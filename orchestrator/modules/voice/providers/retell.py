"""
PRD-203 V·S4: Retell streaming transport adapter
=================================================

Retell (§8-Qa) is a hosted voice transport: it does STT / TTS / turn-taking /
barge-in at sub-600ms and calls Auto's OWN agent loop for the words via a
**custom-LLM webhook**. The deep Automatos agent — the one real differentiator —
stays ours; only the voice-native hard parts move to the vendor.

This adapter sits at the same architectural seam as ``VoiceServiceClient``
(the self-hosted transport, ``client.py``): a swappable voice transport behind
``modules/voice``. It is pure + streaming, so the "first audio before the full
agent stream completes" property is unit-testable with no vendor, pod, or GPU.

Three concerns, all pure:
  * ``parse_llm_request``  — read Retell's ``response_required`` webhook payload;
  * ``retell_response_frames`` — stream the agent's AI-SDK text out as Retell
    custom-LLM frames, yielding each the moment its text arrives (streaming);
  * ``verify_webhook_signature`` — HMAC auth on the inbound webhook (fail-closed).

The self-hosted pod path (``VoiceServiceClient``) stays as the fallback; retiring
the Pipecat/GPU pod is a cross-repo automatos-voice coordination (§8-Qa), NOT done
here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)

# Retell custom-LLM interaction types we act on.
INTERACTION_RESPONSE_REQUIRED = "response_required"
INTERACTION_REMINDER = "reminder_required"
# WebSocket-lane interaction types (PRD-207: Retell's custom-LLM transport is
# WebSocket-only — dynamic variables arrive once in call_details, pings must
# be echoed, update_only owes no response).
INTERACTION_CALL_DETAILS = "call_details"
INTERACTION_UPDATE_ONLY = "update_only"
INTERACTION_PING_PONG = "ping_pong"


def wrap_ws_response(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap one custom-LLM response frame in Retell's WebSocket envelope."""
    return {"response_type": "response", **frame}


@dataclass(frozen=True)
class RetellLLMRequest:
    """The parsed slice of a Retell custom-LLM webhook we need to answer a turn."""

    response_id: int
    user_text: str
    interaction_type: str
    workspace_id: Optional[str]
    agent_id: Optional[str]
    call_id: Optional[str]
    # PRD-207 S2: who is talking and which on-screen thread to bind — set by
    # the S1 mint as dynamic variables; cross-validated against the mint row
    # before anything is written (HMAC alone never authorises binding).
    user_id: Optional[str] = None
    chat_id: Optional[str] = None


def parse_llm_request(payload: Dict[str, Any]) -> RetellLLMRequest:
    """Extract the answerable turn from a Retell custom-LLM webhook body.

    Retell posts the full running ``transcript`` each turn; the latest user
    utterance is the last ``role == "user"`` entry. Per-call context
    (workspace / agent) rides in ``call.retell_llm_dynamic_variables`` — set when
    the web/phone call is created — so the webhook, which carries no user JWT,
    still knows which workspace and agent to route to.
    """
    response_id = int(payload.get("response_id") or 0)
    interaction_type = str(payload.get("interaction_type") or "")

    transcript = payload.get("transcript") or []
    user_text = ""
    for entry in reversed(transcript):
        if isinstance(entry, dict) and entry.get("role") == "user":
            user_text = str(entry.get("content") or "").strip()
            break

    call = payload.get("call") or {}
    dynamic = call.get("retell_llm_dynamic_variables") or {}
    workspace_id = dynamic.get("workspace_id") or None
    agent_id = dynamic.get("agent_id") or None
    call_id = call.get("call_id") or None
    user_id = dynamic.get("user_id") or None
    chat_id = dynamic.get("chat_id") or None

    return RetellLLMRequest(
        response_id=response_id,
        user_text=user_text,
        interaction_type=interaction_type,
        workspace_id=str(workspace_id) if workspace_id else None,
        agent_id=str(agent_id) if agent_id else None,
        call_id=str(call_id) if call_id else None,
        user_id=str(user_id) if user_id else None,
        chat_id=str(chat_id) if chat_id else None,
    )


def extract_agent_text(chunk: str) -> str:
    """Pull the text out of one AI-SDK stream line (``0:"escaped text"``).

    Non-text frames (tool/data/finish lines: ``2:``/``d:``/…) yield ``""`` so the
    caller skips them. Mirrors the extraction in ``chat_voice._collect_streaming_response``.
    """
    if not chunk.startswith("0:"):
        return ""
    try:
        text_part = json.loads(chunk[2:].strip())
    except (json.JSONDecodeError, ValueError):
        return ""
    return text_part if isinstance(text_part, str) else ""


async def retell_response_frames(
    response_id: int, agent_chunks: AsyncIterator[str]
) -> AsyncIterator[Dict[str, Any]]:
    """Stream the agent's AI-SDK output as Retell custom-LLM response frames.

    THE streaming contract (V·S4): each text chunk is emitted as a
    ``content_complete=False`` frame **the moment it arrives** — Retell begins
    speaking (first audio) before the full agent generation completes, instead of
    the old collect-everything-then-speak posture. A terminal
    ``content_complete=True`` frame closes the turn.
    """
    async for chunk in agent_chunks:
        text = extract_agent_text(chunk)
        if text:
            yield {
                "response_id": response_id,
                "content": text,
                "content_complete": False,
            }
    yield {"response_id": response_id, "content": "", "content_complete": True}


# Retell's actual header shape: ``v={timestamp_ms},d={hex_digest}`` — verified
# against their SDK's webhook-auth source (PRD-207 first-contact finding).
_SIGNATURE_RE = re.compile(r"^v=(\d+),d=([0-9a-fA-F]+)$")
_SIGNATURE_MAX_AGE_MS = 5 * 60 * 1000


_PLAIN_HEX_RE = re.compile(r"^(v1=)?([0-9a-fA-F]{64})$")


def verify_webhook_signature(
    secret: str,
    signature: Optional[str],
    body: bytes,
    *,
    now_ms: Optional[int] = None,
) -> bool:
    """Constant-time verification of Retell's ``x-retell-signature``.

    Retell has TWO observed signing formats and we accept exactly those,
    fail-closed on everything else:

    * **Timestamped** (their SDK's webhook-auth): ``v={ts_ms},d={digest}``
      where ``digest = HMAC-SHA256(secret, raw_body + str(ts)).hexdigest()``,
      refused outside a ±5-minute window (replay protection).
    * **Plain** (observed from their live webhook sender in production —
      first-contact evidence, 2026-07-17): a bare 64-hex
      ``HMAC-SHA256(secret, raw_body).hexdigest()``, optionally ``v1=``-
      prefixed. No timestamp exists in this format, so no window CAN apply.

    SECURITY NOTE (reviewed): the plain format is timestamp-less and
    therefore replayable by anyone who can capture a signed payload
    (requires breaking TLS between Retell and us, or log access). Accepted
    deliberately because (a) rejecting it 401s Retell's REAL events — the
    outage this fix exists to end — and (b) the sole consumer
    (``/api/voice/retell/events``) folds events idempotently by ``call_id``
    with guarded status transitions: a replayed ``call_ended`` re-writes
    identical values and cannot inflate the meter or resurrect a call. The
    speech lane never uses this verifier (minted call_id auth). If Retell's
    sender ever moves to the timestamped scheme exclusively, delete the
    plain branch.

    ``now_ms`` is injectable for deterministic tests.
    """
    if not secret or not signature:
        return False
    provided = signature.strip()

    match = _SIGNATURE_RE.match(provided)
    if match:
        timestamp, digest = match.group(1), match.group(2).lower()
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        try:
            if abs(now - int(timestamp)) > _SIGNATURE_MAX_AGE_MS:
                return False
        except ValueError:
            return False
        expected = hmac.new(
            secret.encode("utf-8"), body + timestamp.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, digest)

    plain = _PLAIN_HEX_RE.match(provided)
    if plain:
        expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, plain.group(2).lower())

    return False
