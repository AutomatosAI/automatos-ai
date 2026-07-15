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
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)

# Retell custom-LLM interaction types we act on.
INTERACTION_RESPONSE_REQUIRED = "response_required"
INTERACTION_REMINDER = "reminder_required"


@dataclass(frozen=True)
class RetellLLMRequest:
    """The parsed slice of a Retell custom-LLM webhook we need to answer a turn."""

    response_id: int
    user_text: str
    interaction_type: str
    workspace_id: Optional[str]
    agent_id: Optional[str]
    call_id: Optional[str]


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

    return RetellLLMRequest(
        response_id=response_id,
        user_text=user_text,
        interaction_type=interaction_type,
        workspace_id=str(workspace_id) if workspace_id else None,
        agent_id=str(agent_id) if agent_id else None,
        call_id=str(call_id) if call_id else None,
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


def verify_webhook_signature(secret: str, signature: Optional[str], body: bytes) -> bool:
    """Constant-time HMAC-SHA256 verification of an inbound Retell webhook.

    Fail-closed: no configured secret or no/!matching signature → ``False``. The
    route rejects with 401 so an unauthenticated caller can never drive the agent.
    """
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    provided = signature.strip()
    if provided.startswith("v1="):  # tolerate a scheme prefix if the vendor sends one
        provided = provided[3:]
    return hmac.compare_digest(expected, provided)
