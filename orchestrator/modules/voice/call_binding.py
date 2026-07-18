"""
PRD-207 S2: the webhook trust boundary — who is talking, and where
===================================================================

HMAC proves a webhook came through Retell — it does NOT authorise the
dynamic variables (anything reaching Retell's dashboard could set them).
Binding a live call to an on-screen thread therefore requires the S1 mint
row to PROVE the mapping server-side:

* workspace var must match the mint row — a mismatch here is a misroute or
  tamper and REFUSES the turn outright (there is no proven workspace to
  fall back into);
* chat/user vars must match the row AND the chat must still exist in the
  workspace owned by that user — any mismatch falls closed to the per-call
  fallback chat, attributed to the MINT-proven user (never the var user);
* an unminted call (the phone lane) is attributed to the workspace's
  steward — its earliest owner/admin/member — and registered as a LOUD
  orphan row so lifecycle events and later turns reuse one thread.

The fallback chat is remembered on ``voice_calls.fallback_chat_id`` — the
merged-unarmed webhook created a new chat every TURN (``get_chat("retell:…")``
can never parse as a UUID), losing the conversation each exchange.

Also home to the S6 provenance stamps: the user turn is saved with a voice
source directly; the assistant reply is saved deep inside the streaming
service, so it is stamped post-stream — bounded to messages created within
THIS turn so an interleaved text reply never wears a voice badge.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import case
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

VOICE_LABEL_ASSISTANT = "Auto · voice"
VOICE_LABEL_USER = "Voice"


def voice_source(role: str) -> Dict[str, Any]:
    """The persisted ``messages.source`` doc for a live-voice message."""
    return {
        "origin": "voice",
        "label": VOICE_LABEL_ASSISTANT if role == "assistant" else VOICE_LABEL_USER,
    }


@dataclass(frozen=True)
class CallBinding:
    chat_id: str  # the thread this turn writes into
    user_id: int  # INTEGER users.id driving attribution (memory owner, Q7)
    workspace_id: str  # str(uuid) — always the PROVEN workspace
    bound: bool  # True = the mint-proven on-screen thread


def _workspace_steward(db: Session, ws_uuid: uuid.UUID) -> Optional[int]:
    """The workspace's earliest owner/admin/member — attribution for calls
    that arrive with no minted user (the phone lane). ``chats.user_id`` is a
    NOT-NULL FK, so 'user_id=0' was never a real option; the steward is the
    honest reading of 'the workspace's phone line'."""
    from core.workspaces.models import WorkspaceMember

    role_rank = case(
        (WorkspaceMember.role == "owner", 0),
        (WorkspaceMember.role == "admin", 1),
        else_=2,
    )
    row = (
        db.query(WorkspaceMember.user_id)
        .filter(WorkspaceMember.workspace_id == ws_uuid)
        .order_by(role_rank, WorkspaceMember.id.asc())
        .first()
    )
    return int(row[0]) if row else None


def _voice_chat_title(first_text: str, attempt: int) -> str:
    """chats carries UNIQUE (user_id, title): a fixed 'Voice call' title
    500s the SECOND call a user ever makes. Stamp the start time so every
    call is its own thread; a same-minute repeat gets a numbered suffix."""
    base = (first_text or "").strip()[:50] or f"Voice call — {datetime.utcnow():%b %d, %H:%M}"
    return base if attempt == 0 else f"{base[:44]} ({attempt + 1})"


def create_voice_chat(db: Session, *, user_id: int, workspace_id, first_text: str = ""):
    """Create the thread a live call lands in, immune to unique_user_title.

    No pre-SELECT (test doubles stub the session); collisions surface as
    IntegrityError on commit and retry with a suffixed title."""
    from sqlalchemy.exc import IntegrityError

    from consumers.chatbot.service import ChatService

    last_err: Optional[IntegrityError] = None
    for attempt in range(3):
        try:
            return ChatService(db).create_chat(
                user_id=int(user_id),
                title=_voice_chat_title(first_text, attempt),
                workspace_id=workspace_id,
            )
        except IntegrityError as err:
            db.rollback()
            last_err = err
            logger.warning(
                "voice_chat_title_collision user=%s attempt=%s", user_id, attempt
            )
    raise last_err  # three suffixed collisions in one minute — genuinely stuck


def _fallback_chat(db: Session, row, ws_uuid: uuid.UUID, user_id: int, first_text: str) -> str:
    """Find-or-create the per-call fallback chat, remembered on the row so
    every later turn of this call lands in the SAME thread."""
    from core.models.core import Chat

    if row.fallback_chat_id:
        try:
            existing = (
                db.query(Chat)
                .filter(Chat.id == uuid.UUID(row.fallback_chat_id), Chat.workspace_id == ws_uuid)
                .first()
            )
            if existing is not None:
                return str(existing.id)
        except ValueError:
            pass

    chat = create_voice_chat(db, user_id=int(user_id), workspace_id=ws_uuid, first_text=first_text)
    row.fallback_chat_id = str(chat.id)
    db.commit()
    return str(chat.id)


def resolve_call_binding(
    db: Session,
    *,
    call_id: Optional[str],
    workspace_id: str,
    user_id_var: Optional[str],
    chat_id_var: Optional[str],
    first_text: str = "",
) -> Optional[CallBinding]:
    """Resolve one webhook turn to (chat, user) — or None to refuse the turn.

    Returns None ONLY when there is no safe place to write: no call_id, a
    workspace var contradicting the mint row, or an unminted call in a
    workspace with no members. Every refusal is logged LOUD.
    """
    from core.models.core import Chat
    from core.models.voice_calls import VoiceCall

    if not call_id:
        logger.warning("voice_live_webhook_rejected reason=missing_call_id workspace=%s", workspace_id)
        return None

    row = db.query(VoiceCall).filter(VoiceCall.call_id == str(call_id)).first()

    if row is not None and row.workspace_id is not None:
        # The mint row is SERVER-BORN truth — the vars were only ever our own
        # values echoed back through Retell. The row alone authorises; vars,
        # WHEN PRESENT, must not contradict it (a contradiction is tamper and
        # refuses outright, §7.5-2). Absent vars are fine — first live
        # contact showed Retell omits them unless the config frame asks.
        ws_uuid = row.workspace_id
        if workspace_id and str(workspace_id) != str(row.workspace_id):
            logger.warning(
                "voice_live_webhook_rejected reason=workspace_mismatch call=%s var=%s row=%s",
                call_id, workspace_id, row.workspace_id,
            )
            return None

        # Mint-proven thread binding: the chat still exists here, owned by the
        # minted user; any PRESENT var must agree with the row.
        if row.chat_id and row.user_id is not None:
            vars_agree = (
                (chat_id_var is None or str(chat_id_var) == str(row.chat_id))
                and (user_id_var is None or str(user_id_var) == str(row.user_id))
            )
            if vars_agree:
                try:
                    chat = (
                        db.query(Chat)
                        .filter(Chat.id == uuid.UUID(row.chat_id), Chat.workspace_id == ws_uuid)
                        .first()
                    )
                except ValueError:
                    chat = None
                if chat is not None and int(chat.user_id) == int(row.user_id):
                    return CallBinding(
                        chat_id=str(row.chat_id),
                        user_id=int(row.user_id),
                        workspace_id=str(ws_uuid),
                        bound=True,
                    )
                logger.warning(
                    "voice_live_webhook_rejected reason=chat_gone_or_reowned call=%s chat=%s",
                    call_id, row.chat_id,
                )
            else:
                logger.warning(
                    "voice_live_webhook_rejected reason=vars_mismatch_mint_row call=%s", call_id
                )
            # fall through → per-call fallback, mint-proven user

        if row.user_id is not None:
            chat_id = _fallback_chat(db, row, ws_uuid, int(row.user_id), first_text)
            return CallBinding(
                chat_id=chat_id, user_id=int(row.user_id), workspace_id=str(ws_uuid), bound=False
            )
        # A minted row with no user should not exist (mint requires one) —
        # treat like the unminted lane below.

    # Unminted call (phone lane, or a row missing its user): the workspace can
    # only come from the var here — absent/garbage refuses the turn.
    try:
        ws_uuid = uuid.UUID(str(workspace_id))
    except (TypeError, ValueError):
        logger.warning("voice_live_webhook_rejected reason=bad_workspace_var call=%s", call_id)
        return None

    # Attribute to the workspace steward, register/reuse the orphan row LOUD.
    steward = _workspace_steward(db, ws_uuid)
    if steward is None:
        logger.warning(
            "voice_live_webhook_rejected reason=no_workspace_member call=%s workspace=%s",
            call_id, ws_uuid,
        )
        return None

    if row is None:
        logger.warning(
            "voice_live_orphan_call call=%s workspace=%s — no mint row; attributing to steward user %s",
            call_id, ws_uuid, steward,
        )
        row = VoiceCall(
            call_id=str(call_id),
            provider="retell",
            workspace_id=ws_uuid,
            user_id=steward,
            status="started",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    elif row.user_id is None:
        row.user_id = steward
        db.commit()

    chat_id = _fallback_chat(db, row, ws_uuid, int(row.user_id), first_text)
    return CallBinding(chat_id=chat_id, user_id=int(row.user_id), workspace_id=str(ws_uuid), bound=False)


def upsert_voice_user_message(db: Session, *, chat_id: str, workspace_id: str, text: str) -> None:
    """One growing message per spoken sentence — never stacked prefixes.

    Retell requests a turn at every pause; if the speaker keeps going, the
    next request carries the SAME utterance, longer ("Is the chat, like, a
    single chat" → "… voice chat?" → "… or is it mixed?"). First live use
    stacked all three as separate messages. When the chat's last message is
    a voice user message and old/new are prefix-related, UPDATE it to the
    longer text (JSONB rebuilt, never mutated); otherwise append through the
    one write path with the voice source stamp.
    """
    from consumers.chatbot.service import ChatService
    from core.models.core import Message

    text = (text or "").strip()
    if not text:
        return
    try:
        chat_uuid = uuid.UUID(str(chat_id))
    except ValueError:
        return

    last = (
        db.query(Message)
        .filter(Message.chat_id == chat_uuid)
        .order_by(Message.created_at.desc())
        .first()
    )
    if last is not None and last.role == "user" and (last.source or {}).get("origin") == "voice":
        old = ""
        for part in last.parts or []:
            if isinstance(part, dict) and part.get("type") == "text":
                old = str(part.get("text") or "")
                break
        if old and (text.startswith(old) or old.startswith(text)):
            longer = text if len(text) >= len(old) else old
            last.parts = [{"type": "text", "text": longer}]  # rebuild, never mutate
            db.commit()
            return

    ChatService(db).save_message(
        chat_id=str(chat_uuid),
        role="user",
        parts=[{"type": "text", "text": text}],
        workspace_id=str(workspace_id),
        source=voice_source("user"),
    )


def stamp_assistant_voice_source(
    db: Session, *, chat_id: str, turn_started_at: datetime
) -> int:
    """S6: stamp this turn's assistant reply with the voice source.

    The assistant message is saved deep inside the streaming service (which
    has no source parameter — that plane belongs to producers), so the
    webhook stamps it after the stream closes: assistant messages in this
    chat created since the turn began, still unstamped. The time bound keeps
    an interleaved TEXT reply from ever wearing a voice badge.
    """
    from core.models.core import Message

    try:
        chat_uuid = uuid.UUID(str(chat_id))
    except ValueError:
        return 0

    stamped = 0
    rows = (
        db.query(Message)
        .filter(
            Message.chat_id == chat_uuid,
            Message.role == "assistant",
            Message.created_at >= turn_started_at,
            Message.source.is_(None),
        )
        .all()
    )
    for msg in rows:
        msg.source = voice_source("assistant")
        stamped += 1
    if stamped:
        db.commit()
    return stamped
