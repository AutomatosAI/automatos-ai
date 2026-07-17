"""PRD-205 S1/S2 — ChatMessenger: the background→chat delivery seam.

The one way a server-side producer (the watcher, scheduled tasks — later
anything) posts an assistant-authored message into a conversation after
the originating HTTP turn is gone. Targeting: the originating chat when
known and valid for the workspace; otherwise the per-user per-workspace
**Auto thread** (``chats.kind='auto'`` — the canonical place Auto speaks
unprompted, an ordinary chat in every other way).

Rules enforced here, once:
- ``chats.user_id`` is INTEGER ``users.id`` — Clerk strings are resolved
  via ``User.clerk_user_id`` (the #513 lesson), never written raw.
- A ``chat_id`` is honoured only if it belongs to the workspace; anything
  else falls back to the Auto thread rather than leaking across chats.
- Producers call ``deliver_background_message`` (fail-soft): a chat
  failure must never break a watcher tick or a scheduled task.
- After a successful post, a ``chat_changed`` NOTIFY rides the existing
  ``board_events`` LISTEN/NOTIFY lane so an open chat receives the
  message live (PRD-205 S7).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

AUTO_CHAT_TITLE = "Auto"
BACKGROUND_LABEL = "Auto · background"


def _resolve_user_int_id(db: Session, clerk_user_id: Optional[str]) -> Optional[int]:
    """Clerk string → integer users.id (the coordinator pattern; #513)."""
    if not clerk_user_id:
        return None
    from core.models.core import User

    row = db.query(User.id).filter(User.clerk_user_id == str(clerk_user_id)).first()
    return int(row[0]) if row else None


def find_or_create_auto_chat(db: Session, workspace_id, user_int_id: int):
    """The per-(workspace, user) Auto thread — find it or create it.

    Race-safe: the partial unique index ``uq_chats_auto_thread`` makes the
    concurrent-create loser IntegrityError; we rollback and re-select.
    Deleting the thread is allowed — it is simply recreated on the next
    background post.
    """
    from core.models.core import Chat

    ws_uuid = workspace_id if isinstance(workspace_id, uuid.UUID) else uuid.UUID(str(workspace_id))

    def _select():
        return (
            db.query(Chat)
            .filter(
                Chat.workspace_id == ws_uuid,
                Chat.user_id == int(user_int_id),
                Chat.kind == "auto",
            )
            .first()
        )

    existing = _select()
    if existing:
        return existing

    try:
        chat = Chat(
            id=uuid.uuid4(),
            user_id=int(user_int_id),
            workspace_id=ws_uuid,
            title=AUTO_CHAT_TITLE,
            visibility="private",
            kind="auto",
        )
        db.add(chat)
        db.commit()
        db.refresh(chat)
        logger.info(
            "[ChatMessenger] Created Auto thread %s for user %s in workspace %s",
            chat.id, user_int_id, ws_uuid,
        )
        return chat
    except IntegrityError:
        db.rollback()
        return _select()


def post_background_message(
    db: Session,
    *,
    workspace_id,
    text: str,
    source: Dict[str, Any],
    chat_id: Optional[str] = None,
    clerk_user_id: Optional[str] = None,
    link_type: Optional[str] = None,
    link_id: Optional[str] = None,
):
    """Post one assistant message from a background producer. Raises on
    hard failures (callers wanting fail-soft use ``deliver_background_message``).

    Returns the saved Message, or None when no target is resolvable
    (no valid chat AND no resolvable user for an Auto thread).
    """
    from consumers.chatbot.service import ChatService
    from core.models.core import Chat

    if not text or not str(text).strip():
        return None

    ws_uuid = workspace_id if isinstance(workspace_id, uuid.UUID) else uuid.UUID(str(workspace_id))

    # 1. Resolve the target chat: originating chat if valid for this
    #    workspace, else the user's Auto thread.
    target_chat = None
    if chat_id:
        try:
            target_chat = (
                db.query(Chat)
                .filter(Chat.id == uuid.UUID(str(chat_id)), Chat.workspace_id == ws_uuid)
                .first()
            )
        except ValueError:
            target_chat = None
        if target_chat is None:
            logger.warning(
                "[ChatMessenger] chat_id %s invalid or outside workspace %s — "
                "falling back to the Auto thread", chat_id, ws_uuid,
            )

    if target_chat is None:
        user_int_id = _resolve_user_int_id(db, clerk_user_id)
        if user_int_id is None:
            logger.warning(
                "[ChatMessenger] No valid chat and no resolvable user "
                "(clerk_user_id=%r) — dropping background message", clerk_user_id,
            )
            return None
        target_chat = find_or_create_auto_chat(db, ws_uuid, user_int_id)

    # 2. Persisted provenance — the badge that survives reload.
    source_doc = {
        "label": BACKGROUND_LABEL,
        **{k: v for k, v in (source or {}).items() if v is not None},
    }
    if link_type is not None:
        source_doc["link_type"] = link_type
    if link_id is not None:
        source_doc["link_id"] = str(link_id)

    # 3. Append via the one existing write path (parts shape identical to
    #    the in-turn assistant writes; bumps chat.updated_at → history
    #    re-sort ships free via PRD-220).
    message = ChatService(db).save_message(
        chat_id=str(target_chat.id),
        role="assistant",
        parts=[{"type": "text", "text": str(text)}],
        workspace_id=str(ws_uuid),
        source=source_doc,
    )

    # 4. Live receive (S7): best-effort NOTIFY on the existing lane.
    try:
        from services.board_events import notify_chat_event

        notify_chat_event(
            db,
            workspace_id=ws_uuid,
            chat_id=str(target_chat.id),
            user_id=int(target_chat.user_id),
        )
    except Exception:  # noqa: BLE001 — the NOTIFY is an optimisation
        logger.debug("[ChatMessenger] chat_changed notify failed", exc_info=True)

    return message


def deliver_background_message(db: Session, **kwargs) -> Optional[object]:
    """Fail-soft wrapper — what producers call. A raising messenger must
    never propagate into a watcher tick or a scheduled task (the
    knowledge_flywheel pattern)."""
    try:
        return post_background_message(db, **kwargs)
    except Exception:  # noqa: BLE001
        logger.warning(
            "[ChatMessenger] background delivery failed (source=%s)",
            (kwargs.get("source") or {}).get("origin"),
            exc_info=True,
        )
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None
