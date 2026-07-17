"""PRD-205 -- ChatMessenger: the background->chat delivery seam.

Server-side producers (the watcher, scheduled tasks -- later anything) post
an assistant-authored message into a chat conversation after the originating
HTTP turn is gone. Delivery targeting: the originating chat when known
(``watches.origin_chat_id`` / ``agent_scheduled_tasks.origin_chat_id``),
else the per-user per-workspace "Auto" thread (``chats.kind='auto'``) --
the canonical place Auto speaks unprompted.

Two layers (S1):

- :func:`post_background_message` -- the mechanics, on a session the CALLER
  owns: resolve the integer user (Clerk string -> ``users.id``, the #513
  lesson), validate the target chat belongs to the workspace, fall back to
  the S2 Auto thread, append via ``ChatService.save_message`` with the exact
  AI-SDK parts shape the in-turn assistant writes use, stamp
  ``messages.source``, and emit the S7 ``chat_changed`` NOTIFY post-commit.
- :func:`deliver_background_message` -- what producers actually call.
  Opens and owns its OWN short-lived session (``core.database`` SessionLocal
  pattern, injectable for tests) so a producer's transaction is never
  touched, and NEVER raises: a chat delivery failure must not break a
  watcher tick or a scheduled task (fail-soft, knowledge_flywheel pattern).

S2 (foundation): the Auto thread -- find-or-create, one per
(workspace_id, user_id), race-safe against the partial unique index
``uq_chats_auto_thread`` (IntegrityError -> re-select). An ordinary chat in
every other way: it shows in the history list, deep-links via
``/chat?chatId=``, and may be deleted (recreated on the next post).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.models.core import Chat, Message, User

logger = logging.getLogger(__name__)

AUTO_CHAT_TITLE = "Auto"
AUTO_CHAT_KIND = "auto"

# The badge label the UI renders for background-authored messages
# ("Auto <middle dot> background" -- escape keeps the source ASCII-only).
AUTO_BACKGROUND_LABEL = "Auto \u00b7 background"


def _coerce_workspace_uuid(workspace_id: Union[str, uuid.UUID]) -> uuid.UUID:
    """Normalize to uuid.UUID; raises ValueError on garbage (boundary check)."""
    if isinstance(workspace_id, uuid.UUID):
        return workspace_id
    return uuid.UUID(str(workspace_id))


def _find_auto_chat(
    db: Session, workspace_id: uuid.UUID, user_int_id: int
) -> Optional[Chat]:
    """The (at most one) live Auto thread for this user in this workspace."""
    return (
        db.query(Chat)
        .filter(
            Chat.workspace_id == workspace_id,
            Chat.user_id == user_int_id,
            Chat.kind == AUTO_CHAT_KIND,
        )
        .first()
    )


def find_or_create_auto_chat(
    db: Session,
    workspace_id: Union[str, uuid.UUID],
    user_int_id: int,
) -> Chat:
    """Return the user's Auto thread in this workspace, creating it if absent.

    Race-safe: a concurrent create surfaces as IntegrityError from the
    ``uq_chats_auto_thread`` partial unique index -> rollback and re-select
    the winner. ``user_int_id`` MUST be the integer ``users.id`` -- never a
    Clerk subject string (the #513 lesson); resolution happens at the
    messenger seam, not here.

    Commits (its own short-lived session by contract -- the S1 wrapper owns
    the session lifecycle; never call this with a request-scoped session
    holding uncommitted producer state).
    """
    ws_uuid = _coerce_workspace_uuid(workspace_id)
    if not isinstance(user_int_id, int):
        raise ValueError(
            f"user_int_id must be the integer users.id, got {type(user_int_id).__name__}"
        )

    existing = _find_auto_chat(db, ws_uuid, user_int_id)
    if existing is not None:
        return existing

    now = datetime.utcnow()
    chat = Chat(
        id=uuid.uuid4(),
        user_id=user_int_id,
        workspace_id=ws_uuid,
        title=AUTO_CHAT_TITLE,
        visibility="private",
        kind=AUTO_CHAT_KIND,
        created_at=now,
        updated_at=now,
    )
    db.add(chat)
    try:
        db.commit()
        db.refresh(chat)
        logger.info(
            "[ChatMessenger] created Auto thread %s (workspace=%s user=%s)",
            chat.id, ws_uuid, user_int_id,
        )
        return chat
    except IntegrityError:
        # Lost the create race -- the partial unique index guarantees exactly
        # one winner; adopt it.
        db.rollback()
        winner = _find_auto_chat(db, ws_uuid, user_int_id)
        if winner is not None:
            return winner
        raise


# ---------------------------------------------------------------------------
# S1 -- post + fail-soft wrapper
# ---------------------------------------------------------------------------


def _resolve_user_int_id(db: Session, clerk_user_id: Optional[str]) -> Optional[int]:
    """Clerk subject string -> integer ``users.id`` (coordinator pattern).

    NEVER write a Clerk string into ``chats.user_id`` (#513). Accepts an
    email as a fallback identifier (the api/chat.py get_user_id precedent).
    Returns None when unresolvable -- the caller decides what that means.
    """
    if not clerk_user_id:
        return None
    principal = str(clerk_user_id)
    row = (
        db.query(User.id)
        .filter(User.clerk_user_id == principal)
        .first()
    )
    if row:
        return int(row[0])
    if "@" in principal:
        row = db.query(User.id).filter(User.email == principal).first()
        if row:
            return int(row[0])
    return None


def _build_source(
    source: Optional[Dict[str, Any]],
    link_type: Optional[str],
    link_id: Optional[str],
) -> Dict[str, Any]:
    """New dict (never mutate the caller's): source + link fields + label."""
    final: Dict[str, Any] = dict(source or {})
    if link_type is not None:
        final["link_type"] = link_type
    if link_id is not None:
        final["link_id"] = str(link_id)
    final.setdefault("label", AUTO_BACKGROUND_LABEL)
    return final


def post_background_message(
    db: Session,
    *,
    workspace_id: Union[str, uuid.UUID],
    text: str,
    source: Optional[Dict[str, Any]] = None,
    chat_id: Optional[str] = None,
    clerk_user_id: Optional[str] = None,
    link_type: Optional[str] = None,
    link_id: Optional[str] = None,
) -> Optional[Message]:
    """Post one assistant-authored background message; returns it or None.

    Target resolution: ``chat_id`` when it exists AND belongs to
    ``workspace_id`` (a foreign or deleted chat is treated as unknown --
    never post across workspaces); otherwise the Auto thread of
    ``clerk_user_id``. No valid chat and no resolvable user -> None (logged).

    The session is the caller's -- :func:`deliver_background_message` is the
    entry point that owns one. ``save_message`` commits; the S7
    ``chat_changed`` NOTIFY is emitted (and committed) after, fail-soft.
    """
    from consumers.chatbot.service import ChatService

    if not text or not str(text).strip():
        logger.debug("[ChatMessenger] empty text -- nothing to post")
        return None
    ws_uuid = _coerce_workspace_uuid(workspace_id)
    chat_service = ChatService(db)

    chat: Optional[Chat] = None
    if chat_id:
        chat = chat_service.get_chat(str(chat_id), workspace_id=ws_uuid)
        if chat is None:
            logger.info(
                "[ChatMessenger] chat %s missing or outside workspace %s -- "
                "falling back to the Auto thread",
                chat_id, ws_uuid,
            )

    if chat is None:
        user_int_id = _resolve_user_int_id(db, clerk_user_id)
        if user_int_id is None:
            logger.info(
                "[ChatMessenger] no valid chat and no resolvable user "
                "(clerk=%r) in workspace %s -- dropping background message",
                clerk_user_id, ws_uuid,
            )
            return None
        chat = find_or_create_auto_chat(db, ws_uuid, user_int_id)

    message = chat_service.save_message(
        chat_id=str(chat.id),
        role="assistant",
        # EXACT AI-SDK shape the in-turn assistant writes use
        # (consumers/chatbot/service.py stream save).
        parts=[{"type": "text", "text": str(text)}],
        workspace_id=str(ws_uuid),
        source=_build_source(source, link_type, link_id),
    )

    # S7 backend: tell any open client this chat changed. Post-commit,
    # fail-soft -- a NOTIFY problem never unwinds the saved message.
    try:
        from services import board_events

        board_events.notify_chat_event(
            db,
            workspace_id=ws_uuid,
            chat_id=chat.id,
            user_id=chat.user_id,
        )
        db.commit()  # pg_notify fires on commit
    except Exception:  # noqa: BLE001 -- delivery beat latency, not correctness
        logger.debug(
            "[ChatMessenger] chat_changed notify failed for chat %s",
            chat.id, exc_info=True,
        )

    logger.info(
        "[ChatMessenger] posted background message %s to chat %s (origin=%s)",
        message.id, chat.id, (source or {}).get("origin"),
    )
    return message


def deliver_background_message(
    *,
    workspace_id: Union[str, uuid.UUID],
    text: str,
    source: Optional[Dict[str, Any]] = None,
    chat_id: Optional[str] = None,
    clerk_user_id: Optional[str] = None,
    link_type: Optional[str] = None,
    link_id: Optional[str] = None,
    session_factory: Optional[Callable[[], Session]] = None,
) -> Optional[Message]:
    """The producer-facing entry point: own session, never raises.

    Opens a short-lived session from ``session_factory`` (tests inject the
    test factory here; default is the app ``SessionLocal``) so the producer's
    transaction stays clean, and swallows-and-logs every failure -- a broken
    chat delivery must never break a watcher tick or a scheduled task.
    """
    db: Optional[Session] = None
    try:
        if session_factory is None:
            from core.database.database import SessionLocal as session_factory
        db = session_factory()
        return post_background_message(
            db,
            workspace_id=workspace_id,
            text=text,
            source=source,
            chat_id=chat_id,
            clerk_user_id=clerk_user_id,
            link_type=link_type,
            link_id=link_id,
        )
    except Exception:  # noqa: BLE001 -- fail-soft by contract
        logger.error(
            "[ChatMessenger] background delivery failed (workspace=%s chat=%s "
            "origin=%s) -- producer unaffected",
            workspace_id, chat_id, (source or {}).get("origin"),
            exc_info=True,
        )
        return None
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass
