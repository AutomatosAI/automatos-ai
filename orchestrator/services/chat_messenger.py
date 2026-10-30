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
- A ``chat_id`` is honoured only if it belongs to the workspace AND is owned
  by the message's owner (``clerk_user_id`` → ``users.id``); anything else
  falls back to the Auto thread rather than leaking across users (P227-RVW-2).
  When no owner is resolvable (agent-created scheduled tasks, whose origin was
  server-captured — not a caller-controllable cross-user vector) the legacy
  workspace-only honour is kept.
- Producers call ``deliver_background_message`` (fail-soft): a chat
  failure must never break a watcher tick or a scheduled task.
- After a successful post, a ``chat_changed`` NOTIFY rides the existing
  ``board_events`` LISTEN/NOTIFY lane so an open chat receives the
  message live (PRD-205 S7).
- After a successful post, Auto's own initiative (a watcher, a scheduled
  task, a Mission's narration, the daily digest) tells the chat's owner
  (``auto_message``, PRD-266 US-012b), on a session of its own.
- A post may carry structured parts (``brief@1``, ``suggestion@1``) after
  its text part (PRD-268 US-C19b, ``services.chat_parts``); the text part is
  always first and always required.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Mapping, Optional, Sequence

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from services.chat_parts import checked_parts

logger = logging.getLogger(__name__)

AUTO_CHAT_TITLE = "Auto"
BACKGROUND_LABEL = "Auto · background"


def _resolve_user_int_id(db: Session, clerk_user_id: Optional[str]) -> Optional[int]:
    """The recorded person → integer users.id (#513): a Clerk id, or a local
    operator's email (F166); an agent id is nobody."""
    from core.auth.actor import resolve_recorded_person

    return resolve_recorded_person(db, clerk_user_id)


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


_CALLER_ORIGIN_KEYS = ("origin_chat_id", "chat_id")


def strip_caller_narration_origin(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """PRD-227 P227-RVW-2 (layer 1): the mission-narration origin
    (``run.config['origin_chat_id']``) must be SERVER-injected only.

    Return a FRESH config dict with any caller/LLM-supplied ``origin_chat_id`` or
    ``chat_id`` removed, so no mission-create path can let a caller aim
    ``Auto · mission`` lifecycle lines at another workspace-member's private chat.
    EVERY entry point that turns caller input into ``run.config`` must route
    through this one helper — a per-path re-implementation is how the blog-create
    path (handlers_blog.py) reopened this exact hole. A caller that HAS a
    server-injected origin sets it back on the returned dict afterwards. Pairs
    with ``post_background_message``'s owner check (layer 2) below: with layer 1
    complete, the only ``chat_id`` reaching the no-owner branch there is
    server-captured (scheduled tasks), never caller-chosen.
    """
    return {k: v for k, v in (config or {}).items() if k not in _CALLER_ORIGIN_KEYS}


def _honoured_chat(db: Session, ws_uuid: uuid.UUID, chat_id: Optional[str],
                   owner_int_id: Optional[int], clerk_user_id: Optional[str]):
    """The originating chat when it is valid for this workspace AND owned by the message's
    owner; None otherwise.

    PRD-227 P227-RVW-2: the workspace filter is cross-tenant-safe but NOT cross-user-safe;
    without the owner check a caller-supplied chat_id (a mission's origin_chat_id, a
    suggestion-card chat_id) could target another workspace-member's private chat.
    """
    from core.models.core import Chat

    if not chat_id:
        return None
    try:
        candidate = (
            db.query(Chat)
            .filter(Chat.id == uuid.UUID(str(chat_id)), Chat.workspace_id == ws_uuid)
            .first()
        )
    except ValueError:
        candidate = None
    if candidate is None:
        logger.warning(
            "[ChatMessenger] chat_id %s invalid or outside workspace %s — "
            "falling back to the Auto thread", chat_id, ws_uuid,
        )
        return None
    # No owner to verify against (agent-created scheduled tasks whose origin_chat_id was
    # server-captured at schedule time — not a caller-controllable cross-user vector): keep
    # the legacy workspace-only honour so those producers keep delivering.
    if owner_int_id is None or int(candidate.user_id) == owner_int_id:
        return candidate
    logger.warning(
        "[ChatMessenger] chat_id %s is owned by another user (not "
        "clerk_user_id=%r) — falling back to the Auto thread",
        chat_id, clerk_user_id,
    )
    return None


def _target_chat(db: Session, ws_uuid: uuid.UUID, chat_id: Optional[str],
                 clerk_user_id: Optional[str]):
    """The chat a background message lands in: the honoured originating chat, else the
    owner's Auto thread; None when neither is resolvable. The owner (clerk → int users.id)
    is resolved once, for the ownership check and the Auto-thread fallback."""
    owner_int_id = _resolve_user_int_id(db, clerk_user_id)
    target_chat = _honoured_chat(db, ws_uuid, chat_id, owner_int_id, clerk_user_id)
    if target_chat is not None:
        return target_chat
    if owner_int_id is None:
        logger.warning(
            "[ChatMessenger] No valid chat and no resolvable user "
            "(clerk_user_id=%r) — dropping background message", clerk_user_id,
        )
        return None
    return find_or_create_auto_chat(db, ws_uuid, owner_int_id)


def _source_doc(source: Optional[Dict[str, Any]], link_type: Optional[str],
                link_id: Optional[str]) -> Dict[str, Any]:
    """Persisted provenance — the badge that survives reload."""
    source_doc = {
        "label": BACKGROUND_LABEL,
        **{k: v for k, v in (source or {}).items() if v is not None},
    }
    if link_type is not None:
        source_doc["link_type"] = link_type
    if link_id is not None:
        source_doc["link_id"] = str(link_id)
    return source_doc


def _notify_chat_changed(db: Session, ws_uuid: uuid.UUID, chat) -> None:
    """Live receive (S7): best-effort NOTIFY on the existing lane."""
    try:
        from services.board_events import notify_chat_event

        notify_chat_event(
            db,
            workspace_id=ws_uuid,
            chat_id=str(chat.id),
            user_id=int(chat.user_id),
        )
    except Exception:  # noqa: BLE001 — the NOTIFY is an optimisation
        logger.debug("[ChatMessenger] chat_changed notify failed", exc_info=True)


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
    parts: Optional[Sequence[Mapping[str, Any]]] = None,
):
    """Post one assistant message from a background producer. Raises on
    hard failures (callers wanting fail-soft use ``deliver_background_message``).

    ``parts`` (PRD-268 US-C19b) are structured parts saved after the text part
    (``services.chat_parts``); a refused part raises ValueError before anything
    is read or written. Without them the message is today's single text part.

    Returns the saved Message, or None when the text is empty or no target is
    resolvable (no valid chat AND no resolvable user for an Auto thread).
    """
    structured = checked_parts(parts)
    if not text or not str(text).strip():
        return None

    ws_uuid = workspace_id if isinstance(workspace_id, uuid.UUID) else uuid.UUID(str(workspace_id))
    target_chat = _target_chat(db, ws_uuid, chat_id, clerk_user_id)
    if target_chat is None:
        return None

    # Append via the one existing write path (the text part is the in-turn
    # assistant shape; bumps chat.updated_at → history re-sort ships free via PRD-220).
    from consumers.chatbot.service import ChatService

    message = ChatService(db).save_message(
        chat_id=str(target_chat.id),
        role="assistant",
        parts=[{"type": "text", "text": str(text)}, *structured],
        workspace_id=str(ws_uuid),
        source=_source_doc(source, link_type, link_id),
    )
    _notify_chat_changed(db, ws_uuid, target_chat)
    return message


def _announce(message: object, source: Optional[Dict[str, Any]]) -> None:
    """PRD-266 US-012b: tell the chat's owner Auto wrote (``auto_message``). Never raises."""
    try:
        from services.notification_events.auto_message import announce

        announce(message, source)
    except Exception:
        logger.exception("[ChatMessenger] the auto_message notice failed")


def deliver_background_message(db: Session, **kwargs) -> Optional[object]:
    """Fail-soft wrapper — what producers call. A raising messenger must
    never propagate into a watcher tick or a scheduled task (the
    knowledge_flywheel pattern).

    PRD-266 US-012b: a delivered message is announced (``auto_message``,
    for Auto's own initiative only); the notice never changes what returns."""
    try:
        message = post_background_message(db, **kwargs)
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
    if message is not None:
        _announce(message, kwargs.get("source"))
    return message
