"""PRD-266 US-012b (§4.4, D22, K11) — 'Auto messaged you': Auto wrote in a person's chat unprompted.

``services/chat_messenger.deliver_background_message`` calls :func:`announce` with the message it
just saved and its producer's ``source``. Only Auto's own initiative is announced
(``ANNOUNCED_ORIGINS``: a scheduled task, the daily digest), never the echo of what the person
did (a question's answer, a click's result, a brief posted because they opened the app: its
source's ``via`` is ``open``, PRD-268 K11). A watcher's and a Mission's posts are not announced:
each already raises its own bell event (``watch_*``, ``mission_*``), so a second row would ring
the same person twice for one event. :func:`announce`
copies plain values and hands :func:`dispatch_auto_message` to ``send_soon``, which runs it on a
session of its own: it finds the chat's owner in its workspace and raises ``auto_message`` for
that person only (in-app by default; one of the three push types, FR-7), linked to the chat.

Scheduled tasks can post several lines, so one chat is told at most once every
``AUTO_MESSAGE_COALESCE_SECONDS``; the daily digest is never held back (its notice counts toward
the window like any other). The window is read from the bell's own rows, so it holds across
workers; a person whose preferences write no bell row for this event is not held back. Best
effort: two notices written at the same instant on two workers may both go.

A notice is a courtesy, never part of the post: nothing here raises.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from uuid import UUID

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.orm import InstanceState

from core.models.core import Chat, Message
from core.services.notification_columns import clamp
from services.brief.model import VIA_OPEN
from services.notification_events.soon import send_soon
from services.push.payload import AUTO_MESSAGE_LINE

logger = logging.getLogger(__name__)

EVENT_TYPE = "auto_message"
LINK_TYPE = "chat"
TITLE = AUTO_MESSAGE_LINE  # the bell says what the phone says
AGENT_NAME = "Auto"
DAILY_DIGEST = "daily_digest"
# Auto's own initiative (K11). Never 'question_answer' or a click's result: the person's own action.
# Never 'watcher' or 'mission': each raises its own bell event already (one row per event).
# Whether watcher and Mission events push to a phone is the owner's decision (K11): not designed here.
ANNOUNCED_ORIGINS: frozenset[str] = frozenset({"scheduled_task", DAILY_DIGEST})
NEVER_COALESCED: frozenset[str] = frozenset({DAILY_DIGEST})
# How a post came to be, when it was the person's own action (PRD-268 K11): a brief on open.
UNANNOUNCED_VIAS: frozenset[str] = frozenset({VIA_OPEN})
AUTO_MESSAGE_COALESCE_SECONDS = 600
EXCERPT_MAX_CHARS = 200

# An auto_message for this person and chat, written within the window. Bound parameters only.
_TOLD_LATELY_SQL = text(
    "SELECT 1 FROM notifications "
    "WHERE workspace_id = CAST(:ws AS uuid) AND user_id = :uid AND event_type = :event "
    "  AND link_type = :link_type AND link_id = :chat "
    "  AND created_at > clock_timestamp() - make_interval(secs => CAST(:window AS double precision)) "
    "LIMIT 1"
)


@dataclass(frozen=True)
class AutoMessage:
    """What a notice needs from a saved message: plain values, read before the post's session moves on."""

    workspace_id: str
    chat_id: str
    excerpt: str
    origin: str


def announce(message: Any, source: Optional[Mapping[str, Any]]) -> None:
    """Raise ``auto_message`` for a saved message Auto posted on its own initiative. Never raises.

    Anything but a saved ``Message`` (None, a mock, an unsaved row) and any other origin is ignored.
    """
    try:
        notice = _notice_of(message, source)
    except Exception:
        logger.exception("[auto_message] the message could not be read for its notice")
        return
    if notice is not None:
        send_soon(dispatch_auto_message(notice), what=EVENT_TYPE)


def _notice_of(message: Any, source: Optional[Mapping[str, Any]]) -> Optional[AutoMessage]:
    """The notice's plain values, or None when the message is not one to announce."""
    origin = source.get("origin") if isinstance(source, Mapping) else None
    via = source.get("via") if isinstance(source, Mapping) else None
    if origin not in ANNOUNCED_ORIGINS or via in UNANNOUNCED_VIAS or not _saved(message):
        return None
    return AutoMessage(
        workspace_id=str(message.workspace_id),
        chat_id=str(message.chat_id),
        excerpt=_excerpt(message.parts),
        origin=str(origin),
    )


def _saved(message: Any) -> bool:
    """A ``Message`` the database holds (a mock with Message's spec has no instance state)."""
    if not isinstance(message, Message):
        return False
    state = sa_inspect(message, raiseerr=False)
    return isinstance(state, InstanceState) and bool(state.has_identity)


def _excerpt(parts: Any) -> str:
    """The message's text, at most ``EXCERPT_MAX_CHARS`` characters."""
    texts = [
        str(part.get("text") or "") for part in (parts or [])
        if isinstance(part, Mapping) and part.get("type") == "text"
    ]
    return clamp(" ".join(" ".join(texts).split()), EXCERPT_MAX_CHARS)


def _open_session() -> Any:
    from core.database.database import SessionLocal

    return SessionLocal()


def _dispatcher(db: Any, workspace_id: str) -> Any:
    from core.services.notification_dispatcher import NotificationDispatcher

    return NotificationDispatcher(db, workspace_id)


async def dispatch_auto_message(notice: AutoMessage) -> None:
    """Tell the chat's owner Auto wrote, on a session of its own; one commit. Never raises."""
    try:
        db = _open_session()
    except Exception:
        logger.exception("[auto_message] no session for the notice of chat %s", notice.chat_id)
        return
    try:
        await _tell_owner(db, notice)
        db.commit()
    except Exception:
        logger.exception("[auto_message] the notice of chat %s was not sent", notice.chat_id)
    finally:
        db.close()


async def _tell_owner(db: Any, notice: AutoMessage) -> None:
    """Dispatch the notice to the chat's owner, unless the chat is gone or was told within the window."""
    owner = _chat_owner(db, notice)
    if owner is None:
        return
    if notice.origin not in NEVER_COALESCED and _told_lately(db, notice, owner):
        return
    await _dispatcher(db, notice.workspace_id).dispatch(
        event_type=EVENT_TYPE,
        title=TITLE,
        message=notice.excerpt,
        link_type=LINK_TYPE,
        link_id=notice.chat_id,
        agent_name=AGENT_NAME,
        user_id=owner,
    )


def _chat_owner(db: Any, notice: AutoMessage) -> Optional[int]:
    """The chat's owner (``chats.user_id``), read in the message's workspace; None when it is gone."""
    owner = db.query(Chat.user_id).filter(
        Chat.id == UUID(notice.chat_id), Chat.workspace_id == UUID(notice.workspace_id),
    ).scalar()
    return int(owner) if owner is not None else None


def _told_lately(db: Any, notice: AutoMessage, owner: int) -> bool:
    """True when the owner's bell got an ``auto_message`` for this chat within the window."""
    return db.execute(_TOLD_LATELY_SQL, {
        "ws": notice.workspace_id,
        "uid": owner,
        "event": EVENT_TYPE,
        "link_type": LINK_TYPE,
        "chat": notice.chat_id,
        "window": AUTO_MESSAGE_COALESCE_SECONDS,
    }).first() is not None


__all__ = [
    "ANNOUNCED_ORIGINS",
    "AUTO_MESSAGE_COALESCE_SECONDS",
    "AutoMessage",
    "UNANNOUNCED_VIAS",
    "announce",
    "dispatch_auto_message",
]
