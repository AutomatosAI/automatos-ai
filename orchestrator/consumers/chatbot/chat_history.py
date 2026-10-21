"""The chat list, ``GET /api/chat/history``: one page of a user's chats in a workspace.

PRD-220 S2 ordered the list by most recent activity (``updated_at``, bumped on every
``save_message``), so a thread that just received a message surfaces first. PRD-266
(§4.9, US-007b) pages it with a chat-id cursor, ``starting_after``: a page holds the
chats that come after that chat in the list's own order. A tie on ``updated_at`` is
broken by ``id`` (both descending), so chats touched in the same instant page with no
gap and no repeat. A cursor that is not the caller's chat in this workspace raises
``ChatCursorNotFound``, which the route answers with 404.

``history_row`` is the row each chat becomes in the route's answer (it moved here from
``api/chat.py``, which is over AGENTS.md's 800-line limit).
"""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Query, Session

from core.models import Chat
from core.utils.timestamps import utc_iso


class ChatCursorNotFound(LookupError):
    """``starting_after`` names no chat of this user in this workspace."""


def _users_chats(db: Session, user_id: int, workspace_id: Optional[uuid.UUID]) -> Query:
    """The user's chats, in ``workspace_id`` when one is given (as the list always was)."""
    query = db.query(Chat).filter(Chat.user_id == user_id)
    if workspace_id is not None:
        query = query.filter(Chat.workspace_id == workspace_id)
    return query


def _cursor_chat(chats: Query, starting_after: str) -> Chat:
    """The chat ``starting_after`` names among ``chats``; ``ChatCursorNotFound`` when none."""
    try:
        chat_id = uuid.UUID(str(starting_after))
    except ValueError:
        raise ChatCursorNotFound(starting_after) from None
    cursor = chats.filter(Chat.id == chat_id).first()
    if cursor is None:
        raise ChatCursorNotFound(starting_after)
    return cursor


def _after(cursor: Chat):
    """The filter for the chats that come after ``cursor`` in the list's order."""
    return or_(
        Chat.updated_at < cursor.updated_at,
        and_(Chat.updated_at == cursor.updated_at, Chat.id < cursor.id),
    )


def history_page(
    db: Session,
    user_id: int,
    limit: int,
    starting_after: Optional[str],
    workspace_id: Optional[uuid.UUID],
) -> List[Chat]:
    """Up to ``limit`` of the user's chats, most recent activity first.

    With ``starting_after`` (a chat id), the chats after that one in the same order;
    raises ``ChatCursorNotFound`` when it is not this user's chat in this workspace.
    """
    chats = _users_chats(db, user_id, workspace_id)
    if starting_after:
        chats = chats.filter(_after(_cursor_chat(chats, starting_after)))
    return chats.order_by(desc(Chat.updated_at), desc(Chat.id)).limit(limit).all()


def history_row(chat: Chat, preview: Optional[str]) -> Dict[str, Any]:
    """One chat as ``GET /api/chat/history`` lists it; ``preview`` is its latest message, cut."""
    return {
        "id": str(chat.id),
        "userId": chat.user_id,
        "title": chat.title,
        "createdAt": utc_iso(chat.created_at),
        "updatedAt": utc_iso(chat.updated_at),
        "visibility": chat.visibility,
        "lastContext": chat.last_context,
        "lastMessagePreview": preview,
        # PRD-205 S7: 'auto' marks the thread where Auto speaks unprompted.
        "kind": chat.kind,
    }
