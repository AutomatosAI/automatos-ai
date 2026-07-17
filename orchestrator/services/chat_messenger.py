"""PRD-205 -- ChatMessenger: the background->chat delivery seam.

Server-side producers (the watcher, scheduled tasks -- later anything) post
an assistant-authored message into a chat conversation after the originating
HTTP turn is gone. Delivery targeting: the originating chat when known
(``watches.origin_chat_id`` / ``agent_scheduled_tasks.origin_chat_id``),
else the per-user per-workspace "Auto" thread (``chats.kind='auto'``) --
the canonical place Auto speaks unprompted.

S2 (this module's foundation): the Auto thread -- find-or-create, one per
(workspace_id, user_id), race-safe against the partial unique index
``uq_chats_auto_thread`` (IntegrityError -> re-select). An ordinary chat in
every other way: it shows in the history list, deep-links via
``/chat?chatId=``, and may be deleted (recreated on the next post).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional, Union

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.models.core import Chat

logger = logging.getLogger(__name__)

AUTO_CHAT_TITLE = "Auto"
AUTO_CHAT_KIND = "auto"


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
