"""``platform_create_task`` with the owner's review kept (F180, reopened 2 Oct).

The board task tool (``handlers_board_tasks.create_board_task``) takes a review
mode and defaults to ``auto``: the ticket closes Done by itself. On night 6, Auto
left the mode out when the owner said "drafts only - nothing gets sent without
me" and "set to wait for me", then told the owner the work would wait for them
(#1205, #1206, #1209). Its reply read the tool's ``supervised`` (Auto's own watch,
PRD-224) as the owner's review.

So the platform action runs through here. When the owner's own words in the
chat, this turn's or the one before it, ask to see the work first, the ticket
waits in Review for them, whatever review mode the model passed. The result then
says in plain words whether the owner reviews it, beside the ``review_mode`` that
was kept. The conversation is the server-injected ``_origin_chat_id``, never a
model argument; with none (a session, a heartbeat, a playbook step) nothing
changes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from modules.tools.discovery import handlers_board_tasks
from modules.tools.discovery.handlers_watches import _origin_chat_id
from services.owner_review import owner_asked_to_review

logger = logging.getLogger(__name__)

# This turn's message and the one before it: "set to wait for me", then "I already said yes. Go ahead."
OWNER_WORDS_LOOKBACK = 2
OWNER_ROLE = "user"
REVIEW_BY_OWNER = "human"
WAITS_FOR_OWNER = "It waits in Review for the owner, who sees it before it is done."
CLOSES_BY_ITSELF = "It closes Done by itself: the owner does not review it first."
OWNER_ASKED = "the owner asked to see it first"


def owner_words(db: Session, workspace_id: UUID, chat_id: Optional[UUID]) -> List[str]:
    """The owner's latest messages in ``chat_id``, newest first. Read in a
    savepoint: a refused read must not abort the transaction the ticket is filed in."""
    if chat_id is None:
        return []
    from core.models.core import Message
    from modules.memory.thread_checkpoint import extract_message_text

    try:
        with db.begin_nested():
            rows = (
                db.query(Message.parts)
                .filter(Message.chat_id == chat_id, Message.workspace_id == workspace_id, Message.role == OWNER_ROLE)
                .order_by(Message.created_at.desc())
                .limit(OWNER_WORDS_LOOKBACK)
                .all()
            )
    except Exception:
        logger.exception("[F180] could not read the owner's words in chat %s", chat_id)
        return []
    return [extract_message_text(row.parts) for row in rows]


def _with_review_meaning(result: Dict[str, Any], kept_for_owner: bool) -> Dict[str, Any]:
    """The result, saying in plain words whether the owner reviews the ticket."""
    if not result.get("success"):
        return result
    waits = result.get("review_mode") == REVIEW_BY_OWNER
    meaning = {"review": WAITS_FOR_OWNER if waits else CLOSES_BY_ITSELF}
    if kept_for_owner:
        meaning["review_set_because"] = OWNER_ASKED
    return {**result, **meaning}


async def create_board_task(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a board task (``platform_create_task``), keeping the owner's review."""
    asked = owner_asked_to_review(owner_words(db, workspace_id, _origin_chat_id(params)))
    kept_for_owner = asked and params.get("review_mode") != REVIEW_BY_OWNER
    if asked:
        params = {**params, "review_mode": REVIEW_BY_OWNER}
    result = await handlers_board_tasks.create_board_task(db, workspace_id, params)
    return _with_review_meaning(result, kept_for_owner)


__all__ = ["CLOSES_BY_ITSELF", "OWNER_ASKED", "WAITS_FOR_OWNER", "create_board_task", "owner_words"]
