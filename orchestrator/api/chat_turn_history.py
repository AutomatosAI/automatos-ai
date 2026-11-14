"""The conversation a chat turn reads (``POST /api/chat``): its saved messages, its attachments, its page.

The route reads it when the request comes in, for the lane that picks the agent
(``api.chat_dispatch``: how long the chat is, what the person said before), and the turn
reads it again once it runs (``api.chat._produce_turn``). A turn queued behind another in the
same chat then reads that turn's reply, or its failure note as a marker
(``consumers.chatbot.model_history``), never a snapshot from before it ended.

A spoken turn (``spokenMode``) reads only the chat's newest messages when
``SPOKEN_MODE_HISTORY_MESSAGES`` says how many (voice latency C-4); a typed turn, and a spoken
one with the setting at 0, read the whole chat.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from api.chat_models import ChatMessageRequest
from config import config
from consumers.chatbot import ChatService
from consumers.chatbot.model_history import USER, history_for_model, rows_for_the_turn
from core.models import Message
from services.page_context import inject_page_preamble

logger = logging.getLogger(__name__)

MIN_SPOKEN_WINDOW = 4  # the lane reads the person's message before this one


def spoken_window(spoken: bool) -> Optional[int]:
    """How many recent messages a spoken turn reads; None: the whole chat (a typed turn, the
    setting at 0, or a value under MIN_SPOKEN_WINDOW)."""
    limit = int(config.SPOKEN_MODE_HISTORY_MESSAGES or 0)
    return limit if spoken and limit >= MIN_SPOKEN_WINDOW else None


def _recent_rows(db: Session, chat_id: str, limit: int) -> List[Any]:
    """The chat's newest ``limit`` messages, oldest first. [] for an id that is no UUID."""
    try:
        chat_uuid = uuid.UUID(str(chat_id))
    except ValueError:
        return []
    newest_first = (
        db.query(Message).filter(Message.chat_id == chat_uuid)
        .order_by(Message.created_at.desc()).limit(limit).all()
    )
    return list(reversed(newest_first))


def _from_a_persons_message(rows: List[Any]) -> List[Any]:
    """``rows`` without the leading rows that are not a person's (a new list)."""
    first = next((index for index, row in enumerate(rows) if getattr(row, "role", None) == USER), len(rows))
    return list(rows[first:])


def _window_rows(chat_service: ChatService, chat_id: str, message_id: Any, recent: int) -> Optional[List[Any]]:
    """The rows a turn reads from the chat's newest ``recent`` messages. The route's read (no
    ``message_id``) takes the window as it is; the turn's read starts at a person's message and
    ends with its own. None when the turn's message is not in the window (the turns queued
    before it filled it): the whole chat is read."""
    window = _recent_rows(chat_service.db, chat_id, recent)
    if message_id is None:
        return window
    if all(getattr(row, "id", None) != message_id for row in window):
        return None
    return _from_a_persons_message(rows_for_the_turn(window, message_id))


def with_attachments(history: List[Dict[str, Any]], current_msg: ChatMessageRequest) -> List[Dict[str, Any]]:
    """PRD-127: the request's ephemeral attachment ids ride on the latest user message.
    Attachments are request-scoped (7-day S3 TTL) and never persisted in chat history;
    AttachmentResolver resolves them inline. The frontend sends ids both at the top
    level and inside file parts."""
    ids = list(current_msg.attachment_ids or [])
    for part in current_msg.parts or []:
        if part.attachment_id and part.attachment_id not in ids:
            ids.append(part.attachment_id)
    logger.info(
        "[PRD-127] chat request attachments: top_level_ids=%s parts=%s collected=%s",
        current_msg.attachment_ids, [p.dict(exclude_none=True) for p in current_msg.parts or []], ids,
    )
    if not ids:
        return history
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("role") == "user":
            logger.info("[PRD-127] injected %d attachment_ids into message_history[%d]", len(ids), index)
            return history[:index] + [{**history[index], "attachment_ids": ids}] + history[index + 1:]
    return history


def turn_history(chat_service: ChatService, chat_id: str, current_msg: ChatMessageRequest,
                 page_ctx: Any, message_id: Any = None, recent: Optional[int] = None) -> List[Dict[str, Any]]:
    """The conversation the turn reads: the chat's messages (history_for_model), the attachments, the page.
    With the turn's own saved message (``message_id``) it is the newest, and the person's later
    messages, which belong to later turns, are left out (``rows_for_the_turn``). With ``recent``
    (``spoken_window``) only the chat's newest ``recent`` messages are read.

    PRD-221 S2 (extends PRD-220): the page context is the structured reference set
    {page, route, tab, selected, filters, visible_ids}, or the legacy {"page": <label>};
    one renderer serves both. It is sanitized against the allow-list (authz-looking
    fields never survive) and injected prompt-side only: the user message is already
    saved clean, so chat titles and reloaded history never show the hint."""
    rows = _window_rows(chat_service, chat_id, message_id, recent) if recent else None
    if rows is None:
        rows = rows_for_the_turn(chat_service.get_messages_by_chat_id(chat_id), message_id)
    return inject_page_preamble(with_attachments(history_for_model(rows), current_msg), page_ctx)


__all__ = ["spoken_window", "turn_history", "with_attachments"]
