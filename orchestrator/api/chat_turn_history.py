"""The conversation a chat turn reads (``POST /api/chat``): its saved messages, its attachments, its page.

The route reads it when the request comes in, for the lane that picks the agent
(``api.chat_dispatch``: how long the chat is, what the person said before), and the turn
reads it again once it runs (``api.chat._produce_turn``). A turn queued behind another in the
same chat then reads that turn's reply, or its failure note as a marker
(``consumers.chatbot.model_history``), never a snapshot from before it ended.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from api.chat_models import ChatMessageRequest
from consumers.chatbot import ChatService
from consumers.chatbot.model_history import history_for_model, rows_for_the_turn
from services.page_context import inject_page_preamble

logger = logging.getLogger(__name__)


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
                 page_ctx: Any, message_id: Any = None) -> List[Dict[str, Any]]:
    """The conversation the turn reads: the chat's messages (history_for_model), the attachments, the page.
    With the turn's own saved message (``message_id``) it is the newest, and the person's later
    messages, which belong to later turns, are left out (``rows_for_the_turn``).

    PRD-221 S2 (extends PRD-220): the page context is the structured reference set
    {page, route, tab, selected, filters, visible_ids}, or the legacy {"page": <label>};
    one renderer serves both. It is sanitized against the allow-list (authz-looking
    fields never survive) and injected prompt-side only: the user message is already
    saved clean, so chat titles and reloaded history never show the hint."""
    rows = rows_for_the_turn(chat_service.get_messages_by_chat_id(chat_id), message_id)
    return inject_page_preamble(with_attachments(history_for_model(rows), current_msg), page_ctx)


__all__ = ["turn_history", "with_attachments"]
