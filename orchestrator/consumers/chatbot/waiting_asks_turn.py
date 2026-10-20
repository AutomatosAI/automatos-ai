"""P256-FIX-N17-3 (F439, nights 16–17): "did it run?" sees the run that waits for the owner's click.

"Did the stock check run?" two minutes after a run-now raised its card was answered from the
board's finished cards (F317's findings) and never named the run still waiting for the click.

Now a message that asks whether something ran, went through, happened or finished reads the
asking person's waiting call cards (``services/waiting_asks.py``: the chat's ``user_id``, this
workspace, pending and unexpired, the last day) and, when any wait, gives the model one note after
the document passages, the named template's and the team's findings (F317's finished cards come
before it, so the rule is read last): each card's words and when it was raised, and the rule. No
waiting card, no note. A widget visitor's turn reads nothing: the cards are the owner's (F155).
"""
from __future__ import annotations

import functools
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_ROLE = "system"
NOTE_HEAD = "Waiting for the owner's click right now (their approval cards, read just now; none has run):"
ROW = "- {card}, raised {at}"
NOTE_RULE = ("Say plainly which of these has not run and waits for the owner's click; a finished run of something "
             "else is not their run.")

_ASKS_DID_IT_RUN = re.compile(
    r"\b(?:did|has|was)\b[^.?!]{0,80}?"
    r"\b(?:run|ran|go through|gone through|went through|happen|happened|finish|finished)\b",
    re.IGNORECASE)


def asks_did_it_run(text: object) -> bool:
    """Whether the owner asks if something ran: 'did the stock check run?', 'has it gone through?'."""
    return bool(_ASKS_DID_IT_RUN.search(str(text or "")))


def waiting_note(asks: List[Dict[str, str]]) -> str:
    """What the model reads: each waiting card with its time, then the rule."""
    return "\n".join([NOTE_HEAD, *(ROW.format(card=ask["card"], at=ask["at"]) for ask in asks), NOTE_RULE])


def _read(chat: Any, chat_id: Any) -> List[Dict[str, str]]:
    """The chat's person's waiting cards, read in a savepoint; [] when there are none or they can't be read."""
    from core.models.core import Chat
    from services.waiting_asks import waiting_asks

    try:
        with chat.db.begin_nested():
            row = (chat.db.query(Chat.user_id)
                   .filter(Chat.id == uuid.UUID(str(chat_id)), Chat.workspace_id == chat.workspace_id).first())
            if row is None:
                return []
            return waiting_asks(chat.db, chat.workspace_id, row[0], datetime.now(timezone.utc))
    except Exception:
        logger.exception("[F439] the cards waiting for the owner's click could not be read for this turn")
        return []


def reads_the_waiting(chat: Any, latest_text: str, chat_id: Any) -> Optional[str]:
    """For 'did it run?' in the owner's chat: the note naming the cards still waiting, else None."""
    if getattr(chat, "widget_mode", False) or not asks_did_it_run(latest_text):
        return None
    asks = _read(chat, chat_id)
    if not asks:
        return None
    logger.info("[F439] %d card(s) wait for the owner's click on a did-it-run question", len(asks))
    return waiting_note(asks)


Turn = Callable[..., AsyncGenerator[Any, None]]


def reads_the_waiting_asks(retrieval_first: Turn) -> Turn:
    """Wrap ``StreamingChatService._retrieval_first`` (over F317's ``reads_what_the_team_found``): 'did it
    run?' reads the person's cards still waiting for their click, and the note goes in after the team's
    findings (see the module)."""
    @functools.wraps(retrieval_first)
    async def wrapped(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]], agent_runtime: Any,
                      chat_id: str, prefetched: List[Any]) -> AsyncGenerator[Any, None]:
        async for frame in retrieval_first(chat, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
            yield frame
        note = reads_the_waiting(chat, latest_text, chat_id)
        if note:
            llm_messages.append({"role": SYSTEM_ROLE, "content": note})
    return wrapped


__all__ = ["NOTE_RULE", "asks_did_it_run", "reads_the_waiting", "reads_the_waiting_asks", "waiting_note"]
