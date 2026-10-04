"""Auto is told, in the turn, which cards the owner named (F241, night 8).

Night 8: 17 of 95 first tries on a card the owner named by number landed on that card.
Nothing in Auto's turn said that #0201 is a card on the owner's board: the turn's late
"Most relevant actions" line ranked social posts, skills and scheduled tasks for
"Approve #0201", and Auto sent the number to platform_submit_social_post; "Let's talk
about #0329" got "What is #0329? Is it a task, a report, or something else?".
``follows_the_owner`` refuses a call that misses the card; this note is the first try.
For each card the owner's message names, the turn says what it is (its title and
column, or that no card has that number) and the call that does what they asked with
it, by their verb (``follows_the_owner.right_call``). It goes in last, after every
cache-stable block, like the turn's ranked actions.

A widget visitor's turn gets no note: the board is the owner's.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, AsyncGenerator, Callable, Dict, List

from sqlalchemy.orm import Session

from modules.tools.discovery.follows_the_owner import right_call
from modules.tools.discovery.owner_turn import (
    MISSION_CARD, RUN_CARD, STEP_CARD, NamedCard, OwnerTurn, card_kind, card_words, cards_named,
)

logger = logging.getLogger(__name__)

CARDS_NAMED = ("The owner's message names cards on their board by number. A card's number is never the id of a "
               "social post, blog post, skill, tool, agent, document, timer or playbook: act on the card itself.")
KIND_WORDS = {MISSION_CARD: "a mission's own card (mission_id takes its number)",
              STEP_CARD: "a step of a mission", RUN_CARD: "a playbook run's card"}
A_CARD = "a card"
SYSTEM_ROLE = "system"


def cards_note(db: Session, workspace_id: Any, latest: str) -> str:
    """The note for the cards ``latest`` names by number, or "" when it names none."""
    cards = cards_named(db, workspace_id, latest)
    if not cards:
        return ""
    turn = OwnerTurn(latest=latest, earlier="", cards=cards)
    return "\n".join([CARDS_NAMED, *(_card_line(turn, card) for card in cards)])


def _card_line(turn: OwnerTurn, card: NamedCard) -> str:
    """'- #0201 ('Reply to Hannah', review), a card. To approve it: …'"""
    if card.task is None:
        return f"- {card_words(card)}."
    return f"- {card_words(card)}, {KIND_WORDS.get(card_kind(card), A_CARD)}. {right_call(turn, card)}"


Turn = Callable[..., AsyncGenerator[Any, None]]


def grounds_the_cards(retrieval_first: Turn) -> Turn:
    """Wrap ``StreamingChatService._retrieval_first``: before the turn's first model
    call, the cards the owner's latest message names go in last, with the call for each."""
    @functools.wraps(retrieval_first)
    async def wrapped(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]], *args: Any,
                      **kwargs: Any) -> AsyncGenerator[Any, None]:
        note = _note_for(chat, latest_text)
        if note:
            llm_messages.append({"role": SYSTEM_ROLE, "content": note})
        async for frame in retrieval_first(chat, latest_text, llm_messages, *args, **kwargs):
            yield frame
    return wrapped


def _note_for(chat: Any, latest_text: str) -> str:
    """The note for this chat's turn; none for a widget visitor. Read in a savepoint, and
    a fault never stops the turn."""
    if getattr(chat, "widget_mode", False) or not latest_text:
        return ""
    try:
        with chat.db.begin_nested():
            return cards_note(chat.db, chat.workspace_id, latest_text)
    except Exception:
        logger.exception("[F241] could not name the owner's cards in the turn")
        return ""


__all__ = ["CARDS_NAMED", "cards_note", "grounds_the_cards"]
