"""F460 (night 18): the work the owner names reaches the model by its card.

"Show me the Taunton quote." read no card: F317's board read runs only for a question, and "show me"
is not one. The model searched the documents and printed the wholesale accounts sheet, which mentions
Taunton, while #1836 and #1837, the quotes, waited in Review. "Did the Bell Yard invoice come back?"
searched Gmail, the shop database and three documents and never the board, where #1848 held it.

Now when the owner's words name a piece of made work ("the Taunton quote", "the Bell Yard invoice":
``described_work.work_named``), the turn reads the cards that hold it with the board's own search
(``described_work.described_cards``: newest first, never a cancelled one) and gives the model ONE note
naming each card's number, title, agent, status and file, with the rule to show it from that card
first. A message that asks for new work ("Write the November price letter") gets no note: an older
card is not what it asked for. A widget visitor's turn reads nothing: the board is the owner's (F155).
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

READ_TOOL = "platform_list_tasks"
NOTE_HEAD = ("The owner named a piece of work the team made. The board holds it on these cards, newest first, "
             "read just now for this message:")
ROW = "- {ref} '{title}', {agent}, {status}, file: {file}"
NO_FILE = "none yet"
NOTE_RULE = ("Show or check that work from its card first: read the card with platform_get_task and its file with "
             "workspace_read_file, and name the card's number. A document that only mentions the same name is "
             "not this work.")
TITLE_CHARS = 100
# A message asking for new work: write, draft, create, prepare or put together, at its start.
_MAKES_NEW = re.compile(r"^\s*(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
                        r"(?:write|draft|create|prepare|put together)\b", re.I)


def _searched_for(chat: Any, latest_text: str) -> List[str]:
    """The search words of the work the owner names, or [] when the turn should read nothing."""
    from modules.tools.discovery.described_work import asks_for_a_change, work_named

    if getattr(chat, "widget_mode", False):
        return []
    if _MAKES_NEW.match(latest_text or "") and not asks_for_a_change(latest_text):
        return []
    return work_named(latest_text)


def _cards(chat: Any, words: List[str]) -> List[Any]:
    """The cards that hold the work; [] when the board can't be read (logged)."""
    from modules.tools.discovery.described_work import described_cards

    try:
        return described_cards(chat.db, UUID(str(chat.workspace_id)), words)
    except Exception:
        logger.exception("[F460] the cards holding the named work could not be read for this turn")
        return []


def named_work_note(cards: List[Any]) -> str:
    """What the model reads: each card that holds the work, then the rule."""
    rows = [ROW.format(ref=card.ref, title=str(getattr(card.task, "title", "") or "")[:TITLE_CHARS],
                       agent=card.agent, status=str(getattr(card.task, "status", "") or "").replace("_", " "),
                       file=card.file_path or NO_FILE)
            for card in cards]
    return "\n".join([NOTE_HEAD, *rows, NOTE_RULE])


def reads_the_named_work(chat: Any, latest_text: str, prefetched: List[Any]) -> Optional[str]:
    """The note for the work the owner names, or None. The read goes in ``prefetched``."""
    words = _searched_for(chat, latest_text)
    cards = _cards(chat, words) if words else []
    if not cards:
        return None
    prefetched.append((READ_TOOL, {"automatic": True, "search": " ".join(words)}))
    logger.info("[F460] %d card(s) hold the work the owner named", len(cards))
    return named_work_note(cards)


__all__ = ["NOTE_HEAD", "NOTE_RULE", "named_work_note", "reads_the_named_work"]
