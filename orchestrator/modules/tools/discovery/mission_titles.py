"""PRD-256 FX-009 (night 12, B3): a mission tool given a mission's title acts on that mission.

The owner names a mission by its number (#0992) or by what it is for ("the spring menu
mission"). A number is read as the ticket tools read it (``mission_refs``); anything else
in ``mission_id`` that is not a mission's id is a title. It names a mission when exactly
one mission card in the workspace has it: the card's own title ("Mission: Plan the
spring menu"), the title without "Mission: ", or, when none matches in full, a title
that holds the words. Several matches are never guessed between: the call is refused,
listing them by number, so the owner can say which. Only this workspace's cards are read.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

MISSION_CARD = "orchestration"
TITLE_PREFIX = "Mission: "
MAX_LISTED = 5
NO_MISSION_TITLED = ("No mission titled '{said}' in this workspace, and '{said}' is not a card's number, so nothing "
                     "was done. platform_list_missions lists each mission with its card's number (#0188).")
SEVERAL_TITLED = ("More than one mission is titled like '{said}': {listed}. Nothing was done: ask the owner which "
                  "one, then call this again with its number as mission_id.")
LISTED = "{number} '{title}'"


def mission_card_titled(db: Session, workspace_id: Any, said: str) -> Tuple[Optional[Any], Optional[str]]:
    """The one mission card in this workspace titled ``said``, or why there is none. A None
    session (a validation-only caller) reads nothing."""
    text = " ".join(str(said or "").split())
    if not text or db is None:
        return None, NO_MISSION_TITLED.format(said=said)
    cards = _titled(db, workspace_id, text, whole=True) or _titled(db, workspace_id, text, whole=False)
    if len(cards) == 1:
        return cards[0], None
    if not cards:
        return None, NO_MISSION_TITLED.format(said=text)
    return None, SEVERAL_TITLED.format(said=text, listed=_listed(db, workspace_id, cards[:MAX_LISTED]))


def _titled(db: Session, workspace_id: Any, text: str, *, whole: bool) -> List[Any]:
    """This workspace's mission cards whose title is ``text`` (``whole``) or holds it,
    newest first; one more than are ever listed, so several are told from a few."""
    from core.models.core import BoardTask

    title = func.lower(BoardTask.title)
    lowered = text.lower()
    matches = (title.in_([lowered, f"{TITLE_PREFIX}{text}".lower()]) if whole
               else title.contains(lowered, autoescape=True))
    return (db.query(BoardTask)
            .filter(BoardTask.workspace_id == workspace_id, BoardTask.source_type == MISSION_CARD, matches)
            .order_by(BoardTask.id.desc())
            .limit(MAX_LISTED + 1)
            .all())


def _listed(db: Session, workspace_id: Any, cards: List[Any]) -> str:
    """"#0992 'Mission: Plan the spring menu', #0990 'Mission: Plan the spring menu v2'"."""
    from services.ticket_numbers import ticket_numbers

    numbers = ticket_numbers(db, workspace_id, cards)
    return ", ".join(LISTED.format(number=numbers.get(card.id) or f"ticket {card.id}", title=card.title)
                     for card in cards)


__all__ = ["mission_card_titled"]
