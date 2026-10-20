"""P256-FIX-N17-9 (night 17, F448) — the cards a ticket's owner names, and their files.

CELLAR's redo of #1754 carried the owner's "the quote you cannot find is card #1751", and
SHOPFRONT's answer said "It is card #1751 … the PDF is in Deliverables". Neither session
could open #1751's quote: no session tool read another card's files, and the host keeps a
session in its own folder (F042, PRD-245; that confinement is not widened here). CELLAR
wrote the quote from memory notes instead.

Only the OWNER's words name a card (``owner_text``): the brief, the owner's answers (never
the agent's question, nor the "You asked" block built from it) and this ticket's own
corrections. Never the standing rules from other cards, the sent-back draft, field memory, a
plan or the brand rules: a number there was written by an agent or about another card.
``named_cards`` reads "card #1751" / "ticket 1751" or the board's own "#1751" (four digits,
so a colour "#222222" or "#333" never matches, and not after "question", "order",
"invoice" and the like), resolves each in this workspace only and keeps the cards that have
Deliverables. The claim lists their files on the ticket (``step_files.named_files_block``)
and keeps them on ``runtime_ref.named_card_files``, which ``read_step_file`` reads.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from sqlalchemy.orm import Session

from services.step_files import NAMED_FILES_KEY

logger = logging.getLogger(__name__)

# "card #1751", "ticket 1751", "Card #0051.3": a card named as one, any case.
CARD_WORDS = re.compile(r"\b(?:card|ticket)\s+#?(\d{1,9}(?:\.\d+)?)", re.IGNORECASE)
# The board's own form, "#1751": exactly NUMBER_DIGITS digits after a '#' no word, '#' or '&' precedes.
BOARD_NUMBER = re.compile(r"(?<![\w#&])#(\d{4}(?:\.\d+)?)(?!\w)")
# A bare #NNNN this close after one of these words is that thing's number, not a card's.
NOT_A_CARD_BEFORE = re.compile(r"\b(?:question|ask|colou?r|hex|rgb|order|invoice)\b", re.IGNORECASE)
NOT_A_CARD_WINDOW = 24
# A brief may quote many numbers; past this many, the rest are not looked up.
NUMBERS_READ_MAX = 20


@dataclass(frozen=True)
class ClaimedFiles:
    """What one claim lists on its ticket: the prompt block and the rows it keeps."""

    block: str
    rows: Tuple[Dict[str, Any], ...]

    def on(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        """``ref`` (a new dict) holding exactly these rows, or none at all."""
        kept = {key: value for key, value in ref.items() if key != NAMED_FILES_KEY}
        return {**kept, NAMED_FILES_KEY: list(self.rows)} if self.rows else kept


def owner_text(task: Any) -> List[str]:
    """The owner's own words on ``task``: its brief, their answers to its questions and
    this ticket's corrections (``ticket_redo.owner_corrections``). A mission step's brief is
    the coordinator's, listing the other steps' tickets ("ticket #0051.3"), so it names none."""
    from services.cli_host_service import session_asks
    from services.ticket_redo import owner_corrections

    ref = task.runtime_ref if isinstance(getattr(task, "runtime_ref", None), dict) else {}
    data = task.planning_data if isinstance(getattr(task, "planning_data", None), dict) else {}
    in_a_mission = getattr(task, "orchestration_run_id", None) is not None
    texts = [] if in_a_mission else [getattr(task, "raw_prompt", None) or getattr(task, "description", None) or ""]
    texts += [str(ask.get("answer") or "") for ask in session_asks(ref)]
    texts += [str(entry.get("a") or "") for entry in data.get("human_qa") or [] if isinstance(entry, dict)]
    texts += owner_corrections(task)
    return [text for text in texts if text.strip()]


def numbers_named(texts: Iterable[str]) -> List[str]:
    """The card numbers ``texts`` name, as "#1751" / "#0051.3", in the order first said."""
    found: List[str] = []
    for text in texts:
        hits = [(m.start(), m.group(1)) for m in CARD_WORDS.finditer(text)]
        hits += [(m.start(), m.group(1)) for m in BOARD_NUMBER.finditer(text) if not _names_another_thing(text, m)]
        found += [f"#{digits}" for _, digits in sorted(hits)]
    return list(dict.fromkeys(found))[:NUMBERS_READ_MAX]


def _names_another_thing(text: str, match: "re.Match[str]") -> bool:
    before = text[max(0, match.start() - NOT_A_CARD_WINDOW):match.start()]
    return NOT_A_CARD_BEFORE.search(before) is not None


def named_cards(db: Session, workspace_id: Any, ticket_id: int, texts: Iterable[str]) -> List[int]:
    """The ids of the cards ``texts`` name in this workspace, this ticket left out, that
    have Deliverables; in the order they were named."""
    from services.ticket_numbers import resolve_ticket_ref

    ids = [resolve_ticket_ref(db, workspace_id, number) for number in numbers_named(texts)]
    wanted = list(dict.fromkeys(i for i in ids if i is not None and int(i) != int(ticket_id)))
    return [card_id for card_id in wanted if _has_files(db, workspace_id, card_id)]


def _has_files(db: Session, workspace_id: Any, card_id: int) -> bool:
    from services.step_files import SESSION_FILE_SOURCE_TYPE, listed_or_raise

    listed = listed_or_raise(db, workspace_id, source_type=SESSION_FILE_SOURCE_TYPE, source_id=str(card_id), limit=1)
    return int(listed.get("total") or 0) > 0


def claimed_files(db: Session, task: Any, carried: Sequence[Dict[str, Any]]) -> ClaimedFiles:
    """The named-card files one claim lists: this claim's, with the ``carried`` rows of
    earlier claims kept (by Deliverable id), except after a brief the owner agreed anew.
    Read BEFORE the prompt is built, which consumes ``review_feedback``. Never fails a
    claim: a lookup that breaks lists none and is logged."""
    from services.step_files import (NAMED_CARDS_LISTED_MAX, named_card_files, named_files_block,
                                     named_files_on, named_rows)
    from services.ticket_redo import BRIEF_AGREED

    texts = owner_text(task)
    fresh: List[Any] = []
    cards: List[int] = []
    if numbers_named(texts):
        try:
            with db.begin_nested():   # a failed lookup rolls back only itself, never the claim
                cards = named_cards(db, task.workspace_id, int(task.id), texts)
                fresh = named_card_files(db, workspace_id=task.workspace_id, ticket_id=int(task.id),
                                         card_ids=cards)
        except Exception:
            logger.exception("[F448] ticket %s: the cards its owner named could not be listed", task.id)
            cards, fresh = [], []
    agreed_anew = getattr(task, "review_feedback", None) == BRIEF_AGREED
    rows = _union(named_rows(fresh), () if agreed_anew else carried)
    block = named_files_block(named_files_on({NAMED_FILES_KEY: rows}),
                              cards_not_listed=max(0, len(cards) - NAMED_CARDS_LISTED_MAX))
    return ClaimedFiles(block=block, rows=tuple(rows))


def _union(fresh: Sequence[Dict[str, Any]], carried: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """This claim's rows, then the carried rows it does not already hold, by Deliverable id."""
    held = {row.get("deliverable_id") for row in fresh}
    return [*fresh, *(dict(row) for row in carried if isinstance(row, dict) and row.get("deliverable_id")
                      and row.get("deliverable_id") not in held)]
