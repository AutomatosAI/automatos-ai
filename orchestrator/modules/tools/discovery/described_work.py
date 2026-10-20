"""P256-FIX-N17-10 (F449, night 17): the made work the owner describes to change it, and its cards.

Ask 170, "make the guji quote 8kg fortnightly instead and tell cellar", filed #1754 "Read the existing
Guji quote from the Knowledge Base" while #1751, the quote, waited in Review; ask 219, "Make the
Tidewater quote shorter…", filed #1757 "Find the existing Tidewater quote in the knowledge base". Both
ran in new folders, could not open #1751, and left two quotes that disagree. A card the owner names by
its number was already guarded (``follows_the_owner._a_copy``); one they name by what it is was not.

``described`` reads "the <name> <noun>" from the owner's words (a piece of made work:
``named_work.WORK_NOUNS``) only when the same clause asks for a change (an edit word: the owner's
send-back, give and update verbs, ``AGAIN``, or ``EDITS``): "draft the newsletter intro" asks for new
work, "make the newsletter intro shorter" for a change. ``described_cards`` finds the cards that hold
that work with the board's own search (``board_task_list.named_filter``), newest first, never a
cancelled one, each with its agent and its newest Deliverable's file. Only ``_a_copy`` reads them: a
card named by what it is never joins the cards named by number (``owner_turn.cards_named``), so no
other rule judges a call against it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from sqlalchemy.orm import Session

from modules.tools.discovery.owner_turn import AGAIN, OTHER_VERB, NamedCard, seq_and_step
from services.named_work import WORK_NOUNS, search_words

# The owner asking for made work to change, beside their send-back, give and update verbs and AGAIN.
EDITS = re.compile(r"\bshort(?:er|en\w*)\b|\blonger\b"
                   r"|\b(?:change|revise|amend|edit|tweak|trim)\s+(?:(?:it|this|that|the|its)\b|#)"
                   r"|\bmake\b[^.;!?\n]*?\binstead\b", re.I)
# Where one clause of the owner's words ends: a sentence's end, a comma, "and".
CLAUSE_ENDS = re.compile(r"[.;!?\n]|, | and ", re.I)
# "the Guji quote", "the Hale & Co invoice": one to three words that name it (never "the"), then what it is.
THE_NAMED_WORK = re.compile(rf"\bthe\s+((?:(?!the\b)[\w'’&-]+\s+){{1,3}}?(?:{WORK_NOUNS}))\b", re.I)
DESCRIBED_LIMIT = 3
CANCELLED = "cancelled"
UNASSIGNED = "unassigned"


@dataclass(frozen=True)
class DescribedCard:
    """A card that holds the work the owner described: its number, the card, its agent's name and
    its newest Deliverable's file (None when it has none)."""

    ref: str
    task: Any
    agent: str
    file_path: Optional[str]

    def named(self) -> NamedCard:
        """The card as the owner's-words rules read a card the owner named."""
        seq, step = seq_and_step(self.ref)
        return NamedCard(ref=self.ref, seq=seq, step=step, task=self.task)

    def listed(self) -> str:
        """The card as one choice of several: "#1751 'Guji quote', QUOTER, review, 14 Oct"."""
        task = self.task
        title = str(getattr(task, "title", "") or "").strip()
        status = str(getattr(task, "status", "") or "").replace("_", " ")
        made = getattr(task, "created_at", None)
        when = f", {made.day} {made:%b}" if made is not None else ""
        return f"{self.ref} '{title}', {self.agent}, {status}{when}"


def asks_for_a_change(text: str) -> bool:
    """Whether ``text`` asks for something done to work that exists: an edit word."""
    return any(pattern.search(text or "") for pattern in (OTHER_VERB, AGAIN, EDITS))


def described(text: str) -> List[str]:
    """The search words of the first "the <name> <noun>" whose clause asks for a change to it; [] when
    no clause does ("draft the newsletter intro" asks for new work)."""
    for clause in CLAUSE_ENDS.split(text or ""):
        found = THE_NAMED_WORK.search(clause)
        if found and asks_for_a_change(clause):
            return search_words(found.group(1))
    return []


def described_cards(db: Session, workspace_id: Any, words: Sequence[str],
                    limit: int = DESCRIBED_LIMIT) -> List[DescribedCard]:
    """This workspace's cards holding every word of ``words`` on their title or a Deliverable on them
    (the board's search), cancelled ones left out, newest first, at most ``limit``. Read in a savepoint:
    a failed read must not abort the caller's transaction."""
    from core.models.core import BoardTask
    from modules.tools.discovery.board_task_list import agent_names, named_filter
    from services.ticket_numbers import ticket_numbers

    if not words:
        return []
    with db.begin_nested():
        query = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.status != CANCELLED)
        tasks = (named_filter(query, " ".join(words))
                 .order_by(BoardTask.created_at.desc(), BoardTask.id.desc()).limit(limit).all())
        numbers, agents = ticket_numbers(db, workspace_id, tasks), agent_names(db, tasks)
        return [DescribedCard(ref=numbers[task.id], task=task, agent=agents.get(task.assigned_agent_id, UNASSIGNED),
                              file_path=newest_file(db, workspace_id, task.id))
                for task in tasks if numbers.get(task.id)]


def newest_file(db: Session, workspace_id: Any, card_id: int) -> Optional[str]:
    """The file of the newest Deliverable on the card, or None when it has none."""
    from services.step_files import SESSION_FILE_SOURCE_TYPE, listed_or_raise

    listed = listed_or_raise(db, workspace_id, source_type=SESSION_FILE_SOURCE_TYPE, source_id=str(card_id), limit=1)
    rows = listed.get("deliverables") or []
    return (rows[0].get("file_path") or None) if rows else None


__all__ = ["DESCRIBED_LIMIT", "EDITS", "DescribedCard", "asks_for_a_change", "described", "described_cards",
           "newest_file"]
