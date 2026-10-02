"""PRD-252 R4 — a ticket's number as people and Auto say it: #0042.

A ticket's number is ``workspace_seq``, given on insert
(``core/models/ticket_numbers.py``). A mission step has none of its own: it is
its mission card's number and its step, #0051.3 (D5). Every surface that names
a ticket (the board, Needs you, the feed, notifications, chat cards and Auto's
replies) formats it here, and Auto's ticket tools take it back here
(``resolve_ticket_ref``) wherever they take a ticket's id.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

from sqlalchemy.orm import Session

from core.models.core import BoardTask
from core.models.ticket_numbers import STEP_SOURCE

NUMBER_DIGITS = 4
# "#0042", "#42", "#0051.3"; a leading '#' says it is a number, not an id.
_NUMBER_REF = re.compile(r"^#(\d+)(?:\.(\d+))?$")


def format_number(seq: Optional[int], step: Optional[int] = None) -> Optional[str]:
    """``#0042``, or ``#0051.3`` for step 3 of ticket #0051; None without a number."""
    if seq is None:
        return None
    number = f"#{int(seq):0{NUMBER_DIGITS}d}"
    return f"{number}.{step}" if step is not None else number


def step_of(task: Any) -> Optional[int]:
    """A mission step's place in its mission (1, 2, …), from its planning data."""
    value = (getattr(task, "planning_data", None) or {}).get("sequence_number")
    return int(value) if isinstance(value, int) or str(value).isdigit() else None


def ticket_numbers(db: Session, workspace_id: Any, tasks: Iterable[Any]) -> Dict[int, Optional[str]]:
    """Each ticket's number, by id. A mission step's comes from its mission card's."""
    tasks = list(tasks)
    steps = [t for t in tasks if getattr(t, "source_type", None) == STEP_SOURCE]
    parent_ids = {t.parent_task_id for t in steps if getattr(t, "parent_task_id", None)}
    parents = dict(
        db.query(BoardTask.id, BoardTask.workspace_seq)
        .filter(BoardTask.id.in_(parent_ids), BoardTask.workspace_id == workspace_id).all()
    ) if parent_ids else {}
    numbers = {t.id: format_number(getattr(t, "workspace_seq", None)) for t in tasks}
    return {**numbers, **{t.id: format_number(parents.get(getattr(t, "parent_task_id", None)), step_of(t))
                          for t in steps}}


def ticket_number(db: Session, task: Any) -> Optional[str]:
    """One ticket's number (see ``ticket_numbers``); a step's needs its workspace."""
    workspace_id = getattr(task, "workspace_id", None)
    if workspace_id is None:
        return None if getattr(task, "source_type", None) == STEP_SOURCE else format_number(
            getattr(task, "workspace_seq", None))
    return ticket_numbers(db, workspace_id, [task]).get(task.id)


def ticket_label(task: Any, number: Optional[str] = None, *, capital: bool = False) -> str:
    """How a message names a ticket: "ticket #0042", or "ticket 612" for one with no
    number of its own here (a mission step, a row from before numbering). Never
    "#612": a '#' now means a number, and #612 may be another ticket's."""
    number = number or (format_number(getattr(task, "workspace_seq", None))
                        if getattr(task, "source_type", None) != STEP_SOURCE else None)
    label = f"ticket {number}" if number else f"ticket {task.id}"
    return label[0].upper() + label[1:] if capital else label


def ticket_label_for(db: Session, workspace_id: Any, task_id: Any, *, capital: bool = False) -> str:
    """``ticket_label`` for a ticket known only by its id (a step's from its mission card)."""
    task = db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == workspace_id).first()
    if task is None:
        return f"{'T' if capital else 't'}icket {task_id}"
    return ticket_label(task, ticket_number(db, task), capital=capital)


def is_number_ref(ref: Any) -> bool:
    """True for "#0042" or "#0051.3": a number, not an id."""
    return isinstance(ref, str) and bool(_NUMBER_REF.match(ref.strip()))


def resolve_ticket_ref(db: Session, workspace_id: Any, ref: Any) -> Optional[int]:
    """The id of the ticket ``ref`` names in this workspace: "#0042", or "#0051.3"
    for a mission step. None when no ticket has that number."""
    match = _NUMBER_REF.match(str(ref).strip())
    if not match:
        return None
    seq, step = int(match.group(1)), match.group(2)
    numbered = db.query(BoardTask.id).filter(
        BoardTask.workspace_id == workspace_id, BoardTask.workspace_seq == seq,
    ).first()
    if numbered is None or step is None:
        return numbered.id if numbered else None
    # A retried step can share its place; the newest is the one in play.
    steps = db.query(BoardTask).filter(
        BoardTask.workspace_id == workspace_id, BoardTask.parent_task_id == numbered.id,
        BoardTask.source_type == STEP_SOURCE,
    ).order_by(BoardTask.id.desc()).all()
    return next((t.id for t in steps if step_of(t) == int(step)), None)
