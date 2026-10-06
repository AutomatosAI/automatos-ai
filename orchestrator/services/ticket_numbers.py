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
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session, object_session
from sqlalchemy.orm.exc import UnmappedInstanceError

from core.models.core import BoardTask
from core.models.ticket_numbers import STEP_SOURCE

NUMBER_DIGITS = 4
# "#0042", "#42", "#0051.3"; a leading '#' says it is a number, not an id. So do a
# leading zero ("0175") and a step ("105.4"): an id has neither (F241).
_DIGITS = re.compile(r"^(\d+)(?:\.(\d+))?$")
# A board number searched for fits the column: no workspace has a billion tickets.
MAX_SEARCHED_DIGITS = 9
# F241 (night 7): Auto's tools were called with "#0175" as 175 (a number with its
# '#' and zeros gone) as often as with an id. Gerard, 7 Oct: "same number everywhere": bare
# digits a person or an agent gives are the board's number (``read_bare_refs``).
# F241 (night 7b): a step's number sent as a JSON number ("task_id": 188.3).
_FRACTION = re.compile(r"^(\d+)\.(\d+)$")
FRACTION_DROPPED_A_ZERO = ("{said} could be {short} or {long}: sent as a number, it lost the 0 at its end. "
                           "Nothing was done. Give the step's number as text with its '#', as the board shows it.")


class TicketId(int):
    """A ticket's database id, as the platform's own code passes it in a structured call (the card
    a tool already holds). It is read as the id, never as a board number: a bare number a person or
    an agent gives is the board's number (``read_bare_refs``)."""


def format_number(seq: Optional[int], step: Optional[int] = None) -> Optional[str]:
    """``#0042``, or ``#0051.3`` for step 3 of ticket #0051; None without a number."""
    if seq is None:
        return None
    number = f"#{int(seq):0{NUMBER_DIGITS}d}"
    return f"{number}.{step}" if step is not None else number


def _steps_in_order(db: Session, workspace_id: Any, card_ids: Iterable[Any]) -> Dict[Any, List[int]]:
    """Each mission card's step ids, in the order they were filed."""
    rows = db.query(BoardTask.id, BoardTask.parent_task_id).filter(
        BoardTask.parent_task_id.in_(set(card_ids)), BoardTask.workspace_id == workspace_id,
        BoardTask.source_type == STEP_SOURCE,
    ).order_by(BoardTask.id).all()
    ordered: Dict[Any, List[int]] = {}
    for step_id, card_id in rows:
        ordered.setdefault(card_id, []).append(step_id)
    return ordered


def ticket_numbers(db: Session, workspace_id: Any, tasks: Iterable[Any]) -> Dict[int, Optional[str]]:
    """Each ticket's number, by id. A mission step's is its card's number and its
    place among the card's steps in the order they were filed (#0051.3): unique,
    where the plan's sequence number is shared by steps that run side by side,
    and fixed, because a mission only ever adds steps."""
    tasks = list(tasks)
    steps = [t for t in tasks if getattr(t, "source_type", None) == STEP_SOURCE]
    card_ids = {t.parent_task_id for t in steps if getattr(t, "parent_task_id", None)}
    cards = dict(
        db.query(BoardTask.id, BoardTask.workspace_seq)
        .filter(BoardTask.id.in_(card_ids), BoardTask.workspace_id == workspace_id).all()
    ) if card_ids else {}
    order = _steps_in_order(db, workspace_id, card_ids) if card_ids else {}
    numbers = {t.id: format_number(getattr(t, "workspace_seq", None)) for t in tasks}
    for t in steps:
        siblings = order.get(getattr(t, "parent_task_id", None), [])
        place = siblings.index(t.id) + 1 if t.id in siblings else None
        numbers[t.id] = format_number(cards.get(getattr(t, "parent_task_id", None)), place) if place else None
    return numbers


def ticket_number(db: Session, task: Any) -> Optional[str]:
    """One ticket's number (see ``ticket_numbers``); a step's needs its workspace."""
    workspace_id = getattr(task, "workspace_id", None)
    if workspace_id is None:
        return None if getattr(task, "source_type", None) == STEP_SOURCE else format_number(
            getattr(task, "workspace_seq", None))
    return ticket_numbers(db, workspace_id, [task]).get(task.id)


def number_of(task: Any) -> Optional[str]:
    """One ticket's number with no list to read it from: its own, or a mission
    step's from its card, read through the session the step was loaded in, as a
    relationship loads its parent. A list numbers its tickets in one read
    (``ticket_numbers``) instead. None for a step outside a session."""
    if getattr(task, "source_type", None) != STEP_SOURCE:
        return format_number(getattr(task, "workspace_seq", None))
    db = _session_of(task)
    return ticket_number(db, task) if db is not None else None


def _session_of(task: Any) -> Optional[Session]:
    try:
        return object_session(task)
    except UnmappedInstanceError:  # a plain object standing in for a ticket
        return None


def ticket_label(task: Any, number: Optional[str] = None, *, capital: bool = False) -> str:
    """How a message names a ticket: "ticket #0042" ("ticket #0051.3" for a mission
    step), or "ticket 612" for one with no number (a row from before numbering).
    Never "#612": a '#' now means a number, and #612 may be another ticket's."""
    number = number or number_of(task)
    label = f"ticket {number}" if number else f"ticket {task.id}"
    return label[0].upper() + label[1:] if capital else label


def ticket_label_for(db: Session, workspace_id: Any, task_id: Any, *, capital: bool = False) -> str:
    """``ticket_label`` for a ticket known only by its id (a step's from its mission card)."""
    task = db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == workspace_id).first()
    if task is None:
        return f"{'T' if capital else 't'}icket {task_id}"
    return ticket_label(task, ticket_number(db, task), capital=capital)


def title_or_number(search: str) -> Any:
    """The board search's filter: a ticket whose title has ``search`` in it or, when ``search`` is a
    ticket's number ("#0892", "0892" or "892"), the ticket with that board number (Gerard, 7 Oct:
    "same number everywhere")."""
    titled = BoardTask.title.ilike(f"%{search}%")
    seq = _whole_number(search)
    return or_(titled, BoardTask.workspace_seq == seq) if seq is not None else titled


def _whole_number(text: Any) -> Optional[int]:
    """892 for "#0892", "0892" or "892"; None for anything else, a step's number included."""
    said = str(text).strip()
    match = _DIGITS.match(said[1:] if said.startswith("#") else said)
    if not match or match.group(2) is not None or len(match.group(1)) > MAX_SEARCHED_DIGITS:
        return None
    return int(match.group(1))


def _number_parts(ref: Any) -> Optional[Tuple[int, Optional[int]]]:
    """(number, step) for a ticket named by its number: "#0042", "#0051.3", "0175"
    or "105.4". None for anything else, an id ("42", 42) included."""
    text = str(ref).strip()
    hashed = text.startswith("#")
    match = _DIGITS.match(text[1:] if hashed else text)
    if not match or isinstance(ref, (int, float)):
        return None
    seq, step = match.group(1), match.group(2)
    if hashed or step is not None or (len(seq) > 1 and seq.startswith("0")):
        return int(seq), (int(step) if step is not None else None)
    return None


def is_number_ref(ref: Any) -> bool:
    """True for "#0042", "#0051.3", "0175" or "105.4": a number, not an id."""
    return isinstance(ref, str) and _number_parts(ref) is not None


def is_bare_ref(ref: Any) -> bool:
    """True for 175 or "175": a number without its '#' (or, when no ticket has that number, an id).
    False for a ``TicketId``: the platform's own id is never read as a number."""
    if isinstance(ref, (bool, TicketId)):
        return False
    return isinstance(ref, int) or (isinstance(ref, str) and ref.strip().isdigit() and not is_number_ref(ref))


def spoken_ref(db: Session, workspace_id: Any, ref: Any) -> Tuple[Any, Optional[str]]:
    """``ref`` as the ticket tools read it, or why it can't be read.

    F241 (night 7b): Auto sent #0188.3 as the JSON number 188.3, and it was read as
    ticket 188. An id is never fractional, so a fraction is a mission step's number
    with its '#' gone. The number has dropped any 0 at its end, so 176.1 is refused
    when ticket #0176 has a tenth step. A whole number (188.0) is that integer."""
    if isinstance(ref, bool) or not isinstance(ref, float):
        return ref, None
    if ref.is_integer():
        return int(ref), None
    match = _FRACTION.match(repr(ref))
    if not match:
        return ref, None
    seq, step = int(match.group(1)), match.group(2)
    if len(step) == 1 and _step_count(db, workspace_id, seq) >= int(step) * 10:
        return ref, FRACTION_DROPPED_A_ZERO.format(said=repr(ref), short=format_number(seq, int(step)),
                                                   long=format_number(seq, int(step) * 10))
    return format_number(seq, int(step)), None


def _step_count(db: Session, workspace_id: Any, seq: int) -> int:
    """How many steps ticket ``seq`` (a mission's card) has."""
    card = db.query(BoardTask.id).filter(BoardTask.workspace_id == workspace_id,
                                         BoardTask.workspace_seq == seq).first()
    return len(_steps_in_order(db, workspace_id, [card.id]).get(card.id, [])) if card else 0


def read_bare_refs(db: Session, workspace_id: Any, refs: Iterable[Any]) -> Dict[int, int]:
    """The ticket ids that bare refs (892, "892") name in this workspace, by ref. A ref that names
    no ticket is left out.

    A bare number is the board's number: #0892, in Auto's chat, in the platform tools and in a
    session alike (Gerard, 7 Oct: "same number everywhere, it has to be easy"). It is read as an id
    only when no ticket in the workspace has that number (an id a tool's answer gave).

    F369 (night 10c, chat 16bb619c): "tag that ticket", the one Auto had just called #0892, went as
    task_id 892, also the id of #0708; the tie was refused naming both, and Auto asked the owner
    which they meant. The number is the one meant, everywhere."""
    wanted = {int(str(r).strip()) for r in refs}
    if not wanted:
        return {}
    rows = db.query(BoardTask.id, BoardTask.workspace_seq).filter(
        BoardTask.workspace_id == workspace_id,
        or_(BoardTask.id.in_(sorted(wanted)), BoardTask.workspace_seq.in_(sorted(wanted))),
    ).all()
    as_ids = {r.id: r.id for r in rows if r.id in wanted}
    as_numbers = {r.workspace_seq: r.id for r in rows if getattr(r, "workspace_seq", None) in wanted}
    return {**as_ids, **as_numbers}


def resolve_ticket_ref(db: Session, workspace_id: Any, ref: Any) -> Optional[int]:
    """The id of the ticket ``ref`` names in this workspace: "#0042" (or "0042"),
    or "#0051.3" for a mission step. None when no ticket has that number."""
    parts = _number_parts(ref)
    if parts is None:
        return None
    seq, step = parts
    numbered = db.query(BoardTask.id).filter(
        BoardTask.workspace_id == workspace_id, BoardTask.workspace_seq == seq,
    ).first()
    if numbered is None or step is None:
        return numbered.id if numbered else None
    siblings = _steps_in_order(db, workspace_id, [numbered.id]).get(numbered.id, [])
    place = int(step)
    return siblings[place - 1] if 0 < place <= len(siblings) else None
