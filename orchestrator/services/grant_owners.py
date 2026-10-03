"""F091-E1 (night 3): every card says whose job it belongs to.

Night 3's cards read "Agent #294 · board_task:612" or a bare tool name — the
persona could not tell which agent was asking or which ticket the answer would
move. Each listed grant now carries ``owner``: the agent (asker, else the
agent the call runs as) with its name, and the ticket — the board task it
blocks, or the ticket a gated call was made from — with its title. Two reads
for the whole list, never one per card.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Set
from uuid import UUID

from core.models.approval_grants import SUBJECT_BOARD_TASK, SUBJECT_TOOL_CALL


def _agent_of(grant: Any) -> Optional[int]:
    return getattr(grant, "asked_by_agent_id", None) or getattr(grant, "agent_id", None)


def _ticket_of(grant: Any) -> Optional[int]:
    if getattr(grant, "subject_type", None) == SUBJECT_BOARD_TASK and str(grant.subject_id).isdigit():
        return int(grant.subject_id)
    raw = (getattr(grant, "details", None) or {}).get("board_task_id")
    return int(raw) if raw is not None and str(raw).isdigit() else None


def grant_owners(db: Any, workspace_id: Any, grants: Iterable[Any]) -> Dict[int, Dict[str, Any]]:
    """``{grant id: {"agent": {id, name} | None, "ticket": {id, title, number} | None}}``.
    PRD-252 R4: the ticket's ``number`` (#0042) is read with its title."""
    from core.models.core import Agent

    rows: List[Any] = list(grants)
    agent_ids = {a for a in (_agent_of(g) for g in rows) if a}
    step_cards = _step_cards(db, workspace_id, rows)
    ticket_of = {g.id: _ticket_of(g) or step_cards.get(str(g.subject_id)) for g in rows}
    ticket_ids = {t for t in ticket_of.values() if t}
    names = dict(
        db.query(Agent.id, Agent.name).filter(Agent.id.in_(agent_ids), Agent.workspace_id == workspace_id).all()
    ) if agent_ids else {}
    tickets = _tickets(db, workspace_id, ticket_ids)
    owners: Dict[int, Dict[str, Any]] = {}
    for g in rows:
        agent, ticket = _agent_of(g), ticket_of[g.id]
        known = tickets.get(ticket, {})
        owners[g.id] = {
            "agent": {"id": agent, "name": names.get(agent)} if agent else None,
            "ticket": {"id": ticket, "title": known.get("title"), "number": known.get("number")} if ticket else None,
        }
    return owners


def _step_cards(db: Any, workspace_id: Any, grants: List[Any]) -> Dict[str, int]:
    """A mission step's question is staged on its task (the clarification ladder:
    subject ``tool_call``, the orchestration task's id); its ticket is the step's
    card. F246: question #1178 from #0139.1 was listed with no card number."""
    from core.models.core import BoardTask

    task_ids = {_uuid(g.subject_id) for g in grants if getattr(g, "subject_type", None) == SUBJECT_TOOL_CALL}
    task_ids.discard(None)
    if not task_ids:
        return {}
    cards = (
        db.query(BoardTask.id, BoardTask.orchestration_task_id)
        .filter(BoardTask.orchestration_task_id.in_(task_ids), BoardTask.workspace_id == workspace_id).all()
    )
    return {str(task_id): card_id for card_id, task_id in cards}


def _uuid(value: Any) -> Optional[UUID]:
    """``value`` as a UUID, or None: a gated tool call's subject is a call hash."""
    try:
        return UUID(str(value))
    except ValueError:
        return None


def _tickets(db: Any, workspace_id: Any, ticket_ids: Set[int]) -> Dict[int, Dict[str, Any]]:
    """Each ticket's title and number, in one read (a mission step's number needs its card's)."""
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_numbers

    if not ticket_ids:
        return {}
    rows = [
        SimpleNamespace(id=r[0], title=r[1], workspace_seq=r[2], source_type=r[3], parent_task_id=r[4])
        for r in db.query(BoardTask.id, BoardTask.title, BoardTask.workspace_seq, BoardTask.source_type,
                          BoardTask.parent_task_id)
        .filter(BoardTask.id.in_(ticket_ids), BoardTask.workspace_id == workspace_id).all()
    ]
    numbers = ticket_numbers(db, workspace_id, rows)
    return {r.id: {"title": r.title, "number": numbers.get(r.id)} for r in rows}
