"""F091-E1 (night 3): every card says whose job it belongs to.

Night 3's cards read "Agent #294 · board_task:612" or a bare tool name — the
persona could not tell which agent was asking or which ticket the answer would
move. Each listed grant now carries ``owner``: the agent (asker, else the
agent the call runs as) with its name, and the ticket — the board task it
blocks, or the ticket a gated call was made from — with its title. Two reads
for the whole list, never one per card.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from core.models.approval_grants import SUBJECT_BOARD_TASK


def _agent_of(grant: Any) -> Optional[int]:
    return getattr(grant, "asked_by_agent_id", None) or getattr(grant, "agent_id", None)


def _ticket_of(grant: Any) -> Optional[int]:
    if getattr(grant, "subject_type", None) == SUBJECT_BOARD_TASK and str(grant.subject_id).isdigit():
        return int(grant.subject_id)
    raw = (getattr(grant, "details", None) or {}).get("board_task_id")
    return int(raw) if raw is not None and str(raw).isdigit() else None


def grant_owners(db: Any, workspace_id: Any, grants: Iterable[Any]) -> Dict[int, Dict[str, Any]]:
    """``{grant id: {"agent": {id, name} | None, "ticket": {id, title} | None}}``."""
    from core.models.core import Agent, BoardTask

    rows: List[Any] = list(grants)
    agent_ids = {a for a in (_agent_of(g) for g in rows) if a}
    ticket_ids = {t for t in (_ticket_of(g) for g in rows) if t}
    names = dict(
        db.query(Agent.id, Agent.name).filter(Agent.id.in_(agent_ids), Agent.workspace_id == workspace_id).all()
    ) if agent_ids else {}
    titles = dict(
        db.query(BoardTask.id, BoardTask.title)
        .filter(BoardTask.id.in_(ticket_ids), BoardTask.workspace_id == workspace_id).all()
    ) if ticket_ids else {}
    owners: Dict[int, Dict[str, Any]] = {}
    for g in rows:
        agent, ticket = _agent_of(g), _ticket_of(g)
        owners[g.id] = {
            "agent": {"id": agent, "name": names.get(agent)} if agent else None,
            "ticket": {"id": ticket, "title": titles.get(ticket)} if ticket else None,
        }
    return owners
