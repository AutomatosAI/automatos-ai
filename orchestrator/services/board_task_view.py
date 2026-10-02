"""A board ticket as the board's API serves it (moved out of api/board_tasks.py).

Its columns, its agent's name and icon, (PRD-252 R3) why it waits in Review or
Blocked, as the codes ``core.services.ticket_reasons`` gives, and (R4) its
number, #0042 or a mission step's #0051.3.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from core.models import Agent
from core.models.core import BoardTask
from core.services.ticket_reasons import blocked_code, review_reason
from services.ticket_numbers import ticket_numbers


def board_dict(task: BoardTask, number: Optional[str] = None) -> Dict[str, Any]:
    """A ticket's columns, its number, and why it waits in Review or Blocked."""
    return {**task.to_dict(), "number": number, "review_reason": review_reason(task),
            "blocked_code": blocked_code(task)}


def enrich_with_agents(tasks: List[BoardTask], db: Session, workspace_id: Any) -> List[Dict[str, Any]]:
    """Join agent info onto task dicts.

    Agents are resolved within ``workspace_id`` only: a task whose
    ``assigned_agent_id`` points at another workspace's agent yields no ``agent``
    block rather than leaking that agent's name/icon (defense-in-depth tenant
    isolation — board reads are now reachable by per-workspace SDK keys).
    """
    numbers = ticket_numbers(db, workspace_id, tasks)
    agent_ids = {t.assigned_agent_id for t in tasks if t.assigned_agent_id}
    if not agent_ids:
        return [board_dict(t, numbers.get(t.id)) for t in tasks]

    agents = {
        a.id: a
        for a in db.query(Agent)
        .filter(Agent.id.in_(agent_ids), Agent.workspace_id == workspace_id)
        .all()
    }

    result = []
    for t in tasks:
        d = board_dict(t, numbers.get(t.id))
        agent = agents.get(t.assigned_agent_id)
        if agent:
            d["agent"] = {
                "id": agent.id,
                "name": agent.name,
                "agent_icon": getattr(agent, "premium_icon", None),
            }
        result.append(d)
    return result
