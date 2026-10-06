"""A board ticket as the board's API serves it (moved out of api/board_tasks.py).

Its columns, its agent's name and icon, (PRD-252 R3) why it waits in Review or
Blocked, as the codes ``core.services.ticket_reasons`` gives, and (R4) its
number, #0042 or a mission step's #0051.3. PRE-11 (7 Oct): whether the owner added
its answer to knowledge (``knowledge_document_id``, services/owner_knowledge.py).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from core.models import Agent
from core.models.core import BoardTask
from core.services.ticket_reasons import blocked_code, review_reason
from services.owner_knowledge import CARD_TAG, with_knowledge_ids
from services.ticket_numbers import ticket_numbers
from services.ticket_redo import kept_draft, times_sent_back


def board_dict(task: BoardTask, number: Optional[str] = None) -> Dict[str, Any]:
    """A ticket's columns, its number, why it waits in Review or Blocked, how
    often it was sent back (R2: the review panel suggests Discuss after three),
    and (F243) the draft a failed redo keeps on the card's face."""
    return {**task.to_dict(), "number": number, "review_reason": review_reason(task),
            "blocked_code": blocked_code(task), "times_sent_back": times_sent_back(task),
            "kept_draft": kept_draft(task)}


def enrich_with_agents(tasks: List[BoardTask], db: Session, workspace_id: Any) -> List[Dict[str, Any]]:
    """Join agent info onto task dicts, and (PRE-11) each card's
    ``knowledge_document_id``: the owner's copy of its answer, or None.

    Agents are resolved within ``workspace_id`` only: a task whose
    ``assigned_agent_id`` points at another workspace's agent yields no ``agent``
    block rather than leaking that agent's name/icon (defense-in-depth tenant
    isolation — board reads are now reachable by per-workspace SDK keys).
    """
    numbers = ticket_numbers(db, workspace_id, tasks)
    agents = _workspace_agents(db, workspace_id, tasks)
    shown = [_with_agent(board_dict(t, numbers.get(t.id)), agents.get(t.assigned_agent_id)) for t in tasks]
    return with_knowledge_ids(db, workspace_id, shown, CARD_TAG)


def _workspace_agents(db: Session, workspace_id: Any, tasks: List[BoardTask]) -> Dict[int, Any]:
    """The tickets' agents, by id, from this workspace only."""
    agent_ids = {t.assigned_agent_id for t in tasks if t.assigned_agent_id}
    if not agent_ids:
        return {}
    return {
        a.id: a
        for a in db.query(Agent)
        .filter(Agent.id.in_(agent_ids), Agent.workspace_id == workspace_id)
        .all()
    }


def _with_agent(served: Dict[str, Any], agent: Any) -> Dict[str, Any]:
    if not agent:
        return served
    return {**served, "agent": {"id": agent.id, "name": agent.name,
                                "agent_icon": getattr(agent, "premium_icon", None)}}
