"""platform_list_tasks: the board, one page at a time, with one total (F424).

Night 15: the action took no offset, so a board bigger than one page could not
be read in full, and the page's own length went out as ``total`` next to the
filter's count. Six identical asks in one second counted the review column as
66, 66, 66, 31, 66 and 13. Now ``total`` is the count for the filter on every
page, ``offset``/``next_offset`` walk the rest, and the order is total
(created_at, then id: Postgres stamps one transaction's rows with one
created_at), so consecutive pages never repeat or skip a card.

F426 (night 15): Auto could not find "the Guji blog draft" waiting in Review
without its number: the action had no search. ``search`` keeps the cards whose
title, or a Deliverable on them (its title or file name), holds every word that
names the work (services/named_work.py).
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import String, and_, cast, column, exists, or_, table
from sqlalchemy.orm import Session

from modules.tools.discovery.handlers_board_tasks import _widget_turn, task_visitor_view
from services.deliverable_tags import CARD_SOURCE_TYPE
from services.named_work import search_patterns
from services.ticket_refs import by_ticket_number

# The page size the model may ask for. "Close all the blocked tasks" needs to
# SEE them all; 50 hid 121 blocked tasks behind a page (2026-09-02).
MAX_LIST_TASKS_LIMIT = 200
DEFAULT_LIST_TASKS_LIMIT = 20
# The Deliverables registered on a card (source_type 'task', the card's id): what search reads.
CARD_DELIVERABLES = table(
    "deliverables", column("workspace_id"), column("source_type"), column("source_id"),
    column("title"), column("file_path"), column("deleted_at"),
)


def _page(params: Dict[str, Any]) -> Tuple[int, int]:
    """(offset, limit) from the call: offset never negative, limit 1..MAX."""
    limit = min(max(int(params.get("limit") or DEFAULT_LIST_TASKS_LIMIT), 1), MAX_LIST_TASKS_LIMIT)
    offset = max(int(params.get("offset") or 0), 0)
    return offset, limit


def _envelope(rows: List[Dict[str, Any]], total: int, offset: int, limit: int) -> Dict[str, Any]:
    """One page and where it sits: ``total`` is the filter's, never the page's."""
    after = offset + len(rows)
    return {"success": True, "tasks": rows, "total": total, "offset": offset, "limit": limit,
            "returned": len(rows), "next_offset": after if after < total else None}


def _agent_filter(db: Session, workspace_id: UUID, name: str) -> Optional[int]:
    """The id of the workspace's agent called ``name`` (any case), or None."""
    from sqlalchemy import func as sa_func

    from core.models import Agent

    agent = db.query(Agent).filter(
        Agent.workspace_id == workspace_id,
        sa_func.lower(Agent.name) == name.lower(),
    ).first()
    return agent.id if agent else None


def _filtered(query: Any, params: Dict[str, Any]) -> Any:
    """Status, priority, tags and search filters. BoardTask.tags is JSONB, so .contains([...])
    compiles to ``tags @> [...]``: a task carrying ALL the tags. HARNESS finds its
    own '[HARNESS]' tasks (tagged 'harness' + 'rx:{id}') this way."""
    from core.models.core import BoardTask

    if params.get("status"):
        query = query.filter(BoardTask.status == params["status"])
    if params.get("priority"):
        query = query.filter(BoardTask.priority == params["priority"])
    tags = params.get("tags")
    tags = [tags] if isinstance(tags, str) else tags
    if isinstance(tags, list) and tags:
        query = query.filter(BoardTask.tags.contains(tags))
    return _named(query, params.get("search"))


def _named(query: Any, raw: Any) -> Any:
    """F426: the cards whose title, or a Deliverable on them, holds every word of ``raw``."""
    from core.models.core import BoardTask

    d = CARD_DELIVERABLES.c
    for pattern in search_patterns(raw):
        on_card = exists().where(and_(
            d.workspace_id == BoardTask.workspace_id, d.source_type == CARD_SOURCE_TYPE,
            d.source_id == cast(BoardTask.id, String), d.deleted_at.is_(None),
            or_(d.title.ilike(pattern), d.file_path.ilike(pattern)),
        ))
        query = query.filter(or_(BoardTask.title.ilike(pattern), on_card))
    return query


def _agent_names(db: Session, tasks: List[Any]) -> Dict[int, str]:
    agent_ids = {t.assigned_agent_id for t in tasks if t.assigned_agent_id}
    if not agent_ids:
        return {}
    from core.models import Agent

    return {a.id: a.name for a in db.query(Agent).filter(Agent.id.in_(agent_ids)).all()}


def _owner_row(task: Any, agents: Dict[int, str]) -> Dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "status": task.status,
        "priority": task.priority,
        "tags": task.tags or [],
        "assigned_agent": agents.get(task.assigned_agent_id, "unassigned"),
        "created_at": str(task.created_at) if task.created_at else None,
        "started_at": str(task.started_at) if task.started_at else None,
        "completed_at": str(task.completed_at) if task.completed_at else None,
        "error_message": task.error_message,
    }


@by_ticket_number  # PRD-252 R4: takes #0042, answers with numbers
async def list_board_tasks(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """One page of the workspace's board tasks, newest first, with the filter's total."""
    from core.models.core import BoardTask

    offset, limit = _page(params)
    query = _filtered(db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id), params)
    agent_name = params.get("assigned_agent_name")
    if agent_name:
        agent_id = _agent_filter(db, workspace_id, agent_name)
        if agent_id is None and _widget_turn():
            # F155: a widget turn cannot tell a name no agent has from an agent
            # with no tickets: who the agents are is agents:read's, not tasks:read's.
            return _envelope([], 0, offset, limit)
        if agent_id is None:
            return {**_envelope([], 0, offset, limit), "note": f"No agent named '{agent_name}' found"}
        query = query.filter(BoardTask.assigned_agent_id == agent_id)

    total = int(query.count())
    tasks = query.order_by(BoardTask.created_at.desc(), BoardTask.id.desc()).offset(offset).limit(limit).all()
    if _widget_turn():
        return _envelope([task_visitor_view(t) for t in tasks], total, offset, limit)
    agents = _agent_names(db, tasks)
    return _envelope([_owner_row(t, agents) for t in tasks], total, offset, limit)
