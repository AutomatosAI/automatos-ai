"""Board task CRUD, assignment, and status handlers for PlatformActionExecutor (PRD-72)."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def create_board_task(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a board task (called by agents via platform_create_task)."""
    from core.models.core import BoardTask

    title = params.get("title")
    description = params.get("description")
    if not title or not description:
        return {"success": False, "error": "title and description are required"}

    # Resolve assigned agent by name
    assigned_agent_id = None
    agent_name = params.get("assigned_agent_name")
    if agent_name:
        from core.models import Agent
        from sqlalchemy import func as sa_func
        agent = db.query(Agent).filter(
            Agent.workspace_id == workspace_id,
            sa_func.lower(Agent.name) == agent_name.lower(),
        ).first()
        if agent:
            assigned_agent_id = agent.id

    # Build planning_data if approval_action or other planning fields provided
    planning_data = params.get("planning_data")
    if not planning_data and params.get("approval_action"):
        planning_data = {"approval_action": params["approval_action"]}

    # Determine initial status — tasks with approval_action go to review
    initial_status = params.get("status", "assigned" if assigned_agent_id else "inbox")
    if planning_data and planning_data.get("approval_action"):
        initial_status = "review"

    task = BoardTask(
        workspace_id=workspace_id,
        title=title,
        description=description,
        priority=params.get("priority", "medium"),
        assigned_agent_id=assigned_agent_id,
        status=initial_status,
        created_by_type="agent",
        created_by_id=str(params.get("_agent_id", "")),
        parent_task_id=params.get("parent_task_id"),
        tags=params.get("tags", []),
        planning_data=planning_data,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    return {
        "success": True,
        "task_id": task.id,
        "status": task.status,
        "title": task.title,
    }


async def list_board_tasks(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List board tasks with optional filters."""
    from core.models.core import BoardTask

    query = db.query(BoardTask).filter(
        BoardTask.workspace_id == workspace_id,
    )

    status = params.get("status")
    if status:
        query = query.filter(BoardTask.status == status)

    priority = params.get("priority")
    if priority:
        query = query.filter(BoardTask.priority == priority)

    agent_name = params.get("assigned_agent_name")
    if agent_name:
        from core.models import Agent
        from sqlalchemy import func as sa_func
        agent = db.query(Agent).filter(
            Agent.workspace_id == workspace_id,
            sa_func.lower(Agent.name) == agent_name.lower(),
        ).first()
        if agent:
            query = query.filter(BoardTask.assigned_agent_id == agent.id)
        else:
            return {"success": True, "tasks": [], "total": 0, "note": f"No agent named '{agent_name}' found"}

    limit = min(int(params.get("limit", 20)), 50)
    tasks = query.order_by(BoardTask.created_at.desc()).limit(limit).all()

    # Enrich with agent names
    agent_ids = {t.assigned_agent_id for t in tasks if t.assigned_agent_id}
    agents_map = {}
    if agent_ids:
        from core.models import Agent
        for a in db.query(Agent).filter(Agent.id.in_(agent_ids)).all():
            agents_map[a.id] = a.name

    result = []
    for t in tasks:
        result.append({
            "id": t.id,
            "title": t.title,
            "status": t.status,
            "priority": t.priority,
            "assigned_agent": agents_map.get(t.assigned_agent_id, "unassigned"),
            "created_at": str(t.created_at) if t.created_at else None,
            "started_at": str(t.started_at) if t.started_at else None,
            "completed_at": str(t.completed_at) if t.completed_at else None,
            "error_message": t.error_message,
        })

    return {"success": True, "tasks": result, "total": len(result)}


async def get_board_task(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get full details of a single board task."""
    from core.models.core import BoardTask

    task_id = params.get("task_id")
    if not task_id:
        return {"success": False, "error": "task_id is required"}

    task = db.query(BoardTask).filter(
        BoardTask.id == int(task_id),
        BoardTask.workspace_id == workspace_id,
    ).first()

    if not task:
        return {"success": False, "error": f"Task {task_id} not found"}

    # Resolve agent name
    agent_name = None
    if task.assigned_agent_id:
        from core.models import Agent
        agent = db.query(Agent).get(task.assigned_agent_id)
        agent_name = agent.name if agent else None

    return {
        "success": True,
        "task": {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "raw_prompt": task.raw_prompt,
            "status": task.status,
            "priority": task.priority,
            "review_mode": task.review_mode,
            "assigned_agent": agent_name or "unassigned",
            "tags": task.tags or [],
            "result": str(task.result)[:2000] if task.result else None,
            "error_message": task.error_message,
            "created_at": str(task.created_at) if task.created_at else None,
            "started_at": str(task.started_at) if task.started_at else None,
            "completed_at": str(task.completed_at) if task.completed_at else None,
        },
    }


async def assign_board_task(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Assign a board task to an agent by name."""
    from core.models.core import BoardTask
    from core.models import Agent
    from sqlalchemy import func as sa_func

    task_id = params.get("task_id")
    agent_name = params.get("agent_name")
    if not task_id or not agent_name:
        return {"success": False, "error": "task_id and agent_name are required"}

    task = db.query(BoardTask).filter(
        BoardTask.id == int(task_id),
        BoardTask.workspace_id == workspace_id,
    ).first()
    if not task:
        return {"success": False, "error": f"Task {task_id} not found"}

    agent = db.query(Agent).filter(
        Agent.workspace_id == workspace_id,
        sa_func.lower(Agent.name) == agent_name.lower(),
    ).first()
    if not agent:
        return {"success": False, "error": f"Agent '{agent_name}' not found"}

    task.assigned_agent_id = agent.id
    if task.status == "inbox":
        task.status = "assigned"
    db.commit()

    return {
        "success": True,
        "task_id": task.id,
        "assigned_agent": agent.name,
        "status": task.status,
    }


async def update_board_task_status(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Update a board task's status. Moving to in_progress triggers execution."""
    from core.models.core import BoardTask

    task_id = params.get("task_id")
    new_status = params.get("status")
    if not task_id or not new_status:
        return {"success": False, "error": "task_id and status are required"}

    valid = {"inbox", "assigned", "in_progress", "review", "done"}
    if new_status not in valid:
        return {"success": False, "error": f"Invalid status: {new_status}. Must be one of {valid}"}

    task = db.query(BoardTask).filter(
        BoardTask.id == int(task_id),
        BoardTask.workspace_id == workspace_id,
    ).first()
    if not task:
        return {"success": False, "error": f"Task {task_id} not found"}

    task.status = new_status
    if new_status == "in_progress" and not task.started_at:
        task.started_at = datetime.now(timezone.utc)
    if new_status in ("done", "review") and not task.completed_at:
        task.completed_at = datetime.now(timezone.utc)

    db.commit()

    # Trigger agent execution if moved to in_progress with an assigned agent
    if new_status == "in_progress" and task.assigned_agent_id:
        from api.board_tasks import _launch_task_execution
        _launch_task_execution(
            task_id=task.id,
            agent_id=task.assigned_agent_id,
            workspace_id=str(workspace_id),
            prompt=task.raw_prompt or task.description or task.title,
            review_mode=task.review_mode or "auto",
        )

    return {
        "success": True,
        "task_id": task.id,
        "status": task.status,
        "triggered_execution": new_status == "in_progress" and task.assigned_agent_id is not None,
    }
