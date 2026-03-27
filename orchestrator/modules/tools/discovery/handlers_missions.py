"""Mission handlers for PlatformActionExecutor (PRD-82A)."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def create_mission(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create and launch a mission via CoordinatorService."""
    goal = params.get("goal")
    if not goal:
        return {"success": False, "error": "goal is required"}

    config = params.get("config") or {}
    created_by = str(params.get("_agent_id", "agent"))

    try:
        from services.coordinator_service import CoordinatorService

        coordinator = CoordinatorService()
        run = await coordinator.create_mission(
            db=db,
            workspace_id=workspace_id,
            goal=goal,
            created_by=created_by,
            config=config,
        )

        # Summarize the plan for the caller
        plan = run.plan or {}
        tasks = plan.get("tasks", [])
        task_summary = [
            {"title": t.get("title", ""), "agent_role": t.get("agent_role", ""), "sequence": t.get("sequence_number", 0)}
            for t in tasks[:10]
        ]

        return {
            "success": True,
            "mission_id": run.id,
            "state": run.state,
            "goal": run.goal[:200] if run.goal else "",
            "task_count": len(tasks),
            "tasks": task_summary,
            "message": f"Mission {run.id} created with {len(tasks)} tasks. The coordinator will execute them automatically.",
        }

    except Exception as e:
        logger.error("[Missions] create_mission failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to create mission: {str(e)[:300]}"}


async def list_missions(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List missions in the workspace."""
    from core.models.orchestration import OrchestrationRun

    query = db.query(OrchestrationRun).filter(
        OrchestrationRun.workspace_id == workspace_id,
    )

    state = params.get("state")
    if state:
        query = query.filter(OrchestrationRun.state == state)

    limit = min(int(params.get("limit", 10)), 50)
    runs = query.order_by(OrchestrationRun.created_at.desc()).limit(limit).all()

    result = []
    for r in runs:
        plan = r.plan or {}
        result.append({
            "id": r.id,
            "goal": (r.goal or "")[:150],
            "state": r.state,
            "task_count": len(plan.get("tasks", [])),
            "created_at": str(r.created_at) if r.created_at else None,
            "completed_at": str(r.completed_at) if r.completed_at else None,
            "created_by": r.created_by,
        })

    return {"success": True, "missions": result, "total": len(result)}


async def get_mission(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get full details of a specific mission."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    mission_id = params.get("mission_id")
    if not mission_id:
        return {"success": False, "error": "mission_id is required"}

    run = db.query(OrchestrationRun).filter(
        OrchestrationRun.id == int(mission_id),
        OrchestrationRun.workspace_id == workspace_id,
    ).first()

    if not run:
        return {"success": False, "error": f"Mission {mission_id} not found"}

    # Get tasks
    tasks = db.query(OrchestrationTask).filter(
        OrchestrationTask.run_id == run.id,
    ).order_by(OrchestrationTask.sequence_number).all()

    task_details = []
    for t in tasks:
        task_details.append({
            "id": t.id,
            "title": t.title,
            "state": t.state,
            "agent_role": t.agent_role,
            "sequence": t.sequence_number,
            "result_summary": str(t.result)[:500] if t.result else None,
            "error": t.error_message if hasattr(t, "error_message") else None,
        })

    return {
        "success": True,
        "mission": {
            "id": run.id,
            "goal": run.goal,
            "state": run.state,
            "config": run.config,
            "plan": run.plan,
            "created_by": run.created_by,
            "created_at": str(run.created_at) if run.created_at else None,
            "completed_at": str(run.completed_at) if run.completed_at else None,
            "tasks": task_details,
        },
    }
