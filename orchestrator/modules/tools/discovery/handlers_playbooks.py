"""Playbook CRUD + execution handlers for PlatformActionExecutor."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def list_playbooks(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate

    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )

    status_filter = params.get("status_filter", "all")
    if status_filter != "all" and hasattr(WorkflowTemplate, "status"):
        query = query.filter(WorkflowTemplate.status == status_filter)

    playbooks = query.order_by(WorkflowTemplate.id).all()

    return {
        "success": True,
        "playbooks": [
            {
                "id": r.id,
                "name": r.name,
                "template_id": r.template_id,
                "description": (r.description or "")[:200],
                "tags": r.tags or [],
                "created_at": r.created_at.isoformat() if hasattr(r, "created_at") and r.created_at else None,
            }
            for r in playbooks
        ],
        "count": len(playbooks),
    }


async def get_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate

    playbook_id = params.get("playbook_id")
    playbook_name = params.get("playbook_name")

    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if playbook_id:
        query = query.filter(WorkflowTemplate.id == playbook_id)
    elif playbook_name:
        query = query.filter(WorkflowTemplate.name.ilike(f"%{playbook_name}%"))
    else:
        return {"success": False, "error": "Provide playbook_name or playbook_id"}

    playbook = query.first()
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    # Count executions
    exec_count = 0
    try:
        from core.models.core import RecipeExecution
        exec_count = (
            db.query(RecipeExecution)
            .filter(RecipeExecution.recipe_id == playbook.id)
            .count()
        )
    except Exception:
        pass

    steps = playbook.steps or []

    return {
        "success": True,
        "playbook": {
            "id": playbook.id,
            "name": playbook.name,
            "template_id": playbook.template_id,
            "description": playbook.description,
            "tags": playbook.tags or [],
            "step_count": len(steps),
            "steps": [
                {
                    "index": i,
                    "prompt_preview": (s.get("prompt_template", "") or "")[:120],
                    "agent_id": s.get("agent_id"),
                    "error_handling": s.get("error_handling", "stop"),
                    "output_key": s.get("output_key"),
                }
                for i, s in enumerate(steps[:10])
                if isinstance(s, dict)
            ],
            "total_executions": exec_count,
        },
    }


async def create_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    import uuid

    name = params.get("name")
    description = params.get("description")
    if not name or not description:
        return {"success": False, "error": "Missing required: name and description"}

    tags = params.get("tags", [])
    template_id = f"custom-{uuid.uuid4().hex[:8]}"

    playbook = WorkflowTemplate(
        name=name,
        template_id=template_id,
        description=description,
        workspace_id=workspace_id,
        owner_type="workspace",
        owner_id=str(workspace_id),
        created_by="platform",
        tags=tags,
        template_definition={"steps": [], "agents": [], "config": {}, "variables": []},
    )
    db.add(playbook)
    db.flush()

    logger.info(f"[PlatformExecutor] Created playbook '{name}' (id={playbook.id}) in workspace {workspace_id}")

    return {
        "success": True,
        "playbook": {
            "id": playbook.id,
            "name": playbook.name,
            "template_id": playbook.template_id,
            "description": playbook.description,
        },
        "message": f"Playbook '{name}' created successfully. Add steps via the playbook editor.",
    }


async def update_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate

    playbook_id = params.get("playbook_id")
    if not playbook_id:
        return {"success": False, "error": "Missing required parameter: playbook_id"}

    playbook = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == playbook_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    changes = []
    if params.get("name"):
        playbook.name = params["name"]
        changes.append(f"name -> '{params['name']}'")
    if params.get("description") is not None:
        playbook.description = params["description"]
        changes.append("description updated")
    if params.get("tags") is not None:
        playbook.tags = params["tags"]
        changes.append(f"tags -> {params['tags']}")
    if params.get("execution_config") is not None:
        playbook.execution_config = params["execution_config"]
        changes.append("execution_config updated")
    if params.get("schedule_config") is not None:
        playbook.schedule_config = params["schedule_config"]
        changes.append("schedule_config updated")

    if not changes:
        return {"success": True, "message": "No changes specified", "playbook_id": playbook.id}

    db.flush()
    logger.info(f"[PlatformExecutor] Updated playbook {playbook.id}: {', '.join(changes)}")

    return {
        "success": True,
        "playbook_id": playbook.id,
        "changes": changes,
        "message": f"Playbook '{playbook.name}' updated: {', '.join(changes)}",
    }


async def _validate_agent_id(db: Session, workspace_id: UUID, agent_id) -> tuple:
    """Validate agent_id exists in workspace. Returns (valid_id: int | None, error: str | None)."""
    if agent_id is None:
        return None, None
    from core.models import Agent
    try:
        aid = int(agent_id)
    except (ValueError, TypeError):
        return None, f"agent_id must be an integer, got: {agent_id!r}"
    agent = db.query(Agent).filter(Agent.id == aid, Agent.workspace_id == workspace_id).first()
    if not agent:
        valid = db.query(Agent.id, Agent.name).filter(Agent.workspace_id == workspace_id, Agent.status == "active").all()
        agent_list = ", ".join(f"{a.id}={a.name}" for a in valid[:20])
        return None, f"agent_id {aid} does not exist in this workspace. Valid agents: [{agent_list}]"
    return aid, None


async def add_playbook_step(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    from sqlalchemy.orm.attributes import flag_modified
    import uuid

    playbook_id = params.get("playbook_id")
    prompt_template = params.get("prompt_template")
    if not playbook_id or not prompt_template:
        return {"success": False, "error": "Missing required: playbook_id and prompt_template"}

    agent_id, err = await _validate_agent_id(db, workspace_id, params.get("agent_id"))
    if err:
        return {"success": False, "error": err}

    playbook = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == playbook_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    steps = list(playbook.steps or [])
    order = params.get("order", len(steps))

    step = {
        "step_id": uuid.uuid4().hex[:12],
        "step_number": order + 1,
        "prompt_template": prompt_template,
        "agent_id": agent_id,
        "error_handling": params.get("error_handling", "stop"),
        "output_key": params.get("output_key"),
    }

    if order >= len(steps):
        steps.append(step)
    else:
        steps.insert(order, step)

    # Re-number all steps
    for i, s in enumerate(steps):
        s["step_number"] = i + 1

    playbook.steps = steps
    flag_modified(playbook, "steps")
    db.flush()

    logger.info(f"[PlatformExecutor] Added step to playbook {playbook.id} (now {len(steps)} steps)")

    return {
        "success": True,
        "playbook_id": playbook.id,
        "step_index": order if order < len(steps) else len(steps) - 1,
        "total_steps": len(steps),
        "message": f"Step added to playbook '{playbook.name}' (now {len(steps)} steps).",
    }


async def update_playbook_step(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    from sqlalchemy.orm.attributes import flag_modified

    playbook_id = params.get("playbook_id")
    step_index = params.get("step_index")
    if playbook_id is None or step_index is None:
        return {"success": False, "error": "Missing required: playbook_id and step_index"}

    playbook = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == playbook_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    steps = list(playbook.steps or [])
    if step_index < 0 or step_index >= len(steps):
        return {"success": False, "error": f"step_index {step_index} out of range (0-{len(steps)-1})"}

    if "agent_id" in params and params["agent_id"] is not None:
        valid_id, err = await _validate_agent_id(db, workspace_id, params["agent_id"])
        if err:
            return {"success": False, "error": err}
        params["agent_id"] = valid_id

    step = steps[step_index]
    changes = []

    for field in ("prompt_template", "agent_id", "order", "error_handling", "output_key"):
        if field in params and params[field] is not None:
            step[field] = params[field]
            changes.append(f"{field} updated")

    if not changes:
        return {"success": True, "message": "No changes specified", "playbook_id": playbook.id}

    playbook.steps = steps
    flag_modified(playbook, "steps")
    db.flush()

    logger.info(f"[PlatformExecutor] Updated step {step_index} of playbook {playbook.id}: {', '.join(changes)}")

    return {
        "success": True,
        "playbook_id": playbook.id,
        "step_index": step_index,
        "changes": changes,
        "message": f"Step {step_index} of '{playbook.name}' updated: {', '.join(changes)}",
    }


async def delete_playbook_step(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    from sqlalchemy.orm.attributes import flag_modified

    playbook_id = params.get("playbook_id")
    step_index = params.get("step_index")
    if playbook_id is None or step_index is None:
        return {"success": False, "error": "Missing required: playbook_id and step_index"}

    playbook = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == playbook_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    steps = list(playbook.steps or [])
    if step_index < 0 or step_index >= len(steps):
        return {"success": False, "error": f"step_index {step_index} out of range (0-{len(steps)-1})"}

    removed = steps.pop(step_index)

    # Re-number remaining steps
    for i, s in enumerate(steps):
        s["step_number"] = i + 1

    playbook.steps = steps
    flag_modified(playbook, "steps")
    db.flush()

    logger.info(f"[PlatformExecutor] Deleted step {step_index} from playbook {playbook.id} (now {len(steps)} steps)")

    return {
        "success": True,
        "playbook_id": playbook.id,
        "deleted_step_index": step_index,
        "remaining_steps": len(steps),
        "message": f"Step {step_index} removed from '{playbook.name}' ({len(steps)} steps remaining).",
    }


async def schedule_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Set a cron schedule on a playbook so it runs automatically."""
    from core.models.core import WorkflowTemplate

    playbook_id = params.get("playbook_id")
    playbook_name = params.get("playbook_name")
    cron_expression = params.get("cron_expression")

    if not cron_expression:
        return {"success": False, "error": "Missing required parameter: cron_expression"}

    # Validate cron expression
    parts = cron_expression.strip().split()
    if len(parts) != 5:
        return {"success": False, "error": f"Invalid cron expression: expected 5 fields, got {len(parts)}. Format: minute hour day_of_month month day_of_week"}

    # Resolve playbook
    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if playbook_id:
        query = query.filter(WorkflowTemplate.id == playbook_id)
    elif playbook_name:
        query = query.filter(WorkflowTemplate.name.ilike(f"%{playbook_name}%"))
    else:
        return {"success": False, "error": "Provide playbook_id or playbook_name"}

    playbook = query.first()
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    timezone = params.get("timezone", "UTC")
    enabled = params.get("enabled", True)

    schedule_config = {
        "type": "cron",
        "cron_expression": cron_expression,
        "timezone": timezone,
        "enabled": enabled,
    }
    playbook.schedule_config = schedule_config
    db.flush()

    # Sync with APScheduler if available
    try:
        from services.playbook_scheduler import PlaybookSchedulerService
        scheduler = PlaybookSchedulerService()
        if enabled:
            scheduler.schedule_playbook(playbook.id, cron_expression, timezone)
        else:
            scheduler.unschedule_playbook(playbook.id)
    except Exception as e:
        logger.warning("[PlatformExecutor] Scheduler sync failed for playbook %d: %s", playbook.id, e)

    logger.info(
        "[PlatformExecutor] Scheduled playbook '%s' (id=%d) with cron '%s' tz=%s enabled=%s",
        playbook.name, playbook.id, cron_expression, timezone, enabled,
    )

    return {
        "success": True,
        "playbook_id": playbook.id,
        "playbook_name": playbook.name,
        "schedule_config": schedule_config,
        "message": f"Playbook '{playbook.name}' scheduled: {cron_expression} ({timezone}). {'Active now.' if enabled else 'Paused — set enabled=true to activate.'}",
    }


async def execute_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger a playbook run asynchronously. Returns execution_id immediately."""
    from core.models.core import WorkflowTemplate, RecipeExecution
    import uuid

    playbook_id = params.get("playbook_id")
    playbook_name = params.get("playbook_name")
    input_data = params.get("input_data") or {}

    # Resolve playbook
    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if playbook_id:
        query = query.filter(WorkflowTemplate.id == playbook_id)
    elif playbook_name:
        query = query.filter(WorkflowTemplate.name.ilike(f"%{playbook_name}%"))
    else:
        return {"success": False, "error": "Provide playbook_id or playbook_name"}

    playbook = query.first()
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    # Concurrency guard -- return error to agent if workspace is at capacity
    from services.concurrency_guard import check_concurrency
    concurrency = await check_concurrency(workspace_id, db)
    if not concurrency.allowed:
        logger.warning(
            "[PlatformExecutor] Concurrency limit reached for workspace %s: %s",
            workspace_id, concurrency.reason,
        )
        return {"status": "error", "error": concurrency.reason}

    # Create execution record
    execution_id = f"exec-{uuid.uuid4().hex[:12]}"
    execution = RecipeExecution(
        execution_id=execution_id,
        recipe_id=playbook.id,
        workspace_id=workspace_id,
        status="pending",
        input_data=input_data,
        triggered_by="platform_action",
    )
    db.add(execution)
    db.commit()  # Must commit before async task (it opens its own session)

    # Launch async execution fire-and-forget
    try:
        from api.recipe_executor import launch_recipe_task
        launch_recipe_task(
            recipe_execution_id=execution_id,
            recipe_id=playbook.id,
            workspace_id=workspace_id,
            input_data=input_data,
        )
    except Exception as e:
        logger.error("[PlatformExecutor] Failed to launch playbook task: %s", e)
        # Mark execution as failed so it doesn't stay "pending" forever
        execution.status = "failed"
        execution.error_message = f"Failed to enqueue: {str(e)[:500]}"
        db.commit()
        return {"success": False, "error": f"Playbook triggered but failed to launch: {str(e)[:200]}"}

    logger.info(
        "[PlatformExecutor] Triggered playbook '%s' (id=%d) -- execution_id=%s",
        playbook.name, playbook.id, execution_id,
    )

    return {
        "success": True,
        "execution_id": execution_id,
        "playbook_id": playbook.id,
        "playbook_name": playbook.name,
        "status": "pending",
        "message": f"Playbook '{playbook.name}' triggered. Track with execution_id: {execution_id}",
    }


async def get_playbook_execution(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Check status/results of a playbook execution."""
    from core.models.core import RecipeExecution

    execution_id = params.get("execution_id")
    playbook_id = params.get("playbook_id")

    if execution_id:
        execution = (
            db.query(RecipeExecution)
            .filter(
                RecipeExecution.execution_id == execution_id,
                RecipeExecution.workspace_id == workspace_id,
            )
            .first()
        )
        if not execution:
            return {"success": False, "error": f"Execution '{execution_id}' not found"}

        # Summarize step_results (200 char preview per step)
        step_summaries = []
        for i, step in enumerate(execution.step_results or []):
            if isinstance(step, dict):
                output = str(step.get("output", step.get("result", "")))[:200]
                step_summaries.append({
                    "step": i,
                    "status": step.get("status", "unknown"),
                    "output_preview": output,
                })

        return {
            "success": True,
            "execution": {
                "execution_id": execution.execution_id,
                "playbook_id": execution.recipe_id,
                "status": execution.status,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "error_message": execution.error_message,
                "step_results": step_summaries,
                "current_step": execution.current_step,
            },
        }

    elif playbook_id:
        # List recent executions for this playbook
        executions = (
            db.query(RecipeExecution)
            .filter(
                RecipeExecution.recipe_id == playbook_id,
                RecipeExecution.workspace_id == workspace_id,
            )
            .order_by(RecipeExecution.started_at.desc())
            .limit(5)
            .all()
        )

        return {
            "success": True,
            "executions": [
                {
                    "execution_id": e.execution_id,
                    "status": e.status,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                    "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                    "error_message": e.error_message,
                }
                for e in executions
            ],
            "count": len(executions),
        }

    return {"success": False, "error": "Provide execution_id or playbook_id"}


async def delete_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a playbook with full cleanup."""
    from core.models.core import WorkflowTemplate

    playbook_id = params.get("playbook_id")
    playbook_name = params.get("playbook_name")

    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if playbook_id:
        query = query.filter(WorkflowTemplate.id == playbook_id)
    elif playbook_name:
        query = query.filter(WorkflowTemplate.name.ilike(f"%{playbook_name}%"))
    else:
        return {"success": False, "error": "Provide playbook_id or playbook_name"}

    playbook = query.first()
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    # Guard against system playbooks
    if getattr(playbook, "is_system", False):
        return {"success": False, "error": "System playbooks cannot be deleted"}

    playbook_info = {"id": playbook.id, "name": playbook.name}
    cleanup_notes = []

    # Trigger subscription cleanup (non-fatal)
    try:
        from api.workflow_recipes import _cleanup_trigger_subscriptions
        _cleanup_trigger_subscriptions(playbook.id, db)
        cleanup_notes.append("Trigger subscriptions deactivated")
    except Exception as e:
        logger.warning("[PlatformExecutor] Trigger cleanup failed for playbook %d: %s", playbook.id, e)
        cleanup_notes.append(f"Trigger cleanup failed: {e}")

    # Mem0 memory cleanup (non-fatal)
    try:
        import httpx
        from config import config
        mem0_url = config.MEM0_API_URL
        if mem0_url:
            import asyncio
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = {}
                if config.MEM0_API_KEY:
                    headers["Authorization"] = f"Bearer {config.MEM0_API_KEY}"
                await client.delete(
                    f"{mem0_url}/api/v1/memories/",
                    params={"user_id": f"playbook-{playbook.id}"},
                    headers=headers,
                )
            cleanup_notes.append("Playbook memories cleaned up")
    except Exception as e:
        logger.debug("[PlatformExecutor] Mem0 cleanup skipped for playbook %d: %s", playbook.id, e)

    # Delete the playbook (cascades to executions via FK)
    db.delete(playbook)
    db.flush()
    cleanup_notes.append("Database record deleted")

    logger.info("[PlatformExecutor] Deleted playbook %s -- %s", playbook_info, ", ".join(cleanup_notes))

    return {
        "success": True,
        "deleted_playbook": playbook_info,
        "cleanup": cleanup_notes,
        "message": f"Playbook '{playbook_info['name']}' (ID {playbook_info['id']}) deleted.",
    }
