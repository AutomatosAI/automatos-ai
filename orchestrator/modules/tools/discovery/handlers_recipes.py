"""Recipe CRUD + execution handlers for PlatformActionExecutor."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def list_recipes(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate

    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )

    status_filter = params.get("status_filter", "all")
    if status_filter != "all" and hasattr(WorkflowTemplate, "status"):
        query = query.filter(WorkflowTemplate.status == status_filter)

    recipes = query.order_by(WorkflowTemplate.id).all()

    return {
        "success": True,
        "recipes": [
            {
                "id": r.id,
                "name": r.name,
                "template_id": r.template_id,
                "description": (r.description or "")[:200],
                "tags": r.tags or [],
                "created_at": r.created_at.isoformat() if hasattr(r, "created_at") and r.created_at else None,
            }
            for r in recipes
        ],
        "count": len(recipes),
    }


async def get_recipe(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate

    recipe_id = params.get("recipe_id")
    recipe_name = params.get("recipe_name")

    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if recipe_id:
        query = query.filter(WorkflowTemplate.id == recipe_id)
    elif recipe_name:
        query = query.filter(WorkflowTemplate.name.ilike(f"%{recipe_name}%"))
    else:
        return {"success": False, "error": "Provide recipe_name or recipe_id"}

    recipe = query.first()
    if not recipe:
        return {"success": False, "error": "Recipe not found"}

    # Count executions
    exec_count = 0
    try:
        from core.models.core import RecipeExecution
        exec_count = (
            db.query(RecipeExecution)
            .filter(RecipeExecution.recipe_id == recipe.id)
            .count()
        )
    except Exception:
        pass

    steps = recipe.steps or []

    return {
        "success": True,
        "recipe": {
            "id": recipe.id,
            "name": recipe.name,
            "template_id": recipe.template_id,
            "description": recipe.description,
            "tags": recipe.tags or [],
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


async def create_recipe(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    import uuid

    name = params.get("name")
    description = params.get("description")
    if not name or not description:
        return {"success": False, "error": "Missing required: name and description"}

    tags = params.get("tags", [])
    template_id = f"custom-{uuid.uuid4().hex[:8]}"

    recipe = WorkflowTemplate(
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
    db.add(recipe)
    db.flush()

    logger.info(f"[PlatformExecutor] Created recipe '{name}' (id={recipe.id}) in workspace {workspace_id}")

    return {
        "success": True,
        "recipe": {
            "id": recipe.id,
            "name": recipe.name,
            "template_id": recipe.template_id,
            "description": recipe.description,
        },
        "message": f"Recipe '{name}' created successfully. Add steps via the recipe editor.",
    }


async def update_recipe(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate

    recipe_id = params.get("recipe_id")
    if not recipe_id:
        return {"success": False, "error": "Missing required parameter: recipe_id"}

    recipe = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == recipe_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not recipe:
        return {"success": False, "error": "Recipe not found"}

    changes = []
    if params.get("name"):
        recipe.name = params["name"]
        changes.append(f"name -> '{params['name']}'")
    if params.get("description") is not None:
        recipe.description = params["description"]
        changes.append("description updated")
    if params.get("tags") is not None:
        recipe.tags = params["tags"]
        changes.append(f"tags -> {params['tags']}")
    if params.get("execution_config") is not None:
        recipe.execution_config = params["execution_config"]
        changes.append("execution_config updated")
    if params.get("schedule_config") is not None:
        recipe.schedule_config = params["schedule_config"]
        changes.append("schedule_config updated")

    if not changes:
        return {"success": True, "message": "No changes specified", "recipe_id": recipe.id}

    db.flush()
    logger.info(f"[PlatformExecutor] Updated recipe {recipe.id}: {', '.join(changes)}")

    return {
        "success": True,
        "recipe_id": recipe.id,
        "changes": changes,
        "message": f"Recipe '{recipe.name}' updated: {', '.join(changes)}",
    }


async def add_recipe_step(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    from sqlalchemy.orm.attributes import flag_modified
    import uuid

    recipe_id = params.get("recipe_id")
    prompt_template = params.get("prompt_template")
    if not recipe_id or not prompt_template:
        return {"success": False, "error": "Missing required: recipe_id and prompt_template"}

    recipe = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == recipe_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not recipe:
        return {"success": False, "error": "Recipe not found"}

    steps = list(recipe.steps or [])
    order = params.get("order", len(steps))

    step = {
        "step_id": uuid.uuid4().hex[:12],
        "step_number": order + 1,
        "prompt_template": prompt_template,
        "agent_id": params.get("agent_id"),
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

    recipe.steps = steps
    flag_modified(recipe, "steps")
    db.flush()

    logger.info(f"[PlatformExecutor] Added step to recipe {recipe.id} (now {len(steps)} steps)")

    return {
        "success": True,
        "recipe_id": recipe.id,
        "step_index": order if order < len(steps) else len(steps) - 1,
        "total_steps": len(steps),
        "message": f"Step added to recipe '{recipe.name}' (now {len(steps)} steps).",
    }


async def update_recipe_step(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    from sqlalchemy.orm.attributes import flag_modified

    recipe_id = params.get("recipe_id")
    step_index = params.get("step_index")
    if recipe_id is None or step_index is None:
        return {"success": False, "error": "Missing required: recipe_id and step_index"}

    recipe = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == recipe_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not recipe:
        return {"success": False, "error": "Recipe not found"}

    steps = list(recipe.steps or [])
    if step_index < 0 or step_index >= len(steps):
        return {"success": False, "error": f"step_index {step_index} out of range (0-{len(steps)-1})"}

    step = steps[step_index]
    changes = []

    for field in ("prompt_template", "agent_id", "order", "error_handling", "output_key"):
        if field in params and params[field] is not None:
            step[field] = params[field]
            changes.append(f"{field} updated")

    if not changes:
        return {"success": True, "message": "No changes specified", "recipe_id": recipe.id}

    recipe.steps = steps
    flag_modified(recipe, "steps")
    db.flush()

    logger.info(f"[PlatformExecutor] Updated step {step_index} of recipe {recipe.id}: {', '.join(changes)}")

    return {
        "success": True,
        "recipe_id": recipe.id,
        "step_index": step_index,
        "changes": changes,
        "message": f"Step {step_index} of '{recipe.name}' updated: {', '.join(changes)}",
    }


async def delete_recipe_step(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    from sqlalchemy.orm.attributes import flag_modified

    recipe_id = params.get("recipe_id")
    step_index = params.get("step_index")
    if recipe_id is None or step_index is None:
        return {"success": False, "error": "Missing required: recipe_id and step_index"}

    recipe = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == recipe_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not recipe:
        return {"success": False, "error": "Recipe not found"}

    steps = list(recipe.steps or [])
    if step_index < 0 or step_index >= len(steps):
        return {"success": False, "error": f"step_index {step_index} out of range (0-{len(steps)-1})"}

    removed = steps.pop(step_index)

    # Re-number remaining steps
    for i, s in enumerate(steps):
        s["step_number"] = i + 1

    recipe.steps = steps
    flag_modified(recipe, "steps")
    db.flush()

    logger.info(f"[PlatformExecutor] Deleted step {step_index} from recipe {recipe.id} (now {len(steps)} steps)")

    return {
        "success": True,
        "recipe_id": recipe.id,
        "deleted_step_index": step_index,
        "remaining_steps": len(steps),
        "message": f"Step {step_index} removed from '{recipe.name}' ({len(steps)} steps remaining).",
    }


async def execute_recipe(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger a recipe run asynchronously. Returns execution_id immediately."""
    from core.models.core import WorkflowTemplate, RecipeExecution
    import uuid

    recipe_id = params.get("recipe_id")
    recipe_name = params.get("recipe_name")
    input_data = params.get("input_data") or {}

    # Resolve recipe
    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if recipe_id:
        query = query.filter(WorkflowTemplate.id == recipe_id)
    elif recipe_name:
        query = query.filter(WorkflowTemplate.name.ilike(f"%{recipe_name}%"))
    else:
        return {"success": False, "error": "Provide recipe_id or recipe_name"}

    recipe = query.first()
    if not recipe:
        return {"success": False, "error": "Recipe not found"}

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
        recipe_id=recipe.id,
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
            recipe_id=recipe.id,
            workspace_id=workspace_id,
            input_data=input_data,
        )
    except Exception as e:
        logger.error("[PlatformExecutor] Failed to launch recipe task: %s", e)
        # Mark execution as failed so it doesn't stay "pending" forever
        execution.status = "failed"
        execution.error_message = f"Failed to enqueue: {str(e)[:500]}"
        db.commit()
        return {"success": False, "error": f"Recipe triggered but failed to launch: {str(e)[:200]}"}

    logger.info(
        "[PlatformExecutor] Triggered recipe '%s' (id=%d) -- execution_id=%s",
        recipe.name, recipe.id, execution_id,
    )

    return {
        "success": True,
        "execution_id": execution_id,
        "recipe_id": recipe.id,
        "recipe_name": recipe.name,
        "status": "pending",
        "message": f"Recipe '{recipe.name}' triggered. Track with execution_id: {execution_id}",
    }


async def get_recipe_execution(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Check status/results of a recipe execution."""
    from core.models.core import RecipeExecution

    execution_id = params.get("execution_id")
    recipe_id = params.get("recipe_id")

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
                "recipe_id": execution.recipe_id,
                "status": execution.status,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "error_message": execution.error_message,
                "step_results": step_summaries,
                "current_step": execution.current_step,
            },
        }

    elif recipe_id:
        # List recent executions for this recipe
        executions = (
            db.query(RecipeExecution)
            .filter(
                RecipeExecution.recipe_id == recipe_id,
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

    return {"success": False, "error": "Provide execution_id or recipe_id"}


async def delete_recipe(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a recipe with full cleanup."""
    from core.models.core import WorkflowTemplate

    recipe_id = params.get("recipe_id")
    recipe_name = params.get("recipe_name")

    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if recipe_id:
        query = query.filter(WorkflowTemplate.id == recipe_id)
    elif recipe_name:
        query = query.filter(WorkflowTemplate.name.ilike(f"%{recipe_name}%"))
    else:
        return {"success": False, "error": "Provide recipe_id or recipe_name"}

    recipe = query.first()
    if not recipe:
        return {"success": False, "error": "Recipe not found"}

    # Guard against system recipes
    if getattr(recipe, "is_system", False):
        return {"success": False, "error": "System recipes cannot be deleted"}

    recipe_info = {"id": recipe.id, "name": recipe.name}
    cleanup_notes = []

    # Trigger subscription cleanup (non-fatal)
    try:
        from api.workflow_recipes import _cleanup_trigger_subscriptions
        _cleanup_trigger_subscriptions(recipe.id, db)
        cleanup_notes.append("Trigger subscriptions deactivated")
    except Exception as e:
        logger.warning("[PlatformExecutor] Trigger cleanup failed for recipe %d: %s", recipe.id, e)
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
                    params={"user_id": f"recipe-{recipe.id}"},
                    headers=headers,
                )
            cleanup_notes.append("Recipe memories cleaned up")
    except Exception as e:
        logger.debug("[PlatformExecutor] Mem0 cleanup skipped for recipe %d: %s", recipe.id, e)

    # Delete the recipe (cascades to executions via FK)
    db.delete(recipe)
    db.flush()
    cleanup_notes.append("Database record deleted")

    logger.info("[PlatformExecutor] Deleted recipe %s -- %s", recipe_info, ", ".join(cleanup_notes))

    return {
        "success": True,
        "deleted_recipe": recipe_info,
        "cleanup": cleanup_notes,
        "message": f"Recipe '{recipe_info['name']}' (ID {recipe_info['id']}) deleted.",
    }
