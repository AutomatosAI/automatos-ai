"""
Workflow Recipes API
====================

CRUD operations for workflow recipes that users can browse,
customize, and use to create workflows.
"""

import hashlib
import hmac
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func as sa_func
from core.database.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflow-recipes", tags=["workflow-recipes"])

# Import the model from main models file
from core.models import WorkflowTemplate as WorkflowRecipe  # Aliased for transition
from core.models import Agent
from core.models.core import RecipeExecution
from core.models.composio import TriggerSubscription, ComposioEntity
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from config import config


def _sync_cron_schedule(recipe: WorkflowRecipe):
    """Sync a recipe's cron schedule with the RecipeSchedulerService."""
    if not config.RECIPE_SCHEDULER_ENABLED:
        return
    try:
        from services.recipe_scheduler import get_recipe_scheduler
        scheduler = get_recipe_scheduler()
        sc = recipe.schedule_config or {}
        if sc.get("type") == "cron" and sc.get("cron_expression"):
            scheduler.schedule_recipe(recipe)
        else:
            scheduler.unschedule_recipe(recipe.id)
    except Exception as e:
        logger.warning(f"[_sync_cron_schedule] Failed for recipe {recipe.id}: {e}")


def _auto_register_trigger(recipe: WorkflowRecipe, workspace_id, db: Session) -> Optional[str]:
    """
    If recipe.schedule_config is type=trigger with a Composio trigger,
    subscribe via Composio API and store TriggerSubscription.
    Returns the composio_subscription_id on success, None otherwise.

    Non-Composio triggers (custom webhooks) skip Composio registration entirely —
    they only need the webhook_id stored in schedule_config.
    """
    schedule = recipe.schedule_config
    if not schedule or schedule.get("type") != "trigger":
        return None

    trigger_config = schedule.get("trigger_config", {})

    # Only attempt Composio registration for composio-sourced triggers
    source = trigger_config.get("source", "")
    if source != "composio":
        logger.info("[trigger_auto] Non-Composio trigger (source=%s), skipping Composio registration for recipe %d", source, recipe.id)
        return None

    # Support both "trigger_name" (canonical) and "trigger" (UI shorthand)
    trigger_name = (
        trigger_config.get("trigger_name")
        or trigger_config.get("trigger")
    )
    if not trigger_name:
        logger.warning("[trigger_auto] No trigger_name in trigger_config: %s", trigger_config)
        return None

    # Check if a subscription already exists for this recipe
    existing = db.query(TriggerSubscription).filter(
        TriggerSubscription.workflow_id == recipe.id,
        TriggerSubscription.trigger_name == trigger_name,
        TriggerSubscription.is_active == True,
    ).first()
    if existing:
        logger.info("[trigger_auto] Subscription already exists for recipe %d trigger %s", recipe.id, trigger_name)
        return existing.composio_subscription_id

    try:
        from core.composio.client import get_composio_client
        from core.composio.entity_manager import EntityManager

        client = get_composio_client()
        entity_manager = EntityManager(db)
        entity = entity_manager.get_or_create_entity(workspace_id)

        backend_url = config.BACKEND_URL or "http://localhost:8000"
        callback_url = f"{backend_url}/api/composio/webhook"

        result = client.subscribe_to_trigger(
            entity_id=entity["composio_entity_id"],
            trigger_name=trigger_name,
            callback_url=callback_url,
        )

        subscription = TriggerSubscription(
            entity_id=entity["id"],
            trigger_name=trigger_name,
            callback_url=callback_url,
            agent_id=None,
            workflow_id=recipe.id,
            composio_subscription_id=result.get("id"),
            is_active=True,
        )
        db.add(subscription)

        logger.info(
            "[trigger_auto] Registered trigger %s for recipe %d (subscription=%s)",
            trigger_name, recipe.id, result.get("id"),
        )
        return result.get("id")

    except Exception:
        logger.exception("[trigger_auto] Failed to auto-register trigger %s for recipe %d", trigger_name, recipe.id)
        return None


def _cleanup_trigger_subscriptions(recipe_id: int, db: Session) -> None:
    """Deactivate trigger subscriptions for a recipe."""
    subs = db.query(TriggerSubscription).filter(
        TriggerSubscription.workflow_id == recipe_id,
        TriggerSubscription.is_active == True,
    ).all()
    for sub in subs:
        sub.is_active = False
        logger.info("[trigger_auto] Deactivated subscription %d for recipe %d", sub.id, recipe_id)


def _enrich_steps_with_agents(steps: Optional[list], db: Session) -> Optional[list]:
    """Populate agent details for each step in the steps array."""
    if not steps:
        return steps

    # Collect unique agent_ids
    agent_ids = list({step.get('agent_id') for step in steps if step.get('agent_id')})
    if not agent_ids:
        return steps

    # Batch-fetch agents
    agents = db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
    agent_map = {agent.id: agent for agent in agents}

    enriched = []
    for step in steps:
        step_copy = dict(step)
        agent_id = step_copy.get('agent_id')
        if agent_id and agent_id in agent_map:
            agent = agent_map[agent_id]
            model_cfg = agent.model_config or {}
            tool_count = len(agent.skills) if agent.skills else 0
            step_copy['agent'] = {
                'id': agent.id,
                'name': agent.name,
                'model': model_cfg.get('model_id', 'unknown'),
                'provider': model_cfg.get('provider', 'unknown'),
                'tool_count': tool_count,
                'status': agent.status,
            }
        else:
            step_copy['agent'] = None
        enriched.append(step_copy)

    return enriched


@router.get("")
async def list_workflow_recipes(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    is_featured: Optional[bool] = None,
    is_public: Optional[bool] = True,
    search: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    sort_by: str = Query('popularity', regex='^(popularity|created_at|use_count|average_rating|name)$'),
    db: Session = Depends(get_db)
):
    """
    List workflow recipes with filtering and pagination.

    Query Parameters:
    - is_featured: Show only featured recipes
    - is_public: Show only public recipes (default: true)
    - search: Search in name and description
    - skip: Number of records to skip for pagination
    - limit: Maximum number of records to return (1-100)
    - sort_by: Sort field (popularity, created_at, use_count, average_rating, name)
    """
    try:
        query = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id
        )

        # Apply filters

        if is_featured is not None:
            query = query.filter(WorkflowRecipe.is_featured == is_featured)

        if is_public is not None:
            query = query.filter(WorkflowRecipe.is_public == is_public)

        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    WorkflowRecipe.name.ilike(search_pattern),
                    WorkflowRecipe.description.ilike(search_pattern)
                )
            )

        # Get total count before pagination
        total = query.count()

        # Apply sorting
        if sort_by == 'popularity':
            query = query.order_by(WorkflowRecipe.popularity.desc())
        elif sort_by == 'created_at':
            query = query.order_by(WorkflowRecipe.created_at.desc())
        elif sort_by == 'use_count':
            query = query.order_by(WorkflowRecipe.use_count.desc())
        elif sort_by == 'average_rating':
            query = query.order_by(WorkflowRecipe.average_rating.desc())
        elif sort_by == 'name':
            query = query.order_by(WorkflowRecipe.name.asc())

        # Apply pagination
        recipes = query.offset(skip).limit(limit).all()

        return {
            "items": [recipe.to_dict() for recipe in recipes],
            "total": total,
            "skip": skip,
            "limit": limit
        }

    except Exception as e:
        logger.error(f"Error listing workflow recipes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats/dashboard")
async def get_recipe_stats_dashboard(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Aggregate dashboard stats across all workspace recipes and their executions.
    Returns overview metrics, execution status breakdown, and top recipes by usage.
    """
    try:
        # Base query: workspace-scoped recipes
        base_q = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
        )

        total_recipes = base_q.count()
        recipes_with_steps = base_q.filter(WorkflowRecipe.steps != None).count()

        # Aggregate quality_score and success_rate across recipes
        agg = db.query(
            sa_func.avg(WorkflowRecipe.quality_score),
            sa_func.avg(WorkflowRecipe.success_rate),
        ).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
        ).first()
        avg_quality = round(float(agg[0] or 0), 2)
        avg_success = round(float(agg[1] or 0), 1)

        # Recipe IDs for execution queries
        recipe_ids = [r.id for r in base_q.with_entities(WorkflowRecipe.id).all()]

        # Execution stats
        total_executions = 0
        status_breakdown = {"completed": 0, "failed": 0, "running": 0, "pending": 0}
        if recipe_ids:
            exec_q = db.query(RecipeExecution).filter(
                RecipeExecution.recipe_id.in_(recipe_ids),
            )
            total_executions = exec_q.count()

            status_rows = db.query(
                RecipeExecution.status,
                sa_func.count(RecipeExecution.id),
            ).filter(
                RecipeExecution.recipe_id.in_(recipe_ids),
            ).group_by(RecipeExecution.status).all()

            for status_val, cnt in status_rows:
                if status_val in status_breakdown:
                    status_breakdown[status_val] = cnt

        # Top recipes by use_count
        top_recipes_orm = base_q.order_by(
            WorkflowRecipe.use_count.desc()
        ).limit(10).all()

        top_recipes = []
        for r in top_recipes_orm:
            top_recipes.append({
                "id": r.id,
                "template_id": r.template_id,
                "name": r.name,
                "use_count": r.use_count or 0,
                "success_rate": r.success_rate or 0.0,
                "quality_score": r.quality_score,
                "steps_count": len(r.steps) if r.steps else 0,
                "last_used_at": r.last_used_at.isoformat() if r.last_used_at else None,
            })

        return {
            "overview": {
                "total_recipes": total_recipes,
                "total_executions": total_executions,
                "avg_quality_score": avg_quality,
                "avg_success_rate": avg_success,
            },
            "status_breakdown": status_breakdown,
            "top_recipes": top_recipes,
        }

    except Exception as e:
        logger.error(f"Error getting recipe dashboard stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{recipe_id}")
async def get_workflow_recipe(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get a single workflow recipe by its template_id.
    Returns recipe data with agent details populated for each step in the steps array."""
    try:
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        result = recipe.to_dict()
        # Enrich steps with agent details
        result['steps'] = _enrich_steps_with_agents(recipe.steps, db)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recipe {recipe_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("")
async def create_workflow_recipe(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    recipe_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Create a new workflow recipe.

    Required fields:
    - template_id: Unique identifier (e.g., "my-custom-recipe")
    - name: Display name
    - description: Description of what the recipe does
    - template_definition: JSON structure with steps, agents, config
    - steps: Array of step definitions (required, each step needs step_id, order, agent_id, prompt_template)

    Optional fields:
    - tags: Array of tags
    - inputs: Input schema
    - outputs: Output schema
    - execution_config: Runtime behavior config (defaults provided if omitted)
    - schedule_config: Scheduling configuration
    - recommended_agents: Array of agent type names
    - required_tools: Array of tool names
    - is_public: Boolean (default: true)
    - is_featured: Boolean (default: false)
    """
    try:
        # Validate required fields
        required_fields = ['template_id', 'name', 'description', 'template_definition', 'steps']

        for field in required_fields:
            if field not in recipe_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        # Check if template_id already exists
        existing = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_data['template_id']
        ).first()

        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Recipe with ID '{recipe_data['template_id']}' already exists"
            )

        # Apply execution_config defaults if not provided
        # Timeouts in seconds (executor normalises legacy ms values automatically)
        execution_config = recipe_data.get('execution_config') or {
            'mode': 'sequential',
            'max_retries': 1,
            'timeout_per_step': 300,
            'total_timeout': 600,
            'auto_learning': True,
        }

        # Ensure webhook_id in schedule_config for trigger/webhook types
        schedule_config = recipe_data.get('schedule_config')
        if schedule_config and schedule_config.get('type') in ('trigger', 'webhook'):
            if 'webhook_id' not in schedule_config:
                schedule_config['webhook_id'] = uuid4().hex

        # Create recipe (validation happens after assignment)
        recipe = WorkflowRecipe(
            workspace_id=ctx.workspace_id,
            template_id=recipe_data['template_id'],
            name=recipe_data['name'],
            description=recipe_data['description'],
            template_definition=recipe_data['template_definition'],
            tags=recipe_data.get('tags', []),
            steps=recipe_data['steps'],
            inputs=recipe_data.get('inputs'),
            outputs=recipe_data.get('outputs'),
            execution_config=execution_config,
            schedule_config=schedule_config,
            recommended_agents=recipe_data.get('recommended_agents', []),
            required_tools=recipe_data.get('required_tools', []),
            is_public=recipe_data.get('is_public', True),
            is_featured=recipe_data.get('is_featured', False),
            is_system=False,  # User-created recipes are never system recipes
            preview_image=recipe_data.get('preview_image'),
            documentation_url=recipe_data.get('documentation_url'),
            version=recipe_data.get('version', '1.0'),
            created_by=recipe_data.get('created_by', ctx.user.email if ctx.user and ctx.user.email else "anonymous")
        )

        # Validate steps structure
        is_valid, error = recipe.validate_steps()
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid steps: {error}")

        # Validate execution_config structure
        is_valid, error = recipe.validate_execution_config()
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid execution_config: {error}")

        # Validate schedule_config structure
        is_valid, error = recipe.validate_schedule_config()
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid schedule_config: {error}")

        # Validate agent_id references exist in workspace
        agent_ids = [step.get('agent_id') for step in recipe.steps if step.get('agent_id')]
        if agent_ids:
            existing_agents = db.query(Agent.id).filter(
                Agent.id.in_(agent_ids),
                Agent.workspace_id == ctx.workspace_id
            ).all()
            existing_ids = {a.id for a in existing_agents}
            missing = [aid for aid in agent_ids if aid not in existing_ids]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent IDs not found in workspace: {missing}"
                )

        db.add(recipe)
        db.commit()
        db.refresh(recipe)

        # Auto-register Composio trigger if schedule_config is trigger type
        trigger_sub_id = _auto_register_trigger(recipe, ctx.workspace_id, db)
        if trigger_sub_id:
            db.commit()

        # Sync cron scheduler
        _sync_cron_schedule(recipe)

        logger.info(f"Created workflow recipe: {recipe.template_id}")

        return {
            "message": "Recipe created successfully",
            "recipe": recipe.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow recipe: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{recipe_id}")
async def update_workflow_recipe(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    recipe_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Update an existing workflow recipe.
    System recipes cannot be modified.
    """
    try:
        logger.info(f"[update_recipe] PUT {recipe_id} - fields: {list(recipe_data.keys())}")

        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        if recipe.is_system:
            raise HTTPException(
                status_code=403,
                detail="System recipes cannot be modified"
            )

        # Ensure webhook_id in schedule_config for trigger/webhook types
        if 'schedule_config' in recipe_data:
            sc = recipe_data['schedule_config']
            if sc and sc.get('type') in ('trigger', 'webhook') and 'webhook_id' not in sc:
                # Preserve existing webhook_id on update — generating a new one
                # breaks any external automation (JIRA, GitHub, etc.) already
                # configured to POST to the old URL.
                existing_wh = (recipe.schedule_config or {}).get('webhook_id')
                sc['webhook_id'] = existing_wh or uuid4().hex

        # Update fields if provided
        updatable_fields = [
            'name', 'description', 'tags',
            'template_definition', 'steps', 'inputs', 'outputs',
            'execution_config', 'schedule_config',
            'recommended_agents', 'required_tools',
            'is_public', 'is_featured',
            'preview_image', 'documentation_url', 'version', 'changelog'
        ]

        for field in updatable_fields:
            if field in recipe_data:
                setattr(recipe, field, recipe_data[field])

        # Validate steps if updated
        if 'steps' in recipe_data:
            is_valid, error = recipe.validate_steps()
            if not is_valid:
                logger.warning(f"[update_recipe] Steps validation failed for {recipe_id}: {error}")
                raise HTTPException(status_code=400, detail=f"Invalid steps: {error}")

            # Validate agent_id references exist in workspace
            agent_ids = [step.get('agent_id') for step in (recipe.steps or []) if step.get('agent_id')]
            if agent_ids:
                existing_agents = db.query(Agent.id).filter(
                    Agent.id.in_(agent_ids),
                    Agent.workspace_id == ctx.workspace_id
                ).all()
                existing_ids = {a.id for a in existing_agents}
                missing = [aid for aid in agent_ids if aid not in existing_ids]
                if missing:
                    logger.warning(f"[update_recipe] Agent IDs not found for {recipe_id}: {missing}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Agent IDs not found in workspace: {missing}"
                    )

        # Validate execution_config if updated
        if 'execution_config' in recipe_data:
            is_valid, error = recipe.validate_execution_config()
            if not is_valid:
                logger.warning(f"[update_recipe] execution_config validation failed for {recipe_id}: {error} | data: {recipe_data.get('execution_config')}")
                raise HTTPException(status_code=400, detail=f"Invalid execution_config: {error}")

        # Validate schedule_config if updated
        if 'schedule_config' in recipe_data:
            is_valid, error = recipe.validate_schedule_config()
            if not is_valid:
                logger.warning(f"[update_recipe] schedule_config validation failed for {recipe_id}: {error}")
                raise HTTPException(status_code=400, detail=f"Invalid schedule_config: {error}")

        recipe.updated_at = datetime.now()
        db.commit()
        db.refresh(recipe)

        # Re-register trigger if schedule_config changed
        # Only deactivate old subscriptions if new registration succeeds
        if 'schedule_config' in recipe_data:
            new_sub_id = _auto_register_trigger(recipe, ctx.workspace_id, db)
            if new_sub_id:
                # New subscription created — deactivate old ones (except the new one)
                _cleanup_trigger_subscriptions(recipe.id, db)
                # Re-activate the newly created one (cleanup may have caught it)
                new_sub = db.query(TriggerSubscription).filter(
                    TriggerSubscription.composio_subscription_id == new_sub_id,
                    TriggerSubscription.workflow_id == recipe.id,
                ).first()
                if new_sub:
                    new_sub.is_active = True
            db.commit()

        # Sync cron scheduler
        _sync_cron_schedule(recipe)

        logger.info(f"Updated workflow recipe: {recipe_id}")

        return {
            "message": "Recipe updated successfully",
            "recipe": recipe.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating recipe {recipe_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{recipe_id}")
async def delete_workflow_recipe(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Delete a workflow recipe.
    System recipes cannot be deleted.
    """
    try:
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        if recipe.is_system:
            raise HTTPException(
                status_code=403,
                detail="System recipes cannot be deleted"
            )

        # Unschedule cron job if any
        if config.RECIPE_SCHEDULER_ENABLED:
            try:
                from services.recipe_scheduler import get_recipe_scheduler
                get_recipe_scheduler().unschedule_recipe(recipe.id)
            except Exception:
                pass

        # Cleanup trigger subscriptions before deleting
        _cleanup_trigger_subscriptions(recipe.id, db)

        # Cleanup Mem0 memories scoped to this recipe
        try:
            from core.services.recipe_memory_service import RecipeMemoryService
            from modules.memory.integrations.mem0_client import Mem0Client
            mem0 = Mem0Client()
            template_id = recipe.template_id or str(recipe.id)
            recipe_scope = f"ws_{ctx.workspace_id}_recipe_{template_id}"
            # Delete all memories under the recipe scope
            all_mems = mem0.get_all(user_id=recipe_scope, limit=200)
            deleted_count = 0
            for mem in all_mems:
                mem_id = mem.get("id") if isinstance(mem, dict) else None
                if mem_id:
                    mem0.delete(mem_id)
                    deleted_count += 1
            if deleted_count:
                logger.info(f"[delete_recipe] Cleaned up {deleted_count} Mem0 memories for scope {recipe_scope}")
        except Exception as e:
            logger.info(f"[delete_recipe] Mem0 cleanup skipped: {e}")

        db.delete(recipe)
        db.commit()

        logger.info(f"Deleted workflow recipe: {recipe_id}")

        return {
            "message": "Recipe deleted successfully",
            "recipe_id": recipe_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting recipe {recipe_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{recipe_id}/use")
async def record_recipe_usage(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Record that a recipe was used to create a workflow.
    Updates use_count and last_used_at.
    """
    try:
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        recipe.use_count += 1
        recipe.last_used_at = datetime.now()
        db.commit()

        return {
            "message": "Recipe usage recorded",
            "recipe_id": recipe_id,
            "use_count": recipe.use_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording recipe usage for {recipe_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/categories/list")
async def list_recipe_categories(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get list of all unique tags used across workspace recipes"""
    try:
        recipes = db.query(WorkflowRecipe.tags).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.is_public == True
        ).all()

        # Aggregate tags across all recipes
        tag_counts: dict = {}
        for (tags,) in recipes:
            if tags:
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "categories": [
                {"name": name, "count": count}
                for name, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            ]
        }

    except Exception as e:
        logger.error(f"Error listing recipe categories: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/featured/list")
async def list_featured_recipes(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get featured workflow recipes"""
    try:
        recipes = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.is_featured == True,
            WorkflowRecipe.is_public == True
        ).order_by(
            WorkflowRecipe.popularity.desc()
        ).limit(limit).all()

        return {
            "items": [recipe.to_dict() for recipe in recipes],
            "total": len(recipes)
        }

    except Exception as e:
        logger.error(f"Error listing featured recipes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{recipe_id}/execute")
async def execute_recipe(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    body: Dict[str, Any] = Body(default={}),
    db: Session = Depends(get_db)
):
    """
    Execute a recipe directly — step by step, no 9-stage pipeline.

    Creates a RecipeExecution record and launches execute_recipe_direct()
    as an async task. Each step calls its assigned agent with filtered
    Composio actions. Results stored in RecipeExecution.step_results.

    Body (optional):
    - input_data: Dict matching the recipe's inputs schema
    """
    try:
        logger.info(f"[execute_recipe] Starting direct execution for recipe_id={recipe_id}, workspace={ctx.workspace_id}")

        # Fetch recipe and validate ownership
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            logger.warning(f"[execute_recipe] Recipe not found: {recipe_id}")
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        if not recipe.steps:
            raise HTTPException(status_code=400, detail="Recipe has no steps to execute")

        logger.info(f"[execute_recipe] Recipe found: {recipe.name}, steps={len(recipe.steps or [])}")

        input_data = body.get('input_data') or {}

        # Fill in defaults for missing required inputs
        if recipe.inputs:
            for param_name, param_def in recipe.inputs.items():
                if isinstance(param_def, dict) and param_name not in input_data:
                    default = param_def.get('default', '')
                    input_data[param_name] = default

        # Create RecipeExecution record
        recipe_execution_id = f"exec-{uuid4().hex[:12]}"
        recipe_execution = RecipeExecution(
            execution_id=recipe_execution_id,
            recipe_id=recipe.id,
            workspace_id=ctx.workspace_id,
            status='pending',
            input_data=input_data,
            current_step=0,
            triggered_by=ctx.user.email if ctx.user else 'anonymous',
            execution_metadata={
                'execution_type': 'recipe_direct',
                'total_steps': len(recipe.steps),
            },
        )
        db.add(recipe_execution)

        # Update recipe usage stats
        recipe.use_count += 1
        recipe.last_used_at = datetime.now()

        db.commit()

        logger.info(f"[execute_recipe] Created execution {recipe_execution_id}, launching direct executor")

        # Launch direct executor as async task (crash-safe)
        from api.recipe_executor import launch_recipe_task
        launch_recipe_task(
            recipe_execution_id=recipe_execution_id,
            recipe_id=recipe.id,
            workspace_id=ctx.workspace_id,
            input_data=input_data,
        )

        return {
            "recipe_execution_id": recipe_execution_id,
            "recipe_id": recipe_id,
            "status": "started",
            "total_steps": len(recipe.steps),
            "message": "Recipe execution started (direct mode)",
        }

    except HTTPException as he:
        logger.warning(f"[execute_recipe] HTTPException: status={he.status_code}, detail={he.detail}")
        raise
    except Exception as e:
        logger.error(f"[execute_recipe] Unhandled error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{recipe_id}/executions/{execution_id}")
async def get_recipe_execution_detail(
    recipe_id: str,
    execution_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get detailed status of a specific recipe execution.

    Returns step-level results, current progress, and overall status.
    Used by frontend for polling execution progress.
    """
    try:
        # Validate recipe ownership — try template_id first, fall back to integer id
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe and recipe_id.isdigit():
            recipe = db.query(WorkflowRecipe).filter(
                WorkflowRecipe.owner_type == 'workspace',
                WorkflowRecipe.workspace_id == ctx.workspace_id,
                WorkflowRecipe.id == int(recipe_id)
            ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        # Get execution — try execution_id string first, fall back to numeric DB id
        execution = db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id,
            RecipeExecution.recipe_id == recipe.id
        ).first()

        if not execution and execution_id.isdigit():
            execution = db.query(RecipeExecution).filter(
                RecipeExecution.id == int(execution_id),
                RecipeExecution.recipe_id == recipe.id
            ).first()

        if not execution:
            raise HTTPException(
                status_code=404,
                detail=f"Execution '{execution_id}' not found for recipe '{recipe_id}'"
            )

        total_steps = len(recipe.steps or [])
        step_results = execution.step_results or []

        return {
            "execution_id": execution.execution_id,
            "recipe_id": recipe_id,
            "recipe_name": recipe.name,
            "status": execution.status,
            "current_step": execution.current_step or 0,
            "total_steps": total_steps,
            "step_results": step_results,
            "input_data": execution.input_data,
            "output_data": execution.output_data,
            "error_message": execution.error_message,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "triggered_by": execution.triggered_by,
            "execution_metadata": execution.execution_metadata,
            "total_duration_ms": (
                execution.output_data.get("total_duration_ms")
                if execution.output_data else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution detail: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================================================
# STEP LOG ENDPOINT (Lazy-load full logs from S3)
# ===================================================================

@router.get("/{recipe_id}/executions/{execution_id}/steps/{step_order}/logs")
async def get_step_full_logs(
    recipe_id: str,
    execution_id: str,
    step_order: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Fetch full verbose step log from S3 on demand.

    The DB stores only compact summaries. This endpoint fetches the
    full agent output, tool call results, and message history from S3.
    """
    try:
        # Validate recipe ownership — try template_id first, fall back to integer id
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe and recipe_id.isdigit():
            recipe = db.query(WorkflowRecipe).filter(
                WorkflowRecipe.owner_type == 'workspace',
                WorkflowRecipe.workspace_id == ctx.workspace_id,
                WorkflowRecipe.id == int(recipe_id)
            ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        # Validate execution — try execution_id string first, fall back to numeric DB id
        execution = db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id,
            RecipeExecution.recipe_id == recipe.id
        ).first()

        if not execution and execution_id.isdigit():
            execution = db.query(RecipeExecution).filter(
                RecipeExecution.id == int(execution_id),
                RecipeExecution.recipe_id == recipe.id
            ).first()

        if not execution:
            raise HTTPException(
                status_code=404,
                detail=f"Execution '{execution_id}' not found for recipe '{recipe_id}'"
            )

        # Check if step result has a log_url
        step_results = execution.step_results or []
        log_url = None
        for sr in step_results:
            if isinstance(sr, dict) and sr.get("order") == step_order:
                log_url = sr.get("log_url")
                break

        if not log_url:
            raise HTTPException(
                status_code=404,
                detail=f"No S3 log found for step {step_order}"
            )

        # Fetch from S3
        import boto3
        import json as json_mod
        from config import config

        # Parse s3://bucket/key from log_url
        s3_path = log_url.replace("s3://", "")
        bucket = s3_path.split("/", 1)[0]
        key = s3_path.split("/", 1)[1]

        s3 = boto3.client(
            "s3",
            region_name=config.AWS_REGION or "us-east-1",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )

        response = s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read().decode("utf-8")
        log_data = json_mod.loads(body)

        return {
            "step_order": step_order,
            "execution_id": execution_id,
            "log_url": log_url,
            "log_data": log_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching step logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching step logs: {str(e)}")


# ===================================================================
# SELF-LEARNING ENDPOINTS (Learn, Quality, Suggestions, Executions)
# ===================================================================

@router.post("/{recipe_id}/learn")
async def analyze_execution_learning(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    body: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Trigger learning analysis on a completed recipe execution (Stage 6).

    Body:
    - execution_id: The execution_id to analyze (required)

    Returns patterns, suggestions, and performance_metrics.
    """
    try:
        execution_id = body.get('execution_id')
        if not execution_id:
            raise HTTPException(status_code=400, detail="execution_id is required")

        # Validate recipe ownership
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        # Validate execution — try execution_id string first, fall back to numeric DB id
        execution = db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id,
            RecipeExecution.recipe_id == recipe.id
        ).first()

        if not execution and execution_id.isdigit():
            execution = db.query(RecipeExecution).filter(
                RecipeExecution.id == int(execution_id),
                RecipeExecution.recipe_id == recipe.id
            ).first()

        if not execution:
            raise HTTPException(
                status_code=404,
                detail=f"Execution '{execution_id}' not found for recipe '{recipe_id}'"
            )

        from core.services.recipe_learning_service import RecipeLearningService
        service = RecipeLearningService(db=db)
        result = service.analyze_execution(execution_id)

        return result

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Execution analysis resource not found for recipe {recipe_id}: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail="Execution or recipe not found")
    except Exception as e:
        logger.error(f"Error analyzing execution for recipe {recipe_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{recipe_id}/assess-quality")
async def assess_execution_quality(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    body: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Trigger quality assessment on a recipe execution (Stage 7).

    Body:
    - execution_id: The execution_id to assess (required)
    - learnings: Optional learnings dict from /learn endpoint for reliability scoring

    Returns quality_score, breakdown, grade, and bottlenecks.
    """
    try:
        execution_id = body.get('execution_id')
        if not execution_id:
            raise HTTPException(status_code=400, detail="execution_id is required")

        # Validate recipe ownership
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        # Validate execution — try execution_id string first, fall back to numeric DB id
        execution = db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id,
            RecipeExecution.recipe_id == recipe.id
        ).first()

        if not execution and execution_id.isdigit():
            execution = db.query(RecipeExecution).filter(
                RecipeExecution.id == int(execution_id),
                RecipeExecution.recipe_id == recipe.id
            ).first()

        if not execution:
            raise HTTPException(
                status_code=404,
                detail=f"Execution '{execution_id}' not found for recipe '{recipe_id}'"
            )

        learnings = body.get('learnings')

        from core.services.recipe_quality_service import RecipeQualityService
        service = RecipeQualityService(db=db)
        result = service.assess_quality(execution_id, learnings=learnings)

        return result

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Quality assessment resource not found for recipe {recipe_id}: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail="Execution or recipe not found")
    except Exception as e:
        logger.error(f"Error assessing quality for recipe {recipe_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{recipe_id}/suggestions")
async def get_recipe_suggestions(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get improvement suggestions from the recipe's learning_data.

    Returns latest suggestions, patterns, and performance metrics
    extracted from previous learning analyses.
    """
    try:
        # Validate recipe ownership
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        learning_data = recipe.learning_data or {}

        return {
            "recipe_id": recipe_id,
            "quality_score": recipe.quality_score,
            "suggestions": learning_data.get("latest_suggestions", []),
            "patterns": learning_data.get("latest_patterns", []),
            "performance_metrics": learning_data.get("latest_performance"),
            "last_analyzed_at": learning_data.get("last_analyzed_at"),
            "analysis_count": len(learning_data.get("analyses", [])),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting suggestions for recipe {recipe_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{recipe_id}/executions")
async def list_recipe_executions(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    status: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    List executions for a recipe with optional status filter.

    Query Parameters:
    - status: Filter by execution status (pending, running, completed, failed, cancelled)
    - skip: Pagination offset
    - limit: Pagination limit (1-100, default 20)

    Returns list of executions with quality scores.
    """
    try:
        # Validate recipe ownership
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        query = db.query(RecipeExecution).filter(
            RecipeExecution.recipe_id == recipe.id
        )

        if status:
            query = query.filter(RecipeExecution.status == status)

        total = query.count()

        executions = query.order_by(
            RecipeExecution.started_at.desc()
        ).offset(skip).limit(limit).all()

        return {
            "items": [ex.to_dict() for ex in executions],
            "total": total,
            "skip": skip,
            "limit": limit,
            "recipe_id": recipe_id,
            "recipe_quality_score": recipe.quality_score,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing executions for recipe {recipe_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================================================
# MARKETPLACE ENDPOINTS
# ===================================================================

@router.post("/submit")
async def submit_recipe_to_marketplace(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    recipe_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Submit a workspace recipe to the marketplace for approval.
    Trusted users (5+ approved items) auto-publish; others go to approval queue.

    Required fields:
    - recipe_id: ID of the workspace recipe to submit (template_id)
    - category: Optional marketplace category (uses recipe category if not provided)
    - icon: Optional marketplace icon URL
    """
    try:
        recipe_id = recipe_data.get('recipe_id')
        if not recipe_id:
            raise HTTPException(status_code=400, detail="recipe_id is required")

        # Get workspace recipe
        workspace_recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.template_id == recipe_id,
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.owner_type == 'workspace'
        ).first()

        if not workspace_recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found in workspace")

        # Look up database user ID
        from core.models.core import User as UserModel
        user_id_int = None
        if ctx.user and ctx.user.id:
            user = db.query(UserModel).filter(UserModel.clerk_user_id == ctx.user.id).first()
            if not user and ctx.user.email:
                user = db.query(UserModel).filter(UserModel.email == ctx.user.email).first()
            if user:
                user_id_int = user.id

        # Check if user is trusted (5+ approved marketplace items - agents OR recipes)
        from core.models.core import Agent
        approved_agent_count = db.query(Agent).filter(
            Agent.original_creator_id == user_id_int,
            Agent.owner_type == 'marketplace',
            Agent.is_approved == True
        ).count() if user_id_int else 0

        approved_recipe_count = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.original_creator_id == user_id_int,
            WorkflowRecipe.owner_type == 'marketplace',
            WorkflowRecipe.is_approved == True
        ).count() if user_id_int else 0

        total_approved = approved_agent_count + approved_recipe_count
        is_trusted = total_approved >= 5

        logger.info(f"User approval status - User ID: {user_id_int}, Approved items: {total_approved}, Is trusted: {is_trusted}")

        # Check if recipe already exists in marketplace
        existing = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.name == workspace_recipe.name,
            WorkflowRecipe.owner_type == 'marketplace'
        ).first()

        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"A marketplace recipe with name '{workspace_recipe.name}' already exists"
            )

        # Clone to marketplace
        marketplace_recipe = WorkflowRecipe(
            template_id=f"marketplace-{workspace_recipe.template_id}-{datetime.now().timestamp()}",
            name=workspace_recipe.name,
            description=workspace_recipe.description,
            template_definition=workspace_recipe.template_definition,
            tags=workspace_recipe.tags,
            steps=workspace_recipe.steps,
            inputs=workspace_recipe.inputs,
            outputs=workspace_recipe.outputs,
            execution_config=workspace_recipe.execution_config,
            schedule_config=workspace_recipe.schedule_config,
            recommended_agents=workspace_recipe.recommended_agents,
            required_tools=workspace_recipe.required_tools,
            preview_image=workspace_recipe.preview_image,
            documentation_url=workspace_recipe.documentation_url,
            version=workspace_recipe.version or '1.0',

            # Marketplace ownership
            owner_type='marketplace',
            owner_id='marketplace',
            workspace_id=None,

            # Creator tracking
            original_creator_id=user_id_int,
            created_by_user_id=user_id_int,
            cloned_from_id=workspace_recipe.id,

            # Approval
            is_approved=is_trusted,
            marketplace_category=recipe_data.get('category') or (workspace_recipe.tags[0] if workspace_recipe.tags else 'General'),
            marketplace_icon=recipe_data.get('icon') or workspace_recipe.marketplace_icon,

            # Visibility
            is_public=True,
            is_featured=False,
            is_system=False,

            # Stats
            install_count=0,
            use_count=0,

            created_by=(ctx.user.email if ctx.user and ctx.user.email else "system")
        )

        db.add(marketplace_recipe)
        db.commit()
        db.refresh(marketplace_recipe)

        logger.info(f"Marketplace recipe created - ID: {marketplace_recipe.id}, Name: {marketplace_recipe.name}, Approved: {marketplace_recipe.is_approved}")

        message = "Recipe published to marketplace successfully" if is_trusted else "Recipe submitted for marketplace approval"

        return {
            "success": True,
            "message": message,
            "item_id": marketplace_recipe.id,
            "auto_approved": is_trusted
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting recipe to marketplace: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/install/{recipe_id}")
async def install_recipe_from_marketplace(
    recipe_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Install a marketplace recipe to the user's workspace.
    Automatically clones the recipe and handles name collisions.
    Optionally auto-clones referenced agents if available in marketplace.
    """
    try:
        # Get the marketplace recipe
        marketplace_recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.id == recipe_id,
            WorkflowRecipe.owner_type == 'marketplace',
            WorkflowRecipe.is_approved == True
        ).first()

        if not marketplace_recipe:
            raise HTTPException(status_code=404, detail="Marketplace recipe not found")

        cloned_items = []
        warnings = []

        # Look up database user ID
        from core.models.core import User as UserModel
        user_id_int = None
        if ctx.user and ctx.user.id:
            user = db.query(UserModel).filter(UserModel.clerk_user_id == ctx.user.id).first()
            if not user and ctx.user.email:
                user = db.query(UserModel).filter(UserModel.email == ctx.user.email).first()
            if user:
                user_id_int = user.id

        # Check if recipe name already exists in workspace
        name_exists = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.name == marketplace_recipe.name,
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.owner_type == 'workspace'
        ).first() is not None

        recipe_name = f"{marketplace_recipe.name} (Copy)" if name_exists else marketplace_recipe.name

        # Generate unique template_id
        base_template_id = marketplace_recipe.template_id.replace('marketplace-', '')
        template_id = base_template_id
        counter = 1
        while db.query(WorkflowRecipe).filter(
            WorkflowRecipe.template_id == template_id,
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.owner_type == 'workspace'
        ).first():
            template_id = f"{base_template_id}-{counter}"
            counter += 1

        # Clone recipe to workspace
        cloned_recipe = WorkflowRecipe(
            template_id=template_id,
            name=recipe_name,
            description=marketplace_recipe.description,
            template_definition=marketplace_recipe.template_definition,
            tags=marketplace_recipe.tags,
            steps=marketplace_recipe.steps,
            inputs=marketplace_recipe.inputs,
            outputs=marketplace_recipe.outputs,
            execution_config=marketplace_recipe.execution_config,
            schedule_config=marketplace_recipe.schedule_config,
            recommended_agents=marketplace_recipe.recommended_agents,
            required_tools=marketplace_recipe.required_tools,
            preview_image=marketplace_recipe.preview_image,
            documentation_url=marketplace_recipe.documentation_url,
            version=marketplace_recipe.version,

            # Ownership swap
            owner_type='workspace',
            owner_id=str(ctx.workspace_id),
            workspace_id=ctx.workspace_id,
            created_by_user_id=user_id_int,

            # Tracking
            cloned_from_id=marketplace_recipe.id,
            original_creator_id=marketplace_recipe.original_creator_id,

            # Visibility
            is_public=True,
            is_featured=False,
            is_system=False,
            is_approved=True,

            # Stats
            install_count=0,
            use_count=0,

            created_by=(ctx.user.email if ctx.user and ctx.user.email else "system")
        )

        db.add(cloned_recipe)
        db.flush()

        cloned_items.append({
            "type": "recipe",
            "name": recipe_name,
            "id": cloned_recipe.id,
            "template_id": cloned_recipe.template_id
        })

        # TODO: Auto-clone referenced agents if available in marketplace
        # This would require parsing template_definition and checking for agent references

        # Increment marketplace recipe install count
        marketplace_recipe.install_count += 1

        # Record installation in marketplace_installs using a savepoint so
        # failures don't roll back the main recipe install.
        from sqlalchemy import text
        install_query = text("""
            INSERT INTO marketplace_installs (user_id, marketplace_recipe_id, cloned_recipe_id, version, installed_at)
            VALUES (:user_id, :marketplace_recipe_id, :cloned_recipe_id, :version, NOW())
            ON CONFLICT DO NOTHING
        """)

        try:
            with db.begin_nested():
                db.execute(install_query, {
                    "user_id": user_id_int,
                    "marketplace_recipe_id": marketplace_recipe.id,
                    "cloned_recipe_id": cloned_recipe.id,
                    "version": marketplace_recipe.version
                })
        except Exception as e:
            # If marketplace_installs table doesn't have recipe columns yet, log warning
            logger.warning(f"Could not record recipe install in marketplace_installs: {e}")
            warnings.append("Install tracking not available for recipes yet")

        db.commit()

        return {
            "success": True,
            "message": f"{marketplace_recipe.name} installed successfully",
            "cloned_items": cloned_items,
            "warnings": warnings
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error installing marketplace recipe {recipe_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================================================
# RECIPE WEBHOOK ENDPOINT (no auth — URL is the secret)
# ===================================================================

webhook_router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])


@webhook_router.get("/recipe/{webhook_id}")
async def recipe_webhook_verify(
    webhook_id: str,
    db: Session = Depends(get_db),
):
    """Verification endpoint — Jira/GitHub/Slack send GET to validate the URL exists."""
    recipe = db.query(WorkflowRecipe).filter(
        WorkflowRecipe.schedule_config["webhook_id"].astext == webhook_id,
        WorkflowRecipe.owner_type == "workspace",
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Unknown webhook")

    return {"status": "ok", "recipe": recipe.name}


@webhook_router.post("/recipe/{webhook_id}")
async def recipe_webhook(
    webhook_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Trigger a recipe execution via webhook.

    The webhook_id is a persistent secret stored in the recipe's
    schedule_config.webhook_id. No authentication required — the
    URL itself is the credential.

    Body (optional):
    - Any JSON payload — passed as input_data to the recipe executor.
    - Also accepts form-encoded payloads (GitHub ping events).
    """
    import json as _json

    # Parse body from any content type
    content_type = request.headers.get("content-type", "")
    try:
        if "application/json" in content_type:
            body = await request.json()
        elif "form" in content_type:
            form = await request.form()
            # GitHub form-encoded wraps JSON in a "payload" field
            payload_str = form.get("payload", "{}")
            body = _json.loads(payload_str) if isinstance(payload_str, str) else {}
        else:
            # Try JSON first, fall back to empty dict
            raw = await request.body()
            try:
                body = _json.loads(raw) if raw else {}
            except (ValueError, _json.JSONDecodeError):
                body = {"raw": raw.decode("utf-8", errors="replace")}
    except Exception:
        body = {}

    # Look up recipe by webhook_id in schedule_config
    recipe = db.query(WorkflowRecipe).filter(
        WorkflowRecipe.schedule_config["webhook_id"].astext == webhook_id,
        WorkflowRecipe.owner_type == "workspace",
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Unknown webhook")

    # Verify HMAC signature if a webhook secret is configured
    webhook_secret = (recipe.schedule_config or {}).get("webhook_secret") or config.WEBHOOK_SECRET
    if webhook_secret:
        sig_header = (
            request.headers.get("x-hub-signature-256")
            or request.headers.get("x-composio-signature")
            or request.headers.get("x-webhook-signature")
        )
        if sig_header:
            raw_body = await request.body()
            expected_sig = sig_header.removeprefix("sha256=")
            computed = hmac.new(
                webhook_secret.encode("utf-8"),
                raw_body,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(computed, expected_sig):
                logger.warning("[webhook] HMAC signature mismatch for recipe webhook %s", webhook_id)
                raise HTTPException(status_code=401, detail="Invalid webhook signature")

    if not recipe.steps:
        raise HTTPException(status_code=400, detail="Recipe has no steps")

    execution_id = f"webhook-{uuid4().hex[:12]}"
    execution = RecipeExecution(
        execution_id=execution_id,
        recipe_id=recipe.id,
        workspace_id=recipe.workspace_id,
        status="pending",
        input_data=body,
        triggered_by="webhook",
        execution_metadata={
            "execution_type": "webhook",
            "webhook_id": webhook_id,
            "total_steps": len(recipe.steps),
        },
    )
    db.add(execution)
    recipe.use_count += 1
    recipe.last_used_at = datetime.now()
    db.commit()

    logger.info("[webhook] Recipe %d (%s) triggered via webhook %s, execution=%s",
                recipe.id, recipe.name, webhook_id, execution_id)

    from api.recipe_executor import launch_recipe_task
    launch_recipe_task(
        recipe_execution_id=execution_id,
        recipe_id=recipe.id,
        workspace_id=recipe.workspace_id,
        input_data=body,
    )

    return {
        "status": "started",
        "execution_id": execution_id,
        "recipe_name": recipe.name,
        "total_steps": len(recipe.steps),
    }
