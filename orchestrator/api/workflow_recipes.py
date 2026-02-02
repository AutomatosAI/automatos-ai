"""
Workflow Recipes API
====================

CRUD operations for workflow recipes that users can browse,
customize, and use to create workflows.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from core.database.database import get_db
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflow-recipes", tags=["workflow-recipes"])

# Import the model from main models file
from core.models import WorkflowTemplate as WorkflowRecipe  # Aliased for transition
from core.models import Workflow, WorkflowExecution, Agent, workflow_agents
from core.models.core import RecipeExecution
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext


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
        raise HTTPException(status_code=500, detail=f"Error listing recipes: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Error getting recipe: {str(e)}")


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
        execution_config = recipe_data.get('execution_config') or {
            'mode': 'sequential',
            'max_retries': 1,
            'retry_delay': 5,
            'per_step_timeout': 300,
            'total_timeout': 1800,
            'quality_threshold': 0.7,
            'auto_learn': True,
        }

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
            schedule_config=recipe_data.get('schedule_config'),
            recommended_agents=recipe_data.get('recommended_agents', []),
            required_tools=recipe_data.get('required_tools', []),
            is_public=recipe_data.get('is_public', True),
            is_featured=recipe_data.get('is_featured', False),
            is_system=False,  # User-created recipes are never system recipes
            preview_image=recipe_data.get('preview_image'),
            documentation_url=recipe_data.get('documentation_url'),
            version=recipe_data.get('version', '1.0'),
            created_by=recipe_data.get('created_by', ctx.user.email if ctx.user else "anonymous")
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
        raise HTTPException(status_code=500, detail=f"Error creating recipe: {str(e)}")


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
                    raise HTTPException(
                        status_code=400,
                        detail=f"Agent IDs not found in workspace: {missing}"
                    )

        # Validate execution_config if updated
        if 'execution_config' in recipe_data:
            is_valid, error = recipe.validate_execution_config()
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid execution_config: {error}")

        # Validate schedule_config if updated
        if 'schedule_config' in recipe_data:
            is_valid, error = recipe.validate_schedule_config()
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid schedule_config: {error}")

        recipe.updated_at = datetime.now()
        db.commit()
        db.refresh(recipe)

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
        raise HTTPException(status_code=500, detail=f"Error updating recipe: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Error deleting recipe: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Error recording usage: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Error listing categories: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Error listing featured recipes: {str(e)}")


@router.post("/{recipe_id}/execute")
async def execute_recipe(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    body: Dict[str, Any] = Body(default={}),
    db: Session = Depends(get_db)
):
    """
    Execute a recipe directly via the 9-stage workflow engine.

    Creates a Workflow from recipe data, a WorkflowExecution record, and launches
    execute_workflow_with_progress as an async task. Also creates a RecipeExecution
    record for recipe-specific tracking.

    Body (optional):
    - input_data: Dict matching the recipe's inputs schema
    """
    try:
        logger.info(f"[execute_recipe] Starting for recipe_id={recipe_id}, workspace={ctx.workspace_id}")

        # Fetch recipe and validate ownership
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            logger.warning(f"[execute_recipe] Recipe not found: {recipe_id}")
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        logger.info(f"[execute_recipe] Recipe found: {recipe.name}, steps={len(recipe.steps or [])}")

        input_data = body.get('input_data') or {}

        # Fill in defaults for missing required inputs (don't block execution)
        if recipe.inputs:
            for param_name, param_def in recipe.inputs.items():
                if isinstance(param_def, dict) and param_name not in input_data:
                    default = param_def.get('default', '')
                    input_data[param_name] = default
                    if param_def.get('required'):
                        logger.info(f"[execute_recipe] Auto-filled missing required input '{param_name}' with default")

        # Build workflow_definition from recipe steps (RECIPE mode)
        recipe_steps = recipe.steps or []
        workflow_definition = {
            "steps": recipe_steps,
            "config": recipe.execution_config or {},
            "goal": recipe.description,
            "category": "recipe",
            "priority": "medium",
            "version": recipe.version or "1.0",
            "recipe_template_id": recipe.template_id,
        }

        # Create a Workflow from recipe data
        timestamp = datetime.now().strftime("%H%M%S")
        workflow = Workflow(
            workspace_id=ctx.workspace_id,
            name=f"{recipe.name} {timestamp}",
            description=recipe.description,
            workflow_definition=workflow_definition,
            status='active',
            created_by=ctx.user.email if ctx.user else 'system',
        )
        db.add(workflow)
        db.flush()  # Get workflow.id without committing

        # Associate agents from recipe steps to the Workflow
        agent_ids_from_steps = list({
            step.get('agent_id') for step in recipe_steps if step.get('agent_id')
        })
        if agent_ids_from_steps:
            agent_objects = db.query(Agent).filter(
                Agent.id.in_(agent_ids_from_steps),
                Agent.workspace_id == ctx.workspace_id
            ).all()
            workflow.agents.extend(agent_objects)

        # Pick first step's agent as default agent_id for WorkflowExecution (required field)
        default_agent_id = None
        if recipe_steps and recipe_steps[0].get('agent_id'):
            default_agent_id = recipe_steps[0]['agent_id']
        if not default_agent_id and agent_ids_from_steps:
            default_agent_id = agent_ids_from_steps[0]
        if not default_agent_id:
            # Fallback: get any active agent in workspace
            fallback_agent = db.query(Agent).filter(
                Agent.workspace_id == ctx.workspace_id,
                Agent.status == 'active'
            ).first()
            if fallback_agent:
                default_agent_id = fallback_agent.id
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No active agents available to execute this recipe"
                )

        # Create WorkflowExecution record
        workflow_execution = WorkflowExecution(
            workflow_id=workflow.id,
            agent_id=default_agent_id,
            workspace_id=ctx.workspace_id,
            input_data=input_data or {},
            status='pending',
        )
        db.add(workflow_execution)
        db.flush()

        # Create RecipeExecution record for recipe-specific tracking
        recipe_execution_id = f"exec-{uuid4().hex[:12]}"
        recipe_execution = RecipeExecution(
            execution_id=recipe_execution_id,
            recipe_id=recipe.id,
            workspace_id=ctx.workspace_id,
            status='running',
            input_data=input_data,
            current_stage=1,
            current_step=0,
            triggered_by=ctx.user.email if ctx.user else 'anonymous',
            execution_metadata={
                'workflow_id': workflow.id,
                'workflow_execution_id': workflow_execution.id,
            },
        )
        db.add(recipe_execution)

        # Update recipe usage stats
        recipe.use_count += 1
        recipe.last_used_at = datetime.now()

        db.commit()
        db.refresh(workflow)
        db.refresh(workflow_execution)

        logger.info(
            f"Recipe execution started: recipe={recipe_id}, "
            f"workflow_id={workflow.id}, execution_id={workflow_execution.id}"
        )

        # Launch execute_workflow_with_progress as async task
        import asyncio
        from api.workflows import execute_workflow_with_progress
        asyncio.create_task(
            execute_workflow_with_progress(workflow_execution.id, {})
        )

        return {
            "workflow_id": workflow.id,
            "workflow_execution_id": workflow_execution.id,
            "recipe_execution_id": recipe_execution.execution_id,
            "status": "started",
            "message": "Recipe execution started",
        }

    except HTTPException as he:
        logger.warning(f"[execute_recipe] HTTPException: status={he.status_code}, detail={he.detail}")
        raise
    except Exception as e:
        logger.error(f"[execute_recipe] Unhandled error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error executing recipe: {str(e)}")


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

        # Validate execution belongs to this recipe
        execution = db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id,
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
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing execution for recipe {recipe_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing execution: {str(e)}")


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

        # Validate execution belongs to this recipe
        execution = db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id,
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
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error assessing quality for recipe {recipe_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error assessing quality: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Error getting suggestions: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Error listing executions: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Error submitting recipe: {str(e)}")


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

        # Record installation in marketplace_installs
        from sqlalchemy import text
        install_query = text("""
            INSERT INTO marketplace_installs (user_id, marketplace_recipe_id, cloned_recipe_id, version, installed_at)
            VALUES (:user_id, :marketplace_recipe_id, :cloned_recipe_id, :version, NOW())
            ON CONFLICT DO NOTHING
        """)

        try:
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
        raise HTTPException(status_code=500, detail=f"Error installing recipe: {str(e)}")
