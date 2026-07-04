"""
Board Task Bridge — Recipe ↔ Kanban Board Integration
======================================================

Creates and updates BoardTask rows when recipe executions run,
so recipes appear on the Kanban board alongside manual tasks.

All functions are non-blocking: callers wrap in try/except.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from core.models.core import BoardTask

logger = logging.getLogger(__name__)


def create_recipe_board_task(
    db: Session,
    recipe,          # WorkflowRecipe ORM instance
    execution,       # RecipeExecution ORM instance
) -> Optional[BoardTask]:
    """Create a BoardTask linked to a recipe execution (idempotent)."""
    steps = recipe.steps or []
    first_agent_id = steps[0].get('agent_id') if steps else None
    total_steps = len(steps)
    triggered_by = getattr(execution, 'triggered_by', 'manual') or 'manual'

    # Check for existing task (partial unique index prevents duplicates at DB level)
    existing = db.query(BoardTask.id).filter(
        BoardTask.source_type == 'recipe',
        BoardTask.source_id == execution.execution_id,
    ).first()
    if existing:
        return None

    task = BoardTask(
        workspace_id=execution.workspace_id,
        title=f"Recipe: {recipe.name}",
        description=recipe.description,
        status='in_progress',
        priority='medium',
        review_mode='auto',
        assigned_agent_id=first_agent_id,
        created_by_type='recipe',
        source_type='recipe',
        source_id=execution.execution_id,
        tags=['recipe', triggered_by],
        started_at=datetime.now(timezone.utc),
        planning_data={
            'recipe_id': recipe.id,
            'total_steps': total_steps,
            'execution_id': execution.execution_id,
        },
    )
    db.add(task)
    db.commit()

    logger.info(
        "[board_bridge] Created board task for recipe execution %s",
        execution.execution_id,
    )
    return None


def update_recipe_board_task_progress(
    db: Session,
    execution_id: str,
    current_step: int,
    total_steps: int,
) -> None:
    """Update step progress on the linked BoardTask."""
    task = db.query(BoardTask).filter(
        BoardTask.source_type == 'recipe',
        BoardTask.source_id == execution_id,
    ).first()

    if not task:
        return

    planning = dict(task.planning_data or {})
    planning['step_progress'] = {'current': current_step, 'total': total_steps}
    task.planning_data = planning

    # Flag JSONB mutation for SQLAlchemy
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(task, 'planning_data')
    db.commit()


def complete_recipe_board_task(
    db: Session,
    execution_id: str,
    *,
    success: bool,
    result: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Move the linked BoardTask to done (success) or review (failure)."""
    task = db.query(BoardTask).filter(
        BoardTask.source_type == 'recipe',
        BoardTask.source_id == execution_id,
    ).first()

    if not task:
        return

    # PRD-185 S4: honor the success flag — a failed playbook must show 'failed',
    # not silently close 'done' (that hid a ~17-day OpenRouter 402 outage where
    # the board reported green while every run failed).
    task.status = 'done' if success else 'failed'
    task.completed_at = datetime.now(timezone.utc)

    if result:
        task.result = result
    if error_message:
        task.error_message = error_message

    db.commit()

    logger.info(
        "[board_bridge] Completed board task for execution %s → %s",
        execution_id,
        task.status,
    )
