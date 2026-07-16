"""
Orchestration Board Bridge — Mission ↔ Kanban Board Integration
================================================================

Creates and syncs BoardTask rows for orchestration runs (missions) and
orchestration tasks, so the mission lifecycle is visible on the kanban board.

Design:
- Mission → parent BoardTask (source_type='orchestration')
- Each orchestration task → child BoardTask (source_type='orchestration_task')
- State changes on orchestration tasks sync to board task status via BOARD_STATUS_MAP
- All writes happen in the caller's transaction (no internal commit)
- Non-fatal: log warnings on missing board tasks, never crash the coordinator

Source: PRD-82A Section 4.3, PRD-101 Section 7.2
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from core.models.core import BoardTask
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import (
    BOARD_STATUS_MAP,
    TERMINAL_RUN_STATES,
    RunState,
    TaskState,
)

# Priority → SLA deadline hours
_PRIORITY_SLA_HOURS: dict[str, int] = {
    "urgent": 4,
    "high": 12,
    "medium": 24,
    "low": 72,
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BOARD_STATUS_MAP uses PRD terminology; BoardTask.status uses kanban terms.
# This mapping bridges the two vocabularies.
# ---------------------------------------------------------------------------

_ORCHESTRATION_TO_BOARD_STATUS: dict[str, str] = {
    "backlog": "inbox",
    "todo": "inbox",
    "in_progress": "in_progress",
    "in_review": "review",
    "done": "done",
    "blocked": "blocked",
    "cancelled": "done",
}


def _resolve_board_status(task_state: TaskState) -> str:
    """
    Map an orchestration TaskState to a BoardTask status string.

    Uses BOARD_STATUS_MAP (PRD terms) → _ORCHESTRATION_TO_BOARD_STATUS (kanban terms).
    Falls back to 'inbox' for unknown states.
    """
    prd_status = BOARD_STATUS_MAP.get(task_state, "inbox")
    return _ORCHESTRATION_TO_BOARD_STATUS.get(prd_status, "inbox")


# ---------------------------------------------------------------------------
# Mission-level board task
# ---------------------------------------------------------------------------


def create_mission_board_task(
    db: Session,
    run: OrchestrationRun,
) -> Optional[BoardTask]:
    """
    Create a parent BoardTask for a mission (orchestration run).

    The board task uses source_type='orchestration' and links via
    orchestration_run_id FK. Idempotent: returns None if already exists.

    Args:
        db: SQLAlchemy session (caller manages transaction).
        run: The OrchestrationRun to create a board task for.

    Returns:
        The created BoardTask, or None if one already exists.
    """
    # Idempotent check
    existing = db.query(BoardTask.id).filter(
        BoardTask.source_type == "orchestration",
        BoardTask.orchestration_run_id == run.id,
    ).first()
    if existing:
        logger.debug(
            "Board task already exists for mission run %s (board_task_id=%s)",
            run.id,
            existing.id,
        )
        return None

    board_task = BoardTask(
        workspace_id=run.workspace_id,
        title=f"Mission: {run.goal[:200]}" if run.goal else "Mission",
        description=run.goal,
        status="inbox",
        priority="medium",
        review_mode="auto",
        created_by_type="system",
        created_by_id="coordinator",
        source_type="orchestration",
        orchestration_run_id=run.id,
        tags=["mission", "orchestration"],
        sla_deadline=datetime.now(timezone.utc) + timedelta(hours=_PRIORITY_SLA_HOURS.get("medium", 24)),
    )
    db.add(board_task)
    db.flush()  # Get the ID without committing

    logger.info(
        "Created mission board task %s for run %s",
        board_task.id,
        run.id,
    )
    return board_task


# ---------------------------------------------------------------------------
# Task-level board task
# ---------------------------------------------------------------------------


def create_task_board_task(
    db: Session,
    run: OrchestrationRun,
    task: OrchestrationTask,
) -> Optional[BoardTask]:
    """
    Create a child BoardTask for an orchestration task.

    Links to the parent mission board task via parent_task_id, and sets
    source_type='orchestration_task' with orchestration_task_id FK.
    Idempotent: returns None if already exists.

    Args:
        db: SQLAlchemy session (caller manages transaction).
        run: The parent OrchestrationRun (used to find mission board task).
        task: The OrchestrationTask to create a board task for.

    Returns:
        The created BoardTask, or None if one already exists.
    """
    # Idempotent check
    existing = db.query(BoardTask.id).filter(
        BoardTask.source_type == "orchestration_task",
        BoardTask.orchestration_task_id == task.id,
    ).first()
    if existing:
        logger.debug(
            "Board task already exists for orchestration task %s (board_task_id=%s)",
            task.id,
            existing.id,
        )
        return None

    # Find parent mission board task
    parent_board_task = db.query(BoardTask).filter(
        BoardTask.source_type == "orchestration",
        BoardTask.orchestration_run_id == run.id,
    ).first()

    parent_id = parent_board_task.id if parent_board_task else None
    if parent_id is None:
        logger.warning(
            "No parent board task found for run %s when creating task board task for %s",
            run.id,
            task.id,
        )

    # Resolve initial board status from orchestration task state
    task_state = TaskState(task.state)
    board_status = _resolve_board_status(task_state)

    board_task = BoardTask(
        workspace_id=run.workspace_id,
        title=task.title,
        description=task.description,
        status=board_status,
        priority="medium",
        review_mode="auto",
        assigned_agent_id=task.assigned_agent_id,
        created_by_type="system",
        created_by_id="coordinator",
        parent_task_id=parent_id,
        source_type="orchestration_task",
        orchestration_task_id=task.id,
        tags=["mission", f"mission:{(run.goal or 'Mission')[:60]}"],
        planning_data={
            "sequence_number": task.sequence_number,
            "agent_role": task.agent_role,
        },
        sla_deadline=datetime.now(timezone.utc) + timedelta(hours=_PRIORITY_SLA_HOURS.get("medium", 24)),
    )
    db.add(board_task)
    db.flush()

    logger.info(
        "Created task board task %s for orchestration task %s (seq=%s)",
        board_task.id,
        task.id,
        task.sequence_number,
    )
    return board_task


# ---------------------------------------------------------------------------
# Board status sync
# ---------------------------------------------------------------------------


def sync_board_status(
    db: Session,
    task: OrchestrationTask,
) -> None:
    """
    Sync a BoardTask's status to match the orchestration task's current state.

    Reads task.state, maps via BOARD_STATUS_MAP → kanban status, and updates
    the linked BoardTask. Handles missing board task gracefully (log warning).

    Args:
        db: SQLAlchemy session (caller manages transaction).
        task: The OrchestrationTask whose linked board task should be updated.
    """
    board_task = db.query(BoardTask).filter(
        BoardTask.source_type == "orchestration_task",
        BoardTask.orchestration_task_id == task.id,
    ).first()

    if board_task is None:
        logger.warning(
            "No linked board task found for orchestration task %s — skipping sync",
            task.id,
        )
        return

    task_state = TaskState(task.state)
    new_status = _resolve_board_status(task_state)

    if board_task.status == new_status:
        return  # No change needed

    old_status = board_task.status
    board_task.status = new_status

    # Sync agent assignment if it changed
    if task.assigned_agent_id and board_task.assigned_agent_id != task.assigned_agent_id:
        board_task.assigned_agent_id = task.assigned_agent_id

    # Sync started_at when moving to in_progress
    if new_status == "in_progress" and board_task.started_at is None:
        board_task.started_at = task.started_at

    # Sync completed_at when moving to done
    if new_status == "done" and board_task.completed_at is None:
        board_task.completed_at = task.completed_at

    # Store task output as result when task completes or is verified
    if task_state in (TaskState.COMPLETED, TaskState.VERIFIED) and task.output:
        board_task.result = task.output

    # Store failure info and blocked metadata
    if task_state in (TaskState.FAILED, TaskState.STALLED):
        board_task.error_message = task.failure_detail or task.failure_reason_code
        if new_status == "blocked" and board_task.blocked_at is None:
            board_task.blocked_at = datetime.now(timezone.utc)
            board_task.blocked_reason = task.failure_detail or task.failure_reason_code

    # Clear blocked fields when unblocked
    if new_status != "blocked" and old_status == "blocked":
        board_task.blocked_at = None
        board_task.blocked_reason = None

    db.flush()

    logger.info(
        "Synced board task %s status: %s → %s (orchestration task %s state=%s)",
        board_task.id,
        old_status,
        new_status,
        task.id,
        task_state.value,
    )


# ---------------------------------------------------------------------------
# Mission-level board status sync
# ---------------------------------------------------------------------------

_RUN_STATE_TO_BOARD_STATUS: dict[str, str] = {
    RunState.PENDING.value: "inbox",
    RunState.PLANNING.value: "inbox",
    RunState.AWAITING_APPROVAL.value: "review",
    RunState.RUNNING.value: "in_progress",
    RunState.PAUSED.value: "blocked",
    RunState.REPLANNING.value: "in_progress",
    RunState.VERIFYING.value: "in_progress",
    RunState.AWAITING_HUMAN.value: "review",
    RunState.COMPLETED.value: "done",
    # PRD-204 S4 (OS-review F023): a failed mission's card used to render
    # "done" -- the board was lying. "failed" is a valid board status
    # (api/board_tasks.VALID_STATUSES). cancelled stays "done": the user
    # deliberately closed it.
    RunState.FAILED.value: "failed",
    RunState.CANCELLED.value: "done",
}


def sync_mission_board_status(
    db: Session,
    run: OrchestrationRun,
) -> None:
    """
    Sync the mission-level BoardTask status to match the run's current state.

    Should be called whenever the run transitions state — especially to
    terminal states (completed, failed, cancelled) so the mission card
    moves out of inbox/in_progress on the board.

    Args:
        db: SQLAlchemy session (caller manages transaction).
        run: The OrchestrationRun whose board card should be updated.
    """
    board_task = db.query(BoardTask).filter(
        BoardTask.source_type == "orchestration",
        BoardTask.orchestration_run_id == run.id,
    ).first()

    if board_task is None:
        logger.debug(
            "No mission board task found for run %s — skipping sync",
            run.id,
        )
        return

    run_state = RunState(run.state)
    new_status = _RUN_STATE_TO_BOARD_STATUS.get(run_state.value, "inbox")

    if board_task.status == new_status:
        return

    old_status = board_task.status
    board_task.status = new_status

    # Sync timestamps
    if new_status == "in_progress" and board_task.started_at is None:
        board_task.started_at = run.started_at or datetime.now(timezone.utc)

    # "failed" is terminal for the card too (PRD-204 S4) -- stamp completion.
    if new_status in ("done", "failed") and board_task.completed_at is None:
        board_task.completed_at = run.completed_at or datetime.now(timezone.utc)

    # Store failure info when mission fails
    if run_state == RunState.FAILED:
        board_task.error_message = run.stop_detail or run.stop_reason or "Mission failed"

    # Blocked metadata for paused runs
    if new_status == "blocked" and old_status != "blocked":
        board_task.blocked_at = datetime.now(timezone.utc)
        board_task.blocked_reason = "Mission paused"
    elif new_status != "blocked" and old_status == "blocked":
        board_task.blocked_at = None
        board_task.blocked_reason = None

    db.flush()

    logger.info(
        "Synced mission board task %s status: %s → %s (run %s state=%s)",
        board_task.id,
        old_status,
        new_status,
        run.id,
        run_state.value,
    )
