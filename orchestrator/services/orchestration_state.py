"""
Orchestration State Transition Service — PRD-82A
=================================================

Validated state transitions with dual-write event emission for both
orchestration tasks and runs.

Design principles:
- Dual-write: state change on row + append-only event in SAME transaction.
- Optimistic locking: StaleDataError → ConflictError (caller retries).
- Caller manages transaction boundary — no internal commit/rollback.
- Timestamps: started_at set on entering active, completed_at on terminal.

Source: PRD-82A Section 5 (principles), PRD-101 Section 3.9
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from core.models.orchestration import (
    OrchestrationEvent,
    OrchestrationRun,
    OrchestrationTask,
    RunTransition,
    TaskTransition,
)
from core.models.orchestration_enums import (
    ALLOWED_RUN_TRANSITIONS,
    ALLOWED_TASK_TRANSITIONS,
    RUN_STATE_TYPE,
    TASK_STATE_TYPE,
    TERMINAL_RUN_STATES,
    TERMINAL_TASK_STATES,
    ActorType,
    EventType,
    RunState,
    StateType,
    TaskState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed by the state machine."""

    def __init__(self, entity_type: str, entity_id: Any, current_state: str, target_state: str):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.current_state = current_state
        self.target_state = target_state
        super().__init__(
            f"Invalid {entity_type} transition: {current_state} → {target_state} "
            f"(entity {entity_id})"
        )


class ConflictError(Exception):
    """Raised when optimistic locking detects a concurrent modification."""

    def __init__(self, entity_type: str, entity_id: Any):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(
            f"Optimistic lock conflict on {entity_type} {entity_id} — "
            f"another transaction modified this row"
        )


# ---------------------------------------------------------------------------
# Task transitions
# ---------------------------------------------------------------------------


def transition_task(
    db: Session,
    task: OrchestrationTask,
    new_state: TaskState,
    actor_type: ActorType,
    actor_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> TaskTransition:
    """
    Transition an orchestration task to a new state with dual-write event.

    Validates the transition against ALLOWED_TASK_TRANSITIONS, updates
    state/state_type/timestamps, creates an OrchestrationEvent in the
    same transaction, and returns a frozen TaskTransition record.

    Args:
        db: SQLAlchemy session (caller manages transaction).
        task: The OrchestrationTask ORM instance to transition.
        new_state: Target TaskState.
        actor_type: Who is triggering this transition.
        actor_id: Optional identifier of the actor (agent ID, user ID, etc.).
        reason: Optional human-readable reason for the transition.

    Returns:
        Frozen TaskTransition record (PRD-123 Pattern #1).

    Raises:
        InvalidTransitionError: If the transition is not allowed.
        ConflictError: If optimistic locking detects a concurrent modification.
    """
    current_state = TaskState(task.state)

    # Validate transition
    allowed = ALLOWED_TASK_TRANSITIONS.get(current_state, frozenset())
    if new_state not in allowed:
        raise InvalidTransitionError(
            entity_type="task",
            entity_id=task.id,
            current_state=current_state.value,
            target_state=new_state.value,
        )

    old_state_value = task.state
    now = datetime.now(timezone.utc)
    new_state_type = TASK_STATE_TYPE[new_state]

    # Update state fields
    task.state = new_state.value
    task.state_type = new_state_type.value

    # Update timestamps based on new state
    if new_state == TaskState.RUNNING and task.started_at is None:
        task.started_at = now

    if new_state in TERMINAL_TASK_STATES:
        task.completed_at = now

    # Build event payload
    payload = {}
    if reason:
        payload["reason"] = reason
    if task.failure_reason_code and new_state in (TaskState.FAILED, TaskState.SKIPPED):
        payload["failure_reason_code"] = task.failure_reason_code

    # Create event (dual-write)
    event = OrchestrationEvent(
        run_id=task.run_id,
        task_id=task.id,
        event_type=_task_state_to_event_type(new_state).value,
        actor_type=actor_type.value,
        actor_id=actor_id,
        old_state=old_state_value,
        new_state=new_state.value,
        payload=payload if payload else None,
    )
    db.add(event)

    # Flush to trigger optimistic lock check — caller manages rollback
    try:
        db.flush()
    except StaleDataError:
        raise ConflictError(entity_type="task", entity_id=task.id)

    # PRD-123 Pattern #1: Produce frozen transition record
    transition = TaskTransition(
        task_id=task.id,
        from_state=old_state_value,
        to_state=new_state.value,
        triggered_by=actor_type.value,
        reason=reason,
    )

    logger.info(
        "Task %s transitioned: %s → %s (actor=%s/%s)",
        task.id,
        old_state_value,
        new_state.value,
        actor_type.value,
        actor_id,
    )

    return transition


# ---------------------------------------------------------------------------
# Run transitions
# ---------------------------------------------------------------------------


def transition_run(
    db: Session,
    run: OrchestrationRun,
    new_state: RunState,
    actor_type: ActorType,
    actor_id: Optional[str] = None,
    reason: Optional[str] = None,
    stop_reason: Optional[str] = None,
    stop_detail: Optional[str] = None,
) -> RunTransition:
    """
    Transition an orchestration run to a new state with dual-write event.

    Same pattern as transition_task but for runs. Returns a frozen
    RunTransition record (PRD-123 Pattern #1).

    Args:
        db: SQLAlchemy session (caller manages transaction).
        run: The OrchestrationRun ORM instance to transition.
        new_state: Target RunState.
        actor_type: Who is triggering this transition.
        actor_id: Optional identifier of the actor.
        reason: Optional human-readable reason for the transition.
        stop_reason: Named stop reason (PRD-123 Pattern #6), set on terminal transitions.
        stop_detail: Human-readable detail about why the mission stopped.

    Returns:
        Frozen RunTransition record (PRD-123 Pattern #1).

    Raises:
        InvalidTransitionError: If the transition is not allowed.
        ConflictError: If optimistic locking detects a concurrent modification.
    """
    current_state = RunState(run.state)

    # Validate transition
    allowed = ALLOWED_RUN_TRANSITIONS.get(current_state, frozenset())
    if new_state not in allowed:
        raise InvalidTransitionError(
            entity_type="run",
            entity_id=run.id,
            current_state=current_state.value,
            target_state=new_state.value,
        )

    old_state_value = run.state
    now = datetime.now(timezone.utc)
    new_state_type = RUN_STATE_TYPE[new_state]

    # Update state fields
    run.state = new_state.value
    run.state_type = new_state_type.value

    # Update timestamps based on new state
    if new_state == RunState.RUNNING and run.started_at is None:
        run.started_at = now

    if new_state in TERMINAL_RUN_STATES:
        run.completed_at = now
        # PRD-123 Pattern #6: Named Stop Reasons
        if stop_reason:
            run.stop_reason = stop_reason
        if stop_detail:
            run.stop_detail = stop_detail

    # Build event payload
    payload = {}
    if reason:
        payload["reason"] = reason
    if stop_reason:
        payload["stop_reason"] = stop_reason
    if stop_detail:
        payload["stop_detail"] = stop_detail

    # Create event (dual-write)
    event = OrchestrationEvent(
        run_id=run.id,
        task_id=None,
        event_type=_run_state_to_event_type(new_state).value,
        actor_type=actor_type.value,
        actor_id=actor_id,
        old_state=old_state_value,
        new_state=new_state.value,
        payload=payload if payload else None,
    )
    db.add(event)

    # Flush to trigger optimistic lock check — caller manages rollback
    try:
        db.flush()
    except StaleDataError:
        raise ConflictError(entity_type="run", entity_id=run.id)

    # Sync mission board card status (non-fatal)
    try:
        from services.orchestration_board_bridge import sync_mission_board_status
        sync_mission_board_status(db, run)
    except Exception:
        logger.warning(
            "Failed to sync mission board status for run %s", run.id, exc_info=True,
        )

    # PRD-204 S3: mission terminal choke point -- EVERY run transition to a
    # terminal state flows through here (coordinator, reconciler, joiner,
    # human cancel/reject), so one fail-soft hook covers all producers.
    # watch_ingest_terminal never raises (documented fail-soft seam); the
    # extra guard mirrors the board-sync seam above so even a broken import
    # cannot fail the transition.
    if new_state in TERMINAL_RUN_STATES:
        try:
            from services.watch_hooks import watch_ingest_terminal

            cost_snapshot = {
                "tokens_used": run.tokens_used or 0,
                "budget_spent": run.budget_spent or {},
            }
            watch_ingest_terminal(
                db,
                workspace_id=run.workspace_id,
                target_type="mission",
                target_id=str(run.id),
                terminal_state=new_state.value,
                summary=stop_detail or reason,
                cost_snapshot=cost_snapshot,
                output_pointer=(
                    f"mission:{run.id}:output_summary"
                    if run.output_summary is not None
                    else None
                ),
            )
        except Exception:
            logger.warning(
                "Watch terminal hook failed for run %s (non-fatal)",
                run.id,
                exc_info=True,
            )

    # PRD-123 Pattern #1: Produce frozen transition record
    transition = RunTransition(
        run_id=run.id,
        from_state=old_state_value,
        to_state=new_state.value,
        triggered_by=actor_type.value,
        stop_reason=stop_reason,
    )

    logger.info(
        "Run %s transitioned: %s → %s (actor=%s/%s)",
        run.id,
        old_state_value,
        new_state.value,
        actor_type.value,
        actor_id,
    )

    return transition


# ---------------------------------------------------------------------------
# Non-transition event emission
# ---------------------------------------------------------------------------


def emit_event(
    db: Session,
    run_id: Any,
    event_type: EventType,
    actor_type: ActorType,
    actor_id: Optional[str] = None,
    task_id: Any = None,
    payload: Optional[dict] = None,
) -> OrchestrationEvent:
    """
    Emit a non-transition event (budget warning, stall detection, etc.).

    Use this for events that don't change state but need to be recorded
    in the audit trail. For state changes, use transition_task/transition_run.

    Args:
        db: SQLAlchemy session (caller manages transaction).
        run_id: The orchestration run this event belongs to.
        event_type: The type of event to emit.
        actor_type: Who is emitting this event.
        actor_id: Optional identifier of the actor.
        task_id: Optional task this event relates to.
        payload: Optional arbitrary event data.

    Returns:
        The created OrchestrationEvent.
    """
    event = OrchestrationEvent(
        run_id=run_id,
        task_id=task_id,
        event_type=event_type.value,
        actor_type=actor_type.value,
        actor_id=actor_id,
        old_state=None,
        new_state=None,
        payload=payload,
    )
    db.add(event)
    db.flush()

    logger.info(
        "Event emitted: %s for run=%s task=%s (actor=%s/%s)",
        event_type.value,
        run_id,
        task_id,
        actor_type.value,
        actor_id,
    )

    return event


# ---------------------------------------------------------------------------
# Internal helpers — state-to-event-type mapping
# ---------------------------------------------------------------------------


def _task_state_to_event_type(state: TaskState) -> EventType:
    """Map a TaskState to its corresponding EventType for transition events."""
    mapping = {
        TaskState.QUEUED: EventType.TASK_QUEUED,
        TaskState.ASSIGNED: EventType.TASK_ASSIGNED,
        TaskState.RUNNING: EventType.TASK_STARTED,
        TaskState.COMPLETED: EventType.TASK_OUTPUT_SUBMITTED,
        TaskState.VERIFYING: EventType.TASK_VERIFICATION_STARTED,
        TaskState.VERIFIED: EventType.TASK_VERIFICATION_PASSED,
        TaskState.FAILED: EventType.TASK_FAILED,
        TaskState.SKIPPED: EventType.TASK_SKIPPED,
        TaskState.STALLED: EventType.TASK_STALLED,
        TaskState.RETRYING: EventType.TASK_RETRYING,
    }
    return mapping.get(state, EventType.TASK_STARTED)


def _run_state_to_event_type(state: RunState) -> EventType:
    """Map a RunState to its corresponding EventType for transition events."""
    mapping = {
        RunState.PLANNING: EventType.RUN_PLANNING_STARTED,
        RunState.AWAITING_APPROVAL: EventType.RUN_PLAN_READY,
        RunState.RUNNING: EventType.RUN_STARTED,
        RunState.PAUSED: EventType.RUN_PAUSED,
        RunState.REPLANNING: EventType.RUN_REPLANNING,
        RunState.VERIFYING: EventType.RUN_VERIFYING,
        RunState.AWAITING_HUMAN: EventType.RUN_AWAITING_HUMAN,
        RunState.COMPLETED: EventType.RUN_COMPLETED,
        RunState.FAILED: EventType.RUN_FAILED,
        RunState.CANCELLED: EventType.RUN_CANCELLED,
    }
    return mapping.get(state, EventType.RUN_STARTED)
