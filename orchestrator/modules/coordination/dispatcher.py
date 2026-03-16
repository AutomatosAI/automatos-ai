"""
Mission Dispatcher — PRD-82A Sequential Mission Coordinator
============================================================

Sequential dispatch of orchestration tasks to roster agents via
execute_with_prompt(). Enforces one-task-at-a-time execution per mission.

Key patterns:
- Optimistic claim: raw SQL UPDATE with version_id check prevents double-dispatch
- Sequential enforcement: skip dispatch if any task is ASSIGNED or RUNNING
- AgentMatcher selects best agent; no match → immediate fail with NO_AGENT_AVAILABLE
- Board task created before dispatch for kanban visibility
- Agent output stored on task.output; tokens tracked per task and per run

Source: PRD-82A Sections 4.4 (claim pattern), 8 (failure codes), 9 (budget tracking)
        PRD-102 Section 6 (dispatcher design)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from uuid import UUID

from sqlalchemy import and_, text
from sqlalchemy.orm import Session

from core.models.core import Agent
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import (
    ActorType,
    EventType,
    FailureReasonCode,
    TaskState,
)
from modules.coordination.agent_matcher import AgentMatcher, MatchResult
from services.orchestration_board_bridge import create_task_board_task, sync_board_status
from services.orchestration_deps import DependencyResolver
from services.orchestration_state import (
    ConflictError,
    emit_event,
    transition_task,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchResult:
    """Immutable result of a dispatch attempt."""

    dispatched: bool
    task_id: Optional[UUID] = None
    agent_id: Optional[int] = None
    agent_name: Optional[str] = None
    skipped_reason: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# MissionDispatcher
# ---------------------------------------------------------------------------


class MissionDispatcher:
    """
    Sequential task dispatcher for orchestration missions.

    Stateless — all data comes from DB. Caller manages transactions.
    """

    @staticmethod
    def has_active_task(db: Session, run_id: UUID) -> bool:
        """
        Check if any task for this run is currently ASSIGNED or RUNNING.

        Sequential enforcement: only one task at a time per mission.
        """
        active_count = (
            db.query(OrchestrationTask.id)
            .filter(
                and_(
                    OrchestrationTask.run_id == run_id,
                    OrchestrationTask.state.in_([
                        TaskState.ASSIGNED.value,
                        TaskState.RUNNING.value,
                    ]),
                )
            )
            .count()
        )
        return active_count > 0

    @staticmethod
    def claim_task(
        db: Session,
        task: OrchestrationTask,
        agent_id: int,
    ) -> bool:
        """
        Atomically claim a queued task using optimistic locking.

        Uses raw SQL to perform a conditional UPDATE that only succeeds if
        the task is still in 'queued' state with the expected version_id.
        This prevents double-dispatch under concurrent ticks.

        Args:
            db: SQLAlchemy session.
            task: The OrchestrationTask to claim (must be in 'queued' state).
            agent_id: The agent ID to assign.

        Returns:
            True if claim succeeded, False if another instance claimed it.
        """
        result = db.execute(
            text("""
                UPDATE orchestration_tasks
                SET state = :new_state,
                    state_type = 'active',
                    assigned_agent_id = :agent_id,
                    version_id = version_id + 1,
                    updated_at = NOW()
                WHERE id = :task_id
                  AND state = 'queued'
                  AND version_id = :expected_version
            """),
            {
                "new_state": TaskState.ASSIGNED.value,
                "agent_id": agent_id,
                "task_id": str(task.id),
                "expected_version": task.version_id,
            },
        )

        claimed = result.rowcount > 0

        if claimed:
            # Refresh ORM instance to pick up the raw SQL changes
            db.expire(task)
            logger.info(
                "Claimed task %s for agent %d (version %d → %d)",
                task.id,
                agent_id,
                task.version_id - 1 if task.version_id else 0,
                task.version_id,
            )
        else:
            logger.info(
                "Claim failed for task %s (likely already claimed by another instance)",
                task.id,
            )

        return claimed

    @staticmethod
    def dispatch_next(
        db: Session,
        run: OrchestrationRun,
        agents: Sequence[Agent],
    ) -> DispatchResult:
        """
        Find and dispatch the next queued task for a mission run.

        Enforces sequential execution: if any task is ASSIGNED or RUNNING,
        returns without dispatching. Otherwise, finds the first ready task
        (dependencies met), selects an agent, claims the task, and emits
        the TASK_ASSIGNED event.

        This method does NOT call execute_with_prompt() — that is the
        responsibility of the CoordinatorService, which runs the async
        execution after dispatch succeeds.

        Args:
            db: SQLAlchemy session (caller manages transaction).
            run: The OrchestrationRun to dispatch for.
            agents: Candidate roster agents for the workspace.

        Returns:
            DispatchResult with dispatch outcome.
        """
        run_id = run.id

        # --- Sequential enforcement ---
        if MissionDispatcher.has_active_task(db, run_id):
            return DispatchResult(
                dispatched=False,
                skipped_reason="active_task_exists",
            )

        # --- Find already-queued or retrying tasks first ---
        actionable = (
            db.query(OrchestrationTask)
            .filter(
                and_(
                    OrchestrationTask.run_id == run_id,
                    OrchestrationTask.state.in_([
                        TaskState.QUEUED.value,
                        TaskState.RETRYING.value,
                    ]),
                )
            )
            .order_by(OrchestrationTask.sequence_number)
            .first()
        )

        if actionable:
            task = actionable
        else:
            # --- Find ready tasks (pending with all deps met) ---
            ready_tasks = DependencyResolver.get_ready_tasks(db, run_id)
            if not ready_tasks:
                return DispatchResult(
                    dispatched=False,
                    skipped_reason="no_ready_tasks",
                )

            # Pick first by sequence_number (already ordered by get_ready_tasks)
            task = ready_tasks[0]

            # --- Transition pending → queued ---
            try:
                transition_task(
                    db=db,
                    task=task,
                    new_state=TaskState.QUEUED,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="dispatcher",
                )
            except ConflictError:
                logger.warning("Conflict transitioning task %s to queued", task.id)
                return DispatchResult(
                    dispatched=False,
                    skipped_reason="conflict_on_queue",
                )

        # --- Agent matching ---
        match_result: Optional[MatchResult] = AgentMatcher.match(
            db=db,
            task=task,
            agents=agents,
            task_spec={
                "agent_role": task.agent_role,
                "required_tools": (
                    task.input_context.get("required_tools", [])
                    if isinstance(task.input_context, dict)
                    else []
                ),
            },
        )

        if match_result is None:
            # No agent available — fail task immediately
            task.failure_reason_code = FailureReasonCode.NO_AGENT_AVAILABLE.value
            task.failure_detail = (
                f"No roster agent matched for role '{task.agent_role}'. "
                "Create an agent with the required tools/skills or remove the requirement."
            )
            transition_task(
                db=db,
                task=task,
                new_state=TaskState.FAILED,
                actor_type=ActorType.COORDINATOR,
                actor_id="dispatcher",
                reason="No agent available",
            )
            sync_board_status(db, task)
            logger.warning(
                "Task %s failed: no agent available for role '%s'",
                task.id,
                task.agent_role,
            )
            return DispatchResult(
                dispatched=False,
                error=f"no_agent_available for role '{task.agent_role}'",
            )

        # --- Optimistic claim (queued → assigned) ---
        claimed = MissionDispatcher.claim_task(db, task, match_result.agent_id)
        if not claimed:
            return DispatchResult(
                dispatched=False,
                skipped_reason="claim_failed",
            )

        # --- Emit TASK_ASSIGNED event (claim bypasses transition_task) ---
        emit_event(
            db=db,
            run_id=run_id,
            event_type=EventType.TASK_ASSIGNED,
            actor_type=ActorType.COORDINATOR,
            actor_id="dispatcher",
            task_id=task.id,
            payload={
                "agent_id": match_result.agent_id,
                "agent_name": match_result.agent_name,
                "match_score": match_result.total_score,
            },
        )

        # --- Create board task for kanban visibility ---
        try:
            create_task_board_task(db, run, task)
        except Exception:
            logger.warning(
                "Failed to create board task for orchestration task %s",
                task.id,
                exc_info=True,
            )

        # Sync board status to reflect assigned state
        sync_board_status(db, task)

        logger.info(
            "Dispatched task %s (seq=%d) to agent %s (id=%d, score=%.3f)",
            task.id,
            task.sequence_number,
            match_result.agent_name,
            match_result.agent_id,
            match_result.total_score,
        )

        return DispatchResult(
            dispatched=True,
            task_id=task.id,
            agent_id=match_result.agent_id,
            agent_name=match_result.agent_name,
        )

    @staticmethod
    def record_task_completion(
        db: Session,
        task: OrchestrationTask,
        result: Dict[str, Any],
    ) -> None:
        """
        Record the output from execute_with_prompt() on a task.

        Called by the CoordinatorService after async execution completes.
        Transitions the task from RUNNING → COMPLETED and records output + tokens.

        Args:
            db: SQLAlchemy session (caller manages transaction).
            task: The OrchestrationTask that completed.
            result: The dict returned by execute_with_prompt().
        """
        status = result.get("status", "error")

        if status == "success":
            # Store output
            task.output = result.get("result", "")
            task.output_metadata = {
                "model": result.get("execution", {}).get("model"),
                "provider": result.get("execution", {}).get("provider"),
                "execution_time": result.get("execution", {}).get("time"),
                "attempt": result.get("execution", {}).get("attempt"),
                "tool_iterations": result.get("execution", {}).get("tool_iterations"),
            }

            # Token tracking (PRD-82A Section 9)
            tokens = result.get("execution", {}).get("tokens_used", 0)
            task.tokens_used = (task.tokens_used or 0) + tokens

            transition_task(
                db=db,
                task=task,
                new_state=TaskState.COMPLETED,
                actor_type=ActorType.AGENT,
                actor_id=str(task.assigned_agent_id),
            )

            logger.info(
                "Task %s completed: %d tokens used",
                task.id,
                tokens,
            )
        else:
            # Agent error
            error_msg = result.get("error", "Unknown error")
            task.failure_reason_code = FailureReasonCode.AGENT_ERROR.value
            task.failure_detail = error_msg[:2000]  # Truncate to prevent oversized rows

            transition_task(
                db=db,
                task=task,
                new_state=TaskState.FAILED,
                actor_type=ActorType.AGENT,
                actor_id=str(task.assigned_agent_id),
                reason=f"Agent error: {error_msg[:200]}",
            )

            logger.warning(
                "Task %s failed with agent error: %s",
                task.id,
                error_msg[:200],
            )

        sync_board_status(db, task)

    @staticmethod
    def record_task_running(
        db: Session,
        task: OrchestrationTask,
    ) -> None:
        """
        Transition task from ASSIGNED → RUNNING when execute_with_prompt() is called.

        Args:
            db: SQLAlchemy session (caller manages transaction).
            task: The OrchestrationTask being executed.
        """
        transition_task(
            db=db,
            task=task,
            new_state=TaskState.RUNNING,
            actor_type=ActorType.COORDINATOR,
            actor_id="dispatcher",
        )
        sync_board_status(db, task)

    @staticmethod
    def build_task_prompt(task: OrchestrationTask) -> str:
        """
        Build the user prompt for execute_with_prompt() from task data.

        Includes task title, description, and any input context from
        upstream task outputs or retry feedback.
        """
        parts = [f"# Task: {task.title}"]

        if task.description:
            parts.append(f"\n{task.description}")

        # Include input context (upstream outputs, retry feedback, etc.)
        if isinstance(task.input_context, dict):
            upstream_outputs = task.input_context.get("upstream_outputs")
            if upstream_outputs:
                parts.append("\n## Previous Task Outputs")
                for output in upstream_outputs:
                    parts.append(
                        f"\n### {output.get('title', 'Previous Task')}\n"
                        f"{output.get('output', '')}"
                    )

            retry_feedback = task.input_context.get("retry_feedback")
            if retry_feedback:
                parts.append(
                    f"\n## Feedback from Previous Attempt\n"
                    f"Your previous output was rejected. Here is the feedback:\n"
                    f"{retry_feedback}"
                )

            verification_criteria = task.input_context.get("verification_criteria_hint")
            if verification_criteria:
                parts.append(
                    f"\n## Quality Requirements\n{verification_criteria}"
                )

        return "\n".join(parts)
