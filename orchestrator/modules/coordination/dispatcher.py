"""
Mission Dispatcher — PRD-82A/82C Parallel Mission Coordinator
==============================================================

Parallel dispatch of orchestration tasks to roster agents via
execute_with_prompt(). Dispatches up to max_concurrent tasks simultaneously.

Key patterns:
- Optimistic claim: raw SQL UPDATE with version_id check prevents double-dispatch
- Parallel dispatch: dispatch_ready() sends up to max_concurrent tasks per tick
- AgentMatcher selects best agent; no match → immediate fail with NO_AGENT_AVAILABLE
- Board task created before dispatch for kanban visibility
- Agent output stored on task.output; tokens tracked per task and per run

Source: PRD-82A Sections 4.4 (claim pattern), 8 (failure codes), 9 (budget tracking)
        PRD-82C Section US-002 (parallel dispatch)
        PRD-102 Section 6 (dispatcher design)
"""

import logging
from dataclasses import dataclass
import warnings
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import and_, text
from sqlalchemy.orm import Session

from config import Config
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
    def count_active_tasks(db: Session, run_id: UUID) -> int:
        """
        Count tasks in ASSIGNED or RUNNING state for this run.

        Used by dispatch_ready() to determine available dispatch slots.
        """
        return (
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

    @staticmethod
    def has_active_task(db: Session, run_id: UUID) -> bool:
        """
        Check if any task for this run is currently ASSIGNED or RUNNING.

        .. deprecated:: 82C
            Use :meth:`count_active_tasks` with :meth:`dispatch_ready` instead.
        """
        warnings.warn(
            "has_active_task() is deprecated, use count_active_tasks() + dispatch_ready()",
            DeprecationWarning,
            stacklevel=2,
        )
        return MissionDispatcher.count_active_tasks(db, run_id) > 0

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
                  AND state IN ('queued', 'retrying')
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

        .. deprecated:: 82C
            Use :meth:`dispatch_ready` for parallel dispatch.

        Enforces sequential execution: if any task is ASSIGNED or RUNNING,
        returns without dispatching.

        Args:
            db: SQLAlchemy session (caller manages transaction).
            run: The OrchestrationRun to dispatch for.
            agents: Candidate roster agents for the workspace.

        Returns:
            DispatchResult with dispatch outcome.
        """
        warnings.warn(
            "dispatch_next() is deprecated, use dispatch_ready() for parallel dispatch",
            DeprecationWarning,
            stacklevel=2,
        )
        run_id = run.id

        # --- Sequential enforcement (legacy behavior) ---
        if MissionDispatcher.count_active_tasks(db, run_id) > 0:
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
            task = ready_tasks[0]

        return MissionDispatcher._dispatch_single(db, run, task, agents)

    @staticmethod
    def _dispatch_single(
        db: Session,
        run: OrchestrationRun,
        task: OrchestrationTask,
        agents: Sequence[Agent],
    ) -> DispatchResult:
        """
        Dispatch a single task: transition to queued, match agent, claim, emit event.

        Extracted from dispatch_next() so both dispatch_next() and dispatch_ready()
        share identical dispatch logic.

        Args:
            db: SQLAlchemy session (caller manages transaction).
            run: The OrchestrationRun.
            task: The task to dispatch (must be PENDING, QUEUED, or RETRYING).
            agents: Candidate roster agents.

        Returns:
            DispatchResult with dispatch outcome.
        """
        run_id = run.id

        # If task is pending, transition to queued first
        if task.state == TaskState.PENDING.value:
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

        # --- Emit TASK_ASSIGNED event ---
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
    def dispatch_ready(
        db: Session,
        run: OrchestrationRun,
        agents: Sequence[Agent],
    ) -> List[DispatchResult]:
        """
        Dispatch up to (max_concurrent - active_count) ready tasks.

        Parallel dispatch: finds all ready tasks (dependencies met), dispatches
        up to the available slot count. Each task gets its own agent via
        AgentMatcher.

        Args:
            db: SQLAlchemy session (caller manages transaction).
            run: The OrchestrationRun to dispatch for.
            agents: Candidate roster agents for the workspace.

        Returns:
            List of DispatchResult — one per dispatch attempt.
        """
        run_id = run.id
        max_concurrent = run.max_concurrent or 1

        # --- Calculate available slots ---
        active_count = MissionDispatcher.count_active_tasks(db, run_id)
        available_slots = max_concurrent - active_count

        if available_slots <= 0:
            return [
                DispatchResult(
                    dispatched=False,
                    skipped_reason="max_concurrent_reached",
                )
            ]

        # --- Find already-queued or retrying tasks ---
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
            .all()
        )

        # --- Find ready tasks (pending with all deps met) ---
        ready_tasks = DependencyResolver.get_ready_tasks(db, run_id)

        # Combine: actionable first, then ready (deduplicated)
        seen_ids = set()
        candidates: List[OrchestrationTask] = []
        for task in actionable:
            if task.id not in seen_ids:
                seen_ids.add(task.id)
                candidates.append(task)
        for task in ready_tasks:
            if task.id not in seen_ids:
                seen_ids.add(task.id)
                candidates.append(task)

        if not candidates:
            return [
                DispatchResult(
                    dispatched=False,
                    skipped_reason="no_ready_tasks",
                )
            ]

        # --- Dispatch up to available_slots ---
        results: List[DispatchResult] = []
        for task in candidates[:available_slots]:
            result = MissionDispatcher._dispatch_single(db, run, task, agents)
            results.append(result)

        logger.info(
            "dispatch_ready(run=%s): %d candidates, %d slots, %d dispatched",
            run_id,
            len(candidates),
            available_slots,
            sum(1 for r in results if r.dispatched),
        )

        return results if results else [
            DispatchResult(dispatched=False, skipped_reason="no_ready_tasks")
        ]

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
            # Agent error — retry if attempts remain, else fail permanently
            error_msg = result.get("error", "Unknown error")
            max_retries = task.max_retries or Config.COORDINATOR_MAX_TASK_RETRIES
            current_attempt = (task.attempt_number or 0) + 1
            task.attempt_number = current_attempt

            if current_attempt < max_retries:
                # Re-queue for retry
                task.assigned_agent_id = None
                task.failure_detail = error_msg[:2000]
                transition_task(
                    db=db,
                    task=task,
                    new_state=TaskState.QUEUED,
                    actor_type=ActorType.AGENT,
                    actor_id=str(task.assigned_agent_id or "unknown"),
                    reason=f"Agent error, retrying (attempt {current_attempt}/{max_retries}): {error_msg[:200]}",
                )
                logger.warning(
                    "Task %s agent error → re-queued (attempt %d/%d): %s",
                    task.id, current_attempt, max_retries, error_msg[:200],
                )
            else:
                # Retries exhausted
                task.failure_reason_code = FailureReasonCode.AGENT_ERROR.value
                task.failure_detail = error_msg[:2000]
                transition_task(
                    db=db,
                    task=task,
                    new_state=TaskState.FAILED,
                    actor_type=ActorType.AGENT,
                    actor_id=str(task.assigned_agent_id),
                    reason=f"Agent error after {current_attempt} attempts: {error_msg[:200]}",
                )
                logger.warning(
                    "Task %s failed after %d attempts: %s",
                    task.id, current_attempt, error_msg[:200],
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

            # Surface verification feedback from failed attempts
            verification_feedback = task.input_context.get("verification_feedback")
            if verification_feedback:
                parts.append(
                    f"\n## Feedback from Previous Attempt\n"
                    f"Your previous output was rejected. Here is the feedback:\n"
                    f"Reason: {verification_feedback.get('reasoning', 'Unknown')}\n"
                    f"Failed checks: {', '.join(verification_feedback.get('failures', []))}"
                )

        # Inject required output format from verification_criteria
        vc = task.verification_criteria if hasattr(task, 'verification_criteria') else None
        if isinstance(vc, list):
            section_checks = [
                c for c in vc
                if c.get("type") == "required_sections" and isinstance(c.get("value"), list)
            ]
            if section_checks:
                sections = section_checks[0]["value"]
                parts.append(
                    "\n## Required Output Format\n"
                    "Your response MUST include these markdown sections:\n"
                    + "\n".join(f"- {s}" for s in sections)
                    + "\n\nWrite comprehensive, detailed content under each section."
                )

        return "\n".join(parts)
