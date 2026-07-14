"""
Mission Reconciler — PRD-82A Sequential Mission Coordinator
============================================================

Reconciles active missions on every coordinator tick:
- Verification: COMPLETED tasks → verify via VerificationService → VERIFIED/RETRYING/FAILED
- Stall detection: ASSIGNED tasks >60s, RUNNING tasks >300s → mark stalled
- Stalled recovery: stalled → assigned for re-dispatch
- Completion check: all tasks terminal & verified → advance run to verifying
- Failure check: any task failed with retries exhausted → fail the mission

Stateless — all data comes from DB. Caller manages transactions.

Source: PRD-82A Sections 4.2 (transitions), 8 (failure codes), 11 (retry guardrails)
        PRD-102 Section 4.3-4.5 (reconciler design)
        PRD-103 Sections 3-5 (verification service)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.orm import Session

from config import Config
from core.database.database import end_open_transaction
from core.models.core import Agent
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import (
    ActorType,
    EventType,
    FailureReasonCode,
    RunState,
    TaskState,
    TERMINAL_TASK_STATES,
    DONE_TASK_STATES,
)
from modules.coordination.verification import (
    VERDICT_FAIL,
    VERDICT_PARTIAL,
    VERDICT_PASS,
    VerificationResult,
    VerificationService,
)
from services.orchestration_board_bridge import sync_board_status
from services.orchestration_state import (
    ConflictError,
    emit_event,
    transition_run,
    transition_task,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory capture helpers (PRD-131d Phase 2)
# ---------------------------------------------------------------------------


async def _store_task_failure_safe(db: Session, task: OrchestrationTask) -> None:
    """Store a permanently-failed task into memory. Never raises."""
    try:
        from core.services.mission_memory_service import MissionMemoryService
        await MissionMemoryService(db=db).store_task_failure(task=task)
    except Exception:
        logger.warning(
            "[Reconciler] Task failure memory capture skipped for task %s",
            task.id,
            exc_info=True,
        )


async def _store_retry_recovery_safe(db: Session, task: OrchestrationTask) -> None:
    """Store a successful recovery after retry. Never raises. Skips first-attempt wins."""
    if (task.attempt_number or 0) <= 0:
        return
    try:
        from core.services.mission_memory_service import MissionMemoryService
        await MissionMemoryService(db=db).store_retry_recovery(task=task)
    except Exception:
        logger.warning(
            "[Reconciler] Retry recovery memory capture skipped for task %s",
            task.id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconcileResult:
    """Immutable result of a reconciliation pass for one mission run."""

    run_id: UUID
    stalls_detected: int = 0
    stalls_recovered: int = 0
    tasks_failed: int = 0
    tasks_verified: int = 0
    tasks_verification_failed: int = 0
    run_advanced: bool = False
    run_new_state: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# MissionReconciler
# ---------------------------------------------------------------------------


class MissionReconciler:
    """
    Reconciles orchestration missions by detecting stalls, checking
    completions, and advancing run state.

    Stateless — all data comes from DB. Caller manages transactions.
    """

    @staticmethod
    async def reconcile(db: Session, run: OrchestrationRun) -> ReconcileResult:
        """
        Run a full reconciliation pass for a single mission run.

        Steps:
        1. Verify completed tasks (COMPLETED → VERIFYING → VERIFIED/RETRYING/FAILED)
        2. Detect stalls (ASSIGNED >60s, RUNNING >300s)
        3. Check if all tasks terminal → advance run or fail run

        Args:
            db: SQLAlchemy session (caller manages transaction).
            run: The OrchestrationRun to reconcile.

        Returns:
            ReconcileResult with counts and state changes.
        """
        run_id = run.id
        stalls_detected = 0
        stalls_recovered = 0
        tasks_failed = 0
        tasks_verified = 0
        tasks_verification_failed = 0

        try:
            # --- Verification phase (completed → verifying → verdict) ---
            verify_counts = await MissionReconciler._verify_completed_tasks(db, run)
            tasks_verified = verify_counts[0]
            tasks_verification_failed = verify_counts[1]

            # --- Stall detection + recovery ---
            stall_counts = await MissionReconciler._detect_and_recover_stalls(db, run_id)
            stalls_detected = stall_counts[0]
            stalls_recovered = stall_counts[1]

            # --- Escalate repeated stalls ---
            if stalls_detected > 0:
                try:
                    from services.escalation_service import escalate_stalled_task

                    stalled_tasks = (
                        db.query(OrchestrationTask)
                        .filter(
                            OrchestrationTask.run_id == run_id,
                            OrchestrationTask.state == TaskState.STALLED.value,
                        )
                        .all()
                    )
                    for st in stalled_tasks:
                        escalate_stalled_task(
                            db=db,
                            workspace_id=run.workspace_id,
                            task_id=st.id,
                            task_title=st.title or "Untitled",
                            stall_count=st.attempt_number or 1,
                            agent_name=str(st.assigned_agent_id or "Unknown"),
                        )
                except Exception as e:
                    logger.warning("Escalation service error (non-fatal): %s", e)

            # --- Check task terminal states ---
            all_tasks = (
                db.query(OrchestrationTask)
                .filter(OrchestrationTask.run_id == run_id)
                .all()
            )

            if not all_tasks:
                return ReconcileResult(run_id=run_id)

            done_tasks = [
                t for t in all_tasks
                if TaskState(t.state) in DONE_TASK_STATES
            ]
            all_terminal = len(done_tasks) == len(all_tasks)

            # Count failures (tasks that exhausted retries)
            failed_tasks = [
                t for t in all_tasks
                if TaskState(t.state) == TaskState.FAILED
            ]
            tasks_failed = len(failed_tasks)

            # --- Advance run state if appropriate ---
            if all_terminal:
                return MissionReconciler._advance_run_on_completion(
                    db=db,
                    run=run,
                    all_tasks=all_tasks,
                    failed_tasks=failed_tasks,
                    stalls_detected=stalls_detected,
                    stalls_recovered=stalls_recovered,
                    tasks_failed=tasks_failed,
                    tasks_verified=tasks_verified,
                    tasks_verification_failed=tasks_verification_failed,
                )

            # --- Check for fatal failure (task failed, retries exhausted) ---
            if failed_tasks:
                fatal = MissionReconciler._check_fatal_failure(
                    db=db,
                    run=run,
                    failed_tasks=failed_tasks,
                )
                if fatal:
                    return ReconcileResult(
                        run_id=run_id,
                        stalls_detected=stalls_detected,
                        stalls_recovered=stalls_recovered,
                        tasks_failed=tasks_failed,
                        tasks_verified=tasks_verified,
                        tasks_verification_failed=tasks_verification_failed,
                        run_advanced=True,
                        run_new_state=RunState.FAILED.value,
                    )

            return ReconcileResult(
                run_id=run_id,
                stalls_detected=stalls_detected,
                stalls_recovered=stalls_recovered,
                tasks_failed=tasks_failed,
                tasks_verified=tasks_verified,
                tasks_verification_failed=tasks_verification_failed,
            )

        except Exception:
            logger.error(
                "Error reconciling run %s", run_id, exc_info=True,
            )
            return ReconcileResult(
                run_id=run_id,
                stalls_detected=stalls_detected,
                stalls_recovered=stalls_recovered,
                tasks_failed=tasks_failed,
                tasks_verified=tasks_verified,
                tasks_verification_failed=tasks_verification_failed,
                error="reconciliation_error",
            )

    # -----------------------------------------------------------------------
    # Verification of completed tasks
    # -----------------------------------------------------------------------

    @staticmethod
    async def _verify_completed_tasks(
        db: Session,
        run: OrchestrationRun,
    ) -> tuple:
        """
        Find tasks in COMPLETED state, verify them via VerificationService.

        Flow per task:
          COMPLETED → VERIFYING → verify_task() → VERIFIED / RETRYING / FAILED

        Returns (tasks_verified, tasks_verification_failed) counts.
        """
        run_id = run.id
        tasks_verified = 0
        tasks_verification_failed = 0

        completed_tasks: List[OrchestrationTask] = (
            db.query(OrchestrationTask)
            .filter(
                and_(
                    OrchestrationTask.run_id == run_id,
                    OrchestrationTask.state == TaskState.COMPLETED.value,
                )
            )
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        if not completed_tasks:
            return (0, 0)

        # Check if verification should be skipped (benchmark/testing mode)
        skip_verification = (run.config or {}).get("skip_verification", False)

        if skip_verification:
            for task in completed_tasks:
                # Must go COMPLETED → VERIFYING → VERIFIED (state machine requires it)
                transition_task(
                    db=db,
                    task=task,
                    new_state=TaskState.VERIFYING,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="reconciler",
                    reason="Verification skipped — transitioning through VERIFYING",
                )
                await MissionReconciler._apply_verdict_pass(db, task)
                emit_event(
                    db=db,
                    run_id=run_id,
                    event_type=EventType.TASK_VERIFICATION_COMPLETED,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="reconciler",
                    task_id=task.id,
                    payload={
                        "verdict": VERDICT_PASS,
                        "scores": {},
                        "reasoning": "Verification skipped (skip_verification=True)",
                        "confidence": 1.0,
                        "deterministic_passed": True,
                        "tokens_used": 0,
                    },
                )
                db.flush()
            logger.info(
                "Skipped verification for %d tasks in run %s (skip_verification=True)",
                len(completed_tasks),
                run_id,
            )
            return (len(completed_tasks), 0)

        verification_service = VerificationService()

        for task in completed_tasks:
            # Transition COMPLETED → VERIFYING
            try:
                transition_task(
                    db=db,
                    task=task,
                    new_state=TaskState.VERIFYING,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="reconciler",
                    reason="Starting verification",
                )
                sync_board_status(db, task)
            except ConflictError:
                logger.warning(
                    "Conflict transitioning task %s to verifying", task.id,
                )
                continue

            # Resolve executor model from assigned agent
            executor_model = MissionReconciler._get_executor_model(db, task)

            # Parse verification criteria from task spec
            criteria = task.verification_criteria if task.verification_criteria else None

            # Strip required_sections from synthesis tasks — exact heading
            # matching causes PARTIAL downgrades that burn tokens on retries.
            # The LLM judge already evaluates content quality.
            if criteria and task.task_type == "synthesis":
                criteria = [
                    c for c in criteria
                    if c.get("type") != "required_sections"
                ]
                if not criteria:
                    criteria = None

            # Capture every scalar verify_task needs BEFORE committing. After
            # end_open_transaction() the session is expired (expire_on_commit
            # default), so touching task.title/description/output/id would
            # re-SELECT and re-open a transaction — exactly what we must avoid
            # across the long LLM await below.
            t_title = task.title
            t_desc = task.description or ""
            t_output = task.output or ""
            t_id = task.id

            # End the read transaction so the connection sits idle (not
            # idle-in-transaction) for the whole verify LLM call. PRD-135 /
            # W1-S9: the agents SELECT in _get_executor_model held across this
            # await was the documented 9-hour idle-in-transaction leak. The
            # pending VERIFYING transition is committed here, so verification
            # becomes durable per task rather than at end-of-tick.
            end_open_transaction(db)

            # Run verification
            try:
                result: VerificationResult = await verification_service.verify_task(
                    task_title=t_title,
                    task_description=t_desc,
                    output=t_output,
                    verification_criteria=criteria,
                    executor_model=executor_model,
                    run_id=run_id,
                    task_id=t_id,
                )
            except Exception:
                logger.error(
                    "Verification service error for task %s", task.id,
                    exc_info=True,
                )
                # Treat verification error as partial (escalate)
                result = VerificationResult(
                    verdict=VERDICT_PARTIAL,
                    reasoning="Verification service raised an exception",
                )

            # Emit verification result event with scores
            emit_event(
                db=db,
                run_id=run_id,
                event_type=EventType.TASK_VERIFICATION_COMPLETED,
                actor_type=ActorType.COORDINATOR,
                actor_id="reconciler",
                task_id=task.id,
                payload={
                    "verdict": result.verdict,
                    "scores": result.scores,
                    "reasoning": result.reasoning,
                    "confidence": result.confidence,
                    "deterministic_passed": result.deterministic_passed,
                    "tokens_used": result.tokens_used,
                },
            )

            # Update token tracking
            if result.tokens_used > 0:
                task.tokens_used = (task.tokens_used or 0) + result.tokens_used
                run.tokens_used = (run.tokens_used or 0) + result.tokens_used

                # Check budget warning
                if (
                    run.token_budget_estimate
                    and run.tokens_used > run.token_budget_estimate * 1.5
                ):
                    emit_event(
                        db=db,
                        run_id=run_id,
                        event_type=EventType.BUDGET_WARNING,
                        actor_type=ActorType.COORDINATOR,
                        actor_id="reconciler",
                        payload={
                            "tokens_used": run.tokens_used,
                            "token_budget_estimate": run.token_budget_estimate,
                            "ratio": round(run.tokens_used / run.token_budget_estimate, 2),
                        },
                    )

            # Apply the verdict (PRD-200 S1 — the judge now gates once).
            # A FAIL verdict requeues the task once with the verifier's
            # feedback so the agent can revise instead of the empty/wrong
            # output flowing straight into synthesis; PARTIAL stays advisory
            # (the retry-storm scar tissue). The routing lives in the pure,
            # unit-tested _apply_verdict.
            requeued = await MissionReconciler._apply_verdict(db, task, result)
            if requeued:
                tasks_verification_failed += 1
            else:
                tasks_verified += 1

            db.flush()

        return (tasks_verified, tasks_verification_failed)

    @staticmethod
    async def _apply_verdict(
        db: Session,
        task: OrchestrationTask,
        result: VerificationResult,
    ) -> bool:
        """Route a verification verdict (PRD-200 S1).

        Returns True if the task was requeued (RETRYING — awaiting another
        execution), False if it was passed through to VERIFIED.

        - FAIL, with verification-requeue budget remaining → requeue once
          with the verifier's feedback (via _apply_verdict_fail) so the agent
          revises. Empty output (the judge's hard FAIL) is the highest-value
          catch — it now gets one revision instead of flowing into synthesis.
        - FAIL at the requeue cap, PARTIAL, or PASS → VERIFIED. A non-PASS
          verdict has its feedback annotated in output_metadata for downstream
          consumers (advisory — the retreat deliberately kept for PARTIAL and
          for a FAIL that has already had its one revision).
        """
        if result.verdict == VERDICT_FAIL and MissionReconciler._apply_verdict_fail(
            db, task, result
        ):
            return True

        if result.verdict != VERDICT_PASS:
            # Store review feedback on the task so downstream consumers
            # (synthesis tasks, reports, humans) can see it.
            task.output_metadata = {
                **(task.output_metadata or {}),
                "review_feedback": {
                    "verdict": result.verdict,
                    "reasoning": result.reasoning,
                    "scores": result.scores,
                    "suggestions": result.suggestions or result.deterministic_failures,
                },
            }
            logger.info(
                "Task %s verification %s — feedback stored, proceeding (advisory)",
                task.id,
                result.verdict.upper(),
            )

        await MissionReconciler._apply_verdict_pass(db, task)
        return False

    @staticmethod
    def _get_executor_model(db: Session, task: OrchestrationTask) -> Optional[str]:
        """Get the model used by the task's assigned agent."""
        if not task.assigned_agent_id:
            return None
        agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
        if not agent or not agent.model_config:
            return None
        return agent.model_config.get("model_id")

    @staticmethod
    async def _apply_verdict_pass(
        db: Session,
        task: OrchestrationTask,
    ) -> None:
        """Handle PASS verdict: transition task to VERIFIED."""
        try:
            transition_task(
                db=db,
                task=task,
                new_state=TaskState.VERIFIED,
                actor_type=ActorType.COORDINATOR,
                actor_id="reconciler",
                reason="Verification passed",
            )
            sync_board_status(db, task)
            logger.info(
                "Task %s verified (pass)", task.id,
            )
            # PRD-131d Phase 2: capture recoveries (passed after >=1 retry)
            await _store_retry_recovery_safe(db, task)
        except ConflictError:
            logger.warning("Conflict transitioning task %s to verified", task.id)

    @staticmethod
    def _apply_verdict_fail(
        db: Session,
        task: OrchestrationTask,
        result: VerificationResult,
    ) -> bool:
        """Requeue a task once on a FAIL verdict with the verifier's feedback
        (PRD-200 S1; Mission Zero P3).

        The task is re-queued for revision (→ RETRYING) with ``previous_output``
        and ``verification_feedback`` stashed into ``input_context`` — the keys
        ``MissionDispatcher.build_task_prompt`` reads back to build a revision
        prompt so the agent fixes the specific failure instead of repeating it.

        Capped at ``COORDINATOR_MAX_VERIFICATION_REQUEUES`` verification-driven
        requeues — a budget SEPARATE from agent-error retries (``attempt_number``
        / ``COORDINATOR_MAX_TASK_RETRIES``) and from the LLM-judge's own
        malformed-response retry count. Tracked in ``input_context`` (no new
        column). The counter is deliberately independent of ``attempt_number``
        so an agent-retried task still gets its one verification revision.

        Returns True if the task was requeued; False if the requeue budget is
        spent or the transition conflicts. On False the caller falls back to
        advisory VERIFIED — a FAIL verdict never fails the mission on its own,
        which is the retreat this gate re-opens only one notch.
        """
        requeues = (task.input_context or {}).get("verification_requeues", 0)
        if requeues >= Config.COORDINATOR_MAX_VERIFICATION_REQUEUES:
            return False

        attempt = task.attempt_number or 0
        # Stash previous output so the retry revises instead of rewriting from
        # scratch — saves ~80% of tokens per retry.
        previous_output = task.output or ""
        task.failure_reason_code = FailureReasonCode.VERIFICATION_FAIL.value
        task.attempt_number = attempt + 1

        # Immutable replace so SQLAlchemy detects the JSONB mutation.
        task.input_context = {
            **(task.input_context or {}),
            "previous_output": previous_output,
            "verification_requeues": requeues + 1,
            "verification_feedback": {
                "attempt": attempt + 1,
                "reasoning": result.reasoning,
                "scores": result.scores,
                "failures": result.deterministic_failures,
            },
        }

        try:
            transition_task(
                db=db,
                task=task,
                new_state=TaskState.RETRYING,
                actor_type=ActorType.COORDINATOR,
                actor_id="reconciler",
                reason=f"Verification failed, requeuing once for revision: {result.reasoning}",
            )
            sync_board_status(db, task)
            logger.info(
                "Task %s verification FAIL → requeued for revision (requeue %d/%d): %s",
                task.id,
                requeues + 1,
                Config.COORDINATOR_MAX_VERIFICATION_REQUEUES,
                result.reasoning,
            )
            return True
        except ConflictError:
            logger.warning(
                "Conflict requeuing task %s after verification fail", task.id
            )
            return False

    # -----------------------------------------------------------------------
    # Stall detection and recovery
    # -----------------------------------------------------------------------

    @staticmethod
    async def _detect_and_recover_stalls(
        db: Session,
        run_id: UUID,
    ) -> tuple:
        """
        Detect stalled tasks and recover them by transitioning back to ASSIGNED.

        Returns (stalls_detected, stalls_recovered) counts.
        """
        now = datetime.now(timezone.utc)
        assigned_threshold = Config.COORDINATOR_ASSIGNED_STALL_THRESHOLD_SECONDS
        running_threshold = Config.COORDINATOR_RUNNING_STALL_THRESHOLD_SECONDS

        stalls_detected = 0
        stalls_recovered = 0

        # Find ASSIGNED or RUNNING tasks for this run
        active_tasks: List[OrchestrationTask] = (
            db.query(OrchestrationTask)
            .filter(
                and_(
                    OrchestrationTask.run_id == run_id,
                    OrchestrationTask.state.in_([
                        TaskState.ASSIGNED.value,
                        TaskState.RUNNING.value,
                    ]),
                )
            )
            .all()
        )

        for task in active_tasks:
            task_state = TaskState(task.state)
            threshold = (
                assigned_threshold
                if task_state == TaskState.ASSIGNED
                else running_threshold
            )

            # Use updated_at as the last activity timestamp
            last_activity = task.updated_at or task.created_at
            if last_activity is None:
                continue

            # Ensure timezone-aware comparison
            if last_activity.tzinfo is None:
                last_activity = last_activity.replace(tzinfo=timezone.utc)

            elapsed = (now - last_activity).total_seconds()

            if elapsed > threshold:
                stalls_detected += 1
                logger.warning(
                    "Stall detected: task %s in state %s for %.0fs "
                    "(threshold: %ds, run: %s)",
                    task.id,
                    task_state.value,
                    elapsed,
                    threshold,
                    run_id,
                )

                # Transition to STALLED
                try:
                    task.failure_reason_code = FailureReasonCode.AGENT_TIMEOUT.value
                    transition_task(
                        db=db,
                        task=task,
                        new_state=TaskState.STALLED,
                        actor_type=ActorType.SCHEDULER,
                        actor_id="reconciler",
                        reason=f"Stall detected: {task_state.value} for {elapsed:.0f}s",
                    )
                    sync_board_status(db, task)

                    # Emit stall event with details
                    emit_event(
                        db=db,
                        run_id=run_id,
                        event_type=EventType.STALL_DETECTED,
                        actor_type=ActorType.SCHEDULER,
                        actor_id="reconciler",
                        task_id=task.id,
                        payload={
                            "previous_state": task_state.value,
                            "elapsed_seconds": round(elapsed),
                            "threshold_seconds": threshold,
                        },
                    )
                except ConflictError:
                    logger.warning(
                        "Conflict marking task %s as stalled (concurrent modification)",
                        task.id,
                    )
                    continue

                # Recover: transition STALLED → QUEUED for re-dispatch (not ASSIGNED,
        # which would block the dispatcher's has_active_task check)
                recovered = await MissionReconciler._recover_stalled_task(db, task)
                if recovered:
                    stalls_recovered += 1

        return (stalls_detected, stalls_recovered)

    @staticmethod
    async def _recover_stalled_task(db: Session, task: OrchestrationTask) -> bool:
        """
        Recover a stalled task by transitioning back to QUEUED for re-dispatch.

        Increments the attempt_number. If max_retries exhausted, transitions
        to FAILED instead.

        Returns True if recovery succeeded, False if task was failed.
        """
        max_retries = task.max_retries or Config.COORDINATOR_MAX_TASK_RETRIES

        if (task.attempt_number or 0) >= max_retries:
            # Retries exhausted — fail the task
            task.failure_reason_code = FailureReasonCode.MAX_RETRIES_EXHAUSTED.value
            task.failure_detail = (
                f"Task stalled {task.attempt_number} times, "
                f"exceeding max_retries={max_retries}"
            )
            try:
                transition_task(
                    db=db,
                    task=task,
                    new_state=TaskState.FAILED,
                    actor_type=ActorType.SCHEDULER,
                    actor_id="reconciler",
                    reason="Max retries exhausted after stall",
                )
                sync_board_status(db, task)
                # PRD-131d Phase 2: capture stall-fatal failure
                await _store_task_failure_safe(db, task)
            except ConflictError:
                logger.warning(
                    "Conflict failing stalled task %s", task.id,
                )
            return False

        # Increment attempt, clear agent, and re-queue for dispatch
        task.attempt_number = (task.attempt_number or 0) + 1
        task.assigned_agent_id = None

        try:
            transition_task(
                db=db,
                task=task,
                new_state=TaskState.QUEUED,
                actor_type=ActorType.SCHEDULER,
                actor_id="reconciler",
                reason=f"Stall recovery (attempt {task.attempt_number})",
            )
            sync_board_status(db, task)
            logger.info(
                "Recovered stalled task %s → queued (attempt %d/%d)",
                task.id,
                task.attempt_number,
                max_retries,
            )
            return True
        except ConflictError:
            logger.warning(
                "Conflict recovering stalled task %s", task.id,
            )
            return False

    # -----------------------------------------------------------------------
    # Run completion / failure advancement
    # -----------------------------------------------------------------------

    @staticmethod
    def _advance_run_on_completion(
        db: Session,
        run: OrchestrationRun,
        all_tasks: List[OrchestrationTask],
        failed_tasks: List[OrchestrationTask],
        stalls_detected: int,
        stalls_recovered: int,
        tasks_failed: int,
        tasks_verified: int = 0,
        tasks_verification_failed: int = 0,
    ) -> ReconcileResult:
        """
        When all tasks are terminal, advance the run to verifying or failed.

        All verified → run transitions to verifying.
        Any failed → run transitions to failed.
        """
        run_id = run.id
        result_kwargs = dict(
            run_id=run_id,
            stalls_detected=stalls_detected,
            stalls_recovered=stalls_recovered,
            tasks_failed=tasks_failed,
            tasks_verified=tasks_verified,
            tasks_verification_failed=tasks_verification_failed,
        )

        verified_tasks = [
            t for t in all_tasks
            if TaskState(t.state) == TaskState.VERIFIED
        ]

        if len(verified_tasks) == len(all_tasks):
            # All tasks verified — advance run to verifying
            try:
                transition_run(
                    db=db,
                    run=run,
                    new_state=RunState.VERIFYING,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="reconciler",
                    reason="All tasks verified",
                )
                logger.info(
                    "Run %s → verifying (all %d tasks verified)",
                    run_id,
                    len(all_tasks),
                )
                return ReconcileResult(
                    **result_kwargs,
                    run_advanced=True,
                    run_new_state=RunState.VERIFYING.value,
                )
            except ConflictError:
                logger.warning("Conflict advancing run %s to verifying", run_id)

        elif failed_tasks:
            # Some tasks failed — fail the run
            failed_titles = [t.title for t in failed_tasks[:5]]
            try:
                transition_run(
                    db=db,
                    run=run,
                    new_state=RunState.FAILED,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="reconciler",
                    reason=f"Tasks failed: {', '.join(failed_titles)}",
                    stop_reason="dependency_failed",
                    stop_detail=f"Tasks failed: {', '.join(failed_titles)}",
                )
                logger.info(
                    "Run %s → failed (%d of %d tasks failed)",
                    run_id,
                    len(failed_tasks),
                    len(all_tasks),
                )
                return ReconcileResult(
                    **result_kwargs,
                    run_advanced=True,
                    run_new_state=RunState.FAILED.value,
                )
            except ConflictError:
                logger.warning("Conflict advancing run %s to failed", run_id)

        else:
            # Mix of verified and skipped — still fail (skipped means upstream failed)
            skipped_tasks = [
                t for t in all_tasks
                if TaskState(t.state) == TaskState.SKIPPED
            ]
            if skipped_tasks:
                try:
                    transition_run(
                        db=db,
                        run=run,
                        new_state=RunState.FAILED,
                        actor_type=ActorType.COORDINATOR,
                        actor_id="reconciler",
                        reason=f"{len(skipped_tasks)} tasks skipped due to dependency failure",
                        stop_reason="dependency_failed",
                        stop_detail=f"{len(skipped_tasks)} tasks skipped due to upstream failure",
                    )
                    return ReconcileResult(
                        **result_kwargs,
                        run_advanced=True,
                        run_new_state=RunState.FAILED.value,
                    )
                except ConflictError:
                    logger.warning("Conflict advancing run %s to failed", run_id)

        return ReconcileResult(**result_kwargs)

    @staticmethod
    def _check_fatal_failure(
        db: Session,
        run: OrchestrationRun,
        failed_tasks: List[OrchestrationTask],
    ) -> bool:
        """
        Check if any failed task should cause the entire run to fail.

        A task failure is fatal when:
        - failure_reason_code is max_retries_exhausted, no_agent_available,
          or dependency_failed
        - attempt_number >= max_retries

        Returns True if the run was transitioned to failed.
        """
        fatal_codes = {
            FailureReasonCode.MAX_RETRIES_EXHAUSTED.value,
            FailureReasonCode.NO_AGENT_AVAILABLE.value,
            FailureReasonCode.DEPENDENCY_FAILED.value,
        }

        for task in failed_tasks:
            is_fatal = (
                task.failure_reason_code in fatal_codes
                or (task.attempt_number or 0) >= (task.max_retries or Config.COORDINATOR_MAX_TASK_RETRIES)
            )

            if is_fatal:
                # Skip all remaining pending/queued tasks
                MissionReconciler._skip_remaining_tasks(
                    db=db,
                    run_id=run.id,
                    reason=f"Upstream task '{task.title}' failed: {task.failure_reason_code}",
                )

                try:
                    _stop = "max_retries" if "retries" in str(task.failure_reason_code).lower() else "dependency_failed"
                    transition_run(
                        db=db,
                        run=run,
                        new_state=RunState.FAILED,
                        actor_type=ActorType.COORDINATOR,
                        actor_id="reconciler",
                        reason=(
                            f"Task '{task.title}' failed fatally: "
                            f"{task.failure_reason_code}"
                        ),
                        stop_reason=_stop,
                        stop_detail=f"Task '{task.title}' failed: {task.failure_reason_code}",
                    )
                    emit_event(
                        db=db,
                        run_id=run.id,
                        event_type=EventType.RUN_FAILED,
                        actor_type=ActorType.COORDINATOR,
                        actor_id="reconciler",
                        payload={
                            "fatal_task_id": str(task.id),
                            "fatal_task_title": task.title,
                            "failure_reason": task.failure_reason_code,
                        },
                    )
                    VerificationService.clear_cache(run.id)
                    logger.info(
                        "Run %s failed due to fatal task failure: %s (%s)",
                        run.id,
                        task.title,
                        task.failure_reason_code,
                    )
                    return True
                except ConflictError:
                    logger.warning("Conflict failing run %s", run.id)
                    return False

        return False

    @staticmethod
    def _skip_remaining_tasks(
        db: Session,
        run_id: UUID,
        reason: str,
    ) -> int:
        """
        Skip all pending and queued tasks for a run (dependency failed or cancelled).

        Returns the number of tasks skipped.
        """
        skippable_tasks: List[OrchestrationTask] = (
            db.query(OrchestrationTask)
            .filter(
                and_(
                    OrchestrationTask.run_id == run_id,
                    OrchestrationTask.state.in_([
                        TaskState.PENDING.value,
                        TaskState.QUEUED.value,
                    ]),
                )
            )
            .all()
        )

        skipped = 0
        for task in skippable_tasks:
            task.failure_reason_code = FailureReasonCode.DEPENDENCY_FAILED.value
            task.failure_detail = reason
            try:
                transition_task(
                    db=db,
                    task=task,
                    new_state=TaskState.SKIPPED,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="reconciler",
                    reason=reason,
                )
                sync_board_status(db, task)
                skipped += 1
            except (ConflictError, Exception):
                logger.warning(
                    "Could not skip task %s: %s",
                    task.id,
                    reason,
                    exc_info=True,
                )

        if skipped:
            logger.info(
                "Skipped %d pending/queued tasks for run %s: %s",
                skipped,
                run_id,
                reason,
            )

        return skipped
