"""
Coordinator Service — PRD-82A Sequential Mission Coordinator
=============================================================

Main orchestration service: 5s tick loop, mission lifecycle methods,
and the glue between planner, dispatcher, reconciler, and verifier.

Key patterns:
- DB-authoritative, stateless coordinator — every tick reads from DB
- SessionLocal per tick (no stored DB session on singleton)
- Sequential dispatch via MissionDispatcher
- Output summary built when all tasks verified (Section 6)
- Soft budget tracking with warning events (Section 9)

Source: PRD-82A Sections 6, 8, 9, 12 (US-014)
        PRD-102 Section 3.3 (coordinator design)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.orm import Session

from config import Config
from core.models.core import Agent
from core.models.orchestration import (
    OrchestrationEvent,
    OrchestrationRun,
    OrchestrationTask,
    OrchestrationTaskDependency,
)
from core.models.orchestration_enums import (
    ActorType,
    EventType,
    FailureReasonCode,
    RunState,
    TaskState,
    TERMINAL_RUN_STATES,
    DONE_TASK_STATES,
)
from modules.coordination.dispatcher import MissionDispatcher
from modules.coordination.planner import (
    DecompositionResult,
    MissionPlanner,
    PlanValidationError,
)
from modules.coordination.reconciler import MissionReconciler
from services.orchestration_board_bridge import (
    create_mission_board_task,
    create_task_board_task,
    sync_board_status,
)
from services.orchestration_deps import DependencyResolver
from services.orchestration_state import (
    ConflictError,
    InvalidTransitionError,
    emit_event,
    transition_run,
    transition_task,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CoordinatorService
# ---------------------------------------------------------------------------


class CoordinatorService:
    """
    Stateless coordinator that orchestrates sequential missions.

    - ``tick()`` runs every 5s: dispatches next tasks + reconciles active runs.
    - Lifecycle methods: create, approve, reject, review, pause, resume, cancel.
    - No stored DB session — each method/tick acquires its own via SessionLocal.
    """

    def __init__(self):
        self._tick_running: bool = False
        self._scheduler = None
        self._owns_scheduler: bool = False

    # ------------------------------------------------------------------
    # Scheduler lifecycle
    # ------------------------------------------------------------------

    async def start(self, scheduler=None) -> None:
        """
        Register the coordinator tick on the shared scheduler.

        Args:
            scheduler: Shared APScheduler instance from UnifiedScheduler.
                       If None, creates a local scheduler (useful for tests).
        """
        if scheduler:
            self._scheduler = scheduler
            self._owns_scheduler = False
        else:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.jobstores.memory import MemoryJobStore

            self._scheduler = AsyncIOScheduler(
                jobstores={"default": MemoryJobStore()},
            )
            self._scheduler.start()
            self._owns_scheduler = True

        self._scheduler.add_job(
            self.tick,
            "interval",
            seconds=Config.COORDINATOR_TICK_INTERVAL_SECONDS,
            id="coordinator_tick",
            replace_existing=True,
            max_instances=1,
        )
        logger.info(
            "[Coordinator] Started tick loop (interval: %ds)",
            Config.COORDINATOR_TICK_INTERVAL_SECONDS,
        )

    async def stop(self) -> None:
        """Stop the scheduler if this service owns it."""
        if self._scheduler and self._owns_scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        logger.info("[Coordinator] Service stopped")

    # ------------------------------------------------------------------
    # Tick (5s interval)
    # ------------------------------------------------------------------

    async def tick(self) -> Dict[str, Any]:
        """
        Main coordinator tick — called every 5s by the scheduler.

        Finds all active (running) missions, then for each:
          1. Dispatch phase: try to dispatch the next queued task
          2. Reconcile phase: check for stalls, completions, failures

        Returns a summary dict of what happened.
        """
        if self._tick_running:
            return {"status": "skipped", "reason": "already_running"}

        self._tick_running = True
        summary: Dict[str, Any] = {
            "status": "success",
            "runs_processed": 0,
            "dispatches": 0,
            "reconciliations": 0,
            "errors": [],
        }

        try:
            from core.database.database import SessionLocal

            db = SessionLocal()
            try:
                # Find active missions (running state only — not paused/awaiting)
                active_runs: List[OrchestrationRun] = (
                    db.query(OrchestrationRun)
                    .filter(
                        OrchestrationRun.state == RunState.RUNNING.value,
                    )
                    .all()
                )

                if not active_runs:
                    return summary

                for run in active_runs:
                    try:
                        await self._process_run(db, run)
                        summary["runs_processed"] += 1
                        db.commit()
                    except Exception:
                        logger.error(
                            "Error processing run %s", run.id, exc_info=True,
                        )
                        db.rollback()
                        summary["errors"].append(str(run.id))

            finally:
                db.close()

        except Exception:
            logger.error("Coordinator tick failed", exc_info=True)
            summary["status"] = "error"
        finally:
            self._tick_running = False

        return summary

    async def _process_run(
        self,
        db: Session,
        run: OrchestrationRun,
    ) -> None:
        """
        Process a single active mission run: dispatch phase + reconcile phase.
        """
        workspace_id = run.workspace_id

        # Load roster agents for this workspace
        agents: List[Agent] = (
            db.query(Agent)
            .filter(
                and_(
                    Agent.workspace_id == workspace_id,
                    Agent.status == "active",
                )
            )
            .all()
        )

        # --- Dispatch phase ---
        dispatch_result = MissionDispatcher.dispatch_next(db, run, agents)

        if dispatch_result.dispatched:
            # Task was claimed and assigned — now execute async
            task = (
                db.query(OrchestrationTask)
                .filter(OrchestrationTask.id == dispatch_result.task_id)
                .first()
            )

            if task:
                await self._execute_task(db, run, task, dispatch_result.agent_id)

        # --- Reconcile phase ---
        await MissionReconciler.reconcile(db, run)

        # --- Check if run advanced to verifying → build summary ---
        db.refresh(run)
        if RunState(run.state) == RunState.VERIFYING:
            self._build_and_advance_to_awaiting_human(db, run)

    async def _execute_task(
        self,
        db: Session,
        run: OrchestrationRun,
        task: OrchestrationTask,
        agent_id: int,
    ) -> None:
        """
        Execute a dispatched task via AgentFactory.execute_with_prompt().

        Transitions: ASSIGNED → RUNNING → COMPLETED/FAILED.
        """
        from modules.agents.factory.agent_factory import AgentFactory

        # Transition to RUNNING
        try:
            MissionDispatcher.record_task_running(db, task)
            db.flush()
        except (ConflictError, InvalidTransitionError):
            logger.warning(
                "Could not transition task %s to running", task.id,
                exc_info=True,
            )
            return

        # Build the prompt
        prompt = MissionDispatcher.build_task_prompt(task)

        # Execute via AgentFactory
        factory = AgentFactory(db_session=db)

        # Ensure agent is activated so we can bump max_tokens for mission tasks
        agent_runtime = factory.active_agents.get(agent_id)
        if not agent_runtime:
            agent_runtime = await factory.activate_agent(agent_id, workspace_dir="/tmp/automatos_workspace")

        # Mission tasks need longer outputs than the 2000-token default
        if agent_runtime and hasattr(agent_runtime, "llm_manager"):
            original_max_tokens = agent_runtime.llm_manager.config.max_tokens
            agent_runtime.llm_manager.config.max_tokens = max(original_max_tokens, 4096)

        try:
            result = await factory.execute_with_prompt(
                agent=agent_id,
                prompt=prompt,
                max_retries=0,  # Coordinator manages retries, not AgentFactory
                max_tool_iterations=10,
            )
        except Exception as exc:
            logger.error(
                "execute_with_prompt failed for task %s: %s",
                task.id,
                exc,
                exc_info=True,
            )
            result = {"status": "error", "error": str(exc)}

        # Record completion/failure
        MissionDispatcher.record_task_completion(db, task, result)

        # Update run-level token tracking (PRD-82A Section 9)
        task_tokens = result.get("execution", {}).get("tokens_used", 0)
        if task_tokens:
            run.tokens_used = (run.tokens_used or 0) + task_tokens

            # Budget warning check
            if (
                run.token_budget_estimate
                and run.tokens_used > run.token_budget_estimate * 1.5
            ):
                emit_event(
                    db=db,
                    run_id=run.id,
                    event_type=EventType.BUDGET_WARNING,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="coordinator",
                    payload={
                        "tokens_used": run.tokens_used,
                        "token_budget_estimate": run.token_budget_estimate,
                        "ratio": round(run.tokens_used / run.token_budget_estimate, 2),
                    },
                )

    # ------------------------------------------------------------------
    # Lifecycle: create_mission
    # ------------------------------------------------------------------

    async def create_mission(
        self,
        db: Session,
        workspace_id: UUID,
        goal: str,
        created_by: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationRun:
        """
        Create a new mission: plan → create DB rows → board tasks → await approval.

        Args:
            db: SQLAlchemy session (caller manages transaction).
            workspace_id: Owning workspace UUID.
            goal: Natural-language goal string.
            created_by: Clerk user ID (e.g., 'user_xxx').
            config: Optional mission config overrides.

        Returns:
            The created OrchestrationRun.

        Raises:
            PlanValidationError: if planner cannot produce a valid plan.
        """
        mission_config = config or {}

        # Create the run in PENDING state
        run = OrchestrationRun(
            workspace_id=workspace_id,
            goal=goal,
            created_by=created_by,
            config=mission_config,
            state=RunState.PENDING.value,
            state_type="initial",
            max_retries=mission_config.get(
                "max_retries", Config.COORDINATOR_MAX_TASK_RETRIES,
            ),
            max_concurrent=1,  # Sequential only in 82A
        )
        db.add(run)
        db.flush()  # Get the run.id

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_CREATED,
            actor_type=ActorType.HUMAN,
            actor_id=created_by,
            payload={"goal": goal[:500]},
        )

        logger.info(
            "Mission created: run=%s workspace=%s goal='%s'",
            run.id,
            workspace_id,
            goal[:80],
        )

        # Transition to PLANNING
        transition_run(
            db=db,
            run=run,
            new_state=RunState.PLANNING,
            actor_type=ActorType.COORDINATOR,
            actor_id="coordinator",
        )

        # Load roster agents
        agents: List[Agent] = (
            db.query(Agent)
            .filter(
                and_(
                    Agent.workspace_id == workspace_id,
                    Agent.status == "active",
                )
            )
            .all()
        )

        # Decompose goal into task DAG
        try:
            decomposition = await MissionPlanner.decompose(
                goal=goal,
                workspace_id=workspace_id,
                agents=agents,
                config=mission_config,
            )
        except PlanValidationError:
            transition_run(
                db=db,
                run=run,
                new_state=RunState.FAILED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                reason="Plan validation failed after all retries",
            )
            raise

        # Store the plan on the run
        run.plan = {
            "tasks": [
                {
                    "temp_id": t.temp_id,
                    "title": t.title,
                    "description": t.description,
                    "agent_role": t.agent_role,
                    "sequence_number": t.sequence_number,
                    "task_type": t.task_type,
                }
                for t in decomposition.tasks
            ],
            "dependencies": [
                {
                    "from": d.from_task_temp_id,
                    "to": d.to_task_temp_id,
                }
                for d in decomposition.dependencies
            ],
        }
        run.token_budget_estimate = decomposition.token_estimate

        # Create OrchestrationTask rows
        temp_id_to_task: Dict[str, OrchestrationTask] = {}
        for planned in decomposition.tasks:
            task = OrchestrationTask(
                run_id=run.id,
                title=planned.title,
                description=planned.description,
                task_type=planned.task_type,
                sequence_number=planned.sequence_number,
                agent_role=planned.agent_role,
                state=TaskState.PENDING.value,
                state_type="initial",
                verification_criteria=planned.verification_criteria or None,
                input_context={
                    "required_tools": planned.required_tools,
                } if planned.required_tools else None,
                max_retries=run.max_retries,
            )
            db.add(task)
            db.flush()  # Get task.id
            temp_id_to_task[planned.temp_id] = task

            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.TASK_CREATED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                task_id=task.id,
                payload={
                    "title": planned.title,
                    "sequence_number": planned.sequence_number,
                    "agent_role": planned.agent_role,
                },
            )

        # Create dependency edges
        for dep in decomposition.dependencies:
            from_task = temp_id_to_task.get(dep.from_task_temp_id)
            to_task = temp_id_to_task.get(dep.to_task_temp_id)
            if from_task and to_task:
                dep_row = OrchestrationTaskDependency(
                    task_id=to_task.id,
                    depends_on_task_id=from_task.id,
                )
                db.add(dep_row)

        db.flush()

        # Create board tasks for kanban visibility
        try:
            create_mission_board_task(db, run)
            for task in temp_id_to_task.values():
                create_task_board_task(db, run, task)
        except Exception:
            logger.warning(
                "Failed to create board tasks for mission %s",
                run.id,
                exc_info=True,
            )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_PLAN_READY,
            actor_type=ActorType.COORDINATOR,
            actor_id="coordinator",
            payload={
                "task_count": len(decomposition.tasks),
                "token_estimate": decomposition.token_estimate,
            },
        )

        # Auto-approve or await approval
        auto_approve = mission_config.get("auto_approve", False)
        if auto_approve:
            transition_run(
                db=db,
                run=run,
                new_state=RunState.RUNNING,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                reason="Auto-approved",
            )
            self._queue_initial_tasks(db, run)
            logger.info("Mission %s auto-approved → running", run.id)
        else:
            transition_run(
                db=db,
                run=run,
                new_state=RunState.AWAITING_APPROVAL,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
            )
            logger.info("Mission %s → awaiting_approval", run.id)

        return run

    # ------------------------------------------------------------------
    # Lifecycle: approve_plan
    # ------------------------------------------------------------------

    def approve_plan(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationRun:
        """
        Approve a mission plan and start execution.

        Transitions: awaiting_approval → running.
        Queues initial tasks (those with no dependencies) as QUEUED.
        """
        run = self._get_run(db, run_id)

        if modifications:
            # Apply plan modifications (e.g., reorder, remove tasks)
            run.plan = {**(run.plan or {}), "modifications": modifications}

        transition_run(
            db=db,
            run=run,
            new_state=RunState.RUNNING,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_APPROVED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
            payload={"modifications": modifications} if modifications else None,
        )

        self._queue_initial_tasks(db, run)

        logger.info("Mission %s approved by %s → running", run_id, actor_id)
        return run

    # ------------------------------------------------------------------
    # Lifecycle: reject_plan
    # ------------------------------------------------------------------

    def reject_plan(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
        reason: str,
    ) -> OrchestrationRun:
        """Reject a mission plan. Transitions: awaiting_approval → failed."""
        run = self._get_run(db, run_id)

        transition_run(
            db=db,
            run=run,
            new_state=RunState.FAILED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
            reason=reason,
        )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_REJECTED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
            payload={"reason": reason},
        )

        # Skip all pending tasks
        MissionReconciler._skip_remaining_tasks(
            db=db,
            run_id=run.id,
            reason=f"Plan rejected: {reason}",
        )

        logger.info("Mission %s rejected by %s: %s", run_id, actor_id, reason)
        return run

    # ------------------------------------------------------------------
    # Lifecycle: review_mission
    # ------------------------------------------------------------------

    def review_mission(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
        verdict: str,
        task_feedback: Optional[Dict[str, str]] = None,
    ) -> OrchestrationRun:
        """
        Submit human review after all tasks are verified.

        Args:
            verdict: 'accept' or 'reject'
            task_feedback: Optional dict mapping task_id → feedback string.
                          On reject, tasks with feedback get re-queued.
        """
        run = self._get_run(db, run_id)

        if verdict == "accept":
            transition_run(
                db=db,
                run=run,
                new_state=RunState.COMPLETED,
                actor_type=ActorType.HUMAN,
                actor_id=actor_id,
            )
            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.RUN_COMPLETED,
                actor_type=ActorType.HUMAN,
                actor_id=actor_id,
            )
            logger.info("Mission %s accepted by %s → completed", run_id, actor_id)

        elif verdict == "reject":
            # Re-queue specific tasks with feedback
            requeued_count = 0
            if task_feedback:
                for task_id_str, feedback in task_feedback.items():
                    task = (
                        db.query(OrchestrationTask)
                        .filter(
                            and_(
                                OrchestrationTask.id == task_id_str,
                                OrchestrationTask.run_id == run_id,
                            )
                        )
                        .first()
                    )
                    if task:
                        self._requeue_task_with_feedback(
                            db, task, feedback, actor_id,
                        )
                        requeued_count += 1

            # Transition run back to running for re-execution
            transition_run(
                db=db,
                run=run,
                new_state=RunState.RUNNING,
                actor_type=ActorType.HUMAN,
                actor_id=actor_id,
                reason=f"Human rejected — {requeued_count} tasks re-queued",
            )

            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.RUN_RESUMED,
                actor_type=ActorType.HUMAN,
                actor_id=actor_id,
                payload={
                    "verdict": "reject",
                    "tasks_requeued": requeued_count,
                },
            )

            logger.info(
                "Mission %s rejected by %s — %d tasks re-queued",
                run_id,
                actor_id,
                requeued_count,
            )

        return run

    # ------------------------------------------------------------------
    # Lifecycle: pause / resume / cancel
    # ------------------------------------------------------------------

    def pause_mission(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
    ) -> OrchestrationRun:
        """Pause a running mission. Running tasks continue but no new dispatches."""
        run = self._get_run(db, run_id)

        transition_run(
            db=db,
            run=run,
            new_state=RunState.PAUSED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_PAUSED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        logger.info("Mission %s paused by %s", run_id, actor_id)
        return run

    def resume_mission(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
    ) -> OrchestrationRun:
        """Resume a paused mission."""
        run = self._get_run(db, run_id)

        transition_run(
            db=db,
            run=run,
            new_state=RunState.RUNNING,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_RESUMED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        logger.info("Mission %s resumed by %s", run_id, actor_id)
        return run

    def cancel_mission(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
    ) -> OrchestrationRun:
        """Cancel a mission. Running tasks continue to completion; no new dispatches."""
        run = self._get_run(db, run_id)

        transition_run(
            db=db,
            run=run,
            new_state=RunState.CANCELLED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_CANCELLED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        # Skip all pending/queued tasks
        MissionReconciler._skip_remaining_tasks(
            db=db,
            run_id=run.id,
            reason="Mission cancelled",
        )

        logger.info("Mission %s cancelled by %s", run_id, actor_id)
        return run

    # ------------------------------------------------------------------
    # Output summary builder (PRD-82A Section 6)
    # ------------------------------------------------------------------

    @staticmethod
    def build_output_summary(
        db: Session,
        run: OrchestrationRun,
    ) -> Dict[str, Any]:
        """
        Build the mission output summary from verified task results.

        This is a structured aggregation — no LLM call needed.
        Stored in run.output_summary JSONB.
        """
        tasks: List[OrchestrationTask] = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run.id)
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        now = datetime.now(timezone.utc)
        started = run.started_at
        total_duration = (
            (now - started).total_seconds()
            if started
            else 0
        )

        verified_count = sum(
            1 for t in tasks if TaskState(t.state) == TaskState.VERIFIED
        )
        failed_count = sum(
            1 for t in tasks if TaskState(t.state) == TaskState.FAILED
        )

        task_summaries = []
        for task in tasks:
            # Get agent name if assigned
            agent_name = None
            if task.assigned_agent_id:
                agent = db.query(Agent).get(task.assigned_agent_id)
                if agent:
                    agent_name = agent.name

            output_excerpt = ""
            if task.output:
                output_excerpt = task.output[:500]

            task_summaries.append({
                "sequence": task.sequence_number,
                "title": task.title,
                "agent": agent_name,
                "verdict": (
                    "pass" if TaskState(task.state) == TaskState.VERIFIED
                    else "fail" if TaskState(task.state) == TaskState.FAILED
                    else TaskState(task.state).value
                ),
                "output_excerpt": output_excerpt,
                "tokens_used": task.tokens_used or 0,
            })

        return {
            "goal": run.goal,
            "tasks_completed": verified_count,
            "tasks_failed": failed_count,
            "total_duration_seconds": round(total_duration),
            "task_summaries": task_summaries,
            "generated_at": now.isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_run(db: Session, run_id: UUID) -> OrchestrationRun:
        """Load an OrchestrationRun by ID. Raises ValueError if not found."""
        run = db.query(OrchestrationRun).get(run_id)
        if run is None:
            raise ValueError(f"Mission run not found: {run_id}")
        return run

    @staticmethod
    def _queue_initial_tasks(
        db: Session,
        run: OrchestrationRun,
    ) -> int:
        """
        Queue tasks that have no dependencies (roots of the DAG).

        Transitions root tasks from PENDING → QUEUED so the dispatcher
        picks them up on the next tick.

        Returns count of tasks queued.
        """
        ready = DependencyResolver.get_ready_tasks(db, run.id)
        queued = 0

        for task in ready:
            try:
                transition_task(
                    db=db,
                    task=task,
                    new_state=TaskState.QUEUED,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="coordinator",
                )
                sync_board_status(db, task)
                queued += 1
            except (ConflictError, InvalidTransitionError):
                logger.warning(
                    "Could not queue task %s", task.id, exc_info=True,
                )

        if queued:
            logger.info(
                "Queued %d initial tasks for mission %s", queued, run.id,
            )
        return queued

    @staticmethod
    def _requeue_task_with_feedback(
        db: Session,
        task: OrchestrationTask,
        feedback: str,
        actor_id: str,
    ) -> None:
        """
        Re-queue a verified/failed task with human feedback for retry.

        Injects feedback into task.input_context so the agent sees it
        on the next execution.
        """
        # Inject feedback — immutable replace for JSONB dirty detection
        task.input_context = {
            **(task.input_context or {}),
            "retry_feedback": feedback,
        }

        task.failure_reason_code = FailureReasonCode.VERIFICATION_REJECT.value
        task.attempt_number = (task.attempt_number or 0) + 1

        # Transition verified/failed → retrying → assigned pipeline
        try:
            transition_task(
                db=db,
                task=task,
                new_state=TaskState.RETRYING,
                actor_type=ActorType.HUMAN,
                actor_id=actor_id,
                reason=f"Human rejected with feedback: {feedback[:200]}",
            )
            transition_task(
                db=db,
                task=task,
                new_state=TaskState.ASSIGNED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                reason="Re-dispatching with human feedback",
            )
            sync_board_status(db, task)
        except (ConflictError, InvalidTransitionError):
            logger.warning(
                "Could not requeue task %s with feedback",
                task.id,
                exc_info=True,
            )

    def _build_and_advance_to_awaiting_human(
        self,
        db: Session,
        run: OrchestrationRun,
    ) -> None:
        """
        Build output summary and transition run from verifying → awaiting_human.
        """
        try:
            summary = self.build_output_summary(db, run)
            run.output_summary = summary

            transition_run(
                db=db,
                run=run,
                new_state=RunState.AWAITING_HUMAN,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                reason="All tasks verified — awaiting human review",
            )

            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.RUN_AWAITING_HUMAN,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                payload={
                    "tasks_completed": summary["tasks_completed"],
                    "total_duration_seconds": summary["total_duration_seconds"],
                },
            )

            logger.info(
                "Mission %s → awaiting_human (summary: %d tasks, %ds)",
                run.id,
                summary["tasks_completed"],
                summary["total_duration_seconds"],
            )

        except Exception:
            logger.error(
                "Failed to build output summary for run %s",
                run.id,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Singleton accessor (matches HeartbeatService pattern)
# ---------------------------------------------------------------------------

_coordinator_service: Optional[CoordinatorService] = None


def get_coordinator_service() -> CoordinatorService:
    """Get or create the singleton CoordinatorService."""
    global _coordinator_service
    if _coordinator_service is None:
        _coordinator_service = CoordinatorService()
    return _coordinator_service
