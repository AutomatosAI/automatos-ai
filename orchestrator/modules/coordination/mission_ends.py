"""How a mission ends, once its steps are done or none of them can move (F268, night 7b;
F283, night 8).

#0176 failed on night 7 when the AI credit ran out: two automatic re-plans replaced
the failed part of its plan, and when a step failed for good, every step still waiting
was skipped. On night 7b the owner resumed it (F247). Its four stranded steps ran and
the owner approved every one, and the mission still ended "failed: 8 tasks skipped
due to upstream failure", and stayed in Needs-you until the owner cancelled it.

- A step a re-plan replaced (``replaced_by_replan``) is no longer part of the plan, so
  it counts for nothing when the mission ends: a mission whose live steps all passed
  completes (``counts_only_its_live_steps``, around the reconciler's end rule).
- A retried mission's steps that were skipped because a step failed
  (``dependency_failed``) wait to run again with its failed steps
  (``mission_retry.retry_failed``): the plan runs as written.

F283 (night 8): a mission never reached "failed", so nothing recovered it. #0352.2
failed the mission's own check at 04:58:29 ("It still has placeholders where its
content belongs: [Cafe Name].", attempt 1 of 3). The step after it waited for it, and
the mission said "running" for 20 minutes with nothing moving, until the owner
cancelled it; #0433's two written steps did the same. Pause and Resume changed
nothing, Replan answered that the mission must be 'failed', and Needs you never
counted it: a step that fails the check is not a "fatal" failure, and a mission with
a step still waiting never ends.

- ``fails_when_nothing_can_move`` (around the reconciler's fatal-failure rule) and
  ``counts_only_its_live_steps`` (around its end rule): a running mission fails when
  a step failed and nothing else in it can move: no step is queued, being worked,
  back for a redo or being checked, and no waiting step has had every step it builds
  on pass. Its stop reason is ``dependency_failed`` and its stop_detail names the step
  by its card number and says why ("Step #0352.2 failed the mission's check: …").
  The steps that waited for it are skipped for the failure (``dependency_failed``),
  so Resume runs them again (``mission_retry.retry_failed``), and Replan is offered.
  The failed step's card ends Failed with its mission
  (``mission_card_result.carries_the_missions_result``).
- A step that failed the mission's check is not failed out of turn by the old
  fatal rule (its attempts are spent by then, ``unfinished_work``): the steps that do
  not wait for it finish first, and the mission then fails saying why.
- Both rules read the steps again under the mission's row lock, so a change committed
  before it (a send-back, a pause, a cancel) is seen, and a change to the mission that
  another request is still writing leaves the decision to the next tick (5 s). The
  tick never waits on the lock (F105).
- A completed mission says what happened (``says_what_it_completed``): "All 4 steps
  verified.", and after a re-plan "8 steps of the earlier plan were replaced when it
  was re-planned and did not run." Resumed on night 8, #0176 said "All tasks verified
  successfully" with its 8 replaced steps still showing skipped.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.models.orchestration_enums import (
    DONE_TASK_STATES,
    ActorType,
    EventType,
    FailureReasonCode,
    RunState,
    StopReason,
    TaskState,
)

logger = logging.getLogger(__name__)

# coordinator_service.replan_mission writes this on the steps a re-plan replaced.
REPLACED_BY_REPLAN = "replaced_by_replan"
# A step in one of these can still move without the owner: queued, being worked, back
# for a redo, done and waiting for its check or being checked (a step held for the
# owner's check waits in verifying), or stalled (the reconciler queues it again).
MOVING_STATES = frozenset(state.value for state in (
    TaskState.QUEUED, TaskState.ASSIGNED, TaskState.RUNNING, TaskState.RETRYING,
    TaskState.COMPLETED, TaskState.VERIFYING, TaskState.STALLED,
))
FAILED_THE_CHECK = "Step {step} failed the mission's check: {why}"
FAILED_STEP = "Step {step} failed: {why}"
ALSO_FAILED = "{steps} failed too."
NO_REASON = "no reason was recorded"
STOP_DETAIL_CHARS = 2000
ALL_VERIFIED = "All {count} steps verified."
ONE_VERIFIED = "Its one step verified."
REPLACED_STEPS = "{count} steps of the earlier plan were replaced when it was re-planned and did not run."
REPLACED_STEP = "1 step of the earlier plan was replaced when it was re-planned and did not run."
STEP_CARD = "orchestration_task"
RECONCILER = "reconciler"
# The counts a reconcile pass reports, passed through when the mission does not end.
RESULT_COUNTS = ("stalls_detected", "stalls_recovered", "tasks_failed", "tasks_verified",
                 "tasks_verification_failed")


def superseded(task: Any) -> bool:
    """A step a re-plan replaced: skipped, and no longer part of the plan."""
    return task.state == TaskState.SKIPPED.value and task.failure_reason_code == REPLACED_BY_REPLAN


def skipped_for_a_failure(task: Any) -> bool:
    """A step skipped because another step failed for good: it never ran."""
    return (task.state == TaskState.SKIPPED.value
            and task.failure_reason_code == FailureReasonCode.DEPENDENCY_FAILED.value)


def failed_the_check(task: Any) -> bool:
    """A step that failed the mission's own check (``unfinished_work``)."""
    return (task.state == TaskState.FAILED.value
            and task.failure_reason_code == FailureReasonCode.VERIFICATION_FAIL.value)


def live_steps(steps: Sequence[Any]) -> List[Any]:
    """The steps of the mission's plan: every step but those a re-plan replaced."""
    return [task for task in steps if not superseded(task)]


def in_sequence(task: Any) -> tuple:
    """A step's place in the plan, for sorting."""
    return (task.sequence_number or 0, str(task.id))


def steps_under_lock(db: Any, run: Any) -> Optional[List[Any]]:
    """The mission's steps as they stand under its row lock, or None when another
    change holds the lock or the mission no longer runs. What this session had
    written is flushed first; the run and its steps are then read again, so a
    change committed before the lock is seen."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    db.flush()
    held = (db.query(OrchestrationRun.id).filter(OrchestrationRun.id == run.id,
                                                 OrchestrationRun.workspace_id == run.workspace_id)
            .with_for_update(skip_locked=True).scalar())
    if held is None:
        logger.info("[F283] mission %s is held by another change: the next tick decides", run.id)
        return None
    db.refresh(run)
    if run.state != RunState.RUNNING.value:
        return None
    return db.query(OrchestrationTask).filter(OrchestrationTask.run_id == run.id).populate_existing().all()


def counts_only_its_live_steps(advance: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``MissionReconciler._advance_run_on_completion``: it judges the mission by
    the steps of its plan, not the ones a re-plan replaced, as they stand under the
    mission's lock (F283). A step sent back meanwhile keeps the mission running, and a
    failed step fails it saying which step and why (``fail_if_stuck``)."""
    @functools.wraps(advance)
    def wrapped(db: Any, run: Any, all_tasks: Any, *args: Any, **kwargs: Any) -> Any:
        steps = steps_under_lock(db, run)
        live = live_steps(steps or []) or list(steps or [])
        if not live or any(TaskState(task.state) not in DONE_TASK_STATES for task in live):
            return _result(run, kwargs)
        if fail_if_stuck(db, run, live):
            return _result(run, kwargs, failed=True)
        return advance(db, run, live, *args, **{**kwargs, "failed_tasks": []})
    return wrapped


def _result(run: Any, counts: Dict[str, Any], *, failed: bool = False) -> Any:
    """The reconcile pass's result: the mission as it was, or failed here."""
    from modules.coordination.reconciler import ReconcileResult

    ended = {"run_advanced": True, "run_new_state": RunState.FAILED.value} if failed else {}
    return ReconcileResult(run_id=run.id, **{key: counts[key] for key in RESULT_COUNTS if key in counts}, **ended)


def fails_when_nothing_can_move(check: Callable[..., bool]) -> Callable[..., bool]:
    """Wrap ``MissionReconciler._check_fatal_failure``: it reads the failed steps under
    the mission's lock. A step that failed fatally fails its mission at once, as
    before, unless it failed the mission's check; then, as for any failed step, its
    mission fails once nothing else in it can move (F283)."""
    @functools.wraps(check)
    def wrapped(db: Any, run: Any, failed_tasks: Any) -> bool:
        steps = steps_under_lock(db, run)
        if not steps:
            return False
        fatal_now = [task for task in live_steps(steps)
                     if task.state == TaskState.FAILED.value and not failed_the_check(task)]
        if fatal_now and check(db, run, fatal_now):
            return True
        return fail_if_stuck(db, run, steps)
    return wrapped


def fail_if_stuck(db: Any, run: Any, steps: Sequence[Any]) -> bool:
    """Fail ``run`` when one of its ``steps`` failed and none of the others can move.
    True when it failed."""
    live = live_steps(steps)
    failed = sorted((task for task in live if task.state == TaskState.FAILED.value), key=in_sequence)
    if not failed or any(task.state in MOVING_STATES for task in live) or _a_step_is_ready(db, run):
        return False
    return _fail(db, run, failed)


def _a_step_is_ready(db: Any, run: Any) -> bool:
    """A waiting step every step it builds on has passed: the dispatcher starts it."""
    from services.orchestration_deps import DependencyResolver

    return bool(DependencyResolver.get_ready_tasks(db, run.id))


def _fail(db: Any, run: Any, failed: List[Any]) -> bool:
    """The mission fails saying which step stopped it and why, and the steps still
    waiting are skipped for the failure."""
    from modules.coordination.reconciler import MissionReconciler
    from modules.coordination.verification import VerificationService
    from services.orchestration_state import ConflictError, emit_event, transition_run

    why = why_it_stopped(db, run, failed)
    try:
        transition_run(db=db, run=run, new_state=RunState.FAILED, actor_type=ActorType.COORDINATOR,
                       actor_id=RECONCILER, reason=why, stop_reason=StopReason.DEPENDENCY_FAILED.value,
                       stop_detail=why)
    except ConflictError:
        logger.warning("[F283] conflict failing run %s: the next tick decides", run.id)
        return False
    MissionReconciler._skip_remaining_tasks(db=db, run_id=run.id, reason=why)
    emit_event(db=db, run_id=run.id, event_type=EventType.RUN_FAILED, actor_type=ActorType.COORDINATOR,
               actor_id=RECONCILER, payload={"fatal_task_id": str(failed[0].id), "fatal_task_title": failed[0].title,
                                             "failure_reason": failed[0].failure_reason_code,
                                             "nothing_can_move": True})
    VerificationService.clear_cache(run.id)
    logger.info("[F283] run %s failed, nothing else could move: %s", run.id, why)
    return True


def why_it_stopped(db: Any, run: Any, failed: Sequence[Any]) -> str:
    """The mission's stop_detail: the first failed step by its number, and why."""
    numbers = step_numbers(db, run, failed)
    first = failed[0]
    template = FAILED_THE_CHECK if failed_the_check(first) else FAILED_STEP
    words = template.format(step=_named(first, numbers),
                            why=str(first.failure_detail or first.failure_reason_code or NO_REASON).strip())
    if len(failed) > 1:
        words = f"{words} {ALSO_FAILED.format(steps=', '.join(_named(task, numbers) for task in failed[1:]))}"
    return words[:STOP_DETAIL_CHARS]


def _named(task: Any, numbers: Dict[Any, str]) -> str:
    return numbers.get(task.id) or f"'{task.title}'"


def step_cards(db: Any, run: Any, steps: Sequence[Any]) -> Dict[Any, Any]:
    """Each step's card on the board, by the step's id, in the mission's workspace."""
    from core.models.core import BoardTask

    ids = [task.id for task in steps]
    if not ids:
        return {}
    cards = db.query(BoardTask).filter(BoardTask.source_type == STEP_CARD, BoardTask.workspace_id == run.workspace_id,
                                       BoardTask.orchestration_task_id.in_(ids)).all()
    return {card.orchestration_task_id: card for card in cards}


def step_numbers(db: Any, run: Any, steps: Sequence[Any]) -> Dict[Any, str]:
    """Each step's number as the board shows it (#0352.2), by the step's id."""
    from services.ticket_numbers import ticket_numbers

    cards = step_cards(db, run, steps)
    numbers = ticket_numbers(db, run.workspace_id, cards.values()) if cards else {}
    return {step_id: numbers[card.id] for step_id, card in cards.items() if numbers.get(card.id)}


def completion_words(db: Any, run: Any) -> str:
    """What a completed mission did, in plain words: its verified steps, and the steps
    of an earlier plan that a re-plan replaced and that never ran."""
    from core.models.orchestration import OrchestrationTask

    steps = db.query(OrchestrationTask).filter(OrchestrationTask.run_id == run.id).all()
    verified = sum(1 for task in steps if task.state == TaskState.VERIFIED.value)
    replaced = sum(1 for task in steps if superseded(task))
    words = ONE_VERIFIED if verified == 1 else ALL_VERIFIED.format(count=verified)
    if replaced:
        words = f"{words} {REPLACED_STEP if replaced == 1 else REPLACED_STEPS.format(count=replaced)}"
    return words


def says_what_it_completed(complete: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``CoordinatorService._complete_verified_run``: a completed mission's
    stop_detail says what ran and what a re-plan replaced (``completion_words``)."""
    @functools.wraps(complete)
    async def wrapped(self: Any, db: Any, run: Any) -> None:
        await complete(self, db, run)
        if run.state == RunState.COMPLETED.value:
            run.stop_detail = completion_words(db, run)
            db.flush()
    return wrapped


__all__ = [
    "REPLACED_BY_REPLAN", "completion_words", "counts_only_its_live_steps", "fail_if_stuck", "failed_the_check",
    "fails_when_nothing_can_move", "in_sequence", "live_steps", "says_what_it_completed", "skipped_for_a_failure",
    "step_cards", "step_numbers", "steps_under_lock", "superseded", "why_it_stopped",
]
