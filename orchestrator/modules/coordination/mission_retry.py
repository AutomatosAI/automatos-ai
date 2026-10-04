"""F247 (night 7): a mission that failed can be retried, from its page or by Auto.

#0176 failed while the AI credit was out, and then nothing could run it:
- Resume was refused ("expected 'paused'");
- Replan was refused ("replanned 2 times, maximum is 2": both went automatically);
- the board refuses a mission's steps ("Retry or change it from the mission").
Four step cards could run from nowhere.

Resume now retries a failed mission. It runs the same plan again from where it
stopped, and spends no re-plan:
- each failed step waits to run again, with its attempts reset. It starts once the
  steps it builds on have passed;
- a step that passed keeps its result;
- the progress ledger starts again, so old churn is never read as a new stall;
- the mission then resumes as a paused one does, and F153 raises a spent budget.

``retries_a_failed_mission`` wraps ``CoordinatorService.resume_mission`` (Auto's
platform_resume_mission). The mission page's Resume goes through the API, which
does the same (``api/mission_retry.py``).

F283 (night 8):
- Resume on the mission that failed while planning on 26 September (no plan, no
  steps) left it "running" with nothing to run for over an hour. A failed mission
  with no steps has nothing to resume: Resume says so and points to Replan
  (``NothingToResume``), and the mission stays failed.
- F268: Resume on #0176 completed it, and its card stayed Cancelled with
  "8 tasks skipped due to upstream failure" on it: the owner had dismissed the
  failed card on night 7b, and the mission's card follows its mission only while
  the card is open (F245). A failed mission run again (Resume, Replan) reopens its
  card, without the old failure, and the card ends Done when the mission completes.
- Pause → Resume on #0352 and #0433 left their failed steps failed, and the
  missions sat "running". A paused mission whose step failed for good runs that
  step again when it resumes (``runs_its_failed_steps_again``), as Resume on a failed
  mission does; otherwise it would only fail again at the next tick (``mission_ends``).
- A replan replaces the steps the failure skipped too
  (``replaces_what_the_failure_skipped``). They belong to the part of the plan it
  re-did, and left skipped they failed the replanned mission at its end ("1 tasks
  skipped due to upstream failure") however its new steps went.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, List

from sqlalchemy.orm import Session

from services.orchestration_state import ConflictError

logger = logging.getLogger(__name__)

RETRY_REASON = "Retry: the failed steps run again, with the same plan"
LEDGER_KEY = "progress_ledger"
NO_STEPS_TO_RESUME = "This mission failed while planning, so it has no steps to resume: use Replan."
RESUMED_NOTE = "Resumed: the mission runs its failed steps again."
REPLANNED_NOTE = "Re-planned: the mission runs its new plan."
REPLACED_DETAIL = "Replaced during replan #{number}"


class NothingToResume(ConflictError, ValueError):
    """Resume refused: the mission failed before it had any steps. A conflict to the
    missions API (409), a ValueError to Auto's tool (a refusal), as
    ``owner_checks.WaitsForTheOwnersCheck`` is."""

    def __init__(self, run_id: Any):
        self.entity_type, self.entity_id = "run", run_id
        Exception.__init__(self, NO_STEPS_TO_RESUME)


def retries_a_failed_mission(resume: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``CoordinatorService.resume_mission``: a failed mission is made ready to
    run again (``retry_failed``) and then resumed; a paused one is resumed and its
    failed steps run again (F283); any other is resumed as before."""
    @functools.wraps(resume)
    def wrapped(self: Any, db: Session, run_id: Any, actor_id: str) -> Any:
        from core.models.orchestration_enums import RunState

        run = self._get_run(db, run_id)
        was = RunState(run.state)
        if was == RunState.FAILED:
            retry_failed(db, run, actor_id)
        resumed = resume(self, db, run_id, actor_id)
        if was == RunState.PAUSED:
            runs_its_failed_steps_again(db, resumed, actor_id)
        return resumed
    return wrapped


def runs_its_failed_steps_again(db: Session, run: Any, actor_id: str) -> List[Any]:
    """F283: a resumed mission's steps that failed for good wait to run again, with
    fresh attempts. Returns them."""
    from core.models.orchestration import OrchestrationTask
    from core.models.orchestration_enums import TaskState

    failed = db.query(OrchestrationTask).filter(
        OrchestrationTask.run_id == run.id, OrchestrationTask.state == TaskState.FAILED.value,
    ).order_by(OrchestrationTask.sequence_number).all()
    for task in failed:
        _wait_again(db, task, actor_id)
    if failed:
        logger.info("[F283] run %s resumed by %s: %d failed step(s) run again", run.id, actor_id, len(failed))
    return failed


def retry_failed(db: Session, run: Any, actor_id: str) -> List[Any]:
    """Turn a failed ``run`` into a paused one whose failed steps wait to run again.
    Returns those steps. Resume runs it. A mission with no steps is refused
    (``NothingToResume``) and left as it is."""
    from core.models.orchestration import OrchestrationTask
    from core.models.orchestration_enums import ActorType, RunState, TaskState
    from services.mission_card_result import reopen_the_missions_card
    from services.orchestration_state import transition_run

    from modules.coordination.mission_ends import skipped_for_a_failure

    steps = db.query(OrchestrationTask).filter(
        OrchestrationTask.run_id == run.id).order_by(OrchestrationTask.sequence_number).all()
    if not steps:
        raise NothingToResume(run.id)
    # F268 (night 7b): the steps skipped because a step failed run again too; #0176
    # retried its four failed steps, and its eight skipped ones failed it again.
    failed = [t for t in steps if t.state == TaskState.FAILED.value or skipped_for_a_failure(t)]
    transition_run(db=db, run=run, new_state=RunState.PAUSED, actor_type=ActorType.HUMAN,
                   actor_id=actor_id, reason=RETRY_REASON)
    run.completed_at = None
    run.config = {key: value for key, value in (run.config or {}).items() if key != LEDGER_KEY}
    for task in failed:
        _wait_again(db, task, actor_id)
    reopen_the_missions_card(db, run, note=RESUMED_NOTE)
    logger.info("[coordinator] run %s retried by %s: %d failed or skipped step(s) wait to run again",
                run.id, actor_id, len(failed))
    return failed


def _wait_again(db: Session, task: Any, actor_id: str) -> None:
    """One failed step back to waiting for its inputs: a fresh attempt, no failure, no
    agent (the dispatcher picks one when it is ready), and its card back in the Inbox."""
    from core.models.orchestration_enums import ActorType, TaskState
    from services.orchestration_board_bridge import sync_board_status
    from services.orchestration_state import transition_task

    task.attempt_number = 0
    task.failure_detail = None
    task.failure_reason_code = None
    task.completed_at = None
    task.assigned_agent_id = None
    transition_task(db=db, task=task, new_state=TaskState.PENDING, actor_type=ActorType.HUMAN,
                    actor_id=actor_id, reason=RETRY_REASON)
    sync_board_status(db, task)
    _card_without_the_failure(db, task)


def _card_without_the_failure(db: Session, task: Any) -> None:
    """A step card back in the Inbox no longer shows the failure: the step waits to run again.
    F268: a step skipped for the failure shows Cancelled, which no mission sync reopens
    (F245); the owner never cancels a step alone, so the retry reopens it."""
    from core.models.core import BoardTask
    from services.orchestration_board_bridge import STEP_CARD_SOURCE_TYPE

    card = db.query(BoardTask).filter(BoardTask.source_type == STEP_CARD_SOURCE_TYPE,
                                      BoardTask.orchestration_task_id == task.id).first()
    if card is not None and card.status == "cancelled":
        card.status = "inbox"
    if card is not None and card.status == "inbox":
        card.error_message = None
        card.completed_at = None


def replaces_what_the_failure_skipped(replan: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``CoordinatorService.replan_mission``: once a replan stands, the steps the
    failure skipped are replaced too, and a failed mission's card is reopened."""
    @functools.wraps(replan)
    async def wrapped(self: Any, db: Session, run_id: Any, *args: Any, **kwargs: Any) -> Any:
        from core.models.orchestration_enums import RunState
        from services.mission_card_result import reopen_the_missions_card

        was_failed = RunState(self._get_run(db, run_id).state) == RunState.FAILED
        run = await replan(self, db, run_id, *args, **kwargs)
        _replaced_too(db, run)
        if was_failed:
            reopen_the_missions_card(db, run, note=REPLANNED_NOTE)
        return run
    return wrapped


def _replaced_too(db: Session, run: Any) -> int:
    """The steps skipped for the failure a replan re-did: replaced by it, as the failed
    step was, so they count for nothing when the mission ends."""
    from core.models.orchestration import OrchestrationTask
    from core.models.orchestration_enums import TaskState

    from modules.coordination.mission_ends import REPLACED_BY_REPLAN, skipped_for_a_failure

    skipped = [task for task in db.query(OrchestrationTask).filter(
        OrchestrationTask.run_id == run.id, OrchestrationTask.state == TaskState.SKIPPED.value).all()
        if skipped_for_a_failure(task)]
    for task in skipped:
        task.failure_reason_code = REPLACED_BY_REPLAN
        task.failure_detail = REPLACED_DETAIL.format(number=run.replan_count)
    if skipped:
        db.flush()
        logger.info("[F283] run %s: replan #%s replaced %d step(s) its failure had skipped",
                    run.id, run.replan_count, len(skipped))
    return len(skipped)


__all__ = ["NO_STEPS_TO_RESUME", "NothingToResume", "RETRY_REASON", "replaces_what_the_failure_skipped",
           "retries_a_failed_mission", "retry_failed", "runs_its_failed_steps_again"]
