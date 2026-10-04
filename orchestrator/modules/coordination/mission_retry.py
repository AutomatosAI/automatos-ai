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
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, List

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

RETRY_REASON = "Retry: the failed steps run again, with the same plan"
LEDGER_KEY = "progress_ledger"


def retries_a_failed_mission(resume: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``CoordinatorService.resume_mission``: a failed mission is made ready to
    run again (``retry_failed``) and then resumed; any other is resumed as before."""
    @functools.wraps(resume)
    def wrapped(self: Any, db: Session, run_id: Any, actor_id: str) -> Any:
        from core.models.orchestration_enums import RunState

        run = self._get_run(db, run_id)
        if RunState(run.state) == RunState.FAILED:
            retry_failed(db, run, actor_id)
        return resume(self, db, run_id, actor_id)
    return wrapped


def retry_failed(db: Session, run: Any, actor_id: str) -> List[Any]:
    """Turn a failed ``run`` into a paused one whose failed steps wait to run again.
    Returns those steps. Resume runs it."""
    from core.models.orchestration import OrchestrationTask
    from core.models.orchestration_enums import ActorType, RunState, TaskState
    from services.orchestration_state import transition_run

    from modules.coordination.mission_ends import skipped_for_a_failure

    # F268 (night 7b): the steps skipped because a step failed run again too; #0176
    # retried its four failed steps, and its eight skipped ones failed it again.
    steps = db.query(OrchestrationTask).filter(
        OrchestrationTask.run_id == run.id,
        OrchestrationTask.state.in_([TaskState.FAILED.value, TaskState.SKIPPED.value]),
    ).order_by(OrchestrationTask.sequence_number).all()
    failed = [t for t in steps if t.state == TaskState.FAILED.value or skipped_for_a_failure(t)]
    transition_run(db=db, run=run, new_state=RunState.PAUSED, actor_type=ActorType.HUMAN,
                   actor_id=actor_id, reason=RETRY_REASON)
    run.completed_at = None
    run.config = {key: value for key, value in (run.config or {}).items() if key != LEDGER_KEY}
    for task in failed:
        _wait_again(db, task, actor_id)
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


__all__ = ["RETRY_REASON", "retries_a_failed_mission", "retry_failed"]
