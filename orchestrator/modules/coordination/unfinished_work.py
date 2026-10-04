"""F248 (night 7): a mission step that still holds a template's placeholders after
its revision fails. It never ends "verified".

Step #0126.3 was marked verified with "[Number]" and "[Your Name/Company Name]" in
it. Mission verification is advisory: a FAIL is revised once, and a FAIL after that
revision passes through to VERIFIED with its feedback noted. Placeholders are the
exception, because a step with slots left in has not done its work. The failure
names the slots, so the owner sees what was missing and can re-plan.

F286 (night 8): so is an answer that is still not the work after its revision: a note
the agent left itself, or an answer that says it could not do the work
(``non_answers``). #0400.2 was verified with "This information is still missing." and
its mission "completed" with that sentence as its result. Now the step fails, quoting
it, and its mission fails saying so (F283, ``mission_ends``).

F283 (night 8): #0352.2 failed the check after its one revision, "attempt 1 of 3",
and nothing ever ran it again. Decision: a step whose work the check finds unfinished
gets the rest of its attempts (``max_retries``, the budget an agent error gets), each
a revision of its own last answer with the check's words, and fails for good only
when they are spent (``redo_or_fail_unfinished``). Why: the attempts are the step's
own budget and the step says so; a revision with the exact failure ("[Cafe Name]")
is the cheapest way to finish (it revises, it does not start again), and a slip fixed
there costs the owner nothing, where a failed step stops the whole mission until the
owner acts. The budget bounds what a step that cannot finish (Gmail not connected)
costs: one attempt more than before at most, with the default of 3. Any other FAIL
keeps PRD-200's single revision and then passes as advisory, as before.
"""
from __future__ import annotations

import logging
from typing import Any, List

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

RETRY_REASON = "The mission's check found the work unfinished, attempt {attempt} of {limit}: {why}"
FAILED_REASON = "The mission's check found the work unfinished after {attempts} attempts: {why}"
DETAIL_CHARS = 2000
REASON_CHARS = 300


def unfinished_failures(result: Any) -> List[str]:
    """The verdict's deterministic failures that say the work is not done: placeholders
    left in it, or an answer that is not the work."""
    from core.services.placeholders import UNFINISHED
    from modules.coordination.non_answers import NOT_THE_WORK

    unfinished = (UNFINISHED, *NOT_THE_WORK)
    return [f for f in getattr(result, "deterministic_failures", None) or [] if str(f).startswith(unfinished)]


def redo_or_fail_unfinished(db: Session, task: Any, result: Any, failures: List[str]) -> None:
    """The step is revised again with the check's words while it has attempts left,
    else it fails for good, saying what is not done."""
    from config import Config
    from core.models.orchestration_enums import FailureReasonCode

    detail = " ".join(failures)[:DETAIL_CHARS]
    attempt = (task.attempt_number or 0) + 1
    limit = task.max_retries or Config.COORDINATOR_MAX_TASK_RETRIES
    task.failure_reason_code = FailureReasonCode.VERIFICATION_FAIL.value
    task.attempt_number = attempt
    if attempt < limit:
        _revise_again(db, task, result, detail, limit)
    else:
        _fail(db, task, detail)


def _revise_again(db: Session, task: Any, result: Any, detail: str, limit: int) -> None:
    """Back to its agent as a revision of its last answer, with the check's words (the
    keys ``MissionDispatcher.build_task_prompt`` reads)."""
    from core.models.orchestration_enums import ActorType, TaskState
    from services.orchestration_board_bridge import sync_board_status
    from services.orchestration_state import transition_task

    task.input_context = {
        **(task.input_context or {}),
        "previous_output": task.output or "",
        "verification_feedback": {"attempt": task.attempt_number, "reasoning": getattr(result, "reasoning", "") or detail,
                                  "scores": getattr(result, "scores", None) or {},
                                  "failures": list(getattr(result, "deterministic_failures", None) or [])},
    }
    reason = RETRY_REASON.format(attempt=task.attempt_number + 1, limit=limit, why=detail)[:REASON_CHARS]
    transition_task(db=db, task=task, new_state=TaskState.RETRYING, actor_type=ActorType.COORDINATOR,
                    actor_id="reconciler", reason=reason)
    sync_board_status(db, task)
    logger.info("Task %s: the work is not done, revised again (%s)", task.id, reason)


def _fail(db: Session, task: Any, detail: str) -> None:
    """The step fails for good, saying what is not done."""
    from core.models.orchestration_enums import ActorType, TaskState
    from services.orchestration_board_bridge import sync_board_status
    from services.orchestration_state import transition_task

    task.failure_detail = detail
    reason = FAILED_REASON.format(attempts=task.attempt_number, why=detail)[:REASON_CHARS]
    transition_task(db=db, task=task, new_state=TaskState.FAILED, actor_type=ActorType.COORDINATOR,
                    actor_id="reconciler", reason=reason)
    sync_board_status(db, task)
    logger.warning("Task %s failed verification: the work is not done after its attempts (%s)", task.id, detail[:200])


__all__ = ["redo_or_fail_unfinished", "unfinished_failures"]
