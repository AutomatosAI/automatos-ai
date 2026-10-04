"""A mission that checks each step runs one step at a time (F267, night 7b).

#0188 waited for the owner's check of every step (``approval_mode: step_by_step``), but
its steps with no dependencies started side by side, so two of them waited in Review
at once. The owner sent #0188.2 back, and its redo did not run for 2 min 10 s: the
mission stays paused while any step waits for the owner (``owner_checks.carry_on``),
and #0188.3 still waited, until the owner decided it too. #0176.10 and #0176.11 waited
2 min 45 s and 2 min 28 s the same way.

``one_step_at_a_time`` wraps the dispatcher. A mission that checks each step
(``owner_checks.checks_each_step``) runs one step at a time (``max_concurrent`` 1),
and starts nothing while a step is busy: assigned, running, done and not yet checked,
or waiting for the owner. So only one step ever waits for the owner, a step sent back
resumes its mission at once, and the next tick (seconds) runs its redo. Any other
mission is dispatched as before.

Night 8: the owner of #0383 had said "don't move on until I've said it's right", and
#0383.1 started while #0383.2, sent back, waited for its redo: a step being redone
(RETRYING) was not busy, and a step that had waited in the queue went first.
``waits_its_turn`` wraps the dispatch of each step: while a step is being redone, no
other step starts, and the redo goes first. A step marked as waiting for the owner's
check (``owner_checks.WAITING_KEY``) holds the others too.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, List

from sqlalchemy.orm import Session

from core.models.orchestration_enums import TaskState

ONE_AT_A_TIME = "one_step_at_a_time"
# A step every other step waits for: assigned, running, done and not yet checked, or
# being checked (a step that waits for the owner's check waits here).
BUSY = (TaskState.ASSIGNED.value, TaskState.RUNNING.value, TaskState.COMPLETED.value, TaskState.VERIFYING.value)
# A step sent back to be redone: it alone may start until it is busy again.
REDO = TaskState.RETRYING.value
# A step waiting to run, whose mark of waiting for the owner (left by an earlier round) holds nothing.
WAITING_TO_RUN = (TaskState.PENDING.value, TaskState.QUEUED.value, REDO)
# A step that has ended holds nothing.
ENDED = (TaskState.VERIFIED.value, TaskState.FAILED.value, TaskState.SKIPPED.value)


def one_step_at_a_time(dispatch: Callable[..., List[Any]]) -> Callable[..., List[Any]]:
    """Wrap ``MissionDispatcher.dispatch_ready`` (see the module)."""
    @functools.wraps(dispatch)
    def wrapped(db: Session, run: Any, agents: Any) -> List[Any]:
        from modules.coordination.owner_checks import checks_each_step

        if not checks_each_step(getattr(run, "config", None)):
            return dispatch(db, run, agents)
        if (run.max_concurrent or 1) != 1:
            run.max_concurrent = 1      # the mission's own setting from now on: one step at a time
        if a_step_is_busy(db, run.id):
            from modules.coordination.dispatcher import DispatchResult

            return [DispatchResult(dispatched=False, skipped_reason=ONE_AT_A_TIME)]
        return dispatch(db, run, agents)
    return wrapped


def waits_its_turn(dispatch_single: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``MissionDispatcher._dispatch_single``: in a mission that checks each step,
    a step starts only when no other step holds it back (see the module)."""
    @functools.wraps(dispatch_single)
    def wrapped(db: Session, run: Any, task: Any, agents: Any) -> Any:
        from modules.coordination.owner_checks import checks_each_step

        if checks_each_step(getattr(run, "config", None)) and another_step_goes_first(db, run.id, task.id):
            from modules.coordination.dispatcher import DispatchResult

            return DispatchResult(dispatched=False, task_id=task.id, skipped_reason=ONE_AT_A_TIME)
        return dispatch_single(db, run, task, agents)
    return wrapped


def a_step_is_busy(db: Session, run_id: Any) -> bool:
    """A step of the mission is assigned, running, done and not yet checked, or
    checked and waiting for the owner."""
    return any(_holds_the_others(state, context) for _, state, context in _open_steps(db, run_id))


def another_step_goes_first(db: Session, run_id: Any, task_id: Any) -> bool:
    """Another step of the mission holds step ``task_id`` back: one that is busy or
    waits for the owner's check, or one being redone while this one is not."""
    steps = _open_steps(db, run_id)
    this_is_a_redo = any(step_id == task_id and state == REDO for step_id, state, _ in steps)
    return any(_holds_the_others(state, context) or (state == REDO and not this_is_a_redo)
               for step_id, state, context in steps if step_id != task_id)


def _holds_the_others(state: str, context: Any) -> bool:
    """Whether a step every other step waits for: busy, or marked as waiting for the
    owner's check and not back to waiting to run."""
    from modules.coordination.owner_checks import WAITING_KEY

    if state in BUSY:
        return True
    return isinstance(context, dict) and bool(context.get(WAITING_KEY)) and state not in WAITING_TO_RUN


def _open_steps(db: Session, run_id: Any) -> List[Any]:
    """``(id, state, input_context)`` of each step of the mission that has not ended,
    as the database has them now (a step loaded earlier in the tick may be stale)."""
    from core.models.orchestration import OrchestrationTask

    return db.query(OrchestrationTask.id, OrchestrationTask.state, OrchestrationTask.input_context).filter(
        OrchestrationTask.run_id == run_id, OrchestrationTask.state.notin_(ENDED)).all()


__all__ = ["ONE_AT_A_TIME", "a_step_is_busy", "another_step_goes_first", "one_step_at_a_time", "waits_its_turn"]
