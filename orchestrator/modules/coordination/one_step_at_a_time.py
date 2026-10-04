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
"""
from __future__ import annotations

import functools
from typing import Any, Callable, List

from sqlalchemy.orm import Session

ONE_AT_A_TIME = "one_step_at_a_time"


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


def a_step_is_busy(db: Session, run_id: Any) -> bool:
    """A step of the mission is assigned, running, done and not yet checked, or
    checked and waiting for the owner."""
    from core.models.orchestration import OrchestrationTask
    from core.models.orchestration_enums import TaskState

    busy = (TaskState.ASSIGNED.value, TaskState.RUNNING.value, TaskState.COMPLETED.value, TaskState.VERIFYING.value)
    return db.query(OrchestrationTask.id).filter(
        OrchestrationTask.run_id == run_id, OrchestrationTask.state.in_(busy)).first() is not None


__all__ = ["ONE_AT_A_TIME", "a_step_is_busy", "one_step_at_a_time"]
