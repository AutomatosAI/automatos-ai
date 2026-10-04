"""How a mission ends, once its steps are done (F268, night 7b).

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
"""
from __future__ import annotations

import functools
from typing import Any, Callable

from core.models.orchestration_enums import FailureReasonCode, TaskState

# coordinator_service.replan_mission writes this on the steps a re-plan replaced.
REPLACED_BY_REPLAN = "replaced_by_replan"


def superseded(task: Any) -> bool:
    """A step a re-plan replaced: skipped, and no longer part of the plan."""
    return task.state == TaskState.SKIPPED.value and task.failure_reason_code == REPLACED_BY_REPLAN


def skipped_for_a_failure(task: Any) -> bool:
    """A step skipped because another step failed for good: it never ran."""
    return (task.state == TaskState.SKIPPED.value
            and task.failure_reason_code == FailureReasonCode.DEPENDENCY_FAILED.value)


def counts_only_its_live_steps(advance: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``MissionReconciler._advance_run_on_completion``: it judges the mission by
    the steps of its plan, not the ones a re-plan replaced."""
    @functools.wraps(advance)
    def wrapped(db: Any, run: Any, all_tasks: Any, *args: Any, **kwargs: Any) -> Any:
        live = [task for task in all_tasks if not superseded(task)]
        return advance(db, run, live or all_tasks, *args, **kwargs)
    return wrapped


__all__ = ["REPLACED_BY_REPLAN", "counts_only_its_live_steps", "skipped_for_a_failure", "superseded"]
