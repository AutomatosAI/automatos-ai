"""PRD-268 §6.2.2 — failing streaks: one card per streak, and a failing agent checks in less often.

A streak is one agent failing with one error class, run after run. Counted from the
agent's newest run back, a successful run ends it and so does a failure of another class;
a skipped run (quiet hours, a cap, the switches) read nothing and neither counts nor ends
it. The count comes from ``proactive_runs``, so no column holds it. Error classes compare
as the streak's key writes them (``error_class_name``): "Timeout" and "timeout" are one.

The interval stays the agent's own for the first two failures. From the third it doubles
on every failure, up to 24 hours, and the first success brings it back (the count is 0).
The 24 hours are the PRD's, whatever a workspace's interval cap: a failing agent still
checks in once a day.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable

from modules.proactive.dedupe import error_class_name
from modules.proactive.window import SUCCESSFUL_OUTCOMES

__all__ = [
    "DOUBLING_FROM",
    "FAILED",
    "MAX_STREAK_INTERVAL_MINUTES",
    "consecutive_failures",
    "is_due",
    "next_interval",
]

# The run outcome of a failure (core.models.proactive RUN_ERROR).
FAILED = "error"
# The failure, counted in a row, whose run doubles the interval first.
DOUBLING_FROM = 3
MAX_STREAK_INTERVAL_MINUTES = 24 * 60


def consecutive_failures(runs_newest_first: Iterable[Any], error_class: str) -> int:
    """How many of an agent's newest runs failed with ``error_class`` in a row.

    Runs carry ``outcome`` and ``error_class``, newest first. Counting stops at a
    successful run or a failure of another class; skipped runs are passed over.
    """
    wanted = error_class_name(error_class)
    count = 0
    for run in runs_newest_first:
        if run.outcome in SUCCESSFUL_OUTCOMES:
            break
        if run.outcome != FAILED:
            continue
        if error_class_name(run.error_class) != wanted:
            break
        count += 1
    return count


def next_interval(base_minutes: int, consecutive_failures: int) -> int:
    """The minutes until an agent's next check-in: ``base_minutes`` until the third failure
    in a row, then doubled per failure, up to 24 hours (or the base, when that is longer).
    No failures, as after a success, is the base: the reset."""
    if base_minutes <= 0 or consecutive_failures < 0:
        raise ValueError("the interval is positive and the failure count is not negative")
    ceiling = max(base_minutes, MAX_STREAK_INTERVAL_MINUTES)
    interval = base_minutes
    for _ in range(consecutive_failures - DOUBLING_FROM + 1):
        interval *= 2
        if interval >= ceiling:
            return ceiling
    return interval


def is_due(last_run_at: datetime, base_minutes: int, consecutive_failures: int, now: datetime) -> bool:
    """True when ``next_interval`` has passed since the agent's last run: a streak's later
    ticks are skipped until then.

    ``last_run_at`` is the agent's last run that was not skipped: a skipped tick never
    moves the clock on, or a streak's skips would keep the agent from ever being due.
    """
    return now - last_run_at >= timedelta(minutes=next_interval(base_minutes, consecutive_failures))
