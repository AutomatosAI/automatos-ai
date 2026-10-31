"""PRD-268 §6.2.1 — what a check-in reads: everything new since its last successful run.

OpenDuet's window rule: a check-in reads what changed since its last **successful** run,
else the last 24 hours. A failed run's reads are not trusted, so a run that errored, or
was skipped before it read anything, never moves the window on: the next run reads from
the last success again.

A run's window ends at ``window_until``, the moment it read up to. The next window starts
there, not at the run row's ``created_at``: a change made while a run was calling the
model is read by the next run, not lost between the two.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

__all__ = [
    "LOOKBACK",
    "SUCCESSFUL_OUTCOMES",
    "check_in_window",
    "last_success_at",
]

# With no successful run to start from, a check-in reads the last day.
LOOKBACK = timedelta(hours=24)

# The run outcomes that read and finished (core.models.proactive RUN_SILENT and
# RUN_SUGGESTED): an error's reads are not trusted, and a skipped run read nothing.
SUCCESSFUL_OUTCOMES = frozenset({"silent", "suggested"})


def _read_until(run: Any) -> Optional[datetime]:
    return run.window_until or run.created_at


def last_success_at(runs: Iterable[Any]) -> Optional[datetime]:
    """What the latest successful run read up to (its ``window_until``, else its
    ``created_at``), from runs with ``outcome`` in any order; None when none succeeded."""
    times = [_read_until(run) for run in runs if run.outcome in SUCCESSFUL_OUTCOMES]
    times = [moment for moment in times if moment is not None]
    return max(times) if times else None


def check_in_window(last_success_at: Optional[datetime], now: datetime) -> datetime:
    """Where a check-in's window starts: the last successful run, else ``now`` less a day.

    A last success stamped after ``now`` (another host's clock running ahead) starts the
    window at ``now``, so the window is empty rather than reversed. Both times are
    timezone-aware.
    """
    if last_success_at is None:
        return now - LOOKBACK
    return min(last_success_at, now)
