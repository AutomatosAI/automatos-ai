"""PRD-268 §6.9 — the caps that bound what proactive work costs and how much of it reaches a person.

``Caps`` holds the §6.9 defaults; a workspace's own values (``settings.proactive``, owners
and admins) replace them field by field with ``dataclasses.replace``, which refuses a cap
that is not a whole number of 0 or more, and an interval range that is empty. 0 is a cap
that allows nothing.

Each check takes the counts its caller read and returns None when the work may go ahead,
else a ``CapRefusal`` naming the cap: the runner records its ``outcome`` as the run's, and
the suggestion tool tells the model which cap it reached.

The open-per-person cap is not a refusal: the oldest open suggestions expire to make
room (``oldest_to_expire``).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable, Optional, Tuple

__all__ = [
    "CAPPED",
    "CapRefusal",
    "Caps",
    "clamp_interval",
    "oldest_to_expire",
    "run_refusal",
    "suggestion_refusal",
]

# The run outcome of a refusal (core.models.proactive RUN_SKIPPED_CAP).
CAPPED = "skipped_cap"

MIN_INTERVAL_MINUTES = 30
MAX_INTERVAL_MINUTES = 24 * 60
MAX_MODEL_RUNS_PER_DAY = 48
MAX_SUGGESTIONS_PER_RUN = 3
MAX_NEW_PER_PERSON_PER_DAY = 10
MAX_OPEN_PER_PERSON = 20
MAX_OPEN_PER_AGENT = 3
MAX_PUSHES_PER_PERSON_PER_DAY = 2


@dataclass(frozen=True)
class Caps:
    """The §6.9 caps, one workspace's: the defaults unless its owner set its own."""

    # A check-in's interval, per agent; a migrated heartbeat keeps its own within these.
    min_interval_minutes: int = MIN_INTERVAL_MINUTES
    max_interval_minutes: int = MAX_INTERVAL_MINUTES
    # Check-in runs that reach the model, per workspace per day: a silent run doesn't count.
    max_model_runs_per_day: int = MAX_MODEL_RUNS_PER_DAY
    max_suggestions_per_run: int = MAX_SUGGESTIONS_PER_RUN
    max_new_per_person_per_day: int = MAX_NEW_PER_PERSON_PER_DAY
    # Past this the oldest open suggestions expire; nothing is refused.
    max_open_per_person: int = MAX_OPEN_PER_PERSON
    max_open_per_agent: int = MAX_OPEN_PER_AGENT
    # Read by the suggestion push (W2).
    max_pushes_per_person_per_day: int = MAX_PUSHES_PER_PERSON_PER_DAY

    def __post_init__(self) -> None:
        """Refuse a cap a workspace's settings got wrong, before anything is checked against it."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"the cap {field.name} is a whole number of 0 or more, not {value!r}")
        if not 0 < self.min_interval_minutes <= self.max_interval_minutes:
            raise ValueError("the check-in interval runs from at least a minute up to its maximum")


@dataclass(frozen=True)
class CapRefusal:
    """Work a cap refused: the run outcome to record, the cap's name, its limit and the count."""

    cap: str
    limit: int
    count: int
    outcome: str = CAPPED


def _reached(caps: Caps, cap: str, count: int) -> Optional[CapRefusal]:
    limit = getattr(caps, cap)
    return CapRefusal(cap=cap, limit=limit, count=count) if count >= limit else None


def run_refusal(caps: Caps, model_runs_today: int) -> Optional[CapRefusal]:
    """None when a check-in may call the model, given the workspace's runs that reached it
    today; else the refusal of ``max_model_runs_per_day``."""
    return _reached(caps, "max_model_runs_per_day", model_runs_today)


def suggestion_refusal(caps: Caps, *, made_this_run: int, new_today_for_person: int,
                       open_for_agent: int) -> Optional[CapRefusal]:
    """None when one more suggestion may be made, else the first cap it would pass: per
    run, then new per person per day, then open per source agent (pass 0 for none)."""
    checks = (
        ("max_suggestions_per_run", made_this_run),
        ("max_new_per_person_per_day", new_today_for_person),
        ("max_open_per_agent", open_for_agent),
    )
    for cap, count in checks:
        refusal = _reached(caps, cap, count)
        if refusal is not None:
            return refusal
    return None


def oldest_to_expire(caps: Caps, open_rows: Iterable[Any]) -> Tuple[Any, ...]:
    """The person's open suggestions to expire, oldest ``created_at`` first, so that one
    more fits under ``max_open_per_person``; empty when there is room."""
    rows = sorted(open_rows, key=lambda row: row.created_at)
    excess = len(rows) - caps.max_open_per_person + 1
    return tuple(rows[:excess]) if excess > 0 else ()


def clamp_interval(caps: Caps, minutes: int) -> int:
    """A check-in interval within the caps' bounds: 30 minutes to 24 hours by default."""
    return max(caps.min_interval_minutes, min(int(minutes), caps.max_interval_minutes))
