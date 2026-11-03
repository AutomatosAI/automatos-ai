"""PRD-268 §6.3, K9 — when a follow-up may be, and the first cap one more would break.

A follow-up is Auto's word to a person: "I'll check again at 15:10". Its time comes from
``in_minutes`` or an ISO ``at`` that carries its offset; a time with no offset is refused,
since nobody can say whose 15:10 it is. ``follow_up_time`` gives it in UTC.

``FollowUpCaps`` holds the K9 defaults (``config.PROACTIVE_FOLLOW_UP_*`` sets the platform's,
``services.proactive.follow_ups.platform_caps``). ``first_broken`` names the first cap a
request breaks, in this order: sooner than ``min_minutes`` from now, later than ``max_days``,
``max_pending`` follow-ups already pending for the person from the agent,
``max_per_day`` scheduled for them by the agent in the last 24 hours (fired or not), and
``max_per_workspace_per_day`` in the workspace. The counts are read under the workspace's
lock by the caller, which inserts in the same transaction.

A follow-up may be about one thing, of a kind in ``ABOUT_KINDS``. The watcher already supervises a launched
Mission, a playbook run and a ticket to a verdict (``WATCHED_KINDS``): a follow-up about one
is refused with "use a watch" (§6.3).

Pure: no database, no clock (``now`` is the caller's).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

__all__ = [
    "ABOUT_KINDS",
    "BOTH_TIMES",
    "CapBroken",
    "FollowUpCaps",
    "FollowUpCounts",
    "FollowUpTimeRefused",
    "NAIVE_TIME",
    "NOT_A_TIME",
    "NO_TIME",
    "WATCHED_KINDS",
    "count_broken",
    "first_broken",
    "follow_up_time",
    "time_broken",
]

MIN_MINUTES = 10
MAX_DAYS = 7
MAX_PENDING = 5
MAX_PER_DAY = 20
MAX_PER_WORKSPACE_PER_DAY = 100
REASON_CHARS = 500
ABOUT_KINDS = ("mission", "playbook_run", "ticket", "other")
WATCHED_KINDS = frozenset({"mission", "playbook_run", "ticket"})

# Why a time could not be read: none given, both given, no offset, or not a time at all.
NO_TIME = "no_time"
BOTH_TIMES = "both_times"
NAIVE_TIME = "naive_time"
NOT_A_TIME = "not_a_time"
_ISO_Z = "Z"
_UTC_OFFSET = "+00:00"
# More digits than this is no number of minutes anyone means: refused, never overflowed.
MAX_DIGITS = 9


@dataclass(frozen=True)
class FollowUpCaps:
    """The §6.3 caps on Auto's follow-ups, and the longest reason one keeps."""

    min_minutes: int = MIN_MINUTES
    max_days: int = MAX_DAYS
    max_pending: int = MAX_PENDING
    max_per_day: int = MAX_PER_DAY
    max_per_workspace_per_day: int = MAX_PER_WORKSPACE_PER_DAY
    reason_chars: int = REASON_CHARS

    def __post_init__(self) -> None:
        """Refuse caps set wrong, before any request is checked against them."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"the follow-up cap {field.name} is a whole number of 0 or more, not {value!r}")
        if timedelta(minutes=self.min_minutes) > timedelta(days=self.max_days):
            raise ValueError("the earliest follow-up is later than the latest one")


@dataclass(frozen=True)
class FollowUpCounts:
    """What the caps count, read under the workspace's lock: the person's pending follow-ups
    from the agent, those the agent scheduled for them in the last 24 hours, and the
    workspace's in the last 24 hours."""

    pending: int = 0
    today: int = 0
    workspace_today: int = 0


@dataclass(frozen=True)
class CapBroken:
    """The cap a request breaks: its name (a ``FollowUpCaps`` field) and its value."""

    cap: str
    limit: int


class FollowUpTimeRefused(ValueError):
    """A follow-up time that cannot be read; ``code`` says why (``NO_TIME`` and the rest)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _minutes(value: Any) -> int:
    """``in_minutes`` as a whole number: an int, a whole float, or ASCII digits as text."""
    if isinstance(value, bool):
        raise FollowUpTimeRefused(NOT_A_TIME)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    said = value.strip() if isinstance(value, str) else ""
    if said.isascii() and said.isdigit() and len(said) <= MAX_DIGITS:
        return int(said)
    raise FollowUpTimeRefused(NOT_A_TIME)


def _at(value: Any) -> datetime:
    """``at`` as an aware time in UTC; a time with no offset is refused."""
    if not isinstance(value, str) or not value.strip():
        raise FollowUpTimeRefused(NOT_A_TIME)
    said = value.strip()
    try:
        at = datetime.fromisoformat(said[:-1] + _UTC_OFFSET if said.endswith(_ISO_Z) else said)
    except ValueError:
        raise FollowUpTimeRefused(NOT_A_TIME) from None
    if at.tzinfo is None or at.utcoffset() is None:
        raise FollowUpTimeRefused(NAIVE_TIME)
    return at.astimezone(timezone.utc)


def follow_up_time(in_minutes: Any, at: Any, now: datetime) -> datetime:
    """The follow-up's time in UTC, from exactly one of ``in_minutes`` (from ``now``) and an
    ISO ``at`` with its offset (an empty one is none). ``FollowUpTimeRefused`` when neither,
    both, or neither reads, a time past the calendar's ends included."""
    has_minutes, has_at = in_minutes not in (None, ""), at not in (None, "")
    if has_minutes and has_at:
        raise FollowUpTimeRefused(BOTH_TIMES)
    try:
        if has_minutes:
            return (now + timedelta(minutes=_minutes(in_minutes))).astimezone(timezone.utc)
        if has_at:
            return _at(at)
    except OverflowError:
        raise FollowUpTimeRefused(NOT_A_TIME) from None
    raise FollowUpTimeRefused(NO_TIME)


def time_broken(caps: FollowUpCaps, at: datetime, now: datetime) -> Optional[CapBroken]:
    """``min_minutes`` when ``at`` is sooner than that from ``now``, ``max_days`` when it is
    later than that; None when it is in between (both ends allowed)."""
    ahead = at - now
    if ahead < timedelta(minutes=caps.min_minutes):
        return CapBroken("min_minutes", caps.min_minutes)
    if ahead > timedelta(days=caps.max_days):
        return CapBroken("max_days", caps.max_days)
    return None


def count_broken(caps: FollowUpCaps, counts: FollowUpCounts) -> Optional[CapBroken]:
    """The first count cap one more follow-up would pass: pending, per day, per workspace."""
    checks = (("max_pending", counts.pending), ("max_per_day", counts.today),
              ("max_per_workspace_per_day", counts.workspace_today))
    for cap, count in checks:
        limit = getattr(caps, cap)
        if count >= limit:
            return CapBroken(cap, limit)
    return None


def first_broken(caps: FollowUpCaps, at: datetime, now: datetime, counts: FollowUpCounts) -> Optional[CapBroken]:
    """The first cap a follow-up at ``at`` breaks, given ``counts``; None when it fits them all."""
    return time_broken(caps, at, now) or count_broken(caps, counts)
