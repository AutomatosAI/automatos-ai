"""PRD-251C Wave 2 (C1; US-C203): a weekly or monthly plan makes its batch at once.

A plan's rhythm (``make.rhythm``, ``plans.validate_make``):

* ``daily`` keeps PRD-251B's timing: each post is made on its day (``plans.make_at``).
* ``weekly``: at ``make.batch_day`` and ``make.time`` (the plan's timezone), every slot of the
  7 days from the next midnight is made: the week's batch, keyed by the ISO week of its first
  day ("2026-W42").
* ``monthly``: at ``make.batch_date`` and ``make.time``, every slot of the next calendar month:
  "2026-11".

A slot is due once the moment of the batch that holds it has come, while its own time has not,
nothing holds its key, and the batch did not skip it. So a slot that appears after its batch
was made (a new row, a moved slot, a plan saved mid-week) is made at the next tick, as a daily
slot is; a slot moved before its batch's moment is made now. The tick's per-tick cap holds: a
large batch is made over several ticks.

The plan records its batches in ``make.batches``: ``{key: {"skipped": {slot key: why},
"announced": ISO}}``. A slot the tick could not make (no connected channel posts its format,
no render minutes left) is skipped and recorded, never half-made; once every slot of a batch
is made or skipped, "Your week is ready" goes out once. A slot whose own time came while the
tick had not made it (an outage, a backlog across every plan) is no longer made: when the
batch is announced it is recorded as skipped and the notice names it (``passed_unmade``).
A record is dropped ``KEEP_BATCH_DAYS`` after its batch began. All of it is pure: the tick
reads and writes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from modules.socials import plans

BATCHES = "batches"
SKIPPED = "skipped"
ANNOUNCED = "announced"
WEEK_DAYS = 7
KEEP_BATCH_DAYS = 70
_FIRST_OF_MONTH = 1


@dataclass(frozen=True)
class Window:
    """The local days a batch covers, [start, end), and when it is made (UTC)."""

    start: date
    end: date
    moment: datetime
    key: str


def rhythm_of(plan: Any) -> str:
    return plans.make_settings(plan)["rhythm"]


def is_batched(plan: Any) -> bool:
    return rhythm_of(plan) != plans.DAILY


def _next_month(day: date) -> date:
    return (day.replace(day=28) + timedelta(days=4)).replace(day=_FIRST_OF_MONTH)


def _previous_month(day: date) -> date:
    return (day.replace(day=_FIRST_OF_MONTH) - timedelta(days=1)).replace(day=_FIRST_OF_MONTH)


def _window_start(settings: Dict[str, Any], day: date) -> date:
    if settings["rhythm"] == plans.MONTHLY:
        return day.replace(day=_FIRST_OF_MONTH)
    first_weekday = (plans.WEEKDAYS.index(settings["batch_day"]) + 1) % WEEK_DAYS
    return day - timedelta(days=(day.weekday() - first_weekday) % WEEK_DAYS)


def _key(settings: Dict[str, Any], start: date) -> str:
    if settings["rhythm"] == plans.MONTHLY:
        return f"{start.year}-{start.month:02d}"
    iso = start.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def window_of(plan: Any, day: date) -> Optional[Window]:
    """The batch window holding local ``day``; None for a daily plan."""
    settings = plans.make_settings(plan)
    if settings["rhythm"] == plans.DAILY:
        return None
    start = _window_start(settings, day)
    if settings["rhythm"] == plans.MONTHLY:
        end = _next_month(start)
        made_on = _previous_month(start).replace(day=settings["batch_date"])
    else:
        end, made_on = start + timedelta(days=WEEK_DAYS), start - timedelta(days=1)
    moment = plans.local_to_utc(made_on, settings["time"], plans.zone_of(plan))
    return Window(start=start, end=end, moment=moment, key=_key(settings, start))


def batch_key(plan: Any, day: date) -> Optional[str]:
    """The key of the batch a slot on local ``day`` belongs to; None for a daily plan."""
    window = window_of(plan, day)
    return window.key if window is not None else None


def made_through(plan: Any, now: datetime) -> Optional[Window]:
    """The latest batch window whose moment has come by ``now`` (its days and everything
    before them are the plan's made-ahead span); None for a daily plan."""
    today = now.astimezone(plans.zone_of(plan)).date()
    current = window_of(plan, today)
    if current is None:
        return None
    following = window_of(plan, current.end)
    return following if following.moment <= now else current


def next_window(plan: Any, now: datetime) -> Optional[Window]:
    """The first batch still to be made after ``now`` that holds a day of the plan; None for a
    daily plan or one whose dates hold no later batch."""
    latest = made_through(plan, now)
    starts_on, ends_on = getattr(plan, "starts_on", None), getattr(plan, "ends_on", None)
    if latest is None or starts_on is None or ends_on is None:
        return None
    upcoming = window_of(plan, max(latest.end, starts_on))
    return upcoming if upcoming.start <= ends_on else None


def label(plan: Any, window: Window) -> str:
    """How a notice names the batch: "the week of 12 Oct" or "November 2026"."""
    if rhythm_of(plan) == plans.MONTHLY:
        return f"{window.start:%B %Y}"
    return f"the week of {window.start.day} {window.start:%b}"


# ── the records ────────────────────────────────────────────────────────────


def records(plan: Any) -> Dict[str, Dict[str, Any]]:
    found = (getattr(plan, "make", None) or {}).get(BATCHES)
    return found if isinstance(found, dict) else {}


def skipped_keys(plan: Any) -> Set[str]:
    """Every slot a batch skipped, across the plan's batches."""
    return {key for record in records(plan).values() if isinstance(record, dict) for key in (record.get(SKIPPED) or {})}


def _start_of_key(key: str) -> Optional[date]:
    try:
        if "-W" in key:
            year, week = key.split("-W")
            return date.fromisocalendar(int(year), int(week), 1)
        year, month = key.split("-")
        return date(int(year), int(month), _FIRST_OF_MONTH)
    except ValueError:
        return None


def _kept(found: Dict[str, Any], today: date) -> Dict[str, Any]:
    """The batch records still worth keeping: those that began less than KEEP_BATCH_DAYS ago."""
    floor = today - timedelta(days=KEEP_BATCH_DAYS)
    return {key: record for key, record in found.items() if (_start_of_key(key) or today) >= floor}


def with_record(plan: Any, key: str, change: Dict[str, Any], today: date) -> Dict[str, Any]:
    """The plan's ``make`` with batch ``key``'s record updated by ``change``: a new object."""
    make = dict(getattr(plan, "make", None) or {})
    found = _kept(records(plan), today)
    current = dict(found.get(key) or {})
    return {**make, BATCHES: {**found, key: {**current, **change}}}


def with_skip(plan: Any, key: str, slot_key: str, why: str, today: date) -> Dict[str, Any]:
    return with_skips(plan, key, (slot_key,), why, today)


def with_skips(plan: Any, key: str, slot_keys: Iterable[str], why: str, today: date) -> Dict[str, Any]:
    """The plan's ``make`` with ``slot_keys`` recorded as skipped by batch ``key``, for ``why``."""
    skipped = dict((records(plan).get(key) or {}).get(SKIPPED) or {})
    return with_record(plan, key, {SKIPPED: {**skipped, **{slot_key: why for slot_key in slot_keys}}}, today)


def announced(plan: Any, key: str) -> bool:
    return bool((records(plan).get(key) or {}).get(ANNOUNCED))


# ── what is due ────────────────────────────────────────────────────────────


def due_slots(plan: Any, now: datetime, taken: Iterable[str]) -> List[plans.Slot]:
    """A batched plan's slots to make now (the module docstring), earliest first."""
    through = made_through(plan, now)
    if through is None:
        return []
    horizon = plans.local_to_utc(through.end, "00:00", plans.zone_of(plan))
    done = {*taken, *skipped_keys(plan)}
    reach = horizon + timedelta(days=plans.MAX_MOVE_DAYS + 1)
    return [
        slot for slot in plans.expand_slots(plan, now, reach)
        if slot.key not in done and (slot.local_date < through.end or slot.at < horizon)
    ]


def _utc(moment: Optional[datetime]) -> Optional[datetime]:
    """A stored time as UTC (SQLite gives it back without a zone)."""
    if moment is None:
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def passed_unmade(plan: Any, key: str, now: datetime, taken: Iterable[str]) -> List[plans.Slot]:
    """The slots of batch ``key`` whose own time came while nothing held them and the batch had
    not skipped them: due once (after the batch's moment, and after the plan was made), and the
    tick fell behind. Never made now: the batch's notice names them."""
    start = _start_of_key(key)
    window = window_of(plan, start) if start is not None else None
    if window is None:
        return []
    created = _utc(getattr(plan, "created_at", None))
    since = max(window.moment, created) if created is not None else window.moment
    done = {*taken, *skipped_keys(plan)}
    return [slot for slot in plans.expand_slots(plan, since, now) if slot.key not in done and batch_key(plan, slot.local_date) == key]


def pending(plan: Any, key: str, now: datetime, taken: Iterable[str]) -> List[plans.Slot]:
    """The slots of batch ``key`` still to make: their time has not come, nothing holds them
    and the batch did not skip them."""
    done = {*taken, *skipped_keys(plan)}
    return [slot for slot in due_slots(plan, now, ()) if slot.key not in done and batch_key(plan, slot.local_date) == key]
