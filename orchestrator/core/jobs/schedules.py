"""The scheduler queues; it never runs (PRD-276 §6.4, Wave 2b).

A timed piece of work is declared once with ``schedule(scheduler, kind, schedule_id, trigger)``.
APScheduler is then only a timer: each time it fires, ``tick`` runs, which

* locks the schedule's cursor (``jobs.job_schedules``: the last slot it queued);
* works out every slot of the trigger since that cursor, up to now;
* queues them by the kind's catch-up policy: ``all`` (every missed slot, at most
  ``JOBS_CATCH_UP_MAX``, the newest kept and the rest logged and recorded as dropped), ``latest``
  (the most recent only) or ``none`` (only a slot that is due right now);
* moves the cursor past them, in the same transaction, so a slot is queued and recorded, or neither.

Each slot's job is keyed ``<schedule_id>:<slot>``, so a second scheduler firing the same slot, while
the scheduler lock briefly slips, queues nothing more. The slot is the time the tick is *for*, not
when it ran, and the handler finds it in its payload as ``slot``. Interval triggers are anchored
(``interval``) so every process computes the same slots.

When it is declared, a schedule also ticks once at once: the start-up catch-up for slots missed
while no scheduler ran. A brand-new schedule's cursor starts at that moment, so it never backfills.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from core.jobs.registry import CATCH_UP_ALL, CATCH_UP_LATEST, kind_of

logger = logging.getLogger(__name__)

PLATFORM = "platform"  # the schedule_id of a platform-wide schedule, such as metering
ANCHOR = datetime(2026, 1, 1, tzinfo=timezone.utc)  # every interval trigger counts from here
ON_TIME_SECONDS = 60  # under "none", a slot this recent is the one firing now, not a missed one
# A job's key is "<schedule_id>:<slot>" in a 255-character column; a slot's ISO time is at most 32.
MAX_SCHEDULE_ID = 200

# A share lock on a live workspace's row, held to the tick's commit: closing the workspace (its
# soft delete) waits for a tick in progress, so no tick seeds a cursor or queues a slot after it.
_LIVE = text("SELECT 1 FROM workspaces WHERE id = CAST(:ws AS uuid) AND deleted_at IS NULL FOR SHARE")

_SEED = text("INSERT INTO jobs.job_schedules (kind, schedule_id, workspace_id, last_slot) "
             "VALUES (:kind, :sid, CAST(:ws AS uuid), :now) ON CONFLICT (kind, schedule_id) DO NOTHING")
_LOCK = text("SELECT last_slot FROM jobs.job_schedules WHERE kind = :kind AND schedule_id = :sid FOR UPDATE")
_MOVE = text("UPDATE jobs.job_schedules SET last_slot = :slot, updated_at = clock_timestamp() "
             "WHERE kind = :kind AND schedule_id = :sid")


@dataclass(frozen=True)
class Schedule:
    """One declared schedule: what it queues, for whom, and when."""

    kind: str
    schedule_id: str
    trigger: Any
    workspace_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.schedule_id) > MAX_SCHEDULE_ID:
            raise ValueError(f"schedule_id is {len(self.schedule_id)} characters; at most {MAX_SCHEDULE_ID}, "
                             "so its slots' job keys fit")


_SCHEDULES: Dict[Tuple[str, str], Schedule] = {}


def interval(**every: Any) -> Any:
    """An interval trigger counted from ``ANCHOR``, in UTC, so every process has the same slots."""
    from apscheduler.triggers.interval import IntervalTrigger

    return IntervalTrigger(start_date=ANCHOR, timezone=timezone.utc, **every)


def due_slots(trigger: Any, after: datetime, until: datetime, keep: int) -> Tuple[List[datetime], int]:
    """The newest ``keep`` slots of ``trigger`` after ``after`` and no later than ``until`` (oldest
    first), and how many such slots there are in all. Every slot is counted, however long the
    backlog, so the last one kept is always the latest; only ``keep`` are held at once."""
    kept: deque = deque(maxlen=max(1, keep))
    total, moment = 0, after + timedelta(microseconds=1)
    while True:
        slot = trigger.get_next_fire_time(None, moment)
        if slot is None or slot > until:
            return list(kept), total
        kept.append(slot)
        total += 1
        moment = slot + timedelta(microseconds=1)


def chosen(policy: str, slots: List[datetime], total: int, now: datetime, cap: int) -> Tuple[List[datetime], int]:
    """The slots ``policy`` queues from the newest due ``slots`` (of ``total`` due), and how many "all"
    had to drop past its cap (at least 1, whatever ``JOBS_CATCH_UP_MAX`` says)."""
    if not slots:
        return [], 0
    if policy == CATCH_UP_ALL:
        queue = slots[-max(1, cap):]
        return queue, total - len(queue)
    if policy == CATCH_UP_LATEST:
        return slots[-1:], 0
    return [s for s in slots[-1:] if (now - s).total_seconds() <= ON_TIME_SECONDS], 0


def tick(db: Any, spec: Schedule, now: Optional[datetime] = None) -> int:
    """Queues the schedule's due slots by its kind's policy and moves its cursor, in one
    transaction; answers how many jobs went in (a slot already queued, by a second scheduler,
    adds none). A schedule of a workspace that is closed or gone does nothing: no cursor is made
    again after a purge deleted it, and no job is queued for it (§6.8). A live workspace's row is
    share-locked to the commit, so it can't be closed in the middle of a tick. The cursor's row
    lock makes two overlapping ticks of one schedule take turns. Commits."""
    from config import config
    from core.jobs.queue import enqueue

    if spec.workspace_id and db.execute(_LIVE, {"ws": spec.workspace_id}).first() is None:
        return 0
    now = now or datetime.now(timezone.utc)
    cap = max(1, int(config.JOBS_CATCH_UP_MAX))
    params = {"kind": spec.kind, "sid": spec.schedule_id}
    db.execute(_SEED, {**params, "ws": spec.workspace_id, "now": now})
    last_slot = db.execute(_LOCK, params).scalar()
    slots, total = due_slots(spec.trigger, last_slot, now, cap)
    queue, dropped = chosen(kind_of(spec.kind).catch_up, slots, total, now, cap)
    queued = 0
    for slot in queue:
        stamp = slot.isoformat()
        queued += enqueue(db, spec.kind, {**spec.payload, "slot": stamp}, workspace_id=spec.workspace_id,
                          key=f"{spec.schedule_id}:{stamp}") is not None
    if slots:
        db.execute(_MOVE, {**params, "slot": slots[-1]})
    db.commit()
    if dropped:
        _report_dropped(spec, dropped)
    return queued


def _report_dropped(spec: Schedule, dropped: int) -> None:
    from core.utils.exception_telemetry import record_error

    logger.warning("[jobs] schedule %s of %s missed %d slot(s) past the catch-up cap; they were skipped",
                   spec.schedule_id, spec.kind, dropped)
    record_error(subsystem="jobs", operation="catch_up", workspace_id=spec.workspace_id,
                 error=RuntimeError(f"{dropped} missed slot(s) of {spec.kind} skipped past the catch-up cap"),
                 extra={"schedule_id": spec.schedule_id, "kind": spec.kind, "dropped": dropped})


def job_id(kind: str, schedule_id: str) -> str:
    """The APScheduler job that wakes this schedule."""
    return f"schedule:{kind}:{schedule_id}"


def schedule(scheduler: Any, kind: str, schedule_id: str, trigger: Any, *, workspace_id: Any = None,
             payload: Optional[Dict[str, Any]] = None) -> Schedule:
    """Declares a schedule on ``scheduler``: its trigger wakes ``tick``, which queues ``kind``'s jobs.
    It also ticks once at once, for the slots missed while no scheduler ran. Replaces an earlier
    declaration of the same (kind, schedule_id)."""
    kind_of(kind)  # a kind nobody registered is refused here, not at its first slot
    spec = Schedule(kind, schedule_id, trigger, str(workspace_id) if workspace_id else None, dict(payload or {}))
    _SCHEDULES[(kind, schedule_id)] = spec
    common = {"args": [kind, schedule_id], "replace_existing": True, "max_instances": 1}
    scheduler.add_job(fire, trigger, id=job_id(kind, schedule_id), coalesce=True, misfire_grace_time=None, **common)
    # However late the catch-up starts (a busy boot), it runs: APScheduler's default grace would skip it.
    scheduler.add_job(fire, "date", id=f"{job_id(kind, schedule_id)}:catch-up", misfire_grace_time=None, **common)
    return spec


async def fire(kind: str, schedule_id: str) -> int:
    """APScheduler's job: one ``tick`` of a declared schedule, off the event loop."""
    from core.jobs.worker import on_own_session

    spec = _SCHEDULES.get((kind, schedule_id))
    if spec is None:
        logger.warning("[jobs] schedule %s of %s fired but is no longer declared", schedule_id, kind)
        return 0
    try:
        return await asyncio.to_thread(on_own_session, tick, spec)
    except Exception:  # noqa: BLE001 — logged; the cursor didn't move, so the next tick queues this slot too
        logger.exception("[jobs] schedule %s of %s could not queue its slot", schedule_id, kind)
        return 0


def declared(kind: str, schedule_id: str) -> Optional[Schedule]:
    """A declared schedule, or None."""
    return _SCHEDULES.get((kind, schedule_id))


def forget(kind: str, schedule_id: str) -> None:
    """Removes a declaration (tests)."""
    _SCHEDULES.pop((kind, schedule_id), None)


def unschedule(scheduler: Any, kind: str, schedule_id: str) -> None:
    """Undoes ``schedule``: its timer and any catch-up not yet run come off ``scheduler``, and the
    declaration goes. The cursor stays, so declaring it again catches up from where it stopped."""
    for apscheduler_id in (job_id(kind, schedule_id), f"{job_id(kind, schedule_id)}:catch-up"):
        if scheduler.get_job(apscheduler_id) is not None:
            scheduler.remove_job(apscheduler_id)
    forget(kind, schedule_id)
