"""PRD-268 §6.2 — ``proactive_runs``, the check-in audit, is kept 90 days.

``prune_batches`` deletes the rows written longer ago than the retention:
``config.PROACTIVE_RUNS_RETENTION_DAYS`` (default 90), never under ``MIN_RETENTION_DAYS``
whatever the config holds, so a zero or a negative value never empties the audit. As
``services.audit_retention`` sweeps ``audit_logs``, it deletes in batches of at most
``PRUNE_BATCH`` rows, each one bound ``DELETE`` committed on its own, so no batch holds its
locks while the next is chosen.

It touches ``proactive_runs`` only. A suggestion's ``run_id`` is a plain value, not a foreign
key (``core.models.proactive``), so a suggestion outlives its run; follow-ups
(``agent_scheduled_tasks``) and the brief's claim (``mobile_digest_posts``) are other tables.

``run_prune`` is the job, on a session of its own, registered by ``register`` on the leader's
scheduler from ``services.schedule_reconcile``, as the push receipts and the daily digest are.
It runs once a day at ``PRUNE_HOUR_UTC``:``PRUNE_MINUTE`` UTC: a cron time, not a 24-hour
interval, which every restart of the leader would start counting again. It never raises: a
failure is logged with its traceback and the count deleted before it, and rolled back (the
batches before it stay deleted); the scheduler and its other jobs go on. Its log lines hold
counts only, by run kind: no workspace, agent or row content.

No index leads with ``created_at`` (``ix_proactive_runs_workspace_agent_created`` leads with
the workspace), so choosing a batch scans the table. Once a day that is a few scans of at most
the retention's check-ins: one row per check-in run, a table that stays small.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from config import config
from core.models.proactive import ProactiveRun

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_RETENTION_DAYS",
    "PRUNE_BATCH",
    "PRUNE_HOUR_UTC",
    "PRUNE_JOB_ID",
    "PRUNE_MINUTE",
    "cutoff",
    "prune_batches",
    "register",
    "retention_days",
    "run_prune",
]

MIN_RETENTION_DAYS = 1
# As services.audit_retention: a backlog never holds the table in one statement.
PRUNE_BATCH = 5000
PRUNE_JOB_ID = "proactive_runs_prune"
# Once a day, after the 03:00 memory promotion and the 03:15 credit overhead.
PRUNE_HOUR_UTC = 4
PRUNE_MINUTE = 45
# A run the leader starts late (busy, or just restarted) still runs within this hour.
PRUNE_MISFIRE_GRACE_SECONDS = 3600

SessionFactory = Callable[[], Session]


def retention_days() -> int:
    """How many days a run row is kept: the config's value, never under ``MIN_RETENTION_DAYS``."""
    return max(MIN_RETENTION_DAYS, int(config.PROACTIVE_RUNS_RETENTION_DAYS))


def cutoff(now: datetime, days: int) -> datetime:
    """Rows written before this are past the retention (pure); the minimum binds here too."""
    return now - timedelta(days=max(MIN_RETENTION_DAYS, int(days)))


def _delete_batch(db: Session, before: datetime, batch_size: int) -> Tuple[str, ...]:
    """Delete at most ``batch_size`` run rows written before ``before`` and commit; their kinds."""
    table = ProactiveRun.__table__
    oldest = select(table.c.id).where(table.c.created_at < before).limit(batch_size)
    kinds = db.execute(delete(table).where(table.c.id.in_(oldest)).returning(table.c.kind)).scalars().all()
    db.commit()
    return tuple(kinds)


def prune_batches(db: Session, *, days: int, now: Optional[datetime] = None,
                  batch_size: int = PRUNE_BATCH) -> Iterator[Tuple[str, ...]]:
    """Delete the run rows written more than ``days`` days before ``now``, a batch at a time,
    each committed before it is yielded; yields each batch's run kinds, and stops after the
    first batch that comes back short. Lazy: nothing is deleted until it is iterated."""
    before = cutoff(now or datetime.now(timezone.utc), days)
    size = max(1, int(batch_size))
    while True:
        kinds = _delete_batch(db, before, size)
        yield kinds
        if len(kinds) < size:
            return


def _by_kind(deleted: Counter) -> Dict[str, int]:
    return dict(sorted(deleted.items()))


def _open_session() -> Session:
    from core.database.database import SessionLocal

    return SessionLocal()


def _close(db: Optional[Session], *, rollback: bool) -> None:
    """Roll back (after a failure), then close; either one failing is logged, never raised, and
    a failed rollback still closes."""
    if db is None:
        return
    if rollback:
        try:
            db.rollback()
        except Exception:
            logger.exception("[proactive] runs prune: the session did not roll back")
    try:
        db.close()
    except Exception:
        logger.exception("[proactive] runs prune: the session did not close cleanly")


def run_prune(*, session_factory: Optional[SessionFactory] = None,
              now: Optional[datetime] = None) -> Dict[str, int]:
    """The job: prune ``proactive_runs`` on a session of its own; how many rows went, by kind.

    Never raises: a failure is logged with its traceback and rolled back, and the counts are
    the batches committed before it, which stay deleted; the next day's run goes on from there.
    """
    db: Optional[Session] = None
    deleted: Counter = Counter()
    try:
        days = retention_days()
        db = (session_factory or _open_session)()
        for kinds in prune_batches(db, days=days, now=now, batch_size=PRUNE_BATCH):
            deleted = deleted + Counter(kinds)
    except Exception:
        logger.exception("[proactive] runs prune failed after %d row(s) deleted, by kind %s; the rest stay "
                         "until the next run", sum(deleted.values()), _by_kind(deleted))
        _close(db, rollback=True)
        return _by_kind(deleted)
    _close(db, rollback=False)
    logger.info("[proactive] runs prune: %d proactive_runs row(s) older than %d days deleted, by kind %s",
                sum(deleted.values()), days, _by_kind(deleted))
    return _by_kind(deleted)


def register(scheduler: Any) -> bool:
    """The prune job on the leader's scheduler, daily; ``False`` when this worker hosts none.

    No job args: a Redis job store pickles them (``schedule_reconcile._tick``), and the job
    needs none.
    """
    if scheduler is None or not getattr(scheduler, "running", False):
        return False
    from apscheduler.triggers.cron import CronTrigger

    scheduler.add_job(
        run_prune, CronTrigger(hour=PRUNE_HOUR_UTC, minute=PRUNE_MINUTE, timezone=timezone.utc),
        id=PRUNE_JOB_ID, replace_existing=True, max_instances=1, coalesce=True,
        misfire_grace_time=PRUNE_MISFIRE_GRACE_SECONDS,
    )
    logger.info("[proactive] runs prune: daily at %02d:%02d UTC", PRUNE_HOUR_UTC, PRUNE_MINUTE)
    return True
