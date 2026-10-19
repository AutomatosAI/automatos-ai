"""HARNESS runs: the run table, the per-workspace lock and the sweep's cron (PRD-263 PR-1B).

* **Run table (US-009, US-010).** Every HARNESS run is one ``harness_runs`` row,
  opened as ``running`` when the run starts and closed with its outcome when it
  ends, so status and history read one database row that every worker sees.
  Completed runs' verdicts are the week-by-week history DIAGNOSE reads (US-014b).
* **Lock (US-011).** One run per workspace at a time, whatever the uvicorn worker
  or Railway replica count: a session-level Postgres advisory lock in its two-key
  form (``HARNESS_LOCK_NAMESPACE``, ``hashtext(<workspace id>)``), held on one
  dedicated connection for the whole run and released when it ends. A held lock
  makes another run a no-op; ``lock_is_held`` reads ``pg_locks`` without taking it.
  Holding it, the sweep skips a workspace whose judged week is settled: it already
  has a completed or dormant run (a failed one is retried).
* **Cron (US-012).** The sweep fires on ``config.HARNESS_SWEEP_CRON`` read in UTC,
  through ``schedule_util.scheduler_cron_trigger`` and never
  ``CronTrigger.from_crontab``, which reads a numeric weekday a day late (F240).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import UUID

from sqlalchemy import func, text

if TYPE_CHECKING:
    from apscheduler.triggers.cron import CronTrigger
    from sqlalchemy.engine import Connection, Engine
    from sqlalchemy.orm import Session

    from core.models.harness import HarnessRun

logger = logging.getLogger(__name__)

# The HARNESS key space for two-key advisory locks ('hrns' in hex). A two-key lock
# never meets a one-key lock such as BOOT_LOCK_ID (47111, core/database/boot_lock.py).
HARNESS_LOCK_NAMESPACE = 0x68726E73
# pg_locks shows a two-key advisory lock with objsubid 2 (a one-key lock: 1).
TWO_KEY_OBJSUBID = 2
# Why a run left ``running`` with no lock holder reads (and is closed as) failed.
INTERRUPTED = "interrupted"
FAILURE_PREFIX = "failed: "
# A run judges the last complete ISO week: the week that held the moment a week ago.
WEEK = timedelta(days=7)
# harness_runs.counts keys read outside DIAGNOSE: how many changes the run suggested
# (written by HarnessService._run_pipeline), and, on a run imported from the HARNESS
# files (US-023), how many changes the loop before PRD-263 applied in it. A report-only
# run applies nothing and has no IMPORTED_APPLIED key.
PRESCRIPTION_COUNT = "prescriptions"
IMPORTED_APPLIED = "applied"

_TRY_LOCK = text("SELECT pg_try_advisory_lock(:namespace, hashtext(:key))")
_UNLOCK = text("SELECT pg_advisory_unlock(:namespace, hashtext(:key))")
_LOCK_HELD = text(
    "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND granted "
    "AND database = (SELECT oid FROM pg_database WHERE datname = current_database()) "
    "AND classid = CAST(:namespace AS oid) AND objid = CAST(hashtext(:key) AS oid) "
    "AND objsubid = :objsubid)"
)


def last_complete_iso_week(now: Optional[datetime] = None) -> str:
    """The ISO week a run starting at ``now`` judges, as ``YYYY-Www`` (UTC)."""
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc) - WEEK
    year, week, _weekday = moment.isocalendar()
    return f"{year}-W{week:02d}"


def failure(reason: str) -> str:
    """The ``report_status`` of a run or audit report that failed for ``reason``."""
    return f"{FAILURE_PREFIX}{reason}"


def failure_reason(report_status: Optional[str]) -> str:
    """The reason a ``failed: <reason>`` report_status names ("unknown" if none)."""
    reason = (report_status or "").removeprefix(FAILURE_PREFIX).strip()
    return reason or "unknown"


# --------------------------------------------------------------------------- lock


def _lock_params(workspace_id: UUID) -> Dict[str, Any]:
    return {"namespace": HARNESS_LOCK_NAMESPACE, "key": str(workspace_id)}


@contextmanager
def workspace_lock(engine: "Engine", workspace_id: UUID) -> Iterator[bool]:
    """Yield True when this run holds the workspace's HARNESS lock, False when another does.

    The lock is session-level, on one dedicated connection kept for the whole
    block (the boot_lock pattern). The transaction that took it commits at once,
    so the connection waits idle, not idle-in-transaction, while the run awaits
    its phases; the lock outlives that commit and is released on the way out.
    """
    params = _lock_params(workspace_id)
    with engine.connect() as conn:
        held = bool(conn.execute(_TRY_LOCK, params).scalar())
        conn.commit()
        try:
            yield held
        finally:
            if held:
                _unlock(conn, params)


def _unlock(conn: "Connection", params: Dict[str, Any]) -> None:
    """Release the lock; when that fails, drop the connection (which releases it) so
    the pool never hands out a connection still holding it."""
    try:
        conn.execute(_UNLOCK, params)
        conn.commit()
    except Exception:
        logger.exception("[HARNESS] Advisory unlock failed for %s; dropping its connection", params["key"])
        conn.invalidate()


def lock_is_held(db: "Session", workspace_id: UUID) -> bool:
    """Whether a run holds the workspace's HARNESS lock, read from pg_locks without taking it."""
    params = {**_lock_params(workspace_id), "objsubid": TWO_KEY_OBJSUBID}
    return bool(db.execute(_LOCK_HELD, params).scalar())


# ---------------------------------------------------------------------- run table


def completed_runs(db: "Session", workspace_id: UUID) -> int:
    """How many completed runs the workspace has."""
    from core.models.harness import RUN_COMPLETED, HarnessRun

    count = (
        db.query(func.count(HarnessRun.id))
        .filter(HarnessRun.workspace_id == workspace_id, HarnessRun.status == RUN_COMPLETED)
        .scalar()
    )
    return int(count or 0)


def week_is_settled(db: "Session", workspace_id: UUID, cur_window: str) -> bool:
    """Whether the workspace already has a run judging ``cur_window`` that ended
    completed or dormant, so the sweep has nothing left to do for that week.

    A dormant run counts: a second scheduler (another replica) reaching the same
    workspace minutes later would only write the same dormancy again. A failed or
    interrupted run does not, so the next run retries the week (US-011: a run
    closed as interrupted "proceeds").
    """
    from core.models.harness import RUN_FAILED, RUN_RUNNING, HarnessRun

    found = (
        db.query(HarnessRun.id)
        .filter(
            HarnessRun.workspace_id == workspace_id,
            HarnessRun.cur_window == cur_window,
            HarnessRun.status.notin_((RUN_RUNNING, RUN_FAILED)),
        )
        .first()
    )
    return found is not None


def judged_verdicts(db: "Session", workspace_id: UUID, windows: Sequence[str]) -> List[Tuple[str, Any]]:
    """The (cur_window, verdicts) of the workspace's completed runs that judged one of
    ``windows``, oldest first: DIAGNOSE's week-by-week verdict history (US-014b)."""
    from core.models.harness import RUN_COMPLETED, HarnessRun

    rows = (
        db.query(HarnessRun.cur_window, HarnessRun.verdicts)
        .filter(
            HarnessRun.workspace_id == workspace_id,
            HarnessRun.status == RUN_COMPLETED,
            HarnessRun.cur_window.in_(list(windows)),
        )
        .order_by(HarnessRun.id)
        .all()
    )
    return [(window, verdicts) for window, verdicts in rows]


def latest_run(db: "Session", workspace_id: UUID) -> Optional["HarnessRun"]:
    """The workspace's most recent run, or None before its first."""
    runs = recent_runs(db, workspace_id, 1)
    return runs[0] if runs else None


def recent_runs(db: "Session", workspace_id: UUID, limit: int) -> List["HarnessRun"]:
    """The workspace's ``limit`` most recent runs, newest first.

    Ordered by when each run started, then id: rows imported from the HARNESS files
    on the volume (``scripts/harness_backfill_runs.py``, US-023) are inserted after
    newer runs, so their ids say nothing about when they ran.
    """
    from core.models.harness import HarnessRun

    return (
        db.query(HarnessRun)
        .filter(HarnessRun.workspace_id == workspace_id)
        .order_by(HarnessRun.started_at.desc(), HarnessRun.id.desc())
        .limit(limit)
        .all()
    )


def recorded_windows(db: "Session", workspace_id: UUID) -> Set[str]:
    """Every ISO week the workspace has a run row for, whatever its status."""
    from core.models.harness import HarnessRun

    rows = db.query(HarnessRun.cur_window).filter(HarnessRun.workspace_id == workspace_id).distinct().all()
    return {window for (window,) in rows}


def record_imported_runs(db: "Session", workspace_id: UUID, runs: Sequence[Mapping[str, Any]]) -> int:
    """Insert finished runs read from the HARNESS files on the volume (US-023) and commit.

    Each mapping carries the row's columns (``cur_window``, ``started_at``,
    ``finished_at``, ``status``, ``iteration`` and optionally ``counts``,
    ``prescriptions`` and ``report_status``). The caller has already left out every
    week in recorded_windows. Returns how many rows were inserted.
    """
    from core.models.harness import HarnessRun

    db.add_all([HarnessRun(workspace_id=workspace_id, **dict(run)) for run in runs])
    db.commit()
    return len(runs)


def open_run(db: "Session", workspace_id: UUID, cur_window: str) -> int:
    """Open the run's row as ``running`` (ORG-1) and return its id.

    Its iteration starts at the completed runs before it; ``close_run`` sets it on
    completion. Committed at once, so every worker sees the run.
    """
    from core.models.harness import RUN_RUNNING, HarnessRun

    run = HarnessRun(
        workspace_id=workspace_id,
        cur_window=cur_window,
        started_at=datetime.now(timezone.utc),
        status=RUN_RUNNING,
        iteration=completed_runs(db, workspace_id),
    )
    db.add(run)
    db.flush()
    run_id = run.id
    db.commit()
    return run_id


def close_run(
    db: "Session",
    run_id: int,
    status: str,
    *,
    iteration: Optional[int] = None,
    coverage: Optional[float] = None,
    counts: Optional[Dict[str, Any]] = None,
    verdicts: Optional[Dict[str, Any]] = None,
    prescriptions: Optional[List[Dict[str, Any]]] = None,
    report_md: Optional[str] = None,
    report_status: Optional[str] = None,
) -> None:
    """Close the run's row with its outcome (ORG-11) and commit.

    ``finished_at`` is now; a field given as None keeps what the row holds.
    """
    from core.models.harness import HarnessRun

    fields = {
        "iteration": iteration,
        "coverage": coverage,
        "counts": counts,
        "verdicts": verdicts,
        "prescriptions": prescriptions,
        "report_md": report_md,
        "report_status": report_status,
    }
    values = {
        "status": status,
        "finished_at": datetime.now(timezone.utc),
        **{name: value for name, value in fields.items() if value is not None},
    }
    db.query(HarnessRun).filter(HarnessRun.id == run_id).update(values, synchronize_session=False)
    db.commit()


def close_interrupted(db: "Session", workspace_id: UUID) -> int:
    """Close the workspace's ``running`` rows as failed ("interrupted"); return how many.

    Only a run that holds the workspace's lock calls this, so no running row has a
    holder: each was left by a run that died (a restart, a lost replica) before it
    could close its row.
    """
    from core.models.harness import RUN_FAILED, RUN_RUNNING, HarnessRun

    closed = (
        db.query(HarnessRun)
        .filter(HarnessRun.workspace_id == workspace_id, HarnessRun.status == RUN_RUNNING)
        .update(
            {
                "status": RUN_FAILED,
                "finished_at": datetime.now(timezone.utc),
                "report_status": failure(INTERRUPTED),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    if closed:
        logger.warning("[HARNESS] Closed %d interrupted run(s) for %s", closed, workspace_id)
    return closed


# --------------------------------------------------------------------------- cron


def sweep_trigger() -> "CronTrigger":
    """The sweep's trigger: ``config.HARNESS_SWEEP_CRON`` in UTC. ValueError for a bad cron."""
    from config import config
    from services.schedule_util import scheduler_cron_trigger

    return scheduler_cron_trigger(config.HARNESS_SWEEP_CRON, timezone=timezone.utc)


def next_sweep_run(now: Optional[datetime] = None) -> Optional[str]:
    """When the sweep next fires, ISO 8601 in UTC, computed from its trigger so any
    worker can answer. None when HARNESS_SWEEP_CRON cannot be read (logged)."""
    from config import config

    try:
        fire = sweep_trigger().get_next_fire_time(None, now or datetime.now(timezone.utc))
    except ValueError:
        logger.exception("[HARNESS] HARNESS_SWEEP_CRON %r is not a cron expression", config.HARNESS_SWEEP_CRON)
        return None
    return fire.astimezone(timezone.utc).isoformat() if fire else None
