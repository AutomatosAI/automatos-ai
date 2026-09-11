"""Leader reconcile tick — the scheduler worker catches up with the DB.

Production runs several uvicorn workers; exactly one holds the scheduler lock
and hosts APScheduler (services.scheduler). Every write that changes what
should be scheduled — a scheduled task created, paused, resumed or cancelled,
a heartbeat toggled or re-configured — lands on whichever worker served the
request and only touches THAT worker's (usually absent) scheduler. Before this
tick the leader learned of the change at its next restart.

The tick runs on the leader every ``RECONCILE_INTERVAL_SECONDS`` and diffs the
DB against the registered jobs for both sources:

* ``agent_scheduled_tasks`` rows ↔ ``scheduled_task_<id>`` jobs
* ``agents.configuration.heartbeat`` blocks ↔ ``agent_hb_<id>`` jobs

Both reconcilers are idempotent, so a request-worker edit that DID land on the
leader is simply confirmed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.scheduled_task_service import RECONCILE_INTERVAL_SECONDS, ScheduledTaskService

logger = logging.getLogger(__name__)

RECONCILE_JOB_ID = "schedule_reconcile"


def run_reconcile_once(scheduler: Any, db: Optional[Any] = None) -> Dict[str, Any]:
    """One pass over both sources. Opens (and closes) its own session unless given one."""
    from services.heartbeat_service import get_heartbeat_service

    owns_db = db is None
    if owns_db:
        from core.database.database import SessionLocal

        db = SessionLocal()
    try:
        tasks = ScheduledTaskService(db, workspace_id=None).reconcile_with_scheduler(scheduler)
        heartbeats = get_heartbeat_service().reconcile_agent_heartbeats(db)
    finally:
        if owns_db:
            db.close()
    return {"tasks": tasks, "heartbeats": heartbeats}


def _tick(scheduler: Any) -> None:
    try:
        run_reconcile_once(scheduler)
    except Exception as exc:  # noqa: BLE001 — a failed pass must not kill the job
        logger.warning("[ScheduleReconcile] pass failed: %s", exc)


async def start_schedule_reconcile(scheduler: Any) -> bool:
    """Register the tick on the leader's APScheduler and run the first pass now.

    Returns False when this worker hosts no running scheduler (nothing to do).
    """
    if scheduler is None or not getattr(scheduler, "running", False):
        return False
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler.add_job(
        _tick,
        IntervalTrigger(seconds=RECONCILE_INTERVAL_SECONDS),
        id=RECONCILE_JOB_ID,
        args=[scheduler],
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    _tick(scheduler)
    logger.info("[ScheduleReconcile] tick registered every %ds", RECONCILE_INTERVAL_SECONDS)
    return True
