"""The credit meter's scheduled jobs, on the unified scheduler (PRD-265).

- ``credits_meter_tick`` — every ``CREDITS_METER_INTERVAL_SECONDS``: debit the
  model calls made since the last tick and send balance alerts
  (``meter.meter_pending``).
- ``credits_overhead_nightly`` — 03:15 UTC: record yesterday's absorbed overhead
  per workspace (``meter.record_overhead``), for margin reporting.

Both are plain functions (the Redis job store keeps references, not closures)
and run on the scheduler's thread pool, off the event loop. The local edition
never meters, so they do nothing there.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

METER_JOB_ID = "credits_meter_tick"
OVERHEAD_JOB_ID = "credits_overhead_nightly"
OVERHEAD_HOUR_UTC = 3
OVERHEAD_MINUTE = 15


def _metered_edition() -> bool:
    from config import config

    return config.AUTH_EDITION != "local"


def meter_tick() -> None:
    """Debit the calls made since the last tick."""
    if not _metered_edition():
        return
    from core.database.database import SessionLocal
    from services.credits.meter import meter_pending

    db = SessionLocal()
    try:
        debited = meter_pending(db)
        if debited:
            logger.info("[credits] metered %d calls", debited)
    finally:
        db.close()


def overhead_nightly() -> None:
    """Record yesterday's absorbed overhead."""
    if not _metered_edition():
        return
    from core.database.database import SessionLocal
    from services.credits.meter import record_overhead

    db = SessionLocal()
    try:
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        logger.info("[credits] overhead for %s: %d workspaces", yesterday, record_overhead(db, yesterday))
    finally:
        db.close()


def register_credit_jobs(scheduler: Any) -> None:
    """Put both jobs on ``scheduler`` (replacing any earlier registration)."""
    from config import config

    scheduler.add_job(meter_tick, "interval", seconds=int(config.CREDITS_METER_INTERVAL_SECONDS),
                      id=METER_JOB_ID, replace_existing=True, max_instances=1)
    scheduler.add_job(overhead_nightly, "cron", hour=OVERHEAD_HOUR_UTC, minute=OVERHEAD_MINUTE,
                      id=OVERHEAD_JOB_ID, replace_existing=True, max_instances=1)
    logger.info("[credits] meter every %ss; overhead recorded daily at %02d:%02d UTC",
                config.CREDITS_METER_INTERVAL_SECONDS, OVERHEAD_HOUR_UTC, OVERHEAD_MINUTE)
