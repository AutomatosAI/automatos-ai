"""The credit meter's scheduled jobs (PRD-265), queued by their schedules (PRD-276 §6.4, Wave 2b).

Each is a job kind with a platform schedule: the scheduler only queues a slot, and a worker runs it.

- ``credits.meter`` — every ``CREDITS_METER_INTERVAL_SECONDS``: debit the model calls made since the
  last tick and send balance alerts (``meter.meter_pending``). Catch-up ``latest``: one late tick
  debits everything still waiting, and the ledger never debits a call twice (§11).
- ``credits.overhead`` — 03:15 UTC: record the absorbed overhead per workspace for the day before
  the slot (``meter.record_overhead``), for margin reporting. Catch-up ``all``: each missed night
  records its own day, and a day recorded twice changes nothing (the ledger's key per workspace
  and day).
- ``credits.expiry`` — every hour: expire months past their grace and top-ups past their date
  (``expiry.expire_due``, PRD-265 §4.3). Catch-up ``latest``.

Each runs off the event loop. The local edition never meters, so they do nothing there.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from core.jobs.registry import CATCH_UP_ALL, CATCH_UP_LATEST, job_handler

logger = logging.getLogger(__name__)

METER_KIND = "credits.meter"
OVERHEAD_KIND = "credits.overhead"
EXPIRY_KIND = "credits.expiry"
EXPIRY_INTERVAL_HOURS = 1
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


def overhead_nightly(day: Optional[date] = None) -> None:
    """Record ``day``'s absorbed overhead (yesterday's, by default)."""
    if not _metered_edition():
        return
    from core.database.database import SessionLocal
    from services.credits.meter import record_overhead

    day = day or (datetime.now(timezone.utc) - timedelta(days=1)).date()
    db = SessionLocal()
    try:
        logger.info("[credits] overhead for %s: %d workspaces", day, record_overhead(db, day))
    finally:
        db.close()


def expiry_hourly() -> None:
    """Expire the credits whose time is up."""
    if not _metered_edition():
        return
    from core.database.database import SessionLocal
    from services.credits.expiry import expire_due

    db = SessionLocal()
    try:
        expired = expire_due(db)
        if any(expired.values()):
            logger.info("[credits] expired %d monthly and %d top-up credits", expired["monthly"], expired["topup"])
    finally:
        db.close()


def _slot(job: Any) -> datetime:
    return datetime.fromisoformat(job.payload["slot"])


@job_handler(METER_KIND, catch_up=CATCH_UP_LATEST, concurrency=1, workspace_concurrency=1)
async def run_meter(job: Any) -> None:
    """A metering slot."""
    await asyncio.to_thread(meter_tick)


@job_handler(OVERHEAD_KIND, catch_up=CATCH_UP_ALL, concurrency=1, workspace_concurrency=1)
async def run_overhead(job: Any) -> None:
    """A nightly slot: the overhead of the day before it."""
    await asyncio.to_thread(overhead_nightly, (_slot(job) - timedelta(days=1)).date())


@job_handler(EXPIRY_KIND, catch_up=CATCH_UP_LATEST, concurrency=1, workspace_concurrency=1)
async def run_expiry(job: Any) -> None:
    """An hourly expiry slot."""
    await asyncio.to_thread(expiry_hourly)


def register_credit_jobs(scheduler: Any) -> None:
    """Declares the three platform schedules on ``scheduler`` (replacing any earlier declaration)."""
    from apscheduler.triggers.cron import CronTrigger

    from config import config
    from core.jobs.schedules import PLATFORM, interval, schedule

    schedule(scheduler, METER_KIND, PLATFORM, interval(seconds=int(config.CREDITS_METER_INTERVAL_SECONDS)))
    schedule(scheduler, OVERHEAD_KIND, PLATFORM,
             CronTrigger(hour=OVERHEAD_HOUR_UTC, minute=OVERHEAD_MINUTE, timezone=timezone.utc))
    schedule(scheduler, EXPIRY_KIND, PLATFORM, interval(hours=EXPIRY_INTERVAL_HOURS))
    logger.info("[credits] meter every %ss; overhead recorded daily at %02d:%02d UTC",
                config.CREDITS_METER_INTERVAL_SECONDS, OVERHEAD_HOUR_UTC, OVERHEAD_MINUTE)
