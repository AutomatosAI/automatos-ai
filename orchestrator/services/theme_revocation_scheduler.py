"""ThemeRevocationScheduler — the local edition's pull of the hosted theme revocation feed (PRD-261 C-05h).

Registers one interval job on the UnifiedScheduler that runs
``services.themes.revocation_pull.pull_feed`` every ``THEME_REVOCATIONS_PULL_MINUTES`` (60 by
default), the first run ``FIRST_RUN_DELAY_SECONDS`` after start, one run at a time
(``max_instances=1``). It is never registered in the hosted edition (which serves the feed) or
with an empty ``THEME_REVOCATIONS_URL`` (QB-25, still open: the plan's default is on).

The job is the module-level ``run_scheduled_pull``, so a RedisJobStore can store it by reference
(a scheduler or service instance refuses to be serialized). It runs the synchronous pull in a
worker thread with its own session; a failed pull is recorded on the sync row by ``pull_feed``
and logged here, never raised (a stale list degrades to the rows already held, it must not kill
the job or boot). The admin's Check now (``api/admin_theme_revocation_sync.py``) runs the same
pull on demand, and is the only pull when the shared scheduler is off (``main.py`` starts it only
when HEARTBEAT, RECIPE or COORDINATOR is on): the sync status then shows the list going stale.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from apscheduler.triggers.interval import IntervalTrigger

from config import config
from services.themes.revocation_pull import PullOutcome, feed_url, pull_feed

logger = logging.getLogger(__name__)

JOB_ID = "theme_revocation_pull"
PULL_MINUTES_SETTING = "THEME_REVOCATIONS_PULL_MINUTES"
DEFAULT_PULL_MINUTES = 60
MIN_PULL_MINUTES = 1
FIRST_RUN_DELAY_SECONDS = 60


def pull_minutes() -> int:
    """``THEME_REVOCATIONS_PULL_MINUTES``; a value under one minute is read as the default and logged."""
    minutes = int(getattr(config, PULL_MINUTES_SETTING, DEFAULT_PULL_MINUTES))
    if minutes < MIN_PULL_MINUTES:
        logger.warning("[ThemeRevocations] %s=%d is under a minute: pulling every %d minutes",
                       PULL_MINUTES_SETTING, minutes, DEFAULT_PULL_MINUTES)
        return DEFAULT_PULL_MINUTES
    return minutes


def _pull_in_session() -> PullOutcome:
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        return pull_feed(db)
    finally:
        db.close()


async def run_scheduled_pull() -> None:
    """The job: one pull in a worker thread; a failure is logged, never raised."""
    try:
        outcome = await asyncio.to_thread(_pull_in_session)
    except Exception:
        logger.exception("[ThemeRevocations] Scheduled pull failed")
        return
    if outcome.ok:
        logger.info("[ThemeRevocations] Scheduled pull: %d revocations from the feed", outcome.rows)
    else:
        logger.warning("[ThemeRevocations] Scheduled pull failed: %s", outcome.error)


class ThemeRevocationScheduler:
    """Registers the revocation-feed pull on the shared APScheduler (local edition only)."""

    async def start(self, scheduler: Any) -> bool:
        """Register the pull job; False (nothing registered) in the hosted edition or with no feed URL."""
        if not config.IS_LOCAL_EDITION:
            logger.info("[ThemeRevocations] Hosted edition: it serves the revocation feed, no pull scheduled")
            return False
        if not feed_url():
            logger.info("[ThemeRevocations] THEME_REVOCATIONS_URL is empty: no pull scheduled")
            return False
        minutes = pull_minutes()
        first_run = datetime.now(timezone.utc) + timedelta(seconds=FIRST_RUN_DELAY_SECONDS)
        scheduler.add_job(
            run_scheduled_pull,
            IntervalTrigger(minutes=minutes),
            id=JOB_ID,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            next_run_time=first_run,
        )
        logger.info("[ThemeRevocations] Pulling the revocation feed every %d minutes, first in %d s",
                    minutes, FIRST_RUN_DELAY_SECONDS)
        return True


_instance: Optional[ThemeRevocationScheduler] = None


def get_theme_revocation_scheduler() -> ThemeRevocationScheduler:
    """The one ThemeRevocationScheduler (main.py starts it on the shared scheduler)."""
    global _instance
    if _instance is None:
        _instance = ThemeRevocationScheduler()
    return _instance
