"""
UnifiedScheduler
=================
Singleton owning a single APScheduler AsyncIOScheduler instance.
HeartbeatService and RecipeSchedulerService register their jobs on this
shared scheduler instead of each spinning up their own.

This ensures:
- One worker holds the fcntl lock → one scheduler across all uvicorn workers
- Single Redis/memory jobstore connection
- Centralised start/stop lifecycle
"""

import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore

logger = logging.getLogger(__name__)


class UnifiedScheduler:
    """Owns the single APScheduler instance shared by all scheduling services."""

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None

    @property
    def apscheduler(self) -> Optional[AsyncIOScheduler]:
        return self._scheduler

    def start(self):
        """Create and start the shared AsyncIOScheduler."""
        if self._scheduler and self._scheduler.running:
            logger.warning("[Scheduler] Already running, skipping start")
            return

        jobstores = {"default": MemoryJobStore()}

        try:
            from config import config as app_config

            if app_config.REDIS_URL:
                from apscheduler.jobstores.redis import RedisJobStore

                jobstores["default"] = RedisJobStore(url=app_config.REDIS_URL)
                logger.info("[Scheduler] Using Redis job store")
        except Exception:
            logger.info("[Scheduler] Using memory job store")

        self._scheduler = AsyncIOScheduler(jobstores=jobstores)
        self._scheduler.start()
        logger.info("[Scheduler] Unified scheduler started")

    def stop(self):
        """Shut down the shared scheduler (stops all registered jobs)."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("[Scheduler] Unified scheduler stopped")
            self._scheduler = None


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_unified_scheduler: Optional[UnifiedScheduler] = None


def get_unified_scheduler() -> UnifiedScheduler:
    global _unified_scheduler
    if _unified_scheduler is None:
        _unified_scheduler = UnifiedScheduler()
    return _unified_scheduler
