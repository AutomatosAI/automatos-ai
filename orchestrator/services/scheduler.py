"""
UnifiedScheduler
=================
Singleton owning a single APScheduler AsyncIOScheduler instance.
HeartbeatService and RecipeSchedulerService register their jobs on this
shared scheduler instead of each spinning up their own.

This ensures:
- One process holds the scheduler lock (a Postgres advisory lock, PRD-276 §6.4,
  ``core/jobs/leadership.py``) → one scheduler across workers and containers;
  the others wait and take over if it dies, and it pauses if it loses the lock
- An in-memory job store: every service rebuilds its jobs from the database when
  it starts (scheduled tasks and social posts through the reconcile tick), so the
  schedule never needs sharing across processes
- Centralised start/stop lifecycle
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from services.end_user_retention import schedules_the_retention_sweep
from services.product_purge import schedules_the_closed_purge
from services.run_callbacks import schedules_the_callback_sweep

logger = logging.getLogger(__name__)


class UnifiedScheduler:
    """Owns the single APScheduler instance shared by all scheduling services."""

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._leadership: Any = None
        self._leader_stop: Optional[asyncio.Event] = None
        self._leader_task: Optional[asyncio.Task] = None

    @property
    def apscheduler(self) -> Optional[AsyncIOScheduler]:
        return self._scheduler

    @schedules_the_callback_sweep  # PRD-276 W1b: callbacks no completion writer queued are queued
    @schedules_the_retention_sweep  # PRD-274 W3: what is past an end user's retention, daily
    @schedules_the_closed_purge  # PRD-274 W5c: product workspaces closed past their grace, daily
    def start(self):
        """Create and start the shared AsyncIOScheduler."""
        if self._scheduler and self._scheduler.running:
            logger.warning("[Scheduler] Already running, skipping start")
            return

        self._scheduler = AsyncIOScheduler(jobstores={"default": MemoryJobStore()})
        self._scheduler.start()
        logger.info("[Scheduler] Unified scheduler started")

        # PRD-265: the credit meter and the nightly overhead record.
        from services.credits.jobs import register_credit_jobs

        register_credit_jobs(self._scheduler)

        # PRD-261 C-05h: the local edition's pull of the hosted theme revocation feed (QB-25).
        # It registers nothing in SaaS or with an empty THEME_REVOCATIONS_URL. Guarded like
        # ComposioSyncScheduler: a failure is logged and the shared scheduler keeps running.
        try:
            from services.theme_revocation_scheduler import get_theme_revocation_scheduler

            if get_theme_revocation_scheduler().start(scheduler=self._scheduler):
                logger.info("[Scheduler] ThemeRevocationScheduler started on unified scheduler")
        except Exception:
            logger.exception("[Scheduler] Could not start ThemeRevocationScheduler (non-fatal)")

    async def lead(self, register_services: Callable[[Any], Awaitable[None]], every_seconds: float) -> bool:
        """Tries the scheduler lock now, then keeps checking it in the background. The first time
        this process holds it, the scheduler starts and ``register_services`` puts the scheduled
        services on it; losing the lock pauses it, and holding it again resumes it. True when this
        process leads after the first try."""
        from core.jobs.leadership import leadership_for

        async def start_services() -> None:
            self.start()
            await register_services(self._scheduler)

        self._leadership = leadership_for(start_services, self.set_paused)
        leading = await self._leadership.check()
        self._leader_stop = asyncio.Event()
        self._leader_task = asyncio.create_task(self._leadership.run(self._leader_stop, every_seconds))
        return leading

    def set_paused(self, paused: bool) -> None:
        """Pauses the scheduler (no job fires) or resumes it; nothing when it isn't running."""
        if self._scheduler is None or not self._scheduler.running:
            return
        (self._scheduler.pause if paused else self._scheduler.resume)()

    def stop(self):
        """Shut down the shared scheduler (stops all registered jobs) and give back the scheduler lock."""
        if self._leader_stop is not None:
            self._leader_stop.set()
        if self._leader_task is not None:
            self._leader_task.cancel()
            self._leader_task = None
        if self._leadership is not None:
            self._leadership.release()
            self._leadership = None
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
