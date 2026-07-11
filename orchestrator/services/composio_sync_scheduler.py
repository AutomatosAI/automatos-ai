"""ComposioSyncScheduler — daily Composio action-metadata refresh (PRD-177 S3).

Registers a daily cron job on the UnifiedScheduler that runs
``jobs.sync_composio_actions.sync_all_composio_actions()`` to keep the
``composio_action_metadata`` classifications fresh. That table backs the
destructive-action gate (F018) — without a scheduler it only filled on
app-enable / manual trigger, so a cold table left the gate with nothing to
check. Runs alongside the nightly edge recompute (EdgeBuilderScheduler), one
hour later by default so the two heavy jobs don't overlap.

Mirrors EdgeBuilderScheduler exactly (same shared APScheduler, same singleton
factory idiom). Failure to sync is logged, never raised — a stale metadata
table degrades gracefully to the fail-closed gate, it must not crash boot.
"""

import asyncio
import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)


class ComposioSyncScheduler:
    """Registers the daily Composio action-metadata sync on the shared scheduler."""

    JOB_ID = "composio_action_sync_daily"

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None

    async def start(self, scheduler: AsyncIOScheduler):
        """Register the Composio metadata sync cron job."""
        from config import config as app_config

        self._scheduler = scheduler
        hour = int(getattr(app_config, "COMPOSIO_SYNC_HOUR_UTC", 4))

        self._scheduler.add_job(
            self._run_sync,
            "cron",
            hour=hour,
            minute=0,
            id=self.JOB_ID,
            replace_existing=True,
            max_instances=1,
        )
        logger.info("[ComposioSync] Scheduled daily metadata sync at %02d:00 UTC", hour)

        # PRD-194 S6 (P2-13): startup assertion — connected apps with an
        # EMPTY composio_action_metadata table means the destructive gate is
        # running blind on the keyword heuristic. Say so at boot, loudly,
        # instead of failing silently for months. Logged, never raised: a
        # stale table degrades to the fail-closed keyword floor and must not
        # crash boot (module docstring contract).
        await asyncio.to_thread(self._startup_metadata_check)

    @staticmethod
    def _startup_metadata_check() -> None:
        try:
            from core.database.database import SessionLocal
            from modules.tools.sync.composio_action_sync import (
                check_action_metadata_populated,
            )

            db = SessionLocal()
            try:
                ok, detail = check_action_metadata_populated(db)
            finally:
                db.close()
            if ok:
                logger.info("[ComposioSync] startup metadata check: %s", detail)
            else:
                logger.error("[ComposioSync] STARTUP ASSERTION FAILED: %s", detail)
        except Exception:
            logger.exception("[ComposioSync] startup metadata check errored")

    async def _run_sync(self):
        """Execute the Composio action metadata sync."""
        try:
            from jobs.sync_composio_actions import sync_all_composio_actions

            result = await sync_all_composio_actions()
            logger.info(
                "[ComposioSync] Daily sync complete — status=%s classified=%s",
                result.get("status"),
                result.get("classified"),
            )
        except Exception:
            logger.exception("[ComposioSync] Daily sync failed")


_instance: Optional[ComposioSyncScheduler] = None


def get_composio_sync_scheduler() -> ComposioSyncScheduler:
    global _instance
    if _instance is None:
        _instance = ComposioSyncScheduler()
    return _instance
