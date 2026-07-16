"""
PlaybookSchedulerService
========================
Fires cron-scheduled playbooks using APScheduler.

Follows the same pattern as HeartbeatService:
- APScheduler AsyncIOScheduler with memory (or Redis) job store
- Starts in main.py lifespan
- Loads all cron playbooks on startup, adds/removes jobs on playbook create/update
"""

import asyncio
import logging
from typing import Optional
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


class PlaybookSchedulerService:

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._owns_scheduler: bool = False  # True when we created our own scheduler (tests)
        # PRD-204 S4: once-per-breaker-open-period latch for playbook_benched.
        # In-memory is deliberate and sufficient: the scheduler is a
        # single-owner process (fcntl lock in main.py lifespan), so exactly
        # one instance observes every skip. Cleared when the breaker closes
        # (the check passes again); a process restart re-notifies at most
        # once per still-open breaker, which is acceptable for an alert.
        self._benched_notified: set[int] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, scheduler: Optional[AsyncIOScheduler] = None):
        """Initialize scheduler and load all cron playbooks from DB.

        Args:
            scheduler: Shared APScheduler instance from UnifiedScheduler.
                        If None, creates a local scheduler (useful for tests).
        """
        if scheduler:
            self._scheduler = scheduler
            self._owns_scheduler = False
        else:
            # Standalone mode (tests / backwards compat)
            jobstores = {"default": MemoryJobStore()}
            try:
                from config import config as app_config
                if app_config.REDIS_URL:
                    from apscheduler.jobstores.redis import RedisJobStore
                    jobstores["default"] = RedisJobStore(url=app_config.REDIS_URL)
                    logger.info("[PlaybookScheduler] Using Redis job store (standalone)")
            except Exception:
                pass
            self._scheduler = AsyncIOScheduler(jobstores=jobstores)
            self._scheduler.start()
            self._owns_scheduler = True

        await self._load_cron_playbooks()
        logger.info("[PlaybookScheduler] Service started")

    async def stop(self):
        """Remove playbook jobs. Only shuts down scheduler if we own it."""
        if self._scheduler and self._owns_scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("[PlaybookScheduler] Standalone scheduler stopped")
        logger.info("[PlaybookScheduler] Service stopped")

    # ------------------------------------------------------------------
    # Load from DB
    # ------------------------------------------------------------------

    async def _load_cron_playbooks(self):
        """Query all cron-type playbooks and schedule them."""
        from core.database.database import SessionLocal
        from core.models import WorkflowTemplate as WorkflowPlaybook
        from sqlalchemy import text

        db = SessionLocal()
        try:
            playbooks = db.query(WorkflowPlaybook).filter(
                WorkflowPlaybook.schedule_config.isnot(None),
                WorkflowPlaybook.workspace_id.isnot(None),
                WorkflowPlaybook.steps.isnot(None),
            ).all()

            count = 0
            for playbook in playbooks:
                sc = playbook.schedule_config or {}
                if sc.get("type") != "cron":
                    continue
                expr = sc.get("cron_expression")
                if not expr:
                    continue
                self.schedule_playbook(playbook)
                count += 1

            logger.info("[PlaybookScheduler] Loaded %d cron playbooks", count)
        except Exception as e:
            logger.error("[PlaybookScheduler] Failed to load cron playbooks: %s", e, exc_info=True)
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Schedule / unschedule
    # ------------------------------------------------------------------

    def schedule_playbook(self, playbook):
        """Add or replace APScheduler job for a cron playbook."""
        sc = playbook.schedule_config or {}
        expr = sc.get("cron_expression")
        if not expr:
            logger.warning("[PlaybookScheduler] No cron_expression for playbook %d, skipping", playbook.id)
            return

        job_id = f"playbook_cron_{playbook.id}"

        try:
            trigger = CronTrigger.from_crontab(expr)
        except ValueError as e:
            logger.error("[PlaybookScheduler] Invalid cron expression '%s' for playbook %d: %s", expr, playbook.id, e)
            return

        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

        self._scheduler.add_job(
            self._fire_playbook,
            trigger,
            id=job_id,
            args=[playbook.id, str(playbook.workspace_id)],
            replace_existing=True,
            max_instances=1,
        )
        logger.info("[PlaybookScheduler] Scheduled playbook %d (%s) with cron '%s'", playbook.id, getattr(playbook, 'name', ''), expr)

    def unschedule_playbook(self, playbook_id: int):
        """Remove a scheduled cron job by playbook id."""
        job_id = f"playbook_cron_{playbook_id}"
        if self._scheduler and self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
            logger.info("[PlaybookScheduler] Unscheduled playbook %d", playbook_id)

    # ------------------------------------------------------------------
    # Fire
    # ------------------------------------------------------------------

    async def _fire_playbook(self, playbook_id: int, workspace_id: str):
        """Execute a cron-triggered playbook. Same pattern as webhook handler."""
        from core.database.database import SessionLocal
        from core.models import WorkflowTemplate as WorkflowPlaybook
        from core.models.core import RecipeExecution
        from uuid import uuid4

        db = SessionLocal()
        try:
            playbook = db.query(WorkflowPlaybook).filter(WorkflowPlaybook.id == playbook_id).first()
            if not playbook:
                logger.warning("[PlaybookScheduler] Playbook %d no longer exists, unscheduling", playbook_id)
                self.unschedule_playbook(playbook_id)
                return

            # Skip if schedule was changed away from cron
            sc = playbook.schedule_config or {}
            if sc.get("type") != "cron":
                logger.info("[PlaybookScheduler] Playbook %d is no longer cron-type, unscheduling", playbook_id)
                self.unschedule_playbook(playbook_id)
                return

            if not playbook.steps:
                logger.warning("[PlaybookScheduler] Playbook %d has no steps, skipping", playbook_id)
                return

            # PRD-185 S4: repeated-failure circuit breaker. A cron playbook that
            # fails on every run re-fires forever (the 2026-06 daily 402 spam).
            # Once the last N terminal runs are all failures, stop re-firing until
            # a human intervenes; a manual run that succeeds breaks the streak and
            # auto-resets. Checked BEFORE creating an execution row so an open
            # breaker adds no history noise and stays stably open.
            from services.playbook_breaker import breaker_is_open
            if breaker_is_open(db, playbook.id):
                from config import config as _cfg
                logger.warning(
                    "[PlaybookScheduler] Circuit breaker OPEN for playbook %d (%s) — "
                    "last %d runs all failed; skipping cron re-fire until a manual run succeeds",
                    playbook.id, playbook.name, _cfg.PLAYBOOK_BREAKER_THRESHOLD,
                )
                # PRD-204 S4: the bench used to be a log line only. Notify
                # the workspace once per breaker-open period (in-memory
                # latch -- see __init__; cleared below when the breaker
                # closes).
                if playbook.id not in self._benched_notified:
                    self._benched_notified.add(playbook.id)
                    await self._notify_playbook_benched(
                        db, playbook, _cfg.PLAYBOOK_BREAKER_THRESHOLD
                    )
                return

            # Breaker closed -- clear the bench latch so the NEXT open period
            # notifies again.
            self._benched_notified.discard(playbook.id)

            execution_id = f"cron-{uuid4().hex[:12]}"
            execution = RecipeExecution(
                execution_id=execution_id,
                recipe_id=playbook.id,
                workspace_id=playbook.workspace_id,
                status="pending",
                input_data={},
                triggered_by="cron_scheduler",
                execution_metadata={
                    "execution_type": "cron_scheduler",
                    "total_steps": len(playbook.steps),
                    "cron_expression": sc.get("cron_expression"),
                },
            )
            db.add(execution)
            db.commit()

            # Concurrency guard — skip this tick if workspace is at capacity
            from services.concurrency_guard import check_concurrency
            concurrency = await check_concurrency(UUID(str(playbook.workspace_id)), db)
            if not concurrency.allowed:
                logger.warning(
                    "[PlaybookScheduler] Concurrency limit reached for workspace %s, "
                    "skipping playbook %d this tick: %s",
                    workspace_id, playbook_id, concurrency.reason,
                )
                # Roll back the pending execution record — cron will retry next tick
                db.delete(execution)
                db.commit()
                return

            logger.info("[PlaybookScheduler] Firing playbook %d (%s), execution=%s", playbook.id, playbook.name, execution_id)

            # PRD-142 W3-S12: cron-fired playbooks launch via the engine seam.
            from services.playbook_engine import get_playbook_engine
            get_playbook_engine().launch(
                recipe_execution_id=execution_id,
                recipe_id=playbook.id,
                workspace_id=UUID(str(playbook.workspace_id)),
                input_data={},
            )
        except Exception as e:
            logger.error("[PlaybookScheduler] Failed to fire playbook %d: %s", playbook_id, e, exc_info=True)
        finally:
            db.close()

    # ------------------------------------------------------------------
    # PRD-204 S4: benched notification
    # ------------------------------------------------------------------

    async def _notify_playbook_benched(self, db, playbook, threshold: int) -> None:
        """Dispatch ``playbook_benched`` -- the scheduler skipped a cron fire
        because the repeated-failure breaker is open. Workspace-wide (a
        scheduled playbook has no single requesting user). Never raises
        into the fire path.

        The dispatcher never commits (it joins the caller's transaction);
        the commit below is THIS caller's: ``_fire_playbook`` owns a
        scheduler-local session whose only pending write on the skip path
        is the notification row, and the session closes right after.
        """
        try:
            from core.services.notification_dispatcher import NotificationDispatcher

            dispatcher = NotificationDispatcher(db, str(playbook.workspace_id))
            await dispatcher.dispatch(
                event_type="playbook_benched",
                title=f"Playbook benched: {playbook.name}",
                message=(
                    f"The last {threshold} runs all failed, so scheduled runs "
                    f"are paused. Fix the cause and run it manually once to "
                    f"re-enable the schedule."
                ),
                link_type="playbook",
                link_id=str(playbook.id),
                status="warning",
            )
            db.commit()
        except Exception:
            logger.error(
                "[PlaybookScheduler] playbook_benched dispatch failed for %s",
                getattr(playbook, "id", "?"),
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return status of all scheduled cron playbook jobs."""
        if not self._scheduler:
            return {"active": False, "jobs": []}

        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "next_run_at": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            })

        return {
            "active": self._scheduler.running if self._scheduler else False,
            "jobs": jobs,
        }


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_playbook_scheduler: Optional[PlaybookSchedulerService] = None


def get_playbook_scheduler() -> PlaybookSchedulerService:
    global _playbook_scheduler
    if _playbook_scheduler is None:
        _playbook_scheduler = PlaybookSchedulerService()
    return _playbook_scheduler
