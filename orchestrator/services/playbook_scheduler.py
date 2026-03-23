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

            from api.recipe_executor import launch_recipe_task
            launch_recipe_task(
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
