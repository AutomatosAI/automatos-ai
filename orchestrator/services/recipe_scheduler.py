"""
RecipeSchedulerService
======================
Fires cron-scheduled recipes using APScheduler.

Follows the same pattern as HeartbeatService:
- APScheduler AsyncIOScheduler with memory (or Redis) job store
- Starts in main.py lifespan
- Loads all cron recipes on startup, adds/removes jobs on recipe create/update
"""

import asyncio
import logging
from typing import Optional
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


class RecipeSchedulerService:

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._owns_scheduler: bool = False  # True when we created our own scheduler (tests)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, scheduler: Optional[AsyncIOScheduler] = None):
        """Initialize scheduler and load all cron recipes from DB.

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
                    logger.info("[RecipeScheduler] Using Redis job store (standalone)")
            except Exception:
                pass
            self._scheduler = AsyncIOScheduler(jobstores=jobstores)
            self._scheduler.start()
            self._owns_scheduler = True

        await self._load_cron_recipes()
        logger.info("[RecipeScheduler] Service started")

    async def stop(self):
        """Remove recipe jobs. Only shuts down scheduler if we own it."""
        if self._scheduler and self._owns_scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("[RecipeScheduler] Standalone scheduler stopped")
        logger.info("[RecipeScheduler] Service stopped")

    # ------------------------------------------------------------------
    # Load from DB
    # ------------------------------------------------------------------

    async def _load_cron_recipes(self):
        """Query all cron-type recipes and schedule them."""
        from core.database.database import SessionLocal
        from core.models import WorkflowTemplate as WorkflowRecipe
        from sqlalchemy import text

        db = SessionLocal()
        try:
            recipes = db.query(WorkflowRecipe).filter(
                WorkflowRecipe.schedule_config.isnot(None),
                WorkflowRecipe.workspace_id.isnot(None),
                WorkflowRecipe.steps.isnot(None),
            ).all()

            count = 0
            for recipe in recipes:
                sc = recipe.schedule_config or {}
                if sc.get("type") != "cron":
                    continue
                expr = sc.get("cron_expression")
                if not expr:
                    continue
                self.schedule_recipe(recipe)
                count += 1

            logger.info("[RecipeScheduler] Loaded %d cron recipes", count)
        except Exception as e:
            logger.error("[RecipeScheduler] Failed to load cron recipes: %s", e, exc_info=True)
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Schedule / unschedule
    # ------------------------------------------------------------------

    def schedule_recipe(self, recipe):
        """Add or replace APScheduler job for a cron recipe."""
        sc = recipe.schedule_config or {}
        expr = sc.get("cron_expression")
        if not expr:
            logger.warning("[RecipeScheduler] No cron_expression for recipe %d, skipping", recipe.id)
            return

        job_id = f"recipe_cron_{recipe.id}"

        try:
            trigger = CronTrigger.from_crontab(expr)
        except ValueError as e:
            logger.error("[RecipeScheduler] Invalid cron expression '%s' for recipe %d: %s", expr, recipe.id, e)
            return

        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

        self._scheduler.add_job(
            self._fire_recipe,
            trigger,
            id=job_id,
            args=[recipe.id, str(recipe.workspace_id)],
            replace_existing=True,
            max_instances=1,
        )
        logger.info("[RecipeScheduler] Scheduled recipe %d (%s) with cron '%s'", recipe.id, getattr(recipe, 'name', ''), expr)

    def unschedule_recipe(self, recipe_id: int):
        """Remove a scheduled cron job by recipe id."""
        job_id = f"recipe_cron_{recipe_id}"
        if self._scheduler and self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
            logger.info("[RecipeScheduler] Unscheduled recipe %d", recipe_id)

    # ------------------------------------------------------------------
    # Fire
    # ------------------------------------------------------------------

    async def _fire_recipe(self, recipe_id: int, workspace_id: str):
        """Execute a cron-triggered recipe. Same pattern as webhook handler."""
        from core.database.database import SessionLocal
        from core.models import WorkflowTemplate as WorkflowRecipe
        from core.models.core import RecipeExecution
        from uuid import uuid4

        db = SessionLocal()
        try:
            recipe = db.query(WorkflowRecipe).filter(WorkflowRecipe.id == recipe_id).first()
            if not recipe:
                logger.warning("[RecipeScheduler] Recipe %d no longer exists, unscheduling", recipe_id)
                self.unschedule_recipe(recipe_id)
                return

            # Skip if schedule was changed away from cron
            sc = recipe.schedule_config or {}
            if sc.get("type") != "cron":
                logger.info("[RecipeScheduler] Recipe %d is no longer cron-type, unscheduling", recipe_id)
                self.unschedule_recipe(recipe_id)
                return

            if not recipe.steps:
                logger.warning("[RecipeScheduler] Recipe %d has no steps, skipping", recipe_id)
                return

            execution_id = f"cron-{uuid4().hex[:12]}"
            execution = RecipeExecution(
                execution_id=execution_id,
                recipe_id=recipe.id,
                workspace_id=recipe.workspace_id,
                status="pending",
                input_data={},
                triggered_by="cron_scheduler",
                execution_metadata={
                    "execution_type": "cron_scheduler",
                    "total_steps": len(recipe.steps),
                    "cron_expression": sc.get("cron_expression"),
                },
            )
            db.add(execution)
            db.commit()

            # Concurrency guard — skip this tick if workspace is at capacity
            from services.concurrency_guard import check_concurrency
            concurrency = await check_concurrency(UUID(str(recipe.workspace_id)), db)
            if not concurrency.allowed:
                logger.warning(
                    "[RecipeScheduler] Concurrency limit reached for workspace %s, "
                    "skipping recipe %d this tick: %s",
                    workspace_id, recipe_id, concurrency.reason,
                )
                # Roll back the pending execution record — cron will retry next tick
                db.delete(execution)
                db.commit()
                return

            logger.info("[RecipeScheduler] Firing recipe %d (%s), execution=%s", recipe.id, recipe.name, execution_id)

            from api.recipe_executor import launch_recipe_task
            launch_recipe_task(
                recipe_execution_id=execution_id,
                recipe_id=recipe.id,
                workspace_id=UUID(str(recipe.workspace_id)),
                input_data={},
            )
        except Exception as e:
            logger.error("[RecipeScheduler] Failed to fire recipe %d: %s", recipe_id, e, exc_info=True)
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return status of all scheduled cron recipe jobs."""
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

_recipe_scheduler: Optional[RecipeSchedulerService] = None


def get_recipe_scheduler() -> RecipeSchedulerService:
    global _recipe_scheduler
    if _recipe_scheduler is None:
        _recipe_scheduler = RecipeSchedulerService()
    return _recipe_scheduler
