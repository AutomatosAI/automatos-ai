"""
MemoryJobScheduler — Background Jobs for Unified Memory (PRD-79)
================================================================
Registers three recurring jobs on the UnifiedScheduler:

1. **Session Consolidation** (hourly) — L1 Redis → L2 Postgres
   Expired sessions (end_session called >1hr ago) are scanned, decisions
   and action items extracted, stored in L2, and L1 keys deleted.

2. **Decay Scoring** (hourly) — Ebbinghaus retention scoring on L2
   Updates decay_score for all active L2 rows and archives items
   below the retention threshold (default 0.3).

3. **L2→L3 Promotion** (daily) — Promote important L2 items to Mem0
   Items with high importance and access count get stored in L3 via
   Mem0 with infer=True for fact extraction and deduplication.

All three jobs are resilient: one workspace failure does not stop others.
Each job logs start/end timestamps and summary metrics.
"""

import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)


class MemoryJobScheduler:
    """Registers memory background jobs on the shared APScheduler."""

    JOB_ID_CONSOLIDATION = "memory_session_consolidation"
    JOB_ID_DECAY = "memory_decay_scoring"
    JOB_ID_PROMOTION = "memory_l2_l3_promotion"

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, scheduler: AsyncIOScheduler):
        """Register all three memory jobs on the shared scheduler."""
        from config import config as app_config

        self._scheduler = scheduler

        consolidation_interval = getattr(
            app_config, "MEMORY_CONSOLIDATION_INTERVAL_SECONDS", 3600
        )
        decay_interval = getattr(
            app_config, "MEMORY_DECAY_INTERVAL_SECONDS", 3600
        )
        promotion_hour = getattr(
            app_config, "MEMORY_PROMOTION_HOUR_UTC", 3
        )

        # Hourly: Session consolidation (L1 → L2)
        self._scheduler.add_job(
            self._run_session_consolidation,
            "interval",
            seconds=consolidation_interval,
            id=self.JOB_ID_CONSOLIDATION,
            replace_existing=True,
            max_instances=1,
        )

        # Hourly: Decay scoring on L2
        self._scheduler.add_job(
            self._run_decay,
            "interval",
            seconds=decay_interval,
            id=self.JOB_ID_DECAY,
            replace_existing=True,
            max_instances=1,
        )

        # Daily at promotion_hour UTC: L2 → L3 promotion
        self._scheduler.add_job(
            self._run_promotion,
            "cron",
            hour=promotion_hour,
            minute=0,
            id=self.JOB_ID_PROMOTION,
            replace_existing=True,
            max_instances=1,
        )

        logger.info(
            "[MemoryJobs] Started — consolidation every %ds, "
            "decay every %ds, promotion daily at %02d:00 UTC",
            consolidation_interval,
            decay_interval,
            promotion_hour,
        )

    async def stop(self):
        """Remove all memory jobs from the scheduler."""
        if not self._scheduler:
            return
        for job_id in (
            self.JOB_ID_CONSOLIDATION,
            self.JOB_ID_DECAY,
            self.JOB_ID_PROMOTION,
        ):
            if self._scheduler.get_job(job_id):
                self._scheduler.remove_job(job_id)
        logger.info("[MemoryJobs] Stopped")

    # ------------------------------------------------------------------
    # Job: Session Consolidation (L1 → L2)
    # ------------------------------------------------------------------

    async def _run_session_consolidation(self):
        """Scan expired L1 sessions, extract decisions, store in L2."""
        try:
            from modules.memory.unified_memory_service import (
                get_unified_memory_service,
            )

            service = get_unified_memory_service()
            result = await service.run_session_consolidation()
            logger.info(
                "[MemoryJobs] Session consolidation complete: "
                "scanned=%d, consolidated=%d, items=%d, errors=%d",
                result.get("sessions_scanned", 0),
                result.get("sessions_consolidated", 0),
                result.get("total_items", 0),
                result.get("errors", 0),
            )
        except Exception as e:
            logger.error(
                "[MemoryJobs] Session consolidation failed: %s",
                e,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Job: Decay Scoring (L2)
    # ------------------------------------------------------------------

    async def _run_decay(self):
        """Update decay scores and archive expired L2 items."""
        try:
            from modules.memory.unified_memory_service import (
                get_unified_memory_service,
            )

            service = get_unified_memory_service()
            result = await service.run_decay_all()
            logger.info(
                "[MemoryJobs] Decay scoring complete: "
                "workspaces=%d, decayed=%d, archived=%d, errors=%d",
                result.get("workspaces_processed", 0),
                result.get("total_decayed", 0),
                result.get("total_archived", 0),
                result.get("errors", 0),
            )
        except Exception as e:
            logger.error(
                "[MemoryJobs] Decay scoring failed: %s",
                e,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Job: L2 → L3 Promotion
    # ------------------------------------------------------------------

    async def _run_promotion(self):
        """Promote high-importance L2 items to L3 via Mem0."""
        try:
            from modules.memory.unified_memory_service import (
                get_unified_memory_service,
            )

            service = get_unified_memory_service()
            result = await service.run_promotion_all()
            logger.info(
                "[MemoryJobs] L2→L3 promotion complete: "
                "workspaces=%d, promoted=%d, errors=%d",
                result.get("workspaces_processed", 0),
                result.get("total_promoted", 0),
                result.get("errors", 0),
            )
        except Exception as e:
            logger.error(
                "[MemoryJobs] L2→L3 promotion failed: %s",
                e,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return job status for health checks."""
        if not self._scheduler:
            return {"active": False, "jobs": {}}

        jobs = {}
        for job_id in (
            self.JOB_ID_CONSOLIDATION,
            self.JOB_ID_DECAY,
            self.JOB_ID_PROMOTION,
        ):
            job = self._scheduler.get_job(job_id)
            jobs[job_id] = {
                "active": job is not None,
                "next_run": str(job.next_run_time) if job else None,
            }

        return {"active": True, "jobs": jobs}


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_memory_jobs: Optional[MemoryJobScheduler] = None


def get_memory_job_scheduler() -> MemoryJobScheduler:
    global _memory_jobs
    if _memory_jobs is None:
        _memory_jobs = MemoryJobScheduler()
    return _memory_jobs
