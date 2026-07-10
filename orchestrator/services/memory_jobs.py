"""
MemoryJobScheduler — Background Jobs for Unified Memory (PRD-79)
================================================================
Registers three recurring jobs on the UnifiedScheduler:

1. **Consolidation** (periodic) — contradiction-based invalidation (PRD-159 S4)
   Per workspace, near-duplicate L3 memories merge into one canonical and
   contradictions supersede by recency+confidence (loser removed). This is the
   primary memory lifecycle, replacing the dead L1 session-decision scan.

2. **Decay Scoring** (hourly) — Ebbinghaus retention scoring on L2
   Updates decay_score for all active L2 rows and archives items
   below the retention threshold (default 0.3).

3. **L2→L3 Promotion** (daily) — Promote important L2 items to L3
   Items meeting the type-aware importance policy (PRD-187 S4 —
   no access-count gate) are stored verbatim in the durable store,
   deduped by content hash.

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
    JOB_ID_ARCHIVAL = "memory_graphify_archival"

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

        # Periodic: contradiction-based consolidation (PRD-159 S4)
        self._scheduler.add_job(
            self._run_consolidation,
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

        # Monthly: graphify archival (L2+L3 → workspace knowledge graph)
        archival_enabled = getattr(app_config, "MEMORY_ARCHIVAL_ENABLED", True)
        archival_day = getattr(app_config, "MEMORY_ARCHIVAL_CRON_DAY", 1)
        archival_hour = getattr(app_config, "MEMORY_ARCHIVAL_CRON_HOUR", 3)
        if archival_enabled:
            self._scheduler.add_job(
                self._run_archival,
                "cron",
                day=archival_day,
                hour=archival_hour,
                minute=0,
                id=self.JOB_ID_ARCHIVAL,
                replace_existing=True,
                max_instances=1,
            )

        logger.info(
            "[MemoryJobs] Started — consolidation every %ds, "
            "decay every %ds, promotion daily at %02d:00 UTC, "
            "archival %s (day=%d hour=%02d)",
            consolidation_interval,
            decay_interval,
            promotion_hour,
            "enabled" if archival_enabled else "disabled",
            archival_day,
            archival_hour,
        )

    async def stop(self):
        """Remove all memory jobs from the scheduler."""
        if not self._scheduler:
            return
        for job_id in (
            self.JOB_ID_CONSOLIDATION,
            self.JOB_ID_DECAY,
            self.JOB_ID_PROMOTION,
            self.JOB_ID_ARCHIVAL,
        ):
            if self._scheduler.get_job(job_id):
                self._scheduler.remove_job(job_id)
        logger.info("[MemoryJobs] Stopped")

    # ------------------------------------------------------------------
    # Job: Session Consolidation (L1 → L2)
    # ------------------------------------------------------------------

    async def _run_consolidation(self):
        """PRD-159 S4: contradiction-based consolidation — merge near-duplicates
        and supersede contradictions (recency+confidence) across workspaces. The
        primary memory lifecycle, replacing the dead L1 session-decision scan."""
        try:
            from modules.memory.unified_memory_service import (
                get_unified_memory_service,
            )

            service = get_unified_memory_service()
            result = await service.run_sleep_time_consolidation()
            logger.info(
                "[MemoryJobs] Consolidation complete: "
                "workspaces=%d, merged=%d, superseded=%d, errors=%d",
                result.get("workspaces_processed", 0),
                result.get("merged", 0),
                result.get("superseded", 0),
                result.get("errors", 0),
            )
        except Exception as e:
            logger.error(
                "[MemoryJobs] Consolidation failed: %s",
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
        """Promote policy-eligible L2 items to L3 (durable store)."""
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
    # Job: Graphify Archival (L2+L3 → workspace knowledge graph)
    # ------------------------------------------------------------------

    async def _run_archival(self):
        """Fold aged L2+L3 memories into the workspace graph, then purge."""
        try:
            from services.memory_archival_job import MemoryArchivalJob

            result = await MemoryArchivalJob().run_once()
            logger.info(
                "[MemoryJobs] Graphify archival complete: "
                "workspaces=%d, with_candidates=%d, imported=%d, "
                "l2_archived=%d, l3_archived=%d, errors=%d",
                result.get("workspaces_processed", 0),
                result.get("workspaces_with_candidates", 0),
                result.get("nodes_imported", 0),
                result.get("l2_archived", 0),
                result.get("l3_archived", 0),
                result.get("errors", 0),
            )
        except Exception as e:
            logger.error(
                "[MemoryJobs] Graphify archival failed: %s",
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
            self.JOB_ID_ARCHIVAL,
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
