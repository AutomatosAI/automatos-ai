"""
MemoryJobScheduler — Background Jobs for Unified Memory (PRD-79)
================================================================
Eight recurring memory jobs, each a job kind with a platform schedule (PRD-276 Wave 2c): the
scheduler only queues a slot, and a worker runs it. Each catches up ``latest``: every job sweeps
all workspaces, so one run after downtime does what the missed runs would have.

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

4. **Archival** (monthly), 5. **audit retention** (PRD-196 S5), 6. **Qdrant snapshot** (daily,
   PRD-197 S3), 7. **substrate metrics prune** (PRD-197 S4) and 8. **thread checkpoints**
   (PRD-206 S2).

All are resilient: one workspace failure does not stop others.
Each job logs start/end timestamps and summary metrics.
"""

import logging
from datetime import timezone
from typing import Any, Callable, Optional, Tuple

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.jobs.registry import job_handler

logger = logging.getLogger(__name__)

# (the status key, the job kind, the runner method, the config switch that turns it on or None)
_JOBS: Tuple[Tuple[str, str, str, Optional[str]], ...] = (
    ("memory_session_consolidation", "memory.consolidation", "_run_consolidation", None),
    ("memory_decay_scoring", "memory.decay", "_run_decay", None),
    ("memory_l2_l3_promotion", "memory.promotion", "_run_promotion", None),
    ("memory_graphify_archival", "memory.archival", "_run_archival", "MEMORY_ARCHIVAL_ENABLED"),
    ("audit_retention_sweep", "audit.retention_sweep", "_run_audit_retention", None),  # PRD-196 S5
    ("memory_qdrant_snapshot", "memory.snapshot", "_run_snapshot", "MEMORY_SNAPSHOT_ENABLED"),  # PRD-197 S3
    ("substrate_metrics_prune", "substrate.metrics_prune", "_run_substrate_prune", None),  # PRD-197 S4
    ("thread_checkpoint_sweep", "memory.thread_checkpoints", "_run_thread_checkpoints",
     "THREAD_CHECKPOINT_ENABLED"),  # PRD-206 S2
)


def _trigger(kind: str, config: Any) -> Any:
    """Each job's trigger, from its settings (the defaults are the ones it always had)."""
    from apscheduler.triggers.cron import CronTrigger

    from core.jobs.schedules import interval

    def setting(name: str, default: int) -> int:
        return int(getattr(config, name, default))

    triggers: dict = {
        "memory.consolidation": lambda: interval(seconds=setting("MEMORY_CONSOLIDATION_INTERVAL_SECONDS", 3600)),
        "memory.decay": lambda: interval(seconds=setting("MEMORY_DECAY_INTERVAL_SECONDS", 3600)),
        "memory.promotion": lambda: CronTrigger(hour=setting("MEMORY_PROMOTION_HOUR_UTC", 3), minute=0,
                                                timezone=timezone.utc),
        "memory.archival": lambda: CronTrigger(day=setting("MEMORY_ARCHIVAL_CRON_DAY", 1),
                                               hour=setting("MEMORY_ARCHIVAL_CRON_HOUR", 3), minute=0,
                                               timezone=timezone.utc),
        "audit.retention_sweep": lambda: interval(seconds=setting("AUDIT_RETENTION_SWEEP_INTERVAL_SECONDS", 86400)),
        "memory.snapshot": lambda: CronTrigger(hour=setting("MEMORY_SNAPSHOT_CRON_HOUR_UTC", 4), minute=0,
                                               timezone=timezone.utc),
        "substrate.metrics_prune": lambda: interval(seconds=setting("SUBSTRATE_METRICS_PRUNE_INTERVAL_SECONDS", 86400)),
        "memory.thread_checkpoints": lambda: interval(
            seconds=setting("THREAD_CHECKPOINT_SWEEP_INTERVAL_SECONDS", 900)),
    }
    return triggers[kind]()


def _runs(method: str) -> Callable[[Any], Any]:
    async def handler(job: Any) -> None:
        await getattr(get_memory_job_scheduler(), method)()

    handler.__name__ = f"run{method}"
    return handler


for _status_key, _kind, _method, _switch in _JOBS:  # registered on import, in every worker process
    job_handler(_kind, concurrency=1, workspace_concurrency=1)(_runs(_method))


class MemoryJobScheduler:
    """Declares the memory jobs' schedules on the shared APScheduler, and runs each job."""

    JOB_ID_CONSOLIDATION = "memory_session_consolidation"
    JOB_ID_DECAY = "memory_decay_scoring"
    JOB_ID_PROMOTION = "memory_l2_l3_promotion"
    JOB_ID_ARCHIVAL = "memory_graphify_archival"
    JOB_ID_AUDIT_RETENTION = "audit_retention_sweep"  # PRD-196 S5
    JOB_ID_SNAPSHOT = "memory_qdrant_snapshot"  # PRD-197 S3
    JOB_ID_SUBSTRATE_PRUNE = "substrate_metrics_prune"  # PRD-197 S4
    JOB_ID_THREAD_CHECKPOINT = "thread_checkpoint_sweep"  # PRD-206 S2

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, scheduler: AsyncIOScheduler):
        """Declare each enabled memory job's platform schedule on the shared scheduler."""
        from config import config as app_config
        from core.jobs.schedules import PLATFORM, schedule

        self._scheduler = scheduler
        declared = []
        for status_key, kind, _, switch in _JOBS:
            if switch is not None and not getattr(app_config, switch, True):
                continue
            schedule(scheduler, kind, PLATFORM, _trigger(kind, app_config))
            declared.append(status_key)
        logger.info("[MemoryJobs] Started — scheduled: %s", ", ".join(declared))

    async def stop(self):
        """Remove the memory jobs' schedules from the scheduler."""
        if not self._scheduler:
            return
        from core.jobs.schedules import PLATFORM, unschedule

        for _, kind, _, _ in _JOBS:
            unschedule(self._scheduler, kind, PLATFORM)
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

    async def _run_audit_retention(self):
        """PRD-196 S5: hard-delete audit_logs rows past the retention floor and
        write one summary row per affected workspace. Resilient — a failure here
        never stops the other memory jobs (its own try/except, like the rest)."""
        try:
            from core.database.database import SessionLocal
            from services.audit_retention import sweep_expired_audit_logs

            db = SessionLocal()
            try:
                result = sweep_expired_audit_logs(db)
            finally:
                db.close()
            logger.info(
                "[MemoryJobs] Audit retention sweep: deleted=%d, workspaces=%d, cutoff=%s",
                result.get("total_deleted", 0),
                result.get("workspaces_affected", 0),
                result.get("cutoff"),
            )
        except Exception as e:
            logger.error(
                "[MemoryJobs] Audit retention sweep failed: %s",
                e,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Job: Qdrant memory snapshots (PRD-197 S3)
    # ------------------------------------------------------------------

    async def _run_substrate_prune(self):
        """PRD-197 S4: delete substrate_metric_events rows past retention.
        Fail-soft like the rest."""
        try:
            from config import config as app_config
            from core.database.database import SessionLocal
            from services.substrate_health import prune_substrate_metrics

            db = SessionLocal()
            try:
                deleted = prune_substrate_metrics(
                    db,
                    retention_days=getattr(
                        app_config, "SUBSTRATE_METRICS_RETENTION_DAYS", 14
                    ),
                )
            finally:
                db.close()
            if deleted:
                logger.info(
                    "[MemoryJobs] Substrate metrics prune: deleted=%d", deleted
                )
        except Exception as e:
            logger.error(
                "[MemoryJobs] Substrate metrics prune failed: %s", e,
                exc_info=True,
            )

    async def _run_thread_checkpoints(self):
        """PRD-206 S2: checkpoint recently-idle threads. Candidates: user-kind
        chats idle past the threshold (but touched inside the lookback), with
        enough messages, whose summary watermark is older than their last
        activity. Per-chat failures never stop the batch; fail-soft overall."""
        try:
            from sqlalchemy import text as sql_text

            from config import config as app_config
            from core.database.database import SessionLocal
            from modules.memory.thread_checkpoint import run_thread_checkpoint

            idle_minutes = getattr(app_config, "THREAD_CHECKPOINT_IDLE_MINUTES", 30)
            lookback_hours = getattr(app_config, "THREAD_CHECKPOINT_LOOKBACK_HOURS", 48)
            batch = getattr(app_config, "THREAD_CHECKPOINT_BATCH", 10)
            min_messages = getattr(app_config, "THREAD_CHECKPOINT_MIN_MESSAGES", 4)

            db = SessionLocal()
            try:
                rows = db.execute(
                    sql_text(
                        """
                        SELECT id, workspace_id FROM chats
                        WHERE kind = 'user'
                          AND workspace_id IS NOT NULL
                          AND updated_at <= NOW() - (:idle_min * INTERVAL '1 minute')
                          AND updated_at >= NOW() - (:lookback_h * INTERVAL '1 hour')
                          AND (
                            summary IS NULL
                            OR COALESCE((summary->>'checkpointed_at')::double precision, 0)
                               < EXTRACT(EPOCH FROM updated_at)
                          )
                          AND (
                            SELECT COUNT(*) FROM messages m WHERE m.chat_id = chats.id
                          ) >= :min_msgs
                        ORDER BY updated_at DESC
                        LIMIT :batch
                        """
                    ),
                    {
                        "idle_min": idle_minutes,
                        "lookback_h": lookback_hours,
                        "min_msgs": min_messages,
                        "batch": batch,
                    },
                ).fetchall()

                done = failed = 0
                for chat_id, workspace_id in rows:
                    try:
                        result = await run_thread_checkpoint(
                            db,
                            workspace_id=str(workspace_id),
                            chat_id=str(chat_id),
                            trigger="idle_sweep",
                            min_messages=min_messages,
                        )
                        if result.get("success"):
                            done += 1
                        elif not result.get("skipped"):
                            failed += 1
                    except Exception:
                        failed += 1
                        logger.error(
                            "[MemoryJobs] Thread checkpoint failed for chat=%s",
                            chat_id, exc_info=True,
                        )
            finally:
                db.close()
            if done or failed:
                logger.info(
                    "[MemoryJobs] Thread checkpoint sweep: candidates=%d done=%d failed=%d",
                    len(rows), done, failed,
                )
        except Exception as e:
            logger.error(
                "[MemoryJobs] Thread checkpoint sweep failed: %s", e, exc_info=True,
            )

    async def _run_snapshot(self):
        """PRD-197 S3: snapshot durable_memory + field_memory to the object
        store and prune to retention. Fail-soft like the rest — a snapshot
        failure is logged and never stops the sibling jobs."""
        try:
            from services.qdrant_snapshots import run_snapshot_cycle

            summary = await run_snapshot_cycle()
            failures = [c for c, s in summary.items() if "error" in s]
            if failures:
                logger.warning(
                    "[MemoryJobs] Qdrant snapshot cycle completed with "
                    "failures: %s", ", ".join(failures),
                )
            else:
                logger.info(
                    "[MemoryJobs] Qdrant snapshot cycle complete: %s",
                    {c: s.get("snapshot") for c, s in summary.items()},
                )
        except Exception as e:
            logger.error(
                "[MemoryJobs] Qdrant snapshot cycle failed: %s", e,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return job status for health checks, keyed by each job's status name."""
        if not self._scheduler:
            return {"active": False, "jobs": {}}
        from core.jobs.schedules import PLATFORM, job_id

        jobs = {}
        for status_key, kind, _, _ in _JOBS:
            job = self._scheduler.get_job(job_id(kind, PLATFORM))
            jobs[status_key] = {
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
