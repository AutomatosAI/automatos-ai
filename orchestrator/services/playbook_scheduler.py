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
from typing import Any, Dict, Optional
from uuid import UUID

from apscheduler.events import EVENT_JOB_MISSED
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger

from config import config

logger = logging.getLogger(__name__)

JOB_PREFIX = "playbook_cron_"
# What a schedule saved without a zone has always fired in: the servers run UTC.
SERVER_ZONE = "UTC"
SYNC_SCHEDULED, SYNC_REMOVED, SYNC_DEFERRED, SYNC_OFF = "scheduled", "removed", "deferred", "off"


def cron_trigger(expression: str, zone: str) -> CronTrigger:
    """The trigger for a cron schedule in ``zone``. A cron or zone the scheduler
    cannot use raises ValueError naming it (F132: both used to be swallowed)."""
    try:
        return CronTrigger.from_crontab(expression, timezone=zone)
    except KeyError as exc:  # pytz and zoneinfo raise KeyError subclasses for an unknown zone
        raise ValueError(f"unknown timezone '{zone}'") from exc
    except ValueError as exc:
        raise ValueError(f"invalid cron '{expression}': {exc}") from exc


def is_live_cron(schedule_config: Optional[Dict[str, Any]]) -> bool:
    sc = schedule_config or {}
    return sc.get("type") == "cron" and bool(sc.get("cron_expression")) and sc.get("enabled") is not False


def default_schedule_zone(db, workspace_id) -> str:
    """The zone a schedule saved without one fires in (F132): the workspace's
    orchestrator heartbeat timezone when set, else UTC."""
    from core.models.workspaces import Workspace

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    settings = (getattr(workspace, "settings", None) or {}) if workspace is not None else {}
    heartbeat = (settings.get("orchestrator") or {}).get("heartbeat") or {}
    return heartbeat.get("timezone") or SERVER_ZONE


def with_explicit_zone(schedule_config: Optional[Dict[str, Any]], db, workspace_id) -> Dict[str, Any]:
    """A copy of ``schedule_config`` whose cron names its zone, so what fires is
    what the owner saw (B35: a zone-less cron fired in the server's UTC)."""
    sc = dict(schedule_config or {})
    if sc.get("type") == "cron" and not sc.get("timezone"):
        sc["timezone"] = default_schedule_zone(db, workspace_id)
    return sc


def sync_playbook_schedule(playbook) -> str:
    """Make this worker's scheduler match the playbook's schedule (F132), for the
    UI route and Auto's tools alike.

    SYNC_SCHEDULED / SYNC_REMOVED where this worker hosts the scheduler;
    SYNC_DEFERRED where it does not (production runs several workers): the
    leader's reconcile tick registers the saved schedule within
    RECONCILE_INTERVAL_SECONDS; SYNC_OFF when scheduled runs are switched off.
    A cron or zone the scheduler cannot use raises ValueError.
    """
    if not config.RECIPE_SCHEDULER_ENABLED:
        return SYNC_OFF
    service = get_playbook_scheduler()
    sc = playbook.schedule_config or {}
    if not is_live_cron(sc):
        service.unschedule_playbook(playbook.id)
        return SYNC_REMOVED
    if not service.hosts_jobs:
        cron_trigger(sc["cron_expression"], sc.get("timezone") or SERVER_ZONE)
        return SYNC_DEFERRED
    service.schedule_playbook(playbook)
    return SYNC_SCHEDULED


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
        self._notify_tasks: set = set()

    @property
    def hosts_jobs(self) -> bool:
        """True in the worker that holds the scheduler lock (the leader)."""
        return self._scheduler is not None

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

        self._scheduler.add_listener(self._on_job_missed, EVENT_JOB_MISSED)
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

            count, zoned = 0, []
            for playbook in playbooks:
                sc = playbook.schedule_config or {}
                if not is_live_cron(sc):
                    continue
                try:
                    self.schedule_playbook(playbook)
                except ValueError as exc:
                    logger.error("[PlaybookScheduler] Playbook %d is not scheduled: %s", playbook.id, exc)
                    continue
                count += 1
                if (sc.get("timezone") or SERVER_ZONE) != SERVER_ZONE:
                    zoned.append(playbook.id)

            logger.info("[PlaybookScheduler] Loaded %d cron playbooks", count)
            if zoned:
                # F132: these fired in the server's UTC until now, whatever zone they were saved with.
                logger.warning(
                    "[PlaybookScheduler] %d schedules now fire in their saved zone, not the server's UTC: ids %s",
                    len(zoned), zoned,
                )
        except Exception as e:
            logger.error("[PlaybookScheduler] Failed to load cron playbooks: %s", e, exc_info=True)
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Schedule / unschedule
    # ------------------------------------------------------------------

    def schedule_playbook(self, playbook):
        """Add or replace the cron job for a playbook, in its saved zone (F132:
        the trigger was built without one, so it fired in the server's UTC). A
        cron or zone the scheduler cannot use raises ValueError."""
        sc = playbook.schedule_config or {}
        expr = sc.get("cron_expression")
        if not expr:
            logger.warning("[PlaybookScheduler] No cron_expression for playbook %d, skipping", playbook.id)
            return

        zone = sc.get("timezone") or SERVER_ZONE
        trigger = cron_trigger(expr, zone)
        job_id = f"{JOB_PREFIX}{playbook.id}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
        self._add_job(playbook, trigger)
        logger.info("[PlaybookScheduler] Scheduled playbook %d (%s) with cron '%s' in %s",
                    playbook.id, getattr(playbook, 'name', ''), expr, zone)

    def _add_job(self, playbook, trigger) -> None:
        self._scheduler.add_job(
            self._fire_playbook,
            trigger,
            id=f"{JOB_PREFIX}{playbook.id}",
            args=[playbook.id, str(playbook.workspace_id)],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            # F132: APScheduler's default grace is 1 s; a fire reached later is dropped
            # (and now reported by _on_job_missed).
            misfire_grace_time=config.PLAYBOOK_SCHEDULE_MISFIRE_GRACE_SECONDS,
        )

    def reconcile_with_db(self, db) -> Dict[str, int]:
        """The leader's tick (F132). A schedule saved on a worker that hosts no
        scheduler, or whose sync failed, is registered here; a changed one is
        replaced; one no longer live is removed. Idempotent."""
        if not self.hosts_jobs:
            return {}
        from core.models import WorkflowTemplate as WorkflowPlaybook

        wanted = {
            f"{JOB_PREFIX}{playbook.id}": playbook
            for playbook in db.query(WorkflowPlaybook).filter(
                WorkflowPlaybook.schedule_config.isnot(None),
                WorkflowPlaybook.workspace_id.isnot(None),
                WorkflowPlaybook.steps.isnot(None),
            ).all()
            if is_live_cron(playbook.schedule_config)
        }
        jobs = {job.id: job for job in self._scheduler.get_jobs() if str(job.id).startswith(JOB_PREFIX)}
        counts = {"added": 0, "changed": 0, "removed": 0}
        for job_id, playbook in wanted.items():
            sc = playbook.schedule_config
            try:
                trigger = cron_trigger(sc["cron_expression"], sc.get("timezone") or SERVER_ZONE)
            except ValueError as exc:
                logger.error("[PlaybookScheduler] Playbook %d is not scheduled: %s", playbook.id, exc)
                continue
            job = jobs.get(job_id)
            if job is not None and repr(job.trigger) == repr(trigger):
                continue
            self._add_job(playbook, trigger)
            counts["added" if job is None else "changed"] += 1
        for job_id in jobs.keys() - wanted.keys():
            self._scheduler.remove_job(job_id)
            counts["removed"] += 1
        if any(counts.values()):
            logger.info("[PlaybookScheduler] reconcile: %s", counts)
        return counts

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

            # PRD-222 US-005 — no background burn: a trial workspace gets no
            # scheduled playbook execution until converted. Visible skip.
            try:
                from core.models.workspaces import Workspace
                from services.trial_ledger import is_trial_active_workspace
                # PRD-234 S3: the local edition has no platform-paid trial credit to
                # protect (operator keys / the user's own Claude subscription) — the
                # onboarding trial record must not switch scheduled work off there.
                from config import config as _config
                _local_edition = getattr(_config, "AUTH_EDITION", "saas") == "local"

                _ws = db.query(Workspace).get(playbook.workspace_id)
                if not _local_edition and is_trial_active_workspace(_ws):
                    logger.warning(
                        "[PlaybookScheduler] Skipping playbook %d — trial workspace %s "
                        "(no background burn until converted)",
                        playbook_id, playbook.workspace_id,
                    )
                    return
            except Exception as _e:
                logger.debug("[PlaybookScheduler] trial-skip check failed for playbook %d: %s", playbook_id, _e)

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
                await self._notify_schedule_skipped(
                    db, playbook, f"the workspace was at its run limit ({concurrency.reason})")
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
    # F132: a scheduled run that did not start says so
    # ------------------------------------------------------------------

    def _on_job_missed(self, event) -> None:
        """EVENT_JOB_MISSED: APScheduler drops a fire it reaches after the grace,
        without a word (B34, B60). Log it and ring the owner."""
        job_id = str(getattr(event, "job_id", ""))
        if not job_id.startswith(JOB_PREFIX):
            return
        playbook_id = int(job_id[len(JOB_PREFIX):])
        due = getattr(event, "scheduled_run_time", None)
        when = f"at {due:%H:%M %Z} " if due is not None else ""
        reason = (f"it was due {when}and the scheduler reached it more than "
                  f"{config.PLAYBOOK_SCHEDULE_MISFIRE_GRACE_SECONDS} s late (busy or restarting)")
        logger.warning("[PlaybookScheduler] scheduled run skipped: playbook %d, %s", playbook_id, reason)
        try:
            task = asyncio.get_running_loop().create_task(self._notify_skipped_by_id(playbook_id, reason))
        except RuntimeError:
            logger.warning("[PlaybookScheduler] no event loop to report the skip of playbook %d", playbook_id)
            return
        self._notify_tasks.add(task)
        task.add_done_callback(self._notify_tasks.discard)

    async def _notify_skipped_by_id(self, playbook_id: int, reason: str) -> None:
        from core.database.database import SessionLocal
        from core.models import WorkflowTemplate as WorkflowPlaybook

        db = SessionLocal()
        try:
            playbook = db.query(WorkflowPlaybook).filter(WorkflowPlaybook.id == playbook_id).first()
            if playbook is not None:
                await self._notify_schedule_skipped(db, playbook, reason)
        finally:
            db.close()

    async def _notify_schedule_skipped(self, db, playbook, reason: str) -> None:
        """Dispatch ``playbook_schedule_skipped`` to the workspace. Never raises
        into the fire path; commits its own row like ``_notify_playbook_benched``."""
        try:
            from core.services.notification_dispatcher import NotificationDispatcher

            await NotificationDispatcher(db, str(playbook.workspace_id)).dispatch(
                event_type="playbook_schedule_skipped",
                title=f"Scheduled run skipped: {playbook.name}",
                message=f"The scheduled run did not start: {reason}.",
                link_type="playbook",
                link_id=str(playbook.id),
                status="warning",
            )
            db.commit()
        except Exception:
            logger.error("[PlaybookScheduler] playbook_schedule_skipped dispatch failed for %s",
                         getattr(playbook, "id", "?"), exc_info=True)

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
