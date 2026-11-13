"""Who runs the shared scheduler, and what runs on it (PRD-276 §6.4, Wave 2a).

At boot (``main._boot_phase_2_extensions``), a process with the ``scheduler`` role
(``PROCESS_ROLES``) tries the scheduler lock (``core/jobs/leadership.py``). The one that holds it
starts the unified scheduler and every scheduled service on it; the rest keep trying, and the
first to get the lock after the holder dies starts them in its place. This replaces the
``/tmp/automatos_scheduler.lock`` file lock, which held within one container only.

Each service starts on its own: one that fails is logged and the rest still start.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEDULER_ROLE = "scheduler"
WORKER_ROLE = "worker"
NOTHING_RUNS_THE_QUEUE = ("JOBS_ENABLED is off: the scheduler queues scheduled work, credit metering among it, "
                          "and no worker will run it")

# (the config switch that turns it on, or None for always; its name; its module; the getter whose
# result has ``start(scheduler=...)``), in the order main.py started them.
_Service = Tuple[Optional[str], str, str, str]
_BEFORE_SCHEDULED_TASKS: Tuple[_Service, ...] = (
    ("HEARTBEAT_ENABLED", "HeartbeatService", "services.heartbeat_service", "get_heartbeat_service"),
    ("RECIPE_SCHEDULER_ENABLED", "PlaybookSchedulerService", "services.playbook_scheduler", "get_playbook_scheduler"),
    (None, "TaskReconciler", "services.task_reconciler", "get_task_reconciler"),  # PRD-72
    ("WATCHER_ENABLED", "WatchTicker", "services.watch_ticker", "get_watch_ticker"),  # PRD-204 S5
    ("MEMORY_JOBS_ENABLED", "MemoryJobScheduler", "services.memory_jobs", "get_memory_job_scheduler"),  # PRD-79
    ("FIELD_PROMOTION_ENABLED", "FieldPromotionJobScheduler", "jobs.promote_field_memory",
     "get_field_promotion_scheduler"),  # PRD-178 S4
    (None, "EdgeBuilderScheduler", "services.edge_builder_scheduler", "get_edge_builder_scheduler"),  # PRD-139
    ("COMPOSIO_SYNC_ENABLED", "ComposioSyncScheduler", "services.composio_sync_scheduler",
     "get_composio_sync_scheduler"),  # PRD-177 S3
)
_AFTER_SCHEDULED_TASKS: Tuple[_Service, ...] = (
    ("COORDINATOR_ENABLED", "CoordinatorService", "services.coordinator_service", "get_coordinator_service"),  # PRD-82A
    ("HARNESS_ENABLED", "HarnessService", "services.harness_service", "get_harness_service"),  # PRD-121
)


def scheduler_wanted() -> bool:
    """A scheduled service is on, and this process has the ``scheduler`` role."""
    from config import config
    from core.jobs.worker import roles

    services_on = config.HEARTBEAT_ENABLED or config.RECIPE_SCHEDULER_ENABLED or config.COORDINATOR_ENABLED
    return bool(services_on) and SCHEDULER_ROLE in roles()


async def start_scheduler_leadership() -> bool:
    """Tries the scheduler lock now, and keeps trying in the background; True when this process
    leads. False, and nothing started, when it doesn't want a scheduler at all."""
    if not scheduler_wanted():
        logger.info("[scheduler] no scheduled service on, or no scheduler role here; not running one")
        return False
    from config import config
    from services.scheduler import get_unified_scheduler

    check_the_queue_runs()
    leading = await get_unified_scheduler().lead(start_scheduled_services, float(config.SCHEDULER_LEADER_CHECK_SECONDS))
    if not leading:
        logger.info("[scheduler] another process holds the scheduler lock; this one waits to take over")
    return leading


def check_the_queue_runs() -> bool:
    """The scheduler only queues (PRD-276 §6.4), so something must run the queue: an error, logged and
    recorded, when ``JOBS_ENABLED`` is off; a warning when this process has no worker role, since
    another process must have it. True when this process runs the queue itself."""
    from config import config
    from core.jobs.worker import roles

    if not config.JOBS_ENABLED:
        from core.utils.exception_telemetry import record_error

        logger.error("[scheduler] %s", NOTHING_RUNS_THE_QUEUE)
        record_error(subsystem="jobs", operation="start_scheduler", error=RuntimeError(NOTHING_RUNS_THE_QUEUE))
        return False
    if WORKER_ROLE not in roles():
        logger.warning("[scheduler] this process has no worker role: another process must run the worker "
                       "role with JOBS_ENABLED on, or scheduled work (credit metering among it) is never run")
        return False
    return True


async def start_scheduled_services(scheduler: Any) -> None:
    """Every scheduled service, on the shared ``scheduler``, each guarded on its own."""
    for service in _BEFORE_SCHEDULED_TASKS:
        await _start_service(scheduler, *service)
    await _guarded("scheduled tasks", _load_scheduled_tasks)
    await _guarded("schedule reconcile tick", _start_reconcile, scheduler)
    for service in _AFTER_SCHEDULED_TASKS:
        await _start_service(scheduler, *service)
    logger.info("Unified scheduler started (this worker owns it)")


async def _start_service(scheduler: Any, switch: Optional[str], name: str, module: str, getter: str) -> None:
    from config import config

    if switch is not None and not getattr(config, switch, False):
        return

    async def start() -> None:
        await getattr(importlib.import_module(module), getter)().start(scheduler=scheduler)

    if await _guarded(name, start):
        logger.info("%s started on unified scheduler", name)


async def _guarded(name: str, work: Any, *args: Any) -> bool:
    try:
        await work(*args)
    except Exception as err:  # noqa: BLE001 — logged; the other services still start
        logger.warning("Could not start %s: %s", name, err)
        return False
    return True


async def _load_scheduled_tasks() -> None:
    """PRD-77: agent-scheduled tasks loaded into the scheduler."""
    from core.database.database import SessionLocal
    from services.scheduled_task_service import ScheduledTaskService

    db = SessionLocal()
    try:
        await ScheduledTaskService(db, workspace_id=None).load_active_tasks_to_scheduler()
    finally:
        db.close()


async def _start_reconcile(scheduler: Any) -> None:
    """Scheduled-task and heartbeat changes made on any process reach this scheduler within a minute."""
    from services.schedule_reconcile import start_schedule_reconcile

    await start_schedule_reconcile(scheduler)
