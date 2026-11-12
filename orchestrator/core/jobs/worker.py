"""The worker loop (PRD-276 §6.3): claim due jobs, run their handlers, renew leases, record outcomes.

PRD-161's dispatch loop, generalised to any registered kind. Each tick sweeps expired leases, then
claims, for each kind, as many jobs as the kind's concurrency leaves room for in this process, and
runs each as its own task. A running job's lease is renewed every third of its length; a handler that
raises is logged, recorded, and its job goes back in the queue at the kind's next delay, or dead.
Database work runs off the event loop. Wave 1a polls (``JOBS_POLL_SECONDS``); with no kind
registered a tick does nothing at all.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
import socket
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Set, Tuple

from core.jobs import queue
from core.jobs.registry import JobKind, kind_of, registered

logger = logging.getLogger(__name__)

PRUNE_EVERY_TICKS = 1800  # about an hour at the default poll
RETRY_HANDLERS_EVERY_TICKS = 30  # about a minute: a handler module that failed to load is tried again
# The modules whose ``@job_handler``s define the kinds, imported when a worker starts so that every
# worker process can claim every kind, whichever role it has (PRD-276 §6.7: a worker service in
# Wave 4 has no scheduler and serves no route that would import them otherwise).
HANDLER_MODULES = (
    "services.run_callbacks",  # run_callback.deliver (Wave 1b), run_callback.sweep (Wave 2c)
    "services.destinations.delivery_jobs",  # widget_destination.deliver (Wave 1b)
    "services.credits.jobs",  # credits.meter, credits.overhead, credits.expiry (Wave 2b)
    "services.end_user_retention",  # end_users.retention_sweep (Wave 2c)
    "services.push.receipts",  # push.receipts (Wave 2c)
    "services.mobile.digest",  # mobile.digest (Wave 2c)
    "services.memory_jobs",  # the eight memory jobs (Wave 2c)
)
MIN_POLL_SECONDS = 0.5  # a zero or negative JOBS_POLL_SECONDS would spin the loop
WORKER_ROLE = "worker"


def on_own_session(work: Callable[..., Any], *args: Any) -> Any:
    """``work(db, *args)`` on a session of its own, always closed (threadpool)."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        return work(db, *args)
    finally:
        db.close()


class JobWorker:
    """Runs the registered kinds' jobs in this process."""

    def __init__(self, worker_id: Optional[str] = None, missing_handlers: Tuple[str, ...] = ()) -> None:
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self._inflight: Dict[str, Set[asyncio.Task]] = defaultdict(set)
        self._ticks = 0
        self.missing_handlers = tuple(missing_handlers)  # modules that failed to load, retried until they do

    def inflight(self) -> int:
        return sum(len(tasks) for tasks in self._inflight.values())

    async def tick(self) -> int:
        """One pass: sweep, then claim and start what fits. Answers how many jobs it started."""
        kinds = registered()
        if not kinds:
            return 0
        await asyncio.to_thread(on_own_session, queue.sweep_expired)
        started = 0
        for kind in kinds:
            spec = kind_of(kind)
            room = spec.concurrency - len(self._inflight[kind])
            jobs = await asyncio.to_thread(on_own_session, queue.claim, spec, self.worker_id, room) if room > 0 else []
            for job in jobs:
                task = asyncio.create_task(self._run(spec, job))
                self._inflight[kind].add(task)
                task.add_done_callback(self._inflight[kind].discard)
                started += 1
        return started

    async def _run(self, spec: JobKind, job: queue.Job) -> None:
        renewing = asyncio.create_task(self._keep_renewed(spec, job))
        try:
            await spec.handler(job)
        except Exception as exc:  # noqa: BLE001 — logged, recorded, and the attempt is failed below
            renewing.cancel()
            logger.exception("[jobs] %s %s failed on attempt %d", job.kind, job.id, job.attempts)
            await asyncio.to_thread(_record, job, exc)
            await asyncio.to_thread(on_own_session, queue.fail, job, f"{type(exc).__name__}: {exc}")
            return
        finally:
            renewing.cancel()  # however the handler stopped, cancelled included: its lease is no longer renewed
        if not await asyncio.to_thread(on_own_session, queue.complete, job):
            logger.info("[jobs] %s %s was finished by its handler or lost its claim", job.kind, job.id)

    async def _keep_renewed(self, spec: JobKind, job: queue.Job) -> None:
        while True:
            await asyncio.sleep(max(1.0, spec.lease_seconds / 3))
            if not await asyncio.to_thread(on_own_session, queue.renew, job, spec.lease_seconds):
                logger.warning("[jobs] %s %s lost its claim; its handler's fenced writes will refuse", job.kind, job.id)
                return

    async def run(self, stop: asyncio.Event, poll_seconds: float, retention_days: int) -> None:
        """Ticks until ``stop`` is set, waiting at least ``MIN_POLL_SECONDS`` between ticks; prunes
        finished jobs about once an hour."""
        poll_seconds = max(MIN_POLL_SECONDS, poll_seconds)
        logger.info("[jobs] worker %s started", self.worker_id)
        while not stop.is_set():
            try:
                await self.tick()
                self._ticks += 1
                if self._ticks % PRUNE_EVERY_TICKS == 0:
                    await asyncio.to_thread(on_own_session, queue.prune, retention_days)
                if self.missing_handlers and self._ticks % RETRY_HANDLERS_EVERY_TICKS == 0:
                    self.missing_handlers = load_handlers(self.missing_handlers)
            except Exception:  # noqa: BLE001 — logged; the next tick tries again
                logger.exception("[jobs] worker tick failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                continue


def _record(job: queue.Job, exc: Exception) -> None:
    try:
        from core.utils.exception_telemetry import record_error

        record_error(subsystem="jobs", operation=job.kind, error=exc, workspace_id=job.workspace_id)
    except Exception:  # noqa: BLE001 — telemetry never stops the retry
        logger.warning("[jobs] could not record the failure of %s", job.id, exc_info=True)


def load_handlers(modules: Tuple[str, ...] = HANDLER_MODULES) -> Tuple[str, ...]:
    """Imports ``modules``, so their kinds are registered here; answers those that failed. Each failure
    is logged; the worker runs the rest and tries the failed ones again (``RETRY_HANDLERS_EVERY_TICKS``)."""
    import importlib

    failed = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001 — logged; retried, and its kinds stay queued until it loads
            logger.exception("[jobs] could not load the job handlers in %s; retrying", module)
            failed.append(module)
    if modules != HANDLER_MODULES and not failed:
        logger.info("[jobs] job handlers loaded at last: %s", ", ".join(modules))
    return tuple(failed)


def roles() -> Set[str]:
    """This process's roles (``PROCESS_ROLES``, §6.1): all three by default."""
    from config import config

    return {r.strip() for r in str(config.PROCESS_ROLES).split(",") if r.strip()}


def also_starts_the_job_worker(start_board_dispatch: Callable[..., None]) -> Callable[..., None]:
    """Wraps ``main._start_board_dispatch``: after it, a process with the ``worker`` role starts the job
    worker the same way, once trust passed, unless ``JOBS_ENABLED`` is off."""

    @functools.wraps(start_board_dispatch)
    def started(app_instance: Any, trust_passed: bool) -> None:
        start_board_dispatch(app_instance, trust_passed)
        from config import config

        if not (trust_passed and config.JOBS_ENABLED and WORKER_ROLE in roles()):
            return
        missing = load_handlers()
        app_instance.state.job_worker_stop = asyncio.Event()
        app_instance.state.job_worker_task = asyncio.create_task(JobWorker(missing_handlers=missing).run(
            app_instance.state.job_worker_stop, float(config.JOBS_POLL_SECONDS), int(config.JOBS_RETENTION_DAYS)))
        from core.jobs.metrics import observe_queue

        observe_queue()  # §10-7: queue depth and wait, from the worker role only
        logger.info("Job worker started (PRD-276)")

    return started
