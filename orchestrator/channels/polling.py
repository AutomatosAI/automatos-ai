"""Channel polling has one owner: the worker that runs the unified scheduler.

The backend runs several uvicorn workers (``WEB_CONCURRENCY``). A polling-mode
connection (Telegram's getUpdates and the like) must be polled by exactly one
process: two pollers on one bot make Telegram answer 409 Conflict, and messages
are split between them or lost. Every worker used to start every polling
adapter at boot, and a connect, start, stop or delete started or stopped a
poller in whichever worker served the request.

Now the worker that holds the scheduler lock (``main.py``) owns them all:

* At boot it reconciles its adapters with the database (the active
  polling-mode rows) and registers a tick that does it again every
  ``CHANNEL_POLLING_RECONCILE_SECONDS``, so a change written on any worker
  reaches the owner within one interval (``services/schedule_reconcile.py``
  does the same for scheduled jobs).
* A route or tool that writes a channel row calls :func:`apply_polling_change`:
  on the owner the pass runs at once; any other worker starts and stops no
  poller and leaves the change to the owner's tick.
* After each pass the owner publishes the adapters it runs
  (``channels.polling_status``), so ``ChannelManager.is_running`` answers the
  same on every worker.

With the unified scheduler disabled by config there is no owner: every worker
polls for itself, as before, and boot says so in a warning.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Collection, Dict, Optional

from config import config

from . import polling_status
from .manager import AdapterSpec, ReconcileReport, get_channel_manager

logger = logging.getLogger(__name__)

RECONCILE_JOB_ID = "channel_polling_reconcile"

# One pass at a time on this worker: the database read, the adapters and the
# published set must come from the same pass, or a slower stale pass could
# restart an adapter a newer one had just stopped.
_PASS_LOCK = asyncio.Lock()


def _interval_seconds() -> int:
    return int(config.CHANNEL_POLLING_RECONCILE_SECONDS)


def _config_dict(raw: Any) -> Dict[str, Any]:
    """A row's config as a dict; the JSON column can hand back a string."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def load_polling_specs() -> Dict[str, AdapterSpec]:
    """The adapters that should run: one per active polling-mode connection.

    Webhook-mode rows are skipped: the platform POSTs to
    ``/api/webhooks/ws/{workspace_key}`` and no in-process adapter is needed; a
    poller for a webhook-mode bot would conflict on Telegram (409 on getUpdates).
    A database error raises: an empty answer would stop every bot.
    """
    from core.database.database import SessionLocal
    from core.models.channels import ChannelConnection

    db = SessionLocal()
    try:
        rows = (
            db.query(
                ChannelConnection.id,
                ChannelConnection.workspace_id,
                ChannelConnection.platform,
                ChannelConnection.config,
            )
            .filter(ChannelConnection.status == "active", ChannelConnection.mode == "polling")
            .all()
        )
    finally:
        db.close()
    return {
        str(row.id): AdapterSpec(str(row.workspace_id), row.platform, _config_dict(row.config))
        for row in rows
    }


async def reconcile_polling(*, retry: Collection[str] = ()) -> ReconcileReport:
    """One pass on this worker: read the desired set, converge the adapters,
    publish the ones running. Raises when the database read fails, leaving the
    running adapters as they were. Does nothing once this worker no longer owns
    polling (a pass that waited on the lock while shutdown stopped everything)."""
    manager = get_channel_manager()
    async with _PASS_LOCK:
        if not manager.owns_polling:
            return ReconcileReport()
        desired = await asyncio.to_thread(load_polling_specs)
        report = await manager.reconcile(desired, retry=retry)
        await asyncio.to_thread(
            polling_status.publish, manager.running_ids(), report.stopped + report.failed,
        )
    return report


async def _tick() -> None:
    """The owner's interval job. Registered without args: a Redis job store
    pickles them (``schedule_reconcile._tick``). That store also keeps the job
    across restarts, so a worker that does not own polling (channels turned off
    since) ignores it."""
    if not get_channel_manager().owns_polling:
        return
    try:
        await reconcile_polling()
    except Exception:
        logger.exception("[ChannelPolling] reconcile pass failed; the next tick retries")


def _scheduler_configured() -> bool:
    """main.py takes the scheduler lock only when one of these is on (the same test)."""
    return bool(config.HEARTBEAT_ENABLED or config.RECIPE_SCHEDULER_ENABLED or config.COORDINATOR_ENABLED)


def _scheduler_on_this_worker() -> Optional[Any]:
    """The unified scheduler when this worker won its lock and runs it, else None."""
    from services.scheduler import get_unified_scheduler

    scheduler = get_unified_scheduler().apscheduler
    return scheduler if scheduler is not None and getattr(scheduler, "running", False) else None


def _register_tick(scheduler: Any) -> bool:
    """The interval job on the shared scheduler. False when the job store refused
    it: the adapters still start, and changes then reach them only through
    requests this worker serves."""
    try:
        from apscheduler.triggers.interval import IntervalTrigger

        scheduler.add_job(
            _tick,
            IntervalTrigger(seconds=_interval_seconds()),
            id=RECONCILE_JOB_ID,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
    except Exception:
        logger.exception(
            "[ChannelPolling] could not register the reconcile tick: changes saved on other "
            "workers reach the pollers only at the next restart, and their status expires there"
        )
        return False
    logger.info("[ChannelPolling] this worker owns channel polling; reconcile every %ds", _interval_seconds())
    return True


async def _own_polling(scheduler: Optional[Any]) -> bool:
    """This worker runs its polling adapters: the first pass now and, given the
    shared scheduler, a tick. False when the first pass failed."""
    get_channel_manager().take_polling_ownership()
    if scheduler is not None:
        _register_tick(scheduler)
    try:
        await reconcile_polling()
    except Exception:
        logger.exception("[ChannelPolling] the first reconcile pass failed")
        return False
    return True


async def start_channel_polling() -> bool:
    """Boot, after Phase 2 started (or skipped) the unified scheduler: start the
    polling adapters on the worker that owns them. False when that failed."""
    if not config.CHANNELS_ENABLED:
        return True  # disabled by config — not a failure
    if not _scheduler_configured():
        logger.warning(
            "[ChannelPolling] the unified scheduler is disabled, so no worker owns channel "
            "polling: each worker polls every polling-mode channel itself. With more than "
            "one uvicorn worker each bot is polled once per worker (Telegram answers 409)."
        )
        return await _own_polling(scheduler=None)
    scheduler = _scheduler_on_this_worker()
    if scheduler is None:
        logger.info("[ChannelPolling] the scheduler worker owns channel polling; this worker runs no pollers")
        return True
    return await _own_polling(scheduler)


async def apply_polling_change(connection_id: str) -> bool:
    """After a route or tool wrote ``connection_id``'s row. True when this worker
    applied the change now.

    On the owner the pass runs at once, retrying that connection even if its last
    start failed. Any other worker starts and stops no poller: the owner's tick
    applies the change within one interval. Never raises: the row is written.
    """
    if not get_channel_manager().owns_polling:
        logger.info(
            "[ChannelPolling] channel %s saved; the polling worker applies it within %ds",
            connection_id, _interval_seconds(),
        )
        return False
    try:
        await reconcile_polling(retry=(connection_id,))
    except Exception:
        logger.exception("[ChannelPolling] could not apply channel %s now; the next pass will", connection_id)
        return False
    return True


async def stop_channel_polling() -> None:
    """Shutdown: stop this worker's adapters and drop them from the shared
    running set. Never raises, so the rest of the shutdown runs."""
    manager = get_channel_manager()
    try:
        async with _PASS_LOCK:
            manager.release_polling_ownership()
            stopped = await manager.stop_all()
            if stopped:
                await asyncio.to_thread(polling_status.publish, (), stopped)
    except Exception:
        logger.exception("[ChannelPolling] stopping the polling adapters failed")
        return
    logger.info("[ChannelPolling] stopped %d polling adapter(s)", len(stopped))
