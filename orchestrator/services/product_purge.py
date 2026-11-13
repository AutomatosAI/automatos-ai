"""The purge of a product's closed workspaces (PRD-274 Wave 5c, decision 25).

A product closes a workspace with ``PRODUCT_CLOSE_GRACE_DAYS`` (7) to restore it (``services.product_lifecycle``).
Once a day, at ``PURGE_HOUR_UTC``:``PURGE_MINUTE_UTC``, a platform schedule queues ``products.purge_closed``
(``schedules_the_closed_purge``, on ``UnifiedScheduler.start``), and its job purges every product workspace
closed longer than that, one by one, through ``purge_workspace_sync``: the same guarded purge an admin's
delete runs. A run missed while no scheduler ran is queued when one starts again (catch-up ``latest``: one run
purges everything due).

A purge that is held (its object storage or memories could not be erased) leaves the workspace closed for the
next run; the others are still purged, and then the job fails with how many were held, so the queue retries
it and the dead-job list shows it if they never go.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from datetime import timezone
from typing import Any, Callable, List
from uuid import UUID

from sqlalchemy import text

from core.jobs.registry import CATCH_UP_LATEST, job_handler

logger = logging.getLogger(__name__)

PURGE_KIND = "products.purge_closed"
PURGE_HOUR_UTC = 2
PURGE_MINUTE_UTC = 50
PURGE_LEASE_SECONDS = 900  # each purge erases object storage and memories; the worker renews the lease

_DUE = text("SELECT id::text FROM workspaces WHERE settings->>'product_id' IS NOT NULL AND deleted_at IS NOT NULL "
            "AND deleted_at < (now() AT TIME ZONE 'utc') - make_interval(days => :days) ORDER BY deleted_at")


def _due(db: Any) -> List[str]:
    """The product workspaces closed longer than the grace, the longest first."""
    from services.product_lifecycle import grace

    return list(db.execute(_DUE, {"days": grace().days}).scalars())


def schedules_the_closed_purge(start: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps ``UnifiedScheduler.start``: once the shared scheduler runs, it purges closed product workspaces daily."""

    @functools.wraps(start)
    def started(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = start(self, *args, **kwargs)
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is not None:
            from apscheduler.triggers.cron import CronTrigger

            from core.jobs.schedules import PLATFORM, schedule

            schedule(scheduler, PURGE_KIND, PLATFORM,
                     CronTrigger(hour=PURGE_HOUR_UTC, minute=PURGE_MINUTE_UTC, timezone=timezone.utc))
        return result

    return started


@job_handler(PURGE_KIND, catch_up=CATCH_UP_LATEST, concurrency=1, workspace_concurrency=1,
             lease_seconds=PURGE_LEASE_SECONDS)
async def purge_closed(job: Any) -> None:
    """A daily slot: purges every product workspace past its grace; raises when any purge was held."""
    from core.jobs.worker import on_own_session
    from services.workspace_purge import purge_workspace_sync

    held = []
    for workspace_id in await asyncio.to_thread(on_own_session, _due):
        result = await asyncio.to_thread(purge_workspace_sync, UUID(workspace_id))
        if result.skipped_reason:
            held.append(f"{workspace_id} ({result.skipped_reason})")
        else:
            logger.warning("[products] purged closed workspace %s, past its grace", workspace_id)
    if held:
        raise RuntimeError(f"{len(held)} closed product workspace purge(s) held: {', '.join(held)}")
