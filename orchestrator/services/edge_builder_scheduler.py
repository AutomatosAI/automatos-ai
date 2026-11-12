"""
EdgeBuilderScheduler — Nightly graph edge computation (PRD-139)
===============================================================
Registers a daily cron job on the UnifiedScheduler that runs
``core.services.edge_builder.build_edges()`` to derive:

- tool_routing_edges (action→action co-occurrence)
- tool_routing_affinities (agent/intent preference signals)
- tool_routing_intent_clusters (query embedding clusters)

Runs at 03:00 UTC by default (configurable via EDGE_BUILDER_HOUR_UTC), as a ``graph.edge_builder``
job queued by a platform schedule (PRD-276 Wave 2c; catch-up ``latest``: a night missed while no
scheduler ran is built when one starts, once). Any worker runs it, so its window comes from the
settings when it runs, not from the scheduler that declared it.
The TOOL_ROUTING_GRAPH flag is NOT checked here — edges accumulate
regardless of whether the GraphRouter is enabled at query time.
"""

import logging
from datetime import timedelta, timezone
from typing import Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.jobs.registry import job_handler

logger = logging.getLogger(__name__)

EDGE_BUILDER_KIND = "graph.edge_builder"


def _window() -> timedelta:
    from config import config as app_config

    return timedelta(days=int(getattr(app_config, "EDGE_BUILDER_WINDOW_DAYS", 30)))


class EdgeBuilderScheduler:
    """Declares the nightly edge builder's schedule on the shared APScheduler, and runs it."""

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None

    async def start(self, scheduler: AsyncIOScheduler):
        """Declare the edge builder's nightly platform schedule."""
        from apscheduler.triggers.cron import CronTrigger

        from config import config as app_config
        from core.jobs.schedules import PLATFORM, schedule

        self._scheduler = scheduler
        hour = int(getattr(app_config, "EDGE_BUILDER_HOUR_UTC", 3))
        schedule(scheduler, EDGE_BUILDER_KIND, PLATFORM, CronTrigger(hour=hour, minute=0, timezone=timezone.utc))
        logger.info("[EdgeBuilder] Scheduled nightly at %02d:00 UTC (window=%dd)", hour, _window().days)

    async def _run_edge_builder(self):
        """Execute the edge builder pipeline in its memory-capped child process
        (F416): running out of memory ends the run, never the backend. A run
        that failed in the child has already logged its one error and isn't
        retried (it would most likely fail the same way): the next night runs as
        usual. An error here, outside the child, is logged and raised, so the
        queue retries the run (PRD-276)."""
        try:
            from core.services.edge_builder_worker import build_edges_in_worker

            summary = await build_edges_in_worker(window=_window())
            if summary.failed:
                return
            logger.info(
                "[EdgeBuilder] Nightly run complete — %d edges, %d affinities, %d clusters "
                "(clustered %d of %d queries%s)",
                summary.edges_built,
                summary.affinities_built,
                summary.intent_clusters,
                summary.cluster_queries,
                summary.cluster_queries_total,
                ", a recent sample" if summary.cluster_sampled else "",
            )
        except Exception:
            logger.exception("[EdgeBuilder] Nightly run failed")
            raise


_instance: Optional[EdgeBuilderScheduler] = None


def get_edge_builder_scheduler() -> EdgeBuilderScheduler:
    global _instance
    if _instance is None:
        _instance = EdgeBuilderScheduler()
    return _instance


@job_handler(EDGE_BUILDER_KIND, concurrency=1, workspace_concurrency=1)
async def run_edge_builder(job: Any) -> None:
    """A nightly slot."""
    await get_edge_builder_scheduler()._run_edge_builder()
