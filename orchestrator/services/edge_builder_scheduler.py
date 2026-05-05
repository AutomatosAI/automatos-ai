"""
EdgeBuilderScheduler — Nightly graph edge computation (PRD-139)
===============================================================
Registers a daily cron job on the UnifiedScheduler that runs
``core.services.edge_builder.build_edges()`` to derive:

- tool_routing_edges (action→action co-occurrence)
- tool_routing_affinities (agent/intent preference signals)
- tool_routing_intent_clusters (query embedding clusters)

Runs at 03:00 UTC by default (configurable via EDGE_BUILDER_HOUR_UTC).
The TOOL_ROUTING_GRAPH flag is NOT checked here — edges accumulate
regardless of whether the GraphRouter is enabled at query time.
"""

import logging
from datetime import timedelta
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)


class EdgeBuilderScheduler:
    """Registers the nightly edge builder job on the shared APScheduler."""

    JOB_ID = "edge_builder_nightly"

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None

    async def start(self, scheduler: AsyncIOScheduler):
        """Register the edge builder cron job."""
        from config import config as app_config

        self._scheduler = scheduler

        hour = int(getattr(app_config, "EDGE_BUILDER_HOUR_UTC", 3))
        window_days = int(getattr(app_config, "EDGE_BUILDER_WINDOW_DAYS", 30))

        self._window = timedelta(days=window_days)

        self._scheduler.add_job(
            self._run_edge_builder,
            "cron",
            hour=hour,
            minute=0,
            id=self.JOB_ID,
            replace_existing=True,
            max_instances=1,
        )

        logger.info(
            "[EdgeBuilder] Scheduled nightly at %02d:00 UTC (window=%dd)",
            hour,
            window_days,
        )

    async def _run_edge_builder(self):
        """Execute the edge builder pipeline."""
        try:
            from core.services.edge_builder import build_edges

            summary = await build_edges(window=self._window)
            logger.info(
                "[EdgeBuilder] Nightly run complete — %d edges, %d affinities, %d clusters",
                summary.edges_built,
                summary.affinities_built,
                summary.intent_clusters,
            )
        except Exception:
            logger.exception("[EdgeBuilder] Nightly run failed")


_instance: Optional[EdgeBuilderScheduler] = None


def get_edge_builder_scheduler() -> EdgeBuilderScheduler:
    global _instance
    if _instance is None:
        _instance = EdgeBuilderScheduler()
    return _instance
