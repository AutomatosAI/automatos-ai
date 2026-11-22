"""What each dependency is doing: GET /health's entries and /metrics' gauges (PRD-282 US-202, US-203).

One background ``Watch`` per dependency (Redis, object storage, the search index, the
memory store, the model provider), each on its own timer, HEALTH_DEPENDENCY_CHECK_SECONDS,
with its own time limit, HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS. The checks are in
core/health_dependency_checks.py. GET /health only reads their last results, so it stays
as fast as it was whatever a dependency does, and a dependency never changes its status
code: that is the database's alone, as before (core/health_probe.http_status).

Each entry is ``{"state": "up" | "down" | "not configured", "checked_at": <UTC>}``, or
``{"state": "unchecked", "checked_at": null}`` before its first check finished (and, for
the model provider, while the install names one but no real call has ended yet).

/metrics: ``automatos_dependency_up{dependency=...}``, 1 up and 0 down, set as each
reading is made, and no series while a dependency is not configured or not yet known.
The database's series comes from readiness's database watch (core/health_readiness.py).

``start`` begins every background health check, readiness's included, as the server
starts listening; ``stop`` ends them at shutdown.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from config import config
from core import health_dependency_checks as checks
from core import health_readiness
from core.health_watch import Reading, Watch
from core.monitoring.automatos_metrics import set_dependency_up

# Each entry's name in /health and /metrics, and its check in health_dependency_checks.
DEPENDENCIES: Tuple[Tuple[str, str], ...] = (
    ("redis", "redis_state"),
    ("object_storage", "object_storage_state"),
    ("search_index", "search_index_state"),
    ("memory_store", "memory_store_state"),
    ("model_provider", "model_provider_state"),
)
_GAUGE_VALUES = {checks.UP: True, checks.DOWN: False}  # any other state: no series
_WATCHES, _TASKS = "dependency_watches", "dependency_watch_tasks"  # where each lives on app.state


def record_gauge(name: str, reading: Optional[Reading]) -> None:
    """The dependency's /metrics series, from its current reading (None: none, or stale)."""
    set_dependency_up(name, None if reading is None else _GAUGE_VALUES.get(reading.state))


def dependency_watches() -> Tuple[Watch, ...]:
    """One watch per dependency; each looks its check up when it runs (so a test can swap it)."""
    return tuple(
        Watch(name, lambda attr=attr: getattr(checks, attr)(), every_s=config.HEALTH_DEPENDENCY_CHECK_SECONDS,
              limit_s=config.HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS, failed_state=checks.DOWN,
              on_reading=record_gauge)
        for name, attr in DEPENDENCIES
    )


def start(state: Any) -> None:
    """Starts every background health check for this app, each checking at once."""
    health_readiness.start(state)
    watches = dependency_watches()
    setattr(state, _WATCHES, watches)
    setattr(state, _TASKS, tuple(asyncio.create_task(watch.run(), name=f"health-{watch.name}")
                                 for watch in watches))


async def stop(state: Any) -> None:
    """Ends them all at shutdown. A check still on its thread is left to finish."""
    running = [task for task in getattr(state, _TASKS, ()) if not task.done()]
    for task in running:
        task.cancel()
    if running:
        await asyncio.wait(running)
    await health_readiness.stop(state)


def _entry(reading: Optional[Reading]) -> Dict[str, Optional[str]]:
    if reading is None:
        return {"state": health_readiness.UNCHECKED, "checked_at": None}
    return {"state": reading.state, "checked_at": reading.checked_at.isoformat()}


def components(state: Any) -> Dict[str, Dict[str, Optional[str]]]:
    """Every dependency's entry for /health, read from the last results alone."""
    watches = {watch.name: watch for watch in getattr(state, _WATCHES, ())}
    return {name: _entry(watches[name].last if name in watches else None) for name, _ in DEPENDENCIES}
