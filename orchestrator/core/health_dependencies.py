"""What each dependency is doing: GET /health's entries and /metrics' gauges (PRD-282 US-202, US-203).

One background ``Watch`` per dependency (Redis, object storage, the search index, the
memory store, the model provider), each on its own timer, HEALTH_DEPENDENCY_CHECK_SECONDS,
with its own time limit, HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS. The checks are in
core/health_dependency_checks.py. GET /health only reads their current readings, so it
stays as fast as it was whatever a dependency does, and a dependency never changes its
status code: that is the database's alone, as before (core/health_probe.http_status).

Each entry is ``{"state": "up" | "down" | "not configured", "checked_at": <UTC>}``, or
``{"state": "unchecked", "checked_at": null}``: no check has finished yet, the last one is
stale (core/health_watch.py: 3 intervals plus one time limit, 93 s by default), the role
may not make the check, or the model provider has had no call through the manager in
MODEL_PROVIDER_STALE_AFTER_S. No other state reaches the body.

/metrics: ``automatos_dependency_up{dependency=...}``, 1 up and 0 down, set at every
tick, and no series while a dependency is not configured, unchecked or stale. The
database's series comes from readiness's database watch (core/health_readiness.py).
Each server process keeps its own series (no multiprocess collector).

``start`` begins every background health check, readiness's included, as the server
starts listening; ``stop`` ends them at shutdown.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from config import config
from core import health_dependency_checks as checks
from core import health_readiness
from core.health_watch import Reading, Watch
from core.monitoring.automatos_metrics import set_dependency_up

logger = logging.getLogger(__name__)

# Each entry's name in /health and /metrics, and its check in health_dependency_checks.
DEPENDENCIES: Tuple[Tuple[str, str], ...] = (
    ("redis", "redis_state"),
    ("object_storage", "object_storage_state"),
    ("search_index", "search_index_state"),
    ("memory_store", "memory_store_state"),
    ("model_provider", "model_provider_state"),
)
# The model provider is never called, so its reading is as old as the last call. An
# install in use calls a model every few minutes (chat turns, heartbeats, scheduled
# agents); past a quarter of an hour without one, "up" would be old news, so it reads
# unchecked until the next call.
MODEL_PROVIDER_STALE_AFTER_S = 15 * 60
_ENTRY_STATES = frozenset({checks.UP, checks.DOWN, checks.NOT_CONFIGURED})  # any other: unchecked
_GAUGE_VALUES = {checks.UP: True, checks.DOWN: False}  # any other state: no series
_WATCHES, _TASKS = "dependency_watches", "dependency_watch_tasks"  # where each lives on app.state


def record_gauge(name: str, reading: Optional[Reading]) -> None:
    """The dependency's /metrics series, from its current reading (None: none, or stale)."""
    set_dependency_up(name, None if reading is None else _GAUGE_VALUES.get(reading.state))


def _watch(name: str, attr: str) -> Watch:
    """One dependency's watch; it looks its check up when it runs (so a test can swap it)."""
    return Watch(name, lambda: getattr(checks, attr)(), every_s=config.HEALTH_DEPENDENCY_CHECK_SECONDS,
                 limit_s=config.HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS, failed_state=checks.DOWN,
                 on_reading=record_gauge,
                 stale_after_s=MODEL_PROVIDER_STALE_AFTER_S if name == "model_provider" else None,
                 notes={checks.NOT_PERMITTED: checks.PERMISSIONS[name]} if name in checks.PERMISSIONS else {})


def dependency_watches() -> Tuple[Watch, ...]:
    """One watch per dependency. One that cannot be built is logged and left out: its
    entry reads unchecked, and every other watch still runs."""
    watches = []
    for name, attr in DEPENDENCIES:
        try:
            watches.append(_watch(name, attr))
        except Exception:
            logger.exception("Health: the %s check could not be set up; its entry reads unchecked", name)
    return tuple(watches)


def start(state: Any) -> None:
    """Starts every background health check for this app, each checking at once. The
    readiness watch starts first and as strictly as before; a failure in the dependency
    checks is logged and leaves their entries unchecked, never the server down."""
    health_readiness.start(state)
    try:
        watches = dependency_watches()
        tasks = tuple(asyncio.create_task(watch.run(), name=f"health-{watch.name}") for watch in watches)
    except Exception:
        logger.exception("Health: the dependency checks could not start; their entries read unchecked")
        watches, tasks = (), ()
    setattr(state, _WATCHES, watches)
    setattr(state, _TASKS, tasks)


async def stop(state: Any) -> None:
    """Ends them all at shutdown. A check still on its thread is left to finish."""
    running = [task for task in getattr(state, _TASKS, ()) if not task.done()]
    for task in running:
        task.cancel()
    if running:
        await asyncio.wait(running)
    await health_readiness.stop(state)


def _entry(reading: Optional[Reading]) -> Dict[str, Optional[str]]:
    if reading is None or reading.state not in _ENTRY_STATES:
        return {"state": health_readiness.UNCHECKED, "checked_at": None}
    return {"state": reading.state, "checked_at": reading.checked_at.isoformat()}


def components(state: Any) -> Dict[str, Dict[str, Optional[str]]]:
    """Every dependency's entry for /health, read from the current readings alone."""
    watches = {watch.name: watch for watch in getattr(state, _WATCHES, ())}
    return {name: _entry(watches[name].current() if name in watches else None) for name, _ in DEPENDENCIES}
