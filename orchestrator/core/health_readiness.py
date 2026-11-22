"""GET /health/ready: ready means start-up has finished and the database answers (PRD-282 US-201).

The database check is the one GET /health uses, ``health_probe.database_status``
(F458): it tells a busy pool from a broken one and never waits on a full pool. A
``Watch`` runs it in the background every HEALTH_READY_CHECK_SECONDS, giving it
HEALTH_READY_CHECK_TIMEOUT_SECONDS to answer, so the route reads one attribute and
does no database work of its own.

* ``healthy`` and ``busy`` (the pool is full, the database answers) are ready.
* ``unhealthy`` (no answer, or none within the time limit) and ``stuck`` are not.
* A reading older than the watch's stale bound (core/health_watch.py: three check
  intervals plus one time limit, 18 s by default) is no passed check: 503, with the
  database ``unchecked``. So a check that hangs, or a watch that stops, never leaves
  the last answer up.
* Redis is not part of it (PRD-282 D-5): core runs without it.
* The watch logs the database's state when it changes, with a failure's traceback
  once, never every tick; a direct GET /health logs as it always did.

Start-up order. The watch starts as the server starts listening (main._start_serving),
so its first check runs beside the background boot; the READY stage waits for that
first result before it sets app.state.ready (main._mark_ready). So the route never
answers ready before a check has been made, and a healthy install waits at most one
check longer than it did, a SELECT 1 that has normally finished long before.
"""
from __future__ import annotations

import asyncio
from typing import Any, Tuple

from config import config
from core import health_probe
from core.health_watch import Watch

READY_STATES = frozenset({health_probe.HEALTHY, health_probe.DATABASE_BUSY})
UNCHECKED = "unchecked"  # no database check has finished, or the last one is stale
_WATCH, _TASK = "database_watch", "database_watch_task"  # where each lives on app.state


def _database_status() -> str:
    # Looked up per call, so a test can replace it. A failure raises, so the watch can log
    # it once, at the state change, rather than every tick.
    return health_probe.database_status(raise_errors=True)


def database_watch() -> Watch:
    """The database check readiness reads, on its own timer and time limit."""
    return Watch("database", _database_status, every_s=config.HEALTH_READY_CHECK_SECONDS,
                 limit_s=config.HEALTH_READY_CHECK_TIMEOUT_SECONDS, failed_state=health_probe.UNHEALTHY)


def start(state: Any) -> None:
    """Starts this app's database watch, its first check at once. Needs a running loop."""
    watch = database_watch()
    setattr(state, _WATCH, watch)
    setattr(state, _TASK, asyncio.create_task(watch.run(), name="health-ready-database"))


async def first_check(state: Any) -> None:
    """Waits for the watch's first check (bounded by its time limit), if one runs."""
    watch = getattr(state, _WATCH, None)
    if watch is not None:
        await watch.first_reading()


async def stop(state: Any) -> None:
    """Ends the watch at shutdown. A check still on its thread is left to finish."""
    task = getattr(state, _TASK, None)
    if task is None or task.done():
        return
    task.cancel()
    await asyncio.wait({task})


def answer(state: Any) -> Tuple[int, dict]:
    """The readiness status code and body, read from app.state alone."""
    if not getattr(state, "ready", False):
        return health_probe.HTTP_UNAVAILABLE, {"status": "starting", "message": "Boot in progress"}
    watch = getattr(state, _WATCH, None)
    reading = watch.current() if watch is not None else None
    if reading is None:  # no check has finished, or the last one is stale
        return health_probe.HTTP_UNAVAILABLE, {"status": "not_ready", "database": UNCHECKED}
    if reading.state in READY_STATES:
        return health_probe.HTTP_OK, {"status": "ready"}
    return health_probe.HTTP_UNAVAILABLE, {
        "status": "not_ready",
        "database": reading.state,
        "checked_at": reading.checked_at.isoformat(),
    }
