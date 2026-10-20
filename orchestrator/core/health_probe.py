"""What GET /health reads (F229, F458), none of it waiting long.

/health is the probe Docker's healthcheck and the entrypoint's liveness watchdog
both run (docker-healthcheck.sh: curl with a 10 s limit; docker-entrypoint.sh,
F416: three misses in a row stop the backend so its container restarts). So what
it reads answers at once:

* **The database.** Night 18 (F458): the pool (10 + 20) ran dry at 08:01Z. The
  probe's SELECT 1 waited for a connection with the pool's 30 s timeout, missed
  three probes in a row, and the watchdog stopped a backend that was still serving,
  dropping every turn and agent run in flight. Now a full pool is read from the
  pool's counters, with no checkout: the database is "busy" and /health answers
  degraded, at once. A pool seen full at every probe for HEALTH_POOL_STUCK_SECONDS
  is "stuck": its connections are not coming back and a restart is what frees
  them, so /health answers 503 and the watchdog restarts it. The room check and the
  checkout are not atomic: a request that takes the last connection in between
  makes one probe wait, which is one miss, never the three in a row a restart needs.
* **The host's CPU and memory,** read without sleeping (F229).

All plain functions: the route is a plain ``def`` and runs in the threadpool,
never on the event loop (F105, F229).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

from config import config
from core.database import pool_watch

logger = logging.getLogger(__name__)

HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"
DATABASE_BUSY = "busy"    # the pool is full now
DATABASE_STUCK = "stuck"  # ...and has been at every probe for HEALTH_POOL_STUCK_SECONDS
HTTP_OK = 200
HTTP_UNAVAILABLE = 503

# How a component's own state counts toward the overall status.
_COUNTS_AS = {DATABASE_BUSY: DEGRADED, DATABASE_STUCK: UNHEALTHY}


class FullPoolClock:
    """When the pool was first seen full, with no free sighting since."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._full_since: Optional[float] = None

    def saw_room(self) -> None:
        with self._lock:
            self._full_since = None

    def saw_full(self, now: float) -> float:
        """Seconds the pool has been full, counted from its first full sighting."""
        with self._lock:
            if self._full_since is None:
                self._full_since = now
            return now - self._full_since


_full_pool = FullPoolClock()


def database_status(clock: FullPoolClock = _full_pool) -> str:
    """``healthy`` or ``unhealthy`` (a SELECT 1 on a free connection), or, with no
    connection free, ``busy`` / ``stuck`` without asking the pool for one."""
    if not pool_watch.pool_has_room():
        full_for = clock.saw_full(time.monotonic())
        return DATABASE_STUCK if full_for >= config.HEALTH_POOL_STUCK_SECONDS else DATABASE_BUSY
    clock.saw_room()
    return _select_one()


def _select_one() -> str:
    from sqlalchemy import text

    from core.database import database

    try:
        db = database.SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            return HEALTHY
        finally:
            db.close()
    except Exception:
        logger.exception("Health: database check failed")
        return UNHEALTHY


def overall_status(components: Dict[str, str]) -> str:
    """unhealthy if any component is, else degraded if any is, else healthy."""
    levels = {_COUNTS_AS.get(state, state) for state in components.values()}
    if UNHEALTHY in levels:
        return UNHEALTHY
    if DEGRADED in levels:
        return DEGRADED
    return HEALTHY


def http_status(components: Dict[str, str]) -> int:
    """503 only for a stuck pool, the one state a restart cures. A database that is
    down still answers 200 (unhealthy): restarting the backend would not bring it back."""
    return HTTP_UNAVAILABLE if components.get("database") == DATABASE_STUCK else HTTP_OK


def host_metrics() -> dict:
    """The process host's CPU and memory (F229: the CPU reading never sleeps;
    ``interval=None`` is the use since the previous reading). Empty when psutil
    cannot read them."""
    import psutil

    try:
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        return {
            "cpu_percent": round(cpu_pct, 1),
            "memory_used_percent": round(mem.percent, 1),
            "memory_available_mb": round(mem.available / (1024 * 1024), 0),
        }
    except Exception:  # noqa: BLE001 — the health answer goes out without metrics
        return {}
