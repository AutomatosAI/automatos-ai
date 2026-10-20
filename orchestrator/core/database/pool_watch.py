"""F458 (night 18): how full the connection pool is, read without waiting on it.

The app has one pool (core/database/database.py: 10 + 20 connections, a 30 s
wait). Asking it for a connection when none is free blocks for that whole wait,
so two callers read its counters first instead:

* the best-effort writers (core/best_effort.py, F105), which drop a row rather
  than queue behind requests;
* GET /health (core/health_probe.py). Night 18: the pool ran dry at 08:01Z, the
  probe's SELECT 1 waited for a connection past curl's 10 s three times in a row,
  and the entrypoint's liveness watchdog stopped a backend that was still serving.

Nothing here checks a connection out.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _app_pool() -> Any:
    from core.database import database

    return database.engine.pool


def pool_has_room(pool: Optional[Any] = None) -> bool:
    """A connection is idle, or one more may be opened (the app's pool when none
    is given). Unreadable → True: a faked or foreign pool never stops a caller."""
    try:
        pool = _app_pool() if pool is None else pool
        return pool.checkedin() > 0 or pool.overflow() < pool._max_overflow
    except Exception:  # noqa: BLE001 — a faked or foreign pool: just go ahead
        return True


def pool_numbers(pool: Optional[Any] = None) -> Dict[str, int]:
    """Connections checked out, and the most the pool opens. Empty when unreadable."""
    try:
        pool = _app_pool() if pool is None else pool
        return {"checked_out": int(pool.checkedout()), "limit": int(pool.size() + pool._max_overflow)}
    except Exception:  # noqa: BLE001 — a faked or foreign pool has no numbers to show
        return {}
