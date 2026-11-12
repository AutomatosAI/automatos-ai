"""One scheduler across processes and containers (PRD-276 §6.4, Wave 2a).

A Postgres session advisory lock picks the one process that runs the shared scheduler. It replaces
the ``/tmp`` file lock, which held within one container only.

* **The lock is only as good as its connection.** A session lock goes quietly when its connection
  drops (a failover, an idle timeout), and a pooler in transaction mode can't hold one. So it's
  held on a dedicated connection, outside the app's pool, with TCP keepalive (``SchedulerLock``).
* **Checked, and given back.** Every ``SCHEDULER_LEADER_CHECK_SECONDS`` the holder checks its
  connection. When the lock is gone it pauses the scheduler, so it stops ticking until it holds
  the lock again; a waiting process tries the lock as often, and starts the scheduled services
  the first time it gets it (``Leadership``). When the holder dies, another process takes over
  within that interval.
* **The lock isn't the safety.** It keeps the work to one scheduler most of the time. The per-slot
  key and the ``job_schedules`` cursor (Wave 2b) make a second scheduler harmless when it slips.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

SCHEDULER_LOCK_KEY = 2_760_000_001  # one fixed advisory-lock key: PRD-276's scheduler
KEEPALIVE = {"keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 3}
_TRY = text("SELECT pg_try_advisory_lock(:key)")
_ALIVE = text("SELECT 1")
_UNLOCK = text("SELECT pg_advisory_unlock(:key)")


class SchedulerLock:
    """The scheduler's advisory lock, on a connection of its own (never one from the app's pool)."""

    def __init__(self, url: str, key: int = SCHEDULER_LOCK_KEY) -> None:
        self._url, self._key = url, key
        self._engine: Any = None
        self._conn: Any = None
        self._holding = False  # this session took the lock; an open connection alone proves nothing

    def _connect(self) -> Any:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import NullPool

        if self._engine is None:
            self._engine = create_engine(self._url, poolclass=NullPool, connect_args=KEEPALIVE)
        return self._engine.connect().execution_options(isolation_level="AUTOCOMMIT")

    def try_acquire(self) -> bool:
        """Takes the lock if no other session holds it; True when this one now does."""
        try:
            if self._conn is None:
                self._conn = self._connect()
            self._holding = bool(self._conn.execute(_TRY, {"key": self._key}).scalar())
            return self._holding
        except Exception:  # noqa: BLE001 — logged; the next check tries again on a new connection
            logger.warning("[scheduler] could not try the scheduler lock", exc_info=True)
            self._drop()
            return False

    def held(self) -> bool:
        """True while this process holds the lock: it took it, and its session, and so its lock, is
        still alive."""
        if self._conn is None or not self._holding:
            return False
        try:
            self._conn.execute(_ALIVE)
            return True
        except Exception:  # noqa: BLE001 — the connection is gone, and the lock with it
            logger.warning("[scheduler] the scheduler lock's connection was lost", exc_info=True)
            self._drop()
            return False

    def release(self) -> None:
        """Gives the lock back at once (unlocked, then its session closed), so the next process can
        take it without waiting for the server to notice the session is gone."""
        if self._conn is not None and self._holding:
            try:
                self._conn.execute(_UNLOCK, {"key": self._key})
            except Exception:  # noqa: BLE001 — closing the session releases it all the same
                logger.debug("[scheduler] unlocking the scheduler lock failed; closing its session", exc_info=True)
        self._drop()

    def _drop(self) -> None:
        conn, self._conn, self._holding = self._conn, None, False
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 — a dead connection may refuse to close; it is gone either way
                logger.debug("[scheduler] closing the lock's connection failed", exc_info=True)


class Leadership:
    """Keeps this process's scheduler running only while it holds the lock.

    ``start_services`` runs once, the first time the lock is held: it starts the shared scheduler
    and registers the scheduled services on it. ``set_paused`` pauses or resumes that scheduler
    when the lock is lost or held again.
    """

    def __init__(self, lock: SchedulerLock, start_services: Callable[[], Awaitable[None]],
                 set_paused: Callable[[bool], None]) -> None:
        self._lock, self._start_services, self._set_paused = lock, start_services, set_paused
        self.started = False
        self.leading = False

    async def check(self) -> bool:
        """One check: hold or take the lock, then start, pause or resume. True while leading."""
        holds = await asyncio.to_thread(self._lock.held) or await asyncio.to_thread(self._lock.try_acquire)
        if holds and not self.started:
            await self._start_services()
            self.started = True
            logger.info("[scheduler] this process holds the scheduler lock and runs the scheduler")
        elif self.started and holds != self.leading:
            self._set_paused(not holds)
            logger.warning("[scheduler] scheduler lock %s; scheduler %s", "held again" if holds else "lost",
                           "resumed" if holds else "paused until it is held again")
        self.leading = holds
        return holds

    def release(self) -> None:
        """Gives the lock back now (shutdown)."""
        self._lock.release()
        self.leading = False

    async def run(self, stop: asyncio.Event, every_seconds: float) -> None:
        """Checks every ``every_seconds`` until ``stop`` is set, then gives the lock back."""
        try:
            while not stop.is_set():
                try:
                    await self.check()
                except Exception:  # noqa: BLE001 — logged; the next check tries again
                    logger.exception("[scheduler] leadership check failed")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=every_seconds)
                except asyncio.TimeoutError:
                    continue
        finally:
            await asyncio.to_thread(self._lock.release)


def database_url() -> str:
    """The URL the lock's own connection uses: the app's database."""
    from core.database.database import get_database_url

    return get_database_url()


def leadership_for(start_services: Callable[[], Awaitable[None]], set_paused: Callable[[bool], None],
                   lock: Optional[SchedulerLock] = None) -> Leadership:
    """A ``Leadership`` on the app's database (tests pass their own lock)."""
    return Leadership(lock or SchedulerLock(database_url()), start_services, set_paused)
