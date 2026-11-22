"""Health checks that run in the background, so a health route only reads (PRD-282).

A ``Watch`` is one check with its own timer and its own time limit. ``run`` checks at
once, then every ``every_s`` seconds, until it is cancelled. ``last`` is the latest
``Reading``, or None before the first check has finished: one attribute read, which
never waits.

The checks are synchronous clients (a SELECT 1, a ping), so each runs on a daemon
thread of its own, never on the event loop, and the wait for it ends at ``limit_s``:
a check that has not answered by then reads as ``failed_state``. A check still running
when the next one is due is not started again, so a dependency that hangs holds one
thread, never a growing number; and a daemon thread never holds up the process's exit.
A check that raises is logged and reads as ``failed_state`` too. ``on_reading``, when
given, is told each reading as it is made (the dependency gauges in /metrics, US-203).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# What first_reading waits beyond one check's time limit: the thread's start and the
# loop's turn, never a second check.
_FIRST_READING_SLACK_S = 1.0


@dataclass(frozen=True)
class Reading:
    """What one check found, and when it finished (UTC)."""

    state: str
    checked_at: datetime


def _on_a_thread(name: str, check: Callable[[], str]) -> concurrent.futures.Future:
    """Runs ``check`` on a daemon thread of its own; its answer arrives in the future."""
    future: concurrent.futures.Future = concurrent.futures.Future()

    def run() -> None:
        if future.set_running_or_notify_cancel():
            future.set_result(check())

    threading.Thread(target=run, name=f"health-check-{name}", daemon=True).start()
    return future


class Watch:
    """One background check: its timer, its time limit and its last reading."""

    def __init__(self, name: str, check: Callable[[], str], *, every_s: float, limit_s: float,
                 failed_state: str, on_reading: Optional[Callable[[str, Reading], None]] = None) -> None:
        self.name = name
        self.last: Optional[Reading] = None
        self._check = check
        self._every_s = every_s
        self._limit_s = limit_s
        self._failed_state = failed_state
        self._on_reading = on_reading
        self._running: Optional[concurrent.futures.Future] = None
        self._first = asyncio.Event()

    def _guarded_check(self) -> str:
        try:
            return self._check()
        except Exception:
            logger.exception("Health: the %s check failed", self.name)
            return self._failed_state

    async def _answer(self) -> str:
        if self._running is not None and not self._running.done():
            return self._failed_state  # the last check has not answered yet: no second thread
        self._running = _on_a_thread(self.name, self._guarded_check)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(self._running), self._limit_s)
        except asyncio.TimeoutError:
            return self._failed_state

    async def check_now(self) -> Reading:
        """One check, its wait ended by the time limit; the reading becomes ``last``."""
        reading = Reading(await self._answer(), datetime.now(timezone.utc))
        if self.last is None or self.last.state != reading.state:
            logger.info("Health: %s is %s", self.name, reading.state)
        self.last = reading
        self._first.set()
        if self._on_reading is not None:
            self._on_reading(self.name, reading)
        return reading

    async def first_reading(self) -> Optional[Reading]:
        """Waits for ``run``'s first check to finish, at most its time limit (and a
        little), and returns ``last``: None only if no check has finished by then."""
        try:
            await asyncio.wait_for(self._first.wait(), self._limit_s + _FIRST_READING_SLACK_S)
        except asyncio.TimeoutError:
            logger.warning("Health: no %s check had finished within %ss", self.name, self._limit_s)
        return self.last

    async def run(self) -> None:
        """Checks at once, then every ``every_s`` seconds, until cancelled."""
        while True:
            await self.check_now()
            await asyncio.sleep(self._every_s)
