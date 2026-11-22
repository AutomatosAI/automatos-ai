"""Health checks that run in the background, so a health route only reads (PRD-282).

A ``Watch`` is one check with its own timer and its own time limit. ``run`` checks at
once, then every ``every_s`` seconds, until it is cancelled; an error in one tick is
logged and the next tick comes anyway, so the watch never stops on its own.

* **Its answer has an age.** ``last`` is the latest ``Reading``; ``current()`` is that
  reading while it is younger than ``stale_after_s`` (by default STALE_AFTER_CHECKS
  intervals plus one time limit), else None. A reader treats None as "no check
  passed", so a watch that has stopped answering can never leave a stale answer up.
* **Off the loop, one thread at most.** The checks are synchronous clients (a SELECT 1,
  a ping), so each runs on a daemon thread of its own. The wait for it ends at
  ``limit_s``: a check that has not answered by then is recorded as ``failed_state``,
  stamped when its time limit ran out. While that thread still runs no second check is
  started, and no new reading is made, so the reading ages; when the thread does
  finish, its answer is recorded then. A daemon thread never holds up the process's exit.
* **Quiet logs.** A check that raises reads as ``failed_state``. The watch logs a state
  only when it changes (a failure with its traceback, once) and when it recovers.
* ``on_reading``, when given, is told the current reading after every tick (None when
  there is none or it is stale); a callback that fails is logged and the reading stands.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# A reading is stale after this many check intervals plus one time limit: a watch that
# has missed that many ticks in a row is not answering, whatever its last reading said.
STALE_AFTER_CHECKS = 3
# What first_reading waits beyond one check's time limit: the thread's start and the
# loop's turn, never a second check.
_FIRST_READING_SLACK_S = 1.0

Outcome = Tuple[str, Optional[BaseException]]


@dataclass(frozen=True)
class Reading:
    """What one check found, and when it finished (UTC)."""

    state: str
    checked_at: datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _on_a_thread(name: str, check: Callable[[], Outcome]) -> concurrent.futures.Future:
    """Runs ``check`` on a daemon thread of its own; its outcome arrives in the future."""
    future: concurrent.futures.Future = concurrent.futures.Future()

    def run() -> None:
        if future.set_running_or_notify_cancel():
            future.set_result(check())

    threading.Thread(target=run, name=f"health-check-{name}", daemon=True).start()
    return future


class Watch:
    """One background check: its timer, its time limit and its last reading."""

    def __init__(self, name: str, check: Callable[[], str], *, every_s: float, limit_s: float,
                 failed_state: str, on_reading: Optional[Callable[[str, Optional[Reading]], None]] = None,
                 stale_after_s: Optional[float] = None) -> None:
        self.name = name
        self.last: Optional[Reading] = None
        self.stale_after_s = stale_after_s if stale_after_s is not None else STALE_AFTER_CHECKS * every_s + limit_s
        self._check = check
        self._every_s = every_s
        self._limit_s = limit_s
        self._failed_state = failed_state
        self._on_reading = on_reading
        self._running: Optional[asyncio.Future] = None
        self._first = asyncio.Event()

    def current(self, now: Optional[datetime] = None) -> Optional[Reading]:
        """The last reading while it is fresh; None before the first, or once it is stale."""
        reading = self.last
        if reading is None:
            return None
        age_s = ((now or _now()) - reading.checked_at).total_seconds()
        return reading if age_s <= self.stale_after_s else None

    def _guarded_check(self) -> Outcome:
        try:
            return self._check(), None
        except Exception as exc:  # noqa: BLE001 — handed to the watch, which logs it at the state change
            return self._failed_state, exc

    async def check_now(self) -> Optional[Reading]:
        """One tick: a check, its wait ended by the time limit. Returns ``last``."""
        if self._running is not None and not self._running.done():
            self._publish()  # the last check's thread still runs: no second one, and the reading ages
            return self.last
        running = asyncio.wrap_future(_on_a_thread(self.name, self._guarded_check))
        self._running = running
        done, _ = await asyncio.wait({running}, timeout=self._limit_s)
        if running in done:
            self._record(*running.result())
        else:
            self._record(self._failed_state, None, timed_out=True)
            running.add_done_callback(self._late_answer)
        return self.last

    def _late_answer(self, running: asyncio.Future) -> None:
        """The answer of a check that outlived its time limit, recorded when it came."""
        if not running.cancelled():
            self._record(*running.result())

    def _record(self, state: str, error: Optional[BaseException], *, timed_out: bool = False) -> None:
        previous, reading = self.last, Reading(state, _now())
        self.last = reading
        self._first.set()
        if previous is None or previous.state != state:
            self._log_change(state, error, timed_out)
        self._publish()

    def _log_change(self, state: str, error: Optional[BaseException], timed_out: bool) -> None:
        if error is not None:
            logger.warning("Health: %s is %s", self.name, state, exc_info=error)
        elif timed_out:
            logger.warning("Health: %s is %s: no answer within %gs", self.name, state, self._limit_s)
        else:
            logger.info("Health: %s is %s", self.name, state)

    def _publish(self) -> None:
        if self._on_reading is None:
            return
        try:
            self._on_reading(self.name, self.current())
        except Exception:
            logger.exception("Health: passing on the %s reading failed; the reading stands", self.name)

    async def first_reading(self) -> Optional[Reading]:
        """Waits for ``run``'s first check to finish, at most its time limit (and a
        little), and returns ``last``: None only if no check has finished by then."""
        try:
            await asyncio.wait_for(self._first.wait(), self._limit_s + _FIRST_READING_SLACK_S)
        except asyncio.TimeoutError:
            logger.warning("Health: no %s check had finished within %ss", self.name, self._limit_s)
        return self.last

    async def run(self) -> None:
        """Checks at once, then every ``every_s`` seconds, until cancelled. An error in a
        tick is logged and the watch goes on: it never stops by itself."""
        while True:
            try:
                await self.check_now()
            except Exception:
                logger.exception("Health: a %s check failed outside its thread; the watch goes on", self.name)
            await asyncio.sleep(self._every_s)
