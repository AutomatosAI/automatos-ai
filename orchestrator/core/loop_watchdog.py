"""F105 (26 Sep): a frozen event loop names what froze it.

Each F105 freezer so far was found by reading gaps in the backend log: an
embedding line, then nothing, then "Candidate sources" (the document search);
a Composio hint line, then 1.4 to 5.6 s of nothing (the Composio lookups). One
freeze, 4.6 s in the first memory turn after the 26 Sep 01:41Z restart, left no
gap shape to read and was never named.

A daemon thread pings the loop every PING_INTERVAL_S with call_soon_threadsafe.
When a ping goes unanswered for LOOP_STALL_LOG_SECONDS, it takes the loop
thread's stack at that moment, while the stall is still running, as code lines
only (traceback.format_stack: file, line, function and source line; never locals
or argument values). It waits for the loop to answer, then logs one WARNING with
the stall's length and that stack; a stall still running after STILL_STALLED_S
is logged then, as ongoing, so a hang is named too. At most DUMPS_PER_WINDOW such
WARNINGs are logged a minute; the next one counts the stalls left out.
LOOP_STALL_LOG_SECONDS=0 starts nothing. The watchdog never raises into the app,
and its thread is a daemon, so it ends with the process; stop() ends it at once.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import traceback
from collections import deque
from typing import Deque, List, Optional

logger = logging.getLogger(__name__)

PING_INTERVAL_S = 0.5
STILL_STALLED_S = 30.0
DUMPS_PER_WINDOW = 10
DUMP_WINDOW_S = 60.0
STACK_FRAMES = 40


class LoopWatchdog:
    """Watches one event loop from a thread of its own."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        loop_thread_id: int,
        stall_s: float,
        *,
        ping_s: float = PING_INTERVAL_S,
        dumps_per_window: int = DUMPS_PER_WINDOW,
        window_s: float = DUMP_WINDOW_S,
    ) -> None:
        self._loop = loop
        self._loop_thread_id = loop_thread_id
        self._stall_s = stall_s
        self._ping_s = ping_s
        self._dumps_per_window = dumps_per_window
        self._window_s = window_s
        self._dumped: Deque[float] = deque()
        self._left_out = 0
        self._stop = threading.Event()
        self._pending: Optional[threading.Event] = None
        self._thread = threading.Thread(target=self._watch, name="loop-watchdog", daemon=True)

    def start(self) -> "LoopWatchdog":
        self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> None:
        """Stop watching. Safe on the loop's own thread: the pending ping is
        answered here, so the watchdog never waits on a loop that is waiting on it."""
        self._stop.set()
        pending = self._pending
        if pending is not None:
            pending.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _watch(self) -> None:
        try:
            while not self._done():
                answered = self._pending = threading.Event()
                sent = time.monotonic()
                self._loop.call_soon_threadsafe(answered.set)
                if not self._answered_within(answered, self._stall_s):
                    if self._done():
                        return
                    stack = self._loop_stack()
                    self._answered_within(answered, max(0.0, STILL_STALLED_S - self._stall_s))
                    if self._done():
                        return
                    self._report(time.monotonic() - sent, stack, ongoing=not answered.is_set())
                    while not self._answered_within(answered, STILL_STALLED_S):  # a hang, logged
                        if self._done():
                            return
                self._stop.wait(self._ping_s)
        except RuntimeError:
            return  # the loop closed between the check and the ping
        except Exception:  # noqa: BLE001 — a watchdog never takes the app down
            logger.debug("[loop-watchdog] stopped watching", exc_info=True)

    def _done(self) -> bool:
        return self._stop.is_set() or self._loop.is_closed()

    def _answered_within(self, answered: threading.Event, seconds: float) -> bool:
        """The loop answered within ``seconds``; False sooner if we stop or the
        loop closes (a closed loop never answers, and is no stall)."""
        deadline = time.monotonic() + seconds
        while True:
            left = deadline - time.monotonic()
            if answered.wait(min(max(left, 0.0), self._ping_s)):
                return True
            if left <= 0 or self._done():
                return False

    def _loop_stack(self) -> List[str]:
        frame = sys._current_frames().get(self._loop_thread_id)
        return traceback.format_stack(frame, limit=STACK_FRAMES) if frame is not None else []

    def _report(self, stalled_s: float, stack: List[str], *, ongoing: bool = False) -> None:
        now = time.monotonic()
        while self._dumped and now - self._dumped[0] > self._window_s:
            self._dumped.popleft()
        if len(self._dumped) >= self._dumps_per_window:
            self._left_out += 1
            return
        self._dumped.append(now)
        left_out, self._left_out = self._left_out, 0
        logger.warning(
            "[loop-watchdog] the event loop %s for %.1fs%s; its thread was at:\n%s",
            "has stood still, and still does," if ongoing else "stood still",
            stalled_s,
            f" ({left_out} more stalls since the last one were not dumped: over "
            f"{self._dumps_per_window} a minute)" if left_out else "",
            "".join(stack) or "  (the loop thread was not found)\n",
        )


def start_loop_watchdog(stall_s: Optional[float] = None, **options: float) -> Optional[LoopWatchdog]:
    """Watch the running loop (call this on the loop's thread). Returns None,
    having started nothing, when LOOP_STALL_LOG_SECONDS (or ``stall_s``) is 0,
    or if the watchdog cannot start."""
    try:
        if stall_s is None:
            from config import config

            stall_s = float(config.LOOP_STALL_LOG_SECONDS)
        if stall_s <= 0:
            return None
        loop = asyncio.get_running_loop()
        return LoopWatchdog(loop, threading.get_ident(), stall_s, **options).start()
    except Exception:  # noqa: BLE001 — a watchdog never takes the app down
        logger.warning("[loop-watchdog] not started", exc_info=True)
        return None
