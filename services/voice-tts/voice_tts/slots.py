"""The synthesis slots: how many requests may be speaking at once, counted for as long as they really are.

A request takes a slot when it is admitted, and its sentences are spoken on this pool's
threads, one at a time. A slot is given back only when BOTH the request has ended AND
the synthesis thread it started has finished. A request whose client went away, or whose
handler was cancelled, ends at once, but Kokoro cannot be stopped mid-sentence: its
thread keeps the CPU until the sentence is done, and so it keeps the slot. Connecting
and dropping again and again therefore never runs more syntheses than there are slots.

The pool has as many threads as there are slots and each slot runs one synthesis at a
time, so a synthesis never waits for a thread. Past the last slot, admission refuses at
once (429 ``tts_busy``): nothing queues.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class Slots:
    """``limit`` slots over a pool of ``limit`` synthesis threads. Used from the event loop only."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.held = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=limit, thread_name_prefix="kokoro")

    def try_acquire(self) -> Optional["Slot"]:
        """A slot, or None when every one is held."""
        if self.held >= self.limit:
            return None
        self.held += 1
        return Slot(self, asyncio.get_running_loop())

    def _give_back(self) -> None:
        self.held -= 1

    def shutdown(self) -> None:
        """Stop taking work; a synthesis already running finishes on its own."""
        self.executor.shutdown(wait=False, cancel_futures=True)


class Slot:
    """One request's slot: held until the request is closed and its last synthesis has finished."""

    def __init__(self, slots: Slots, loop: asyncio.AbstractEventLoop) -> None:
        self._slots = slots
        self._loop = loop
        self._running: Optional[concurrent.futures.Future] = None
        self._closed = False
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        """``fn(*args)`` on a synthesis thread. A cancelled wait leaves the thread running, and the slot held."""
        future = self._slots.executor.submit(fn, *args)
        self._running = future
        future.add_done_callback(self._thread_finished)
        return await asyncio.wrap_future(future, loop=self._loop)

    def _thread_finished(self, future: concurrent.futures.Future) -> None:
        # Called on the synthesis thread (or at once, for a future cancelled before it ran).
        try:
            self._loop.call_soon_threadsafe(self._settle, future)
        except RuntimeError:
            return  # the loop has closed: the service is stopping

    def _settle(self, future: concurrent.futures.Future) -> None:
        if self._running is future:
            self._running = None
        self._release_if_done()

    def close(self) -> None:
        """The request has ended. The slot goes back now, or when its synthesis finishes."""
        self._closed = True
        self._release_if_done()

    def _release_if_done(self) -> None:
        if self._closed and self._running is None and not self._released:
            self._released = True
            self._slots._give_back()
