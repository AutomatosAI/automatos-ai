"""A per-process sliding-window counter, for a limit that must hold when the shared store is down.

The shared limiter (``api.widgets.rate_limit.RateLimitStore``) counts in Redis across every worker
and lets a request through when Redis cannot be reached. A route that nobody has authenticated and
that writes to the database (PRD-277's sign-in start and callback) must stay bounded all the same,
so it hands this counter to the shared one as ``when_unavailable``: with Redis down, each worker
counts on its own, with the same limit and window. That is looser than the shared count by the
number of workers, and never unlimited.

Memory is bounded: at most ``max_keys`` keys, the least recently used dropped first, and each key
keeps at most ``limit`` timestamps. Each key's marks are an immutable tuple, replaced on every
check under one lock.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Tuple

DEFAULT_MAX_KEYS = 10_000

Answer = Tuple[bool, int, int, int]  # (allowed, limit, remaining, reset_seconds), as RateLimitStore.check


class LocalWindow:
    """Sliding-window counts per key, in this process only."""

    def __init__(self, max_keys: int = DEFAULT_MAX_KEYS, clock: Callable[[], float] = time.monotonic) -> None:
        self._max_keys = max_keys
        self._clock = clock
        self._lock = threading.Lock()
        self._marks: "OrderedDict[str, Tuple[float, ...]]" = OrderedDict()

    def check(self, key: str, limit: int, window: int) -> Answer:
        """Count one request for ``key``; refused once ``limit`` fell inside the last ``window`` seconds."""
        with self._lock:
            now = self._clock()
            live = tuple(mark for mark in self._marks.pop(key, ()) if mark > now - window)
            self._marks[key] = (*live, now)[-limit:]
            while len(self._marks) > self._max_keys:
                self._marks.popitem(last=False)
        if len(live) >= limit:
            return False, limit, 0, max(1, int(live[0] + window - now) + 1)
        return True, limit, max(0, limit - len(live) - 1), int(window)


__all__ = ["DEFAULT_MAX_KEYS", "LocalWindow"]
