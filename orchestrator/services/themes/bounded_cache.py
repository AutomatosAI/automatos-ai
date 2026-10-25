"""A small bounded map that drops its least recently used entry, safe across threadpool workers.

The theme services keep checked, content-addressed data in process memory: ``looks.py`` the
checked ``looks/schema.json`` documents by the file's sha256, ``serve.py`` a package's checked
boot files by its digest. What they keep never changes for its key, so an entry has no expiry,
only a bound. The routes that read them are plain ``def`` (FastAPI's threadpool), so every read
and write holds the lock. Every caller gets the same object back, so callers only read it.
"""

from __future__ import annotations

import threading
from typing import Generic, Hashable, Optional, TypeVar

from cachetools import LRUCache

V = TypeVar("V")


class BoundedCache(Generic[V]):
    """At most ``size`` entries; adding one to a full cache drops the one least recently read or added."""

    def __init__(self, size: int) -> None:
        self._entries: LRUCache = LRUCache(maxsize=size)
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Optional[V]:
        """The entry under ``key`` (now the most recently used), or None."""
        with self._lock:
            try:
                return self._entries[key]
            except KeyError:
                return None

    def put(self, key: Hashable, value: V) -> None:
        """Keep ``value`` under ``key``, dropping the least recently used entry when full."""
        with self._lock:
            self._entries[key] = value

    def clear(self) -> None:
        """Drop every entry (tests; a process restart does the same)."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


__all__ = ["BoundedCache"]
