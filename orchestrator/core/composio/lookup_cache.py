"""F105 (26 Sep): a Composio lookup a turn has just made is answered from memory.

Every chat turn, playbook step and agent run with Composio apps asked the SDK
again for the same answers: the hint enrichment fetched each matched app's whole
action list (tools.get, 1.4-5.6 s a turn), and the tool search asked a question
the SDK had just answered. An app's actions change when Composio publishes a
new toolkit version, not between turns, so both answers are kept for
COMPOSIO_LOOKUP_CACHE_TTL_SECONDS (10 minutes by default):

* APP_ACTIONS, an app's action list, by app. It is fetched as a placeholder
  user, so it is the same for every workspace. Callers only read it.
* STEP_SEARCHES, a step search's results, by the workspace's Composio entity,
  query, toolkits, limit and named actions: the SDK answers it as that entity.
  Callers hand the schemas on to a turn's tools, so they get copies.

A failed lookup is not kept, so the next turn asks again. The lookups run on
several threads (core.composio.off_loop), hence the locks.
"""
from __future__ import annotations

import threading
from typing import Any, Hashable, Optional

from cachetools import TTLCache

APP_ACTIONS_KEPT = 64
STEP_SEARCHES_KEPT = 512


class _Kept:
    """A TTL cache behind a lock, built on first use from the configured TTL."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._entries: Optional[TTLCache] = None

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            return self._cache().get(key)

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._cache()[key] = value

    def clear(self) -> None:
        with self._lock:
            self._entries = None

    def _cache(self) -> TTLCache:
        if self._entries is None:
            from config import config

            self._entries = TTLCache(maxsize=self._maxsize, ttl=config.COMPOSIO_LOOKUP_CACHE_TTL_SECONDS)
        return self._entries


APP_ACTIONS = _Kept(APP_ACTIONS_KEPT)
STEP_SEARCHES = _Kept(STEP_SEARCHES_KEPT)


def forget_all() -> None:
    """Drop every kept answer (tests; a new TTL takes effect from here)."""
    APP_ACTIONS.clear()
    STEP_SEARCHES.clear()
