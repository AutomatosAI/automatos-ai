"""One SDK client per key, kept for a few minutes (voice latency D-4).

A provider object is built per ``LLMManager``, and a spoken turn's manager is new every
turn, so every turn built a new SDK client and HTTP pool before its first model call.
With ``LLM_CLIENT_REUSE`` on, the Anthropic provider asks this store instead: the same
key (SDK class, API key, base URL, keep-alive) gets the client built for it the first
time, until its time is up.

Rules the callers rely on:

* **The store never closes a client.** An entry that expires, is pushed out by the size
  cap or is dropped by ``forget_clients()`` only loses the store's reference. A call in
  flight, or a long-lived stream, still holds the client through its provider, and the
  SDK closes the HTTP client when the last reference is collected.
* ``build`` runs OUTSIDE the lock. It may wait on the database pool and run a PBKDF2
  (``anthropic_workspace.workspace_for_key``), and one slow build must not hold up a
  thread asking for another key. The price is a rare double build for one key; one
  client is kept and every caller gets that one.
* Every call drops every entry past its time, whichever key was asked for.
* The key holds the API key string. The client stored beside it holds the same key in
  plain, and the store is never logged or printed. A fingerprint is not used: the
  repo's fingerprint is the slow PBKDF2 this store takes off the per-turn path.

Sync clients only. An async client is bound to its event loop and would need the loop
in its key.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Tuple, TypeVar

T = TypeVar("T")

MAX_CLIENTS = 64


@dataclass(frozen=True)
class _Kept:
    client: Any
    until: float


_lock = threading.Lock()
_clients: Dict[Tuple[Hashable, ...], _Kept] = {}


def _purge(now: float) -> None:
    """Drop every entry past its time. The caller holds the lock. Nothing is closed."""
    for key in [key for key, kept in _clients.items() if kept.until <= now]:
        del _clients[key]


def shared_client(build: Callable[[], T], *key: Hashable, keep_seconds: float) -> T:
    """The client kept for ``key``, or the one ``build`` makes now, kept for ``keep_seconds``."""
    now = time.monotonic()
    with _lock:
        _purge(now)
        kept = _clients.get(key)
        if kept is not None:
            return kept.client
    client = build()
    with _lock:
        kept = _clients.get(key)
        if kept is not None:
            return kept.client  # another thread stored first; ours was never used
        if len(_clients) >= MAX_CLIENTS:
            del _clients[min(_clients, key=lambda k: _clients[k].until)]
        _clients[key] = _Kept(client, now + keep_seconds)
        return client


def forget_clients() -> None:
    """Drop every kept client (a key was saved or deleted). Nothing is closed."""
    with _lock:
        _clients.clear()


__all__ = ["MAX_CLIENTS", "forget_clients", "shared_client"]
