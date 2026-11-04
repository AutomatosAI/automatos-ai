"""The running polling adapters, readable from every worker.

One worker owns channel polling (``channels.polling``); the others hold no
adapters, so their ``ChannelManager.is_running`` could only answer False. After
each pass the owner writes one Redis key per adapter it runs, with a TTL of a
few reconcile intervals, and deletes the key of each adapter it stopped. Every
worker reads the keys. A key outlives a dead owner by at most the TTL.

Without Redis (a supported local shape) nothing is published and the other
workers fall back to the database's status column.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from config import config

logger = logging.getLogger(__name__)

KEY_PREFIX = "channels:polling:running:"
#: A key lives this many reconcile intervals: a tick or two may be late without
#: a running adapter reading as stopped, and a dead owner's keys expire soon.
TTL_INTERVALS = 3


def running_key(connection_id: str) -> str:
    return f"{KEY_PREFIX}{connection_id}"


def ttl_seconds() -> int:
    return int(config.CHANNEL_POLLING_RECONCILE_SECONDS) * TTL_INTERVALS


def _redis() -> Optional[Any]:
    """The redis-py client, or None without Redis."""
    try:
        from core.redis.client import get_redis_client

        wrapper = get_redis_client()
        return wrapper.get_redis() if wrapper is not None else None
    except Exception:  # noqa: BLE001 — no Redis is a supported local shape
        return None


def publish(running: Iterable[str], stopped: Iterable[str]) -> None:
    """The owner's view after a pass: refresh a key per running adapter and drop
    the stopped ones. Never raises: the adapters run whether or not it lands."""
    client = _redis()
    if client is None:
        return
    try:
        pipe = client.pipeline(transaction=False)
        for connection_id in running:
            pipe.set(running_key(connection_id), "1", ex=ttl_seconds())
        for connection_id in stopped:
            pipe.delete(running_key(connection_id))
        pipe.execute()
    except Exception:  # noqa: BLE001 — status is advisory; the next pass publishes again
        logger.warning("[ChannelPolling] could not publish the running adapters", exc_info=True)


def is_running(connection_id: str) -> bool:
    """True when the owner published ``connection_id`` as running. Raises on a
    Redis error: the caller (``ChannelManager.is_running``) isolates it."""
    client = _redis()
    if client is None:
        return False
    return bool(client.exists(running_key(connection_id)))
