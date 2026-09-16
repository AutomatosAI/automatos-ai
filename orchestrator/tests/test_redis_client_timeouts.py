"""The shared Redis pool bounds every socket operation.

Several services call the sync client on the event loop (memory L1, the cache,
the rate limiter); before 2026-09-16 the pool had no timeouts, so one dead or
remote connection could hold the whole process for as long as the kernel let it.
"""
from __future__ import annotations

import asyncio

from core.redis import client as redis_client_module


def test_sync_pool_is_bounded_and_health_checked():
    rc = redis_client_module.RedisClient(host="127.0.0.1", port=1)
    kwargs = rc.pool.connection_kwargs
    assert kwargs["socket_timeout"] == redis_client_module.REDIS_SOCKET_TIMEOUT_S
    assert kwargs["socket_connect_timeout"] == redis_client_module.REDIS_CONNECT_TIMEOUT_S
    assert kwargs["health_check_interval"] == redis_client_module.REDIS_HEALTH_CHECK_INTERVAL_S
    assert kwargs["socket_keepalive"] is True
    assert 0 < redis_client_module.REDIS_SOCKET_TIMEOUT_S <= 5
    assert 0 < redis_client_module.REDIS_CONNECT_TIMEOUT_S <= 5


def test_async_pubsub_bounds_connect_but_not_the_idle_read(monkeypatch):
    captured: dict = {}

    class _FakePubSub:
        async def subscribe(self, channel):
            captured["channel"] = channel

    class _FakeRedis:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def pubsub(self):
            return _FakePubSub()

    monkeypatch.setattr(redis_client_module.aioredis, "Redis", _FakeRedis)
    rc = redis_client_module.RedisClient(host="127.0.0.1", port=1)
    _, pubsub = asyncio.run(rc.get_async_pubsub("events"))
    assert captured["channel"] == "events"
    assert captured["socket_connect_timeout"] == redis_client_module.REDIS_CONNECT_TIMEOUT_S
    assert captured["health_check_interval"] == redis_client_module.REDIS_HEALTH_CHECK_INTERVAL_S
    assert "socket_timeout" not in captured, "a subscription idles by design — no read timeout"
