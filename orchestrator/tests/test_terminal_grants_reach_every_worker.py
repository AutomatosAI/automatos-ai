"""A Canvas terminal grant minted on one backend worker reaches the host through any worker.

The grants went to Redis through ``get_redis_client()``, the ``RedisClient`` wrapper, which
has no list commands: ``rpush`` raised AttributeError, the failure was logged at debug, and
every grant fell back to the minting process's own store. With more than one uvicorn worker
the host's heartbeat usually landed on another worker, which had no grant, so the terminal
did not open. Grants now go through the redis-py client (``wrapper.get_redis()``).
"""
from __future__ import annotations

import time

from services import cli_host_service as svc

HOST = "host-7"
LIFETIME_S = 60


class _Lists:
    """The redis-py list commands the grants use, over a dict."""

    def __init__(self) -> None:
        self.lists = {}

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value.encode("utf-8"))

    def expire(self, key, seconds):
        return True

    def lrange(self, key, start, end):
        return list(self.lists.get(key, []))

    def delete(self, key):
        self.lists.pop(key, None)


class _Wrapper:
    """Like ``RedisClient``: no list commands of its own, only ``get_redis()``."""

    def __init__(self, client: _Lists) -> None:
        self._client = client

    def get_redis(self) -> _Lists:
        return self._client


def test_a_grant_minted_on_one_worker_is_read_on_another(monkeypatch):
    shared = _Lists()
    monkeypatch.setattr("core.redis.client.get_redis_client", lambda: _Wrapper(shared))
    grant = {"token": "t-1", "expires_at": time.time() + LIFETIME_S}

    svc.push_terminal_grant(HOST, grant)
    svc._TERMINAL_GRANTS.clear()  # another worker: none of this process's store

    assert svc.pop_terminal_grants(HOST) == [grant]
    assert shared.lists == {}


def test_without_redis_the_grant_stays_in_this_process(monkeypatch):
    monkeypatch.setattr("core.redis.client.get_redis_client", lambda: None)
    grant = {"token": "t-2", "expires_at": time.time() + LIFETIME_S}

    svc.push_terminal_grant(HOST, grant)

    assert svc.pop_terminal_grants(HOST) == [grant]
