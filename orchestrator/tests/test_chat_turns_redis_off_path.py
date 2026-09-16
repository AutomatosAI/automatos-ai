"""Redis never sits on a chat turn's delivery path.

2026-09-16: the local stack's registry was pointed at a remote Redis (the
checkout's .env carried the Railway URL) and every reply stalled mid-sentence —
the cross-process cancel poll ran between chunks, and redis-py's nested retries
turned a 1 s read timeout into minutes per poll. These tests pin the shape that
cannot stall: the poll lives in a watcher task beside the producer, every marker
op is bounded, the registry client retries nothing, and every ending is logged.
"""
from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from services import chat_turns
from services.chat_turns import TurnRegistry, run_detached_turn


class _Hook:
    def __init__(self) -> None:
        self.calls: list = []
        self.done = asyncio.Event()

    async def __call__(self, **kwargs) -> None:
        self.calls.append(kwargs)
        self.done.set()


class _RedisBackedRegistry(TurnRegistry):
    """A registry that believes Redis is configured, without dialling one."""

    def __init__(self) -> None:
        super().__init__()
        self.polls = 0

    def _client(self):  # non-None → the cancel watcher is started
        return object()

    async def mark_inflight(self, chat_id: str) -> None:
        return None

    async def clear_inflight(self, chat_id: str) -> None:
        return None


class _HangingRegistry(_RedisBackedRegistry):
    async def cancel_requested(self, chat_id: str) -> bool:
        self.polls += 1
        await asyncio.Event().wait()  # a Redis that never answers
        return False


class _MarkedRegistry(_RedisBackedRegistry):
    async def cancel_requested(self, chat_id: str) -> bool:
        self.polls += 1
        return True  # another worker set the cancel marker


async def _chunks(produced: list, n: int = 5, delay: float = 0.01):
    for i in range(n):
        await asyncio.sleep(delay)
        produced.append(i)
        yield f'0:"chunk {i}"\n'


def _watchers() -> list:
    return [t for t in asyncio.all_tasks() if t.get_name().startswith("chat-turn-cancel-watch:")]


def test_a_redis_that_never_answers_never_delays_a_chunk(monkeypatch):
    monkeypatch.setattr(chat_turns, "CANCEL_POLL_S", 0.001)

    async def scenario():
        reg = _HangingRegistry()
        hook = _Hook()
        produced: list = []
        started = time.monotonic()
        got = [
            c async for c in run_detached_turn(
                chat_id="c1", produce=lambda: _chunks(produced), on_complete=hook, registry=reg
            )
        ]
        elapsed = time.monotonic() - started
        await asyncio.wait_for(hook.done.wait(), timeout=2)
        await asyncio.sleep(0.05)  # let the cancelled watcher finish
        return got, produced, elapsed, hook.calls, reg.polls, _watchers()

    got, produced, elapsed, calls, polls, watchers = asyncio.run(scenario())
    assert produced == [0, 1, 2, 3, 4] and len(got) == 5
    assert elapsed < 1.0, f"the turn waited on Redis: {elapsed:.2f}s"
    assert polls >= 1, "the watcher did poll (and hung) while the turn streamed"
    assert calls == [{"completed": True, "cancelled": False, "client_gone": False}]
    assert watchers == [], "the watcher must not outlive the turn"


def test_a_cancel_marker_from_another_worker_stops_the_turn(monkeypatch):
    monkeypatch.setattr(chat_turns, "CANCEL_POLL_S", 0.01)

    async def endless():
        i = 0
        while True:
            await asyncio.sleep(0.005)
            i += 1
            yield f'0:"tick {i}"\n'

    async def scenario():
        reg = _MarkedRegistry()
        hook = _Hook()
        started = time.monotonic()
        got = [c async for c in run_detached_turn(chat_id="c2", produce=endless, on_complete=hook, registry=reg)]
        await asyncio.wait_for(hook.done.wait(), timeout=2)
        return got, hook.calls, time.monotonic() - started

    got, calls, elapsed = asyncio.run(scenario())
    assert len(got) < 200 and elapsed < 2.0
    assert calls == [{"completed": False, "cancelled": True, "client_gone": False}]


def test_every_marker_op_is_bounded(monkeypatch):
    monkeypatch.setattr(chat_turns, "REDIS_OP_TIMEOUT_S", 0.05)

    class _FrozenClient:
        async def get(self, key):
            await asyncio.Event().wait()

    async def scenario():
        reg = TurnRegistry()
        reg._redis_resolved = True
        reg._redis = _FrozenClient()
        started = time.monotonic()
        result = await reg._redis_call("get", "chat:turn:cancel:x")
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(scenario())
    assert result is None
    assert elapsed < 1.0, f"a marker read waited {elapsed:.2f}s"


def test_registry_client_retries_nothing_and_times_out_fast(monkeypatch):
    import core.redis.client as redis_client_module

    monkeypatch.setattr(
        redis_client_module,
        "get_redis_client",
        lambda: SimpleNamespace(host="127.0.0.1", port=1, password=None, db=0),
    )
    client = TurnRegistry()._client()
    assert client is not None
    assert client.get_retry()._retries == 0
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == 1
    assert kwargs["socket_connect_timeout"] == 1
    assert kwargs["retry"]._retries == 0


def test_a_completed_turn_and_a_failed_close_are_both_logged(caplog):
    class _Leaky:
        """Two chunks, then a close that fails the way a generator mid-await does."""

        def __init__(self) -> None:
            self._i = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._i >= 2:
                raise StopAsyncIteration
            self._i += 1
            return f'0:"c{self._i}"\n'

        async def aclose(self):
            raise RuntimeError("aclose(): asynchronous generator is already running")

    caplog.set_level(logging.INFO, logger="services.chat_turns")

    async def scenario():
        reg = TurnRegistry()
        reg._redis_resolved = True
        reg._redis = None
        hook = _Hook()
        got = [c async for c in run_detached_turn(chat_id="c5", produce=_Leaky, on_complete=hook, registry=reg)]
        await asyncio.wait_for(hook.done.wait(), timeout=2)
        return got

    got = asyncio.run(scenario())
    assert len(got) == 2
    messages = [r.getMessage() for r in caplog.records]
    assert any("turn started for chat c5" in m for m in messages)
    assert any("turn completed for chat c5 (2 chunks" in m for m in messages)
    assert any("producer aclose failed for chat c5" in m for m in messages)
    assert any(r.levelno == logging.WARNING and "aclose failed" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("name", ["REDIS_OP_TIMEOUT_S", "CANCEL_POLL_S"])
def test_bounds_are_short(name):
    assert 0 < getattr(chat_turns, name) <= 5
