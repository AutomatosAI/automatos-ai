"""PRD-237 S7 — a chat turn finishes even when the browser goes away.

Pure asyncio tests of ``services.chat_turns`` — no app, no DB, no Redis (the
registry is pinned to process-local mode). Covers the four outcomes the design
promises:

1. the consumer closing early (disconnect) does NOT cancel the producer, the
   turn completes and ``on_complete`` sees ``client_gone=True``;
2. a fully consumed turn reports ``client_gone=False`` (no duplicate notify);
3. an explicit ``request_cancel`` stops a running turn (``cancelled=True``);
4. a producer exception surfaces as the ``e:`` error frame, never a lost loop.
"""
from __future__ import annotations

import asyncio

import pytest

from services import chat_turns
from services.chat_turns import TurnRegistry, error_frame, run_detached_turn


def _local_registry() -> TurnRegistry:
    reg = TurnRegistry()
    reg._redis_resolved = True  # never dial Redis in tests
    reg._redis = None
    return reg


class _Hook:
    def __init__(self) -> None:
        self.calls: list = []
        self.done = asyncio.Event()

    async def __call__(self, **kwargs) -> None:
        self.calls.append(kwargs)
        self.done.set()


async def _chunks(produced: list, n: int = 3, delay: float = 0.01):
    for i in range(n):
        await asyncio.sleep(delay)
        produced.append(i)
        yield f'0:"chunk {i}"\n'


def test_disconnect_does_not_cancel_the_turn():
    async def scenario():
        reg = _local_registry()
        produced: list = []
        hook = _Hook()
        stream = run_detached_turn(
            chat_id="c1", produce=lambda: _chunks(produced), on_complete=hook, registry=reg
        )
        first = await stream.__anext__()
        assert first == '0:"chunk 0"\n'
        await stream.aclose()  # the browser went away after one chunk
        await asyncio.wait_for(hook.done.wait(), timeout=2)
        return produced, hook.calls

    produced, calls = asyncio.run(scenario())
    assert produced == [0, 1, 2], "the producer must run to completion after the disconnect"
    assert calls == [{"completed": True, "cancelled": False, "client_gone": True}]


def test_full_consumption_reports_client_present():
    async def scenario():
        reg = _local_registry()
        hook = _Hook()
        got = [c async for c in run_detached_turn(chat_id="c2", produce=lambda: _chunks([]), on_complete=hook, registry=reg)]
        await asyncio.wait_for(hook.done.wait(), timeout=2)
        return got, hook.calls, await reg.is_in_flight("c2")

    got, calls, in_flight = asyncio.run(scenario())
    assert got == ['0:"chunk 0"\n', '0:"chunk 1"\n', '0:"chunk 2"\n']
    assert calls == [{"completed": True, "cancelled": False, "client_gone": False}]
    assert in_flight is False


def test_request_cancel_stops_a_running_turn():
    async def endless():
        i = 0
        while True:
            await asyncio.sleep(0.005)
            i += 1
            yield f'0:"tick {i}"\n'

    async def scenario():
        reg = _local_registry()
        hook = _Hook()
        stream = run_detached_turn(chat_id="c3", produce=endless, on_complete=hook, registry=reg)
        await stream.__anext__()
        assert await reg.is_in_flight("c3") is True
        assert await reg.request_cancel("c3") is True
        # The consumer sees the stream end instead of hanging.
        rest = [c async for c in stream]
        await asyncio.wait_for(hook.done.wait(), timeout=2)
        return rest, hook.calls, await reg.is_in_flight("c3")

    rest, calls, in_flight = asyncio.run(scenario())
    assert len(rest) < 50
    assert calls == [{"completed": False, "cancelled": True, "client_gone": False}]
    assert in_flight is False


def test_producer_exception_becomes_error_frame():
    async def broken():
        yield '0:"partial"\n'
        raise RuntimeError("boom")

    async def scenario():
        reg = _local_registry()
        hook = _Hook()
        got = [c async for c in run_detached_turn(chat_id="c4", produce=broken, on_complete=hook, registry=reg)]
        await asyncio.wait_for(hook.done.wait(), timeout=2)
        return got, hook.calls

    got, calls = asyncio.run(scenario())
    assert got == ['0:"partial"\n', error_frame("boom")]
    assert calls == [{"completed": False, "cancelled": False, "client_gone": False}]


def test_registry_without_redis_is_honest_about_unknown_turns():
    async def scenario():
        reg = _local_registry()
        return await reg.request_cancel("nobody"), await reg.is_in_flight("nobody"), await reg.cancel_requested("nobody")

    cancelled, in_flight, requested = asyncio.run(scenario())
    assert cancelled is False
    assert in_flight is False
    assert requested is False


def test_error_frame_is_the_ai_sdk_error_line():
    # PRD-239 S4: the same {"message", "code"} object the streaming handler emits,
    # so the client parses one shape wherever a turn fails.
    assert error_frame('bad "quote"') == 'e:{"message": "bad \\"quote\\""}\n'
    assert error_frame("turn died", code="turn_failed") == 'e:{"message": "turn died", "code": "turn_failed"}\n'


def test_singleton_registry():
    assert chat_turns.get_turn_registry() is chat_turns.get_turn_registry()


@pytest.mark.parametrize("ttl", [chat_turns.INFLIGHT_TTL_S, chat_turns.CANCEL_TTL_S])
def test_marker_ttls_are_bounded(ttl):
    # A forgotten marker must expire on its own — no marker may live for hours.
    assert 0 < ttl <= 60 * 60
