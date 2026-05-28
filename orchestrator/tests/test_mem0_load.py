"""PRD-141 US-008 GATE: Mem0 concurrency / thread-pool-starvation load test.

Phase 1 replaced ``run_in_executor``-wrapped Mem0 calls (US-005) with direct
``await`` on an ``httpx.AsyncClient`` (US-003). The executor path bounded
concurrency by the default ThreadPoolExecutor (~min(32, cpu+4) workers), so a
burst of memory ops queued in waves and could starve other executor users. The
async path runs every call as a coroutine on the one event loop, bounded only
by the httpx connection pool — no thread pool involved.

This test fires 50 concurrent searches against a fake transport with a fixed
per-call latency and asserts they complete in roughly ONE call's time (one
wave), not 50x. It is a structural concurrency proof against a mock, not a
throughput benchmark of the live Mem0 server — the 24h INBUILD soak (US-008)
covers real-world behaviour.

Run with ``-s`` to see the captured latency numbers.
"""
import asyncio
import importlib.util
import pathlib
import time

import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "mem0_client_load_mod",
    _ROOT / "modules" / "memory" / "integrations" / "mem0_client.py",
)
mem0_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mem0_mod)

Mem0Client = mem0_mod.Mem0Client
_CircuitBreaker = mem0_mod._CircuitBreaker

_CONCURRENCY = 50
_PER_CALL_S = 0.05  # simulated Mem0 round-trip


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []
        self.text = ""

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_50_concurrent_searches_no_starvation(monkeypatch, capsys):
    """50 concurrent searches complete in ~one wave and none are dropped."""
    breaker = _CircuitBreaker(threshold=3, cooldown_seconds=60)
    monkeypatch.setattr(Mem0Client, "_breakers", {"_global": breaker})
    client = Mem0Client(api_url="http://mem0.test", api_key="test-key")

    in_flight = 0
    max_in_flight = 0

    async def fake_request(self, method, url, **kwargs):
        # Track concurrency: if the calls were serialized (executor wave / pool
        # starvation) max_in_flight would stay at 1.
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await asyncio.sleep(_PER_CALL_S)
            return _FakeResponse(200, payload=[])
        finally:
            in_flight -= 1

    monkeypatch.setattr(mem0_mod.httpx.AsyncClient, "request", fake_request)

    start = time.monotonic()
    results = await asyncio.gather(
        *(
            client.search(query="q", user_id=f"u{i}", limit=1)
            for i in range(_CONCURRENCY)
        )
    )
    elapsed = time.monotonic() - start

    serial_estimate = _PER_CALL_S * _CONCURRENCY
    speedup = serial_estimate / elapsed if elapsed else float("inf")

    # Every call returned (no drops) and the breaker never tripped.
    assert len(results) == _CONCURRENCY
    assert all(r == [] for r in results)
    assert not breaker.is_open
    assert breaker.failures == 0

    # All 50 were genuinely in flight at once (async), not run in serial waves.
    assert max_in_flight == _CONCURRENCY
    # Wall time is ~one call, not 50; allow generous CI headroom (>=5x speedup).
    assert elapsed < _PER_CALL_S * 10, (
        f"expected ~{_PER_CALL_S}s (concurrent), got {elapsed:.3f}s — "
        "calls appear serialized"
    )

    with capsys.disabled():
        print(
            f"\n[US-008 load] {_CONCURRENCY} concurrent searches: "
            f"wall={elapsed*1000:.0f}ms, serial_estimate={serial_estimate*1000:.0f}ms, "
            f"speedup={speedup:.1f}x, peak_in_flight={max_in_flight}"
        )

    await client.aclose()
