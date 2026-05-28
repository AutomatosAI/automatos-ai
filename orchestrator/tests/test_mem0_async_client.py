"""PRD-141 US-003: Mem0Client async httpx.AsyncClient conversion.

These tests pin the async contract of the rewritten client:
- ``_request`` is a coroutine driven by a pooled ``httpx.AsyncClient``;
- retries sleep via ``await asyncio.sleep`` (never the blocking ``time.sleep``);
- the client instance is reused across calls (connection pooling);
- write operations honour the larger ``write_timeout``.

PRD-141 US-004 adds per-workspace circuit-breaker coverage: breakers live on
``Mem0Client._breakers`` (keyed by workspace_id), so a failure in one workspace
does not trip every other workspace.

The module is loaded by path (mirroring test_mem0_circuit_breaker.py) so the
test patches the same objects the client uses — ``Mem0Client._breakers`` and the
module's ``asyncio``.
"""
import importlib.util
import pathlib

import httpx
import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "mem0_client_async_mod",
    _ROOT / "modules" / "memory" / "integrations" / "mem0_client.py",
)
mem0_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mem0_mod)

Mem0Client = mem0_mod.Mem0Client
_CircuitBreaker = mem0_mod._CircuitBreaker


class _FakeResponse:
    """Stand-in for httpx.Response — only the attributes the client reads."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


async def _instant_sleep(_delay):
    """Async stand-in for asyncio.sleep so retry backoff doesn't slow tests."""
    return None


def _fresh_client(monkeypatch):
    """A client with an isolated breaker registry (reset between tests).

    Replaces the class-level ``_breakers`` dict with a fresh one seeded with the
    default ``"_global"`` breaker, so a test that calls ``_request`` without a
    ``workspace_id`` shares that exact object. monkeypatch restores the original
    registry after the test, keeping per-workspace state from leaking.
    """
    breaker = _CircuitBreaker(threshold=3, cooldown_seconds=300)
    monkeypatch.setattr(Mem0Client, "_breakers", {"_global": breaker})
    return Mem0Client(api_url="http://mem0.test", api_key="test-key"), breaker


@pytest.mark.asyncio
async def test_async_request_success(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)

    async def fake_request(self, method, url, **kwargs):
        return _FakeResponse(200, payload={})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    resp = await client._request("GET", "http://mem0.test/api/v1/memories/")

    assert resp.status_code == 200
    assert breaker.failures == 0
    assert not breaker.is_open
    await client.aclose()


@pytest.mark.asyncio
async def test_async_retry_with_asyncio_sleep(monkeypatch):
    """A transient 5xx retries once, awaiting asyncio.sleep between attempts."""
    client, breaker = _fresh_client(monkeypatch)
    call_count = 0

    async def fake_500(self, method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(500)

    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_500)
    monkeypatch.setattr(mem0_mod.asyncio, "sleep", fake_sleep)

    resp = await client._request("GET", "http://mem0.test/api/v1/memories/")

    assert resp.status_code == 500
    assert call_count == 2          # initial attempt + one retry
    assert len(sleep_calls) == 1    # backoff went through await asyncio.sleep
    assert breaker.failures == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_connection_pooling(monkeypatch):
    """_get_client returns one pooled AsyncClient, recreated only after aclose."""
    client, _ = _fresh_client(monkeypatch)

    c1 = client._get_client()
    c2 = client._get_client()
    assert c1 is c2                       # reused, not recreated per call
    assert isinstance(c1, httpx.AsyncClient)

    await client.aclose()
    c3 = client._get_client()
    assert c3 is not c1                   # closed client is replaced
    await client.aclose()


@pytest.mark.asyncio
async def test_write_timeout_respected(monkeypatch):
    """add() (a write) passes the larger write_timeout down to httpx."""
    client, _ = _fresh_client(monkeypatch)
    # Pin the timeouts explicitly. test_unified_memory swaps the global config
    # for a MagicMock at import time, so when these files run together the
    # config-derived timeouts would otherwise both collapse to float(MagicMock)
    # == 1.0 and this assertion (write > read) would fail spuriously.
    client.timeout = 3.0
    client.write_timeout = 15.0
    captured = {}

    async def fake_request(self, method, url, **kwargs):
        captured.update(kwargs)
        return _FakeResponse(200, payload={})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    await client.add(
        messages=[{"role": "user", "content": "hello"}],
        user_id="u1",
    )

    assert captured["timeout"] == client.write_timeout
    assert client.write_timeout > client.timeout   # write budget exceeds read
    await client.aclose()


# ── US-004: per-workspace circuit breaker ───────────────────────────


@pytest.mark.asyncio
async def test_per_workspace_breaker_isolation(monkeypatch):
    """A Mem0 outage in one workspace must not trip the breaker for others."""
    # Seed workspace "A" with an explicit threshold=3 breaker so the test is
    # deterministic regardless of the ambient MEM0_CIRCUIT_* config; "B" is
    # created lazily and, never having failed, stays closed whatever its config.
    breaker_a = _CircuitBreaker(threshold=3, cooldown_seconds=300)
    monkeypatch.setattr(Mem0Client, "_breakers", {"A": breaker_a})
    client = Mem0Client(api_url="http://mem0.test", api_key="test-key")

    async def fake_500(self, method, url, **kwargs):
        return _FakeResponse(500)

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_500)
    monkeypatch.setattr(mem0_mod.asyncio, "sleep", _instant_sleep)

    # Drive workspace "A" to its failure threshold; never touch "B".
    for _ in range(3):
        await client._request(
            "GET", "http://mem0.test/api/v1/memories/", workspace_id="A"
        )

    breaker_b = Mem0Client._get_breaker("B")
    assert breaker_a.is_open                 # A is now failing fast
    assert not breaker_b.is_open             # B is unaffected
    assert breaker_a is not breaker_b        # genuinely separate breakers
    await client.aclose()


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold(monkeypatch):
    """The per-workspace breaker opens after `threshold` consecutive failures."""
    # Explicit threshold=3 breaker — don't inherit the threshold from config.
    breaker = _CircuitBreaker(threshold=3, cooldown_seconds=300)
    monkeypatch.setattr(Mem0Client, "_breakers", {"ws": breaker})
    client = Mem0Client(api_url="http://mem0.test", api_key="test-key")

    async def fake_500(self, method, url, **kwargs):
        return _FakeResponse(500)

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_500)
    monkeypatch.setattr(mem0_mod.asyncio, "sleep", _instant_sleep)

    for _ in range(3):
        await client._request(
            "GET", "http://mem0.test/api/v1/memories/", workspace_id="ws"
        )

    assert breaker.is_open
    assert breaker.failures == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_breaker_half_open_probe(monkeypatch):
    """After cooldown an open breaker allows one probe; a success closes it."""
    # Force workspace "ws" open, with the last failure pushed beyond cooldown so
    # allow_request() returns the half-open probe.
    breaker = _CircuitBreaker(threshold=3, cooldown_seconds=300)
    breaker.is_open = True
    breaker.failures = 3
    breaker.last_failure_time = mem0_mod.time.monotonic() - (breaker.cooldown + 1)
    monkeypatch.setattr(Mem0Client, "_breakers", {"ws": breaker})
    client = Mem0Client(api_url="http://mem0.test", api_key="test-key")

    async def fake_ok(self, method, url, **kwargs):
        return _FakeResponse(200, payload={})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_ok)

    resp = await client._request(
        "GET", "http://mem0.test/api/v1/memories/", workspace_id="ws"
    )
    assert resp.status_code == 200
    assert not breaker.is_open      # successful probe closed the breaker
    assert breaker.failures == 0
    await client.aclose()
