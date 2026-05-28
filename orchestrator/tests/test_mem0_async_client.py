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
    """add() (a write) passes the 5.0s write_timeout (US-007) down to httpx."""
    client, _ = _fresh_client(monkeypatch)
    # Pin the timeouts explicitly to the US-007 defaults (read 3.0 / write 5.0).
    # test_unified_memory swaps the global config for a MagicMock at import time,
    # so when these files run together the config-derived timeouts would
    # otherwise both collapse to float(MagicMock) == 1.0 and this assertion
    # (write > read) would fail spuriously.
    client.timeout = 3.0
    client.write_timeout = 5.0
    captured = {}

    async def fake_request(self, method, url, **kwargs):
        captured.update(kwargs)
        return _FakeResponse(200, payload={})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    await client.add(
        messages=[{"role": "user", "content": "hello"}],
        user_id="u1",
    )

    assert captured["timeout"] == 5.0             # the 5.0s write timeout is applied
    assert client.write_timeout > client.timeout  # write budget exceeds read
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


@pytest.mark.asyncio
async def test_half_open_404_closes_breaker(monkeypatch):
    """A half-open probe that lands on an empty namespace (404) must close.

    Regression for the case where a recovered Mem0 answers a search for a
    workspace that simply has no memories yet: that 404 means "reachable, no
    rows", so it has to count as a healthy call. If a 404 were treated as a
    non-success the breaker would stay OPEN forever for any empty namespace,
    because every half-open probe would keep landing on the same empty 404.
    """
    breaker = _CircuitBreaker(threshold=3, cooldown_seconds=300)
    breaker.is_open = True
    breaker.failures = 3
    breaker.last_failure_time = mem0_mod.time.monotonic() - (breaker.cooldown + 1)
    monkeypatch.setattr(Mem0Client, "_breakers", {"ws": breaker})
    client = Mem0Client(api_url="http://mem0.test", api_key="test-key")

    async def fake_404(self, method, url, **kwargs):
        return _FakeResponse(404, payload=[])

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_404)

    resp = await client._request(
        "GET", "http://mem0.test/api/v1/memories/", workspace_id="ws"
    )
    assert resp.status_code == 404
    assert not breaker.is_open      # empty-namespace 404 still closed the breaker
    assert breaker.failures == 0
    await client.aclose()


# ── US-005: UnifiedMemoryService awaits Mem0 directly (no executor) ──


def test_unified_memory_no_executor():
    """UnifiedMemoryService must await the async Mem0 client directly.

    Wrapping a Mem0 call in an executor would consume a thread per memory op
    and — now that the client is async — schedule a coroutine on a thread that
    never awaits it, silently dropping the write. A source-level check is
    deliberate: importing UnifiedMemoryService here would drag in the full
    package + Redis/config import chain, so we assert the structural contract
    directly on the service source.
    """
    source = (_ROOT / "modules" / "memory" / "unified_memory_service.py").read_text()

    # Built at runtime so this test file does not itself trip the repo grep
    # gate that forbids executor-wrapped Mem0 calls (US-005 criterion).
    executor_call = "run_in_" + "executor"

    mem0_calls = 0
    for line in source.splitlines():
        if "self._mem0." in line:
            mem0_calls += 1
            assert "await self._mem0." in line, f"non-awaited Mem0 call: {line.strip()}"
        if executor_call in line and "mem0" in line:
            raise AssertionError(f"Mem0 call wrapped in an executor: {line.strip()}")

    assert mem0_calls, "expected self._mem0 calls in unified_memory_service.py"


# ── US-006: proactive Mem0 health probe ─────────────────────────────


@pytest.mark.asyncio
async def test_health_check_bypasses_breaker(monkeypatch):
    """health_check reaches Mem0 even with the breaker OPEN and maps the result
    to reachability: any <500 response (incl. 401/404) is up; a transport error
    is down. The probe must bypass the breaker because it is the signal that
    decides breaker state."""
    breaker = _CircuitBreaker(threshold=3, cooldown_seconds=300)
    breaker.force_open()  # breaker OPEN — would short-circuit a normal _request
    monkeypatch.setattr(Mem0Client, "_breakers", {"_global": breaker})
    client = Mem0Client(api_url="http://mem0.test", api_key="test-key")

    async def fake_401(self, method, url, **kwargs):
        return _FakeResponse(401)

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_401)
    assert await client.health_check() is True  # reachable despite open breaker

    async def fake_down(self, method, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_down)
    assert await client.health_check() is False  # transport error == down
    await client.aclose()


@pytest.mark.asyncio
async def test_health_probe_trips_all_breakers(monkeypatch):
    """A failing health probe force-opens every known workspace breaker so the
    whole platform fails fast at once."""
    breakers = {
        "A": _CircuitBreaker(threshold=3, cooldown_seconds=300),
        "B": _CircuitBreaker(threshold=3, cooldown_seconds=300),
        "_global": _CircuitBreaker(threshold=3, cooldown_seconds=300),
    }
    monkeypatch.setattr(Mem0Client, "_breakers", breakers)
    client = Mem0Client(api_url="http://mem0.test", api_key="test-key")

    async def fake_unhealthy(self):
        return False

    monkeypatch.setattr(Mem0Client, "health_check", fake_unhealthy)

    healthy = await client.run_health_probe()

    assert healthy is False
    assert all(b.is_open for b in breakers.values())  # every workspace tripped
    await client.aclose()


@pytest.mark.asyncio
async def test_health_probe_resets_on_recovery(monkeypatch):
    """A successful probe closes every breaker that a prior outage had opened."""
    breakers = {
        "A": _CircuitBreaker(threshold=3, cooldown_seconds=300),
        "B": _CircuitBreaker(threshold=3, cooldown_seconds=300),
    }
    for b in breakers.values():
        b.force_open()  # simulate a prior outage that tripped everything
    monkeypatch.setattr(Mem0Client, "_breakers", breakers)
    client = Mem0Client(api_url="http://mem0.test", api_key="test-key")

    async def fake_healthy(self):
        return True

    monkeypatch.setattr(Mem0Client, "health_check", fake_healthy)

    healthy = await client.run_health_probe()

    assert healthy is True
    assert not any(b.is_open for b in breakers.values())  # all closed again
    assert all(b.failures == 0 for b in breakers.values())
    await client.aclose()


# ── US-007: tightened Mem0 config defaults ──────────────────────────


def test_mem0_config_defaults_tightened():
    """Pin the US-007 default budgets directly on config.py source.

    A source-level check is deliberate: test_unified_memory swaps the global
    ``config`` for a MagicMock at import time, so reading the live attributes in
    a combined run is unreliable. Asserting on the source guards the shipped
    defaults regardless of collection order.
    """
    source = (_ROOT / "config.py").read_text()
    assert 'MEM0_WRITE_TIMEOUT_SECONDS", "5.0"' in source   # 15.0 -> 5.0
    assert 'MEM0_CIRCUIT_COOLDOWN_SECONDS", "60"' in source  # 300 -> 60
    # Unchanged by US-007.
    assert 'MEM0_TIMEOUT_SECONDS", "3.0"' in source
    assert 'MEM0_CIRCUIT_THRESHOLD", "3"' in source
