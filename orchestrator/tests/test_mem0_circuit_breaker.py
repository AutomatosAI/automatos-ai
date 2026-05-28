"""PRD-137 Fix #6 / PRD-141 US-003: Mem0 circuit breaker HTTP status handling.

Converted to the async httpx interface in PRD-141 US-003: the client now drives
a pooled ``httpx.AsyncClient``, so these tests mock ``httpx.AsyncClient.request``
(an async method) rather than ``requests.request``, await ``_request``, and let
backoff go through ``asyncio.sleep`` instead of ``time.sleep``.
"""
import importlib.util
import pathlib

import httpx
import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "mem0_client_mod",
    _ROOT / "modules" / "memory" / "integrations" / "mem0_client.py",
)
mem0_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mem0_mod)

Mem0Client = mem0_mod.Mem0Client
_CircuitBreaker = mem0_mod._CircuitBreaker


class _FakeResponse:
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
    # PRD-141 US-004: the module-level _breaker singleton is gone; breakers are
    # now per-workspace on Mem0Client._breakers. These tests call _request with
    # no workspace_id, so they share the "_global" breaker — seed a fresh one and
    # let monkeypatch restore the registry afterwards.
    breaker = _CircuitBreaker(threshold=3, cooldown_seconds=300)
    monkeypatch.setattr(Mem0Client, "_breakers", {"_global": breaker})
    return Mem0Client(api_url="http://mem0.test", api_key="test-key"), breaker


# ── 2xx → success ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_2xx_records_success(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)

    async def fake(self, method, url, **kwargs):
        return _FakeResponse(200, payload={})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake)
    resp = await client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp.status_code == 200
    assert breaker.failures == 0
    assert not breaker.is_open
    await client.aclose()


# ── 4xx client errors → no breaker action, no retry ────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_client_error_no_breaker_action(monkeypatch, status):
    client, breaker = _fresh_client(monkeypatch)
    call_count = 0

    async def fake(self, method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(status)

    monkeypatch.setattr(httpx.AsyncClient, "request", fake)
    resp = await client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp.status_code == status
    assert call_count == 1  # no retry
    assert breaker.failures == 0
    assert not breaker.is_open
    await client.aclose()


# ── 5xx → retry once, then breaker failure ──────────────────────────


@pytest.mark.asyncio
async def test_500_retries_then_records_failure(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)
    call_count = 0

    async def fake_500(self, method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(500)

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_500)
    monkeypatch.setattr(mem0_mod.asyncio, "sleep", _instant_sleep)
    resp = await client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp.status_code == 500
    assert call_count == 2  # initial + 1 retry
    assert breaker.failures == 1
    await client.aclose()


# ── 429 → retry once, then breaker failure ──────────────────────────


@pytest.mark.asyncio
async def test_429_retries_then_records_failure(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)
    call_count = 0

    async def fake_429(self, method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(429)

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_429)
    monkeypatch.setattr(mem0_mod.asyncio, "sleep", _instant_sleep)
    resp = await client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp.status_code == 429
    assert call_count == 2
    assert breaker.failures == 1
    await client.aclose()


# ── Breaker opens after threshold consecutive failures ──────────────


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)

    async def fake(self, method, url, **kwargs):
        return _FakeResponse(500)

    monkeypatch.setattr(httpx.AsyncClient, "request", fake)
    monkeypatch.setattr(mem0_mod.asyncio, "sleep", _instant_sleep)

    for _ in range(3):
        await client._request("GET", "http://mem0.test/api/v1/memories/")

    assert breaker.is_open
    assert breaker.failures == 3
    await client.aclose()


# ── Open breaker skips request entirely ─────────────────────────────


@pytest.mark.asyncio
async def test_open_breaker_skips_request(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)
    breaker.is_open = True
    breaker.last_failure_time = mem0_mod.time.monotonic()

    called = False

    async def should_not_call(self, method, url, **kwargs):
        nonlocal called
        called = True
        return _FakeResponse(200)

    monkeypatch.setattr(httpx.AsyncClient, "request", should_not_call)
    resp = await client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp is None
    assert not called
    await client.aclose()


# ── Missing API URL disables client ─────────────────────────────────


@pytest.mark.asyncio
async def test_no_api_url_disables_client(monkeypatch):
    # The client disables itself when no api_url is configured (api_key alone
    # is not enough). A disabled client never issues an HTTP request. Blank the
    # config fallback too, since __init__ uses (arg or config.MEM0_API_URL or "").
    from config import config as _config
    monkeypatch.setattr(_config, "MEM0_API_URL", "")
    client = Mem0Client(api_url="", api_key="")
    assert client.api_url == ""

    called = False

    async def should_not_call(self, method, url, **kwargs):
        nonlocal called
        called = True
        return _FakeResponse(200)

    monkeypatch.setattr(httpx.AsyncClient, "request", should_not_call)
    resp = await client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp is None
    assert not called
    await client.aclose()


# ── 5xx on retry followed by 2xx closes breaker ────────────────────


@pytest.mark.asyncio
async def test_success_after_transient_failure_closes_breaker(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)
    responses = iter([_FakeResponse(500), _FakeResponse(200, payload={})])

    async def fake(self, method, url, **kwargs):
        return next(responses)

    monkeypatch.setattr(httpx.AsyncClient, "request", fake)
    monkeypatch.setattr(mem0_mod.asyncio, "sleep", _instant_sleep)

    resp = await client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp.status_code == 200
    assert breaker.failures == 0
    assert not breaker.is_open
    await client.aclose()
