"""PRD-141 US-003: Mem0Client async httpx.AsyncClient conversion.

These tests pin the async contract of the rewritten client:
- ``_request`` is a coroutine driven by a pooled ``httpx.AsyncClient``;
- retries sleep via ``await asyncio.sleep`` (never the blocking ``time.sleep``);
- the client instance is reused across calls (connection pooling);
- write operations honour the larger ``write_timeout``.

The module is loaded by path (mirroring test_mem0_circuit_breaker.py) so the
test patches the same module object the client closes over (``_breaker``,
``asyncio``).
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


def _fresh_client(monkeypatch):
    """A client with an isolated (reset) module breaker."""
    breaker = _CircuitBreaker(threshold=3, cooldown_seconds=300)
    monkeypatch.setattr(mem0_mod, "_breaker", breaker)
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
