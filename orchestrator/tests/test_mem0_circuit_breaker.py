"""PRD-137 Fix #6: Mem0 circuit breaker HTTP status handling."""
import importlib.util
import pathlib

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


def _fresh_client(monkeypatch):
    breaker = _CircuitBreaker(threshold=3, cooldown_seconds=300)
    monkeypatch.setattr(mem0_mod, "_breaker", breaker)
    return Mem0Client(api_url="http://mem0.test", api_key="test-key"), breaker


# ── 2xx → success ──────────────────────────────────────────────────


def test_2xx_records_success(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)
    monkeypatch.setattr(
        mem0_mod.requests, "request",
        lambda *a, **kw: _FakeResponse(200, payload={}),
    )
    resp = client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp.status_code == 200
    assert breaker.failures == 0
    assert not breaker.is_open


# ── 4xx client errors → no breaker action, no retry ────────────────


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_error_no_breaker_action(monkeypatch, status):
    client, breaker = _fresh_client(monkeypatch)
    call_count = 0

    def fake(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(status)

    monkeypatch.setattr(mem0_mod.requests, "request", fake)
    resp = client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp.status_code == status
    assert call_count == 1  # no retry
    assert breaker.failures == 0
    assert not breaker.is_open


# ── 5xx → retry once, then breaker failure ──────────────────────────


def test_500_retries_then_records_failure(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)
    call_count = 0

    def fake_500(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(500)

    monkeypatch.setattr(mem0_mod.requests, "request", fake_500)
    monkeypatch.setattr(mem0_mod.time, "sleep", lambda _: None)
    resp = client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp.status_code == 500
    assert call_count == 2  # initial + 1 retry
    assert breaker.failures == 1


# ── 429 → retry once, then breaker failure ──────────────────────────


def test_429_retries_then_records_failure(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)
    call_count = 0

    def fake_429(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(429)

    monkeypatch.setattr(mem0_mod.requests, "request", fake_429)
    monkeypatch.setattr(mem0_mod.time, "sleep", lambda _: None)
    resp = client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp.status_code == 429
    assert call_count == 2
    assert breaker.failures == 1


# ── Breaker opens after threshold consecutive failures ──────────────


def test_breaker_opens_after_threshold(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)
    monkeypatch.setattr(
        mem0_mod.requests, "request",
        lambda *a, **kw: _FakeResponse(500),
    )
    monkeypatch.setattr(mem0_mod.time, "sleep", lambda _: None)

    for _ in range(3):
        client._request("GET", "http://mem0.test/api/v1/memories/")

    assert breaker.is_open
    assert breaker.failures == 3


# ── Open breaker skips request entirely ─────────────────────────────


def test_open_breaker_skips_request(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)
    breaker.is_open = True
    breaker.last_failure_time = mem0_mod.time.monotonic()

    called = False

    def should_not_call(*a, **kw):
        nonlocal called
        called = True

    monkeypatch.setattr(mem0_mod.requests, "request", should_not_call)
    resp = client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp is None
    assert not called


# ── Missing API key disables client ─────────────────────────────────


def test_no_api_key_disables_client(monkeypatch):
    breaker = _CircuitBreaker(threshold=3, cooldown_seconds=300)
    monkeypatch.setattr(mem0_mod, "_breaker", breaker)
    client = Mem0Client(api_url="http://mem0.test", api_key="")
    assert client.api_url == ""

    called = False

    def should_not_call(*a, **kw):
        nonlocal called
        called = True

    monkeypatch.setattr(mem0_mod.requests, "request", should_not_call)
    resp = client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp is None
    assert not called


# ── 5xx on retry followed by 2xx closes breaker ────────────────────


def test_success_after_transient_failure_closes_breaker(monkeypatch):
    client, breaker = _fresh_client(monkeypatch)
    responses = iter([_FakeResponse(500), _FakeResponse(200, payload={})])
    monkeypatch.setattr(
        mem0_mod.requests, "request",
        lambda *a, **kw: next(responses),
    )
    monkeypatch.setattr(mem0_mod.time, "sleep", lambda _: None)

    resp = client._request("GET", "http://mem0.test/api/v1/memories/")
    assert resp.status_code == 200
    assert breaker.failures == 0
    assert not breaker.is_open
