"""PRD-194 S5 (P2-13, security §1.2.c) — Redis-backed shared widget rate limiter.

The previous ``RateLimitStore`` was a per-process in-memory dict: it reset on
every deploy, each of the 4 uvicorn workers kept its own window, and by its
own docstring the check was "only active when the key ID is already known".
This is the widget dossier's one sanctioned internal REPLACE (§E, §J-5): the
in-memory window is DELETED and the store is a Redis sorted-set sliding
window (the ``core/security/rate_limiter.py`` idiom, via the platform Redis
client), so:

- the window is **shared across workers** (two store instances on one Redis
  see one window);
- the identifier is resolved **pre-handler** — the FIRST request is gated,
  keyed on the presented key (hashed) or the client IP;
- the two money-spending endpoints (``/chat``, ``/callback``) carry a
  **per-IP ceiling that applies even when a key is presented**;
- **Redis down ⇒ fail OPEN, loudly** (locked decision) — a cache outage
  must not brick the widget, and must not stay silent (counter + ERROR log).

Pure: dict-backed fake Redis injected via the store's ``redis_factory`` seam;
full ASGI middleware invocations with a recording downstream app. No live
Redis, no network.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import api.widgets.rate_limit as rl_mod  # noqa: E402
from api.widgets.rate_limit import RateLimitStore, WidgetRateLimitMiddleware  # noqa: E402
from config import config  # noqa: E402


# ---------------------------------------------------------------- fake redis

class _FakePipeline:
    def __init__(self, r):
        self._r = r
        self._ops = []

    def __getattr__(self, name):
        def _queue(*args, **kwargs):
            self._ops.append((name, args, kwargs))
            return self

        return _queue

    def execute(self):
        return [getattr(self._r, name)(*args, **kwargs) for name, args, kwargs in self._ops]


class _FakeRedis:
    """Dict-backed sorted-set subset honouring exactly the commands we use."""

    def __init__(self):
        self.z: dict[str, dict[str, float]] = {}

    def zremrangebyscore(self, key, mn, mx):
        d = self.z.get(key, {})
        mn_f = float("-inf") if mn in ("-inf",) else float(mn)
        mx_f = float(mx)
        doomed = [m for m, s in d.items() if mn_f <= s <= mx_f]
        for m in doomed:
            d.pop(m)
        return len(doomed)

    def zcard(self, key):
        return len(self.z.get(key, {}))

    def zrange(self, key, start, end, withscores=False):
        items = sorted(self.z.get(key, {}).items(), key=lambda kv: kv[1])
        sel = items[start:] if end == -1 else items[start:end + 1]
        return sel if withscores else [m for m, _ in sel]

    def zadd(self, key, mapping):
        self.z.setdefault(key, {}).update(mapping)
        return len(mapping)

    def expire(self, key, ttl):
        return True

    def pipeline(self):
        return _FakePipeline(self)


class _PoisonedRedis:
    def pipeline(self):
        raise AssertionError("redis must not be touched for this path")


def _prime(fake: _FakeRedis, identifier: str, n: int):
    """Fill *identifier*'s window with n fresh marks (as other workers would)."""
    now = time.time()
    key = f"widget:rl:{identifier}"
    for i in range(n):
        fake.zadd(key, {f"prime-{i}": now - 0.001 * i})


# ---------------------------------------------------------------- ASGI rig

class _App:
    def __init__(self):
        self.calls: list[str] = []

    async def __call__(self, scope, receive, send):
        self.calls.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})


def _mw(app, fake_redis):
    return WidgetRateLimitMiddleware(app, store=RateLimitStore(redis_factory=lambda: fake_redis))


async def _call(mw, path, headers=None, method="POST", client=("9.9.9.9", 1234)):
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(k.lower().encode(), str(v).encode()) for k, v in (headers or {}).items()],
        "query_string": b"",
        "client": client,
    }
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    return sent


def _status(sent):
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def _headers(sent):
    start = next(m for m in sent if m["type"] == "http.response.start")
    return {k.decode(): v.decode() for k, v in start.get("headers", [])}


# ---------------------------------------------------------------- store

def test_rate_limit_shared_across_workers():
    """Two limiter instances on ONE Redis share the window — the property the
    deleted in-memory dict could never have."""
    fake = _FakeRedis()
    worker_a = RateLimitStore(redis_factory=lambda: fake)
    worker_b = RateLimitStore(redis_factory=lambda: fake)

    assert worker_a.check("key:shared", 3)[0] is True
    assert worker_b.check("key:shared", 3)[0] is True
    assert worker_a.check("key:shared", 3)[0] is True
    # Fourth request — made by the OTHER worker — is denied: shared state.
    allowed, limit, remaining, reset = worker_b.check("key:shared", 3)
    assert allowed is False
    assert remaining == 0
    assert reset >= 1


def test_rate_limit_window_slides():
    """Marks older than the window are dropped before counting."""
    fake = _FakeRedis()
    store = RateLimitStore(redis_factory=lambda: fake)
    window = config.WIDGET_RATE_LIMIT_WINDOW_SECONDS
    stale = time.time() - window - 5
    fake.zadd("widget:rl:ip:1.2.3.4", {f"old-{i}": stale for i in range(50)})
    allowed, _, remaining, _ = store.check("ip:1.2.3.4", 2)
    assert allowed is True  # the 50 stale marks were evicted, not counted
    assert remaining == 1


def test_limiter_degrades_open_on_redis_down(monkeypatch, caplog):
    """LOCKED DECISION: Redis unreachable ⇒ allow, count, log ERROR — the
    widget must not brick on a cache outage, and the outage must be loud."""
    store = RateLimitStore(redis_factory=lambda: None)
    before = rl_mod.get_redis_failure_count()
    with caplog.at_level(logging.ERROR):
        allowed, limit, remaining, reset = store.check("key:x", 5)
    assert allowed is True
    assert rl_mod.get_redis_failure_count() == before + 1
    assert any("FAIL-OPEN" in r.message for r in caplog.records)

    class _Exploding:
        def pipeline(self):
            raise ConnectionError("redis down")

    store2 = RateLimitStore(redis_factory=lambda: _Exploding())
    assert store2.check("key:x", 5)[0] is True
    assert rl_mod.get_redis_failure_count() == before + 2


# ---------------------------------------------------------------- middleware

def test_rate_limit_gates_first_request_by_ip():
    """A request with NO api key is limited by client_ip on its FIRST hit —
    the old middleware only checked once a key id was already known."""
    fake = _FakeRedis()
    _prime(fake, "ip:9.9.9.9", config.WIDGET_RATE_LIMIT_PUBLIC_PER_WINDOW)
    app = _App()
    sent = asyncio.run(_call(_mw(app, fake), "/api/widgets/config"))
    assert _status(sent) == 429
    assert app.calls == []  # never reached the handler
    body = json.loads(next(m["body"] for m in sent if m["type"] == "http.response.body"))
    assert body["detail"] == "Rate limit exceeded"
    assert "retry-after" in _headers(sent)


def test_allowed_request_passes_with_headers():
    fake = _FakeRedis()
    app = _App()
    sent = asyncio.run(_call(_mw(app, fake), "/api/widgets/config",
                             headers={"Authorization": "Bearer ak_pub_abc123"}))
    assert _status(sent) == 200
    assert app.calls == ["/api/widgets/config"]
    h = _headers(sent)
    assert h["x-ratelimit-limit"] == str(config.WIDGET_RATE_LIMIT_PUBLIC_PER_WINDOW)
    assert int(h["x-ratelimit-remaining"]) >= 0
    assert int(h["x-ratelimit-reset"]) >= 1


def test_key_identifier_is_hashed_not_raw():
    """No key material in Redis: the bucket key is a SHA-256 digest."""
    fake = _FakeRedis()
    app = _App()
    token = "ak_pub_supersecrettoken"
    asyncio.run(_call(_mw(app, fake), "/api/widgets/config",
                      headers={"Authorization": f"Bearer {token}"}))
    assert fake.z, "a window key must have been created"
    assert all(token not in k for k in fake.z)


def test_callback_per_ip_ceiling():
    """The callback endpoint rejects over-ceiling per-IP EVEN with a valid
    key presented (was: per-IP explicitly deferred until Redis)."""
    fake = _FakeRedis()
    _prime(fake, "ipceil:callback:9.9.9.9", config.WIDGET_CALLBACK_IP_LIMIT_PER_WINDOW)
    app = _App()
    sent = asyncio.run(_call(_mw(app, fake), "/api/widgets/callback",
                             headers={"Authorization": "Bearer ak_pub_abc123"}))
    assert _status(sent) == 429
    assert app.calls == []


def test_chat_per_ip_ceiling():
    fake = _FakeRedis()
    _prime(fake, "ipceil:chat:9.9.9.9", config.WIDGET_CHAT_IP_LIMIT_PER_WINDOW)
    app = _App()
    sent = asyncio.run(_call(_mw(app, fake), "/api/widgets/chat",
                             headers={"Authorization": "Bearer ak_pub_abc123"}))
    assert _status(sent) == 429
    assert app.calls == []


def test_non_money_endpoint_has_no_ip_ceiling():
    """Other widget endpoints carry only the per-key/per-IP base limit — a
    keyed request under its key limit passes even from a hot IP."""
    fake = _FakeRedis()
    _prime(fake, "ipceil:chat:9.9.9.9", 10_000)  # irrelevant to /config
    app = _App()
    sent = asyncio.run(_call(_mw(app, fake), "/api/widgets/config",
                             headers={"Authorization": "Bearer ak_pub_abc123"}))
    assert _status(sent) == 200


def test_non_widget_paths_untouched():
    app = _App()
    mw = WidgetRateLimitMiddleware(app, store=RateLimitStore(redis_factory=lambda: _PoisonedRedis()))
    sent = asyncio.run(_call(mw, "/api/agents"))
    assert _status(sent) == 200
    assert app.calls == ["/api/agents"]
    assert "x-ratelimit-limit" not in _headers(sent)


def test_options_preflight_passes_ungated():
    app = _App()
    mw = WidgetRateLimitMiddleware(app, store=RateLimitStore(redis_factory=lambda: _PoisonedRedis()))
    sent = asyncio.run(_call(mw, "/api/widgets/chat", method="OPTIONS"))
    assert _status(sent) == 200


# ---------------------------------------------------------------- REPLACE pin

def test_in_memory_window_is_gone():
    """The dossier's sanctioned REPLACE (§E, §J-5): the per-process dict is
    DELETED, not left mounted beside the Redis store (CLAUDE.md §5)."""
    src = Path(rl_mod.__file__).read_text()
    assert "defaultdict" not in src
    assert "from threading import Lock" not in src
    assert "core/redis/client.py" in src  # the shared store is the Redis one
