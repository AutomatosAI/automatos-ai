"""PRD-194 S2 (P2-13, security §1.1/§1.3.c) — webhook replay guard + event dedup.

Until this change none of the three EXTERNAL webhook lanes (Composio
``/webhook``, workspace ``/ws/{key}``, playbook ``/recipe/{id}``) rejected a
redelivered event: a provider retry re-ran the same agent/playbook, burning
tokens and re-firing side-effects. These tests pin the new contract:

- the SAME event id delivered twice executes once — the redelivery is a
  fast no-op ack (``status: duplicate``), nothing routed, nothing run;
- a provider timestamp outside the skew window is a replay ⇒ 401;
- Redis down ⇒ the guard fails OPEN for dedup (the lane keeps working,
  loudly) — locked decision: availability beats replay protection when the
  guard store is down;
- the Shopify ``/events`` path (PRD-189 S3's debounce) is NOT routed
  through this guard.

Pure: hand-built Starlette Requests, a dict-backed fake Redis patched at the
module boundary (``webhook_dedup._get_redis``), stub/poisoned DBs; no network,
no live Redis, no real Composio.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from starlette.requests import Request  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from api import composio as composio_api  # noqa: E402
from api import webhooks as webhooks_api  # noqa: E402
from api import workflow_recipes as recipes_api  # noqa: E402
from services import webhook_dedup  # noqa: E402
from config import config  # noqa: E402


# ---------------------------------------------------------------- helpers

def _make_request(headers: dict, body: bytes = b"{}") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/webhooks/test",
        "headers": [(k.lower().encode(), str(v).encode()) for k, v in headers.items()],
        "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _run(coro):
    return asyncio.run(coro)


class _FakeRedis:
    """Dict-backed stand-in honouring the exact SET NX EX contract we use."""

    def __init__(self):
        self.store: dict = {}
        self.last_ex = None

    def set(self, key, value, nx=False, ex=None):
        self.last_ex = ex
        if nx and key in self.store:
            return None  # redis-py returns None when NX blocks the write
        self.store[key] = value
        return True


class _ExplodingRedis:
    def set(self, *a, **kw):
        raise ConnectionError("redis down")


class _PoisonedDb:
    """A DB the handler must never reach on the guarded paths."""

    def query(self, *args, **kwargs):
        raise AssertionError("db must not be touched on this path")


class _StubQuery:
    def __init__(self, single=None, many=None):
        self._single = single
        self._many = many or []

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._single

    def all(self):
        return self._many


class _RoutingDb:
    def __init__(self, workspace=None, channel_rows=None):
        self._workspace = workspace
        self._channel_rows = channel_rows or []

    def query(self, model, *args, **kwargs):
        if model.__name__ == "ChannelConnection":
            return _StubQuery(many=self._channel_rows)
        return _StubQuery(single=self._workspace)


class _RecipeDb:
    def __init__(self, recipe):
        self._recipe = recipe

    def query(self, *args, **kwargs):
        return _StubQuery(single=self._recipe)

    def add(self, obj):
        raise AssertionError("no RecipeExecution may be created on a duplicate")


def _fake_workspace(settings=None):
    return SimpleNamespace(id=uuid4(), is_active=True, settings=settings or {})


@pytest.fixture()
def fake_redis(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(webhook_dedup, "_get_redis", lambda: fake)
    return fake


# ---------------------------------------------------------------- unit: guard

def test_seen_before_first_false_then_true(fake_redis):
    """SETNX semantics: first delivery marks and passes, redelivery is seen."""
    assert _run(webhook_dedup.seen_before("composio", "evt_1")) is False
    assert _run(webhook_dedup.seen_before("composio", "evt_1")) is True
    # marks self-expire with the configured TTL
    assert fake_redis.last_ex == config.WEBHOOK_DEDUP_TTL_SECONDS
    # a different lane is a different key — no cross-lane collision
    assert _run(webhook_dedup.seen_before("recipe:whX", "evt_1")) is False


def test_seen_before_without_event_id_is_noop(monkeypatch):
    """No event id ⇒ nothing to dedup on ⇒ process (and never touch Redis)."""
    monkeypatch.setattr(
        webhook_dedup, "_get_redis",
        lambda: (_ for _ in ()).throw(AssertionError("redis must not be touched")),
    )
    assert _run(webhook_dedup.seen_before("composio", None)) is False
    assert _run(webhook_dedup.seen_before("composio", "")) is False


def test_timestamp_skew_contract():
    now = int(time.time())
    assert webhook_dedup.timestamp_is_stale(str(now)) is False
    assert webhook_dedup.timestamp_is_stale(None) is False  # no header → no check
    assert webhook_dedup.timestamp_is_stale("") is False
    stale = now - (config.WEBHOOK_TIMESTAMP_SKEW_SECONDS + 60)
    assert webhook_dedup.timestamp_is_stale(str(stale)) is True
    future = now + (config.WEBHOOK_TIMESTAMP_SKEW_SECONDS + 60)
    assert webhook_dedup.timestamp_is_stale(str(future)) is True
    # present-but-garbage timestamp fails closed
    assert webhook_dedup.timestamp_is_stale("not-a-number") is True


def test_redis_down_fails_open(monkeypatch, caplog):
    """LOCKED DECISION: Redis unavailable ⇒ process the event, log loudly.

    Availability of the lane beats replay protection when the guard store is
    down — a cache outage must not brick webhook ingest.
    """
    monkeypatch.setattr(webhook_dedup, "_get_redis", lambda: None)
    with caplog.at_level(logging.ERROR):
        assert _run(webhook_dedup.seen_before("composio", "evt_9")) is False
        assert _run(webhook_dedup.seen_before("composio", "evt_9")) is False
    assert any("replay/dedup guard DOWN" in r.message for r in caplog.records)

    monkeypatch.setattr(webhook_dedup, "_get_redis", lambda: _ExplodingRedis())
    with caplog.at_level(logging.ERROR):
        assert _run(webhook_dedup.seen_before("composio", "evt_9")) is False


# ---------------------------------------------------------------- Composio lane

def test_composio_replay_is_noop(monkeypatch, fake_redis):
    """The same webhook-id delivered twice dispatches once; the redelivery is
    a fast no-op ack (was: two full dispatches)."""
    monkeypatch.setattr(composio_api.config, "COMPOSIO_WEBHOOK_SECRET", None)
    headers = {"webhook-id": "wh_dup_1", "webhook-timestamp": str(int(time.time()))}

    # First delivery passes the guard and gets processed (400: no trigger_name
    # in the body — proof it went past dedup into the handler proper).
    with pytest.raises(HTTPException) as ei:
        _run(composio_api.handle_webhook(
            request=_make_request(headers), x_composio_signature=None, db=_PoisonedDb()))
    assert ei.value.status_code == 400

    # Redelivery: fast no-op ack — no parse, no routing, no dispatch, no DB.
    result = _run(composio_api.handle_webhook(
        request=_make_request(headers), x_composio_signature=None, db=_PoisonedDb()))
    assert result == {"status": "duplicate", "webhook_id": "wh_dup_1"}


def test_stale_timestamp_rejected(monkeypatch, fake_redis):
    """A webhook-timestamp older than the skew window ⇒ 401 (replay defence)."""
    monkeypatch.setattr(composio_api.config, "COMPOSIO_WEBHOOK_SECRET", None)
    stale = str(int(time.time()) - config.WEBHOOK_TIMESTAMP_SKEW_SECONDS - 120)
    req = _make_request({"webhook-id": "wh_old", "webhook-timestamp": stale})
    with pytest.raises(HTTPException) as ei:
        _run(composio_api.handle_webhook(request=req, x_composio_signature=None, db=_PoisonedDb()))
    assert ei.value.status_code == 401
    # rejected replays must NOT consume a dedup slot
    assert not fake_redis.store


# ---------------------------------------------------------------- workspace lane

def test_workspace_webhook_dedup_on_event_id(monkeypatch, fake_redis):
    """A redelivered Telegram update_id does not re-run the agent."""
    monkeypatch.setattr(webhooks_api.config, "WEBHOOK_SECRET", None)
    ws = _fake_workspace()

    async def _must_not_execute(*a, **kw):
        raise AssertionError("agent must not execute on a duplicate")

    monkeypatch.setattr(webhooks_api, "_execute_agent_sync", _must_not_execute)

    # Simulate the first delivery having been accepted already.
    fake_redis.store[f"webhook:dedup:ws:{ws.id}:4242"] = "1"

    body = b'{"update_id": 4242, "message": {"chat": {"id": 7}, "text": "hi"}}'
    req = _make_request({"content-type": "application/json"}, body=body)
    result = _run(webhooks_api.general_workspace_webhook(
        workspace_key="k", request=req, db=_RoutingDb(ws)))
    assert result == {"status": "duplicate_ignored", "event_id": "4242"}


def test_workspace_webhook_stale_timestamp_rejected(monkeypatch, fake_redis):
    monkeypatch.setattr(webhooks_api.config, "WEBHOOK_SECRET", None)
    ws = _fake_workspace()
    stale = str(int(time.time()) - config.WEBHOOK_TIMESTAMP_SKEW_SECONDS - 120)
    req = _make_request(
        {"content-type": "application/json", "webhook-timestamp": stale},
        body=b'{"update_id": 1}',
    )
    with pytest.raises(HTTPException) as ei:
        _run(webhooks_api.general_workspace_webhook(
            workspace_key="k", request=req, db=_RoutingDb(ws)))
    assert ei.value.status_code == 401


def test_workspace_dedup_scoped_per_workspace(fake_redis, monkeypatch):
    """Telegram update_ids are only unique per bot — two workspaces must not
    share a dedup slot for the same numeric id."""
    ws_a, ws_b = uuid4(), uuid4()
    assert _run(webhook_dedup.seen_before(f"ws:{ws_a}", "4242")) is False
    assert _run(webhook_dedup.seen_before(f"ws:{ws_b}", "4242")) is False
    assert _run(webhook_dedup.seen_before(f"ws:{ws_a}", "4242")) is True


# ---------------------------------------------------------------- playbook lane

def _fake_recipe():
    return SimpleNamespace(
        schedule_config={"webhook_id": "wh1"}, steps=[], name="r", workspace_id=uuid4()
    )


def test_recipe_webhook_dedup_is_noop(monkeypatch, fake_redis):
    """A redelivered event id does not create a second RecipeExecution."""
    monkeypatch.setattr(recipes_api.config, "WEBHOOK_SECRET", None)
    headers = {"content-type": "application/json", "webhook-id": "evt_9"}

    # First delivery passes the guard (400: recipe has no steps — past dedup).
    with pytest.raises(HTTPException) as ei:
        _run(recipes_api.recipe_webhook(
            webhook_id="wh1", request=_make_request(headers), db=_RecipeDb(_fake_recipe())))
    assert ei.value.status_code == 400

    # Redelivery: no-op ack; _RecipeDb.add raises if an execution row is built.
    result = _run(recipes_api.recipe_webhook(
        webhook_id="wh1", request=_make_request(headers), db=_RecipeDb(_fake_recipe())))
    assert result == {"status": "duplicate", "event_id": "evt_9"}


def test_recipe_webhook_stale_timestamp_rejected(monkeypatch, fake_redis):
    monkeypatch.setattr(recipes_api.config, "WEBHOOK_SECRET", None)
    stale = str(int(time.time()) - config.WEBHOOK_TIMESTAMP_SKEW_SECONDS - 120)
    req = _make_request(
        {"content-type": "application/json", "webhook-timestamp": stale})
    with pytest.raises(HTTPException) as ei:
        _run(recipes_api.recipe_webhook(
            webhook_id="wh1", request=req, db=_RecipeDb(_fake_recipe())))
    assert ei.value.status_code == 401


# ---------------------------------------------------------------- composition

def test_dedup_does_not_touch_shopify_events():
    """PRD-189 S3 owns the Shopify /events debounce — this guard must not be
    wired into api/shopify.py (compose, do not collide)."""
    shopify_src = Path(composio_api.__file__).parent.joinpath("shopify.py").read_text()
    assert "webhook_dedup" not in shopify_src
