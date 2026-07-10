"""PRD-189 S3 (F2) — /events debounce + coalesce + already-running guard.

The old handler fired ``create_task(_sync_catalog_for_workspace(...))`` per
catalog event: a merchant bulk edit emitting N webhooks launched N concurrent
full Bulk-Op re-syncs (each an embedding-bearing full rebuild) with the task
reference dropped — GC-collectable mid-flight. These tests pin the fixed
contract, mirroring the in-process debounce shape ``GraphifyService`` ships:

* a webhook burst inside the window produces exactly ONE re-sync, with the
  coalesced event count in its reason;
* while a re-sync is in flight a further event never launches a second
  concurrent full sync — the window re-arms and the follow-up runs after;
* the debounce window is read from ``config.SHOPIFY_SYNC_DEBOUNCE_SECONDS``
  at call time (never an inline ``os.getenv``);
* the fired task's reference is HELD for the duration of the flight.

Pure: the re-sync itself is mocked at the ``_sync_catalog_for_workspace``
seam (the ``test_prd183_s1_catalog_webhook.py`` posture); no DB, no network.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402

import pytest  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from api import shopify  # noqa: E402


class _FakeQuery:
    def __init__(self, workspace):
        self._workspace = workspace

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._workspace


class _FakeWorkspace:
    def __init__(self, wid="ws-123", shop="inbuilduk.myshopify.com"):
        self.id = wid
        self.is_active = True
        self.settings = {"shopify_domain": shop}


class _FakeDb:
    def __init__(self, workspace):
        self._workspace = workspace

    def query(self, *args, **kwargs):
        return _FakeQuery(self._workspace)


@pytest.fixture(autouse=True)
def _fresh_debounce_state():
    """Per-test isolation for the module-level debounce state."""
    shopify._catalog_debounce_handles.clear()
    shopify._catalog_pending_events.clear()
    shopify._catalog_sync_tasks.clear()
    yield
    for handle in shopify._catalog_debounce_handles.values():
        handle.cancel()
    shopify._catalog_debounce_handles.clear()
    shopify._catalog_pending_events.clear()
    shopify._catalog_sync_tasks.clear()


def _run(coro):
    """Run a scenario coroutine on a fresh loop and drain leftover tasks."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        return result
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _event(event: str = "products/update", shop: str = "inbuilduk.myshopify.com"):
    return shopify.EventRequest(shop=shop, event=event)


def test_webhook_burst_coalesces_to_one_resync(monkeypatch):
    """N catalog events inside the debounce window → exactly ONE
    ``_product_sync_impl`` run (via the ``_sync_catalog_for_workspace`` seam),
    not N concurrent ones — with the coalesced burst named in the reason."""
    ws = _FakeWorkspace()
    db = _FakeDb(ws)
    monkeypatch.setattr(shopify.config, "SHOPIFY_SYNC_DEBOUNCE_SECONDS", 0.02)

    calls = []

    async def _fake_sync(workspace_id, event):
        calls.append({"workspace_id": workspace_id, "event": event})

    monkeypatch.setattr(shopify, "_sync_catalog_for_workspace", _fake_sync)

    async def _scenario():
        for event in (
            "products/update",
            "inventory_levels/update",
            "products/update",
            "collections/update",
            "products/update",
        ):
            result = await shopify.forward_event(
                request=_event(event), db=db, _auth=None
            )
            assert result["status"] == "received"
        await asyncio.sleep(0.1)  # let the window expire and the one task run

    _run(_scenario())

    assert len(calls) == 1, "a webhook burst must coalesce to one re-sync"
    assert calls[0]["workspace_id"] == "ws-123"
    assert calls[0]["event"] == "products/update (+4 coalesced)"


def test_webhook_already_running_is_skipped(monkeypatch):
    """While a re-sync is in flight, a further event must not launch a second
    concurrent full sync; the coalesced follow-up runs once the flight ends
    (mid-flight changes are never silently dropped)."""
    ws = _FakeWorkspace()
    db = _FakeDb(ws)
    monkeypatch.setattr(shopify.config, "SHOPIFY_SYNC_DEBOUNCE_SECONDS", 0.02)

    started = []
    release = asyncio.Event()  # loop-agnostic until first wait (py3.10+)

    async def _slow_sync(workspace_id, event):
        started.append(event)
        await release.wait()

    monkeypatch.setattr(shopify, "_sync_catalog_for_workspace", _slow_sync)

    async def _scenario():
        await shopify.forward_event(request=_event(), db=db, _auth=None)
        await asyncio.sleep(0.06)  # window expires → sync starts and blocks
        assert started == ["products/update"]
        # The task reference is HELD while in flight (the GC guard the old
        # fire-and-forget create_task never had).
        held = shopify._catalog_sync_tasks.get("ws-123")
        assert held is not None and not held.done()

        # A further event while in flight: no second concurrent launch.
        await shopify.forward_event(
            request=_event("inventory_levels/update"), db=db, _auth=None
        )
        await asyncio.sleep(0.06)  # its window expires against the guard
        assert len(started) == 1, "no concurrent second sync while one runs"

        release.set()  # in-flight sync completes
        await asyncio.sleep(0.08)  # re-armed window expires → follow-up runs
        assert len(started) == 2, "mid-flight events get their follow-up sync"
        assert started[1] == "inventory_levels/update"
        assert shopify._catalog_sync_tasks.get("ws-123") is None

    _run(_scenario())


def test_webhook_debounce_window_from_config(monkeypatch):
    """The window is ``config.SHOPIFY_SYNC_DEBOUNCE_SECONDS`` read at call
    time — a long window defers the re-sync, a short one fires it."""
    ws = _FakeWorkspace()
    db = _FakeDb(ws)

    fired = {"count": 0}

    async def _fake_sync(workspace_id, event):
        fired["count"] += 1

    monkeypatch.setattr(shopify, "_sync_catalog_for_workspace", _fake_sync)

    # Long window: nothing fires within the settle time.
    monkeypatch.setattr(shopify.config, "SHOPIFY_SYNC_DEBOUNCE_SECONDS", 30.0)

    async def _long_window():
        await shopify.forward_event(request=_event(), db=db, _auth=None)
        await asyncio.sleep(0.1)

    _run(_long_window())
    assert fired["count"] == 0, "the re-sync must wait out the configured window"

    # Short window: the same event fires within the settle time.
    shopify._catalog_debounce_handles.clear()
    shopify._catalog_pending_events.clear()
    monkeypatch.setattr(shopify.config, "SHOPIFY_SYNC_DEBOUNCE_SECONDS", 0.02)

    async def _short_window():
        await shopify.forward_event(request=_event(), db=db, _auth=None)
        await asyncio.sleep(0.1)

    _run(_short_window())
    assert fired["count"] == 1


def test_workspaces_debounce_independently(monkeypatch):
    """The debounce is per workspace — two shops bursting together produce one
    re-sync EACH, not one global."""
    ws_a = _FakeWorkspace(wid="ws-a", shop="a.myshopify.com")
    ws_b = _FakeWorkspace(wid="ws-b", shop="b.myshopify.com")
    monkeypatch.setattr(shopify.config, "SHOPIFY_SYNC_DEBOUNCE_SECONDS", 0.02)

    calls = []

    async def _fake_sync(workspace_id, event):
        calls.append(workspace_id)

    monkeypatch.setattr(shopify, "_sync_catalog_for_workspace", _fake_sync)

    async def _scenario():
        for _ in range(3):
            await shopify.forward_event(
                request=_event(shop="a.myshopify.com"), db=_FakeDb(ws_a), _auth=None
            )
            await shopify.forward_event(
                request=_event(shop="b.myshopify.com"), db=_FakeDb(ws_b), _auth=None
            )
        await asyncio.sleep(0.1)

    _run(_scenario())

    assert sorted(calls) == ["ws-a", "ws-b"]
