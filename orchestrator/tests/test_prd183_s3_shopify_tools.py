"""PRD-183 S3 (F088) — Shopify sync + freshness as platform tools.

Before this, the only catalog refresh was ``POST /api/shopify/sync/products/start``
— a bare HTTP route, not a tool. So Auto could neither run the sync nor even
check when the graph last synced. These handlers close that parity gap through
the canonical 3-file platform-tool registration pattern:

  * ``platform_shopify_sync_catalog``  — run a catalog → graph re-sync and
    report what changed (node/edge/community counts).
  * ``platform_shopify_sync_status``   — read freshness: last sync status,
    timestamp, and counts (never_synced when it has never run).

Both are workspace-scoped: ``workspace_id`` comes from the executor context
(RequestContext), never the params — an agent cannot sync another tenant.

Tests:
  * the two actions are registered + wired into the executor handler map;
  * ``test_platform_sync_tool`` — sync invokes the impl for the executor's
    workspace and surfaces the change counts;
  * status reflects stored ``product_sync`` state, incl. never_synced.

Pure: ``_product_sync_impl`` and the Workspace query are patched at the
boundary; no Composio, no network, no DB.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from modules.tools.discovery import handlers_shopify  # noqa: E402
from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.actions_shopify import register_shopify_actions  # noqa: E402


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ------------------------------------------------------------------
# Registration + executor wiring
# ------------------------------------------------------------------


def test_shopify_actions_registered():
    reg = ActionRegistry()
    register_shopify_actions(reg)
    names = {a.name for a in reg.get_all()}
    assert "platform_shopify_sync_catalog" in names
    assert "platform_shopify_sync_status" in names


def test_shopify_actions_are_promoted_and_scoped():
    reg = ActionRegistry()
    register_shopify_actions(reg)
    sync = reg.get("platform_shopify_sync_catalog")
    status = reg.get("platform_shopify_sync_status")
    # Promoted so every agent gets them (parity with the manual path).
    assert sync.promoted and status.promoted
    # Sync mutates the graph → write; status is a read.
    assert sync.permission_level == "write"
    assert status.permission_level == "read"


def test_executor_wires_shopify_handlers():
    """The executor's handler map binds both tool names to their handlers."""
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    # __init__ only builds the handler dict (no DB I/O), so a throwaway
    # instance is enough to inspect the wiring.
    execu = PlatformActionExecutor(db=None, workspace_id="ws-9")
    assert execu._handlers["platform_shopify_sync_catalog"] is handlers_shopify.shopify_sync_catalog
    assert execu._handlers["platform_shopify_sync_status"] is handlers_shopify.shopify_sync_status


# ------------------------------------------------------------------
# Behaviour
# ------------------------------------------------------------------


def test_platform_sync_tool(monkeypatch):
    """Sync runs the impl for the executor's workspace and reports change counts."""
    seen = {}

    class _Resp:
        def model_dump(self):
            return {
                "status": "complete",
                "node_count": 42,
                "edge_count": 88,
                "community_count": 5,
                "duration_seconds": 3.1,
            }

    async def _fake_impl(workspace_id, db):
        seen["workspace_id"] = workspace_id
        return _Resp()

    monkeypatch.setattr(handlers_shopify, "_product_sync_impl", _fake_impl)

    res = _run(handlers_shopify.shopify_sync_catalog(db=object(), workspace_id="ws-9", params={}))

    assert res["success"] is True
    assert seen["workspace_id"] == "ws-9"           # scoped to executor workspace
    assert res["node_count"] == 42
    assert res["edge_count"] == 88
    assert res["community_count"] == 5


def test_sync_status_reads_stored_state(monkeypatch):
    """Freshness tool returns the stored product_sync block."""
    stored = {
        "status": "complete",
        "node_count": 10,
        "completed_at": 1_700_000_000.0,
    }

    class _WS:
        settings = {"product_sync": stored}

    class _Q:
        def filter(self, *a, **k):
            return self

        def first(self):
            return _WS()

    class _Db:
        def query(self, *a, **k):
            return _Q()

    res = _run(handlers_shopify.shopify_sync_status(db=_Db(), workspace_id="ws-9", params={}))
    assert res["success"] is True
    assert res["status"] == "complete"
    assert res["node_count"] == 10


def test_sync_status_never_synced(monkeypatch):
    """A workspace that never synced reports never_synced, not an error."""

    class _WS:
        settings = {}

    class _Q:
        def filter(self, *a, **k):
            return self

        def first(self):
            return _WS()

    class _Db:
        def query(self, *a, **k):
            return _Q()

    res = _run(handlers_shopify.shopify_sync_status(db=_Db(), workspace_id="ws-9", params={}))
    assert res["success"] is True
    assert res["status"] == "never_synced"
