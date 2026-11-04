"""F105 (4 Nov): the Usage page and the Command Center feed never freeze the event loop.

A self-hosted install's loop watchdog named them: ``GET /api/analytics/usage`` held the
loop for 30.0 s and 29.6 s at ``.all()``, and ``GET /api/activity/feed`` for 30.4 s in
``ActivityService._fetch_chats``. Both were ``async def`` routes over a synchronous
session, so a query waiting for a pooled connection (pool_timeout 30 s) stopped every
other request in the process, the phone's included.

Pinned statically (AGENTS.md: a route that touches the database without awaiting
anything is a plain ``def``): every route in the two routers that awaits nothing is a
plain ``def``, and the Read (digest) route runs its coroutine on a loop of its own.
The OpenRouter routes still await their HTTP call, so their key lookup, and the
activity sync's upsert, are checked to run on a worker thread, not the loop's.
"""
from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ORCH = Path(__file__).resolve().parents[1]
ROUTERS = (ORCH / "api" / "llm_analytics.py", ORCH / "api" / "activity.py")
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
# The super-admin cross-workspace dashboards (admin_router) are still async def over
# the session: 133 and 163 lines each, so making them plain defs means splitting them
# first (the changed-code shape rule). Not on a workspace's own pages.
STILL_ON_THE_LOOP = {"get_admin_cost_analytics", "get_admin_dashboard"}


def _routes(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(isinstance(d, ast.Call) and getattr(d.func, "attr", "") in HTTP_METHODS for d in node.decorator_list):
            yield node


def _awaits(node: ast.AST) -> bool:
    return any(isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith)) for n in ast.walk(node))


@pytest.mark.parametrize("path", ROUTERS, ids=lambda p: p.name)
def test_every_route_that_awaits_nothing_is_a_plain_def(path):
    offenders = {r.name for r in _routes(path) if isinstance(r, ast.AsyncFunctionDef) and not _awaits(r)}
    assert offenders - STILL_ON_THE_LOOP == set(), f"async def over a synchronous session runs on the event loop: {offenders}"


def test_the_usage_and_feed_routes_are_plain_defs():
    names = {r.name: r for path in ROUTERS for r in _routes(path)}
    for name in ("get_usage", "get_summary", "get_costs", "get_cost_projections",
                 "get_activity_feed", "get_board_stats", "get_workspace_digest"):
        assert isinstance(names[name], ast.FunctionDef), f"{name} must be a plain def"


def test_a_usage_row_keeps_its_key_and_facts_per_grouping():
    from api.llm_analytics import _usage_key_and_facts

    agents = {"7": {"label": "Writer"}}
    assert _usage_key_and_facts(("m-1",), "model", agents) == ("m-1", {"model_id": "m-1", "label": "m-1"})
    assert _usage_key_and_facts((None,), "model", agents)[0] == "unknown"
    assert _usage_key_and_facts((7,), "agent", agents) == ("7", {"label": "Writer"})
    assert _usage_key_and_facts((None,), "tier", agents) == ("unknown", {})
    key, facts = _usage_key_and_facts(("openrouter",), "provider", agents)
    assert facts["label"] == facts["provider_label"] and key == facts["provider"]


def _thread_recorder(result):
    seen = []

    def record(*args, **kwargs):
        seen.append(threading.get_ident())
        return result
    return seen, record


def test_the_openrouter_key_is_looked_up_off_the_loop(monkeypatch):
    from api import llm_analytics
    from core.llm import openrouter_analytics

    seen, resolve = _thread_recorder("sk-or-test")
    monkeypatch.setattr(llm_analytics, "_resolve_openrouter_key", resolve)

    class _Service:
        async def get_credits(self, api_key):
            assert api_key == "sk-or-test"
            return {"total_credits": 10.0, "total_usage": 2.5}

    monkeypatch.setattr(openrouter_analytics, "OpenRouterAnalyticsService", _Service)

    async def call():
        loop_thread = threading.get_ident()
        out = await llm_analytics.get_openrouter_credits(ctx=SimpleNamespace(workspace_id="ws-1"), db=object())
        return loop_thread, out

    loop_thread, out = asyncio.run(call())
    assert out.total_credits == 10.0
    assert seen and seen[0] != loop_thread, "the key lookup reads the database: it must not run on the loop"


def test_the_activity_sync_upserts_off_the_loop(monkeypatch):
    from core.llm import openrouter_analytics

    seen, upsert = _thread_recorder({"synced": 1, "skipped": 0})
    svc = openrouter_analytics.OpenRouterAnalyticsService()
    monkeypatch.setattr(svc, "_upsert_activity_rows", upsert)

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"model": "m", "date": "2026-11-04"}]}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(openrouter_analytics.httpx, "AsyncClient", _Client)

    async def call():
        return threading.get_ident(), await svc.sync_activity("sk-or-test", "ws-1")

    loop_thread, out = asyncio.run(call())
    assert out == {"synced": 1, "skipped": 0}
    assert seen and seen[0] != loop_thread, "the upsert is synchronous SQLAlchemy: it must not run on the loop"
