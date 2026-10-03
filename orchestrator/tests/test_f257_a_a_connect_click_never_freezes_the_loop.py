"""F257 — a Connect click never stands the event loop still.

3 Oct 2026, local build from main: ``POST /api/composio/connect/{app}`` was an
``async def`` making synchronous Composio SDK calls, and the loop watchdog caught
every click: 2.8 s (FAL_AI, 12:33:20Z), 3.0 s (KIEAI), 2.3 s (Higgsfield). The
connect-flow routes are plain ``def`` now, so FastAPI runs them in its threadpool:
both connects, the callback, disconnect, and the three listings that settle a
pending connection against Composio (F258).

F196 is the trap on that hop: ``loop.run_in_executor`` drops context variables, and
with them the req/ws log context and the tenant. FastAPI's threadpool is anyio's,
which runs the route in a copy of the caller's context; the second test drives that
real hop with the platform's own setters. The last test pins the one thing a
listing still needs the loop for: the Shopify catalog sync is a task, created on the
loop's own thread.
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import ast  # noqa: E402
import asyncio  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import anyio  # noqa: E402
import pytest  # noqa: E402
from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

ORCHESTRATOR = Path(__file__).resolve().parents[1]
CONNECT_FLOW_ROUTES = {
    "api/composio.py": ("initiate_connection", "connection_callback", "list_connections", "disconnect_app"),
    "api/tools.py": ("connect_app", "connected", "refresh_connections"),
}


def _functions(relative: str):
    tree = ast.parse((ORCHESTRATOR / relative).read_text(encoding="utf-8"))
    return {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


@pytest.mark.parametrize(("relative", "name"), [(f, n) for f, names in CONNECT_FLOW_ROUTES.items() for n in names])
def test_each_connect_flow_route_is_a_plain_def(relative, name):
    route = _functions(relative)[name]
    assert isinstance(route, ast.FunctionDef), f"{name} makes synchronous Composio calls: a plain def, never async def"
    assert not any(isinstance(node, ast.Await) for node in ast.walk(route))


def test_the_threadpool_hop_keeps_the_request_and_tenant_context():
    from core.auth.hybrid import _enrich_log_context
    from core.utils.logging_adapter import request_id_var, set_request_id, tenant_id_var, workspace_id_var

    app = FastAPI()
    threads = {}

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        set_request_id("req-f257")
        return await call_next(request)

    async def _request_context():  # resolved on the loop, as get_request_context_hybrid is
        threads["loop"] = threading.get_ident()
        _enrich_log_context(SimpleNamespace(workspace_id="ws-c1", user=SimpleNamespace(id=7)))

    @app.get("/connect", dependencies=[Depends(_request_context)])
    def _route():
        threads["route"] = threading.get_ident()
        return {"req": request_id_var.get(), "ws": workspace_id_var.get(), "tenant": tenant_id_var.get()}

    with TestClient(app) as client:
        body = client.get("/connect").json()

    assert body == {"req": "req-f257", "ws": "ws-c1", "tenant": "ws-c1"}
    assert threads["route"] != threads["loop"], "the route ran in the threadpool, off the event loop"


def test_the_shopify_sync_is_started_on_the_loop_from_a_threadpool_route(monkeypatch):
    import tests.conftest as _conftest

    _conftest._restore_real_app_modules()
    from api import tools as tools_api

    fired = {}

    def _fire(workspace_id):
        fired["thread"] = threading.get_ident()
        fired["loop_running"] = asyncio.get_running_loop() is not None  # raises off the loop's thread
        fired["workspace_id"] = workspace_id

    monkeypatch.setattr(tools_api, "_fire_shopify_autosync", _fire)

    async def _listing():
        await anyio.to_thread.run_sync(tools_api._start_shopify_autosync, "ws-shop")
        return threading.get_ident()

    loop_thread = anyio.run(_listing)
    assert fired == {"thread": loop_thread, "loop_running": True, "workspace_id": "ws-shop"}
