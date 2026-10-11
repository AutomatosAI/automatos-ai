"""PRD-256 O2b (#847): the worker's HTTP routes moved out of ``_health_server``.

A move with no behaviour change: the same routes with the same methods (GET
also answers HEAD, as ``add_get`` did), the same middleware order (metrics,
then the internal-token check), and the same answers. The handlers' own
behaviour is covered by the F173/F178/F333 and binary-pieces suites, which run
the worker's server unchanged.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from helpers_workspace_worker import load_worker_main, worker_server  # noqa: E402

WS = "00000000-0000-0000-0000-0000000000c1"


def _worker():
    """What the app reads from the worker: its id, tasks and concurrency (health) and Redis (canvas)."""
    return SimpleNamespace(_worker_id="w-test", _active_tasks={}, concurrency=1, _redis=None)


ROUTES_BEFORE_THE_MOVE = {
    ("GET", "/health"), ("HEAD", "/health"),
    ("GET", "/workspaces/{workspace_id}/files"), ("HEAD", "/workspaces/{workspace_id}/files"),
    ("GET", "/workspaces/{workspace_id}/files/content"), ("HEAD", "/workspaces/{workspace_id}/files/content"),
    ("POST", "/workspaces/{workspace_id}/exec"),
    ("POST", "/workspaces/{workspace_id}/files/write"),
    ("GET", "/workspaces/{workspace_id}/files/grep"), ("HEAD", "/workspaces/{workspace_id}/files/grep"),
    ("GET", "/workspaces/{workspace_id}/files/download"), ("HEAD", "/workspaces/{workspace_id}/files/download"),
    ("POST", "/workspaces/{workspace_id}/git"),
    ("POST", "/workspaces/{workspace_id}/html-to-png"),
    ("POST", "/workspaces/{workspace_id}/canvas/session"),
    ("GET", "/workspaces/{workspace_id}/canvas/session"), ("HEAD", "/workspaces/{workspace_id}/canvas/session"),
    ("DELETE", "/workspaces/{workspace_id}/canvas/session"),
    ("POST", "/workspaces/{workspace_id}/canvas/session/decision"),
    ("POST", "/workspaces/{workspace_id}/canvas/session/auto-accept"),
    ("POST", "/workspaces/{workspace_id}/canvas/session/message"),
}


def test_the_app_serves_the_same_routes_as_before_the_move(monkeypatch, tmp_path):
    load_worker_main(monkeypatch, tmp_path)
    import worker_http

    app = worker_http.build_app(_worker())
    routes = {(r.method, r.resource.canonical) for r in app.router.routes()}
    assert routes == ROUTES_BEFORE_THE_MOVE


def test_the_internal_token_is_still_enforced_and_health_still_public(monkeypatch, tmp_path):
    async def run():
        async with worker_server(monkeypatch, tmp_path) as (base, http):
            async with http.get(f"{base}/health") as health:
                assert health.status == 200 and (await health.json())["status"] == "healthy"
            async with http.post(f"{base}/workspaces/{WS}/exec", data="not json") as bad:
                assert bad.status == 400 and (await bad.json()) == {"error": "Invalid JSON body"}

    asyncio.run(run())


def test_a_configured_internal_token_refuses_a_request_without_it(monkeypatch, tmp_path):
    load_worker_main(monkeypatch, tmp_path)
    import worker_http

    monkeypatch.setenv("WORKER_INTERNAL_TOKEN", "s3cret")
    app = worker_http.build_app(_worker())

    async def run():
        from aiohttp.test_utils import TestClient, TestServer

        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/health")).status == 200
            assert (await client.get(f"/workspaces/{WS}/files")).status == 401
            ok = await client.get(f"/workspaces/{WS}/files", headers={"X-Internal-Token": "s3cret"})
            assert ok.status == 200

    asyncio.run(run())
