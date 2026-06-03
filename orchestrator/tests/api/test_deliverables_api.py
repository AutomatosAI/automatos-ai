"""Integration tests for /api/deliverables (PRD-129).

These tests mount a minimal FastAPI app containing only the deliverables
router, override the two external dependencies (``get_db`` and
``get_request_context_hybrid``), and patch ``DeliverableService`` so the
routes are exercised end-to-end without a running Postgres.

Coverage:
- GET /api/deliverables (list + filter forwarding)
- GET /api/deliverables/stats
- GET /api/deliverables/{id} (found, not found, include_content)
- DELETE /api/deliverables/{id} (success + 404)
- Workspace isolation: the service is always instantiated with the
  ctx.workspace_id, never a client-supplied header value.
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Stub out transitive imports that need jwt/clerk before importing the router
# (mirrors orchestrator/tests/test_agents_api_plugins.py).
# ---------------------------------------------------------------------------
def _fake_dep():  # pragma: no cover — replaced via dependency_overrides
    raise RuntimeError("dependency not overridden")


from fastapi import FastAPI
from fastapi.testclient import TestClient


def _import_router_isolated():
    """Import the deliverables router under stubbed transitive deps, then
    restore sys.modules.

    The router binds its dependencies (``get_db``, ``get_request_context_hybrid``,
    ``DeliverableService``) at its own import time; the tests then swap them via
    FastAPI ``dependency_overrides`` / ``patch``. So the stubs only need to exist
    during this import — restoring sys.modules afterwards stops them leaking into
    sibling modules' collection. (PRD-142 W2-S2b.)
    """
    _keys = (
        "jwt", "jwt.algorithms", "jwt.exceptions", "core.auth.clerk",
        "core.auth.hybrid", "core.auth.dependencies",
        "core.database", "core.database.database",
        "services", "services.deliverable_service",
    )
    _saved = {k: sys.modules.get(k) for k in _keys}
    try:
        for mod_name in [
            "jwt", "jwt.algorithms", "jwt.exceptions",
            "core.auth.clerk",
        ]:
            if mod_name not in sys.modules:
                stub = ModuleType(mod_name)
                stub.get_clerk_auth = MagicMock()
                stub.decode = MagicMock()
                stub.DecodeError = Exception
                stub.ExpiredSignatureError = Exception
                sys.modules[mod_name] = stub

        # core.auth.hybrid must be a real (stub) module exposing the dependency name
        if "core.auth.hybrid" not in sys.modules or not hasattr(
            sys.modules["core.auth.hybrid"], "get_request_context_hybrid"
        ):
            hybrid_stub = ModuleType("core.auth.hybrid")
            hybrid_stub.get_request_context_hybrid = _fake_dep
            sys.modules["core.auth.hybrid"] = hybrid_stub

        # core.auth.dependencies exposes RequestContext — stub as MagicMock class alias.
        if "core.auth.dependencies" not in sys.modules:
            deps_stub = ModuleType("core.auth.dependencies")
            deps_stub.RequestContext = MagicMock
            sys.modules["core.auth.dependencies"] = deps_stub

        # core.database.database imports fail without cryptography / DB env. Stub it.
        if "core.database" not in sys.modules:
            sys.modules["core.database"] = ModuleType("core.database")
        if "core.database.database" not in sys.modules or not hasattr(
            sys.modules["core.database.database"], "get_db"
        ):
            db_stub = ModuleType("core.database.database")
            db_stub.get_db = _fake_dep
            sys.modules["core.database.database"] = db_stub

        # services.deliverable_service imports core.workspace_client which pulls config.
        # Patch DeliverableService as a MagicMock so the router import succeeds.
        if "services" not in sys.modules:
            sys.modules["services"] = ModuleType("services")
        svc_stub = ModuleType("services.deliverable_service")
        svc_stub.DeliverableService = MagicMock()
        sys.modules["services.deliverable_service"] = svc_stub

        from api.deliverables import router as deliverables_router
        from core.auth.hybrid import get_request_context_hybrid
        from core.database.database import get_db
        return deliverables_router, get_request_context_hybrid, get_db
    finally:
        for _k, _v in _saved.items():
            if _v is None:
                sys.modules.pop(_k, None)
            else:
                sys.modules[_k] = _v


deliverables_router, get_request_context_hybrid, get_db = _import_router_isolated()


WORKSPACE_ID = uuid4()
OTHER_WORKSPACE_ID = uuid4()
DELIVERABLE_ID = str(uuid4())


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------

def _make_ctx(workspace_id=WORKSPACE_ID):
    ctx = MagicMock()
    ctx.workspace_id = workspace_id
    ctx.user = MagicMock()
    ctx.user.id = "user_123"
    ctx.auth_type = "clerk"
    return ctx


@pytest.fixture
def ctx():
    return _make_ctx()


@pytest.fixture
def app(ctx):
    app = FastAPI()
    app.include_router(deliverables_router)
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

def _sample_deliverable():
    return {
        "id": DELIVERABLE_ID,
        "workspace_id": str(WORKSPACE_ID),
        "source_type": "heartbeat",
        "source_id": "hb-1",
        "agent_id": 42,
        "agent_name": "Scout",
        "artifact_type": "report",
        "title": "Weekly Report",
        "summary": "Summary",
        "storage_type": "workspace",
        "file_path": "reports/scout/weekly.md",
        "file_name": "weekly.md",
        "file_type": "md",
        "file_size_bytes": 1024,
        "preview_url": None,
        "preview_type": None,
        "extra": {},
        "status": "ready",
        "created_at": "2026-04-10T00:00:00+00:00",
        "updated_at": "2026-04-10T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# LIST
# ---------------------------------------------------------------------------

class TestList:
    def test_list_default(self, client):
        with patch("api.deliverables.DeliverableService") as Svc:
            svc = Svc.return_value
            svc.list_deliverables.return_value = {
                "success": True,
                "deliverables": [_sample_deliverable()],
                "total": 1,
                "limit": 24,
                "offset": 0,
            }

            resp = client.get("/api/deliverables")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total"] == 1
        assert body["deliverables"][0]["id"] == DELIVERABLE_ID

        svc.list_deliverables.assert_called_once()
        kwargs = svc.list_deliverables.call_args.kwargs
        assert kwargs["limit"] == 24
        assert kwargs["offset"] == 0
        assert kwargs["artifact_type"] is None

    def test_list_forwards_filters(self, client):
        with patch("api.deliverables.DeliverableService") as Svc:
            svc = Svc.return_value
            svc.list_deliverables.return_value = {
                "success": True, "deliverables": [], "total": 0,
                "limit": 10, "offset": 5,
            }

            resp = client.get(
                "/api/deliverables",
                params={
                    "artifact_type": "image",
                    "source_type": "chat",
                    "agent_id": 7,
                    "date_from": "2026-01-01T00:00:00Z",
                    "date_to": "2026-04-10T00:00:00Z",
                    "search": "hello",
                    "limit": 10,
                    "offset": 5,
                },
            )

        assert resp.status_code == 200
        kwargs = svc.list_deliverables.call_args.kwargs
        assert kwargs["artifact_type"] == "image"
        assert kwargs["source_type"] == "chat"
        assert kwargs["agent_id"] == 7
        assert kwargs["date_from"] == "2026-01-01T00:00:00Z"
        assert kwargs["date_to"] == "2026-04-10T00:00:00Z"
        assert kwargs["search"] == "hello"
        assert kwargs["limit"] == 10
        assert kwargs["offset"] == 5

    def test_list_rejects_limit_over_100(self, client):
        resp = client.get("/api/deliverables", params={"limit": 500})
        assert resp.status_code == 422

    def test_list_scopes_to_ctx_workspace(self, client, ctx):
        """Service must be constructed with ctx.workspace_id — never a header."""
        with patch("api.deliverables.DeliverableService") as Svc:
            Svc.return_value.list_deliverables.return_value = {
                "success": True, "deliverables": [], "total": 0,
                "limit": 24, "offset": 0,
            }

            client.get(
                "/api/deliverables",
                headers={"X-Workspace-ID": str(OTHER_WORKSPACE_ID)},
            )

        # Second positional arg is the workspace_id
        args, _ = Svc.call_args
        assert args[1] == ctx.workspace_id
        assert args[1] != OTHER_WORKSPACE_ID


# ---------------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats(self, client):
        with patch("api.deliverables.DeliverableService") as Svc:
            Svc.return_value.get_stats.return_value = {
                "success": True,
                "total": 12,
                "by_type": {"report": 5, "image": 7},
                "by_agent": [{"agent_id": 42, "agent_name": "Scout", "count": 12}],
            }

            resp = client.get("/api/deliverables/stats")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 12
        assert body["by_type"]["report"] == 5

    def test_stats_path_not_shadowed_by_id(self, client):
        """Regression: /stats must hit stats handler, not get_deliverable('stats')."""
        with patch("api.deliverables.DeliverableService") as Svc:
            svc = Svc.return_value
            svc.get_stats.return_value = {
                "success": True, "total": 0, "by_type": {}, "by_agent": [],
            }
            svc.get_deliverable = AsyncMock()

            resp = client.get("/api/deliverables/stats")

        assert resp.status_code == 200
        svc.get_stats.assert_called_once()
        svc.get_deliverable.assert_not_called()


# ---------------------------------------------------------------------------
# GET ONE
# ---------------------------------------------------------------------------

class TestGetOne:
    def test_get_found(self, client):
        with patch("api.deliverables.DeliverableService") as Svc:
            Svc.return_value.get_deliverable = AsyncMock(return_value={
                "success": True,
                "deliverable": _sample_deliverable(),
            })

            resp = client.get(f"/api/deliverables/{DELIVERABLE_ID}")

        assert resp.status_code == 200
        assert resp.json()["deliverable"]["id"] == DELIVERABLE_ID

    def test_get_not_found_returns_404(self, client):
        with patch("api.deliverables.DeliverableService") as Svc:
            Svc.return_value.get_deliverable = AsyncMock(return_value={
                "success": False, "error": "Deliverable not found",
            })

            resp = client.get(f"/api/deliverables/{DELIVERABLE_ID}")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Deliverable not found"

    def test_get_include_content_flag(self, client):
        with patch("api.deliverables.DeliverableService") as Svc:
            mock_svc = Svc.return_value
            mock_svc.get_deliverable = AsyncMock(return_value={
                "success": True,
                "deliverable": {**_sample_deliverable(), "content": "# Hello"},
            })

            resp = client.get(
                f"/api/deliverables/{DELIVERABLE_ID}",
                params={"include_content": "true"},
            )

        assert resp.status_code == 200
        mock_svc.get_deliverable.assert_awaited_once_with(
            DELIVERABLE_ID, include_content=True,
        )
        assert resp.json()["deliverable"]["content"] == "# Hello"

    def test_get_scopes_to_ctx_workspace(self, client, ctx):
        with patch("api.deliverables.DeliverableService") as Svc:
            Svc.return_value.get_deliverable = AsyncMock(return_value={
                "success": True, "deliverable": _sample_deliverable(),
            })

            client.get(
                f"/api/deliverables/{DELIVERABLE_ID}",
                headers={"X-Workspace-ID": str(OTHER_WORKSPACE_ID)},
            )

        args, _ = Svc.call_args
        assert args[1] == ctx.workspace_id


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_success(self, client):
        with patch("api.deliverables.DeliverableService") as Svc:
            Svc.return_value.soft_delete.return_value = {
                "success": True, "deliverable_id": DELIVERABLE_ID,
            }

            resp = client.delete(f"/api/deliverables/{DELIVERABLE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["deliverable_id"] == DELIVERABLE_ID

    def test_delete_not_found(self, client):
        with patch("api.deliverables.DeliverableService") as Svc:
            Svc.return_value.soft_delete.return_value = {
                "success": False, "error": "Deliverable not found",
            }

            resp = client.delete(f"/api/deliverables/{DELIVERABLE_ID}")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Deliverable not found"
