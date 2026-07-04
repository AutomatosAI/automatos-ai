"""PRD-143 S6 — obs routers batch 1 locked to the super admin.

Router-wide ``require_super_admin`` (core/auth/super_admin.py, S5) on:
api/heartbeat.py, api/analytics.py, api/analytics_api.py,
api/analytics_real.py, api/analytics_charts.py.

Every endpoint on these routers must 403 for any principal that is not
literally ``system_role == 'super_admin'`` — member, workspace admin/owner,
API-key admin (hybrid.py:783) — and must NOT 401/403 the super admin.

Parametrized over one representative GET per router; the dependency is
router-wide, so one representative proves the whole router (a per-route
decoration could miss future endpoints — that is the point of the AC).
"""
from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock

import pytest

# Dummy POSTGRES_* satisfies the config chain at import (blessed pattern,
# see test_prd143_su_executor_gate.py) — the port points at nothing so any
# fail-soft connect refuses instantly. CI exports real vars (setdefault no-ops).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

_WS = uuid.uuid4()

MEMBER = UserContext(id="u-member", role="member", system_role="user")
WS_ADMIN = UserContext(id="u-ws-admin", role="admin", system_role="user")
WS_OWNER = UserContext(id="u-ws-owner", role="owner", system_role="user")
# hybrid.py:783 — API-key principals carry system_role='admin'.
API_KEY_ADMIN = UserContext(id="api_key", email=None, role="admin", system_role="admin")
SUPER_ADMIN = UserContext(id="u-gerard", role="admin", system_role="super_admin")

# (router module, representative GET path) — one per locked router.
ROUTERS = [
    pytest.param("api.heartbeat", "/api/heartbeat/status", id="heartbeat"),
    pytest.param("api.analytics", "/analytics/dashboard/summary", id="analytics"),
    pytest.param("api.analytics_api", "/api/analytics/dashboard/overview", id="analytics_api"),
    pytest.param("api.analytics_real", "/api/analytics/selection-health", id="analytics_real"),
    pytest.param("api.analytics_charts", "/api/analytics/charts/presets", id="analytics_charts"),
]


def _fake_db() -> MagicMock:
    """Query-chain stub: every chained call returns the query, terminals
    return empty/zero so su-path handlers compute real (empty) payloads."""
    db = MagicMock()
    q = MagicMock()
    for chain in ("filter", "filter_by", "group_by", "order_by", "limit", "offset", "join", "outerjoin", "distinct"):
        getattr(q, chain).return_value = q
    q.scalar.return_value = 0
    q.all.return_value = []
    q.first.return_value = None
    q.count.return_value = 0
    db.query.return_value = q
    result = MagicMock()
    result.fetchall.return_value = []
    result.fetchone.return_value = None
    result.scalar.return_value = 0
    db.execute.return_value = result
    return db


def _client(module_name: str, user: UserContext, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import importlib

    module = importlib.import_module(module_name)

    if module_name == "api.analytics_api":
        # The module-global engine would hit Redis/DB on first touch — stub it.
        engine = MagicMock()

        async def _overview():
            return {"stub": True}

        engine.get_dashboard_overview = _overview
        monkeypatch.setattr(module, "analytics_engine", engine)

    app = FastAPI()
    app.include_router(module.router)

    auth_type = "api_key" if user is API_KEY_ADMIN else "clerk"

    def _override_ctx():
        return RequestContext(workspace_id=_WS, user=user, auth_type=auth_type)

    def _override_db():
        yield _fake_db()

    app.dependency_overrides[get_request_context_hybrid] = _override_ctx
    app.dependency_overrides[get_db] = _override_db
    return TestClient(app, raise_server_exceptions=False)


def _assert_su_403(resp) -> None:
    assert resp.status_code == 403, f"expected 403, got {resp.status_code}: {resp.text}"
    assert resp.json()["detail"] == "Super admin only"


@pytest.mark.parametrize("module_name, path", ROUTERS)
def test_403_for_member(module_name, path, monkeypatch):
    resp = _client(module_name, MEMBER, monkeypatch).get(path)
    _assert_su_403(resp)


@pytest.mark.parametrize("module_name, path", ROUTERS)
def test_403_for_workspace_admin(module_name, path, monkeypatch):
    # Workspace-level admin AND owner without the system role: both refused.
    _assert_su_403(_client(module_name, WS_ADMIN, monkeypatch).get(path))
    _assert_su_403(_client(module_name, WS_OWNER, monkeypatch).get(path))


@pytest.mark.parametrize("module_name, path", ROUTERS)
def test_403_for_api_key_admin(module_name, path, monkeypatch):
    resp = _client(module_name, API_KEY_ADMIN, monkeypatch).get(path)
    _assert_su_403(resp)


@pytest.mark.parametrize("module_name, path", ROUTERS)
def test_not_403_for_super_admin(module_name, path, monkeypatch):
    resp = _client(module_name, SUPER_ADMIN, monkeypatch).get(path)
    assert resp.status_code not in (401, 403), (
        f"super admin must pass the gate; got {resp.status_code}: {resp.text}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
