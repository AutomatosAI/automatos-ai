"""PRD-143 S7 — obs routers batch 2 locked to the super admin.

Router-wide ``require_super_admin`` (core/auth/super_admin.py, S5) on:
api/llm_analytics.py (BOTH routers: router + admin_router),
api/statistics.py, api/composio_analytics.py, api/database_analytics.py,
api/execution_history.py, api/kpi_api.py, api/reports.py.

api/memory_stats.py is NO LONGER router-wide locked: PRD-77's Memory Explorer is
a per-workspace USER feature, so its reads (/browse, /health, /stats/*, /layers)
ride hybrid auth and are workspace-scoped. Only its mutating verbs
(DELETE /{id}, POST /consolidate) keep the super-admin lock on ``admin_router`` —
covered by ``test_p2w2_authz_boundary_sweep::test_obs_tier_mutating_routes_stay_super_admin_locked``.

Every endpoint on these routers must 403 for any principal that is not
literally ``system_role == 'super_admin'`` — member, workspace admin/owner,
API-key admin (hybrid.py:783) — and must NOT 401/403 the super admin.

Same parametrized shape as batch 1 (test_prd143_obs_routers_batch1.py):
one representative GET per router; the dependency is router-wide, so one
representative proves the whole router.
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

# (router module, router attribute, representative GET path) — one per locked
# router. llm_analytics carries TWO routers; both are part of the obs surface.
ROUTERS = [
    # api.llm_analytics "router" moved OFF the su-locked list 2026-07-30:
    # workspace owners/admins may read their own workspace's LLM analytics
    # (every endpoint filters by ctx.workspace_id). New contract lives in the
    # TestLlmAnalyticsWorkspaceAccess block below. Its mutating POST
    # (/openrouter/sync) keeps a route-level su lock, asserted there and by
    # the authz boundary sweep.
    pytest.param("api.llm_analytics", "admin_router", "/api/admin/analytics/costs", id="llm_admin_analytics"),
    pytest.param("api.statistics", "router", "/api/system/agents/statistics", id="statistics"),
    pytest.param("api.composio_analytics", "router", "/api/analytics/composio/apps", id="composio_analytics"),
    pytest.param("api.database_analytics", "router", "/api/database/analytics/stats", id="database_analytics"),
    pytest.param("api.execution_history", "router", "/api/execution-history/workflow/1/latest", id="execution_history"),
    pytest.param("api.kpi_api", "router", "/api/kpi/cost-tracker", id="kpi"),
    pytest.param("api.reports", "router", "/api/reports/stats", id="reports"),
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


def _client(module_name: str, router_attr: str, user: UserContext) -> TestClient:
    import importlib

    module = importlib.import_module(module_name)

    app = FastAPI()
    app.include_router(getattr(module, router_attr))

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


@pytest.mark.parametrize("module_name, router_attr, path", ROUTERS)
def test_403_for_member(module_name, router_attr, path):
    resp = _client(module_name, router_attr, MEMBER).get(path)
    _assert_su_403(resp)


@pytest.mark.parametrize("module_name, router_attr, path", ROUTERS)
def test_403_for_workspace_admin(module_name, router_attr, path):
    # Workspace-level admin AND owner without the system role: both refused.
    _assert_su_403(_client(module_name, router_attr, WS_ADMIN).get(path))
    _assert_su_403(_client(module_name, router_attr, WS_OWNER).get(path))


@pytest.mark.parametrize("module_name, router_attr, path", ROUTERS)
def test_403_for_api_key_admin(module_name, router_attr, path):
    resp = _client(module_name, router_attr, API_KEY_ADMIN).get(path)
    _assert_su_403(resp)


@pytest.mark.parametrize("module_name, router_attr, path", ROUTERS)
def test_not_403_for_super_admin(module_name, router_attr, path):
    resp = _client(module_name, router_attr, SUPER_ADMIN).get(path)
    assert resp.status_code not in (401, 403), (
        f"super admin must pass the gate; got {resp.status_code}: {resp.text}"
    )


# ── 2026-07-30: llm_analytics workspace-scoped router — new contract ────────
#
# Workspace owners/admins read their OWN workspace's LLM analytics
# (require_workspace_admin). Members and API-key principals still 403.
# The mutating POST /openrouter/sync keeps a route-level super-admin lock.

_USAGE = "/api/analytics/llm/usage"
_SYNC = "/api/analytics/llm/openrouter/sync"


def _member_db(is_admin_member: bool) -> MagicMock:
    """Fake db whose raw-SQL membership probe answers the workspace-admin
    check; query-chain terminals stay empty like _fake_db."""
    db = _fake_db()
    result = MagicMock()
    result.fetchone.return_value = (1,) if is_admin_member else None
    result.fetchall.return_value = []
    result.scalar.return_value = 0
    db.execute.return_value = result
    return db


def _ws_client(user: UserContext, is_admin_member: bool) -> TestClient:
    import importlib

    module = importlib.import_module("api.llm_analytics")
    app = FastAPI()
    app.include_router(module.router)

    def _override_ctx():
        return RequestContext(workspace_id=_WS, user=user, auth_type="clerk")

    def _override_db():
        yield _member_db(is_admin_member)

    app.dependency_overrides[get_request_context_hybrid] = _override_ctx
    app.dependency_overrides[get_db] = _override_db
    return TestClient(app, raise_server_exceptions=False)


def test_llm_analytics_member_still_403():
    resp = _ws_client(MEMBER, is_admin_member=False).get(_USAGE)
    assert resp.status_code == 403, f"expected 403, got {resp.status_code}: {resp.text}"
    assert resp.json()["detail"] == "Workspace admin only"


def test_llm_analytics_api_key_admin_still_403():
    # API-key principals (system_role='admin') have no workspace membership —
    # refused by require_workspace_admin.
    resp = _ws_client(API_KEY_ADMIN, is_admin_member=False).get(_USAGE)
    assert resp.status_code == 403, f"expected 403, got {resp.status_code}: {resp.text}"


def test_llm_analytics_workspace_admin_and_owner_pass():
    for user in (WS_ADMIN, WS_OWNER):
        resp = _ws_client(user, is_admin_member=True).get(_USAGE)
        assert resp.status_code not in (401, 403), (
            f"workspace admin/owner must read own analytics; got {resp.status_code}: {resp.text}"
        )


def test_llm_analytics_super_admin_passes_without_membership():
    resp = _ws_client(SUPER_ADMIN, is_admin_member=False).get(_USAGE)
    assert resp.status_code not in (401, 403), (
        f"super admin must pass; got {resp.status_code}: {resp.text}"
    )


def test_llm_analytics_openrouter_sync_stays_super_admin_locked():
    # Even a legitimate workspace admin must not trigger the mutating sync.
    resp = _ws_client(WS_ADMIN, is_admin_member=True).post(_SYNC)
    assert resp.status_code == 403, f"expected 403, got {resp.status_code}: {resp.text}"
    assert resp.json()["detail"] == "Super admin only"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
