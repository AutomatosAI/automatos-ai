"""F451 — GET /api/models/platform-default needs the workspace's agents:read.

The route rode only the hybrid context, so any signed-in principal read the platform's
default model route, member of the workspace or not. It now carries
``require_workspace_permission("agents:read")``, as its sibling reads in
api/models_endpoints.py (``/recommend``, ``/estimate-cost``) do. Every workspace role
holds agents:read (modules/policy/roles.py), so the pages that load it (Settings >
Orchestrator, the agent forms, the marketplace item) keep working for every member;
the local operator keeps it too.

Through the router over HTTP with the blessed fake-db pattern
(tests/test_heartbeat_routes_workspace_gate.py): only the gate's membership lookup is
answered.
"""
from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.models_endpoints import router  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.workspace_permission import PERMISSION_MARKER_ATTR  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.llm import new_agent_model  # noqa: E402

URL = "/api/models/platform-default"
ROUTE = ("anthropic", "claude-haiku-5-5")


@pytest.fixture(autouse=True)
def platform_default(monkeypatch):
    monkeypatch.setattr(new_agent_model, "default_route", lambda: ROUTE)


def _db(role=None) -> MagicMock:
    """A session that answers only the gate's membership lookup, with ``role``."""
    def execute(stmt, params=None):
        res = MagicMock()
        res.fetchone.return_value = (role,) if role and "SELECT wm.role" in str(stmt) else None
        return res

    db = MagicMock()
    db.execute.side_effect = execute
    return db


def _get(user: UserContext, db: MagicMock, auth_type: str = "clerk"):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=uuid.uuid4(), user=user, auth_type=auth_type)
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app, raise_server_exceptions=False).get(URL)


def _member(name: str) -> UserContext:
    return UserContext(id=f"u-{name}", email=f"{name}@example.test", system_role="user",
                       clerk_user_id=f"clerk_{name}")


def test_the_route_carries_the_agents_read_gate():
    [route] = [r for r in router.routes if r.path == URL]
    gates = {getattr(dep.call, PERMISSION_MARKER_ATTR) for dep in route.dependant.dependencies
             if hasattr(dep.call, PERMISSION_MARKER_ATTR)}
    assert gates == {"agents:read"}


def test_a_caller_without_agents_read_is_refused():
    resp = _get(_member("stranger"), _db(role=None))
    assert resp.status_code == 403, resp.text
    assert resp.json() == {"detail": "Permission denied: agents:read"}


@pytest.mark.parametrize("role", ["owner", "admin", "editor", "viewer"])
def test_every_workspace_role_reads_the_platform_default(role):
    resp = _get(_member(role), _db(role=role))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"provider": ROUTE[0], "model_id": ROUTE[1]}


def test_the_local_operator_reads_the_platform_default():
    """AUTH_EDITION=local: the anonymous session is the operator (core/auth/hybrid.py)."""
    operator = UserContext(email="operator@local", system_role="super_admin")
    resp = _get(operator, _db(role=None), auth_type="anonymous")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"provider": ROUTE[0], "model_id": ROUTE[1]}
