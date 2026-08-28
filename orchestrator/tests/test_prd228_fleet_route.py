"""PRD-228 US-002 — GET /api/v1/fleet.

The route is a thin, workspace-scoped shell over the US-001 read-model, guarded
by the board list's exact read dependency (``require_task_context(TASKS_READ)``).
These tests mount the router on a bare app and drive it with FastAPI's
dependency overrides — no DB, no real auth — to prove:

  * a call returns the read-model shape for the CALLER's workspace only (the
    workspace comes from the authenticated context, never a client parameter),
  * two different callers each see only their own workspace,
  * a guard denial (unauthorized) blocks the handler — the service never runs,
  * the route mirrors the board list's read guard (source check).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.fleet as fleet_mod  # noqa: E402
import services.fleet_state as fs  # noqa: E402
from core.database.database import get_db  # noqa: E402

WS_A = uuid4()
WS_B = uuid4()


class _Spy:
    """Records (db, workspace_id) the handler forwards to the read-model."""

    def __init__(self, payload):
        self.calls = []
        self.payload = payload

    def __call__(self, db, workspace_id):
        self.calls.append((db, workspace_id))
        return {**self.payload, "_ws_echo": str(workspace_id)}


def _client(ctx_factory, spy, monkeypatch):
    monkeypatch.setattr(fleet_mod, "get_fleet_state", spy)
    app = FastAPI()
    app.include_router(fleet_mod.router)
    app.dependency_overrides[fleet_mod._require_fleet_read] = ctx_factory
    app.dependency_overrides[get_db] = lambda: SimpleNamespace(name="dummy-session")
    return TestClient(app)


def _ok_ctx(workspace_id):
    def _factory():
        return SimpleNamespace(workspace_id=workspace_id)
    return _factory


def test_returns_read_model_shape_for_caller_workspace(monkeypatch):
    spy = _Spy({"version": 1, "cost_source": "llm_usage", "agents": [{"agent_id": 1}]})
    client = _client(_ok_ctx(WS_A), spy, monkeypatch)

    resp = client.get("/api/v1/fleet")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == 1
    assert body["cost_source"] == "llm_usage"
    assert body["agents"] == [{"agent_id": 1}]
    # Exactly one read, scoped to the caller's workspace.
    assert len(spy.calls) == 1
    assert spy.calls[0][1] == WS_A


def test_uses_caller_context_not_client_supplied_workspace(monkeypatch):
    spy = _Spy({"version": 1, "agents": []})
    client = _client(_ok_ctx(WS_A), spy, monkeypatch)

    # A client trying to widen scope via a query param cannot: the route has no
    # workspace parameter, so the authenticated context wins.
    resp = client.get(f"/api/v1/fleet?workspace_id={WS_B}")
    assert resp.status_code == 200
    assert spy.calls[0][1] == WS_A
    assert spy.calls[0][1] != WS_B


def test_two_callers_each_see_only_their_workspace(monkeypatch):
    spy = _Spy({"version": 1, "agents": []})

    client_a = _client(_ok_ctx(WS_A), spy, monkeypatch)
    assert client_a.get("/api/v1/fleet").json()["_ws_echo"] == str(WS_A)

    client_b = _client(_ok_ctx(WS_B), spy, monkeypatch)
    assert client_b.get("/api/v1/fleet").json()["_ws_echo"] == str(WS_B)

    assert {c[1] for c in spy.calls} == {WS_A, WS_B}


def test_unauthorized_is_denied_and_service_never_runs(monkeypatch):
    spy = _Spy({"version": 1, "agents": []})

    def _deny():
        raise HTTPException(status_code=401, detail="unauthorized")

    client = _client(_deny, spy, monkeypatch)
    resp = client.get("/api/v1/fleet")
    assert resp.status_code == 401
    assert spy.calls == []  # the guard blocked the handler entirely


def test_route_mirrors_board_read_guard():
    src = Path(fleet_mod.__file__).read_text(encoding="utf-8")
    # Same guard the board list uses (api/board_tasks.py list_tasks).
    assert "require_task_context(TASKS_READ)" in src
    assert fleet_mod.router.prefix == "/api/v1/fleet"


# ---------------------------------------------------------------------------
# P228-RVW-2 — a non-cost enrichment source failing must NOT 500 the route.
# This drives the REAL read-model (get_fleet_state is not stubbed) through a
# minimal session, with the watches source monkeypatched to raise.
# ---------------------------------------------------------------------------

class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def group_by(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _RealishSession:
    """Serves the real read-model: agent rows for the Agent query, empty for
    every other source (board/mission/asks/cost)."""

    def __init__(self, agents):
        self._agents = agents

    def query(self, *entities):
        from core.models.core import Agent

        if entities and entities[0] is Agent:
            return _FakeQuery(self._agents)
        return _FakeQuery([])


def test_route_returns_200_when_enrichment_source_fails(monkeypatch):
    """P228-RVW-2: a watches-source failure degrades to safe defaults — the
    route returns 200 (not 500) with the fleet intact, watches defaulted."""
    agent = SimpleNamespace(id=1, name="Solo")

    def _boom(*a, **k):
        raise RuntimeError("watches table locked")

    monkeypatch.setattr(fs, "_watches_source", _boom)

    app = FastAPI()
    app.include_router(fleet_mod.router)
    app.dependency_overrides[fleet_mod._require_fleet_read] = _ok_ctx(WS_A)
    app.dependency_overrides[get_db] = lambda: _RealishSession([agent])
    client = TestClient(app)

    resp = client.get("/api/v1/fleet")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["agents"]) == 1
    assert body["agents"][0]["watches"] == {"active": 0, "needs_attention": 0}
