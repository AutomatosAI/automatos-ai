"""PRD-263 US-016 — the Settings API offers the HARNESS modes that exist, and one schedule.

At 697ca7d6e ``PUT /api/workspaces/current/orchestrator`` accepted only ``full_auto``
and ``manual`` and three schedules (weekly, biweekly, monthly) though the sweep runs
weekly whatever is stored, and its defaults said ``enabled: false`` while HARNESS
ran in every workspace that had never saved the tab. The mode is now the only
switch: ``report_only`` (the default), ``propose`` or ``disabled``. Any other mode or
schedule is a 400 at the boundary, only the payload is validated, and the retired
``enabled``/``disabled`` keys are neither validated, written nor returned. GET and the
PUT response read the mode through services.harness_mode (§11-Q28: a stored
``full_auto`` or ``manual`` reads ``report_only``, a stored ``enabled: false`` or
``disabled: true`` reads ``disabled``).

DB-free: a fake workspace row and Auto agent stand in for Postgres.
"""
from __future__ import annotations

import inspect
import os
import uuid
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.workspaces as workspaces_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from services.harness_mode import HARNESS_MODES, harness_mode  # noqa: E402

WS_ID = uuid.uuid4()
ROUTE = "/api/workspaces/current/orchestrator"


class _Query:
    def __init__(self, db):
        self._db = db

    def get(self, ident):
        return self._db.workspace if ident == WS_ID else None


class _FakeDB:
    """Answers ``db.query(Workspace).get(id)`` and counts commits."""

    def __init__(self, settings):
        self.workspace = SimpleNamespace(id=WS_ID, settings=settings)
        self.commits = 0

    def query(self, _model):
        return _Query(self)

    def commit(self):
        self.commits += 1


@pytest.fixture
def client(monkeypatch):
    """A client on the workspaces router for an owner, with the Auto agent faked."""
    auto = SimpleNamespace(id=1, model_config={}, configuration={}, custom_persona_prompt=None,
                           use_custom_persona=False)
    monkeypatch.setattr(workspaces_api, "_get_or_seed_auto_agent", lambda db, workspace_id: auto)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda instance, key: None)

    def _client(db):
        app = FastAPI()
        app.include_router(workspaces_api.router)
        app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
            workspace_id=WS_ID, user=UserContext(id="owner-1", clerk_user_id="clerk_owner", system_role="user"),
            auth_type="clerk")
        app.dependency_overrides[get_db] = lambda: db
        return TestClient(app)

    return _client


def _stored(harness):
    return {"orchestrator": {"harness": harness}}


# ── PUT ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["report_only", "propose", "disabled"])
def test_put_stores_each_mode_that_exists(client, mode):
    db = _FakeDB({})

    resp = client(db).put(ROUTE, json={"harness": {"mode": mode}})

    assert resp.status_code == 200
    assert db.workspace.settings["orchestrator"]["harness"] == {"mode": mode}
    assert resp.json()["orchestrator"]["harness"] == {"schedule": "weekly", "mode": mode}
    assert db.commits == 1


@pytest.mark.parametrize("harness", [
    {"mode": "full_auto"}, {"mode": "manual"}, {"mode": "auto"},
    {"schedule": "biweekly"}, {"schedule": "monthly"},
])
def test_put_refuses_any_other_mode_or_schedule_and_writes_nothing(client, harness):
    db = _FakeDB({})

    resp = client(db).put(ROUTE, json={"harness": harness})

    assert resp.status_code == 400
    assert "harness." in resp.json()["detail"]
    assert db.workspace.settings == {} and db.commits == 0


def test_a_stored_legacy_value_never_blocks_an_unrelated_save(client):
    db = _FakeDB(_stored({"enabled": False, "schedule": "biweekly", "mode": "full_auto"}))

    resp = client(db).put(ROUTE, json={"thinking_level": "high"})

    assert resp.status_code == 200
    assert resp.json()["orchestrator"]["harness"] == {"schedule": "weekly", "mode": "disabled"}


def test_put_writes_no_retired_key_and_the_chosen_mode_wins_over_a_stored_one(client):
    db = _FakeDB(_stored({"enabled": False, "mode": "full_auto"}))

    resp = client(db).put(ROUTE, json={"harness": {"mode": "report_only", "enabled": True, "disabled": False}})

    assert resp.status_code == 200
    # Stored data is not rewritten (the old enabled: false stays), and the payload's
    # retired keys are not written; the chosen mode is what HARNESS reads.
    assert db.workspace.settings["orchestrator"]["harness"] == {"enabled": False, "mode": "report_only"}
    assert resp.json()["orchestrator"]["harness"] == {"schedule": "weekly", "mode": "report_only"}


# ── GET ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("settings, mode", [
    ({}, "report_only"),
    (_stored({"mode": "full_auto"}), "report_only"),
    (_stored({"mode": "manual", "schedule": "monthly"}), "report_only"),
    (_stored({"enabled": False}), "disabled"),
    (_stored({"enabled": False, "mode": "full_auto", "schedule": "weekly"}), "disabled"),
    (_stored({"disabled": True}), "disabled"),
    (_stored({"mode": "propose"}), "propose"),
])
def test_get_reads_the_mode_through_the_mapping_and_returns_no_retired_key(client, settings, mode):
    resp = client(_FakeDB(settings)).get(ROUTE)

    assert resp.status_code == 200
    assert resp.json()["harness"] == {"schedule": "weekly", "mode": mode}


# ── the vocabulary ───────────────────────────────────────────────────────────

def test_the_mode_is_the_only_switch_and_weekly_the_only_schedule():
    assert HARNESS_MODES == ("report_only", "propose", "disabled")
    assert workspaces_api._VALID_HARNESS_MODES == HARNESS_MODES
    assert workspaces_api._VALID_HARNESS_SCHEDULES == ["weekly"]
    assert workspaces_api._ORCHESTRATOR_DEFAULTS["harness"] == {"schedule": "weekly", "mode": "report_only"}
    assert harness_mode(None) == harness_mode({"orchestrator": None}) == "report_only"


def test_the_routes_are_plain_def_over_their_sync_session():
    """AGENTS.md: a route that touches the database without awaiting is a plain def."""
    assert not inspect.iscoroutinefunction(workspaces_api.get_orchestrator_settings)
    assert not inspect.iscoroutinefunction(workspaces_api.save_orchestrator_settings)
