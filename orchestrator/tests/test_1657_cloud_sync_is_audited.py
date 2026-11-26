"""#1657: a Sync with Automatos Cloud writes one ``marketplace:cloud_sync`` row in ``audit_logs``.

``POST /api/marketplace/cloud-sync`` (``api/marketplace_cloud_sync.py``) runs on a mini FastAPI app over
in-memory SQLite holding ``users`` and ``audit_logs`` (``helpers_audit_sqlite``). The Sync's own work is
replaced by recorders (the real Sync is proven on Postgres by ``test_prd270w1_cloud_sync_routes.py``), so
these tests look only at the row: who ran it, in which workspace, the outcome as counts; never the Cloud's
address, an error's text or the caller's token. Proven:

* a good Sync records ``ok`` with the whole Sync's counts, the plugins waiting and the installs moved, by
  the local operator, and by an enterprise install admin with a fresh sign-in;
* a Sync that ran and failed (502) records ``failed`` and nothing of the error;
* a Sync that couldn't start (409: off, or another running) and a refused caller record nothing;
* when the row can't be written the Sync's answer stands and an ERROR names the action.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.marketplace_cloud_sync as sync_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.principal import AUTH_SESSION_CLAIM_SOURCE, LOCAL_OPERATOR_CLAIM_SOURCE  # noqa: E402
from core.auth.session_lane import SESSION_AUTH_TYPE  # noqa: E402
from core.database.database import get_db  # noqa: E402
from services import admin_audit  # noqa: E402
from services.marketplace_cloud import pull, updates  # noqa: E402
from tests import helpers_audit_sqlite as audit  # noqa: E402

SYNC = "/api/marketplace/cloud-sync"
CLOUD = "https://cloud.example.test"
WS = uuid.uuid4()
OPERATOR_PK, ADMIN_PK = 1, 7
COUNTS = {"new": 2, "updated": 1, "unchanged": 4, "refused": 1}
UNREACHABLE = f"Automatos Cloud couldn't be reached at {CLOUD}"


def _operator() -> RequestContext:
    user = UserContext(id="operator@local", email="operator@local", system_role="user",
                       raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": OPERATOR_PK})
    return RequestContext(workspace_id=WS, user=user, auth_type="anonymous")


def _install_admin(system_role: str = "admin") -> RequestContext:
    """An enterprise install admin signed in on the session lane a minute ago."""
    claims = {"source": AUTH_SESSION_CLAIM_SOURCE, "user_id": ADMIN_PK, "auth_time": time.time() - 60}
    user = UserContext(id="person-1657", email="admin@example.test", system_role=system_role, raw_claims=claims)
    return RequestContext(workspace_id=WS, user=user, auth_type=SESSION_AUTH_TYPE)


@pytest.fixture
def api(monkeypatch):
    """The Sync router over SQLite in the local edition; ``outcome`` is what the next Sync answers."""
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id TEXT, email TEXT)"))
        audit.create_audit_logs(conn)
    session = sessionmaker(bind=engine)()
    state = SimpleNamespace(db=session, ctx=_operator(), outcome=pull.SyncOutcome(ok=True))
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(pull, "sync_catalog", lambda db, factory: state.outcome)
    monkeypatch.setattr(pull, "sync_status", lambda db: {"enabled": True})
    monkeypatch.setattr(sync_api, "run_item_passes", lambda db, outcome, factory, run: {
        "counts": COUNTS, "items": {}, "plugins": {"waiting": 3}})
    monkeypatch.setattr(updates, "apply_cloud_updates", lambda db, factory: updates.CloudUpdates(applied=2, failed=1))
    app = FastAPI()
    app.include_router(sync_api.router)
    step_up.install_step_up_handler(app)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _sync(api, **headers):
    return api.client.post(SYNC, headers=headers)


def test_a_good_sync_records_who_ran_it_and_its_counts_never_the_cloud_or_a_token(api):
    resp = _sync(api, Authorization="Bearer test-token", **{"User-Agent": "audit-test/1"})
    assert resp.status_code == 200, resp.text
    [row] = audit.rows(api.db, admin_audit.CLOUD_SYNC)
    assert (row["user_id"], row["actor_type"], row["workspace_id"]) == (OPERATOR_PK, "user", WS)
    assert (row["resource_type"], row["user_agent"]) == (admin_audit.CLOUD_SYNC_RESOURCE, "audit-test/1")
    assert row["details"] == {"outcome": "ok", "counts": COUNTS, "pluginsWaiting": 3,
                              "updatesApplied": 2, "updatesFailed": 1}
    assert "test-token" not in audit.written(row) and "cloud.example.test" not in audit.written(row)


def test_an_enterprise_install_admins_sync_is_recorded_as_theirs(api, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(config, "ENTERPRISE_CALLS_CLOUD_SYNC", "true")
    api.ctx = _install_admin()
    assert _sync(api).status_code == 200
    [row] = audit.rows(api.db, admin_audit.CLOUD_SYNC)
    assert (row["user_id"], row["workspace_id"], row["details"]["outcome"]) == (ADMIN_PK, WS, "ok")


def test_a_sync_that_ran_and_failed_is_recorded_as_failed_without_the_error(api):
    api.outcome = pull.SyncOutcome(ok=False, error=UNREACHABLE)
    resp = _sync(api)
    assert (resp.status_code, resp.json()["detail"]) == (502, UNREACHABLE)
    [row] = audit.rows(api.db, admin_audit.CLOUD_SYNC)
    assert row["details"] == {"outcome": "failed"} and row["user_id"] == OPERATOR_PK
    assert "cloud.example.test" not in audit.written(row)


@pytest.mark.parametrize("outcome", [pull.SyncOutcome(ok=False, error=pull.SYNC_OFF, off=True),
                                     pull.SyncOutcome(ok=False, error=pull.ALREADY_RUNNING, busy=True)],
                         ids=["off", "busy"])
def test_a_sync_that_could_not_start_records_nothing(api, outcome):
    api.outcome = outcome
    assert _sync(api).status_code == 409
    assert audit.rows(api.db, admin_audit.CLOUD_SYNC) == []


def test_a_refused_caller_records_nothing(api, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(config, "ENTERPRISE_CALLS_CLOUD_SYNC", "true")
    api.ctx = _install_admin()
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "viewer")
    assert _sync(api).status_code == 403  # workspace:manage refused
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    api.ctx = _install_admin(system_role="user")
    assert _sync(api).status_code == 403  # a workspace owner who is not an install admin
    assert audit.rows(api.db, admin_audit.CLOUD_SYNC) == []


def test_the_sync_stands_when_its_row_cannot_be_written_and_an_error_says_so(api, caplog):
    api.db.execute(sa.text("DROP TABLE audit_logs"))
    api.db.commit()
    with caplog.at_level(logging.ERROR, logger=admin_audit.logger.name):
        resp = _sync(api)
    assert resp.status_code == 200 and resp.json()["counts"] == COUNTS
    assert any(admin_audit.CLOUD_SYNC in r.getMessage() for r in caplog.records
               if r.name == admin_audit.logger.name and r.levelno == logging.ERROR)


def test_a_sync_that_raises_after_its_pull_committed_is_recorded_as_failed(api, monkeypatch):
    def _item_pass_breaks(db, outcome, factory, run):
        raise RuntimeError("an item pass broke")

    monkeypatch.setattr(sync_api, "run_item_passes", _item_pass_breaks)
    with pytest.raises(RuntimeError):
        _sync(api)
    [row] = audit.rows(api.db, admin_audit.CLOUD_SYNC)
    assert row["details"] == {"outcome": "failed"} and row["user_id"] == OPERATOR_PK
    assert "an item pass broke" not in audit.written(row)


def test_the_sync_stands_when_who_ran_it_cannot_be_read_and_an_error_says_so(api, monkeypatch, caplog):
    def _no_person(db, ctx):
        raise RuntimeError("users could not be read")

    monkeypatch.setattr(admin_audit, "resolve_user_pk", _no_person)
    rolled_back = []
    rollback = api.db.rollback
    monkeypatch.setattr(api.db, "rollback", lambda: rolled_back.append(True) or rollback())
    with caplog.at_level(logging.ERROR, logger=admin_audit.logger.name):
        resp = _sync(api)
    assert resp.status_code == 200 and resp.json()["counts"] == COUNTS
    assert rolled_back  # on Postgres the failed read aborted the transaction the status read then needs
    assert audit.rows(api.db, admin_audit.CLOUD_SYNC) == []
    assert any(admin_audit.CLOUD_SYNC in r.getMessage() for r in caplog.records
               if r.name == admin_audit.logger.name and r.levelno == logging.ERROR)
