"""PRD-273 US-273-03 — the export job and its routes.

On CI's Postgres, with committed rows (the job runs on sessions of its own, as in production)
swept after each test, and object storage faked:

* a member who is not an owner or admin is refused (403) on all three routes;
* an owner starts an export (202); the background job writes it, uploads it and marks it ready;
  the status shows its file and counts; the download works once and the file is deleted
  after it; a second download is 410;
* one export per workspace at a time (409); a job left running past the timeout no longer
  blocks a new one;
* expiry deletes the file: the sweep deletes a ready file past its link's expiry and the job
  reads expired; its download is 410;
* a failed export records a message that leaks nothing and uploads nothing;
* a passphrase is never stored; an unknown category, a short passphrase and a missing object
  store are refused; exports and downloads are audited; the sweep is registered.
"""
from __future__ import annotations

import io
import json
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import config
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_database_url, get_db
from core.models.workspace_move import WorkspaceMoveJob
from services.workspace_move import jobs, storage
from tests.helpers_prd273_workspace import FakeObjects, seed_user

PASSPHRASE = "a passphrase long enough"


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def maker(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def world(maker, monkeypatch):
    """A committed workspace and its owner; the job's sessions and object storage wired to the
    test; everything the test wrote is deleted afterwards."""
    db = maker()
    ws = uuid.uuid4()
    user_id = seed_user(db)
    db.execute(text("INSERT INTO workspaces (id, name, owner_id) VALUES (:id, 'Move me', :owner)"),
               {"id": str(ws), "owner": user_id})
    db.commit()
    store = FakeObjects({})
    monkeypatch.setattr(jobs, "_open_session", maker)
    monkeypatch.setattr(storage, "_client", lambda: store)
    monkeypatch.setattr(storage, "configured", lambda: True)
    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", "")
    yield SimpleNamespace(ws=ws, user_id=user_id, store=store, db=db)
    db.rollback()
    for statement in ("DELETE FROM audit_logs WHERE workspace_id = :ws",
                      "DELETE FROM workspace_move_jobs WHERE workspace_id = :ws",
                      "DELETE FROM workspaces WHERE id = :ws"):
        db.execute(text(statement), {"ws": str(ws)})
    db.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})
    db.commit()
    db.close()


def _client(world, maker, monkeypatch, *, admin: bool = True, workspace_id=None) -> TestClient:
    from api.workspace_move import router

    monkeypatch.setattr("core.auth.workspace_admin.may_see_own_workspace_health", lambda db, ctx: admin)
    monkeypatch.setattr("api.workspace_move.resolve_user_pk", lambda db, ctx: world.user_id)
    ctx = RequestContext(workspace_id=workspace_id or world.ws,
                         user=UserContext(id="u-owner", email="owner@example.test"), auth_type="clerk")

    def session():
        db = maker()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = session
    return TestClient(app)


def _job(world, job_id) -> WorkspaceMoveJob:
    world.db.expire_all()
    return world.db.get(WorkspaceMoveJob, uuid.UUID(str(job_id)))


def test_a_member_is_refused_on_every_route(world, maker, monkeypatch):
    client = _client(world, maker, monkeypatch, admin=False)
    job_id = uuid.uuid4()
    assert client.post("/api/workspace-move/export", json={}).status_code == 403
    assert client.get(f"/api/workspace-move/export/{job_id}").status_code == 403
    assert client.get(f"/api/workspace-move/export/{job_id}/file").status_code == 403


def test_an_owner_exports_and_the_link_works_once(world, maker, monkeypatch):
    client = _client(world, maker, monkeypatch)
    started = client.post("/api/workspace-move/export", json={"categories": ["chats"]})
    assert started.status_code == 202, started.text
    job_id = started.json()["id"]
    status = client.get(f"/api/workspace-move/export/{job_id}").json()
    assert status["status"] == "ready", status
    assert status["categories"] == ["auto", "chats"] and status["encrypted"] is False
    assert status["file_name"].startswith("move-me-") and status["file_name"].endswith(".automatos-workspace")
    assert status["summary"]["rows"]["chats"] == 0 and status["expires_at"]
    key = _job(world, job_id).storage_key
    assert key == storage.export_key(world.ws, job_id) and key in world.store.objects

    first = client.get(f"/api/workspace-move/export/{job_id}/file")
    assert first.status_code == 200 and first.headers["cache-control"] == "no-store"
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(first.content)).read("manifest.json"))
    assert manifest["source"]["workspace_id"] == str(world.ws) and manifest["categories"] == ["auto", "chats"]
    assert key not in world.store.objects and world.store.deleted == [key]
    assert _job(world, job_id).status == "downloaded" and _job(world, job_id).storage_key is None

    second = client.get(f"/api/workspace-move/export/{job_id}/file")
    assert second.status_code == 410


def test_one_export_per_workspace_at_a_time(world, maker, monkeypatch):
    monkeypatch.setattr(jobs, "run_export", lambda job_id, passphrase=None: None)  # it stays queued
    client = _client(world, maker, monkeypatch)
    assert client.post("/api/workspace-move/export", json={}).status_code == 202
    assert client.post("/api/workspace-move/export", json={}).status_code == 409


def test_a_job_past_the_timeout_no_longer_blocks_a_new_one(world):
    stale = jobs.start_export(world.db, world.ws, categories=None, encrypted=False, requested_by=world.user_id)
    world.db.execute(text("UPDATE workspace_move_jobs SET status = 'running', created_at = :at WHERE id = :id"),
                     {"at": datetime.now(timezone.utc) - timedelta(minutes=config.WORKSPACE_EXPORT_JOB_TIMEOUT_MINUTES + 1),
                      "id": stale.id})
    world.db.commit()
    fresh = jobs.start_export(world.db, world.ws, categories=None, encrypted=False, requested_by=world.user_id)
    assert fresh.id != stale.id
    assert _job(world, stale.id).status == "failed" and "restarted" in _job(world, stale.id).error


def test_expiry_deletes_the_file(world, maker, monkeypatch):
    client = _client(world, maker, monkeypatch)
    job_id = client.post("/api/workspace-move/export", json={}).json()["id"]
    key = _job(world, job_id).storage_key
    world.db.execute(text("UPDATE workspace_move_jobs SET expires_at = :at WHERE id = :id"),
                     {"at": datetime.now(timezone.utc) - timedelta(minutes=1), "id": job_id})
    world.db.commit()
    assert client.get(f"/api/workspace-move/export/{job_id}").json()["status"] == "expired"

    counts = jobs.sweep_exports(session_factory=maker)

    assert counts["expired"] >= 1 and key in world.store.deleted and key not in world.store.objects
    job = _job(world, job_id)
    assert job.status == "expired" and job.storage_key is None
    assert client.get(f"/api/workspace-move/export/{job_id}/file").status_code == 410


def test_a_failed_export_says_so_and_uploads_nothing(world, maker, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("relation secret_internal_name does not exist")

    monkeypatch.setattr(jobs, "write_workspace_file", broken)
    client = _client(world, maker, monkeypatch)
    job_id = client.post("/api/workspace-move/export", json={}).json()["id"]
    status = client.get(f"/api/workspace-move/export/{job_id}").json()
    assert status["status"] == "failed" and status["error"] == jobs.FAILED_GENERIC
    assert "secret_internal_name" not in json.dumps(status) and world.store.objects == {}
    assert client.get(f"/api/workspace-move/export/{job_id}/file").status_code == 410


def test_the_passphrase_is_never_stored_and_the_file_is_sealed(world, maker, monkeypatch):
    client = _client(world, maker, monkeypatch)
    job_id = client.post("/api/workspace-move/export", json={"passphrase": PASSPHRASE}).json()["id"]
    row = world.db.execute(text("SELECT * FROM workspace_move_jobs WHERE id = :id"), {"id": job_id}).mappings().one()
    assert row["encrypted"] is True and PASSPHRASE not in json.dumps(dict(row), default=str)
    content = client.get(f"/api/workspace-move/export/{job_id}/file").content
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(content)).read("manifest.json"))
    assert manifest["encryption"]["cipher"] == "AES-256-GCM"


def test_bad_requests_are_refused(world, maker, monkeypatch):
    client = _client(world, maker, monkeypatch)
    assert client.post("/api/workspace-move/export", json={"categories": ["telemetry"]}).status_code == 422
    assert client.post("/api/workspace-move/export", json={"passphrase": "short"}).status_code == 422
    assert client.post("/api/workspace-move/export", json={"secret": True}).status_code == 422
    assert client.get(f"/api/workspace-move/export/{uuid.uuid4()}").status_code == 404
    monkeypatch.setattr(storage, "configured", lambda: False)
    assert client.post("/api/workspace-move/export", json={}).status_code == 503


def test_exports_and_downloads_are_audited(world, maker, monkeypatch):
    client = _client(world, maker, monkeypatch)
    job_id = client.post("/api/workspace-move/export", json={}).json()["id"]
    client.get(f"/api/workspace-move/export/{job_id}/file")
    actions = world.db.execute(text("SELECT action, details FROM audit_logs WHERE workspace_id = :ws"),
                               {"ws": str(world.ws)}).all()
    assert {action for action, _ in actions} == {jobs.AUDIT_EXPORT, jobs.AUDIT_DOWNLOAD}
    assert all(details["job_id"] == job_id for _, details in actions)


def test_another_workspaces_export_is_not_found(world, maker, monkeypatch):
    theirs = jobs.start_export(world.db, world.ws, categories=None, encrypted=False, requested_by=None)
    foreign = _client(world, maker, monkeypatch, workspace_id=uuid.uuid4())
    assert foreign.get(f"/api/workspace-move/export/{theirs.id}").status_code == 404
    assert foreign.get(f"/api/workspace-move/export/{theirs.id}/file").status_code == 404
    assert _job(world, theirs.id).status == "queued"


def test_the_sweep_is_registered_on_the_leader():
    added = []
    scheduler = SimpleNamespace(running=True, add_job=lambda func, trigger, **kw: added.append((func, kw["id"])))
    assert jobs.register(scheduler) is True
    assert added == [(jobs.sweep_exports, jobs.SWEEP_JOB_ID)]
    assert jobs.register(SimpleNamespace(running=False)) is False
