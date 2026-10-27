"""F453 — a document download is the caller's own workspace's.

``GET /api/documents/{id}/download`` took only the id: no sign-in, no workspace
permission, and a lookup with no workspace filter, so any id was a pre-signed link to
anyone's file. ``GET /api/documents/download?path=`` served any file under the
documents root, every tenant's, to anyone. Both now ride the hybrid context and the
``documents:read`` gate. The id is looked up in the caller's workspace (another
workspace's id answers 404, exactly as a missing one does), and the path must sit in
the caller's own folder ``<root>/<workspace_id>/`` (services.document_downloads), with
``..`` and symlinks resolved before the comparison.

Through the router over HTTP, so the gates are the ones FastAPI actually runs.
"""
from __future__ import annotations

import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Blessed dummy POSTGRES_* for the config chain at import (no-ops in CI, where the
# real ones are set; the real-DB tests below run on CI's Postgres).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

import services.document_downloads as downloads  # noqa: E402
from api.documents import router  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.workspace_permission import PERMISSION_MARKER_ATTR  # noqa: E402
from core.database.database import get_db  # noqa: E402

WS = uuid.uuid4()
OTHER_WS = uuid.uuid4()
BY_ID_ROUTE = "/api/documents/{document_id}/download"
BY_PATH_ROUTE = "/api/documents/download"
PRESIGNED = "https://objects.example.test/signed-link"
NOT_FOUND = {"detail": "Document not found"}


def _member(name: str) -> UserContext:
    return UserContext(id=f"u-{name}", email=f"{name}@example.test", system_role="user",
                       clerk_user_id=f"clerk_{name}")


def _membership_db(role=None) -> MagicMock:
    """A session that answers only the gate's membership lookup, with ``role``."""
    def execute(stmt, params=None):
        res = MagicMock()
        res.fetchone.return_value = (role,) if role and "SELECT wm.role" in str(stmt) else None
        return res

    db = MagicMock()
    db.execute.side_effect = execute
    return db


def _client(db, user: UserContext, workspace_id=WS) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=workspace_id, user=user, auth_type="clerk")
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app, raise_server_exceptions=False)


def _not_signed_in():
    raise HTTPException(status_code=401, detail="Sign in to download documents.")


def _flatten(dependant, acc):
    for sub in dependant.dependencies:
        acc.append(sub.call)
        _flatten(sub, acc)
    return acc


# ── the gates ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [BY_ID_ROUTE, BY_PATH_ROUTE])
def test_both_downloads_carry_the_hybrid_context_and_the_documents_read_gate(path):
    [route] = [r for r in router.routes if r.path == path and "GET" in (getattr(r, "methods", None) or ())]
    calls = _flatten(route.dependant, [])
    assert get_request_context_hybrid in calls
    gates = {getattr(c, PERMISSION_MARKER_ATTR) for c in calls if hasattr(c, PERMISSION_MARKER_ATTR)}
    assert gates == {"documents:read"}


@pytest.mark.parametrize("url", ["/api/documents/1/download", f"{BY_PATH_ROUTE}?path=/var/automatos/documents/a.md"])
def test_a_caller_who_is_not_signed_in_gets_401(url):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_context_hybrid] = _not_signed_in
    app.dependency_overrides[get_db] = lambda: _membership_db()
    resp = TestClient(app, raise_server_exceptions=False).get(url)
    assert resp.status_code == 401, resp.text


@pytest.mark.parametrize("url", ["/api/documents/1/download", f"{BY_PATH_ROUTE}?path=/var/automatos/documents/a.md"])
def test_a_signed_in_stranger_to_the_workspace_is_refused(url):
    resp = _client(_membership_db(role=None), _member("stranger")).get(url)
    assert resp.status_code == 403, resp.text
    assert resp.json() == {"detail": "Permission denied: documents:read"}


# ── GET /api/documents/download?path= ───────────────────────────────────────

@pytest.fixture
def root(tmp_path, monkeypatch):
    """A documents root with this workspace's folder, another workspace's, a file at
    the root itself, and a symlink in our folder that points into theirs."""
    base = tmp_path / "documents"
    ours, theirs = base / str(WS), base / str(OTHER_WS)
    ours.mkdir(parents=True)
    theirs.mkdir(parents=True)
    (ours / "report.md").write_text("# Our report", encoding="utf-8")
    (theirs / "secret.md").write_text("# Their report", encoding="utf-8")
    (base / "loose.md").write_text("# Nobody's report", encoding="utf-8")
    (ours / "theirs.md").symlink_to(theirs / "secret.md")
    monkeypatch.setattr(downloads, "DOCUMENTS_ROOT", base)
    return base


def test_a_file_in_the_callers_own_folder_is_served(root):
    resp = _client(_membership_db(role="viewer"), _member("viewer")).get(
        BY_PATH_ROUTE, params={"path": str(root / str(WS) / "report.md")})
    assert resp.status_code == 200, resp.text
    assert resp.text == "# Our report"
    assert resp.headers["content-type"].startswith("text/markdown")


@pytest.mark.parametrize("relative", [
    f"{OTHER_WS}/secret.md",               # another workspace's folder
    "loose.md",                            # the root itself is no workspace's
    f"{WS}/../{OTHER_WS}/secret.md",       # traversal out of our folder
    f"{WS}/theirs.md",                     # a symlink out of our folder
    f"{WS}/missing.md",                    # nothing there
    str(WS),                               # our folder is not a file
])
def test_a_path_outside_the_callers_own_files_is_not_found(root, relative):
    resp = _client(_membership_db(role="owner"), _member("owner")).get(
        BY_PATH_ROUTE, params={"path": str(root / relative)})
    assert resp.status_code == 404, resp.text
    assert resp.json() == NOT_FOUND


def test_a_workspace_folder_that_links_to_another_is_not_served(root):
    linked = uuid.uuid4()
    (root / str(linked)).symlink_to(root / str(OTHER_WS), target_is_directory=True)
    resp = _client(_membership_db(role="owner"), _member("owner"), workspace_id=linked).get(
        BY_PATH_ROUTE, params={"path": str(root / str(linked) / "secret.md")})
    assert resp.status_code == 404 and resp.json() == NOT_FOUND


def test_a_path_outside_the_root_is_not_found(root):
    resp = _client(_membership_db(role="owner"), _member("owner")).get(BY_PATH_ROUTE, params={"path": "/etc/hosts"})
    assert resp.status_code == 404 and resp.json() == NOT_FOUND


# ── GET /api/documents/{id}/download (real rows) ────────────────────────────

class _Objects:
    """Object storage that holds every key: records what was asked, signs a fixed link."""

    def __init__(self):
        self.heads = []

    def head_object(self, Bucket, Key):
        self.heads.append(Key)

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        return f"{PRESIGNED}?key={Params['Key']}"


@pytest.fixture
def objects(monkeypatch):
    store = _Objects()
    monkeypatch.setattr(downloads, "is_storage_configured", lambda: True)
    monkeypatch.setattr(downloads, "get_s3_client", lambda: store)
    monkeypatch.setattr(downloads, "get_public_s3_client", lambda: store)
    return store


@pytest.fixture
def tenants(db_session, seed_workspace, tmp_path):
    """Two workspaces with a document each, and a signed-in owner of the first."""
    from core.models import Document

    db = db_session
    ours, theirs = uuid.UUID(seed_workspace()), uuid.UUID(seed_workspace())
    tag = uuid.uuid4().hex[:8]
    owner = _member(f"owner-{tag}")
    user_id = db.execute(text("INSERT INTO users (email, username, clerk_user_id) VALUES (:e, :u, :c) RETURNING id"),
                         {"e": owner.email, "u": f"owner-{tag}", "c": owner.clerk_user_id}).scalar()
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(ours), "user": user_id})

    def document(workspace_id, name):
        local = tmp_path / f"{workspace_id}-{name}"
        local.write_text(f"# {name}", encoding="utf-8")
        row = Document(filename=name, original_filename=name, workspace_id=workspace_id, status="processed",
                       file_path=str(local))
        db.add(row)
        db.flush()
        return row.id

    return SimpleNamespace(db=db, ws=ours, owner=owner, ours=document(ours, "ours.md"),
                           theirs=document(theirs, "theirs.md"))


def _owner_client(tenants) -> TestClient:
    return _client(tenants.db, tenants.owner, workspace_id=tenants.ws)


def test_the_owners_own_document_answers_with_a_signed_link(tenants, objects):
    resp = _owner_client(tenants).get(f"/api/documents/{tenants.ours}/download")
    assert resp.status_code == 200, resp.text
    key = f"workspaces/{tenants.ws}/documents/{tenants.ours}_ours.md"
    assert resp.json() == {"url": f"{PRESIGNED}?key={key}", "filename": "ours.md"}
    redirect = _owner_client(tenants).get(f"/api/documents/{tenants.ours}/download?mode=redirect",
                                          follow_redirects=False)
    assert redirect.status_code == 302 and redirect.headers["location"] == f"{PRESIGNED}?key={key}"


def test_another_workspaces_document_is_not_found_like_a_missing_one(tenants, objects):
    client = _owner_client(tenants)
    theirs = client.get(f"/api/documents/{tenants.theirs}/download")
    missing = client.get(f"/api/documents/{tenants.theirs + 100000}/download")
    assert theirs.status_code == missing.status_code == 404, theirs.text
    assert theirs.json() == missing.json() == NOT_FOUND
    assert objects.heads == []                     # no link was even looked up for their file


def test_without_object_storage_the_owners_own_file_is_served_from_disk(tenants, monkeypatch):
    monkeypatch.setattr(downloads, "is_storage_configured", lambda: False)
    resp = _owner_client(tenants).get(f"/api/documents/{tenants.ours}/download")
    assert resp.status_code == 200, resp.text
    assert resp.text == "# ours.md"
    assert _owner_client(tenants).get(f"/api/documents/{tenants.theirs}/download").status_code == 404
