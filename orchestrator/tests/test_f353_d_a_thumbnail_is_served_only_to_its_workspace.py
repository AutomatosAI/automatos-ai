"""F353 (issue #947): a Deliverable's first-page picture is served only to its own workspace.

GET /api/deliverables/{id}/thumbnail reads the caller's workspace from the same
request-context dependency the Deliverable and its file are served behind, looks
the Deliverable up in THAT workspace only, and builds the stored file's name from
the id. So:

* the owner's workspace gets the PNG;
* another workspace, with the same id, gets 404 (and no bytes);
* an id that is not a UUID, or a Deliverable with no picture, gets 404.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

# Dummy POSTGRES_* satisfies the config chain at import (the blessed pattern of
# test_heartbeat_routes_workspace_gate.py). Nothing here touches a database.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.deliverable_thumbnails as thumbnails_api  # noqa: E402
from config import config  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.documents.thumbnails import store  # noqa: E402

OWNER = "00000000-0000-0000-0000-0000000000c1"
STRANGER = "00000000-0000-0000-0000-0000000000c2"
DOC = "5f0c2a8e-2b7d-4c55-9a51-0d6f1e2b2980"
PNG = b"\x89PNG\r\n\x1a\nfirst-page"


class OutputsView:
    """``v_workspace_outputs`` holding one live Deliverable of OWNER's."""

    def __init__(self):
        self.asked = []

    def execute(self, stmt, params):
        sql = str(stmt)
        self.asked.append((sql, dict(params)))
        assert "o.workspace_id = CAST(:ws AS uuid)" in sql
        hit = params["id"] == DOC and params["ws"] == OWNER
        row = SimpleNamespace(id=DOC, workspace_id=OWNER, artifact_type="document", source_type="chat",
                              storage_type="generated", file_path="generated/a.pdf", file_size_bytes=10)
        return SimpleNamespace(fetchone=lambda: row if hit else None)


@pytest.fixture
def stored(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(store, "is_storage_configured", lambda: False)
    store.save_thumbnail(OWNER, DOC, PNG)
    return tmp_path


def _client(workspace_id: str, view: OutputsView) -> TestClient:
    app = FastAPI()
    app.include_router(thumbnails_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: SimpleNamespace(workspace_id=workspace_id)

    def _db():
        yield view

    app.dependency_overrides[get_db] = _db
    return TestClient(app, raise_server_exceptions=False)


def test_the_owner_gets_the_picture(stored):
    resp = _client(OWNER, OutputsView()).get(f"/api/deliverables/{DOC}/thumbnail")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == PNG


def test_another_workspace_gets_404_for_the_same_id(stored):
    view = OutputsView()
    resp = _client(STRANGER, view).get(f"/api/deliverables/{DOC}/thumbnail")
    assert resp.status_code == 404
    assert PNG not in resp.content
    assert view.asked and view.asked[0][1]["ws"] == STRANGER


def test_an_id_that_is_not_a_uuid_is_404_before_anything_is_read(stored):
    view = OutputsView()
    resp = _client(OWNER, view).get("/api/deliverables/not-a-uuid/thumbnail")
    assert resp.status_code == 404
    assert view.asked == []


def test_a_deliverable_without_a_picture_is_404(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(store, "is_storage_configured", lambda: False)
    resp = _client(OWNER, OutputsView()).get(f"/api/deliverables/{DOC}/thumbnail")
    assert resp.status_code == 404


def test_the_route_uses_the_same_workspace_dependency_as_the_document():
    route = next(r for r in thumbnails_api.router.routes if r.path.endswith("/thumbnail"))
    deps = [d.call for d in route.dependant.dependencies]
    assert get_request_context_hybrid in deps
