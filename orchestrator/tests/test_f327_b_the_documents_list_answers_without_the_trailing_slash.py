"""F327 (night 9b) — GET /api/documents answers the same as GET /api/documents/.

The list was registered only at ``/api/documents/``, so the plain path matched
no route and FastAPI's redirect_slashes answered 307 with an empty body: a
caller that does not follow redirects saw nothing. The list is now registered
at both, the way /api/tools and /api/notifications already are, and the
committed route manifest lists both.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.documents import router
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

MANIFEST = Path(__file__).resolve().parent.parent / "reports" / "route-manifest.json"


def _who_are_you():
    raise HTTPException(status_code=401, detail="Sign in to list documents.")


def _no_database():
    """Never reached: the request is answered before the database is opened."""
    raise AssertionError("the list opened the database for a caller who is not signed in")


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_context_hybrid] = _who_are_you
    app.dependency_overrides[get_db] = _no_database
    return TestClient(app)


def test_the_plain_path_answers_the_same_as_the_slashed_one():
    client = _client()
    plain = client.get("/api/documents", follow_redirects=False)
    slashed = client.get("/api/documents/", follow_redirects=False)
    assert plain.status_code == slashed.status_code == 401        # never a 307 with an empty body
    assert plain.json() == slashed.json() == {"detail": "Sign in to list documents."}


def test_the_committed_manifest_lists_both_paths():
    routes = json.loads(MANIFEST.read_text())["routes"]
    listed = {(r.get("method"), r["path"]) for r in routes}
    assert {("GET", "/api/documents"), ("GET", "/api/documents/")} <= listed
