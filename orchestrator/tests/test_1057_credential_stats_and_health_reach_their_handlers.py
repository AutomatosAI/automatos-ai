"""#1057 — ``GET /api/credentials/stats`` and ``/health`` reach their handlers.

Both used to be declared after ``GET /{credential_id}`` (an ``int`` path
parameter). Starlette matches routes in declaration order, so ``stats`` and
``health`` were taken as a credential id and FastAPI answered 422
``int_parsing`` before either handler ran.

Pure: the router is mounted on a bare app with its dependencies overridden,
no DB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import credentials as credentials_api  # noqa: E402
from core.database.database import get_db  # noqa: E402

_STATIC_GET_PATHS = ("/api/credentials/health", "/api/credentials/stats")
_DYNAMIC_GET_PATH = "/api/credentials/{credential_id}"


def _get_paths() -> list[str]:
    return [
        route.path
        for route in credentials_api.router.routes
        if "GET" in getattr(route, "methods", set())
    ]


def test_static_get_routes_are_declared_before_the_credential_id_route():
    paths = _get_paths()
    dynamic_at = paths.index(_DYNAMIC_GET_PATH)
    for static_path in _STATIC_GET_PATHS:
        assert paths.index(static_path) < dynamic_at, (
            f"{static_path} is shadowed by {_DYNAMIC_GET_PATH}"
        )


class _EmptyQuery:
    """Chainable stand-in for a SQLAlchemy query over an empty database."""

    def filter(self, *_args):
        return self

    join = group_by = filter

    def scalar(self):
        return 0

    def all(self):
        return []


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(credentials_api.router)
    ctx = MagicMock(workspace_id="ws-1")
    db = MagicMock()
    db.query.side_effect = lambda *_args: _EmptyQuery()
    app.dependency_overrides[credentials_api.get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def test_health_is_not_parsed_as_a_credential_id():
    response = _client().get("/api/credentials/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"healthy", "unhealthy"}


def test_stats_is_not_parsed_as_a_credential_id():
    response = _client().get("/api/credentials/stats")
    assert response.status_code == 200
    assert response.json()["credentials"]["total"] == 0
