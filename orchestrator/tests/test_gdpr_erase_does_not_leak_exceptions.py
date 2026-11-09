"""POST /api/v1/gdpr/erase and /erase-subject never echo a service exception
back to the caller (CodeQL py/stack-trace-exposure #317 #153).

Both routes used to `return result` straight from services.gdpr_service — if
that service's own cascade ever raised instead of returning (a DB error, a
Qdrant/durable-memory client error), the exception's text would have reached
this irreversible-erase response. They now catch, log with the trace, and
answer a fixed, generic 500.
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import gdpr as gdpr_api  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.workspace_admin import require_workspace_admin  # noqa: E402
from core.database.database import get_db  # noqa: E402

WS = uuid.uuid4()
SECRET = "db password is correct-horse-battery-staple at 10.0.0.5:5432"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(gdpr_api.router)
    user = UserContext(id="u-admin", email="admin@example.test", system_role="user",
                       clerk_user_id="clerk_admin")
    app.dependency_overrides[require_workspace_admin] = lambda: RequestContext(
        workspace_id=WS, user=user, auth_type="clerk")
    app.dependency_overrides[get_db] = lambda: object()
    return TestClient(app, raise_server_exceptions=False)


def test_erase_workspace_failure_does_not_leak_the_exception(monkeypatch):
    import services.gdpr_service as gdpr_service

    def _boom(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(gdpr_service, "erase_workspace", _boom)

    resp = _client().post("/api/v1/gdpr/erase", json={"confirm_workspace_id": str(WS)})

    assert resp.status_code == 500
    assert SECRET not in resp.text
    assert resp.json() == {"detail": "Workspace erasure failed"}


def test_erase_subject_failure_does_not_leak_the_exception(monkeypatch):
    import services.gdpr_service as gdpr_service

    def _boom(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(gdpr_service, "erase_data_subject", _boom)

    resp = _client().post("/api/v1/gdpr/erase-subject", json={"subject_id": "cust-123"})

    assert resp.status_code == 500
    assert SECRET not in resp.text
    assert resp.json() == {"detail": "Subject erasure failed"}
