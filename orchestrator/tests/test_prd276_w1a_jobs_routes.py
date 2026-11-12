"""PRD-276 Wave 1a: the dead-job routes over HTTP, through the real permission dependency.

Listing, retrying or clearing a workspace's dead jobs needs ``workspace:manage``: a retry can send
to systems outside Automatos again. An owner or admin gets through on every route; an editor, a
viewer or a non-member is refused with 403 before the handler runs. The fake database answers the
membership lookup the gate makes and the handlers' job queries (the pattern of
``tests/test_heartbeat_routes_workspace_gate.py``); nothing here touches a real database.
"""
from __future__ import annotations

import uuid
from typing import Optional
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import jobs as jobs_api
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

_WS = uuid.uuid4()
_JOB = uuid.uuid4()
ROUTES = (("GET", "/api/v1/jobs/dead"), ("POST", f"/api/v1/jobs/{_JOB}/retry"), ("DELETE", f"/api/v1/jobs/{_JOB}"))


def _db(role: Optional[str]) -> MagicMock:
    """Answers the gate's membership lookup with ``role`` (None: not a member), and the handlers'
    job queries as a workspace with one dead job; records whether a job query ran."""
    db = MagicMock()
    db.job_queries = []

    def execute(stmt, params=None):
        sql, res = str(stmt), MagicMock()
        res.fetchone.return_value = (role,) if role and "SELECT wm.role" in sql else None
        if "FROM jobs" in sql or "UPDATE jobs" in sql or "DELETE FROM jobs" in sql:
            db.job_queries.append(sql)
        res.all.return_value, res.scalar.return_value = [], 1
        res.first.return_value, res.rowcount = ("test_prd276.kind",), 1
        return res

    db.execute.side_effect = execute
    return db


def _client(role: Optional[str], db: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(jobs_api.router)
    user = UserContext(id=f"u-{role}", email=f"{role}@example.test", system_role="user", clerk_user_id=f"clerk_{role}")
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(workspace_id=_WS, user=user,
                                                                                  auth_type="clerk")

    def session():
        yield db

    app.dependency_overrides[get_db] = session
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("role", ["owner", "admin"])
def test_a_workspace_manager_lists_retries_and_clears_dead_jobs(role):
    db = _db(role)
    client = _client(role, db)
    for method, path in ROUTES:
        answer = client.request(method, path)
        assert answer.status_code == 200, (role, method, path, answer.text)
    assert db.job_queries  # the handlers ran


@pytest.mark.parametrize("role", ["editor", "viewer", None])
def test_anyone_without_workspace_manage_is_refused_before_the_handler(role):
    db = _db(role)
    client = _client(role, db)
    for method, path in ROUTES:
        assert client.request(method, path).status_code == 403, (role, method, path)
    assert db.job_queries == []  # no job was read, retried or cleared
