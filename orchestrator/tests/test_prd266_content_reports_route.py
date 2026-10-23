"""PRD-266 US-016a: ``POST /api/v1/reports`` (``api/content_reports.py``).

The router is thin: Pydantic in (extra fields refused), one service call, errors
mapped. A new report is 201 ``{report_id, status: "received"}``, a duplicate 200
``{report_id, status: "already_received"}``; no person 403; a target the
reporter cannot see 404 ``report_target_not_found``; a failed write 503 that
leaks nothing; over the reporter's limit 429 before anything is written. Any
member may report, a viewer included (``members:read``, held by every role). It
mounts through ``RouterSpec("api.content_reports")``, never main.py.

DB-free: the workspace role and the service are faked where the real ones would
need Postgres.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import api.content_reports as reports_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.policy.roles import workspace_has_permission  # noqa: E402
from router_manifest import MANIFEST_ROUTERS, RouterSpec, load_routers  # noqa: E402
from services.content_reports import service  # noqa: E402

WS_ID = uuid.uuid4()
ROUTE = "/api/v1/reports"
MANIFEST = Path(__file__).resolve().parents[1] / "reports" / "route-manifest.json"
MESSAGE_ID = str(uuid.uuid4())
BODY = {"target_type": "message", "target_id": MESSAGE_ID, "reason": "harmful", "note": "Not OK."}


class _NoRowsDB:
    """Answers every query with no row and records any write."""

    def __init__(self):
        self.writes = []

    def execute(self, *_args, **_kwargs):
        return SimpleNamespace(fetchone=lambda: None, scalar=lambda: None)

    def query(self, *_args, **_kwargs):
        self.writes.append("query")
        raise AssertionError("nothing may be read past the rate limit")

    def add(self, row):
        self.writes.append(row)

    def commit(self):
        self.writes.append("commit")


def _member_ctx(user_id="member-1"):
    return RequestContext(
        workspace_id=WS_ID,
        user=UserContext(id=user_id, clerk_user_id="clerk_member", system_role="user"),
        auth_type="clerk",
    )


def _client(ctx, db=None):
    app = FastAPI()
    app.include_router(reports_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db if db is not None else _NoRowsDB()
    return TestClient(app)


def _as_role(monkeypatch, role):
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: role)


def _filed(report_id=41, duplicate=False):
    return service.FiledReport(
        report_id=report_id, duplicate=duplicate, workspace_id=str(WS_ID), target_type="message",
        target_id=MESSAGE_ID, reason="harmful", note="Not OK.", excerpt="An Auto reply.",
    )


@pytest.fixture
def filed(monkeypatch):
    """The service, faked: answers ``state["result"]`` or raises ``state["raise"]``."""
    state = {"result": _filed(), "raise": None, "calls": []}

    async def _file_report(db, ctx, **kwargs):
        state["calls"].append(kwargs)
        if state["raise"] is not None:
            raise state["raise"]
        return state["result"]

    monkeypatch.setattr(reports_api, "file_report", _file_report)
    return state


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def test_a_new_report_is_201_with_its_id(monkeypatch, filed):
    _as_role(monkeypatch, "editor")

    resp = _client(_member_ctx()).post(ROUTE, json=BODY)

    assert resp.status_code == 201
    assert resp.json() == {"report_id": 41, "status": "received"}
    assert filed["calls"] == [{"target_type": "message", "target_id": MESSAGE_ID,
                               "reason": "harmful", "note": "Not OK."}]


def test_a_duplicate_is_200_with_the_first_reports_id(monkeypatch, filed):
    _as_role(monkeypatch, "editor")
    filed["result"] = _filed(report_id=17, duplicate=True)

    resp = _client(_member_ctx()).post(ROUTE, json=BODY)

    assert resp.status_code == 200
    assert resp.json() == {"report_id": 17, "status": "already_received"}


@pytest.mark.parametrize("role", ["viewer", "editor", "admin", "owner"])
def test_every_member_may_report_a_viewer_included(monkeypatch, filed, role):
    _as_role(monkeypatch, role)

    assert _client(_member_ctx()).post(ROUTE, json=BODY).status_code == 201


def test_the_local_operator_may_report(filed):
    ctx = RequestContext(workspace_id=WS_ID, user=UserContext(id="local", system_role="user"),
                         auth_type="anonymous")

    assert _client(ctx).post(ROUTE, json=BODY).status_code == 201


def test_a_target_the_reporter_cannot_see_is_404_with_its_code(monkeypatch, filed):
    _as_role(monkeypatch, "editor")
    filed["raise"] = service.ReportTargetNotFound()

    resp = _client(_member_ctx()).post(ROUTE, json=BODY)

    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "report_target_not_found"


def test_a_failed_write_is_503_and_leaks_nothing(monkeypatch, filed):
    _as_role(monkeypatch, "editor")
    filed["raise"] = service.ReportNotRecorded()

    resp = _client(_member_ctx()).post(ROUTE, json=BODY)

    assert resp.status_code == 503
    assert resp.json() == {"detail": reports_api.NOT_RECORDED_DETAIL}
    assert "audit" not in resp.text.lower()


# ---------------------------------------------------------------------------
# Through the real service: no person, over the limit
# ---------------------------------------------------------------------------


def test_a_member_with_no_person_row_is_403(monkeypatch):
    """The gate passes (the role resolves), but no ``users`` row is the caller's."""
    _as_role(monkeypatch, "editor")

    resp = _client(_member_ctx()).post(ROUTE, json=BODY)

    assert resp.status_code == 403
    assert resp.json() == {"detail": reports_api.NEEDS_A_PERSON_DETAIL}


def test_an_sdk_key_is_refused_by_the_gate(filed):
    ctx = RequestContext(workspace_id=WS_ID, user=UserContext(id="sdk", system_role="user"),
                         auth_type="sdk_key")

    assert _client(ctx).post(ROUTE, json=BODY).status_code == 403
    assert filed["calls"] == []


def test_over_the_limit_is_429_before_anything_is_read_or_written(monkeypatch):
    _as_role(monkeypatch, "editor")
    seen = []

    async def _refuse(workspace_id, operation, **kwargs):
        seen.append((workspace_id, operation, kwargs["subject_id"]))
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "3600"})

    async def _no_target(*_args, **_kwargs):
        raise AssertionError("the target is not looked up past the limit")

    monkeypatch.setattr(service.rate_limiter, "check_rate_limit", _refuse)
    monkeypatch.setattr(service, "resolve_target", _no_target)
    db = _NoRowsDB()

    resp = _client(_member_ctx(user_id=7), db).post(ROUTE, json=BODY)

    assert resp.status_code == 429
    assert seen == [(str(WS_ID), "content_report", "7")]
    assert db.writes == []


# ---------------------------------------------------------------------------
# The body: what is refused before the service runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("change", [
    {"reason": "boring"},
    {"target_type": "chat"},
    {"reporter_email": "someone@example.test"},
    {"target_id": ""},
    {"target_id": "x" * 65},
    {"note": "n" * 501},
])
def test_a_bad_body_is_422_and_files_nothing(monkeypatch, filed, change):
    _as_role(monkeypatch, "editor")

    resp = _client(_member_ctx()).post(ROUTE, json={**BODY, **change})

    assert resp.status_code == 422
    assert filed["calls"] == []


def test_the_note_is_optional(monkeypatch, filed):
    _as_role(monkeypatch, "viewer")
    body = {k: v for k, v in BODY.items() if k != "note"}

    assert _client(_member_ctx()).post(ROUTE, json=body).status_code == 201
    assert filed["calls"][0]["note"] is None


# ---------------------------------------------------------------------------
# The route's gate, its manifest entry and its mount
# ---------------------------------------------------------------------------


def test_the_route_carries_the_members_read_marker_and_is_in_the_manifest():
    (route,) = [r for r in reports_api.router.routes if getattr(r, "path", None) == ROUTE]
    assert route.methods == {"POST"}
    markers = [getattr(dep.call, permission_mod.PERMISSION_MARKER_ATTR, None)
               for dep in route.dependant.dependencies]
    assert "members:read" in markers

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "POST", "path": ROUTE} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])


@pytest.mark.parametrize("role", ["owner", "admin", "editor", "viewer"])
def test_every_role_holds_members_read(role):
    assert workspace_has_permission(role, reports_api.REPORT_PERMISSION)


def test_the_router_mounts_through_the_manifest():
    assert RouterSpec("api.content_reports") in MANIFEST_ROUTERS

    mounted, failures = load_routers([RouterSpec("api.content_reports")], allow_degraded=False)

    assert failures == []
    assert mounted == [(RouterSpec("api.content_reports"), reports_api.router)]


def test_the_router_is_not_imported_by_main():
    main_py = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    imports = [line.strip() for line in main_py.splitlines() if line.strip().startswith(("import ", "from "))]
    assert not [line for line in imports if "content_reports" in line]
