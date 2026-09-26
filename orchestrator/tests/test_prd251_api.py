"""PRD-251 S0.3b — the /api/socials routes behind the gate.

The real router runs on a mini FastAPI app over in-memory SQLite (the social
tables plus SQLite copies of ``workspaces`` / ``document_templates``), with the
REAL gate (``require_socials_enabled``) and the REAL workspace-permission gate;
only the master switch's system setting and the caller's member role are
stubbed. Pins:

* D1 — master switch off: every route 404s; master on, workspace off: still 404;
* tenant isolation — another workspace's post (or template) never resolves;
* D6 — PATCH of an approved post's copy returns it in needs_approval with a new
  content_hash; publish-now on a stale approval is 409 and the Composio executor
  is never called; a valid approval reaches the Wave 0 seam and answers 501;
* D6 — approve carries the content_hash the reviewer saw: a post edited since
  is 409 with the current hash and stays unapproved, and so is an edit another
  worker commits while the approve request runs (the compare-and-set);
* D6/D7 — every other write is the same compare-and-set (P251-RVW-5): an edit,
  submit, review, schedule or unschedule that another worker's commit overtook
  is 409, writes nothing, and leaves every committed review_log entry in place;
* D7 — approve with an unsourced claim is 422 naming it; the override is 200
  and recorded;
* ``socials:approve`` — a viewer gets 403, an editor 200;
* every route under /api/socials in the REAL app carries the gate (a probe
  subprocess imports ``main.app``), and every route is in the committed
  route manifest.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.socials as socials_api  # noqa: E402
import modules.socials.service as socials_service  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.socials.publisher as publisher  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.core import DocumentTemplate  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials.settings import require_socials_enabled  # noqa: E402

WS_A = uuid.uuid4()
WS_B = uuid.uuid4()
WS_OFF = uuid.uuid4()
NOW = datetime(2026, 9, 23, 12, 0)
MANIFEST = _ORCH / "reports" / "route-manifest.json"
ACTION_PATHS = ("submit", "approve", "request-changes", "reject", "schedule", "unschedule", "publish-now")
FUTURE_SLOT = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _portable(col_type):
    if isinstance(col_type, (JSONB, ARRAY)):
        return sa.JSON()
    if isinstance(col_type, sa.Uuid) or type(col_type).__name__.upper() == "UUID":
        return sa.Uuid()
    return col_type


def _sqlite_copy(table: sa.Table, metadata: sa.MetaData) -> sa.Table:
    """A column-for-column copy SQLite can build (JSONB / ARRAY → JSON, UUID → CHAR(32))."""
    columns = [
        sa.Column(col.name, _portable(col.type), primary_key=col.primary_key)
        for col in table.columns
    ]
    return sa.Table(table.name, metadata, *columns)


def _ctx(workspace_id, user_id="member-1"):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id=user_id, clerk_user_id=f"clerk-{user_id}", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def api(monkeypatch):
    engine = sa.create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    copies = sa.MetaData()
    _sqlite_copy(Workspace.__table__, copies)
    _sqlite_copy(DocumentTemplate.__table__, copies)
    copies.create_all(engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])

    session = sessionmaker(bind=engine)()
    for ws_id, settings in (
        (WS_A, {"socials": {"enabled": True}}),
        (WS_B, {"socials": {"enabled": True}}),
        (WS_OFF, {}),
    ):
        session.add(
            Workspace(
                id=ws_id, name=f"ws-{ws_id.hex[:6]}", plan="basic", plan_limits={},
                settings=settings, onboarding={}, created_at=NOW, updated_at=NOW,
            )
        )
    session.commit()

    state = SimpleNamespace(session=session, ctx=_ctx(WS_A), role="owner", master="true")

    def fake_read_system_setting(category, key):
        assert (category, key) == ("socials", "enabled")
        return state.master

    monkeypatch.setattr(socials_settings, "read_system_setting", fake_read_system_setting)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)

    app = FastAPI()
    app.include_router(socials_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _create(api, **body):
    payload = {"title": "Countdown", "brief": "Three weeks to Lisbon", "copy": {"base": "Three weeks to go."}}
    payload.update(body)
    resp = api.client.post("/api/socials/posts", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _post(api, post_id, action, body=None):
    return api.client.post(f"/api/socials/posts/{post_id}/{action}", json=body)


def _approved(api, **body):
    post = _create(api, **body)
    submitted = _post(api, post["id"], "submit")
    assert submitted.status_code == 200
    resp = _post(api, post["id"], "approve", {"content_hash": submitted.json()["content_hash"]})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _router_routes():
    out = []
    for route in socials_api.router.routes:
        if isinstance(route, APIRoute):
            for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
                out.append((method, route.path))
    return sorted(out)


def _url(path, post_id):
    return path.replace("{post_id}", post_id)


# A body each route accepts, so a 404 proves the lookup, not a validation error.
_VALID_BODY = {
    ("POST", "/api/socials/posts"): {"title": "T"},
    ("PATCH", "/api/socials/posts/{post_id}"): {"title": "Renamed"},
    ("POST", "/api/socials/posts/{post_id}/approve"): {"content_hash": "0" * 64},
    ("POST", "/api/socials/posts/{post_id}/request-changes"): {"comment": "Change it"},
    ("POST", "/api/socials/posts/{post_id}/reject"): {"reason": "No"},
    ("POST", "/api/socials/posts/{post_id}/schedule"): {"scheduled_for": FUTURE_SLOT},
}


# ---------------------------------------------------------------------------
# D1 — the gate
# ---------------------------------------------------------------------------


def test_the_router_serves_every_wave_0_and_wave_1_route():
    assert _router_routes() == sorted(
        [
            ("GET", "/api/socials/posts"),
            ("POST", "/api/socials/posts"),
            ("GET", "/api/socials/posts/{post_id}"),
            ("PATCH", "/api/socials/posts/{post_id}"),
            # Wave 1 (S1.1c): rendering, the rendered files, the render minutes.
            ("POST", "/api/socials/posts/{post_id}/render"),
            ("GET", "/api/socials/posts/{post_id}/media/{file_name}"),
            ("GET", "/api/socials/usage"),
            # Wave 1 (S1.4): the source picker's search.
            ("GET", "/api/socials/sources"),
            # Wave 1 (S1.7): a chart template filled from a report (the infographic).
            ("GET", "/api/socials/sources/reports/{report_id}/chart"),
            # Wave 1 (S1.5): the voice picker's choices, and a voice toolkit's voices.
            ("GET", "/api/socials/voices"),
            ("GET", "/api/socials/voices/{toolkit}"),
            # Wave 1 (S1.8): what a post's slots can be filled with, and the month's media spend.
            ("GET", "/api/socials/footage"),
        ]
        + [("POST", f"/api/socials/posts/{{post_id}}/{a}") for a in ACTION_PATHS]
    )


@pytest.mark.parametrize("method, path", _router_routes())
def test_every_route_is_404_when_the_master_switch_is_off(api, method, path):
    post = _create(api)
    api.master = "false"
    resp = api.client.request(method, _url(path, post["id"]), json={})
    assert resp.status_code == 404


@pytest.mark.parametrize("method, path", _router_routes())
def test_every_route_is_404_when_the_workspace_switch_is_off(api, method, path):
    post = _create(api)
    api.ctx = _ctx(WS_OFF)
    resp = api.client.request(method, _url(path, post["id"]), json={})
    assert resp.status_code == 404


def test_the_gate_answers_before_the_permission_check(api):
    """A viewer with the switch off sees 404, never a 403 that confirms the route."""
    post = _create(api)
    api.master, api.role = "false", "viewer"
    assert _post(api, post["id"], "approve", {}).status_code == 404


def test_every_route_depends_on_the_gate_first():
    for route in socials_api.router.routes:
        assert isinstance(route, APIRoute)
        first = route.dependant.dependencies[0].call
        assert first is require_socials_enabled, route.path


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


def test_another_workspaces_post_is_404_everywhere(api):
    api.ctx = _ctx(WS_B, "member-b")
    theirs = _create(api, title="Theirs")
    api.ctx = _ctx(WS_A)
    mine = _create(api, title="Mine")

    for method, path in _router_routes():
        if "{post_id}" not in path:
            continue
        body = _VALID_BODY.get((method, path))
        resp = api.client.request(method, _url(path, theirs["id"]), json=body)
        assert resp.status_code == 404, (method, path, resp.status_code, resp.text)

    listed = api.client.get("/api/socials/posts").json()
    assert [p["id"] for p in listed["posts"]] == [mine["id"]]
    api.session.expire_all()
    assert api.session.get(SocialPost, uuid.UUID(theirs["id"])).title == "Theirs"


def test_a_template_from_another_workspace_is_refused(api):
    own, foreign = uuid.uuid4(), uuid.uuid4()
    for tpl_id, ws_id in ((own, WS_A), (foreign, WS_B)):
        api.session.execute(
            sa.text(
                "INSERT INTO document_templates (id, workspace_id, name, format, data_schema) "
                "VALUES (:id, :ws, :name, 'pdf', '{}')"
            ),
            {"id": tpl_id.hex, "ws": ws_id.hex, "name": f"tpl-{tpl_id.hex[:6]}"},
        )
    api.session.commit()

    resp = api.client.post("/api/socials/posts", json={"title": "T", "template_id": str(foreign)})
    assert resp.status_code == 422 and "template" in resp.json()["detail"]
    created = _create(api, template_id=str(own))
    assert created["template_id"] == str(own)
    resp = api.client.patch(f"/api/socials/posts/{created['id']}", json={"template_id": str(foreign)})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Create, read, list
# ---------------------------------------------------------------------------


def test_create_read_and_list(api):
    post = _create(api, format="fact_card", variables={"days": {"value": 21, "claim": False}})
    assert post["status"] == "draft" and post["workspace_id"] == str(WS_A)
    assert post["created_by"] == "member-1" and len(post["content_hash"]) == 64
    assert post["copy"] == {"base": "Three weeks to go."}

    assert api.client.get(f"/api/socials/posts/{post['id']}").json()["id"] == post["id"]
    listed = api.client.get("/api/socials/posts").json()
    assert listed["total"] == 1 and listed["posts"][0]["id"] == post["id"]
    assert api.client.get("/api/socials/posts", params={"status": "approved"}).json()["total"] == 0
    assert api.client.get("/api/socials/posts", params={"status": "draft,approved"}).json()["total"] == 1
    assert api.client.get("/api/socials/posts", params={"status": "posted"}).status_code == 422


def test_list_from_to_bounds_a_posts_date(api):
    post = _create(api)
    today = datetime.now(timezone.utc)
    inside = {"from": (today - timedelta(days=1)).isoformat(), "to": (today + timedelta(days=1)).isoformat()}
    outside = {"from": (today + timedelta(days=1)).isoformat(), "to": (today + timedelta(days=2)).isoformat()}
    assert [p["id"] for p in api.client.get("/api/socials/posts", params=inside).json()["posts"]] == [post["id"]]
    assert api.client.get("/api/socials/posts", params=outside).json()["total"] == 0


@pytest.mark.parametrize(
    "body",
    [
        {"title": ""},
        {"title": "T", "format": "gif"},
        {"title": "T", "copy": {"headline": "x"}},
        {"title": "T", "sources": {"users": {"kind": "rumour", "ref": "x"}}},
        {"title": "T", "status": "approved"},
        {"title": "T", "approved_hash": "0" * 64},
    ],
)
def test_malformed_or_privileged_fields_are_refused(api, body):
    assert api.client.post("/api/socials/posts", json=body).status_code == 422


def test_a_viewer_cannot_create_or_edit(api):
    post = _create(api)
    api.role = "viewer"
    assert api.client.post("/api/socials/posts", json={"title": "T"}).status_code == 403
    assert api.client.patch(f"/api/socials/posts/{post['id']}", json={"title": "X"}).status_code == 403
    assert _post(api, post["id"], "submit").status_code == 403
    assert api.client.get(f"/api/socials/posts/{post['id']}").status_code == 200  # read stays open


# ---------------------------------------------------------------------------
# D6 — approval binds to the hash
# ---------------------------------------------------------------------------


def test_patch_of_an_approved_posts_copy_returns_it_in_needs_approval(api):
    approved = _approved(api)
    assert approved["status"] == "approved" and approved["approved_hash"] == approved["content_hash"]

    resp = api.client.patch(f"/api/socials/posts/{approved['id']}", json={"copy": {"base": "Two weeks to go."}})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "needs_approval"
    assert body["content_hash"] != approved["content_hash"]
    assert body["approved_hash"] == approved["approved_hash"] != body["content_hash"]
    assert body["review_log"][-1]["action"] == "approval_voided"


def test_publish_now_on_a_stale_approval_is_409_and_never_reaches_composio(api, monkeypatch):
    from core.composio.tool_executor import ComposioToolExecutor

    composio = MagicMock(name="ComposioToolExecutor.execute", side_effect=AssertionError("must not run"))
    monkeypatch.setattr(ComposioToolExecutor, "execute", composio)
    seam = MagicMock(name="_publish_targets")
    monkeypatch.setattr(publisher, "_publish_targets", seam)

    approved = _approved(api)
    row = api.session.get(SocialPost, uuid.UUID(approved["id"]))
    row.approved_hash = "f" * 64  # approved_hash != content_hash
    api.session.commit()

    resp = _post(api, approved["id"], "publish-now")

    assert resp.status_code == 409
    composio.assert_not_called()
    seam.assert_not_called()


def test_publish_now_after_an_edit_is_409_and_never_reaches_composio(api, monkeypatch):
    from core.composio.tool_executor import ComposioToolExecutor

    composio = MagicMock(name="ComposioToolExecutor.execute")
    monkeypatch.setattr(ComposioToolExecutor, "execute", composio)
    approved = _approved(api)
    api.client.patch(f"/api/socials/posts/{approved['id']}", json={"copy": {"base": "Edited"}})

    assert _post(api, approved["id"], "publish-now").status_code == 409
    composio.assert_not_called()


def test_publish_now_on_a_valid_approval_answers_501_in_wave_0(api, monkeypatch):
    from core.composio.tool_executor import ComposioToolExecutor

    composio = MagicMock(name="ComposioToolExecutor.execute")
    monkeypatch.setattr(ComposioToolExecutor, "execute", composio)
    approved = _approved(api)

    resp = _post(api, approved["id"], "publish-now")

    assert resp.status_code == 501
    assert resp.json()["detail"] == "Channel publishing arrives in Wave 3"
    composio.assert_not_called()


def test_an_illegal_transition_is_409(api):
    post = _create(api)
    resp = _post(api, post["id"], "approve", {"content_hash": post["content_hash"]})
    assert resp.status_code == 409 and "draft" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# D6 — the approval binds to the version the reviewer saw (P251-RVW-2)
# ---------------------------------------------------------------------------


def _get(api, post_id):
    resp = api.client.get(f"/api/socials/posts/{post_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _row(api, post_id):
    """The post as committed, read through a session of its own."""
    session = sessionmaker(bind=api.session.get_bind())()
    try:
        return session.get(SocialPost, uuid.UUID(post_id)).to_dict()
    finally:
        session.close()


@pytest.mark.parametrize("body", [{}, {"content_hash": ""}, {"content_hash": "not-a-hash"}, {"content_hash": "A" * 64}])
def test_approve_requires_the_content_hash_the_reviewer_saw(api, body):
    post = _create(api)
    _post(api, post["id"], "submit")
    assert _post(api, post["id"], "approve", body).status_code == 422
    assert _get(api, post["id"])["status"] == "needs_approval"


def test_approve_with_a_hash_the_post_no_longer_has_is_409_and_changes_nothing(api):
    post = _create(api)
    assert _post(api, post["id"], "submit").status_code == 200
    seen = _get(api, post["id"])  # what the reviewer has on screen
    h1 = seen["content_hash"]

    edited = api.client.patch(f"/api/socials/posts/{post['id']}", json={"copy": {"base": "Edited after the reviewer opened it."}})
    assert edited.status_code == 200
    h2 = edited.json()["content_hash"]
    assert edited.json()["status"] == "needs_approval" and h2 != h1
    before = _get(api, post["id"])

    stale = _post(api, post["id"], "approve", {"content_hash": h1})

    assert stale.status_code == 409
    assert stale.json()["detail"]["content_hash"] == h2
    assert "changed since you opened it" in stale.json()["detail"]["message"]
    after = _get(api, post["id"])
    for field in ("status", "approved_hash", "approved_by", "approved_at", "review_log", "content_hash"):
        assert after[field] == before[field], field
    assert after["status"] == "needs_approval" and after["approved_hash"] is None

    current = _post(api, post["id"], "approve", {"content_hash": h2})
    assert current.status_code == 200, current.text
    assert current.json()["status"] == "approved"
    assert current.json()["approved_hash"] == h2 == current.json()["content_hash"]


def test_an_edit_committed_while_the_approve_runs_is_409_and_the_post_is_not_approved(api, monkeypatch):
    """A second session (another worker) commits a PATCH after the approve route
    has loaded the post and before it commits: the compare-and-set refuses it."""
    post = _create(api)
    assert _post(api, post["id"], "submit").status_code == 200
    h1 = _get(api, post["id"])["content_hash"]

    other_worker = sessionmaker(bind=api.session.get_bind())()
    real_get_post = socials_service.get_post
    edits = []

    def load_then_another_worker_edits(db, workspace_id, post_id):
        loaded = real_get_post(db, workspace_id, post_id)
        if not edits:  # once: right after the approve route's load
            row = other_worker.get(SocialPost, post_id)
            socials_service.update_post(row, "editor-2", {"copy": {"base": "Edited by another worker."}})
            edits.append(row.content_hash)
            other_worker.commit()
        return loaded

    monkeypatch.setattr(socials_service, "get_post", load_then_another_worker_edits)
    try:
        resp = _post(api, post["id"], "approve", {"content_hash": h1, "comment": "Looks right"})
    finally:
        other_worker.close()

    assert edits, "the concurrent edit never ran"
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["content_hash"] == edits[0] != h1
    row = _row(api, post["id"])
    assert row["status"] == "needs_approval"
    assert (row["approved_hash"], row["approved_by"], row["approved_at"]) == (None, None, None)
    assert row["content_hash"] == edits[0]
    assert row["copy"] == {"base": "Edited by another worker."}
    assert [entry["action"] for entry in row["review_log"]] == ["submit"]

    # The reviewer, now shown the edit, can approve it.
    resp = _post(api, post["id"], "approve", {"content_hash": edits[0]})
    assert resp.status_code == 200, resp.text
    assert _row(api, post["id"])["approved_hash"] == edits[0]


# ---------------------------------------------------------------------------
# D6/D7 — every write is a compare-and-set (P251-RVW-5)
# ---------------------------------------------------------------------------

EDITED_BY_ANOTHER_WORKER = {"copy": {"base": "Edited by another worker."}}


@contextmanager
def _another_worker_commits_after_the_load(api, monkeypatch, write):
    """The RVW-2 harness for any route: once, right after the route under test
    has loaded the post and before it commits, a second session (another
    worker) applies ``write`` to the post and commits. Yields the content_hash
    that commit left, once it ran."""
    other_worker = sessionmaker(bind=api.session.get_bind())()
    real_get_post = socials_service.get_post
    committed = []

    def load_then_another_worker_writes(db, workspace_id, post_id):
        loaded = real_get_post(db, workspace_id, post_id)
        if not committed:
            row = other_worker.get(SocialPost, post_id)
            write(row)
            content_hash = row.content_hash
            other_worker.commit()
            committed.append(content_hash)
        return loaded

    monkeypatch.setattr(socials_service, "get_post", load_then_another_worker_writes)
    try:
        yield committed
    finally:
        monkeypatch.setattr(socials_service, "get_post", real_get_post)
        other_worker.close()


def _approve_as(reviewer, seen_hash):
    return lambda row: socials_service.approve(row, reviewer, content_hash=seen_hash)


def _edit_as(editor, changes):
    return lambda row: socials_service.update_post(row, editor, changes)


def _actions(row):
    return [entry["action"] for entry in row["review_log"]]


def _assert_the_approval_covers_the_content(row):
    """Never approved or scheduled on content its approval does not cover (D6),
    and never approved with an unsourced claim unless overridden (D7)."""
    if row["status"] in ("approved", "scheduled"):
        assert row["approved_hash"] == row["content_hash"], row
        unsourced = socials_service.unsourced_claims(SimpleNamespace(**row))
        assert not unsourced or row["override_unsourced"], row


def _assert_lost_the_race(resp, current_hash):
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["content_hash"] == current_hash
    assert "changed since you opened it" in detail["message"]


def test_an_edit_racing_a_committed_approve_is_409_and_the_approval_stands(api, monkeypatch):
    """(a) The PATCH loaded the post in needs_approval; an approve committed
    first. The edit must not land under that approval."""
    post = _create(api)
    assert _post(api, post["id"], "submit").status_code == 200
    seen = _get(api, post["id"])["content_hash"]

    with _another_worker_commits_after_the_load(api, monkeypatch, _approve_as("reviewer-2", seen)) as committed:
        resp = api.client.patch(f"/api/socials/posts/{post['id']}", json={"copy": {"base": "An edit nobody reviewed."}})

    assert committed, "the concurrent approve never ran"
    _assert_lost_the_race(resp, seen)
    row = _row(api, post["id"])
    assert row["status"] == "approved" and row["approved_by"] == "reviewer-2"
    assert row["copy"] == post["copy"]
    assert row["approved_hash"] == row["content_hash"] == seen
    assert _actions(row) == ["submit", "approve"]
    _assert_the_approval_covers_the_content(row)

    # Sent again, now against the approved post, the edit voids the approval (D6).
    resp = api.client.patch(f"/api/socials/posts/{post['id']}", json={"copy": {"base": "An edit nobody reviewed."}})
    assert resp.status_code == 200, resp.text
    row = _row(api, post["id"])
    assert row["status"] == "needs_approval" and row["approved_hash"] != row["content_hash"]
    assert _actions(row) == ["submit", "approve", "approval_voided"]


def test_a_schedule_racing_a_committed_edit_is_409_and_the_void_stands(api, monkeypatch):
    """(b) The schedule loaded the approved post; an edit committed first and
    voided the approval. The post must not be scheduled on the edited copy."""
    approved = _approved(api)

    with _another_worker_commits_after_the_load(api, monkeypatch, _edit_as("editor-2", EDITED_BY_ANOTHER_WORKER)) as committed:
        resp = _post(api, approved["id"], "schedule", {"scheduled_for": FUTURE_SLOT, "timezone": "Europe/Lisbon"})

    assert committed, "the concurrent edit never ran"
    _assert_lost_the_race(resp, committed[0])
    assert committed[0] != approved["content_hash"]
    row = _row(api, approved["id"])
    assert row["status"] == "needs_approval"
    assert (row["scheduled_for"], row["timezone"]) == (None, None)
    assert row["content_hash"] == committed[0] and row["copy"] == EDITED_BY_ANOTHER_WORKER["copy"]
    assert _actions(row) == ["submit", "approve", "approval_voided"]
    _assert_the_approval_covers_the_content(row)


def test_an_unschedule_racing_a_committed_edit_is_409_and_the_void_stands(api, monkeypatch):
    """(c) The unschedule loaded the scheduled post; an edit committed first and
    voided the approval. The post must not go back to approved."""
    approved = _approved(api)
    assert _post(api, approved["id"], "schedule", {"scheduled_for": FUTURE_SLOT}).status_code == 200

    with _another_worker_commits_after_the_load(api, monkeypatch, _edit_as("editor-2", EDITED_BY_ANOTHER_WORKER)) as committed:
        resp = _post(api, approved["id"], "unschedule")

    assert committed, "the concurrent edit never ran"
    _assert_lost_the_race(resp, committed[0])
    row = _row(api, approved["id"])
    assert row["status"] == "needs_approval"
    assert row["content_hash"] == committed[0] != row["approved_hash"]
    assert _actions(row) == ["submit", "approve", "schedule", "approval_voided"]
    _assert_the_approval_covers_the_content(row)


def test_a_claim_added_while_an_approve_commits_is_409_and_never_approved_unsourced(api, monkeypatch):
    """(d) The PATCH adding a claim loaded the post in needs_approval; an approve
    committed first. The post must not be approved with an unsourced claim and
    no override (D7)."""
    post = _create(api)
    assert _post(api, post["id"], "submit").status_code == 200
    seen = _get(api, post["id"])["content_hash"]
    claim = {"variables": {"users": {"value": 1200, "claim": True}}}

    with _another_worker_commits_after_the_load(api, monkeypatch, _approve_as("reviewer-2", seen)) as committed:
        resp = api.client.patch(f"/api/socials/posts/{post['id']}", json=claim)

    assert committed, "the concurrent approve never ran"
    _assert_lost_the_race(resp, seen)
    row = _row(api, post["id"])
    assert row["status"] == "approved" and row["override_unsourced"] is False
    assert row["variables"] == {} and row["approved_hash"] == row["content_hash"] == seen
    assert _actions(row) == ["submit", "approve"]
    _assert_the_approval_covers_the_content(row)

    # Sent again, the claim voids the approval, and approving it then needs a source or an override.
    resp = api.client.patch(f"/api/socials/posts/{post['id']}", json=claim)
    assert resp.status_code == 200 and resp.json()["status"] == "needs_approval"
    resp = _post(api, post["id"], "approve", {"content_hash": resp.json()["content_hash"]})
    assert resp.status_code == 422 and resp.json()["detail"]["claims"] == ["users"]


@pytest.mark.parametrize("action, body", [("request-changes", {"comment": "Tighten the hook"}), ("reject", {"reason": "Off brand"})])
def test_a_review_racing_a_committed_approve_is_409_and_the_approval_stands(api, monkeypatch, action, body):
    post = _create(api)
    assert _post(api, post["id"], "submit").status_code == 200
    seen = _get(api, post["id"])["content_hash"]

    with _another_worker_commits_after_the_load(api, monkeypatch, _approve_as("reviewer-2", seen)) as committed:
        resp = _post(api, post["id"], action, body)

    assert committed, "the concurrent approve never ran"
    _assert_lost_the_race(resp, seen)
    row = _row(api, post["id"])
    assert row["status"] == "approved" and row["approved_by"] == "reviewer-2"
    assert _actions(row) == ["submit", "approve"]
    _assert_the_approval_covers_the_content(row)


def test_a_submit_racing_a_committed_edit_is_409_and_writes_nothing(api, monkeypatch):
    post = _create(api)

    with _another_worker_commits_after_the_load(api, monkeypatch, _edit_as("editor-2", EDITED_BY_ANOTHER_WORKER)) as committed:
        resp = _post(api, post["id"], "submit")

    assert committed, "the concurrent edit never ran"
    _assert_lost_the_race(resp, committed[0])
    row = _row(api, post["id"])
    assert row["status"] == "draft" and row["copy"] == EDITED_BY_ANOTHER_WORKER["copy"]
    assert _actions(row) == []

    # Now shown the edit, the author submits it.
    resp = _post(api, post["id"], "submit")
    assert resp.status_code == 200 and resp.json()["status"] == "needs_approval"


def test_schedule_and_unschedule(api):
    approved = _approved(api)
    resp = _post(api, approved["id"], "schedule", {"scheduled_for": FUTURE_SLOT, "timezone": "Europe/Lisbon"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "scheduled" and resp.json()["timezone"] == "Europe/Lisbon"
    assert _post(api, approved["id"], "schedule", {"scheduled_for": FUTURE_SLOT}).status_code == 409
    resp = _post(api, approved["id"], "unschedule")
    assert resp.status_code == 200 and resp.json()["status"] == "approved"
    assert resp.json()["scheduled_for"] is None


def test_schedule_refuses_a_past_slot(api):
    approved = _approved(api)
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert _post(api, approved["id"], "schedule", {"scheduled_for": past}).status_code == 422


# ---------------------------------------------------------------------------
# D7 — unsourced claims
# ---------------------------------------------------------------------------


def test_approve_with_an_unsourced_claim_is_422_naming_it_then_the_override_is_recorded(api):
    post = _create(api, variables={"users": {"value": 1200, "claim": True}})
    assert _post(api, post["id"], "submit").status_code == 200

    resp = _post(api, post["id"], "approve", {"content_hash": post["content_hash"]})
    assert resp.status_code == 422
    assert resp.json()["detail"]["claims"] == ["users"]
    assert "users" in resp.json()["detail"]["message"]

    resp = _post(api, post["id"], "approve", {"content_hash": post["content_hash"], "override_unsourced": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "approved" and body["override_unsourced"] is True
    assert body["review_log"][-1]["overridden_claims"] == ["users"]


# ---------------------------------------------------------------------------
# socials:approve (D6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action, body", [("approve", {}), ("request-changes", {"comment": "No"}), ("reject", {})])
def test_a_viewer_gets_403_on_review_actions(api, action, body):
    post = _create(api)
    _post(api, post["id"], "submit")
    api.role = "viewer"
    resp = _post(api, post["id"], action, body)
    assert resp.status_code == 403 and "socials:approve" in resp.json()["detail"]


def test_an_editor_can_approve(api):
    post = _create(api)
    _post(api, post["id"], "submit")
    api.role = "editor"
    api.ctx = _ctx(WS_A, "editor-1")
    resp = _post(api, post["id"], "approve", {"content_hash": post["content_hash"]})
    assert resp.status_code == 200
    assert resp.json()["approved_by"] == "editor-1"


def test_request_changes_needs_a_comment_and_reject_archives(api):
    post = _create(api)
    _post(api, post["id"], "submit")
    assert _post(api, post["id"], "request-changes", {}).status_code == 422
    resp = _post(api, post["id"], "request-changes", {"comment": "Use the brand colour"})
    assert resp.status_code == 200 and resp.json()["status"] == "changes_requested"
    _post(api, post["id"], "submit")
    resp = _post(api, post["id"], "reject", {"reason": "Off brand"})
    assert resp.status_code == 200 and resp.json()["status"] == "archived"
    assert resp.json()["review_log"][-1]["comment"] == "Off brand"


# ---------------------------------------------------------------------------
# The REAL app: every /api/socials route carries the gate; all are manifested
# ---------------------------------------------------------------------------

_PROBE = r"""
import json
from fastapi.routing import APIRoute
from main import app
from modules.socials.settings import require_socials_enabled

def flatten(dependant, acc):
    for sub in dependant.dependencies:
        if sub.call is not None:
            acc.append(sub.call)
        flatten(sub, acc)

out = []
for route in app.routes:
    path = getattr(route, "path", "") or ""
    if not path.startswith("/api/socials"):
        continue
    calls = []
    if isinstance(route, APIRoute):
        flatten(route.dependant, calls)
    methods = sorted(set(getattr(route, "methods", None) or []) - {"HEAD", "OPTIONS"})
    out.append({"path": path, "methods": methods, "gated": require_socials_enabled in calls})
print("SOCIALS_ROUTES=" + json.dumps(out))
"""


def test_every_socials_route_in_the_real_app_carries_the_gate():
    env = dict(os.environ)
    env.update(
        {
            "POSTGRES_USER": "test",
            "POSTGRES_PASSWORD": "test",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "59432",
            "POSTGRES_DB": "test",
            "DATABASE_URL": "postgresql://test:test@127.0.0.1:59432/test",
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(_ORCH), env=env, capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("SOCIALS_ROUTES="))
    routes = json.loads(line.split("=", 1)[1])

    served = sorted((m, r["path"]) for r in routes for m in r["methods"])
    assert served == _router_routes(), "the app serves /api/socials routes this router does not declare"
    ungated = [r for r in routes if not r["gated"]]
    assert not ungated, f"/api/socials routes without require_socials_enabled: {ungated}"


def test_every_route_is_in_the_committed_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifested = {(r["method"], r["path"]) for r in manifest["routes"]}
    missing = [route for route in _router_routes() if route not in manifested]
    assert not missing, f"add to reports/route-manifest.json: {missing}"
    assert ("PUT", "/api/workspaces/current/socials") in manifested
