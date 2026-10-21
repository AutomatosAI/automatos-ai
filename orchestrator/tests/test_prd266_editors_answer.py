"""PRD-266 US-006a — editors answer (D16): ``require_workspace_answerer`` on the approval-grant verbs.

Gerard's decision of 20 Oct (D16, #1253) reverses PRD-196's owner/admin posture for answering: an
owner, admin or editor of the workspace lists its approval grants, answers its questions and grants
or denies its approvals; a viewer is refused each (403) and nothing changes. Revoking stays an
owner's or admin's (``require_workspace_admin``): an editor's revoke is 403. Against Postgres:

* the routes, through the real ``api/approval_grants.py`` router, for a Clerk-shaped caller of each
  role (the caller stands in for the hybrid dependency; the gate is real and reads the member rows).
  What a click runs beyond its grant row (the resume, the subject's failure, the chat confirmation,
  the audit) is recorded instead; the grant's own write is real;
* a paired device by its bound person's role (``core/auth/device.py``'s context, read by
  ``users.id``): an editor's phone answers, a viewer's is refused;
* the principals that never answer (API key, SDK key, the saas anonymous principal) and the one that
  always does (the super-admin, as the local operator is), decided without a query;
* Needs-you (``GET /api/activity/needs-you`` and ``GET /api/activity/stats``'s ``needs_attention``):
  an editor's counts include the open question and approval, a viewer's are 0 for both.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.auth import device  # noqa: E402
from core.auth import workspace_admin as admin  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.approval_grants import KIND_APPROVAL, KIND_QUESTION, ApprovalGrant  # noqa: E402

GRANTS = "/api/v1/approval-grants"
OPTIONS = ["Rathmines", "Ranelagh"]
ADMIN_ONLY = "Workspace admin only"


class _NoQueries:
    """A session that fails the test if anything is read: a decision that needs no database."""

    def execute(self, *args, **kwargs):
        raise AssertionError("this decision must not query the database")

    def query(self, *args, **kwargs):
        raise AssertionError("this decision must not query the database")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here).

    The routes commit; here that reaches the last savepoint only, and every seed commits
    (``_settle``), so no route can take the seed with it. Everything rolls back at the end.
    """
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _settle(db) -> None:
    """Release the seed's savepoint: a route's rollback never reaches it."""
    db.commit()


@pytest.fixture
def ws(seed_workspace, db_session) -> uuid.UUID:
    workspace_id = uuid.UUID(seed_workspace())
    _settle(db_session)
    return workspace_id


@pytest.fixture
def ran(monkeypatch):
    """What a click runs beyond its grant row, recorded: the resume and the subject's failure."""
    from api import approval_grants

    seen = SimpleNamespace(resumed=[], failed=[])

    async def _requeue(db, grant):
        seen.resumed.append(grant.id)
        return False

    monkeypatch.setattr(approval_grants, "_requeue_subject", _requeue)
    monkeypatch.setattr(approval_grants, "_fail_subject", lambda db, grant: seen.failed.append(grant.id))
    monkeypatch.setattr(approval_grants, "_confirm_answer_into_chat", lambda db, grant, *, resumed: None)
    monkeypatch.setattr(approval_grants, "_audit", lambda db, ctx, action, grant: None)
    return seen


def _person(db, workspace_id=None, role=None, *, active=True) -> SimpleNamespace:
    """A users row with a Clerk id; a ``role`` member of ``workspace_id`` when a role is given."""
    tag = uuid.uuid4().hex[:12]
    email, clerk_id = f"prd266-editors-{tag}@t.test", f"user_prd266_editors_{tag}"
    pk = db.execute(
        text("INSERT INTO users (username, email, is_active, clerk_user_id) VALUES (:u, :e, true, :c) RETURNING id"),
        {"u": f"prd266-editors-{tag}", "e": email, "c": clerk_id},
    ).scalar()
    if role:
        db.execute(
            text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                 "VALUES (CAST(:ws AS uuid), :uid, :role, :active)"),
            {"ws": str(workspace_id), "uid": pk, "role": role, "active": active},
        )
    _settle(db)
    return SimpleNamespace(pk=pk, email=email, clerk_id=clerk_id)


def _owns(db, workspace_id, person) -> None:
    db.execute(text("UPDATE workspaces SET owner_id = :uid WHERE id = CAST(:ws AS uuid)"),
               {"uid": person.pk, "ws": str(workspace_id)})
    _settle(db)


def _clerk_ctx(workspace_id, person) -> RequestContext:
    """The context the Clerk lane binds for ``person``."""
    user = UserContext(id=person.clerk_id, email=person.email, system_role="user", clerk_user_id=person.clerk_id)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="clerk")


def _device_ctx(workspace_id, person) -> RequestContext:
    """A paired device's context exactly as the lane binds it (``device._user_context``)."""
    user = device._user_context(SimpleNamespace(id=uuid.uuid4()), (person.pk, person.email))
    return RequestContext(workspace_id=workspace_id, user=user, auth_type=device.DEVICE_AUTH_TYPE)


def _grant(db, workspace_id, kind) -> int:
    """A pending question (a parked call's) or approval (a board ticket's) in ``workspace_id``."""
    asks = kind == KIND_QUESTION
    grant = ApprovalGrant(
        workspace_id=workspace_id, kind=kind, status="pending",
        subject_type="tool_call" if asks else "board_task", subject_id=f"prd266-editors-{uuid.uuid4().hex[:8]}",
        question_md="Which café opens first on Sundays?" if asks else None, options=OPTIONS if asks else None,
        reason=None if asks else "Send the price list to 40 cafés",
    )
    db.add(grant)
    _settle(db)
    return grant.id


def _statuses(db, *grant_ids) -> list:
    db.expire_all()
    return [db.get(ApprovalGrant, grant_id).status for grant_id in grant_ids]


def _client(db, ctx, *modules) -> TestClient:
    """A client on ``modules``' routers (the approval grants by default) as ``ctx``."""
    from api import approval_grants

    app = FastAPI()
    for module in modules or (approval_grants,):
        app.include_router(module.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


# --------------------------------------------------------------------------- the routes


@pytest.mark.parametrize("role", ["owner", "admin", "editor"])
def test_an_owner_admin_or_editor_lists_answers_grants_and_denies(ws, db_session, ran, role):
    client = _client(db_session, _clerk_ctx(ws, _person(db_session, ws, role)))
    question, approval, declined = (_grant(db_session, ws, KIND_QUESTION), _grant(db_session, ws, KIND_APPROVAL),
                                    _grant(db_session, ws, KIND_APPROVAL))

    listed = client.get(GRANTS)
    answered = client.post(f"{GRANTS}/{question}/answer", json={"option": OPTIONS[1]})
    granted = client.post(f"{GRANTS}/{approval}/grant")
    denied = client.post(f"{GRANTS}/{declined}/deny")

    assert [r.status_code for r in (listed, answered, granted, denied)] == [200] * 4, \
        [r.text for r in (listed, answered, granted, denied)]
    assert {question, approval, declined} <= {g["id"] for g in listed.json()["grants"]}
    assert _statuses(db_session, question, approval, declined) == ["granted", "granted", "denied"]
    assert (ran.resumed, ran.failed) == ([question, approval], [declined])


def test_a_viewer_is_refused_each_and_nothing_changes(ws, db_session, ran):
    client = _client(db_session, _clerk_ctx(ws, _person(db_session, ws, "viewer")))
    question, approval = _grant(db_session, ws, KIND_QUESTION), _grant(db_session, ws, KIND_APPROVAL)

    refused = [client.get(GRANTS), client.post(f"{GRANTS}/{question}/answer", json={"option": OPTIONS[0]}),
               client.post(f"{GRANTS}/{approval}/grant"), client.post(f"{GRANTS}/{approval}/deny")]

    assert [(r.status_code, r.json()["detail"]) for r in refused] == [(403, admin.ANSWERER_ONLY)] * 4
    assert _statuses(db_session, question, approval) == ["pending", "pending"]
    assert (ran.resumed, ran.failed) == ([], [])


def test_an_editors_revoke_is_refused_and_an_editor_is_still_no_workspace_admin(ws, db_session, ran):
    editor = _person(db_session, ws, "editor")
    approval = _grant(db_session, ws, KIND_APPROVAL)

    response = _client(db_session, _clerk_ctx(ws, editor)).post(f"{GRANTS}/{approval}/revoke")

    assert (response.status_code, response.json()["detail"]) == (403, ADMIN_ONLY)
    assert _statuses(db_session, approval) == ["pending"]
    assert admin.may_see_own_workspace_health(db_session, _clerk_ctx(ws, editor)) is False


@pytest.mark.parametrize("role", ["owner", "admin"])
def test_an_owner_or_admin_still_revokes(ws, db_session, ran, role):
    approval = _grant(db_session, ws, KIND_APPROVAL)

    response = _client(db_session, _clerk_ctx(ws, _person(db_session, ws, role))).post(f"{GRANTS}/{approval}/revoke")

    assert response.status_code == 200, response.text
    assert _statuses(db_session, approval) == ["revoked"]


def test_the_workspaces_owner_answers_without_a_member_row(ws, db_session, ran):
    owner = _person(db_session)
    _owns(db_session, ws, owner)
    question = _grant(db_session, ws, KIND_QUESTION)

    response = _client(db_session, _clerk_ctx(ws, owner)).post(f"{GRANTS}/{question}/answer",
                                                                json={"answer_text": "Ranelagh, at nine"})

    assert response.status_code == 200, response.text
    assert _statuses(db_session, question) == ["granted"]


def test_a_lapsed_editor_and_another_workspaces_editor_are_refused(ws, db_session, seed_workspace, ran):
    lapsed = _person(db_session, ws, "editor", active=False)
    elsewhere = _person(db_session, uuid.UUID(seed_workspace()), "editor")

    for person in (lapsed, elsewhere):
        assert _client(db_session, _clerk_ctx(ws, person)).get(GRANTS).status_code == 403
        assert admin.may_answer_workspace_questions(db_session, _clerk_ctx(ws, person)) is False


def test_an_editor_never_reaches_another_workspaces_question(ws, db_session, seed_workspace, ran):
    elsewhere = uuid.UUID(seed_workspace())
    _settle(db_session)
    theirs = _grant(db_session, elsewhere, KIND_QUESTION)

    response = _client(db_session, _clerk_ctx(ws, _person(db_session, ws, "editor"))).post(
        f"{GRANTS}/{theirs}/answer", json={"option": OPTIONS[0]})

    assert response.status_code == 404
    assert _statuses(db_session, theirs) == ["pending"]


def test_a_clerk_token_carrying_a_devices_claims_names_nobody(ws, db_session):
    """The users.id claim is the device lane's alone: a Clerk context with an editor's
    ``raw_claims`` and no membership of its own is refused (``principal.claimed_user_pk``)."""
    editor, stranger = _person(db_session, ws, "editor"), _person(db_session)
    forged = UserContext(id=stranger.clerk_id, email=stranger.email, system_role="user",
                         clerk_user_id=stranger.clerk_id, raw_claims=_device_ctx(ws, editor).user.raw_claims)

    assert admin.may_answer_workspace_questions(
        db_session, RequestContext(workspace_id=ws, user=forged, auth_type="clerk")) is False


# --------------------------------------------------------------------------- a paired device


def test_a_device_bound_to_an_editor_answers_and_one_bound_to_a_viewer_is_refused(ws, db_session, ran):
    editor, viewer = _person(db_session, ws, "editor"), _person(db_session, ws, "viewer")
    question = _grant(db_session, ws, KIND_QUESTION)
    editors_phone, viewers_phone = (_client(db_session, _device_ctx(ws, editor)),
                                    _client(db_session, _device_ctx(ws, viewer)))

    assert (viewers_phone.get(GRANTS).status_code, editors_phone.get(GRANTS).status_code) == (403, 200)
    refused = viewers_phone.post(f"{GRANTS}/{question}/answer", json={"option": OPTIONS[0]})
    answered = editors_phone.post(f"{GRANTS}/{question}/answer", json={"option": OPTIONS[1]})

    assert (refused.status_code, answered.status_code) == (403, 200), answered.text
    db_session.expire_all()
    row = db_session.get(ApprovalGrant, question)
    assert (row.status, row.answer_text) == ("granted", OPTIONS[1])


def test_a_device_whose_person_is_not_found_is_refused(ws, db_session):
    nobody = SimpleNamespace(pk=-1, email=f"prd266-editors-nobody-{uuid.uuid4().hex[:8]}@t.test")

    assert admin.may_answer_workspace_questions(db_session, _device_ctx(ws, nobody)) is False


# --------------------------------------------------------------------------- no workspace role


def _never_answer() -> list:
    ws_id = uuid.uuid4()
    return [
        RequestContext(workspace_id=ws_id, user=UserContext(id="api_key", system_role="admin"), auth_type="api_key"),
        RequestContext(workspace_id=ws_id, user=UserContext(id="sdk:7", system_role="service"), auth_type="sdk_key"),
        RequestContext(workspace_id=ws_id, user=UserContext(), auth_type="anonymous"),
    ]


@pytest.mark.parametrize("ctx", _never_answer(), ids=["api_key", "sdk_key", "saas_anonymous"])
def test_api_keys_sdk_keys_and_the_saas_anonymous_never_answer(ctx):
    assert admin.may_answer_workspace_questions(_NoQueries(), ctx) is False
    with pytest.raises(HTTPException) as refused:
        asyncio.run(admin.require_workspace_answerer(ctx=ctx, db=_NoQueries()))
    assert (refused.value.status_code, refused.value.detail) == (403, admin.ANSWERER_ONLY)


def test_the_super_admin_answers_as_the_local_operator_does():
    operator = UserContext(email="operator@automatos.local", system_role="super_admin")
    ctx = RequestContext(workspace_id=uuid.uuid4(), user=operator, auth_type="anonymous")

    assert asyncio.run(admin.require_workspace_answerer(ctx=ctx, db=_NoQueries())) is ctx


# --------------------------------------------------------------------------- Needs-you


@pytest.fixture
def quiet_stats(monkeypatch):
    """The stats' other counters at 0, so ``needs_attention`` is read alone (the PRD-252 R5 test's)."""
    from services.activity_service import ActivityService

    for counter in ("_count_working_now", "_count_channels_live", "_count_agents_active", "_count_tasks_in_queue"):
        monkeypatch.setattr(ActivityService, counter, lambda self: 0)
    monkeypatch.setattr(ActivityService, "_count_completed", lambda self, since: 0)


def test_needs_you_counts_questions_and_approvals_for_an_editor_and_not_for_a_viewer(ws, db_session, quiet_stats):
    from api import activity

    _grant(db_session, ws, KIND_QUESTION)
    _grant(db_session, ws, KIND_APPROVAL)
    seen = {}
    for role in ("editor", "viewer"):
        client = _client(db_session, _clerk_ctx(ws, _person(db_session, ws, role)), activity)
        needs_you, stats = client.get("/api/activity/needs-you"), client.get("/api/activity/stats?period=1d")
        assert (needs_you.status_code, stats.status_code) == (200, 200), (needs_you.text, stats.text)
        counts = needs_you.json()["counts"]
        seen[role] = (counts["question"], counts["approval"], stats.json()["needs_attention"])

    assert seen == {"editor": (1, 1, 2), "viewer": (0, 0, 0)}
