"""PRD-277 US-202 joined to Wave 3: a signed-in person passes the workspace gates by ``users.id``.

Against Postgres, through the real session lane (``hybrid.get_request_context_hybrid`` with an
``ams_`` token ``issue_session`` made) and then the real gates (``workspace_permission`` and
``workspace_admin``), each test in a transaction rolled back at the end:

* an editor member gets an editor's permissions and not an admin's;
* an owner with no member row is the owner (the lane admits them, as the Clerk lane does);
* a person in no workspace is ``403 workspace_required`` at the lane, before any gate;
* a deactivated person is ``401 session_ended`` (``revoked``);
* the gates read only the ``users.id`` the lane put on the context: a session context without it
  is refused, whatever its email says;
* the Clerk and device answers are unchanged: a Clerk caller is read by its Clerk user ID even
  when its claims name someone else, and a paired device by its bound person.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.auth import hybrid, session_lane, sessions, workspace_admin  # noqa: E402
from core.auth.workspace_permission import resolve_workspace_role, workspace_permission_granted  # noqa: E402

INSTALL = "https://automatos.example.test"
EDITOR_ONLY, ADMIN_ONLY = "agents:create", "members:invite"


class _BorrowedSession:
    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


@pytest.fixture
def db_session(test_engine):
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _person(db, clerk_id=None) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(text("INSERT INTO users (username, email, is_active, clerk_user_id) "
                           "VALUES (:u, :e, true, :c) RETURNING id"),
                      {"u": f"prd277-w4-{tag}", "e": f"prd277-w4-{tag}@example.test", "c": clerk_id}).scalar_one()


def _member(db, workspace_id, user_id, role) -> None:
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :u, :r, true)"), {"ws": str(workspace_id), "u": user_id, "r": role})


@pytest.fixture
def team(db_session, seed_workspace, monkeypatch):
    """One workspace: an editor member, an owner with no member row, and a person in no workspace."""
    workspace_id = uuid.UUID(seed_workspace())
    editor, owner, outsider = _person(db_session), _person(db_session), _person(db_session)
    _member(db_session, workspace_id, editor, "editor")
    db_session.execute(text("UPDATE workspaces SET owner_id = :o WHERE id = CAST(:ws AS uuid)"),
                       {"o": owner, "ws": str(workspace_id)})
    db_session.commit()
    for name, value in {"AUTH_EDITION": "enterprise", "REQUIRE_AUTH": True, "FRONTEND_URL": INSTALL,
                        "DEFAULT_WORKSPACE_ID": str(workspace_id)}.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(session_lane, "SessionLocal", lambda: _BorrowedSession(db_session))
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    return SimpleNamespace(ws=workspace_id, editor=editor, owner=owner, outsider=outsider, db=db_session)


async def _context(db, user_id):
    issued = sessions.issue_session(db, user_id=user_id, auth_time=datetime.now(timezone.utc))
    request = SimpleNamespace(method="GET", headers={"authorization": "Bearer " + issued.token}, query_params={},
                              state=SimpleNamespace())
    return await hybrid.get_request_context_hybrid(request)


async def _refused(db, user_id) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        await _context(db, user_id)
    return refused.value


@pytest.mark.asyncio
async def test_an_editor_member_has_an_editors_permissions_and_not_an_admins(team):
    ctx = await _context(team.db, team.editor)

    assert ctx.auth_type == session_lane.SESSION_AUTH_TYPE and resolve_workspace_role(team.db, ctx) == "editor"
    assert workspace_permission_granted(team.db, ctx, EDITOR_ONLY) is True
    assert workspace_permission_granted(team.db, ctx, ADMIN_ONLY) is False
    assert workspace_admin.may_answer_workspace_questions(team.db, ctx) is True
    assert workspace_admin.may_see_own_workspace_health(team.db, ctx) is False


@pytest.mark.asyncio
async def test_an_owner_with_no_member_row_is_the_owner(team):
    ctx = await _context(team.db, team.owner)

    assert ctx.workspace_id == team.ws and resolve_workspace_role(team.db, ctx) == "owner"
    assert workspace_permission_granted(team.db, ctx, ADMIN_ONLY) is True
    assert workspace_admin.may_see_own_workspace_health(team.db, ctx) is True


@pytest.mark.asyncio
async def test_a_person_in_no_workspace_is_told_so_before_any_gate(team):
    refused = await _refused(team.db, team.outsider)
    assert refused.status_code == 403 and refused.detail["code"] == "workspace_required"


@pytest.mark.asyncio
async def test_a_deactivated_person_is_refused_on_their_live_session(team):
    issued = sessions.issue_session(team.db, user_id=team.editor, auth_time=datetime.now(timezone.utc))
    team.db.execute(text("UPDATE users SET is_active = false WHERE id = :u"), {"u": team.editor})
    team.db.commit()
    request = SimpleNamespace(method="GET", headers={"authorization": "Bearer " + issued.token}, query_params={},
                              state=SimpleNamespace())

    with pytest.raises(HTTPException) as refused:
        await hybrid.get_request_context_hybrid(request)

    assert refused.value.status_code == 401 and refused.value.detail["reason"] == "revoked"


def _email_of(db, user_id) -> str:
    return db.execute(text("SELECT email FROM users WHERE id = :u"), {"u": user_id}).scalar_one()


def test_a_session_context_is_read_only_by_the_users_id_the_lane_bound(team):
    unbound = SimpleNamespace(auth_type=session_lane.SESSION_AUTH_TYPE, workspace_id=team.ws,
                              user=SimpleNamespace(id=_email_of(team.db, team.editor), email=_email_of(team.db, team.editor),
                                                   clerk_user_id=None, system_role="user", raw_claims={}))

    assert resolve_workspace_role(team.db, unbound) is None
    assert workspace_permission_granted(team.db, unbound, EDITOR_ONLY) is False
    assert workspace_admin.may_answer_workspace_questions(team.db, unbound) is False


def test_a_clerk_caller_is_still_read_by_its_clerk_id_whatever_its_claims_say(team):
    clerk_id = "user_prd277_w4_" + uuid.uuid4().hex[:10]
    viewer = _person(team.db, clerk_id)
    _member(team.db, team.ws, viewer, "viewer")
    forged = {"source": "auth_session", "user_id": team.owner}
    ctx = SimpleNamespace(auth_type="clerk", workspace_id=team.ws,
                          user=SimpleNamespace(clerk_user_id=clerk_id, system_role="user", raw_claims=forged))

    assert resolve_workspace_role(team.db, ctx) == "viewer"
    assert workspace_permission_granted(team.db, ctx, EDITOR_ONLY) is False
    assert workspace_admin.may_see_own_workspace_health(team.db, ctx) is False


def test_a_paired_device_is_still_its_bound_person(team):
    ctx = SimpleNamespace(auth_type="device", workspace_id=team.ws,
                          user=SimpleNamespace(clerk_user_id=None, system_role="user", email=None,
                                               raw_claims={"source": "mobile_device", "user_id": team.editor}))

    assert resolve_workspace_role(team.db, ctx) == "editor"
    assert workspace_permission_granted(team.db, ctx, EDITOR_ONLY) is True
    assert workspace_admin.may_answer_workspace_questions(team.db, ctx) is True
