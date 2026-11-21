"""PRD-277 US-408 (§5.13) against Postgres: a membership's rules live in the people service.

``core.auth.people`` owns them, not the Team routes (``api/team.py``), each test in a transaction
rolled back at the end:

* **Workspace scoping.** ``change_member_role``, ``member_to_remove`` and ``end_membership`` find the
  membership in the workspace named only; one of another workspace is ``MemberNotFound`` and is
  left exactly as it was.
* **One owner stays one owner.** The owner's role is never changed, nobody is made owner (in any
  case or spacing), only a role below owner is set, and the owner's membership never ends
  (``MembershipRefused``, nothing changed).
* **The routes keep their answers.** The same refusals reach the Team page as before: 404 ``Member
  not found``, 400 with the owner messages and ``Invalid role``.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.auth import people  # noqa: E402


@pytest.fixture
def db_session(test_engine):
    """A session whose commit releases a savepoint and whose rollback returns to one."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _person(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
                      {"u": f"prd277-w4b-{tag}", "e": f"prd277-w4b-{tag}@example.test"}).scalar_one()


def _member(db, workspace_id, user_id, role) -> int:
    return db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                           "VALUES (CAST(:ws AS uuid), :u, :r, true) RETURNING id"),
                      {"ws": workspace_id, "u": user_id, "r": role}).scalar_one()


@pytest.fixture
def team(db_session, seed_workspace):
    """Two workspaces: ours with an owner and an editor, and another with an editor of its own."""
    ours, theirs = seed_workspace(), seed_workspace()
    owner, editor, outsider = _person(db_session), _person(db_session), _person(db_session)
    members = NS(owner=_member(db_session, ours, owner, "owner"), editor=_member(db_session, ours, editor, "editor"),
                 theirs=_member(db_session, theirs, outsider, "editor"))
    db_session.commit()
    return NS(db=db_session, ours=ours, theirs=theirs, members=members, owner=owner)


def _state(db, member_id):
    db.expire_all()
    return tuple(db.execute(text("SELECT role, is_active FROM workspace_members WHERE id = :m"),
                            {"m": member_id}).one())


# --------------------------------------------------------------------------- the service


def test_a_role_changes_only_for_a_member_of_the_workspace_named(team):
    assert people.change_member_role(team.db, team.ours, team.members.editor, "viewer") == "editor"
    assert _state(team.db, team.members.editor) == ("viewer", True)

    with pytest.raises(people.MemberNotFound):
        people.change_member_role(team.db, team.ours, team.members.theirs, "admin")
    assert _state(team.db, team.members.theirs) == ("editor", True)


@pytest.mark.parametrize(("target", "role", "message"), [
    ("owner", "admin", people.OWNER_ROLE_FIXED_MESSAGE),
    ("editor", "owner", people.ONE_OWNER_MESSAGE),
    ("editor", " Owner ", people.ONE_OWNER_MESSAGE),
    ("editor", "super_admin", "Invalid role: super_admin"),
])
def test_the_one_owner_stays_the_one_owner(team, target, role, message):
    member_id = getattr(team.members, target)
    before = _state(team.db, member_id)

    with pytest.raises(people.MembershipRefused) as refused:
        people.change_member_role(team.db, team.ours, member_id, role)

    assert str(refused.value) == message and _state(team.db, member_id) == before


def test_a_membership_ends_only_in_the_workspace_named_and_never_the_owners(team):
    with pytest.raises(people.MembershipRefused) as refused:
        people.end_membership(team.db, team.ours, team.members.owner)
    assert str(refused.value) == people.OWNER_NOT_REMOVABLE_MESSAGE
    with pytest.raises(people.MemberNotFound):
        people.end_membership(team.db, team.ours, team.members.theirs)
    assert _state(team.db, team.members.owner) == ("owner", True)
    assert _state(team.db, team.members.theirs) == ("editor", True)

    ended = people.end_membership(team.db, team.ours, team.members.editor)
    assert ended.id == team.members.editor and _state(team.db, team.members.editor) == ("editor", False)


# --------------------------------------------------------------------------- the routes keep their answers


def _change(team, member_id, role):
    import api.team as team_api

    with patch.object(team_api, "_resolve_internal_user_id", return_value=team.owner):
        return team_api.update_member_role(workspace_id=team.ours, member_id=member_id, ctx=NS(), db=team.db,
                                           request=team_api.UpdateMemberRoleRequest(role=role))


def _remove(team, member_id):
    import api.team as team_api

    with patch.object(team_api, "_resolve_internal_user_id", return_value=team.owner):
        return asyncio.run(team_api.remove_member(workspace_id=team.ours, member_id=member_id,
                                                  ctx=NS(user=NS(org_id=None)), db=team.db))


@pytest.mark.parametrize(("target", "role", "status", "detail"), [
    ("theirs", "viewer", 404, "Member not found"),
    ("owner", "viewer", 400, "The owner's role cannot be changed."),
    ("editor", "owner", 400, "A workspace has one owner: make them an admin, editor or viewer."),
    ("editor", "boss", 400, "Invalid role: boss"),
])
def test_a_role_change_route_answers_as_it_did(team, target, role, status, detail):
    with pytest.raises(HTTPException) as refused:
        _change(team, getattr(team.members, target), role)
    assert (refused.value.status_code, refused.value.detail) == (status, detail)


@pytest.mark.parametrize(("target", "status", "detail"), [
    ("theirs", 404, "Member not found"),
    ("owner", 400, "Cannot remove workspace owner"),
])
def test_a_removal_route_answers_as_it_did(team, target, status, detail):
    with pytest.raises(HTTPException) as refused:
        _remove(team, getattr(team.members, target))
    assert (refused.value.status_code, refused.value.detail) == (status, detail)
    assert _state(team.db, getattr(team.members, target))[1] is True


def test_the_routes_still_change_and_remove_an_ordinary_member(team):
    assert _change(team, team.members.editor, "admin") == {"status": "success", "new_role": "admin"}
    assert _remove(team, team.members.editor) == {"status": "success"}
    assert _state(team.db, team.members.editor) == ("admin", False)
