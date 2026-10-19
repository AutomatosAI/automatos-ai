"""Clerk production cutover against real Postgres (``@integration``): returning people.

The unique ``users.email`` / ``users.username`` constraints are what turned a new
Clerk user ID into a 500, so this journey runs against the real schema: a person
who has a row (and a workspace) under their development-instance Clerk ID signs in
with a production-instance Clerk ID and the same email. Skips cleanly when no
Postgres is reachable (CI runs it). See test_clerk_cutover_returning_user.py for
the unit-level pins.
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.auth.hybrid import (  # noqa: E402
    _provision_new_user_workspace,
    _resolve_workspace_for_clerk_user,
)
from core.auth.user_link import ensure_user_row  # noqa: E402
from core.database.database import get_database_url  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for tbl in ("users", "workspaces", "workspace_members", "workspace_invitations"):
                c.execute(text(f"SELECT 1 FROM {tbl} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Clerk cutover suite needs a reachable Postgres with the users/workspaces schema: {exc}")
    yield eng
    eng.dispose()


class _People:
    """Creates users and workspaces for one test, and remembers every row it made."""

    def __init__(self, new_session):
        self._new_session = new_session
        self.tag = uuid.uuid4().hex[:10]
        self.users = []
        self.workspaces = []

    def person(self, label: str) -> dict:
        email = f"cutover-{label}-{self.tag}@example.test"
        s = self._new_session()
        uid = s.execute(
            text("INSERT INTO users (username, email, clerk_user_id, is_active) "
                 "VALUES (:e, :e, :cid, true) RETURNING id"),
            {"e": email, "cid": f"user_dev_{label}_{self.tag}"},
        ).scalar()
        s.commit()
        self.users.append(uid)
        return {"id": uid, "email": email, "new_clerk_id": f"user_prod_{label}_{self.tag}"}

    def workspace(self, owner_id, member_id=None, personal=False) -> str:
        ws = str(uuid.uuid4())
        s = self._new_session()
        s.execute(
            text("INSERT INTO workspaces (id, name, slug, owner_id, is_personal, is_active) "
                 "VALUES (CAST(:id AS uuid), :n, :n, :owner, :personal, true)"),
            {"id": ws, "n": f"cutover-{self.tag}-{len(self.workspaces)}", "owner": owner_id, "personal": personal},
        )
        if member_id is not None:
            s.execute(
                text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active, joined_at) "
                     "VALUES (CAST(:ws AS uuid), :uid, 'editor', true, NOW())"),
                {"ws": ws, "uid": member_id},
            )
        s.commit()
        self.workspaces.append(ws)
        return ws

    def sweep(self):
        s = self._new_session.sweep()
        params = {"users": self.users, "ws": self.workspaces}
        s.execute(text("DELETE FROM workspace_members "
                       "WHERE user_id = ANY(:users) OR CAST(workspace_id AS text) = ANY(:ws)"), params)
        s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) = ANY(:ws) OR owner_id = ANY(:users)"), params)
        s.execute(text("DELETE FROM users WHERE id = ANY(:users)"), params)
        s.commit()


@pytest.fixture
def people(new_session):
    made = _People(new_session)
    yield made
    made.sweep()


def _clerk_id_of(session, user_id):
    return session.execute(text("SELECT clerk_user_id FROM users WHERE id = :id"), {"id": user_id}).scalar()


def _rows_with_email(session, email):
    return session.execute(text("SELECT COUNT(*) FROM users WHERE LOWER(email) = LOWER(:e)"), {"e": email}).scalar()


def _owned_workspaces(session, user_id):
    return session.execute(text("SELECT COUNT(*) FROM workspaces WHERE owner_id = :id"), {"id": user_id}).scalar()


def test_a_returning_owner_lands_in_their_own_workspace_relinked(people, new_session):
    owner = people.person("owner")
    theirs = people.workspace(owner["id"], personal=True)

    db = new_session()
    resolved = _resolve_workspace_for_clerk_user(
        db, clerk_user_id=owner["new_clerk_id"], org_id=None, email=owner["email"].upper(),
    )

    check = new_session()
    assert str(resolved) == theirs
    assert _clerk_id_of(check, owner["id"]) == owner["new_clerk_id"], "the relink is committed"
    assert _rows_with_email(check, owner["email"]) == 1
    assert _owned_workspaces(check, owner["id"]) == 1, "no second personal workspace"


def test_a_returning_member_lands_in_the_workspace_they_joined(people, new_session):
    inviter = people.person("inviter")
    member = people.person("member")
    team = people.workspace(inviter["id"], member_id=member["id"])

    db = new_session()
    resolved = _resolve_workspace_for_clerk_user(
        db, clerk_user_id=member["new_clerk_id"], org_id=None, email=member["email"],
    )

    check = new_session()
    assert str(resolved) == team
    assert _clerk_id_of(check, member["id"]) == member["new_clerk_id"]
    assert _owned_workspaces(check, member["id"]) == 0, "a member is not given a personal workspace"


def test_provisioning_a_returning_owner_is_not_a_500(people, new_session):
    """The report's crash site: _provision_new_user_workspace with a taken email."""
    owner = people.person("direct")
    theirs = people.workspace(owner["id"], personal=True)

    db = new_session()
    provisioned = _provision_new_user_workspace(db, owner["new_clerk_id"], email=owner["email"], name="Direct")

    check = new_session()
    assert str(provisioned) == theirs
    assert _clerk_id_of(check, owner["id"]) == owner["new_clerk_id"]
    assert _rows_with_email(check, owner["email"]) == 1


def test_a_new_person_gets_one_row_however_often_they_arrive(people, new_session):
    email = f"cutover-newcomer-{people.tag}@example.test"
    clerk_id = f"user_prod_newcomer_{people.tag}"

    db = new_session()
    first = ensure_user_row(db, clerk_id, email=email, name="Newcomer")
    people.users.append(first)
    second = ensure_user_row(db, clerk_id, email=email, name="Newcomer")
    db.commit()

    check = new_session()
    assert first == second
    assert _rows_with_email(check, email) == 1
    assert _clerk_id_of(check, first) == clerk_id
