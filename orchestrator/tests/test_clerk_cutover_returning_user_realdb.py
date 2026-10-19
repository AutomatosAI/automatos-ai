"""Clerk production cutover against real Postgres (``@integration``): returning people.

The unique ``users.email`` / ``users.username`` constraints are what turned a new
Clerk user ID into a 500, so this journey runs against the real schema: a person
who has a row (and a workspace) under their development-instance Clerk ID comes
back with a production-instance Clerk ID and the same email.

- Before the cutover step, sign-in gives them a 409: no 500, no second row or
  workspace, and their row stays on its old Clerk ID.
- The cutover step (``core/auth/clerk_cutover.py``) moves the row once, on the
  import's ``external_id`` link and a verified email, and records it in
  ``clerk_user_relinks``. After it they land in the workspace they had.
- A second run moves nothing; a rollback moves the row back and a retried cutover
  re-applies the same pair, still one record.

Skips cleanly when no Postgres is reachable (CI runs it). The unit-level pins are in
test_clerk_cutover_returning_user.py and test_clerk_cutover_relink.py.
"""
from __future__ import annotations

import os
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.auth import clerk_cutover as cutover  # noqa: E402
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
            for tbl in ("users", "workspaces", "workspace_members", "workspace_invitations", "clerk_user_relinks"):
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

    def person(self, label: str, email: str = None) -> dict:
        email = email or f"cutover-{label}-{self.tag}@example.test"
        s = self._new_session()
        uid = s.execute(
            text("INSERT INTO users (username, email, clerk_user_id, is_active) "
                 "VALUES (:e, :e, :cid, true) RETURNING id"),
            {"e": email, "cid": f"user_dev_{label}_{self.tag}"},
        ).scalar()
        s.commit()
        self.users.append(uid)
        return {
            "id": uid, "email": email,
            "old_clerk_id": f"user_dev_{label}_{self.tag}", "new_clerk_id": f"user_prod_{label}_{self.tag}",
        }

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
        s.execute(text("DELETE FROM clerk_user_relinks WHERE user_id = ANY(:users)"), params)
        s.execute(text("DELETE FROM users WHERE id = ANY(:users)"), params)
        s.commit()


@pytest.fixture
def people(new_session):
    made = _People(new_session)
    yield made
    made.sweep()


def _production_user(person, verified=True) -> cutover.ProductionUser:
    emails = frozenset({person["email"].lower()}) if verified else frozenset()
    return cutover.ProductionUser(person["new_clerk_id"], person["old_clerk_id"], emails)


def _run_cutover(new_session, *prod_users):
    db = new_session()
    decisions = cutover.plan_relinks(db, list(prod_users))
    moved = cutover.apply_relinks(db, decisions)
    db.commit()
    return decisions, moved


def _run_rollback(new_session, user_id):
    """Roll back this test's person only: the plan covers every unreverted record."""
    db = new_session()
    mine = [d for d in cutover.plan_rollback(db) if d.user_id == user_id]
    moved = cutover.apply_rollback(db, mine)
    db.commit()
    return moved


def _clerk_id_of(session, user_id):
    return session.execute(text("SELECT clerk_user_id FROM users WHERE id = :id"), {"id": user_id}).scalar()


def _rows_with_email(session, email):
    return session.execute(text("SELECT COUNT(*) FROM users WHERE LOWER(email) = LOWER(:e)"), {"e": email}).scalar()


def _owned_workspaces(session, user_id):
    return session.execute(text("SELECT COUNT(*) FROM workspaces WHERE owner_id = :id"), {"id": user_id}).scalar()


def _records(session, user_id):
    return session.execute(
        text("SELECT old_clerk_user_id, new_clerk_user_id, verified_email, reverted_at IS NOT NULL "
             "FROM clerk_user_relinks WHERE user_id = :id"),
        {"id": user_id},
    ).fetchall()


def _sign_in(new_session, person, email=None):
    return _resolve_workspace_for_clerk_user(
        new_session(), clerk_user_id=person["new_clerk_id"], org_id=None, email=email or person["email"],
    )


def test_before_the_cutover_step_a_returning_owner_gets_a_409_and_nothing_moves(people, new_session):
    owner = people.person("early")
    people.workspace(owner["id"], personal=True)

    with pytest.raises(HTTPException) as raised:
        _sign_in(new_session, owner, email=owner["email"].upper())

    check = new_session()
    assert raised.value.status_code == 409
    assert _clerk_id_of(check, owner["id"]) == owner["old_clerk_id"], "sign-in never rebinds the row"
    assert _rows_with_email(check, owner["email"]) == 1
    assert _owned_workspaces(check, owner["id"]) == 1, "no second personal workspace"


def test_provisioning_a_returning_owner_is_a_409_not_a_500(people, new_session):
    """The report's crash site: _provision_new_user_workspace with a taken email."""
    owner = people.person("direct")
    people.workspace(owner["id"], personal=True)

    with pytest.raises(HTTPException) as raised:
        _provision_new_user_workspace(new_session(), owner["new_clerk_id"], email=owner["email"], name="Direct")

    check = new_session()
    assert raised.value.status_code == 409
    assert _clerk_id_of(check, owner["id"]) == owner["old_clerk_id"]
    assert _rows_with_email(check, owner["email"]) == 1


def test_after_the_cutover_step_a_returning_owner_lands_in_their_workspace(people, new_session):
    owner = people.person("owner", email=f"Cutover-Owner-{uuid.uuid4().hex[:6]}@Example.test")
    theirs = people.workspace(owner["id"], personal=True)

    decisions, moved = _run_cutover(new_session, _production_user(owner))
    resolved = _sign_in(new_session, owner, email=owner["email"].lower())

    check = new_session()
    assert [d.outcome for d in decisions] == [cutover.RELINK] and moved == 1
    assert str(resolved) == theirs
    assert _clerk_id_of(check, owner["id"]) == owner["new_clerk_id"]
    assert _records(check, owner["id"]) == [
        (owner["old_clerk_id"], owner["new_clerk_id"], owner["email"].lower(), False),
    ]
    assert _owned_workspaces(check, owner["id"]) == 1, "no second personal workspace"


def test_after_the_cutover_step_a_returning_member_lands_in_the_workspace_they_joined(people, new_session):
    inviter = people.person("inviter")
    member = people.person("member")
    team = people.workspace(inviter["id"], member_id=member["id"])

    _run_cutover(new_session, _production_user(member))
    resolved = _sign_in(new_session, member)

    check = new_session()
    assert str(resolved) == team
    assert _clerk_id_of(check, member["id"]) == member["new_clerk_id"]
    assert _owned_workspaces(check, member["id"]) == 0, "a member is not given a personal workspace"


def test_an_unverified_email_or_a_taken_production_id_moves_nothing(people, new_session):
    unverified = people.person("unverified")
    blocked = people.person("blocked")
    squatter = people.person("squatter")
    held = new_session()
    held.execute(text("UPDATE users SET clerk_user_id = :new WHERE id = :id"),
                 {"new": blocked["new_clerk_id"], "id": squatter["id"]})
    held.commit()

    decisions, moved = _run_cutover(new_session, _production_user(unverified, verified=False), _production_user(blocked))

    check = new_session()
    assert [d.outcome for d in decisions] == [cutover.EMAIL_NOT_VERIFIED, cutover.TARGET_TAKEN]
    assert moved == 0
    assert _clerk_id_of(check, unverified["id"]) == unverified["old_clerk_id"]
    assert _clerk_id_of(check, blocked["id"]) == blocked["old_clerk_id"]
    assert _records(check, unverified["id"]) == [] and _records(check, blocked["id"]) == []


def test_a_row_moves_once_and_a_rollback_and_retry_keep_one_record(people, new_session):
    person = people.person("once")
    prod = _production_user(person)

    _run_cutover(new_session, prod)
    again, moved_again = _run_cutover(new_session, prod)
    assert [d.outcome for d in again] == [cutover.ALREADY_LINKED] and moved_again == 0

    assert _run_rollback(new_session, person["id"]) == 1
    check = new_session()
    assert _clerk_id_of(check, person["id"]) == person["old_clerk_id"]
    assert [r[3] for r in _records(check, person["id"])] == [True], "the record is stamped, not deleted"

    retry, moved_retry = _run_cutover(new_session, prod)
    check = new_session()
    assert [d.outcome for d in retry] == [cutover.REAPPLY] and moved_retry == 1
    assert _clerk_id_of(check, person["id"]) == person["new_clerk_id"]
    assert len(_records(check, person["id"])) == 1 and _records(check, person["id"])[0][3] is False


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
