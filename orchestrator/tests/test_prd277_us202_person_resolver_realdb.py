"""PRD-277 US-202: one way to ask who a person is, keyed on ``users.id``.

``core.auth.person`` against real Postgres (``@integration``). Every read answers the same for
a person named by ``users.id`` and for the same person named by their Clerk user ID (whose
``users.id`` is read inside the same statement):

* the active member row's role, and nothing for an inactive row or a stranger;
* the legacy owner (``workspaces.owner_id``);
* "owns it, or holds one of these roles in it", the workspace-admin and answerer rule;
* "owns it, or is an active member", the request lane's access check;
* the workspace a person lands in when the request names none;
* a Clerk user ID no row carries names nobody: no role, no access, no workspace;
* a found member row ends the role lookup whatever its role: an owner whose active row has a
  NULL role gets that NULL, not ``owner``, on the Clerk lane and the paired-device lane alike;
* the Clerk lane's last-seen stamp lands on the row its Clerk user ID names, and only there.

The statement checks need no database. Skips the rest when no Postgres is reachable (CI
runs it).
"""
from __future__ import annotations

import os
import uuid
from types import SimpleNamespace
from typing import Optional

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.auth import hybrid  # noqa: E402
from core.auth import person as people  # noqa: E402
from core.auth.clerk_person import CLERK_PERSON_PK_SQL  # noqa: E402
from core.auth.device import DEVICE_AUTH_TYPE  # noqa: E402
from core.auth.workspace_permission import resolve_workspace_role  # noqa: E402
from core.database.database import get_database_url  # noqa: E402

pytestmark = pytest.mark.integration

ADMIN_ROLES = ("owner", "admin")


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for probe in ("SELECT clerk_user_id, last_sign_in FROM users LIMIT 1",
                          "SELECT 1 FROM workspace_members LIMIT 1", "SELECT 1 FROM workspaces LIMIT 1"):
                c.execute(text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the person resolver suite needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


class _Team:
    """One workspace with an owner, an editor, a viewer whose row is inactive, and a stranger."""

    def __init__(self, new_session):
        self._new_session = new_session
        self.tag = uuid.uuid4().hex[:10]
        self.users, self.workspaces = [], []
        s = new_session()
        self.owner, self.editor, self.gone, self.stranger = (self._person(s, label)
                                                             for label in ("owner", "editor", "gone", "stranger"))
        self.ws = self._workspace(s, self.owner[0])
        self._member(s, self.editor[0], "editor", True)
        self._member(s, self.gone[0], "viewer", False)
        s.commit()
        s.close()

    def _person(self, s, label: str) -> tuple:
        clerk_id = f"user_prd277_us202_{label}_{self.tag}"
        pk = s.execute(
            text("INSERT INTO users (username, email, clerk_user_id, is_active) VALUES (:e, :e, :c, true) RETURNING id"),
            {"e": f"prd277-us202-{label}-{self.tag}@example.test", "c": clerk_id},
        ).scalar_one()
        self.users.append(pk)
        return pk, clerk_id

    def _workspace(self, s, owner_pk: int) -> str:
        ws = str(uuid.uuid4())
        s.execute(text("INSERT INTO workspaces (id, name, owner_id, is_active) "
                       "VALUES (CAST(:id AS uuid), 'PRD-277 US-202', :owner, true)"), {"id": ws, "owner": owner_pk})
        self.workspaces.append(ws)
        return ws

    def _member(self, s, user_pk: int, role: Optional[str], active: bool) -> None:
        s.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                       "VALUES (CAST(:ws AS uuid), :uid, :role, :active)"),
                  {"ws": self.ws, "uid": user_pk, "role": role, "active": active})

    def add_member(self, user_pk: int, role: Optional[str]) -> None:
        """One more active member row, committed."""
        s = self._new_session()
        self._member(s, user_pk, role, True)
        s.commit()
        s.close()

    def both_ways(self, who: tuple) -> tuple:
        """The same person, named by ``users.id`` and by Clerk user ID."""
        return people.by_user_pk(who[0]), people.by_clerk_id(who[1])

    def sweep(self) -> None:
        s = self._new_session.sweep()
        s.execute(text("DELETE FROM workspace_members WHERE workspace_id = ANY(CAST(:ws AS uuid[]))"),
                  {"ws": self.workspaces})
        s.execute(text("DELETE FROM workspaces WHERE id = ANY(CAST(:ws AS uuid[]))"), {"ws": self.workspaces})
        s.execute(text("DELETE FROM users WHERE id = ANY(:u)"), {"u": self.users})
        s.commit()


@pytest.fixture
def team(engine, new_session):
    made = _Team(new_session)
    yield made
    made.sweep()


def _role(db, person, ws) -> Optional[str]:
    row = people.member_row(db, person, ws)
    return row[0] if row else None


def _answers(db, person, ws) -> tuple:
    landed = people.default_workspace(db, person)
    return (_role(db, person, ws), people.owns_workspace(db, person, ws),
            people.owns_or_holds_role(db, person, ws, ADMIN_ROLES), people.may_open_workspace(db, person, ws),
            str(landed) if landed else None)


# --------------------------------------------------------------------------- statements, no database


def test_a_clerk_id_is_read_as_its_users_id_in_the_same_statement():
    statement, params = people.person_statement("SELECT 1 FROM t WHERE user_id = {person}", people.by_clerk_id("c"))
    assert str(statement) == f"SELECT 1 FROM t WHERE user_id = {CLERK_PERSON_PK_SQL}"
    assert params == {"cid": "c"}


def test_a_users_id_is_a_bound_parameter():
    statement, params = people.person_statement("SELECT 1 FROM t WHERE user_id = {person}", people.by_user_pk(7),
                                                ws="w")
    assert str(statement) == "SELECT 1 FROM t WHERE user_id = :uid"
    assert params == {"ws": "w", "uid": 7}


def test_nothing_but_the_two_expressions_fills_the_slot():
    with pytest.raises(ValueError):
        people.person_statement("SELECT {person}", people.Person("1 OR 1=1", {}))


# --------------------------------------------------------------------------- the reads, on Postgres


def test_each_read_answers_the_same_both_ways(team, new_session):
    db = new_session()
    expected = {
        team.owner: (None, True, True, True, team.ws),
        team.editor: ("editor", False, False, True, team.ws),
        team.gone: (None, False, False, False, None),
        team.stranger: (None, False, False, False, None),
    }
    for who, answer in expected.items():
        by_pk, by_clerk = team.both_ways(who)
        assert _answers(db, by_pk, team.ws) == answer, who
        assert _answers(db, by_clerk, team.ws) == answer, who


def test_the_answerer_roles_take_the_editor(team, new_session):
    db = new_session()
    for person in team.both_ways(team.editor):
        assert people.owns_or_holds_role(db, person, team.ws, (*ADMIN_ROLES, "editor"))


def test_a_clerk_id_no_row_carries_names_nobody(team, new_session):
    db = new_session()
    nobody = people.by_clerk_id(f"user_prd277_us202_nobody_{team.tag}")
    assert _answers(db, nobody, team.ws) == (None, False, False, False, None)


def test_the_last_seen_stamp_lands_on_the_clerk_ids_row_only(team, new_session):
    hybrid._last_seen_touched.clear()
    db = new_session()
    hybrid._touch_last_seen(db, team.editor[1])
    check = new_session()
    stamped = dict(check.execute(text("SELECT id, last_sign_in IS NOT NULL FROM users WHERE id = ANY(:u)"),
                                 {"u": team.users}).fetchall())
    hybrid._last_seen_touched.clear()
    assert stamped == {team.owner[0]: False, team.editor[0]: True, team.gone[0]: False, team.stranger[0]: False}


def test_an_owners_member_row_with_no_role_ends_the_lookup_on_both_lanes(team, new_session):
    team.add_member(team.owner[0], None)
    db = new_session()
    for person in team.both_ways(team.owner):
        row = people.member_row(db, person, team.ws)
        assert row is not None and row[0] is None
    clerk = SimpleNamespace(user=SimpleNamespace(clerk_user_id=team.owner[1]), workspace_id=team.ws)
    device = SimpleNamespace(auth_type=DEVICE_AUTH_TYPE, user=SimpleNamespace(id=team.owner[0]),
                             workspace_id=team.ws)
    assert resolve_workspace_role(db, clerk) is None
    assert resolve_workspace_role(db, device) is None


def test_an_owner_with_no_member_row_is_the_owner_on_both_lanes(team, new_session):
    db = new_session()
    clerk = SimpleNamespace(user=SimpleNamespace(clerk_user_id=team.owner[1]), workspace_id=team.ws)
    device = SimpleNamespace(auth_type=DEVICE_AUTH_TYPE, user=SimpleNamespace(id=team.owner[0]),
                             workspace_id=team.ws)
    assert resolve_workspace_role(db, clerk) == "owner"
    assert resolve_workspace_role(db, device) == "owner"
