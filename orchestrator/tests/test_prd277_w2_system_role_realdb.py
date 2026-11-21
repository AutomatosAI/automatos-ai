"""PRD-277 Wave 2, US-203: an install-wide role the install stores, read only under OIDC sign-in.

``core.auth.system_role`` against real Postgres (``@integration``):

* ``users.system_role`` is read only when the edition signs people in with OIDC; in ``saas``
  and ``local`` the database is not even asked;
* in ``saas`` the Clerk lane's role is the token's claim whatever the column holds, through the
  real request-context code (``hybrid._clerk_context``) on the real ``users`` row;
* changing the role writes one ``audit_logs`` row (who, the old and the new role) in the same
  commit, and a change whose audit row cannot be written is rolled back;
* a call that changes nothing writes no audit row and, like a change, ends its transaction, so
  no row lock is left held;
* outside OIDC sign-in the role is never written, and an unknown role or person is refused.

Skips the database cases when no Postgres is reachable (CI runs them).
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

from config import config  # noqa: E402
from core.auth import hybrid  # noqa: E402
from core.auth.system_role import (  # noqa: E402
    SYSTEM_ROLE_CHANGED_ACTION,
    SystemRoleAuditFailed,
    SystemRoleChange,
    SystemRoleNotStored,
    UnknownPerson,
    set_system_role,
    stored_system_role,
    stores_system_role,
)
from core.database.database import get_database_url  # noqa: E402

pytestmark = pytest.mark.integration

# A stand-in for a Clerk token, built from parts so no literal looks like one.
CLERK_SHAPED_TOKEN = ".".join(("header", "claims", "signature"))
NO_SUCH_PERSON = -1


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for probe in ("SELECT system_role FROM users LIMIT 1", "SELECT 1 FROM audit_logs LIMIT 1",
                          "SELECT 1 FROM workspaces LIMIT 1"):
                c.execute(text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the system role suite needs a reachable Postgres with the PRD-277 W2 schema: {exc}")
    yield eng
    eng.dispose()


class _Install:
    """The people and workspaces of one test; the sweep deletes them and their audit rows."""

    def __init__(self, new_session):
        self._new_session = new_session
        self.tag = uuid.uuid4().hex[:10]
        self.users, self.workspaces = [], []

    def person(self, label: str, role: str = None) -> dict:
        email = f"prd277-w2-role-{label}-{self.tag}@example.test"
        clerk_id = f"user_prd277_role_{label}_{self.tag}"
        s = self._new_session()
        uid = s.execute(
            text("INSERT INTO users (username, email, clerk_user_id, is_active, system_role) "
                 "VALUES (:e, :e, :cid, true, :role) RETURNING id"),
            {"e": email, "cid": clerk_id, "role": role},
        ).scalar_one()
        s.commit()
        self.users.append(uid)
        return {"id": uid, "email": email, "clerk_id": clerk_id}

    def workspace(self, owner_id: int) -> uuid.UUID:
        ws = uuid.uuid4()
        s = self._new_session()
        s.execute(
            text("INSERT INTO workspaces (id, name, slug, owner_id, is_personal, is_active) "
                 "VALUES (CAST(:id AS uuid), :n, :n, :owner, false, true)"),
            {"id": str(ws), "n": f"prd277-w2-role-{self.tag}-{len(self.workspaces)}", "owner": owner_id},
        )
        s.commit()
        self.workspaces.append(str(ws))
        return ws

    def sweep(self):
        s = self._new_session.sweep()
        params = {"users": self.users, "ws": self.workspaces}
        s.execute(text("DELETE FROM audit_logs WHERE CAST(workspace_id AS text) = ANY(:ws)"), params)
        s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) = ANY(:ws) OR owner_id = ANY(:users)"), params)
        s.execute(text("DELETE FROM users WHERE id = ANY(:users)"), params)
        s.commit()


@pytest.fixture
def install(new_session):
    made = _Install(new_session)
    yield made
    made.sweep()


@pytest.fixture
def edition(monkeypatch):
    """Sets the running edition for the test."""
    def _set(name: str) -> None:
        monkeypatch.setattr(config, "AUTH_EDITION", name)
    return _set


class _NoDatabase:
    """A session that fails the test if it is touched."""

    def __getattr__(self, name):
        pytest.fail(f"the database was used ({name}) where this edition must not read or write the role")


def _role(session, user_id: int):
    return session.execute(text("SELECT system_role FROM users WHERE id = :id"), {"id": user_id}).scalar_one()


def _audit_rows(session, ws) -> list:
    return [dict(r) for r in session.execute(
        text("SELECT user_id, actor_type, action, resource_type, resource_id, details, ip_address, user_agent "
             "FROM audit_logs WHERE workspace_id = CAST(:ws AS uuid) ORDER BY id"),
        {"ws": str(ws)},
    ).mappings()]


# --------------------------------------------------------------------------- reading


def test_the_stored_role_is_read_only_when_people_sign_in_with_oidc(install, new_session, edition):
    person = install.person("reader", role="super_admin")

    edition("enterprise")
    assert stores_system_role() is True
    assert stored_system_role(new_session(), person["id"]) == "super_admin"
    assert stored_system_role(new_session(), NO_SUCH_PERSON) is None

    for other in ("saas", "local"):
        edition(other)
        assert stores_system_role() is False
        assert stored_system_role(_NoDatabase(), person["id"]) is None


class _Clerk:
    """Stands in for Clerk: the token verifies, and its claims carry ``claimed`` as the role."""

    def __init__(self, person: dict, claimed: str):
        self._info = {"clerk_user_id": person["clerk_id"], "email": person["email"],
                      "system_role": claimed, "role": claimed}

    def verify_token(self, token):
        assert token == CLERK_SHAPED_TOKEN
        return {"sub": self._info["clerk_user_id"]}

    def extract_user_info(self, claims):
        return dict(self._info)


@pytest.mark.parametrize("stored, claimed", [("super_admin", "user"), ("user", "super_admin"), ("admin", "user")])
def test_in_saas_the_role_comes_from_clerks_claims_and_the_column_is_ignored(
        install, new_session, edition, monkeypatch, stored, claimed):
    edition("saas")
    person = install.person(f"clerk-{stored}", role=stored)
    ws = install.workspace(person["id"])
    monkeypatch.setattr(hybrid, "get_clerk_auth", lambda: _Clerk(person, claimed))

    ctx = hybrid._clerk_context(new_session(), CLERK_SHAPED_TOKEN, ws, False)

    assert ctx.auth_type == "clerk" and ctx.workspace_id == ws
    assert ctx.user.system_role == claimed
    assert _role(new_session(), person["id"]) == stored       # non-vacuous: the column held another role


# --------------------------------------------------------------------------- changing


def test_changing_the_role_is_audited_in_the_same_commit(install, new_session, edition):
    edition("enterprise")
    actor, target = install.person("actor"), install.person("target")
    ws = install.workspace(actor["id"])
    change_kwargs = {"workspace_id": ws, "actor_user_id": actor["id"], "ip_address": "203.0.113.7",
                     "user_agent": "prd277-w2-test"}

    made = set_system_role(new_session(), user_id=target["id"], role="admin", **change_kwargs)
    cleared = set_system_role(new_session(), user_id=target["id"], role=None, **change_kwargs)
    unchanged = set_system_role(new_session(), user_id=target["id"], role=None, **change_kwargs)

    assert made == SystemRoleChange(user_id=target["id"], old_role=None, new_role="admin") and made.changed
    assert cleared == SystemRoleChange(user_id=target["id"], old_role="admin", new_role=None) and cleared.changed
    assert not unchanged.changed
    check = new_session()
    assert _role(check, target["id"]) is None
    common = {"user_id": actor["id"], "actor_type": "user", "action": SYSTEM_ROLE_CHANGED_ACTION,
              "resource_type": "user", "resource_id": str(target["id"]),
              "ip_address": "203.0.113.7", "user_agent": "prd277-w2-test"}
    assert _audit_rows(check, ws) == [
        {**common, "details": {"old_role": None, "new_role": "admin"}},
        {**common, "details": {"old_role": "admin", "new_role": None}},
    ], "one row per change, none for a change that changed nothing"


def test_a_change_that_changes_nothing_writes_no_audit_row_and_leaves_no_lock_held(install, new_session, edition):
    edition("enterprise")
    target = install.person("noop", role="admin")
    ws = install.workspace(target["id"])
    caller = new_session()

    change = set_system_role(caller, user_id=target["id"], role="admin", workspace_id=ws, actor_user_id=None)

    assert change == SystemRoleChange(user_id=target["id"], old_role="admin", new_role="admin")
    assert not caller.in_transaction(), "the call ended its transaction, as a change does"
    other = new_session()
    locked = other.execute(text("SELECT id FROM users WHERE id = :id FOR UPDATE NOWAIT"),
                           {"id": target["id"]}).scalar_one()          # raises if the row is still locked
    other.rollback()
    assert locked == target["id"]
    assert _audit_rows(new_session(), ws) == []


def test_a_change_made_by_the_system_is_recorded_with_no_person(install, new_session, edition):
    edition("enterprise")
    target = install.person("bootstrap")
    ws = install.workspace(target["id"])

    set_system_role(new_session(), user_id=target["id"], role="super_admin", workspace_id=ws, actor_user_id=None)

    (row,) = _audit_rows(new_session(), ws)
    assert (row["user_id"], row["actor_type"]) == (None, "system")
    assert row["details"] == {"old_role": None, "new_role": "super_admin"}


def test_a_change_whose_audit_row_cannot_be_written_is_rolled_back(install, new_session, edition):
    edition("enterprise")
    actor, target = install.person("actor-gone"), install.person("target-kept", role="user")
    missing_workspace = uuid.uuid4()                    # the audit row's workspace does not exist

    with pytest.raises(SystemRoleAuditFailed):
        set_system_role(new_session(), user_id=target["id"], role="super_admin",
                        workspace_id=missing_workspace, actor_user_id=actor["id"])

    check = new_session()
    assert _role(check, target["id"]) == "user"
    assert _audit_rows(check, missing_workspace) == []


@pytest.mark.parametrize("other", ["saas", "local"])
def test_outside_oidc_sign_in_the_role_is_never_written(edition, other):
    edition(other)
    with pytest.raises(SystemRoleNotStored):
        set_system_role(_NoDatabase(), user_id=1, role="admin", workspace_id=uuid.uuid4(), actor_user_id=None)


@pytest.mark.parametrize("role", ["owner", "Admin", "super_admin ", "", "service"])
def test_an_unknown_role_is_refused_before_any_query(edition, role):
    edition("enterprise")
    with pytest.raises(ValueError):
        set_system_role(_NoDatabase(), user_id=1, role=role, workspace_id=uuid.uuid4(), actor_user_id=None)


def test_an_unknown_person_is_refused_and_nothing_is_audited(install, new_session, edition):
    edition("enterprise")
    actor = install.person("actor-unknown")
    ws = install.workspace(actor["id"])

    with pytest.raises(UnknownPerson):
        set_system_role(new_session(), user_id=NO_SUCH_PERSON, role="admin", workspace_id=ws,
                        actor_user_id=actor["id"])

    assert _audit_rows(new_session(), ws) == []
