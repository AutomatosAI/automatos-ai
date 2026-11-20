"""PRD-277 US-405 against Postgres: making another install admin, and never losing the last one.

Through the real route (``POST /api/v1/auth/people/{user_id}/system-role``, ``api/auth_sessions.py``)
and ``core.auth.people``, each test in a transaction rolled back at the end. Every install admin the
database already holds is made a plain person first, inside that transaction, so "the last admin"
means the ones each test makes.

* An install admin, after step-up, makes another person an install admin: committed, and audited
  ``user:system_role_changed`` (the admin as actor, the old and new role) in the workspace they
  acted from, else the default one. A stale session is sent to sign in
  again and changes nothing.
* They act on a peer (another admin), never on a higher stored role, and give only a role theirs
  includes; a role that is no install role is 422.
* The last active install admin can neither lose the role (409, by the route or the service) nor
  be deactivated (``LastInstallAdmin``, nothing changed); with a second admin, both are allowed.
* ``local`` and ``saas`` have no such route answer: the role is not stored there.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import auth_sessions as routes  # noqa: E402
from config import config  # noqa: E402
from core.auth import hybrid, people, session_lane, sessions, step_up  # noqa: E402
from core.database.database import get_db  # noqa: E402

INSTALL = "https://automatos.example.test"


class _BorrowedSession:
    """The test's session, lent to the lane: ``close()`` is not done."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


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


def _person(db, role=None) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(
        text("INSERT INTO users (username, email, is_active, system_role) VALUES (:u, :e, true, :r) RETURNING id"),
        {"u": f"prd277-w4b-{tag}", "e": f"prd277-w4b-{tag}@example.test", "r": role},
    ).scalar_one()


@pytest.fixture
def install(db_session, seed_workspace, monkeypatch):
    """An enterprise install with no install admin yet, its default workspace, and the routes."""
    workspace_id = uuid.UUID(seed_workspace())
    db_session.execute(text("UPDATE users SET system_role = NULL WHERE system_role IN ('admin', 'super_admin')"))
    db_session.commit()
    for name, value in {"AUTH_EDITION": "enterprise", "REQUIRE_AUTH": True, "FRONTEND_URL": INSTALL,
                        "DEFAULT_WORKSPACE_ID": str(workspace_id), "OIDC_ISSUER": ""}.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(session_lane, "SessionLocal", lambda: _BorrowedSession(db_session))
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    app = FastAPI()
    app.include_router(routes.router)
    step_up.install_step_up_handler(app)
    app.dependency_overrides[get_db] = lambda: db_session
    return SimpleNamespace(db=db_session, workspace_id=workspace_id, client=TestClient(app))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _signed_in(db, user_id, *, minutes_ago=0) -> dict:
    """A bearer header for a session of ``user_id`` whose person signed in ``minutes_ago``."""
    issued = sessions.issue_session(db, user_id=user_id, auth_time=_now() - timedelta(minutes=minutes_ago),
                                    now=_now())
    return {"authorization": "Bearer " + issued.token}


def _give(install, headers, user_id, role):
    return install.client.post(f"/api/v1/auth/people/{user_id}/system-role", headers=headers, json={"role": role})


def _role(db, user_id):
    db.expire_all()
    return db.execute(text("SELECT system_role FROM users WHERE id = :u"), {"u": user_id}).scalar_one()


def _role_rows(db, workspace_id) -> list:
    return db.execute(text("SELECT user_id, details FROM audit_logs WHERE workspace_id = CAST(:ws AS uuid) "
                           "AND action = 'user:system_role_changed' ORDER BY id"), {"ws": str(workspace_id)}).all()


def _requester(install, actor):
    return people.Requester(user_id=actor, workspace_id=None)


# --------------------------------------------------------------------------- making another admin


def test_an_admin_makes_another_person_an_install_admin_audited(install):
    admin, person = _person(install.db, "admin"), _person(install.db)
    install.db.commit()

    answer = _give(install, _signed_in(install.db, admin), person, "admin")

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"user_id": person, "old_role": None, "new_role": "admin", "changed": True}
    assert _role(install.db, person) == "admin"
    assert _role_rows(install.db, install.workspace_id) == [(admin, {"old_role": None, "new_role": "admin"})]


def test_a_stale_session_is_sent_to_sign_in_again_and_changes_nothing(install):
    admin, person = _person(install.db, "admin"), _person(install.db)
    install.db.commit()

    answer = _give(install, _signed_in(install.db, admin, minutes_ago=config.STEP_UP_MAX_AGE_MINUTES + 1),
                   person, "admin")

    assert answer.status_code == 403 and answer.json()["code"] == step_up.STEP_UP_CODE
    assert answer.json()["step_up"]["start"] == step_up.SIGN_IN_AGAIN_START
    assert _role(install.db, person) is None and _role_rows(install.db, install.workspace_id) == []


def test_a_member_cannot_give_a_role(install):
    member, person = _person(install.db), _person(install.db)
    install.db.commit()

    assert _give(install, _signed_in(install.db, member), person, "admin").status_code == 403
    assert _role(install.db, person) is None


def test_an_admin_acts_on_a_peer_but_never_on_a_higher_role_or_beyond_their_own(install):
    admin, peer = _person(install.db, "admin"), _person(install.db, "admin")
    higher, person = _person(install.db, "super_admin"), _person(install.db)
    install.db.commit()
    headers = _signed_in(install.db, admin)

    assert _give(install, headers, higher, "user").status_code == 403
    assert _give(install, headers, person, "super_admin").json()["detail"] == routes.ABOVE_YOUR_ROLE
    assert _give(install, headers, person, "owner").status_code == 422
    assert (_role(install.db, higher), _role(install.db, person)) == ("super_admin", None)
    assert _give(install, headers, peer, "user").status_code == 200  # a peer, with admins left
    assert _role(install.db, peer) == "user"


# --------------------------------------------------------------------------- the last admin


def test_the_last_admin_cannot_lose_the_role(install):
    admin = _person(install.db, "admin")
    install.db.commit()

    answer = _give(install, _signed_in(install.db, admin), admin, "user")

    assert answer.status_code == 409 and answer.json()["detail"] == people.LAST_INSTALL_ADMIN_MESSAGE
    assert _role(install.db, admin) == "admin" and _role_rows(install.db, install.workspace_id) == []
    with pytest.raises(people.LastInstallAdmin):
        people.set_install_role(install.db, admin, "user", _requester(install, admin))
    assert _role(install.db, admin) == "admin"


def test_the_last_admin_cannot_be_deactivated_and_keeps_their_sessions(install):
    admin = _person(install.db, "admin")
    install.db.commit()
    _signed_in(install.db, admin)

    with pytest.raises(people.LastInstallAdmin):
        people.deactivate_person(install.db, admin, _requester(install, None))

    install.db.expire_all()
    assert install.db.execute(text("SELECT is_active FROM users WHERE id = :u"), {"u": admin}).scalar_one() is True
    assert install.db.execute(text("SELECT COUNT(*) FROM auth_sessions WHERE user_id = :u AND revoked_at IS NULL"),
                              {"u": admin}).scalar_one() == 1


def test_a_deactivated_admin_is_not_counted_as_the_one_left(install):
    admin, gone = _person(install.db, "admin"), _person(install.db, "admin")
    install.db.execute(text("UPDATE users SET is_active = false WHERE id = :u"), {"u": gone})
    install.db.commit()

    assert _give(install, _signed_in(install.db, admin), admin, "user").status_code == 409


def test_with_a_second_admin_one_may_step_down_and_be_deactivated(install):
    first, second = _person(install.db, "admin"), _person(install.db, "admin")
    install.db.commit()

    assert _give(install, _signed_in(install.db, first), first, "user").status_code == 200
    with pytest.raises(people.LastInstallAdmin):
        people.deactivate_person(install.db, second, _requester(install, None))
    assert _give(install, _signed_in(install.db, second), first, "admin").status_code == 200
    people.deactivate_person(install.db, first, _requester(install, second))
    install.db.expire_all()
    assert install.db.execute(text("SELECT is_active FROM users WHERE id = :u"), {"u": first}).scalar_one() is False


def _statements_of(db, act) -> list:
    """Every statement ``act`` sends on the test's connection, in order."""
    statements = []
    connection = db.connection()

    def listener(conn, cursor, statement, params, context, many):
        statements.append(statement)

    event.listen(connection, "before_cursor_execute", listener)
    try:
        act()
    finally:
        event.remove(connection, "before_cursor_execute", listener)
    return statements


def test_the_guard_locks_the_admins_before_it_counts_them_and_before_any_change(install):
    first, second = _person(install.db, "admin"), _person(install.db, "admin")
    install.db.commit()

    statements = _statements_of(install.db, lambda: people.set_install_role(install.db, first, "user",
                                                                           _requester(install, second)))

    admins_read = [i for i, sql in enumerate(statements) if "FROM users" in sql and "system_role = ANY" in sql]
    changed = [i for i, sql in enumerate(statements) if sql.lstrip().upper().startswith("UPDATE USERS")]
    assert admins_read and changed and admins_read[0] < changed[0]
    assert all("FOR UPDATE" in statements[i] for i in admins_read)  # counted only through the lock
    assert not any("COUNT(" in sql.upper() and "FROM USERS" in sql.upper() for sql in statements)


def test_a_person_who_is_no_admin_is_never_held_by_the_guard(install):
    admin, person = _person(install.db, "admin"), _person(install.db)
    install.db.commit()

    done = people.deactivate_person(install.db, person, _requester(install, admin))

    assert done.user_id == person


# --------------------------------------------------------------------------- local and saas


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_outside_enterprise_the_route_answers_as_if_it_were_not_served(install, monkeypatch, edition):
    person = _person(install.db)
    install.db.commit()
    monkeypatch.setattr(config, "AUTH_EDITION", edition)

    answer = install.client.post(f"/api/v1/auth/people/{person}/system-role", json={"role": "admin"})

    assert answer.status_code == 404 and _role(install.db, person) is None
