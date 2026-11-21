"""PRD-277 Wave 5 against Postgres: who-am-I and the people waiting for a workspace.

Through the real routes (``api/auth_web.py``, ``api/team_waiting.py``), the session lane and
``core.auth.people``, each test in a transaction rolled back at the end.

* ``GET /api/v1/auth/me`` names the session's person, their stored install role and every workspace
  they can open with their role in it; a person in no workspace is answered too (the shared request
  context would refuse them), with ``in_no_workspace``. An ended session is ``401 session_ended``.
* A workspace's role is the one the permission gate uses: the member row's, else ``owner`` for its
  owner with no member row.
* An install admin with ``members:invite`` lists the people waiting (an identity, no workspace),
  oldest first and a page at a time with the total, and
  places one in their workspace in a role below owner, after step-up: the membership and its
  ``member:added_from_waiting`` row land together. Someone no longer waiting is 409, owner is 400, a
  member who is no install admin is 403, and a stale sign-in is sent to sign in again.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import auth_web, team_waiting  # noqa: E402
from config import config  # noqa: E402
from core.auth import hybrid, people, session_lane, sessions, step_up  # noqa: E402
from core.database.database import get_db  # noqa: E402

INSTALL = "https://automatos.example.test"
ISSUER = "https://idp.example.test/realms/prd277-w5"


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


@pytest.fixture
def install(db_session, seed_workspace, monkeypatch):
    """An enterprise install, its default workspace, and the two routers."""
    workspace_id = uuid.UUID(seed_workspace(name="prd277-w5"))
    for name, value in {"AUTH_EDITION": "enterprise", "REQUIRE_AUTH": True, "FRONTEND_URL": INSTALL,
                        "DEFAULT_WORKSPACE_ID": str(workspace_id), "OIDC_ISSUER": ""}.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(session_lane, "SessionLocal", lambda: _BorrowedSession(db_session))
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    app = FastAPI()
    app.include_router(auth_web.router)
    app.include_router(team_waiting.router)
    step_up.install_step_up_handler(app)
    app.dependency_overrides[get_db] = lambda: db_session
    return SimpleNamespace(db=db_session, workspace_id=workspace_id, client=TestClient(app))


def _person(db, *, role=None, identity=True, created_at=None) -> int:
    tag = uuid.uuid4().hex[:12]
    user_id = db.execute(
        text("INSERT INTO users (username, email, name, is_active, system_role, created_at) "
             "VALUES (:u, :e, :n, true, :r, COALESCE(CAST(:c AS timestamptz), now())) RETURNING id"),
        {"u": f"prd277-w5-{tag}", "e": f"prd277-w5-{tag}@example.test", "n": f"Person {tag}", "r": role,
         "c": created_at},
    ).scalar_one()
    if identity:
        db.execute(text("INSERT INTO user_identities (user_id, issuer, subject) VALUES (:u, :i, :s)"),
                   {"u": user_id, "i": ISSUER, "s": "subject-" + tag})
    return user_id


def _signed_in(db, user_id, *, minutes_ago=0) -> dict:
    issued = sessions.issue_session(db, user_id=user_id, auth_time=datetime.now(timezone.utc)
                                    - timedelta(minutes=minutes_ago), now=datetime.now(timezone.utc))
    return {"authorization": "Bearer " + issued.token}


def _admin(install) -> int:
    admin = _person(install.db, role="admin")
    people.set_membership(install.db, admin, install.workspace_id, "admin")
    install.db.commit()
    return admin


def _waiting_url(install, user_id=None) -> str:
    base = f"/api/workspaces/{install.workspace_id}/team/waiting"
    return base if user_id is None else f"{base}/{user_id}"


def _membership(db, user_id, workspace_id):
    db.expire_all()
    return db.execute(text("SELECT role, is_active FROM workspace_members WHERE user_id = :u "
                           "AND workspace_id = CAST(:w AS uuid)"), {"u": user_id, "w": str(workspace_id)}).first()


# --------------------------------------------------------------------------- who am I


def test_me_names_the_person_their_install_role_and_each_workspace_with_their_role(install):
    admin = _admin(install)

    answer = install.client.get("/api/v1/auth/me", headers=_signed_in(install.db, admin))

    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["edition"] == "enterprise" and body["in_no_workspace"] is False
    assert body["person"]["id"] == admin and body["person"]["system_role"] == "admin"
    assert body["workspaces"] == [{"id": str(install.workspace_id), "name": "prd277-w5", "role": "admin"}]
    assert body["default_workspace_id"] == str(install.workspace_id)
    assert answer.headers["cache-control"] == "no-store"


def test_me_answers_a_person_in_no_workspace_whom_the_request_context_refuses(install):
    waiting = _person(install.db)
    install.db.commit()
    headers = _signed_in(install.db, waiting)

    body = install.client.get("/api/v1/auth/me", headers=headers).json()

    assert body["in_no_workspace"] is True and body["workspaces"] == []
    assert body["person"]["system_role"] == "user"
    refused = install.client.get(_waiting_url(install), headers=headers)
    assert refused.status_code == 403  # the shared context's workspace_required, before any handler


def test_me_reports_the_role_the_gate_uses_the_member_row_over_ownership(install, seed_workspace):
    person = _person(install.db)
    owned_only = uuid.UUID(seed_workspace(name="prd277-w5-owned"))
    owned_and_member = uuid.UUID(seed_workspace(name="prd277-w5-both"))
    install.db.execute(text("UPDATE workspaces SET owner_id = :u WHERE id IN (CAST(:a AS uuid), CAST(:b AS uuid))"),
                       {"u": person, "a": str(owned_only), "b": str(owned_and_member)})
    people.set_membership(install.db, person, owned_and_member, "editor")
    install.db.commit()

    body = install.client.get("/api/v1/auth/me", headers=_signed_in(install.db, person)).json()

    roles = {row["id"]: row["role"] for row in body["workspaces"]}
    assert roles == {str(owned_only): "owner", str(owned_and_member): "editor"}


def test_me_after_the_session_ended_is_session_ended(install):
    person = _person(install.db)
    install.db.commit()
    headers = _signed_in(install.db, person)
    # Ended as the product ends it: the revoke sets the time and the reason together.
    assert sessions.revoke_sessions_of(install.db, person, sessions.REVOKED_BY_ADMIN) == 1
    install.db.commit()

    answer = install.client.get("/api/v1/auth/me", headers=headers)

    assert answer.status_code == 401
    assert answer.json()["detail"]["code"] == "session_ended"


# --------------------------------------------------------------------------- waiting for a workspace


def test_an_install_admin_lists_the_people_waiting_and_not_the_members(install):
    admin = _admin(install)
    waiting, no_identity = _person(install.db), _person(install.db, identity=False)
    install.db.commit()

    headers = _signed_in(install.db, admin)
    total = install.client.get(_waiting_url(install), headers=headers).json()["total"]

    # Oldest first: the two just made are on the last page.
    page_size = people.WAITING_PAGE_MAX
    last = f"{_waiting_url(install)}?limit={page_size}&offset={max(0, total - page_size)}"
    answer = install.client.get(last, headers=headers)

    assert answer.status_code == 200, answer.text
    listed = {row["user_id"]: row for row in answer.json()["people"]}
    assert waiting in listed and admin not in listed and no_identity not in listed
    assert set(listed[waiting]) == {"user_id", "email", "name", "signed_in_first_at"}


def test_the_waiting_list_is_oldest_first_paged_and_states_the_total(install):
    admin = _admin(install)
    first, second, third = (_person(install.db, created_at=f"1990-01-0{day}T00:00:00Z") for day in (1, 2, 3))
    install.db.commit()
    headers = _signed_in(install.db, admin)

    page_one = install.client.get(f"{_waiting_url(install)}?limit=2&offset=0", headers=headers).json()
    page_two = install.client.get(f"{_waiting_url(install)}?limit=2&offset=2", headers=headers).json()
    past_end = install.client.get(f"{_waiting_url(install)}?limit=2&offset={page_one['total']}",
                                  headers=headers).json()

    assert [row["user_id"] for row in page_one["people"]] == [first, second]
    assert page_two["people"][0]["user_id"] == third
    assert page_one["total"] >= 3 and page_one["total"] == page_two["total"] == past_end["total"]
    assert (page_one["limit"], page_one["offset"], past_end["people"]) == (2, 0, [])
    too_big = install.client.get(f"{_waiting_url(install)}?limit={people.WAITING_PAGE_MAX + 1}",
                                 headers=headers)
    assert too_big.status_code == 422


def test_an_install_admin_places_a_waiting_person_audited_and_only_once(install):
    admin = _admin(install)
    waiting = _person(install.db)
    install.db.commit()
    headers = _signed_in(install.db, admin)

    answer = install.client.post(_waiting_url(install, waiting), headers=headers, json={"role": "editor"})

    assert answer.status_code == 200, answer.text
    assert tuple(_membership(install.db, waiting, install.workspace_id)) == ("editor", True)
    rows = install.db.execute(text("SELECT user_id, resource_id, details FROM audit_logs WHERE action = "
                                   "'member:added_from_waiting' AND workspace_id = CAST(:w AS uuid)"),
                              {"w": str(install.workspace_id)}).all()
    assert [tuple(row) for row in rows] == [(admin, str(waiting), {"role": "editor"})]
    again = install.client.post(_waiting_url(install, waiting), headers=headers, json={"role": "viewer"})
    assert again.status_code == 409
    assert tuple(_membership(install.db, waiting, install.workspace_id)) == ("editor", True)


def test_nobody_is_placed_as_owner(install):
    admin = _admin(install)
    waiting = _person(install.db)
    install.db.commit()

    answer = install.client.post(_waiting_url(install, waiting), headers=_signed_in(install.db, admin),
                                 json={"role": "owner"})

    assert answer.status_code == 400
    assert _membership(install.db, waiting, install.workspace_id) is None


def test_a_workspace_admin_who_is_no_install_admin_is_refused(install):
    member = _person(install.db)
    people.set_membership(install.db, member, install.workspace_id, "admin")
    waiting = _person(install.db)
    install.db.commit()
    headers = _signed_in(install.db, member)

    assert install.client.get(_waiting_url(install), headers=headers).status_code == 403
    placed = install.client.post(_waiting_url(install, waiting), headers=headers, json={"role": "viewer"})
    assert placed.status_code == 403
    assert _membership(install.db, waiting, install.workspace_id) is None


def test_a_stale_sign_in_is_sent_to_sign_in_again_and_places_nobody(install):
    admin = _admin(install)
    waiting = _person(install.db)
    install.db.commit()
    stale = _signed_in(install.db, admin, minutes_ago=config.STEP_UP_MAX_AGE_MINUTES + 5)

    answer = install.client.post(_waiting_url(install, waiting), headers=stale, json={"role": "viewer"})

    assert answer.status_code == 403
    assert answer.json()["code"] == "step_up_required"
    assert answer.json()["step_up"]["start"].startswith("/api/v1/auth/oidc/start")
    assert _membership(install.db, waiting, install.workspace_id) is None
