"""PRD-277 Wave 3 (US-301, US-303, US-304) against Postgres: issuing, resolving and ending sessions.

* **Issuing** (``issue_session``): the token is ``ams_`` and 32 random bytes; only its SHA-256 is
  stored; the ``sid`` is kept; the two clocks are the settings'; ONE ``auth:sign_in`` row names the
  person. Outside enterprise nothing is issued. A deactivated person is refused with ONE
  ``auth:sign_in_refused`` row carrying only its reason. When the sign-in row cannot be written,
  no session is kept.
* **Resolving** (through ``hybrid.get_request_context_hybrid``): a header token and a same-origin
  cookie resolve as the person, by ``users.id``; an expired, revoked or unknown session and a
  deactivated person are ``401 session_ended`` with their reason; the workspace rules hold; the
  last-seen stamp and the idle clock are written at most every five minutes; the lane's database
  session is always closed and no token reaches the log.
* **Ending** (``core.auth.people`` and ``api/auth_sessions.py``): sign-out, one session, every
  session of a person and deactivating a person each write ONE row; ending nothing writes none;
  deactivating ends the person's sessions and devices in the same commit; a removal lands even
  when its row cannot be written. The admin routes need step-up and an install admin.

* **Hardening:** an oversized user agent and a junk address still give exactly one row; the
  person's row is read ``FOR SHARE`` before the session is inserted, so a sign-in and a
  deactivation order themselves; a failed insert raises a sign-in failure that names no value.

Each test runs in a transaction rolled back at the end; ``commit`` releases a savepoint.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import event, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import auth_sessions as routes  # noqa: E402
from config import config  # noqa: E402
from core.auth import hybrid, people, session_audit, session_lane, sessions, step_up  # noqa: E402
from core.auth.pairing import hash_secret, mint_token  # noqa: E402
from core.auth.principal import claimed_user_pk  # noqa: E402
from core.auth.system_role import UnknownPerson  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.auth_sessions import AuthSession  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402

INSTALL = "https://automatos.example.test"
TOKEN_LENGTH = len("ams_") + 43  # 32 bytes, URL-safe base64 without padding


# --------------------------------------------------------------------------- fixtures


class _BorrowedSession:
    """The test's session, lent to the lane: ``close()`` is counted, not done."""

    def __init__(self, session):
        self._session = session
        self.closes = 0

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        self.closes += 1


class _RefusingAudit:
    """``AuditService`` when its insert fails: it rolls the session back and answers ``None``."""

    def __init__(self, db):
        self.db = db

    def log(self, **_fields):
        self.db.rollback()
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


def _seed_person(db, *, role=None) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(
        text("INSERT INTO users (username, email, is_active, system_role) VALUES (:u, :e, true, :r) RETURNING id"),
        {"u": f"prd277-w3-{tag}", "e": f"prd277-w3-{tag}@example.test", "r": role},
    ).scalar_one()


def _member(db, workspace_id, user_pk) -> None:
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :uid, 'member', true)"), {"ws": str(workspace_id), "uid": user_pk})


@pytest.fixture
def install(db_session, seed_workspace, monkeypatch):
    """An enterprise install, its default workspace and a person in it, committed to a savepoint."""
    workspace_id = uuid.UUID(seed_workspace())
    person = _seed_person(db_session)
    _member(db_session, workspace_id, person)
    db_session.commit()
    for name, value in {"AUTH_EDITION": "enterprise", "REQUIRE_AUTH": True, "FRONTEND_URL": INSTALL,
                        "DEFAULT_WORKSPACE_ID": str(workspace_id)}.items():
        monkeypatch.setattr(config, name, value)
    borrowed = _BorrowedSession(db_session)
    monkeypatch.setattr(session_lane, "SessionLocal", lambda: borrowed)
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    return SimpleNamespace(workspace_id=workspace_id, person=person, lane=borrowed)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _issue(db, user_id, *, at=None, **extra):
    moment = at or _now()
    return sessions.issue_session(db, user_id=user_id, auth_time=moment, now=moment, **extra)


def _audits(db, workspace_id) -> list:
    return db.execute(text(
        "SELECT action, user_id, resource_type, resource_id, details, ip_address, user_agent FROM audit_logs "
        "WHERE workspace_id = CAST(:ws AS uuid) ORDER BY id"), {"ws": str(workspace_id)}).all()


def _row(db, session_id) -> AuthSession:
    db.expire_all()
    return db.get(AuthSession, session_id)


def _request(method="GET", headers=None):
    return SimpleNamespace(method=method, headers=dict(headers or {}), query_params={}, state=SimpleNamespace())


def _bearer(token, **headers):
    return _request(headers={"authorization": "Bearer " + token, **headers})


async def _resolve(request):
    return await hybrid.get_request_context_hybrid(request)


async def _refused(request) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        await _resolve(request)
    return refused.value


# --------------------------------------------------------------------------- issuing


def test_issuing_keeps_only_the_hash_and_writes_one_sign_in_row(db_session, install):
    moment = _now()
    issued = _issue(db_session, install.person, at=moment, sid="idp-session-1", ip_address="203.0.113.7",
                    user_agent="Browser/1.0")

    assert issued.token.startswith("ams_") and len(issued.token) == TOKEN_LENGTH
    assert issued.token not in repr(issued)
    row = _row(db_session, issued.session_id)
    assert row.token_hash == hash_secret(issued.token) and len(row.token_hash) == 64
    assert issued.token not in {str(value) for value in row.__dict__.values()}
    assert row.idp_session_id == "idp-session-1" and row.user_id == install.person
    assert row.idle_expires_at == moment + timedelta(minutes=config.SESSION_IDLE_MINUTES)
    assert row.absolute_expires_at == moment + timedelta(hours=config.SESSION_ABSOLUTE_HOURS)
    assert (row.last_seen_at, row.auth_time, row.revoked_at) == (moment, moment, None)
    assert _audits(db_session, install.workspace_id) == [
        ("auth:sign_in", install.person, "auth_session", str(issued.session_id), {}, "203.0.113.7", "Browser/1.0"),
    ]


def test_every_sign_in_is_a_new_session(db_session, install):
    first, second = _issue(db_session, install.person), _issue(db_session, install.person)

    assert first.session_id != second.session_id and first.token != second.token
    assert [a[0] for a in _audits(db_session, install.workspace_id)] == ["auth:sign_in", "auth:sign_in"]


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_no_session_is_issued_outside_enterprise(db_session, install, monkeypatch, edition):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)

    with pytest.raises(sessions.SessionsNotIssued):
        _issue(db_session, install.person)

    assert db_session.query(AuthSession).filter(AuthSession.user_id == install.person).count() == 0
    assert _audits(db_session, install.workspace_id) == []


def test_a_deactivated_person_is_refused_with_one_row_naming_only_the_reason(db_session, install):
    db_session.execute(text("UPDATE users SET is_active = false WHERE id = :id"), {"id": install.person})
    db_session.commit()

    with pytest.raises(sessions.SignInRefused) as refused:
        _issue(db_session, install.person, ip_address="203.0.113.7")

    assert refused.value.reason == "deactivated"
    assert db_session.query(AuthSession).filter(AuthSession.user_id == install.person).count() == 0
    assert _audits(db_session, install.workspace_id) == [
        ("auth:sign_in_refused", install.person, "user", str(install.person), {"reason": "deactivated"},
         "203.0.113.7", None),
    ]


def test_an_unknown_person_is_refused(db_session, install):
    with pytest.raises(UnknownPerson):
        _issue(db_session, install.person + 10_000_000)


def test_a_sign_in_whose_row_fails_keeps_no_session(db_session, install, monkeypatch):
    monkeypatch.setattr(session_audit, "AuditService", _RefusingAudit)

    with pytest.raises(session_audit.SessionAuditFailed):
        _issue(db_session, install.person)

    assert db_session.query(AuthSession).filter(AuthSession.user_id == install.person).count() == 0


def test_a_refusal_reason_outside_the_list_is_an_error(db_session, install):
    with pytest.raises(ValueError):
        session_audit.record_sign_in_refused(db_session, reason="a token: " + "ams" + "_x")
    assert _audits(db_session, install.workspace_id) == []


# --------------------------------------------------------------------------- resolving


@pytest.mark.asyncio
async def test_a_header_token_resolves_as_its_person_by_users_id(db_session, install):
    issued = _issue(db_session, install.person)

    ctx = await _resolve(_bearer(issued.token))

    assert ctx.auth_type == "session" and ctx.workspace_id == install.workspace_id
    assert claimed_user_pk(ctx) == install.person and ctx.user.system_role == "user"
    assert ctx.user.raw_claims["session_id"] == str(issued.session_id)
    assert session_lane.session_auth_time(ctx) == pytest.approx(_row(db_session, issued.session_id)
                                                                  .auth_time.timestamp())
    assert install.lane.closes == 1


@pytest.mark.asyncio
async def test_a_same_origin_cookie_resolves_and_a_cross_site_one_is_refused_unread(db_session, install,
                                                                                     monkeypatch):
    issued = _issue(db_session, install.person)
    cookie = {"cookie": f"{config.SESSION_COOKIE_NAME}={issued.token}"}

    ctx = await _resolve(_request("POST", {**cookie, "origin": INSTALL, config.SESSION_CSRF_HEADER.lower(): "1"}))
    assert claimed_user_pk(ctx) == install.person

    monkeypatch.setattr(session_lane, "SessionLocal", lambda: pytest.fail("a cross-site request is never looked up"))
    refused = await _refused(_request("POST", {**cookie, "origin": "https://elsewhere.example.test"}))
    assert refused.status_code == 403


async def _ended(request) -> str:
    refused = await _refused(request)
    assert refused.status_code == 401 and refused.detail["code"] == "session_ended"
    return refused.detail["reason"]


@pytest.mark.asyncio
async def test_an_ended_session_is_401_with_its_reason_and_no_token_in_the_log(db_session, install, caplog):
    idle_gone = _issue(db_session, install.person, at=_now() - timedelta(minutes=config.SESSION_IDLE_MINUTES + 1))
    revoked = _issue(db_session, install.person)
    sessions.revoke_session(db_session, revoked.session_id, sessions.REVOKED_SIGN_OUT)
    db_session.commit()
    unknown, _ = mint_token(sessions.SESSION_TOKEN_PREFIX, sessions.SESSION_TOKEN_BYTES)

    with caplog.at_level(logging.DEBUG):
        reasons = [await _ended(_bearer(token)) for token in (idle_gone.token, revoked.token, unknown)]

    assert reasons == ["expired", "revoked", "unknown"]
    assert not any(token in caplog.text for token in (idle_gone.token, revoked.token, unknown))
    assert install.lane.closes == 3


@pytest.mark.asyncio
async def test_the_absolute_clock_ends_a_session_however_busy(db_session, install):
    issued = _issue(db_session, install.person, at=_now() - timedelta(hours=config.SESSION_ABSOLUTE_HOURS))
    db_session.execute(text("UPDATE auth_sessions SET idle_expires_at = now() + interval '1 hour' WHERE id = :id"),
                       {"id": issued.session_id})

    assert await _ended(_bearer(issued.token)) == "expired"


@pytest.mark.asyncio
async def test_a_deactivated_person_is_refused_on_their_live_session(db_session, install):
    issued = _issue(db_session, install.person)
    db_session.execute(text("UPDATE users SET is_active = false WHERE id = :id"), {"id": install.person})

    assert await _ended(_bearer(issued.token)) == "revoked"


@pytest.mark.asyncio
async def test_the_workspace_rules(db_session, install, seed_workspace):
    outsider = _seed_person(db_session)
    admin = _seed_person(db_session, role="admin")
    other = seed_workspace()
    db_session.commit()
    outsider_token, admin_token = _issue(db_session, outsider).token, _issue(db_session, admin).token

    unnamed = await _refused(_bearer(outsider_token))
    assert unnamed.status_code == 403 and unnamed.detail["code"] == "workspace_required"
    named = await _refused(_bearer(_issue(db_session, install.person).token, **{"x-workspace-id": other}))
    assert (named.status_code, named.detail) == (403, session_lane.NOT_A_MEMBER)
    everywhere = await _refused(_bearer(outsider_token, **{"x-workspace-id": "__all__"}))
    assert everywhere.status_code == 403

    ctx = await _resolve(_bearer(admin_token, **{"x-workspace-id": other}))
    assert ctx.workspace_id == uuid.UUID(other) and ctx.user.system_role == "admin"
    ctx = await _resolve(_bearer(admin_token, **{"x-workspace-id": "__all__"}))
    assert ctx.admin_all_workspaces is True


@pytest.mark.asyncio
async def test_last_seen_and_the_idle_clock_move_at_most_every_five_minutes(db_session, install, monkeypatch):
    start = _now() - timedelta(minutes=1)
    issued = _issue(db_session, install.person, at=start)

    monkeypatch.setattr(session_lane, "_now", lambda: start + timedelta(minutes=4))
    await _resolve(_bearer(issued.token))
    row = _row(db_session, issued.session_id)
    assert row.last_seen_at == start and row.idle_expires_at == start + sessions.idle_lifetime()

    later = start + timedelta(minutes=6)
    monkeypatch.setattr(session_lane, "_now", lambda: later)
    await _resolve(_bearer(issued.token))
    row = _row(db_session, issued.session_id)
    assert row.last_seen_at == later and row.idle_expires_at == later + sessions.idle_lifetime()
    assert row.absolute_expires_at == start + sessions.absolute_lifetime()


def test_the_idle_clock_never_passes_the_absolute_expiry(db_session, install):
    start = _now() - timedelta(hours=config.SESSION_ABSOLUTE_HOURS) + timedelta(minutes=30)
    issued = _issue(db_session, install.person, at=start)
    row = _row(db_session, issued.session_id)

    sessions.touch_session(db_session, row, start + timedelta(hours=config.SESSION_ABSOLUTE_HOURS - 1))

    assert _row(db_session, issued.session_id).idle_expires_at == row.absolute_expires_at


# --------------------------------------------------------------------------- ending


def _requester(install, user_id=None):
    return people.Requester(user_id=user_id, workspace_id=install.workspace_id, ip_address="203.0.113.9")


def test_revoking_ends_only_what_is_live(db_session, install):
    one, two = _issue(db_session, install.person), _issue(db_session, install.person)

    assert sessions.revoke_session(db_session, one.session_id, sessions.REVOKED_BY_ADMIN) is True
    assert sessions.revoke_session(db_session, one.session_id, sessions.REVOKED_BY_ADMIN) is False
    assert sessions.revoke_sessions_of(db_session, install.person, sessions.REVOKED_BY_ADMIN) == 1
    assert sessions.revoke_sessions_of(db_session, install.person, sessions.REVOKED_BY_ADMIN) == 0
    assert {_row(db_session, s.session_id).revoked_reason for s in (one, two)} == {"admin"}


def test_sign_out_writes_one_row_and_ending_nothing_writes_none(db_session, install):
    issued = _issue(db_session, install.person)
    before = len(_audits(db_session, install.workspace_id))

    assert people.sign_out(db_session, issued.session_id, _requester(install, install.person)) is True
    assert people.sign_out(db_session, issued.session_id, _requester(install, install.person)) is False

    assert _audits(db_session, install.workspace_id)[before:] == [
        ("auth:sign_out", install.person, "auth_session", str(issued.session_id), {}, "203.0.113.9", None),
    ]
    assert _row(db_session, issued.session_id).revoked_reason == "sign_out"


def test_ending_every_session_of_a_person_writes_one_row(db_session, install):
    admin = _seed_person(db_session, role="admin")
    for _ in range(2):
        _issue(db_session, install.person)
    before = len(_audits(db_session, install.workspace_id))

    assert people.end_sessions_of(db_session, install.person, _requester(install, admin)) == 2

    assert _audits(db_session, install.workspace_id)[before:] == [
        ("auth:session_revoked", admin, "user", str(install.person), {"sessions": 2}, "203.0.113.9", None),
    ]


def _device(db, install) -> MobileDevice:
    token, digest = mint_token("amd_", 32)
    row = MobileDevice(workspace_id=install.workspace_id, user_id=install.person, kind="paired", status="active",
                       install_id=f"prd277-w3-{uuid.uuid4().hex[:12]}", platform="ios", token_hash=digest,
                       paired_at=_now(), events=[])
    db.add(row)
    db.flush()
    return row


def test_deactivating_ends_the_person_their_sessions_and_devices_in_one_commit(db_session, install):
    admin = _seed_person(db_session, role="admin")
    issued = [_issue(db_session, install.person) for _ in range(2)]
    device = _device(db_session, install)
    before = len(_audits(db_session, install.workspace_id))

    done = people.deactivate_person(db_session, install.person, _requester(install, admin))

    assert (done.sessions_ended, done.devices_revoked) == (2, 1)
    db_session.expire_all()
    assert db_session.execute(text("SELECT is_active FROM users WHERE id = :id"),
                              {"id": install.person}).scalar_one() is False
    assert {_row(db_session, s.session_id).revoked_reason for s in issued} == {"deactivated"}
    assert db_session.get(MobileDevice, device.id).status == "revoked"
    assert _audits(db_session, install.workspace_id)[before:] == [
        ("user:deactivated", admin, "user", str(install.person), {"sessions": 2, "devices": 1}, "203.0.113.9", None),
    ]


def test_a_removal_lands_even_when_its_row_cannot_be_written(db_session, install, monkeypatch, caplog):
    issued = _issue(db_session, install.person)
    monkeypatch.setattr(session_audit, "AuditService", _RefusingAudit)

    with caplog.at_level(logging.ERROR, logger="core.auth.session_audit"):
        assert people.end_sessions_of(db_session, install.person, _requester(install)) == 1

    assert _row(db_session, issued.session_id).revoked_at is not None
    assert "auth:session_revoked" in caplog.text and issued.token not in caplog.text


# --------------------------------------------------------------------------- the routes


@pytest.fixture
def client(db_session, install):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)


def _cleared(answer) -> bool:
    cookie = answer.headers.get("set-cookie", "")
    return cookie.startswith(config.SESSION_COOKIE_NAME + "=") and "max-age=0" in cookie.lower()


def test_sign_out_by_header_is_204_then_401_and_clears_the_cookie_both_times(client, db_session, install):
    issued = _issue(db_session, install.person)
    headers = {"authorization": "Bearer " + issued.token}

    first = client.post("/api/v1/auth/sign-out", headers=headers)
    again = client.post("/api/v1/auth/sign-out", headers=headers)

    assert first.status_code == 204 and _cleared(first)
    assert again.status_code == 401 and _cleared(again)
    assert again.json()["detail"]["code"] == "session_ended" and again.json()["detail"]["reason"] == "revoked"
    assert [a[0] for a in _audits(db_session, install.workspace_id)] == ["auth:sign_in", "auth:sign_out"]


def test_sign_out_by_cookie_needs_the_install_origin(client, db_session, install):
    issued = _issue(db_session, install.person)
    cookie = {"cookie": f"{config.SESSION_COOKIE_NAME}={issued.token}"}

    cross_site = client.post("/api/v1/auth/sign-out", headers={**cookie, "origin": "https://elsewhere.example.test"})
    assert cross_site.status_code == 403 and _row(db_session, issued.session_id).revoked_at is None

    same_origin = client.post("/api/v1/auth/sign-out",
                              headers={**cookie, "origin": INSTALL, config.SESSION_CSRF_HEADER: "1"})
    assert same_origin.status_code == 204 and _row(db_session, issued.session_id).revoked_reason == "sign_out"


@pytest.fixture
def admin(db_session, install, monkeypatch):
    """An install admin's session, with step-up's seam passing (Wave 4 fills it, US-406)."""
    admin_pk = _seed_person(db_session, role="admin")
    db_session.commit()
    monkeypatch.setattr(step_up, "confirm_by_signing_in_again", lambda ctx, max_age, now: None)
    return SimpleNamespace(pk=admin_pk, headers={"authorization": "Bearer " + _issue(db_session, admin_pk).token})


def test_an_admin_route_needs_step_up(client, db_session, install):
    admin_pk = _seed_person(db_session, role="admin")
    victim = _issue(db_session, install.person)
    headers = {"authorization": "Bearer " + _issue(db_session, admin_pk).token}

    answer = client.post(f"/api/v1/auth/sessions/{victim.session_id}/revoke", headers=headers)

    assert answer.status_code == 403 and answer.json()["detail"]["code"] == step_up.STEP_UP_UNAVAILABLE_CODE
    assert _row(db_session, victim.session_id).revoked_at is None


def test_an_admin_ends_one_session_then_every_session_and_lists_them(client, db_session, install, admin):
    one, two = _issue(db_session, install.person), _issue(db_session, install.person)

    assert client.post(f"/api/v1/auth/sessions/{one.session_id}/revoke", headers=admin.headers).json() == {
        "revoked": True}
    assert client.post(f"/api/v1/auth/people/{install.person}/sessions/revoke", headers=admin.headers).json() == {
        "revoked": 1}
    listed = client.get(f"/api/v1/auth/people/{install.person}/sessions", headers=admin.headers).json()["sessions"]

    assert {entry["id"] for entry in listed} == {str(one.session_id), str(two.session_id)}
    assert {entry["state"] for entry in listed} == {"revoked"}
    assert not any(one.token in str(entry) or "token" in "".join(entry) for entry in listed)
    assert _row(db_session, two.session_id).revoked_reason == "admin"


def test_an_admin_deactivates_a_person_but_not_themselves_or_a_higher_role(client, db_session, install, admin):
    _issue(db_session, install.person)
    owner = _seed_person(db_session, role="super_admin")
    db_session.commit()

    assert client.post(f"/api/v1/auth/people/{admin.pk}/deactivate", headers=admin.headers).status_code == 409
    assert client.post(f"/api/v1/auth/people/{owner}/deactivate", headers=admin.headers).status_code == 403
    assert client.post("/api/v1/auth/people/2000000000/deactivate", headers=admin.headers).status_code == 404
    answer = client.post(f"/api/v1/auth/people/{install.person}/deactivate", headers=admin.headers)

    assert answer.status_code == 200 and answer.json() == {"sessions_revoked": 1, "devices_revoked": 0}


def test_a_member_cannot_end_sessions(client, db_session, install, admin):
    member_headers = {"authorization": "Bearer " + _issue(db_session, install.person).token}

    answer = client.post(f"/api/v1/auth/people/{admin.pk}/sessions/revoke", headers=member_headers)

    assert answer.status_code == 403 and answer.json()["detail"] == routes.ADMINS_ONLY


# --------------------------------------------------------------------------- hardening

LONG_AGENT = "Browser/" + "x" * 800
JUNK_ADDRESS = "not-an-address-" + "9" * 60


def _oversized(token: str) -> dict:
    return {"authorization": "Bearer " + token, "user-agent": LONG_AGENT, "x-forwarded-for": JUNK_ADDRESS}


def _only_row(db, workspace_id, before: int, action: str):
    rows = _audits(db, workspace_id)[before:]
    assert [row[0] for row in rows] == [action]
    return rows[0]


def test_an_oversized_agent_and_a_junk_address_still_write_one_sign_out_row(client, db_session, install,
                                                                            monkeypatch):
    monkeypatch.setattr(config, "CLIENT_IP_HEADER", "")
    issued = _issue(db_session, install.person)
    before = len(_audits(db_session, install.workspace_id))

    assert client.post("/api/v1/auth/sign-out", headers=_oversized(issued.token)).status_code == 204

    row = _only_row(db_session, install.workspace_id, before, "auth:sign_out")
    assert (row[5], row[6]) == (None, LONG_AGENT[:500])


def test_an_oversized_agent_and_a_junk_address_still_write_one_admin_row(client, db_session, install, admin,
                                                                        monkeypatch):
    monkeypatch.setattr(config, "CLIENT_IP_HEADER", "")
    victim = _issue(db_session, install.person)
    before = len(_audits(db_session, install.workspace_id))
    headers = {**admin.headers, "user-agent": LONG_AGENT, "x-forwarded-for": JUNK_ADDRESS}

    answer = client.post(f"/api/v1/auth/sessions/{victim.session_id}/revoke", headers=headers)

    assert answer.json() == {"revoked": True}
    row = _only_row(db_session, install.workspace_id, before, "auth:session_revoked")
    assert (row[1], row[5], row[6]) == (admin.pk, None, LONG_AGENT[:500])


def test_a_sign_in_keeps_a_real_address_and_drops_a_junk_one(db_session, install):
    kept = _issue(db_session, install.person, ip_address=" 2001:db8::1 ", user_agent=LONG_AGENT)
    dropped = _issue(db_session, install.person, ip_address=JUNK_ADDRESS)

    assert (_row(db_session, kept.session_id).ip_address, _row(db_session, kept.session_id).user_agent) == (
        "2001:db8::1", LONG_AGENT[:500])
    assert _row(db_session, dropped.session_id).ip_address is None
    assert [row[5] for row in _audits(db_session, install.workspace_id)] == ["2001:db8::1", None]


def test_a_sign_in_reads_the_person_for_share_before_inserting(db_session, install):
    statements = []
    connection = db_session.connection()

    def listener(conn, cursor, statement, params, context, many):
        statements.append(statement)

    event.listen(connection, "before_cursor_execute", listener)
    try:
        _issue(db_session, install.person)
    finally:
        event.remove(connection, "before_cursor_execute", listener)

    locked = [i for i, sql in enumerate(statements) if "FROM users" in sql and "FOR SHARE" in sql]
    inserted = [i for i, sql in enumerate(statements) if sql.lstrip().upper().startswith("INSERT INTO AUTH_SESSIONS")]
    assert locked and inserted and locked[0] < inserted[0]


def test_a_failed_insert_raises_a_sign_in_failure_that_names_no_value(db_session, install, monkeypatch, caplog):
    first = _issue(db_session, install.person)
    taken = _row(db_session, first.session_id).token_hash
    fresh, _ = mint_token(sessions.SESSION_TOKEN_PREFIX, sessions.SESSION_TOKEN_BYTES)
    monkeypatch.setattr(sessions, "mint_token", lambda prefix, nbytes: (fresh, taken))
    before = len(_audits(db_session, install.workspace_id))

    with caplog.at_level(logging.DEBUG, logger="core.auth"):
        with pytest.raises(sessions.SignInFailed) as failed:
            _issue(db_session, install.person, ip_address="198.51.100.23", user_agent="Agent/marker-77")

    raised = failed.value
    assert raised.__cause__ is None and raised.__context__ is None
    for value in (taken, fresh, "198.51.100.23", "Agent/marker-77"):
        assert value not in str(raised) and value not in caplog.text
    assert db_session.query(AuthSession).filter(AuthSession.user_id == install.person).count() == 1
    assert len(_audits(db_session, install.workspace_id)) == before
