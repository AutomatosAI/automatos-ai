"""PRD-277 US-403, US-405, US-406 and §5.6 against Postgres and the fake provider.

Through the real routes (``api/auth_oidc.py``, ``api/auth_sessions.py``), a browser (``TestClient``
on the install's address) and the in-process provider, each test in a transaction rolled back:

* **The access rule** (US-403): a verified token that breaks ``OIDC_REQUIRED_CLAIMS`` is refused
  (``sign_in_not_allowed``), audited ``access_rule``, and leaves no person, link or session; a
  parameter on the callback is never a claim; a value broken after start-up refuses everyone. It
  runs before any person is found or created (a test fails if that order changes).
* **The first admins** (US-405): a person ``ENTERPRISE_BOOTSTRAP_ADMINS`` names by subject, or by an
  email the token marks verified, becomes an install admin and owner of the default workspace at
  first sign-in, audited ``auth:bootstrap_admin``; an unverified email, a repeat sign-in and a
  person the access rule refuses get nothing. The migration Job's seed makes the default
  workspace with no owner, once, and only where sign-in is OIDC.
* **Step-up** (US-406): ``/start?step_up=1`` needs a live session, sends ``max_age=0`` and keeps
  the session's id; the callback refreshes that session's ``auth_time`` for the same person only,
  issues no session and sets no cookie, audited ``auth:step_up``. Another person, a token with no
  or an old ``auth_time``, and an ended session are refused, changing nothing and creating nobody;
  the step-up runs before anything that finds or creates a person.
* **Sign-out at the provider** (§5.6): with an end-session address the answer is 200 with that
  address, carrying only ``client_id`` and ``post_logout_redirect_uri``; without one, or with the
  provider unreachable, it is 204. Either way the session ends here.
* **Nothing secret is logged or answered** over a step-up that confirms, one that is refused, and
  a sign-out at the provider.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import auth_oidc as routes  # noqa: E402
from api import auth_sessions  # noqa: E402
from config import config  # noqa: E402
from core.auth import first_admin, hybrid, oidc_provider, people, session_lane, sign_in_flow  # noqa: E402
from core.auth import step_up, step_up_sign_in  # noqa: E402
from core.auth.oidc_sign_in import OidcSignIn, SIGNED_OUT_PATH, STEP_UP_MAX_AGE  # noqa: E402
from core.auth.pairing import hash_secret  # noqa: E402
from core.auth.sign_in_methods import SignInRegistry  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.seeds.seed_default_workspace import DEFAULT_WORKSPACE_NAME, seed_default_workspace  # noqa: E402
from tests.helpers_fake_oidc_provider import INSTALL, FakeIdentityProvider, Person, point_install_at  # noqa: E402

START = "/api/v1/auth/oidc/start"
SIGN_OUT = "/api/v1/auth/sign-out"


class _BorrowedSession:
    """The test's session, lent to the session lane: ``close()`` is counted, not done."""

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
def install(db_session, seed_workspace, monkeypatch, tmp_path):
    """An enterprise install signing in at the fake provider, with its default workspace."""
    idp = FakeIdentityProvider(tmp_path)
    workspace_id = seed_workspace()
    db_session.commit()
    point_install_at(monkeypatch, idp, DEFAULT_WORKSPACE_ID=workspace_id)
    monkeypatch.setattr(routes, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 300, 299, 600)))
    monkeypatch.setattr(sign_in_flow, "REGISTRY", SignInRegistry({"oidc": OidcSignIn()}))
    monkeypatch.setattr(session_lane, "SessionLocal", lambda: _BorrowedSession(db_session))
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(auth_sessions.router)
    app.dependency_overrides[get_db] = lambda: db_session
    yield SimpleNamespace(idp=idp, client=TestClient(app, base_url=INSTALL), workspace_id=workspace_id,
                          db=db_session)
    idp.close()
    oidc_provider.forget_provider()


def _query(address: str) -> dict:
    return {name: values[0] for name, values in parse_qs(urlsplit(address).query).items()}


def _start(install, step_up_too=False, return_to="/settings"):
    params = {"return_to": return_to, **({"step_up": "1"} if step_up_too else {})}
    started = install.client.get(START, params=params, follow_redirects=False)
    assert started.status_code == 303, started.text
    return started


def _sign_in(install, step_up_too=False, return_to="/settings"):
    """Start, approve at the provider, and send the browser back; the callback's answer."""
    callback = install.idp.approve(_start(install, step_up_too, return_to).headers["location"])
    return install.client.get(callback, follow_redirects=False)


def _refusal(answer) -> str:
    assert answer.status_code == 303
    location = urlsplit(answer.headers["location"])
    assert location.path == routes.SIGN_IN_PAGE and not location.netloc
    return parse_qs(location.query)["error"][0]


def _identity_user(db, subject):
    row = db.execute(text("SELECT user_id FROM user_identities WHERE subject = :s"), {"s": subject}).fetchone()
    return row[0] if row else None


def _count(db, sql: str, **params) -> int:
    return db.execute(text(sql), params).scalar_one()


def _audits(db, workspace_id) -> list:
    return db.execute(text("SELECT action, user_id, details FROM audit_logs WHERE workspace_id = CAST(:ws AS uuid) "
                           "AND action LIKE 'auth:%' ORDER BY id"), {"ws": str(workspace_id)}).all()


def _sets_session_cookie(answer) -> bool:
    return any(line.startswith(config.SESSION_COOKIE_NAME + "=") for line in answer.headers.get_list("set-cookie"))


def _spy(monkeypatch, module, name) -> list:
    """Record each call of ``module.name``, still calling it."""
    calls, real = [], getattr(module, name)

    def recorded(*args, **kwargs):
        calls.append(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, recorded)
    return calls


# --------------------------------------------------------------------------- the access rule (US-403)


def test_a_token_the_rule_refuses_leaves_no_person_and_is_audited(install, monkeypatch):
    monkeypatch.setattr(config, "OIDC_REQUIRED_CLAIMS", "groups=automatos-users")
    install.idp.sign_in_as(Person(subject="outsider", extra={"groups": ["finance"]}))
    users = _count(install.db, "SELECT COUNT(*) FROM users")
    linked = _spy(monkeypatch, people, "link_person")

    assert _refusal(_sign_in(install)) == sign_in_flow.NOT_ALLOWED

    assert linked == []  # refused before anyone was found or created
    assert _identity_user(install.db, "outsider") is None and _count(install.db, "SELECT COUNT(*) FROM users") == users
    assert _count(install.db, "SELECT COUNT(*) FROM auth_sessions") == 0
    assert [(a, uid, d) for a, uid, d in _audits(install.db, install.workspace_id)] == [
        ("auth:sign_in_refused", None, {"reason": "access_rule"})]


def test_a_token_every_rule_holds_for_signs_in(install, monkeypatch):
    monkeypatch.setattr(config, "OIDC_REQUIRED_CLAIMS", "cognito:groups=automatos; https://example.test/dept=ops")
    install.idp.sign_in_as(Person(subject="insider", extra={"cognito:groups": ["automatos", "x"],
                                                            "https://example.test/dept": "ops"}))

    answer = _sign_in(install)

    assert answer.status_code == 303 and answer.headers["location"] == "/settings"
    assert _identity_user(install.db, "insider") is not None and _sets_session_cookie(answer)


def test_a_callback_parameter_is_never_a_claim(install, monkeypatch):
    monkeypatch.setattr(config, "OIDC_REQUIRED_CLAIMS", "groups=automatos-users")
    callback = install.idp.approve(_start(install).headers["location"])

    answer = install.client.get(callback + "&" + urlencode({"groups": "automatos-users"}), follow_redirects=False)

    assert _refusal(answer) == sign_in_flow.NOT_ALLOWED


def test_a_person_already_known_is_refused_once_the_rule_excludes_them_naming_nobody(install, monkeypatch):
    install.idp.sign_in_as(Person(subject="moved-team", extra={"groups": ["old-team"]}))
    _sign_in(install)
    user_id = _identity_user(install.db, "moved-team")
    monkeypatch.setattr(config, "OIDC_REQUIRED_CLAIMS", "groups=automatos-users")

    assert _refusal(_sign_in(install)) == sign_in_flow.NOT_ALLOWED
    assert _count(install.db, "SELECT COUNT(*) FROM auth_sessions WHERE user_id = :u", u=user_id) == 1
    assert _audits(install.db, install.workspace_id)[-1] == ("auth:sign_in_refused", None, {"reason": "access_rule"})


def test_a_rule_broken_after_start_up_refuses_everyone_and_is_not_logged(install, monkeypatch, caplog):
    monkeypatch.setattr(config, "OIDC_REQUIRED_CLAIMS", "groups-without-a-value")
    with caplog.at_level(logging.DEBUG):
        assert _refusal(_sign_in(install)) == sign_in_flow.NOT_ALLOWED
    assert "groups-without-a-value" not in caplog.text


# --------------------------------------------------------------------------- the first admins (US-405)


def _system_role(db, user_id):
    return db.execute(text("SELECT system_role FROM users WHERE id = :u"), {"u": user_id}).scalar_one()


def _roles_in(db, user_id, workspace_id) -> list:
    return [row[0] for row in db.execute(text(
        "SELECT role FROM workspace_members WHERE user_id = :u AND workspace_id = CAST(:ws AS uuid)"),
        {"u": user_id, "ws": str(workspace_id)})]


def _owner_of(db, workspace_id):
    return db.execute(text("SELECT owner_id FROM workspaces WHERE id = CAST(:ws AS uuid)"),
                      {"ws": str(workspace_id)}).scalar_one()


@pytest.mark.parametrize(("entry", "person", "matched_by"), [
    ("sub:first-admin-subject", Person(subject="first-admin-subject"), "sub"),
    ("email:First.Admin@Example.Test", Person(subject="by-email", email="First.Admin@example.test",
                                               email_verified=True), "email"),
])
def test_a_listed_person_becomes_install_admin_and_owner_at_first_sign_in(install, monkeypatch, entry, person,
                                                                          matched_by):
    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "sub:somebody-else, " + entry)
    install.idp.sign_in_as(person)

    assert _sign_in(install).status_code == 303

    user_id = _identity_user(install.db, person.subject)
    assert _system_role(install.db, user_id) == "admin"
    assert _roles_in(install.db, user_id, install.workspace_id) == ["owner"]
    assert _owner_of(install.db, install.workspace_id) == user_id
    granted = [(uid, d) for a, uid, d in _audits(install.db, install.workspace_id) if a == "auth:bootstrap_admin"]
    assert granted == [(user_id, {"matched_by": matched_by, "system_role": "admin", "workspace_role": "owner"})]


@pytest.mark.parametrize("verified", [False, None, "true"])
def test_an_email_the_token_does_not_mark_verified_makes_nobody_admin(install, monkeypatch, verified):
    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "email:claims-to-be@example.test")
    install.idp.sign_in_as(Person(subject="unverified", email="claims-to-be@example.test", email_verified=verified))

    _sign_in(install)

    user_id = _identity_user(install.db, "unverified")
    assert _system_role(install.db, user_id) != "admin" and _roles_in(install.db, user_id, install.workspace_id) == []
    assert "auth:bootstrap_admin" not in [a for a, _uid, _d in _audits(install.db, install.workspace_id)]


def test_a_repeat_sign_in_never_makes_an_admin(install, monkeypatch):
    install.idp.sign_in_as(Person(subject="listed-too-late"))
    _sign_in(install)
    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "sub:listed-too-late")
    granted = _spy(monkeypatch, first_admin, "grant_if_listed")

    assert _sign_in(install).status_code == 303

    assert granted == [] and _system_role(install.db, _identity_user(install.db, "listed-too-late")) != "admin"


def test_a_listed_person_the_access_rule_refuses_gets_nothing(install, monkeypatch):
    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "sub:listed-but-outside")
    monkeypatch.setattr(config, "OIDC_REQUIRED_CLAIMS", "groups=automatos-users")
    install.idp.sign_in_as(Person(subject="listed-but-outside"))
    granted = _spy(monkeypatch, first_admin, "grant_if_listed")

    assert _refusal(_sign_in(install)) == sign_in_flow.NOT_ALLOWED
    assert granted == [] and _identity_user(install.db, "listed-but-outside") is None
    assert _owner_of(install.db, install.workspace_id) is None


def test_the_first_admin_makes_the_default_workspace_when_the_job_has_not(install, monkeypatch):
    missing = str(uuid.uuid4())
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", missing)
    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "sub:early-admin")
    install.idp.sign_in_as(Person(subject="early-admin"))

    assert _sign_in(install).status_code == 303

    user_id = _identity_user(install.db, "early-admin")
    assert _owner_of(install.db, missing) == user_id and _roles_in(install.db, user_id, missing) == ["owner"]


def test_the_job_seeds_the_default_workspace_with_no_owner_once_and_only_for_oidc(install, monkeypatch):
    fresh = str(uuid.uuid4())
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", fresh)

    assert seed_default_workspace(install.db) == {"workspace_id": fresh, "workspace": "created"}
    assert seed_default_workspace(install.db) == {"workspace_id": fresh, "workspace": "present"}
    row = install.db.execute(text("SELECT name, owner_id, is_personal FROM workspaces WHERE id = CAST(:ws AS uuid)"),
                             {"ws": fresh}).one()
    assert tuple(row) == (DEFAULT_WORKSPACE_NAME, None, False)
    for edition in ("local", "saas"):
        monkeypatch.setattr(config, "AUTH_EDITION", edition)
        monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", str(uuid.uuid4()))
        assert "skipped" in seed_default_workspace(install.db)
        assert _count(install.db, "SELECT COUNT(*) FROM workspaces WHERE id = CAST(:ws AS uuid)",
                      ws=config.DEFAULT_WORKSPACE_ID) == 0


# --------------------------------------------------------------------------- step-up (US-406)


def _signed_in(install, subject) -> SimpleNamespace:
    """``subject`` signed in, their session's ``auth_time`` two hours old; the cookie is in the jar."""
    install.idp.sign_in_as(Person(subject=subject))
    answer = _sign_in(install)
    token = answer.cookies[config.SESSION_COOKIE_NAME]
    session_id = install.db.execute(text("SELECT id FROM auth_sessions WHERE token_hash = :h"),
                                    {"h": hash_secret(token)}).scalar_one()
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    install.db.execute(text("UPDATE auth_sessions SET auth_time = :t WHERE id = :id"), {"t": old, "id": session_id})
    install.db.commit()
    return SimpleNamespace(token=token, session_id=session_id, user_id=_identity_user(install.db, subject), old=old)


def _auth_time(db, session_id):
    db.expire_all()
    return db.execute(text("SELECT auth_time FROM auth_sessions WHERE id = :id"), {"id": session_id}).scalar_one()


def _step_ups(install) -> list:
    return [(uid, d) for a, uid, d in _audits(install.db, install.workspace_id) if a == "auth:step_up"]


def test_the_step_up_start_named_in_the_403_is_the_step_up_route():
    assert step_up.SIGN_IN_AGAIN_START == step_up_sign_in.STEP_UP_START == START + "?step_up=1"


def test_a_step_up_without_a_live_session_is_refused_and_stores_nothing(install):
    before = _count(install.db, "SELECT COUNT(*) FROM oidc_pending_sign_ins")

    answer = install.client.get(START, params={"step_up": "1"}, follow_redirects=False)

    assert _refusal(answer) == sign_in_flow.STEP_UP_REFUSED
    assert _count(install.db, "SELECT COUNT(*) FROM oidc_pending_sign_ins") == before
    assert install.idp.authorize_requests == []


def test_a_step_up_sends_max_age_0_and_refreshes_only_the_asking_session(install):
    me = _signed_in(install, "steps-up")
    assert "max_age" not in install.idp.authorize_requests[-1]  # a plain sign-in never asks for it

    started = _start(install, step_up_too=True)
    assert _query(started.headers["location"])["max_age"] == STEP_UP_MAX_AGE == "0"
    state = _query(started.headers["location"])["state"]
    kept = install.db.execute(text("SELECT step_up_session_id FROM oidc_pending_sign_ins WHERE state_hash = :h"),
                              {"h": hash_secret(state)}).scalar_one()
    assert kept == me.session_id
    answer = install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False)

    assert answer.status_code == 303 and answer.headers["location"] == "/settings"
    assert not _sets_session_cookie(answer)
    assert _count(install.db, "SELECT COUNT(*) FROM auth_sessions WHERE user_id = :u", u=me.user_id) == 1
    assert _auth_time(install.db, me.session_id) > datetime.now(timezone.utc) - timedelta(minutes=2)
    assert _step_ups(install) == [(me.user_id, {"confirmed": True})]


def test_another_person_signing_in_is_refused_and_never_switches_the_session(install, monkeypatch):
    install.idp.sign_in_as(Person(subject="someone-known"))
    _sign_in(install)
    me = _signed_in(install, "the-asker")
    users = _count(install.db, "SELECT COUNT(*) FROM users")
    linked = _spy(monkeypatch, people, "link_person")
    granted = _spy(monkeypatch, first_admin, "grant_if_listed")

    for other in ("someone-known", "never-seen-before"):
        started = _start(install, step_up_too=True)
        install.idp.sign_in_as(Person(subject=other))
        answer = install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False)
        assert _refusal(answer) == sign_in_flow.STEP_UP_REFUSED and not _sets_session_cookie(answer)

    assert linked == [] and granted == []  # the step-up ran before anything that finds or creates a person
    assert _identity_user(install.db, "never-seen-before") is None
    assert _count(install.db, "SELECT COUNT(*) FROM users") == users
    assert _auth_time(install.db, me.session_id) == me.old
    assert _step_ups(install) == [(me.user_id, {"confirmed": False, "reason": "another_person"})] * 2


@pytest.mark.parametrize(("behaviour", "reason"), [
    ({"remove_claims": ("auth_time",)}, "no_auth_time"),
    ({"claims": {"auth_time": int(time.time()) - 3600}}, "stale_auth_time"),
])
def test_a_token_that_does_not_prove_a_fresh_sign_in_is_refused(install, behaviour, reason):
    me = _signed_in(install, "not-fresh")
    started = _start(install, step_up_too=True)
    install.idp.sign_in_as(Person(subject="not-fresh"))
    install.idp.misbehave(**behaviour)

    answer = install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False)

    assert _refusal(answer) == sign_in_flow.STEP_UP_REFUSED
    assert _auth_time(install.db, me.session_id) == me.old
    assert _step_ups(install) == [(me.user_id, {"confirmed": False, "reason": reason})]


def test_a_session_that_ended_during_the_step_up_is_not_refreshed(install):
    me = _signed_in(install, "ended-meanwhile")
    started = _start(install, step_up_too=True)
    install.db.execute(text("UPDATE auth_sessions SET revoked_at = now() WHERE id = :id"), {"id": me.session_id})
    install.db.commit()

    answer = install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False)

    assert _refusal(answer) == sign_in_flow.STEP_UP_REFUSED
    assert _auth_time(install.db, me.session_id) == me.old
    assert _step_ups(install) == [(me.user_id, {"confirmed": False, "reason": "session_ended"})]


def test_the_access_rule_applies_to_a_step_up_too(install, monkeypatch):
    me = _signed_in(install, "rule-on-step-up")
    monkeypatch.setattr(config, "OIDC_REQUIRED_CLAIMS", "groups=automatos-users")

    assert _refusal(_sign_in(install, step_up_too=True)) == sign_in_flow.NOT_ALLOWED
    assert _auth_time(install.db, me.session_id) == me.old and _step_ups(install) == []


# --------------------------------------------------------------------------- sign-out at the provider (§5.6)


def test_sign_out_sends_the_browser_to_the_provider_with_only_the_allowed_parameters(install):
    me = _signed_in(install, "signs-out")

    answer = install.client.post(SIGN_OUT, headers={"authorization": "Bearer " + me.token})

    assert answer.status_code == 200 and answer.headers["cache-control"] == "no-store"
    redirect_to = answer.json()["redirect_to"]
    parts = urlsplit(redirect_to)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == install.idp.address("/logout")
    assert _query(redirect_to) == {"client_id": install.idp.client_id,
                                   "post_logout_redirect_uri": INSTALL + SIGNED_OUT_PATH}
    assert me.token not in redirect_to and not any(issued in redirect_to for issued in install.idp.issued)
    cleared = [line for line in answer.headers.get_list("set-cookie") if line.startswith(config.SESSION_COOKIE_NAME)]
    assert cleared and "max-age=0" in cleared[0].lower()
    assert _count(install.db, "SELECT COUNT(*) FROM auth_sessions WHERE id = :id AND revoked_at IS NOT NULL",
                  id=me.session_id) == 1


@pytest.mark.parametrize("provider", ["no-end-session-address", "unreachable"])
def test_without_an_end_session_address_sign_out_is_204_and_still_ends_the_session(install, provider):
    me = _signed_in(install, "signs-out-here")
    if provider == "unreachable":
        install.idp.close()
    else:
        install.idp.misbehave(metadata={"end_session_endpoint": None})
    oidc_provider.forget_provider()

    answer = install.client.post(SIGN_OUT, headers={"authorization": "Bearer " + me.token})

    assert answer.status_code == 204 and not answer.content
    assert _count(install.db, "SELECT COUNT(*) FROM auth_sessions WHERE id = :id AND revoked_at IS NOT NULL",
                  id=me.session_id) == 1


# --------------------------------------------------------------------------- nothing secret logged


def _step_up_round(install, subject, answers) -> list:
    """One step-up signed in as ``subject``; the secret values it involved."""
    started = _start(install, step_up_too=True)
    query = _query(started.headers["location"])
    verifier = install.db.execute(text("SELECT code_verifier FROM oidc_pending_sign_ins WHERE state_hash = :h"),
                                  {"h": hash_secret(query["state"])}).scalar_one()
    install.idp.sign_in_as(Person(subject=subject))
    answers.append(install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False))
    return [query["state"], query["nonce"], started.cookies[routes.SIGN_IN_COOKIE], verifier]


def test_no_secret_reaches_a_log_line_or_an_answer_over_step_up_and_sign_out(install, caplog):
    answers, values = [], [install.idp.client_secret]
    with caplog.at_level(logging.DEBUG):
        me = _signed_in(install, "careful")
        values += _step_up_round(install, "careful", answers)        # confirmed
        values += _step_up_round(install, "an-impostor", answers)    # refused: another person
        answers.append(install.client.post(SIGN_OUT, headers={"authorization": "Bearer " + me.token}))
    assert [answer.status_code for answer in answers] == [303, 303, 200]

    formatter = logging.Formatter("%(name)s %(message)s")
    logged = "\n".join(formatter.format(record) for record in caplog.records
                       if not (record.name == "httpx" and (INSTALL in record.getMessage()
                                                           or "/authorize?" in record.getMessage())))
    for value in (*values, *install.idp.issued, me.token):
        assert value not in logged
    for answer in answers:
        shown = answer.headers.get("location", "") + answer.text
        assert not any(value in shown for value in (*values, *install.idp.issued, me.token))
