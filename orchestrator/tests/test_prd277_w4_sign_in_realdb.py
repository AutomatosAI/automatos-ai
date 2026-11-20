"""PRD-277 US-402, US-404 and US-408 against Postgres and the fake provider: signing in end to end.

Through the real routes (``api/auth_oidc.py``), a browser (``TestClient`` on the install's
address) and the in-process provider (US-407), each test in a transaction rolled back at the end:

* **Start** redirects to the provider with ``code``, the exact callback address, the scopes, a
  state, a nonce and a PKCE S256 challenge, keeps the sign-in server-side for ten minutes as
  digests, and binds it to this browser with a ``__Host-`` cookie.
* **First sign-in** creates the person and the identity link and nothing else (no personal
  workspace, no trial, no membership), is audited, and issues a session in the cookie. With
  ``ENTERPRISE_DEFAULT_ROLE=none`` the person is signed in and in no workspace, and the API says
  so; with a role, they get it in the default workspace at first sign-in only, and a role an
  admin changed survives the next sign-in. A token with only a subject and an email signs in.
  The email is the person's only when the token marks it verified with ``true``.
* **Every refusal** in US-402's list (wrong signature, ``alg: none``, wrong issuer, wrong
  audience, expired, missing or wrong nonce, reused state, replayed code, another host) ends
  without a session, at ``/sign-in?error=…``; one that is a refusal of the person is audited with
  its reason, and looks the same whether or not their address is already known.
* **US-408**: a second, fake identity method and a hook sign a person in with no change to the
  session or permission code; a hook can refuse, or set a membership through ``core.auth.people``.

* **Nothing secret is logged or answered**: over a sign-in that succeeds and over each way one
  fails (a bad signature, a wrong nonce, the token endpoint failing, the provider's own error, a
  replayed callback, an unreachable provider), no log line and no answer holds the client secret,
  a code, an ID or access token, a session token, a state, a nonce or a PKCE verifier.
"""
from __future__ import annotations

import base64
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import auth_oidc as routes  # noqa: E402
from config import config  # noqa: E402
from core.auth import hybrid, oidc_provider, people, session_lane, sign_in_flow  # noqa: E402
from core.auth.oidc_sign_in import CALLBACK_PATH, OidcSignIn  # noqa: E402
from core.auth.pairing import hash_secret  # noqa: E402
from core.auth.sign_in_methods import (  # noqa: E402
    Callback,
    Finished,
    SignInRefusedByHook,
    SignInRegistry,
    Started,
    VerifiedIdentity,
)
from core.database.database import get_db  # noqa: E402
from tests.helpers_fake_oidc_provider import INSTALL, FakeIdentityProvider, Person, point_install_at  # noqa: E402

START = "/api/v1/auth/oidc/start"


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
    app.dependency_overrides[get_db] = lambda: db_session
    yield SimpleNamespace(idp=idp, client=TestClient(app, base_url=INSTALL), workspace_id=workspace_id,
                          db=db_session)
    idp.close()
    oidc_provider.forget_provider()


def _start(install, return_to="/projects?tab=1"):
    started = install.client.get(START, params={"return_to": return_to}, follow_redirects=False)
    assert started.status_code == 303, started.text
    return started


def _sign_in(install, return_to="/projects?tab=1"):
    """Start, approve at the provider, and send the browser back; the callback's answer."""
    callback = install.idp.approve(_start(install, return_to).headers["location"])
    return install.client.get(callback, follow_redirects=False)


def _refusal(answer) -> str:
    assert answer.status_code == 303
    location = urlsplit(answer.headers["location"])
    assert location.path == routes.SIGN_IN_PAGE and not location.netloc
    return parse_qs(location.query)["error"][0]


def _identity_user(db, subject):
    row = db.execute(text("SELECT user_id FROM user_identities WHERE subject = :s"), {"s": subject}).fetchone()
    return row[0] if row else None


def _sessions_of(db, user_id) -> int:
    return db.execute(text("SELECT COUNT(*) FROM auth_sessions WHERE user_id = :u"), {"u": user_id}).scalar_one()


def _all_sessions(db) -> int:
    return db.execute(text("SELECT COUNT(*) FROM auth_sessions")).scalar_one()


def _audits(db, workspace_id) -> list:
    return db.execute(text("SELECT action, user_id, details FROM audit_logs WHERE workspace_id = CAST(:ws AS uuid) "
                           "AND action LIKE 'auth:%' ORDER BY id"), {"ws": str(workspace_id)}).all()


def _membership(db, user_id, workspace_id):
    return db.execute(text("SELECT role FROM workspace_members WHERE user_id = :u AND workspace_id = CAST(:ws AS uuid)"),
                      {"u": user_id, "ws": str(workspace_id)}).fetchall()


# --------------------------------------------------------------------------- start


def test_start_redirects_with_pkce_state_nonce_and_the_exact_callback(install):
    started = _start(install)
    location = urlsplit(started.headers["location"])
    query = {name: values[0] for name, values in parse_qs(location.query).items()}

    assert f"{location.scheme}://{location.netloc}{location.path}" == install.idp.address("/authorize")
    assert query["response_type"] == "code" and query["client_id"] == install.idp.client_id
    assert query["redirect_uri"] == INSTALL + CALLBACK_PATH and query["scope"] == "openid profile email"
    assert query["code_challenge_method"] == "S256" and len(query["code_challenge"]) == 43
    assert len(query["state"]) >= 43 and len(query["nonce"]) >= 43 and query["state"] != query["nonce"]
    cookie = started.headers["set-cookie"]
    assert cookie.startswith(f"{routes.SIGN_IN_COOKIE}={query['state']};")
    assert all(part in cookie.lower() for part in ("secure", "httponly", "samesite=lax", "path=/", "max-age=600"))
    assert started.headers["cache-control"] == "no-store" and started.headers["referrer-policy"] == "no-referrer"
    row = install.db.execute(text("SELECT state_hash, nonce_hash, code_verifier, return_to, created_at, expires_at "
                                  "FROM oidc_pending_sign_ins WHERE state_hash = :h"),
                             {"h": hash_secret(query["state"])}).one()
    assert row.nonce_hash == hash_secret(query["nonce"]) and row.return_to == "/projects?tab=1"
    assert 43 <= len(row.code_verifier) <= 128
    assert row.expires_at - row.created_at == timedelta(seconds=600)


def test_start_where_the_provider_cannot_be_reached_says_so_and_keeps_nothing(install):
    install.idp.close()
    before = install.db.execute(text("SELECT COUNT(*) FROM oidc_pending_sign_ins")).scalar_one()

    answer = install.client.get(START, follow_redirects=False)

    assert _refusal(answer) == sign_in_flow.UNAVAILABLE
    assert install.db.execute(text("SELECT COUNT(*) FROM oidc_pending_sign_ins")).scalar_one() == before


# --------------------------------------------------------------------------- first sign-in (US-404)


def test_a_first_sign_in_creates_the_person_and_only_the_person_and_a_session(install):
    install.idp.sign_in_as(Person(subject="first-sign-in"))

    answer = _sign_in(install)

    assert answer.status_code == 303 and answer.headers["location"] == "/projects?tab=1"
    assert answer.cookies.get(config.SESSION_COOKIE_NAME, "").startswith("ams_")
    assert any(value.startswith(routes.SIGN_IN_COOKIE + "=") and "max-age=0" in value.lower()
               for value in answer.headers.get_list("set-cookie"))
    user_id = _identity_user(install.db, "first-sign-in")
    assert user_id is not None and _sessions_of(install.db, user_id) == 1
    owned = install.db.execute(text("SELECT COUNT(*) FROM workspaces WHERE owner_id = :u"), {"u": user_id})
    assert owned.scalar_one() == 0 and _membership(install.db, user_id, install.workspace_id) == []
    actions = [(action, uid, details) for action, uid, details in _audits(install.db, install.workspace_id)]
    assert actions == [("auth:identity_linked", user_id, {"default_role": None}), ("auth:sign_in", user_id, {})]


@pytest.mark.asyncio
async def test_with_no_default_role_the_person_is_signed_in_and_the_api_says_they_are_in_no_workspace(install):
    token = _sign_in(install).cookies[config.SESSION_COOKIE_NAME]
    request = SimpleNamespace(method="GET", headers={"authorization": "Bearer " + token}, query_params={},
                              state=SimpleNamespace())

    with pytest.raises(HTTPException) as refused:
        await hybrid.get_request_context_hybrid(request)

    assert refused.value.status_code == 403 and refused.value.detail["code"] == "workspace_required"


def test_a_default_role_is_given_at_first_sign_in_only_and_an_admins_change_survives(install, monkeypatch):
    monkeypatch.setattr(config, "ENTERPRISE_DEFAULT_ROLE", "editor")
    install.idp.sign_in_as(Person(subject="gets-a-role"))
    _sign_in(install)
    user_id = _identity_user(install.db, "gets-a-role")
    assert [row[0] for row in _membership(install.db, user_id, install.workspace_id)] == ["editor"]
    assert _audits(install.db, install.workspace_id)[0][2] == {"default_role": "editor"}

    install.db.execute(text("UPDATE workspace_members SET role = 'admin' WHERE user_id = :u"), {"u": user_id})
    install.db.commit()
    assert _sign_in(install).status_code == 303

    assert [row[0] for row in _membership(install.db, user_id, install.workspace_id)] == ["admin"]
    assert _sessions_of(install.db, user_id) == 2


def test_a_token_with_only_a_subject_and_an_email_signs_in(install):
    install.idp.sign_in_as(Person(subject="bare", name=None, email_verified=None))
    install.idp.misbehave(remove_claims=("auth_time", "sid", "email_verified", "name"))

    answer = _sign_in(install)

    assert answer.status_code == 303 and answer.headers["location"] == "/projects?tab=1"
    assert _sessions_of(install.db, _identity_user(install.db, "bare")) == 1


@pytest.mark.parametrize("verified", [None, False, "true", 1])
def test_an_email_the_token_does_not_mark_verified_with_true_is_no_email(install, verified):
    address = f"unverified-{uuid.uuid4().hex[:8]}@example.test"
    install.idp.sign_in_as(Person(subject="unverified", email=address, email_verified=verified))

    _sign_in(install)

    email = install.db.execute(text("SELECT email FROM users WHERE id = :u"),
                               {"u": _identity_user(install.db, "unverified")}).scalar_one()
    assert email != address and email.endswith("@pending")


def test_a_verified_email_is_the_new_persons_address(install):
    address = f"verified-{uuid.uuid4().hex[:8]}@example.test"
    install.idp.sign_in_as(Person(subject="verified", email=address, email_verified=True))
    _sign_in(install)
    email = install.db.execute(text("SELECT email FROM users WHERE id = :u"),
                               {"u": _identity_user(install.db, "verified")}).scalar_one()
    assert email == address


def test_an_address_another_person_holds_is_refused_exactly_like_a_bad_token(install):
    address = f"taken-{uuid.uuid4().hex[:8]}@example.test"
    install.db.execute(text("INSERT INTO users (username, email, is_active) VALUES (:e, :e, true)"), {"e": address})
    install.db.commit()
    install.idp.sign_in_as(Person(subject="newcomer", email=address, email_verified=True))
    taken = _sign_in(install)
    install.idp.misbehave(signature="wrong_key")
    bad_token = _sign_in(install)

    assert _refusal(taken) == _refusal(bad_token) == sign_in_flow.REFUSED
    assert taken.headers["location"] == bad_token.headers["location"] and taken.content == bad_token.content
    assert _identity_user(install.db, "newcomer") is None and _all_sessions(install.db) == 0
    reasons = [details["reason"] for action, _uid, details in _audits(install.db, install.workspace_id)
               if action == "auth:sign_in_refused"]
    assert reasons == ["email_conflict", "invalid_token"]


def test_a_deactivated_person_is_refused_and_audited(install):
    install.idp.sign_in_as(Person(subject="leaver"))
    _sign_in(install)
    user_id = _identity_user(install.db, "leaver")
    install.db.execute(text("UPDATE users SET is_active = false WHERE id = :u"), {"u": user_id})
    install.db.commit()

    assert _refusal(_sign_in(install)) == sign_in_flow.REFUSED
    assert _sessions_of(install.db, user_id) == 1
    refused = [(uid, details) for action, uid, details in _audits(install.db, install.workspace_id)
               if action == "auth:sign_in_refused"]
    assert refused == [(user_id, {"reason": "deactivated"})]


# --------------------------------------------------------------------------- refusals (US-402)


@pytest.mark.parametrize("behaviour", [
    {"signature": "wrong_key"}, {"signature": "none"}, {"signature": "hs256_public"},
    {"claims": {"iss": "https://elsewhere.example.test/realms/prd277"}}, {"claims": {"aud": "another-client"}},
    {"claims": {"exp": 1_000_000_000, "iat": 999_999_000}}, {"nonce": "missing"}, {"nonce": "wrong"},
], ids=["wrong-signature", "alg-none", "hs256-public-key", "wrong-issuer", "wrong-audience", "expired",
        "missing-nonce", "wrong-nonce"])
def test_a_token_that_breaks_a_rule_ends_without_a_session_and_is_audited(install, behaviour):
    install.idp.misbehave(**behaviour)

    assert _refusal(_sign_in(install)) == sign_in_flow.REFUSED

    assert _all_sessions(install.db) == 0 and _identity_user(install.db, install.idp.person.subject) is None
    assert [(action, details) for action, _uid, details in _audits(install.db, install.workspace_id)] == [
        ("auth:sign_in_refused", {"reason": "invalid_token"})]


def test_a_reused_state_is_expired_and_the_code_is_exchanged_once(install):
    callback = install.idp.approve(_start(install).headers["location"])
    first = install.client.get(callback, follow_redirects=False)
    install.client.cookies.set(routes.SIGN_IN_COOKIE, parse_qs(urlsplit(callback).query)["state"][0])

    second = install.client.get(callback, follow_redirects=False)

    assert first.status_code == 303 and first.headers["location"] == "/projects?tab=1"
    assert _refusal(second) == sign_in_flow.EXPIRED
    assert len(install.idp.token_requests) == 1


def test_a_replayed_code_with_a_new_state_is_refused_by_the_provider(install):
    used = parse_qs(urlsplit(install.idp.approve(_start(install).headers["location"])).query)
    install.client.get(f"{CALLBACK_PATH}?code={used['code'][0]}&state={used['state'][0]}", follow_redirects=False)
    fresh = parse_qs(urlsplit(install.idp.approve(_start(install).headers["location"])).query)

    replayed = install.client.get(CALLBACK_PATH, params={"code": used["code"][0], "state": fresh["state"][0]},
                                  follow_redirects=False)

    assert _refusal(replayed) == sign_in_flow.REFUSED
    assert len(install.idp.token_requests) == 2 and _all_sessions(install.db) == 1


def test_a_callback_carried_to_another_browser_is_not_a_sign_in_there(install):
    callback = install.idp.approve(_start(install).headers["location"])
    install.client.cookies.clear()

    assert _refusal(install.client.get(callback, follow_redirects=False)) == sign_in_flow.EXPIRED
    assert install.idp.token_requests == [] and _audits(install.db, install.workspace_id) == []


def test_only_the_latest_sign_in_of_a_browser_can_finish(install):
    earlier = install.idp.approve(_start(install).headers["location"])
    _start(install)

    assert _refusal(install.client.get(earlier, follow_redirects=False)) == sign_in_flow.EXPIRED


def test_an_expired_sign_in_is_refused(install):
    callback = install.idp.approve(_start(install).headers["location"])
    state = parse_qs(urlsplit(callback).query)["state"][0]
    install.db.execute(text("UPDATE oidc_pending_sign_ins SET expires_at = :t WHERE state_hash = :h"),
                       {"t": datetime.now(timezone.utc) - timedelta(seconds=1), "h": hash_secret(state)})
    install.db.commit()

    assert _refusal(install.client.get(callback, follow_redirects=False)) == sign_in_flow.EXPIRED
    assert install.idp.token_requests == []


@pytest.mark.parametrize("return_to", ["//evil.example.test/x", "https://evil.example.test/", "/\\evil.example.test",
                                       "javascript:alert(1)", "evil.example.test", "/ok\r\nLocation: x"])
def test_the_person_is_only_ever_sent_to_a_path_on_this_install(install, return_to):
    answer = _sign_in(install, return_to=return_to)

    assert answer.status_code == 303 and answer.headers["location"] == "/"


@pytest.mark.parametrize("behaviour", [{"callback_error": "access_denied"},
                                       {"callback_iss": "https://elsewhere.example.test"}])
def test_a_provider_error_or_another_issuer_on_the_callback_is_refused_unexchanged(install, behaviour):
    install.idp.misbehave(**behaviour)

    assert _refusal(_sign_in(install)) == sign_in_flow.REFUSED
    assert install.idp.token_requests == [] and _all_sessions(install.db) == 0


@pytest.mark.parametrize("behaviour", [{"token_status": 500}, {"token_body": {"access_token": "x"}}])
def test_a_token_endpoint_that_gives_no_id_token_is_refused(install, behaviour):
    install.idp.misbehave(**behaviour)
    assert _refusal(_sign_in(install)) == sign_in_flow.REFUSED and _all_sessions(install.db) == 0


# --------------------------------------------------------------------------- US-408


class _FakeMethod:
    """A second way of proving identity: whatever its callback names, verified by fiat."""

    def start(self, db, return_to):
        return Started(redirect_to="/fake-sign-in", browser_binding="fake-binding")

    def finish(self, db, callback):
        subject = callback.params["who"][0]
        identity = VerifiedIdentity(issuer="https://fake.example.test", subject=subject, email=None,
                                    email_verified=False, name=None, auth_time=datetime.now(timezone.utc),
                                    idp_session_id=None)
        return Finished(identity=identity, return_to="/after")


def _fake_registry(monkeypatch, hook):
    registry = SignInRegistry({"oidc": OidcSignIn()})
    registry.register_method("fake", _FakeMethod())
    registry.register_hook("test-hook", hook)
    monkeypatch.setattr(sign_in_flow, "REGISTRY", registry)
    return registry


def test_a_second_method_and_a_hook_sign_a_person_in_and_set_a_membership(install, monkeypatch):
    seen = []

    def hook(db, signed_in):
        seen.append((signed_in.created, signed_in.identity.subject))
        people.set_membership(db, signed_in.user_id, install.workspace_id, "viewer")

    _fake_registry(monkeypatch, hook)
    done = sign_in_flow.finish_sign_in(install.db, Callback(params={"who": ("hooked",)}),
                                       sign_in_flow.Browser(), method="fake")

    user_id = _identity_user(install.db, "hooked")
    assert done.return_to == "/after" and done.session.user_id == user_id and seen == [(True, "hooked")]
    assert [row[0] for row in _membership(install.db, user_id, install.workspace_id)] == ["viewer"]
    assert _sessions_of(install.db, user_id) == 1


def test_a_hook_that_refuses_ends_the_sign_in_and_keeps_no_person(install, monkeypatch):
    def hook(db, signed_in):
        raise SignInRefusedByHook("not in the pilot group")

    _fake_registry(monkeypatch, hook)
    with pytest.raises(sign_in_flow.SignInRefusal) as refused:
        sign_in_flow.finish_sign_in(install.db, Callback(params={"who": ("kept-out",)}),
                                    sign_in_flow.Browser(), method="fake")

    assert refused.value.code == sign_in_flow.REFUSED
    assert _identity_user(install.db, "kept-out") is None and _all_sessions(install.db) == 0
    assert [(action, details) for action, _uid, details in _audits(install.db, install.workspace_id)] == [
        ("auth:sign_in_refused", {"reason": "access_rule"})]


# --------------------------------------------------------------------------- nothing secret logged


def _browser_line(record) -> bool:
    """The test's own browser (httpx in TestClient and in ``approve``) logging the address it opened."""
    message = record.getMessage()
    return record.name == "httpx" and (INSTALL in message or "/authorize?" in message)


def _one_round(install, behaviour, answers) -> list:
    """One sign-in under ``behaviour``, its callback sent twice; the secret values it involved."""
    install.idp.misbehave(**behaviour)
    started = _start(install)
    query = parse_qs(urlsplit(started.headers["location"]).query)
    state, nonce = query["state"][0], query["nonce"][0]
    verifier = install.db.execute(text("SELECT code_verifier FROM oidc_pending_sign_ins WHERE state_hash = :h"),
                                  {"h": hash_secret(state)}).scalar_one()
    callback = install.idp.approve(started.headers["location"])
    answers.extend(install.client.get(callback, follow_redirects=False) for _ in range(2))
    return [state, nonce, verifier]


def test_no_secret_reaches_a_log_line_or_an_answer_on_any_path(install, caplog):
    answers, values = [], [install.idp.client_secret]
    pair = f"{install.idp.client_id}:{install.idp.client_secret}".encode("utf-8")
    values.append(base64.b64encode(pair).decode("ascii"))
    with caplog.at_level(logging.DEBUG):
        for behaviour in ({}, {"signature": "wrong_key"}, {"nonce": "wrong"}, {"token_status": 500},
                          {"callback_error": "access_denied"}):
            values += _one_round(install, behaviour, answers)
        install.idp.close()
        answers.append(install.client.get(START, follow_redirects=False))
    session_tokens = [answer.cookies[config.SESSION_COOKIE_NAME] for answer in answers
                      if config.SESSION_COOKIE_NAME in answer.cookies]
    assert len(session_tokens) == 1 and install.idp.issued

    formatter = logging.Formatter("%(name)s %(message)s")
    logged = "\n".join(formatter.format(record) for record in caplog.records if not _browser_line(record))
    for value in (*values, *install.idp.issued, *session_tokens):
        assert value not in logged
    for answer in answers:
        shown = answer.headers.get("location", "") + answer.text
        assert not any(value in shown for value in (*values, *install.idp.issued, *session_tokens))
