"""PRD-277 US-407 (line 3): the sign-in flow against a real Keycloak, not the fake provider.

Runs only in the ``oidc-keycloak`` CI job (``.github/workflows/oidc-keycloak.yml``), which starts a
Keycloak container over TLS with a certificate made at run time, imports a realm made at run time
(one confidential client, one person, a hard-coded ``automatos:team`` claim) and hands this file
its address and the generated credentials in ``PRD277_KEYCLOAK_*``. Anywhere else every test here
is skipped. Against Postgres, each test in a transaction rolled back at the end:

* a person signs in through Keycloak's own login form and gets a session; their identity is
  linked to Keycloak's issuer and subject;
* the access rule reads Keycloak's claim with a colon in its name: it lets the person in when it
  holds and refuses them, keeping no person, when it does not;
* a step-up sends ``max_age=0``, ``prompt=login`` and the ``claims`` request for ``auth_time``,
  Keycloak asks for the password again, and the session's ``auth_time`` is refreshed;
* sign-out answers Keycloak's end-session address with ``client_id`` and the install's
  ``post_logout_redirect_uri``, which Keycloak accepts.
"""
from __future__ import annotations

import html
import os
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
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
from core.auth import hybrid, oidc_provider, session_lane, sign_in_flow  # noqa: E402
from core.auth.oidc_sign_in import AUTH_TIME_CLAIMS_REQUEST, OidcSignIn, SIGNED_OUT_PATH  # noqa: E402
from core.auth.pairing import hash_secret  # noqa: E402
from core.auth.sign_in_methods import SignInRegistry  # noqa: E402
from core.database.database import get_db  # noqa: E402

KEYCLOAK = {name: os.environ.get(f"PRD277_KEYCLOAK_{name}", "") for name in
            ("ISSUER", "CLIENT_ID", "CLIENT_SECRET", "CA_FILE", "USERNAME", "PASSWORD")}
pytestmark = pytest.mark.skipif(not all(KEYCLOAK.values()),
                                reason="only in the oidc-keycloak CI job, which starts Keycloak")

INSTALL = "https://automatos.example.test"
START = "/api/v1/auth/oidc/start"
TEAM_CLAIM = "automatos:team"  # the realm's hard-coded claim, a colon in its name
_LOGIN_FORM = re.compile(r"<form\b[^>]*\bid=\"kc-form-login\"[^>]*>", re.S)
_ACTION = re.compile(r"\baction=\"([^\"]+)\"")


class _BorrowedSession:
    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


@pytest.fixture
def db_session(test_engine):
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
    """An enterprise install whose people sign in at the Keycloak the job started."""
    workspace_id = seed_workspace()
    db_session.commit()
    settings = {
        "AUTH_EDITION": "enterprise", "REQUIRE_AUTH": True, "FRONTEND_URL": INSTALL,
        "OIDC_ISSUER": KEYCLOAK["ISSUER"], "OIDC_CLIENT_ID": KEYCLOAK["CLIENT_ID"],
        "OIDC_CLIENT_SECRET": KEYCLOAK["CLIENT_SECRET"], "OIDC_CA_BUNDLE_FILE": KEYCLOAK["CA_FILE"],
        "OIDC_SCOPES": "openid profile email", "OIDC_CLAIM_EMAIL": "email", "OIDC_CLAIM_NAME": "name",
        "ENTERPRISE_DEFAULT_ROLE": "none", "OIDC_ENDPOINT_HOSTS": "", "OIDC_PENDING_SIGN_INS_MAX": 10_000,
        "OIDC_PENDING_SIGN_INS_PER_ADDRESS": 100, "OIDC_REQUIRED_CLAIMS": "", "ENTERPRISE_BOOTSTRAP_ADMINS": "",
        "DEFAULT_WORKSPACE_ID": workspace_id,
    }
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)
    oidc_provider.forget_provider()
    monkeypatch.setattr(routes, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 300, 299, 600)))
    monkeypatch.setattr(sign_in_flow, "REGISTRY", SignInRegistry({"oidc": OidcSignIn()}))
    monkeypatch.setattr(session_lane, "SessionLocal", lambda: _BorrowedSession(db_session))
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(auth_sessions.router)
    app.dependency_overrides[get_db] = lambda: db_session
    with httpx.Client(verify=KEYCLOAK["CA_FILE"], follow_redirects=False, trust_env=False) as browser:
        yield SimpleNamespace(client=TestClient(app, base_url=INSTALL), browser=browser, db=db_session,
                              workspace_id=workspace_id)
    oidc_provider.forget_provider()


def _password_at_keycloak(browser: httpx.Client, address: str) -> SimpleNamespace:
    """Keycloak's login form, filled in as a person would: the callback address it sends back to, and
    whether it asked for the password (a browser already signed in there is sent straight back)."""
    page = browser.get(address)
    if page.status_code == 302:
        return SimpleNamespace(callback=page.headers["location"], asked=False)
    assert page.status_code == 200, page.status_code
    form = _LOGIN_FORM.search(page.text)
    assert form is not None, "Keycloak showed no login form"
    action = html.unescape(_ACTION.search(form.group(0)).group(1))
    answer = browser.post(action, data={"username": KEYCLOAK["USERNAME"], "password": KEYCLOAK["PASSWORD"],
                                        "credentialId": ""})
    assert answer.status_code == 302, answer.status_code
    return SimpleNamespace(callback=answer.headers["location"], asked=True)


def _sign_in(install, step_up_too=False):
    """Start, sign in at Keycloak, and send the browser back: (the start, Keycloak's step, the callback)."""
    params = {"return_to": "/settings", **({"step_up": "1"} if step_up_too else {})}
    started = install.client.get(START, params=params, follow_redirects=False)
    assert started.status_code == 303, started.text
    at_keycloak = _password_at_keycloak(install.browser, started.headers["location"])
    assert at_keycloak.callback.startswith(INSTALL + "/api/v1/auth/oidc/callback?")
    return started, at_keycloak, install.client.get(at_keycloak.callback, follow_redirects=False)


def _linked_person(db):
    return db.execute(text("SELECT user_id, subject FROM user_identities WHERE issuer = :i"),
                      {"i": KEYCLOAK["ISSUER"]}).fetchone()


def test_a_person_signs_in_through_keycloak_and_gets_a_session(install):
    _, at_keycloak, answer = _sign_in(install)

    assert at_keycloak.asked
    assert answer.status_code == 303 and answer.headers["location"] == "/settings"
    assert answer.cookies.get(config.SESSION_COOKIE_NAME, "").startswith("ams_")
    assert _linked_person(install.db) is not None


def test_the_access_rule_reads_keycloaks_claim_with_a_colon_in_its_name(install, monkeypatch):
    monkeypatch.setattr(config, "OIDC_REQUIRED_CLAIMS", f"{TEAM_CLAIM}=finance")
    _, _, refused = _sign_in(install)
    assert refused.status_code == 303
    assert parse_qs(urlsplit(refused.headers["location"]).query)["error"] == [sign_in_flow.NOT_ALLOWED]
    assert _linked_person(install.db) is None

    monkeypatch.setattr(config, "OIDC_REQUIRED_CLAIMS", f"{TEAM_CLAIM}=ops")
    _, _, allowed = _sign_in(install)  # Keycloak remembers the person: no form this time
    assert allowed.headers["location"] == "/settings" and _linked_person(install.db) is not None


def test_a_step_up_makes_keycloak_ask_again_and_refreshes_the_session(install):
    _, _, answer = _sign_in(install)
    token = answer.cookies[config.SESSION_COOKIE_NAME]
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    install.db.execute(text("UPDATE auth_sessions SET auth_time = :t WHERE token_hash = :h"),
                       {"t": old, "h": hash_secret(token)})
    install.db.commit()

    started, at_keycloak, stepped_up = _sign_in(install, step_up_too=True)

    sent = parse_qs(urlsplit(started.headers["location"]).query)
    # Keycloak's metadata says it takes the claims parameter and lists login among its prompt values.
    assert (sent["max_age"], sent["prompt"], sent["claims"]) == (["0"], ["login"], [AUTH_TIME_CLAIMS_REQUEST])
    assert at_keycloak.asked  # signed in at Keycloak already, and asked for the password all the same
    assert stepped_up.status_code == 303 and stepped_up.headers["location"] == "/settings"
    install.db.expire_all()
    refreshed = install.db.execute(text("SELECT auth_time FROM auth_sessions WHERE token_hash = :h"),
                                   {"h": hash_secret(token)}).scalar_one()
    assert refreshed > datetime.now(timezone.utc) - timedelta(minutes=5)


def test_sign_out_sends_the_browser_to_keycloaks_end_session_address(install):
    _, _, answer = _sign_in(install)
    token = answer.cookies[config.SESSION_COOKIE_NAME]

    signed_out = install.client.post("/api/v1/auth/sign-out", headers={"authorization": "Bearer " + token})

    assert signed_out.status_code == 200
    redirect_to = signed_out.json()["redirect_to"]
    assert redirect_to.startswith(KEYCLOAK["ISSUER"] + "/protocol/openid-connect/logout?")
    query = parse_qs(urlsplit(redirect_to).query)
    assert query == {"client_id": [KEYCLOAK["CLIENT_ID"]], "post_logout_redirect_uri": [INSTALL + SIGNED_OUT_PATH]}
    at_keycloak = install.browser.get(redirect_to)
    assert at_keycloak.status_code in (200, 302), at_keycloak.status_code  # its confirm page, or straight back
