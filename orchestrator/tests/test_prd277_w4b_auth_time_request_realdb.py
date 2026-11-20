"""PRD-277 US-406 (§5.5, §5.9): what the start asks the provider for, so ``auth_time`` can arrive.

* **Every sign-in** sends the ``claims`` parameter asking for ``auth_time`` in the ID token as
  essential, unless the metadata's ``claims_parameter_supported`` is anything but absent or
  ``true``.
* **A step-up** also sends ``max_age=0`` and ``prompt=login``. ``prompt=login`` is left out where
  the metadata lists prompt values without ``login`` (or lists them as something not a list), and
  for an issuer in ``oidc_provider.PROMPT_LOGIN_UNLISTED_HOSTS`` whose metadata lists none.
* **Nothing checked changes.** A provider that ignores every one of them still signs people in,
  with no ``auth_time`` (the epoch), and a step-up there is refused (``step_up_no_auth_time``). A
  provider that sends ``auth_time`` only when asked gives an ordinary session a real one.

The pure part reads ``oidc_provider.parse_metadata`` and ``oidc_sign_in.auth_time_values``; the
rest runs the routes against the fake provider and Postgres, as
``test_prd277_w4b_sign_in_rules_realdb.py`` does, each test in a transaction rolled back after.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.auth import sign_in_flow  # noqa: E402
from core.auth.oidc_provider import parse_metadata  # noqa: E402
from core.auth.oidc_sign_in import AUTH_TIME_CLAIMS_REQUEST, NO_AUTH_TIME, auth_time_values  # noqa: E402
from core.auth.pairing import hash_secret  # noqa: E402
from tests.helpers_fake_oidc_provider import AUTH_TIME_ASKED, Person  # noqa: E402
from tests.test_prd277_w4b_sign_in_rules_realdb import (  # noqa: E402
    _auth_time,
    _refusal,
    _sets_session_cookie,
    _sign_in,
    _signed_in,
    _start,
    db_session,
    install,
)

# The install and its session are that file's fixtures, used here by name.
FIXTURES = (db_session, install)

CODE_FLOW = {"response_type", "client_id", "redirect_uri", "scope", "state", "nonce", "code_challenge",
             "code_challenge_method"}
GOOGLE = "https://accounts.google.com"
OTHER = "https://idp.example.test"


# --------------------------------------------------------------------------- pure: what the metadata allows


def _metadata(issuer=OTHER, **fields):
    doc = {"issuer": issuer, "authorization_endpoint": issuer + "/a", "token_endpoint": issuer + "/t",
           "jwks_uri": issuer + "/k", **fields}
    return parse_metadata(doc, issuer)


@pytest.fixture
def no_listed_hosts(monkeypatch):
    monkeypatch.setattr(config, "OIDC_ENDPOINT_HOSTS", "")


def test_the_claims_request_is_auth_time_in_the_id_token_as_essential():
    assert json.loads(AUTH_TIME_CLAIMS_REQUEST) == {"id_token": {"auth_time": {"essential": True}}} == AUTH_TIME_ASKED


@pytest.mark.parametrize(("fields", "taken"), [
    ({}, True), ({"claims_parameter_supported": True}, True), ({"claims_parameter_supported": False}, False),
    ({"claims_parameter_supported": "true"}, False), ({"claims_parameter_supported": None}, False),
    ({"claims_parameter_supported": 1}, False),
])
def test_claims_is_sent_unless_the_metadata_says_otherwise(no_listed_hosts, fields, taken):
    metadata = _metadata(**fields)
    assert metadata.takes_claims_request is taken
    assert ("claims" in auth_time_values(metadata, step_up=False)) is taken


@pytest.mark.parametrize(("issuer", "fields", "taken"), [
    (OTHER, {}, True), (OTHER, {"prompt_values_supported": ["none", "login", "consent"]}, True),
    (OTHER, {"prompt_values_supported": ["none", "consent"]}, False),
    (OTHER, {"prompt_values_supported": "login"}, False), (OTHER, {"prompt_values_supported": None}, False),
    (GOOGLE, {}, False), (GOOGLE, {"prompt_values_supported": ["login"]}, True),
])
def test_prompt_login_is_sent_where_the_metadata_or_the_provider_documents_it(no_listed_hosts, issuer, fields, taken):
    metadata = _metadata(issuer, **fields)
    assert metadata.takes_prompt_login is taken
    assert (auth_time_values(metadata, step_up=True).get("prompt") == "login") is taken
    assert "prompt" not in auth_time_values(metadata, step_up=False)


def test_a_step_up_always_sends_max_age_0_and_a_sign_in_never_does(no_listed_hosts):
    for metadata in (_metadata(), _metadata(GOOGLE), _metadata(claims_parameter_supported=False,
                                                                prompt_values_supported=[])):
        assert auth_time_values(metadata, step_up=True)["max_age"] == "0"
        assert "max_age" not in auth_time_values(metadata, step_up=False)


def test_what_the_metadata_says_about_them_never_refuses_the_metadata(no_listed_hosts):
    metadata = _metadata(claims_parameter_supported={"odd": 1}, prompt_values_supported=42)
    assert (metadata.takes_claims_request, metadata.takes_prompt_login) == (False, False)


# --------------------------------------------------------------------------- the routes: the exact query


def test_a_sign_in_sends_the_code_flow_and_the_claims_request_and_nothing_else(install):
    _sign_in(install)

    sent = install.idp.authorize_requests[-1]
    assert set(sent) == CODE_FLOW | {"claims"}
    assert json.loads(sent["claims"]) == AUTH_TIME_ASKED


def test_a_step_up_sends_the_claims_request_max_age_0_and_prompt_login(install):
    _signed_in(install, "asks-exactly")

    started = _start(install, step_up_too=True)

    sent = install.idp.authorize_requests[-1]
    assert set(sent) == CODE_FLOW | {"claims", "max_age", "prompt"}
    assert (json.loads(sent["claims"]), sent["max_age"], sent["prompt"]) == (AUTH_TIME_ASKED, "0", "login")
    answer = install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False)
    assert answer.status_code == 303 and answer.headers["location"] == "/settings"


def test_a_provider_that_refuses_the_claims_parameter_is_not_sent_it_and_still_signs_in(install):
    install.idp.misbehave(metadata={"claims_parameter_supported": False})

    answer = _sign_in(install)

    assert set(install.idp.authorize_requests[-1]) == CODE_FLOW
    assert answer.status_code == 303 and _sets_session_cookie(answer)


def test_a_provider_that_lists_prompt_values_without_login_is_never_sent_it(install):
    install.idp.misbehave(metadata={"prompt_values_supported": ["none", "consent"]}, refused_prompts=("login",))
    me = _signed_in(install, "no-prompt-login")

    started = _start(install, step_up_too=True)
    answer = install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False)

    sent = install.idp.authorize_requests[-1]
    assert "prompt" not in sent and sent["max_age"] == "0"
    assert answer.status_code == 303 and answer.headers["location"] == "/settings"
    assert _auth_time(install.db, me.session_id) > datetime.now(timezone.utc) - timedelta(minutes=2)


def test_a_provider_that_answers_prompt_login_with_an_error_refuses_the_step_up(install):
    # The case the metadata rule avoids: the provider lists no prompt values and refuses login.
    install.idp.misbehave(refused_prompts=("login",))
    me = _signed_in(install, "prompt-refused")

    started = _start(install, step_up_too=True)
    answer = install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False)

    assert install.idp.authorize_requests[-1]["prompt"] == "login"
    assert _refusal(answer) == sign_in_flow.REFUSED
    assert _auth_time(install.db, me.session_id) == me.old


# --------------------------------------------------------------------------- nothing checked changes


def _session_auth_time(install, answer):
    """The ``auth_time`` of the session ``answer`` set the cookie for."""
    token = answer.cookies[config.SESSION_COOKIE_NAME]
    install.db.expire_all()
    return install.db.execute(text("SELECT auth_time FROM auth_sessions WHERE token_hash = :h"),
                              {"h": hash_secret(token)}).scalar_one()


def test_a_provider_that_ignores_every_parameter_still_signs_people_in_without_auth_time(install):
    install.idp.misbehave(auth_time="never")
    install.idp.sign_in_as(Person(subject="ignores-it-all"))

    answer = _sign_in(install)

    assert answer.status_code == 303 and answer.headers["location"] == "/settings" and _sets_session_cookie(answer)
    assert _session_auth_time(install, answer) == NO_AUTH_TIME
    me = _signed_in(install, "ignores-it-all")
    started = _start(install, step_up_too=True)
    stepped = install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False)
    assert _refusal(stepped) == sign_in_flow.STEP_UP_NO_AUTH_TIME
    assert _auth_time(install.db, me.session_id) == me.old


def test_a_provider_that_sends_auth_time_only_when_asked_gives_an_ordinary_session_a_real_one(install):
    install.idp.misbehave(auth_time="when_asked")

    answer = _sign_in(install)

    assert "max_age" not in install.idp.authorize_requests[-1]  # asked through claims alone
    assert _session_auth_time(install, answer) > datetime.now(timezone.utc) - timedelta(minutes=2)


def test_not_asked_through_claims_that_provider_sends_none_and_the_session_starts_at_the_epoch(install):
    install.idp.misbehave(auth_time="when_asked", metadata={"claims_parameter_supported": False})

    answer = _sign_in(install)

    assert "claims" not in install.idp.authorize_requests[-1]
    assert answer.status_code == 303 and _session_auth_time(install, answer) == NO_AUTH_TIME
