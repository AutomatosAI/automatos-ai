"""PRD-277 US-406 (§5.5) and §5.6 against Postgres: step-up's edges, and where sign-out may send a browser.

The routes and the fake provider are those of ``test_prd277_w4b_sign_in_rules_realdb.py``; the
clock edges call ``step_up_sign_in.confirm_step_up`` itself on a fixed clock. Each test runs in a
transaction rolled back at the end.

* **Fresh, on a fixed clock**: ``auth_time`` may be earlier than the step-up's start, or later than
  now, by ``CLOCK_SKEW`` and no more (``stale_auth_time``, ``future_auth_time``).
* **The new session**: a confirmed step-up ends the session that asked (``step_up``; its token is
  refused) and issues one that keeps its person, its absolute expiry and the provider's session id
  (the token's ``sid``, else the old one); when the audit row cannot be written, no session changes.
* **Refusals**: a person deactivated while signing in again (``deactivated``); a callback carried to
  another browser with a session of its own, or used twice (``sign_in_expired``, nothing changed);
  an ``auth_time`` in the future or not a number (``sign_in_refused``, the session untouched).
* **Fetch metadata**: ``step_up=1`` sent from another site (``Sec-Fetch-Site`` ``cross-site`` or
  ``same-site``) is refused before anything is stored; this site, the address bar, or a browser that
  sends no such header may start one.
* **Sign-out**: an end-session address on a host that is neither the issuer's nor in
  ``OIDC_ENDPOINT_HOSTS`` is treated as none (204); listed there, it is followed.
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.auth import oidc_provider, people, session_audit, sessions, sign_in_flow, step_up_sign_in  # noqa: E402
from core.auth.sign_in_methods import Finished, VerifiedIdentity  # noqa: E402
from core.auth.step_up_sign_in import CLOCK_SKEW, StepUpRefused  # noqa: E402
from core.models.auth_sessions import AuthSession  # noqa: E402
from tests.helpers_fake_oidc_provider import INSTALL, Person  # noqa: E402
from tests.test_prd277_w4b_sign_in_rules_realdb import (  # noqa: E402
    SIGN_OUT,
    START,
    _auth_time,
    _confirmed,
    _count,
    _new_session,
    _refusal,
    _signed_in,
    _start,
    _step_ups,
    db_session,
    install,
)

# The install and its session are that file's fixtures, used here by name.
FIXTURES = (db_session, install)

ISSUER = "https://idp.example.test/realms/prd277-w4b-edges"
STARTED = datetime(2026, 11, 20, 9, 0, 0, tzinfo=timezone.utc)
NOW = STARTED + timedelta(seconds=30)


# --------------------------------------------------------------------------- fresh, on a fixed clock


@pytest.fixture
def asking(db_session, seed_workspace, monkeypatch):
    """A live session, made at ``STARTED`` less a minute, of a person linked to ``ISSUER``."""
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", seed_workspace())
    subject = "edges-" + uuid.uuid4().hex[:8]
    linked = people.link_person(db_session, issuer=ISSUER, subject=subject, email_verified=False)
    db_session.commit()
    issued = sessions.issue_session(db_session, user_id=linked.user_id, auth_time=STARTED - timedelta(hours=2),
                                    now=STARTED - timedelta(minutes=1))
    return issued, subject


def _confirm(db, issued, subject, auth_time, sid=None) -> sessions.IssuedSession:
    identity = VerifiedIdentity(issuer=ISSUER, subject=subject, email=None, email_verified=False, name=None,
                                auth_time=auth_time, idp_session_id=sid)
    finished = Finished(identity=identity, return_to="/", step_up_session_id=issued.session_id, started_at=STARTED)
    return step_up_sign_in.confirm_step_up(db, finished, ip_address=None, user_agent=None, now=NOW)


@pytest.mark.parametrize("auth_time", [STARTED, STARTED - CLOCK_SKEW, NOW, NOW + CLOCK_SKEW],
                         ids=["at-the-start", "skew-before-the-start", "now", "skew-after-now"])
def test_an_auth_time_within_the_skew_of_start_and_now_confirms(db_session, asking, auth_time):
    issued, subject = asking
    renewed = _confirm(db_session, issued, subject, auth_time)
    db_session.expire_all()
    assert db_session.get(AuthSession, renewed.session_id).auth_time == auth_time
    assert db_session.get(AuthSession, issued.session_id).auth_time == STARTED - timedelta(hours=2)


# --------------------------------------------------------------------------- the new session


@pytest.fixture
def asking_with_sid(db_session, asking):
    """``asking``'s person, with a session of their own that recorded the provider's ``sid``."""
    old, subject = asking
    person = db_session.get(AuthSession, old.session_id).user_id
    issued = sessions.issue_session(db_session, user_id=person, auth_time=STARTED - timedelta(hours=2),
                                    sid="provider-session-1", now=STARTED - timedelta(minutes=1))
    return issued, subject


@pytest.mark.parametrize(("sent", "kept"), [(None, "provider-session-1"), ("provider-session-2", "provider-session-2")],
                         ids=["no-sid-keeps-the-old-one", "a-new-sid-is-taken"])
def test_a_confirmed_step_up_issues_a_new_session_and_ends_the_one_that_asked(db_session, asking_with_sid,
                                                                              sent, kept):
    issued, subject = asking_with_sid

    renewed = _confirm(db_session, issued, subject, NOW, sid=sent)

    db_session.expire_all()
    old, new = db_session.get(AuthSession, issued.session_id), db_session.get(AuthSession, renewed.session_id)
    assert (old.revoked_at, old.revoked_reason) == (NOW, sessions.REVOKED_STEP_UP)
    assert sessions.session_state(sessions.find_session(db_session, issued.token), NOW) == sessions.REVOKED
    assert sessions.find_session(db_session, renewed.token).id == new.id
    assert sessions.session_state(new, NOW) == sessions.LIVE
    assert (new.user_id, new.auth_time, new.idp_session_id) == (old.user_id, NOW, kept)
    assert new.absolute_expires_at == old.absolute_expires_at == renewed.absolute_expires_at
    assert new.idle_expires_at == min(NOW + sessions.idle_lifetime(), old.absolute_expires_at)
    assert (new.created_at, new.last_seen_at) == (NOW, NOW)


def test_a_step_up_whose_audit_row_cannot_be_written_changes_no_session(db_session, asking, monkeypatch):
    issued, subject = asking
    person = db_session.get(AuthSession, issued.session_id).user_id
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", None)  # no workspace to record the row in

    with pytest.raises(session_audit.SessionAuditFailed):
        _confirm(db_session, issued, subject, NOW)

    db_session.expire_all()
    rows = db_session.query(AuthSession).filter(AuthSession.user_id == person).all()
    assert [(row.id, row.revoked_at) for row in rows] == [(issued.session_id, None)]
    assert rows[0].auth_time == STARTED - timedelta(hours=2)


@pytest.mark.parametrize(("auth_time", "reason"), [
    (STARTED - CLOCK_SKEW - timedelta(seconds=1), session_audit.STEP_UP_STALE),
    (NOW + CLOCK_SKEW + timedelta(seconds=1), session_audit.STEP_UP_FUTURE),
], ids=["a-second-too-early", "a-second-too-late"])
def test_an_auth_time_beyond_the_skew_is_refused(db_session, asking, auth_time, reason):
    issued, subject = asking
    with pytest.raises(StepUpRefused) as refused:
        _confirm(db_session, issued, subject, auth_time)
    assert str(refused.value) == reason
    db_session.expire_all()
    assert db_session.get(AuthSession, issued.session_id).auth_time == STARTED - timedelta(hours=2)


# --------------------------------------------------------------------------- refusals through the routes


def test_a_person_deactivated_while_signing_in_again_is_refused(install):
    me = _signed_in(install, "deactivated-meanwhile")
    started = _start(install, step_up_too=True)
    install.db.execute(text("UPDATE users SET is_active = false WHERE id = :u"), {"u": me.user_id})
    install.db.commit()

    answer = install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False)

    assert _refusal(answer) == sign_in_flow.STEP_UP_REFUSED
    assert _auth_time(install.db, me.session_id) == me.old
    assert _step_ups(install) == [(me.user_id, {"confirmed": False, "reason": "deactivated"})]


def test_a_step_up_callback_carried_to_another_session_or_used_twice_changes_nothing(install):
    me = _signed_in(install, "asks-for-step-up")
    started = _start(install, step_up_too=True)
    callback = install.idp.approve(started.headers["location"])
    other = TestClient(install.client.app, base_url=INSTALL)
    install.idp.sign_in_as(Person(subject="other-browser"))
    other_callback = install.idp.approve(other.get(START, follow_redirects=False).headers["location"])
    assert other.get(other_callback, follow_redirects=False).status_code == 303  # its own session
    install.idp.sign_in_as(Person(subject="asks-for-step-up"))

    carried = other.get(callback, follow_redirects=False)

    assert _refusal(carried) == sign_in_flow.EXPIRED
    assert _auth_time(install.db, me.session_id) == me.old and _step_ups(install) == []
    confirmed = install.client.get(callback, follow_redirects=False)
    assert confirmed.status_code == 303  # its own browser: confirmed, with a new session
    _token, new = _new_session(install.db, confirmed)
    assert new.auth_time > me.old
    held = _count(install.db, "SELECT COUNT(*) FROM auth_sessions WHERE user_id = :u", u=me.user_id)

    assert _refusal(install.client.get(callback, follow_redirects=False)) == sign_in_flow.EXPIRED
    assert _count(install.db, "SELECT COUNT(*) FROM auth_sessions WHERE user_id = :u", u=me.user_id) == held
    _token, still = _new_session(install.db, confirmed)
    assert sessions.session_state(still, datetime.now(timezone.utc)) == sessions.LIVE  # the reuse ended nothing
    assert _step_ups(install) == [(me.user_id, _confirmed(new.id))]


@pytest.mark.parametrize("auth_time", [int(time.time()) + 3600, "yesterday", True],
                         ids=["an-hour-ahead", "a-string", "a-boolean"])
def test_an_auth_time_in_the_future_or_not_a_number_refuses_and_leaves_the_session(install, auth_time):
    me = _signed_in(install, "odd-auth-time")
    started = _start(install, step_up_too=True)
    install.idp.misbehave(claims={"auth_time": auth_time})

    answer = install.client.get(install.idp.approve(started.headers["location"]), follow_redirects=False)

    assert _refusal(answer) == sign_in_flow.REFUSED
    assert _auth_time(install.db, me.session_id) == me.old and _step_ups(install) == []


# --------------------------------------------------------------------------- fetch metadata


@pytest.mark.parametrize("sent_from", ["cross-site", "same-site", "Cross-Site"])
def test_a_step_up_started_from_another_site_is_refused_and_stores_nothing(install, sent_from):
    _signed_in(install, "lured-to-step-up")
    pending = _count(install.db, "SELECT COUNT(*) FROM oidc_pending_sign_ins")

    answer = install.client.get(START, params={"step_up": "1"}, headers={"Sec-Fetch-Site": sent_from},
                                follow_redirects=False)

    assert _refusal(answer) == sign_in_flow.STEP_UP_REFUSED
    assert _count(install.db, "SELECT COUNT(*) FROM oidc_pending_sign_ins") == pending
    assert "set-cookie" not in answer.headers  # no sign-in bound to this browser


@pytest.mark.parametrize("sent_from", ["same-origin", "none", None])
def test_a_step_up_from_this_site_the_address_bar_or_an_older_browser_starts(install, sent_from):
    _signed_in(install, "steps-up-here")
    headers = {} if sent_from is None else {"Sec-Fetch-Site": sent_from}

    answer = install.client.get(START, params={"step_up": "1"}, headers=headers, follow_redirects=False)

    assert answer.status_code == 303
    assert answer.headers["location"].startswith(install.idp.address(""))


# --------------------------------------------------------------------------- sign-out's host rule


def test_an_end_session_address_on_another_host_is_treated_as_none_unless_listed(install, monkeypatch):
    foreign = "https://elsewhere.example.test/logout"
    install.idp.misbehave(metadata={"end_session_endpoint": foreign})
    oidc_provider.forget_provider()
    first = _signed_in(install, "signs-out-twice")

    answer = install.client.post(SIGN_OUT, headers={"authorization": "Bearer " + first.token})

    assert answer.status_code == 204 and not answer.content
    assert _count(install.db, "SELECT COUNT(*) FROM auth_sessions WHERE id = :id AND revoked_at IS NOT NULL",
                  id=first.session_id) == 1

    monkeypatch.setattr(config, "OIDC_ENDPOINT_HOSTS", "elsewhere.example.test")
    oidc_provider.forget_provider()
    second = _signed_in(install, "signs-out-twice")
    listed = install.client.post(SIGN_OUT, headers={"authorization": "Bearer " + second.token})
    assert listed.status_code == 200 and listed.json()["redirect_to"].startswith(foreign + "?")
