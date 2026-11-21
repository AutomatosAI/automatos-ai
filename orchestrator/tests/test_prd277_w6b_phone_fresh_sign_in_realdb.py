"""PRD-277 US-603 (open question 13, decided) against Postgres: a phone's sign-in must prove it was fresh.

Through the real routes and the fake provider, on the fixtures of ``test_prd277_w6b_phone_sign_in_realdb.py``.
The callback applies to a phone's sign-in the step-up's own sign-in time rule
(``step_up_sign_in.sign_in_time_refusal``): the ``auth_time`` the provider reports must be there, and not
earlier than the phone's start beyond the clock skew. Otherwise the sign-in is refused before anything
awaits the person: no person made or linked, no pending device row, no confirmation cookie; one
``auth:sign_in_refused`` row with the rule's reason; and the app link answers ``error=sign_in_refused``,
where a phone's refusals before its person is known already go. A browser's ordinary sign-in is unchanged.
"""
from __future__ import annotations

import os
import time
from urllib.parse import urlsplit

from sqlalchemy import text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.auth import session_audit, sign_in_flow  # noqa: E402
from tests.helpers_fake_oidc_provider import Person  # noqa: E402
from tests.test_prd277_w6b_phone_sign_in_realdb import (  # noqa: E402,F401  (fixtures and helpers)
    START,
    Phone,
    _app_answer,
    _awaits,
    _pending,
    _person,
    _signed_in,
    db_session,
    install,
)

AN_HOUR = 3600


def _refusals(db, workspace_id) -> list:
    return db.execute(text(
        "SELECT user_id, details FROM audit_logs WHERE workspace_id = CAST(:ws AS uuid) "
        "AND action = :a ORDER BY id"), {"ws": str(workspace_id), "a": session_audit.SIGN_IN_REFUSED}).all()


def _linked(db, subject) -> int:
    return db.execute(text("SELECT count(*) FROM user_identities WHERE subject = :s"), {"s": subject}).scalar_one()


def test_a_phones_sign_in_without_auth_time_is_refused_before_anything_awaits(install):
    install.idp.sign_in_as(Person(subject="phone-no-auth-time"))
    install.idp.misbehave(auth_time="never")
    phone = Phone(install)

    answer = phone.sign_in()

    assert _app_answer(answer) == {"error": sign_in_flow.REFUSED}
    assert phone.confirmation_token() is None and _linked(install.db, "phone-no-auth-time") == 0
    assert install.db.execute(text("SELECT count(*) FROM mobile_devices WHERE sign_in_install_id = :i"),
                              {"i": phone.install_id}).scalar_one() == 0
    assert [tuple(row) for row in _refusals(install.db, install.workspace_id)] == [
        (None, {"reason": session_audit.STEP_UP_NO_AUTH_TIME})]


def test_a_phones_sign_in_older_than_its_start_is_refused_naming_its_known_person(install):
    install.idp.sign_in_as(Person(subject="phone-stale"))
    _signed_in(Phone(install))
    person = _person(install.db, "phone-stale")
    install.idp.misbehave(claims={"auth_time": int(time.time()) - AN_HOUR})
    phone = Phone(install)

    answer = phone.sign_in()

    assert _app_answer(answer) == {"error": sign_in_flow.REFUSED}
    assert phone.confirmation_token() is None and _pending(install.db, person) == []
    assert [tuple(row) for row in _refusals(install.db, install.workspace_id)] == [
        (person, {"reason": session_audit.STEP_UP_STALE})]


def test_a_phones_fresh_sign_in_still_awaits_the_persons_continue(install):
    install.idp.sign_in_as(Person(subject="phone-fresh"))

    _awaits(Phone(install).sign_in())

    assert _refusals(install.db, install.workspace_id) == []


def test_a_browsers_ordinary_sign_in_without_auth_time_is_unchanged(install):
    install.idp.sign_in_as(Person(subject="browser-no-auth-time"))
    install.idp.misbehave(auth_time="never")
    browser = install.client()

    started = browser.get(START, follow_redirects=False)
    answer = browser.get(install.idp.approve(started.headers["location"]), follow_redirects=False)

    assert answer.status_code == 303 and "error" not in urlsplit(answer.headers["location"]).query
    assert any(value.startswith(config.SESSION_COOKIE_NAME) for value in answer.headers.get_list("set-cookie"))
    assert _refusals(install.db, install.workspace_id) == []
