"""PRD-277 US-603 against Postgres and the fake provider: a phone signs in at Automatos's own start address.

Through the real routes (``api/auth_oidc.py``, ``api/phone_sign_in.py`` and ``api/mobile_devices.py``), the
phone's system browser (``TestClient`` on the install's address, keeping its cookies) and the in-process
provider (US-407), each test in a transaction rolled back at the end (``commit`` releases a savepoint):

* **The flow, end to end.** The phone starts with its own PKCE challenge; the provider is asked for a fresh
  sign-in (``max_age=0``, ``prompt=login``), a browser's start is not. The callback makes no code and no device:
  it records the sign-in as awaiting confirmation, sets the ``__Host-`` confirmation cookie and answers 303 to
  the confirmation page with nothing in its address. The page's read answers who, which app, which platform
  and whether the phone was removed before, and nothing else. Continue makes the code (a minute, once), writes
  ``auth:sign_in`` (a phone's), clears the cookie and answers the app link with the code; the app exchanges it
  with its verifier for an ``amd_`` device token of kind ``oidc``, audited ``device:paired``.
* **Continue** is refused, with nothing made, without the cookie, from another origin or without the
  cross-site header, after the record expired, and the second time. A confirmation token handed to the
  exchange is refused. **Cancel** deletes what waits and clears the cookie.
* **The code** is single use: a second exchange, a wrong verifier (which spends it) and an expired code are each
  the one 401. Signing in again on the same phone keeps its device and replaces its token.
* **Refusals once the person is known** (no workspace, several, the device cap) reach the confirmation page as
  ``?error=<code>``, never the app, are audited ``auth:sign_in_refused`` with that reason, and keep nothing. A
  refusal before (an identity that does not verify, or a token without the claim ``OIDC_CLAIM_SUBJECT``
  names) still goes to the app link. A paused workspace is not one a device can belong to. A phone signs
  in as the person that claim names, as a browser does.
* **One pending sign-in per person per workspace**, expired ones of anybody deleted; the device cap counted
  again inside the exchange; a phone someone else removed says so and its ``device:paired`` row names who.
* **Nothing from the provider reaches the phone.**
* **Devices of kind ``oidc``** hold with pairing on or off, rotate, are revoked by another edition but not by
  pairing switched off, and stop with the person's deactivation (which deletes their pending rows), their
  access ended, their membership ended, and the device's own revoke. As a device they make no pairing code,
  decide no push question and pass no step-up.
"""
from __future__ import annotations

import os
import secrets
import time
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

from api import auth_oidc  # noqa: E402
from api import mobile_devices  # noqa: E402
from api import mobile_push  # noqa: E402
from api import phone_sign_in as phone_routes  # noqa: E402
from api.widgets.rate_limit import RateLimitStore  # noqa: E402
from config import config  # noqa: E402
from core import editions  # noqa: E402
from core.auth import device, hybrid, oidc_provider, people, session_lane, sign_in_flow, step_up  # noqa: E402
from core.auth.oidc_pending import code_challenge  # noqa: E402
from core.auth.pairing import hash_secret, mint_token  # noqa: E402
from core.auth.oidc_sign_in import OidcSignIn  # noqa: E402
from core.auth.phone_sign_in import (  # noqa: E402
    APP_LINKS,
    CONFIRM_COOKIE,
    CONFIRM_PAGE,
    DEVICE_LIMIT_REACHED,
    NO_WORKSPACE,
    SEVERAL_WORKSPACES,
)
from core.auth.principal import claimed_user_pk  # noqa: E402
from core.auth.sign_in_methods import SignInRegistry  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from services.mobile import device_audit, devices, edition_change  # noqa: E402
from tests.helpers_fake_oidc_provider import INSTALL, FakeIdentityProvider, Person, point_install_at  # noqa: E402

START = "/api/v1/auth/oidc/start"
WAITING = "/api/v1/auth/oidc/phone/waiting"
CONTINUE = "/api/v1/auth/oidc/phone/continue"
CANCEL = "/api/v1/auth/oidc/phone/cancel"
EXCHANGE = "/api/v1/mobile/sign-in/exchange"
MARKER = "prd277-w6b-idp-only-" + secrets.token_hex(4)


class _BorrowedSession:
    """The test's session, lent to a lane: ``close()`` is not done."""

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
    """An enterprise install signing in at the fake provider; a first sign-in joins its default workspace."""
    idp = FakeIdentityProvider(tmp_path)
    workspace_id = seed_workspace()
    db_session.commit()
    point_install_at(monkeypatch, idp, DEFAULT_WORKSPACE_ID=workspace_id, ENTERPRISE_DEFAULT_ROLE="editor",
                     ENTERPRISE_DEVICE_PAIRING="true")
    borrowed = _BorrowedSession(db_session)
    monkeypatch.setattr(auth_oidc, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 300, 299, 600)))
    monkeypatch.setattr(mobile_devices, "_rate_store", RateLimitStore(redis_factory=lambda: None))
    monkeypatch.setattr(sign_in_flow, "REGISTRY", SignInRegistry({"oidc": OidcSignIn()}))
    for module in (session_lane, device, edition_change):
        monkeypatch.setattr(module, "SessionLocal", lambda: borrowed)
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    monkeypatch.setattr(edition_change, "_status", edition_change.NOT_RUN)
    app = FastAPI()
    for router in (auth_oidc.router, phone_routes.router, mobile_devices.router):
        app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db_session
    yield SimpleNamespace(idp=idp, db=db_session, workspace_id=workspace_id,
                          client=lambda: TestClient(app, base_url=INSTALL))
    idp.close()
    oidc_provider.forget_provider()


def _same_site() -> dict:
    """What the web app's page sends with a POST: the install's own origin and the cross-site header."""
    return {"origin": INSTALL, config.SESSION_CSRF_HEADER: "1"}


class Phone:
    """One phone: its install id, its PKCE verifier and its own system browser (which keeps cookies)."""

    def __init__(self, install, install_id=None):
        self.install = install
        self.install_id = install_id or f"prd277-w6b-{uuid.uuid4().hex[:12]}"
        self.verifier = secrets.token_urlsafe(48)
        self.browser = install.client()

    def params(self, **changes) -> dict:
        return {"app_link": "go", "code_challenge": code_challenge(self.verifier), "code_challenge_method": "S256",
                "install_id": self.install_id, "platform": "ios", **changes}

    def sign_in(self):
        """Start, approve at the provider, and come back; the callback's answer."""
        started = self.browser.get(START, params=self.params(), follow_redirects=False)
        assert started.status_code == 303, started.text
        return self.browser.get(self.install.idp.approve(started.headers["location"]), follow_redirects=False)

    def confirmation_token(self):
        return self.browser.cookies.get(CONFIRM_COOKIE)

    def press_continue(self, headers=None):
        return self.browser.post(CONTINUE, headers=_same_site() if headers is None else headers)

    def exchange(self, code, verifier=None):
        return self.browser.post(EXCHANGE, json={"code": code, "code_verifier": verifier or self.verifier,
                                                 "name": "A phone", "app_version": "1.0.0"})


def _awaits(answer) -> None:
    """A callback that left the sign-in awaiting confirmation: 303 to the page, nothing in its address."""
    assert answer.status_code == 303, answer.text
    assert answer.headers["location"] == CONFIRM_PAGE


def _app_answer(answer) -> dict:
    """The app link an answer sent the phone to, as its query; asserts it is the fixed link."""
    location = urlsplit(answer.headers["location"] if answer.status_code == 303 else answer.json()["app_link"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == APP_LINKS["go"]
    return {name: values[0] for name, values in parse_qs(location.query).items()}


def _code(phone) -> str:
    """Sign in, Continue: the one-time code the app link carries."""
    _awaits(phone.sign_in())
    pressed = phone.press_continue()
    assert pressed.status_code == 200, pressed.text
    return _app_answer(pressed)["code"]


def _signed_in(phone) -> SimpleNamespace:
    """A phone signed in, confirmed and exchanged: the device's id, token and the person."""
    code = _code(phone)
    answer = phone.exchange(code)
    assert answer.status_code == 200, answer.text
    body = answer.json()
    return SimpleNamespace(id=uuid.UUID(body["device_id"]), token=body["token"], body=body, code=code)


def _page_refusal(answer) -> str:
    assert answer.status_code == 303 and urlsplit(answer.headers["location"]).path == CONFIRM_PAGE
    return parse_qs(urlsplit(answer.headers["location"]).query)["error"][0]


def _person(db, subject) -> int:
    return db.execute(text("SELECT user_id FROM user_identities WHERE subject = :s"), {"s": subject}).scalar_one()


def _pending(db, person) -> list:
    db.expire_all()
    return db.query(MobileDevice).filter(MobileDevice.user_id == person, MobileDevice.status == "pending").all()


def _audits(db, person, workspace_id) -> list:
    return db.execute(text(
        "SELECT action, details FROM audit_logs WHERE workspace_id = CAST(:ws AS uuid) AND user_id = :u "
        "AND (action LIKE 'auth:%' OR action LIKE 'device:%') ORDER BY id"),
        {"ws": str(workspace_id), "u": person}).all()


def _request(token):
    return SimpleNamespace(method="GET", headers={"authorization": "Bearer " + token}, query_params={},
                           state=SimpleNamespace())


async def _refused(token) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        await hybrid.get_request_context_hybrid(_request(token))
    return refused.value


def _row(db, device_id) -> MobileDevice:
    db.expire_all()
    return db.get(MobileDevice, device_id)


def _expire_waiting(db, install_id) -> None:
    db.execute(text("UPDATE mobile_devices SET pairing_expires_at = now() - interval '1 second' "
                    "WHERE sign_in_install_id = :i"), {"i": install_id})
    db.commit()


# --------------------------------------------------------------------------- the flow


@pytest.mark.asyncio
async def test_a_phone_signs_in_confirms_and_exchanges_its_code_for_a_device_token_of_its_person(install):
    install.idp.sign_in_as(Person(subject="phone-person", email="phone.person@example.test"))
    phone = Phone(install)

    answer = phone.sign_in()

    _awaits(answer)
    cookie = next(value for value in answer.headers.get_list("set-cookie") if value.startswith(CONFIRM_COOKIE))
    assert {"HttpOnly", "Secure", "Path=/", "SameSite=lax", "Max-Age=300"} <= {p.strip() for p in cookie.split(";")}
    assert not any(value.startswith(config.SESSION_COOKIE_NAME) for value in answer.headers.get_list("set-cookie"))
    person = _person(install.db, "phone-person")
    assert install.db.execute(text("SELECT count(*) FROM auth_sessions WHERE user_id = :u"),
                              {"u": person}).scalar_one() == 0
    (waiting,) = _pending(install.db, person)
    assert (waiting.kind, waiting.install_id, waiting.pairing_code_hash) == ("oidc", None, None)
    assert waiting.sign_in_confirm_hash == hash_secret(phone.confirmation_token())
    lives = waiting.pairing_expires_at.astimezone(timezone.utc) - datetime.now(timezone.utc)
    assert timedelta(seconds=290) < lives <= timedelta(seconds=300)
    assert [action for action, _ in _audits(install.db, person, install.workspace_id)] == ["auth:identity_linked"]

    shown = phone.browser.get(WAITING)
    assert shown.status_code == 200 and shown.headers["cache-control"] == "no-store"
    assert shown.json() == {"person": "phone.person@example.test", "app": "Automatos Go", "platform": "ios",
                            "removed_before": False}

    pressed = phone.press_continue()
    assert pressed.status_code == 200 and set(pressed.json()) == {"app_link"}
    query = _app_answer(pressed)
    assert set(query) == {"code"} and len(query["code"]) >= 43
    assert phone.confirmation_token() is None
    (coded,) = _pending(install.db, person)
    assert (coded.pairing_code_hash, coded.sign_in_confirm_hash) == (hash_secret(query["code"]), None)

    exchanged = phone.exchange(query["code"])

    assert exchanged.status_code == 200 and exchanged.headers["cache-control"] == "no-store"
    body = exchanged.json()
    assert set(body) == {"device_id", "workspace_id", "token", "rotate_after"} and body["token"].startswith("amd_")
    assert body["workspace_id"] == str(install.workspace_id)
    ctx = await hybrid.get_request_context_hybrid(_request(body["token"]))
    assert ctx.auth_type == device.DEVICE_AUTH_TYPE and claimed_user_pk(ctx) == person
    assert str(ctx.workspace_id) == str(install.workspace_id) and ctx.user.system_role == "user"
    row = _row(install.db, uuid.UUID(body["device_id"]))
    assert (row.kind, row.status, row.install_id, row.platform) == ("oidc", "active", phone.install_id, "ios")
    assert (row.pairing_code_hash, row.sign_in_challenge, row.sign_in_install_id) == (None, None, None)
    actions = _audits(install.db, person, install.workspace_id)
    assert [action for action, _ in actions] == ["auth:identity_linked", "auth:sign_in", device_audit.DEVICE_PAIRED]
    assert actions[1][1] == {"phone": True, "platform": "ios"}
    assert actions[2][1] == {"platform": "ios", "kind": "oidc"}


def test_the_provider_is_asked_for_a_fresh_sign_in_on_a_phones_start_and_not_on_a_browsers(install):
    phone = Phone(install)

    started = phone.browser.get(START, params=phone.params(), follow_redirects=False)
    browser_started = install.client().get(START, follow_redirects=False)

    phone_asked, browser_asked = (
        {name: values[0] for name, values in parse_qs(urlsplit(answer.headers["location"]).query).items()}
        for answer in (started, browser_started))
    assert (phone_asked["max_age"], phone_asked["prompt"]) == ("0", "login")
    assert "max_age" not in browser_asked and "prompt" not in browser_asked
    assert phone_asked["code_challenge"] != code_challenge(phone.verifier) and "install_id" not in phone_asked
    row = install.db.execute(text(
        "SELECT phone_app_link, phone_code_challenge, phone_install_id, phone_platform FROM oidc_pending_sign_ins "
        "WHERE phone_install_id = :i"), {"i": phone.install_id}).one()
    assert tuple(row) == ("go", code_challenge(phone.verifier), phone.install_id, "ios")


def test_a_browsers_sign_in_is_as_it_was(install):
    browser = install.client()
    started = browser.get(START, params={"return_to": "/projects"}, follow_redirects=False)

    answer = browser.get(install.idp.approve(started.headers["location"]), follow_redirects=False)

    assert answer.status_code == 303 and answer.headers["location"] == "/projects"
    assert answer.cookies.get(config.SESSION_COOKIE_NAME, "").startswith("ams_")
    assert CONFIRM_COOKIE not in answer.headers.get("set-cookie", "")
    assert install.db.query(MobileDevice).filter(MobileDevice.kind == "oidc",
                                                 MobileDevice.workspace_id == install.workspace_id).count() == 0


@pytest.mark.parametrize("params", [{"code_challenge_method": "plain"}, {"app_link": "https://elsewhere.example"}])
def test_a_malformed_phone_start_stores_nothing_and_asks_the_provider_nothing(install, params):
    phone = Phone(install)
    before = install.db.execute(text("SELECT count(*) FROM oidc_pending_sign_ins")).scalar_one()

    answer = phone.browser.get(START, params=phone.params(**params), follow_redirects=False)

    assert (answer.status_code, answer.json()) == (400, {"detail": {"code": auth_oidc.INVALID_PHONE_START}})
    assert install.db.execute(text("SELECT count(*) FROM oidc_pending_sign_ins")).scalar_one() == before
    assert install.idp.authorize_requests == []


# --------------------------------------------------------------------------- the confirmation page


def test_continue_makes_a_code_once(install):
    phone = Phone(install)
    _awaits(phone.sign_in())
    token = phone.confirmation_token()

    assert phone.press_continue().status_code == 200
    again = install.client().post(CONTINUE, headers={**_same_site(), "cookie": f"{CONFIRM_COOKIE}={token}"})

    assert (again.status_code, again.json()) == (404, {"detail": {"code": phone_routes.NOTHING_WAITING}})
    assert install.db.query(MobileDevice).filter(MobileDevice.sign_in_install_id == phone.install_id).count() == 1


def test_a_confirmation_token_handed_to_the_exchange_is_refused_and_still_waits(install):
    phone = Phone(install)
    _awaits(phone.sign_in())

    answer = phone.exchange(phone.confirmation_token())

    assert (answer.status_code, answer.json()) == (401, {"detail": {"code": mobile_devices.INVALID_SIGN_IN_CODE}})
    assert phone.browser.get(WAITING).status_code == 200


@pytest.mark.parametrize("sent", [
    {"origin": "https://attacker.example", config.SESSION_CSRF_HEADER: "1"},
    {"origin": "https://attacker.example"},
    {"origin": INSTALL},
    {},
], ids=["another origin", "a form from another site", "no cross-site header", "no origin"])
def test_continue_from_anywhere_but_the_installs_page_is_refused_and_makes_nothing(install, sent):
    phone = Phone(install)
    _awaits(phone.sign_in())

    answer = phone.press_continue(headers=sent)

    assert (answer.status_code, answer.json()["detail"]["code"]) == (403, "cross_site_request_refused")
    (waiting,) = install.db.query(MobileDevice).filter(MobileDevice.sign_in_install_id == phone.install_id).all()
    assert waiting.pairing_code_hash is None and waiting.sign_in_confirm_hash is not None
    assert phone.press_continue().status_code == 200  # the person's own Continue still works


def test_continue_without_the_cookie_or_after_expiry_makes_nothing(install):
    phone = Phone(install)
    _awaits(phone.sign_in())

    assert install.client().post(CONTINUE, headers=_same_site()).status_code == 404
    _expire_waiting(install.db, phone.install_id)
    expired = phone.press_continue()

    assert (expired.status_code, expired.json()) == (404, {"detail": {"code": phone_routes.NOTHING_WAITING}})
    assert install.db.query(MobileDevice).filter(MobileDevice.sign_in_install_id == phone.install_id).count() == 0
    assert phone.browser.get(WAITING).status_code == 404


def test_cancel_deletes_what_waits_and_clears_the_cookie(install):
    phone = Phone(install)
    _awaits(phone.sign_in())

    assert phone.browser.post(CANCEL, headers={"origin": "https://attacker.example"}).status_code == 403
    cancelled = phone.browser.post(CANCEL, headers=_same_site())

    assert cancelled.status_code == 204 and phone.confirmation_token() is None
    assert install.db.query(MobileDevice).filter(MobileDevice.sign_in_install_id == phone.install_id).count() == 0
    assert phone.browser.get(WAITING).status_code == 404


# --------------------------------------------------------------------------- the code


def test_a_code_is_used_once(install):
    phone = Phone(install)
    signed_in = _signed_in(phone)

    again = phone.exchange(signed_in.code)

    assert (again.status_code, again.json()) == (401, {"detail": {"code": mobile_devices.INVALID_SIGN_IN_CODE}})


def test_a_wrong_verifier_spends_the_code(install):
    phone = Phone(install)
    code = _code(phone)

    assert phone.exchange(code, verifier=secrets.token_urlsafe(48)).status_code == 401
    assert phone.exchange(code).status_code == 401
    assert install.db.query(MobileDevice).filter(MobileDevice.kind == "oidc",
                                                 MobileDevice.workspace_id == install.workspace_id).count() == 0


def test_an_expired_code_is_refused_and_gone(install):
    phone = Phone(install)
    code = _code(phone)
    _expire_waiting(install.db, phone.install_id)

    assert phone.exchange(code).status_code == 401
    assert install.db.execute(text("SELECT count(*) FROM mobile_devices WHERE sign_in_install_id = :i"),
                              {"i": phone.install_id}).scalar_one() == 0


@pytest.mark.asyncio
async def test_signing_in_again_on_the_same_phone_keeps_its_device_and_replaces_its_token(install):
    install.idp.sign_in_as(Person(subject="again"))
    phone = Phone(install)
    first = _signed_in(phone)
    phone.verifier = secrets.token_urlsafe(48)

    second = _signed_in(phone)

    assert second.id == first.id and second.token != first.token
    assert (await _refused(first.token)).status_code == 401
    assert claimed_user_pk(await hybrid.get_request_context_hybrid(_request(second.token))) == _person(
        install.db, "again")


# --------------------------------------------------------------------------- refusals once the person is known


def _refusal_rows(db, person) -> list:
    return db.execute(text("SELECT details FROM audit_logs WHERE action = 'auth:sign_in_refused' AND user_id = :u"),
                      {"u": person}).scalars().all()


def test_a_person_in_no_workspace_is_told_so_on_the_page_not_the_app(install, monkeypatch):
    monkeypatch.setattr(config, "ENTERPRISE_DEFAULT_ROLE", "none")
    install.idp.sign_in_as(Person(subject="nowhere"))
    phone = Phone(install)

    answer = phone.sign_in()

    assert _page_refusal(answer) == NO_WORKSPACE and phone.confirmation_token() is None
    person = _person(install.db, "nowhere")
    assert install.db.query(MobileDevice).filter(MobileDevice.user_id == person).count() == 0
    assert _refusal_rows(install.db, person) == [{"reason": NO_WORKSPACE}]


def test_a_person_in_several_workspaces_is_told_so_on_the_page(install, seed_workspace):
    install.idp.sign_in_as(Person(subject="everywhere"))
    _signed_in(Phone(install))
    person = _person(install.db, "everywhere")
    install.db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :u, 'editor', true)"), {"ws": seed_workspace(), "u": person})
    install.db.commit()

    assert _page_refusal(Phone(install).sign_in()) == SEVERAL_WORKSPACES
    assert _refusal_rows(install.db, person) == [{"reason": SEVERAL_WORKSPACES}]


def test_a_paused_workspace_is_not_one_a_device_can_belong_to(install, seed_workspace):
    install.idp.sign_in_as(Person(subject="paused-too"))
    _signed_in(Phone(install))
    person, paused = _person(install.db, "paused-too"), seed_workspace()
    install.db.execute(text("UPDATE workspaces SET paused_at = now() WHERE id = CAST(:ws AS uuid)"), {"ws": paused})
    install.db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :u, 'editor', true)"), {"ws": paused, "u": person})
    install.db.commit()

    assert _signed_in(Phone(install)).body["workspace_id"] == str(install.workspace_id)


def test_a_person_holding_the_most_devices_is_told_so_on_the_page_and_nothing_is_kept(install):
    install.idp.sign_in_as(Person(subject="many-phones"))
    for _ in range(devices.MAX_ACTIVE_DEVICES_PER_PERSON):
        _signed_in(Phone(install))
    person = _person(install.db, "many-phones")

    assert _page_refusal(Phone(install).sign_in()) == DEVICE_LIMIT_REACHED
    assert _pending(install.db, person) == []
    assert _refusal_rows(install.db, person) == [{"reason": DEVICE_LIMIT_REACHED}]


def test_a_refused_identity_is_still_told_so_on_the_app_link(install):
    install.idp.misbehave(signature="wrong_key")
    assert _app_answer(Phone(install).sign_in()) == {"error": sign_in_flow.REFUSED}


def test_a_phone_signs_in_as_the_person_oidc_claim_subject_names(install, monkeypatch):
    monkeypatch.setattr(config, "OIDC_CLAIM_SUBJECT", "oid")
    install.idp.sign_in_as(Person(subject="phone-sub", extra={"oid": "phone-persons-own-id"}))

    signed = _signed_in(Phone(install))

    assert _row(install.db, signed.id).user_id == _person(install.db, "phone-persons-own-id")
    linked_by_sub = text("SELECT COUNT(*) FROM user_identities WHERE subject = 'phone-sub'")
    assert install.db.execute(linked_by_sub).scalar_one() == 0


def test_a_phones_token_without_the_subject_claim_is_told_so_on_the_app_link(install, monkeypatch):
    monkeypatch.setattr(config, "OIDC_CLAIM_SUBJECT", "oid")
    install.idp.sign_in_as(Person(subject="phone-without-oid"))
    assert _app_answer(Phone(install).sign_in()) == {"error": sign_in_flow.REFUSED}


# --------------------------------------------------------------------------- one at a time, the cap, a removed phone


def test_a_new_sign_in_replaces_the_persons_pending_ones_and_clears_anybodys_expired_ones(install):
    install.idp.sign_in_as(Person(subject="two-phones"))
    first, second, third = Phone(install), Phone(install), Phone(install)
    first_code = _code(first)  # holding a code
    _awaits(second.sign_in())  # awaiting Continue
    person = _person(install.db, "two-phones")
    assert first.exchange(first_code).status_code == 401  # the second sign-in replaced it
    other = install.db.execute(text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) "
                                    "RETURNING id"), {"u": f"w6b-{uuid.uuid4().hex[:8]}",
                                                      "e": f"w6b-{uuid.uuid4().hex[:8]}@example.test"}).scalar_one()
    stale = MobileDevice(workspace_id=uuid.UUID(str(install.workspace_id)), user_id=other, kind="oidc",
                         status="pending", sign_in_confirm_hash=hash_secret("stale"), events=[],
                         pairing_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    install.db.add(stale)
    install.db.commit()
    # Keep the id and let go of the object: once the sign-in below deletes the row, reading the
    # object again would ask the database for a row that is no longer there.
    stale_id = stale.id
    install.db.expunge(stale)

    _awaits(third.sign_in())

    assert [row.sign_in_install_id for row in _pending(install.db, person)] == [third.install_id]
    assert install.db.query(MobileDevice.id).filter(MobileDevice.id == stale_id).first() is None
    assert second.press_continue().status_code == 404


def test_the_cap_is_counted_again_inside_the_exchange(install):
    install.idp.sign_in_as(Person(subject="race"))
    for _ in range(devices.MAX_ACTIVE_DEVICES_PER_PERSON - 1):
        _signed_in(Phone(install))
    person = _person(install.db, "race")
    late = Phone(install)
    code = _code(late)  # confirmed while there was room for one more
    install.db.add(MobileDevice(workspace_id=uuid.UUID(str(install.workspace_id)), user_id=person, kind="paired",
                                status="active", install_id="prd277-w6b-tenth", platform="ios",
                                token_hash=mint_token("amd_", 32)[1], paired_at=datetime.now(timezone.utc),
                                events=[]))
    install.db.commit()

    assert late.exchange(code).status_code == 401
    assert install.db.query(MobileDevice).filter(MobileDevice.user_id == person,
                                                 MobileDevice.status == "active").count() == \
        devices.MAX_ACTIVE_DEVICES_PER_PERSON
    assert _pending(install.db, person) == []


def _admin(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(text("INSERT INTO users (username, email, is_active, system_role) "
                           "VALUES (:u, :e, true, 'admin') RETURNING id"),
                      {"u": f"prd277-w6b-{tag}", "e": f"prd277-w6b-{tag}@example.test"}).scalar_one()


@pytest.mark.parametrize("removed_by", ["an admin", "the person"])
def test_a_phone_removed_before_says_so_when_someone_else_removed_it(install, removed_by):
    install.idp.sign_in_as(Person(subject="removed-" + removed_by.replace(" ", "-")))
    phone = Phone(install)
    first = _signed_in(phone)
    person, db = _person(install.db, "removed-" + removed_by.replace(" ", "-")), install.db
    admin = _admin(db)
    remover = admin if removed_by == "an admin" else person
    caller = devices.DeviceCaller(workspace_id=uuid.UUID(str(install.workspace_id)), user_pk=remover,
                                  sees_workspace=remover == admin)
    assert devices.revoke_device(db, caller, first.id).changed
    db.commit()
    phone.verifier = secrets.token_urlsafe(48)

    _awaits(phone.sign_in())
    assert phone.browser.get(WAITING).json()["removed_before"] is (removed_by == "an admin")
    again = phone.exchange(_app_answer(phone.press_continue())["code"])

    assert again.status_code == 200 and uuid.UUID(again.json()["device_id"]) == first.id
    assert _row(db, first.id).status == "active"
    paired = [details for action, details in _audits(db, person, install.workspace_id)
              if action == device_audit.DEVICE_PAIRED]
    assert paired[-1] == {"platform": "ios", "kind": "oidc", "re_enrolled": True, "was_kind": "oidc",
                          "was_status": "revoked", "revoked_by": remover, "revoked_by_self": remover == person}


# --------------------------------------------------------------------------- nothing from the provider


def test_nothing_from_the_identity_provider_reaches_the_phone(install):
    install.idp.sign_in_as(Person(subject="marked-" + MARKER, email=f"{MARKER}@example.test",
                                  extra={"groups": [MARKER], "custom": MARKER}))
    phone = Phone(install)

    answer = phone.sign_in()
    pressed = phone.press_continue()
    exchanged = phone.exchange(_app_answer(pressed)["code"])

    assert exchanged.status_code == 200
    sent_to_app = pressed.json()["app_link"] + exchanged.text + str(dict(exchanged.headers)) + answer.headers["location"]
    assert install.idp.issued and not [value for value in install.idp.issued if value in sent_to_app]
    for leak in (MARKER, install.idp.issuer, install.idp.origin, install.idp.client_id):
        assert leak not in sent_to_app


# --------------------------------------------------------------------------- devices of kind oidc


@pytest.mark.asyncio
async def test_with_pairing_off_a_signed_in_phone_works_and_a_paired_one_does_not(install, monkeypatch):
    paired_token, paired_hash = mint_token("amd_", 32)
    install.idp.sign_in_as(Person(subject="pairing-off"))
    signed_in = _signed_in(Phone(install))
    person = _person(install.db, "pairing-off")
    install.db.add(MobileDevice(workspace_id=uuid.UUID(str(install.workspace_id)), user_id=person, kind="paired",
                                status="active", install_id="prd277-w6b-paired", platform="ios",
                                token_hash=paired_hash, paired_at=datetime.now(timezone.utc), events=[]))
    install.db.commit()
    monkeypatch.setattr(config, "ENTERPRISE_DEVICE_PAIRING", "false")

    assert claimed_user_pk(await hybrid.get_request_context_hybrid(_request(signed_in.token))) == person
    assert (await _refused(paired_token)).status_code == 401
    assert _signed_in(Phone(install)).token.startswith("amd_")


@pytest.mark.asyncio
async def test_a_signed_in_phone_rotates_its_token(install):
    install.idp.sign_in_as(Person(subject="rotates"))
    signed_in = _signed_in(Phone(install))
    client = install.client()

    rotated = client.post(f"/api/v1/mobile/devices/{signed_in.id}/rotate",
                          headers={"authorization": "Bearer " + signed_in.token})

    assert rotated.status_code == 200, rotated.text
    assert claimed_user_pk(await hybrid.get_request_context_hybrid(_request(rotated.json()["token"]))) == _person(
        install.db, "rotates")


def _check(db) -> int:
    return edition_change.revoke_if_edition_changed(db, datetime.now(timezone.utc))


@pytest.mark.asyncio
async def test_another_edition_revokes_a_signed_in_phone_and_pairing_off_does_not(install, monkeypatch):
    edition_change._store_edition(install.db, editions.started_as())
    install.db.commit()
    signed_in = _signed_in(Phone(install))

    monkeypatch.setattr(config, "ENTERPRISE_DEVICE_PAIRING", "false")
    _check(install.db)
    assert _row(install.db, signed_in.id).status == "active"

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    _check(install.db)
    assert _row(install.db, signed_in.id).status == "revoked"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["deactivated", "access ended", "membership ended", "device revoked"])
async def test_a_signed_in_phone_stops_when_its_persons_access_does(install, ending):
    install.idp.sign_in_as(Person(subject="ends-" + ending.replace(" ", "-")))
    signed_in = _signed_in(Phone(install))
    person, db = _person(install.db, "ends-" + ending.replace(" ", "-")), install.db
    requester = people.Requester(_admin(db), install.workspace_id)
    if ending == "deactivated":
        people.deactivate_person(db, person, requester)
    elif ending == "access ended":
        people.end_access_of(db, person, requester)
    elif ending == "membership ended":
        db.execute(text("UPDATE workspace_members SET is_active = false WHERE user_id = :u"), {"u": person})
        db.commit()
    else:  # the person revokes it, as Settings does
        caller = devices.DeviceCaller(workspace_id=uuid.UUID(str(install.workspace_id)), user_pk=person)
        assert devices.revoke_device(db, caller, signed_in.id).changed
        db.commit()

    assert (await _refused(signed_in.token)).status_code == 401


def test_deactivating_a_person_deletes_their_pending_sign_in_and_revokes_only_devices(install):
    install.idp.sign_in_as(Person(subject="pending-then-gone"))
    _signed_in(Phone(install))
    waiting = Phone(install)
    _awaits(waiting.sign_in())
    person = _person(install.db, "pending-then-gone")

    done = people.deactivate_person(install.db, person, people.Requester(_admin(install.db), install.workspace_id))

    assert done.devices_revoked == 1 and _pending(install.db, person) == []
    assert waiting.press_continue().status_code == 404


@pytest.mark.asyncio
async def test_as_a_device_it_makes_no_pairing_code_decides_no_push_and_passes_no_step_up(install):
    install.idp.sign_in_as(Person(subject="device-limits"))
    token = _signed_in(Phone(install)).token
    install.db.execute(text("UPDATE users SET system_role = 'super_admin' WHERE id = :u"),
                       {"u": _person(install.db, "device-limits")})
    install.db.commit()
    ctx = await hybrid.get_request_context_hybrid(_request(token))

    assert ctx.user.system_role == "user"
    for refuse in (lambda: mobile_devices.pairing_code_owner(ctx, db=install.db),
                   lambda: mobile_push.push_consent_decider(ctx),
                   lambda: step_up.check_step_up(ctx, config.STEP_UP_MAX_AGE_MINUTES, time.time())):
        with pytest.raises(HTTPException) as refused:
            refuse()
        assert refused.value.status_code == 403
