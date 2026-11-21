"""PRD-277 US-603 against Postgres and the fake provider: a phone signs in at Automatos's own start address.

Through the real routes (``api/auth_oidc.py`` and ``api/mobile_devices.py``), the phone's system browser
(``TestClient`` on the install's address) and the in-process provider (US-407), each test in a transaction
rolled back at the end (``commit`` releases a savepoint):

* **The flow.** The phone starts with its own PKCE challenge; the start is the browser's (same provider
  redirect, its own state, nonce and PKCE) and keeps what the phone asked for on the sign-in's row. The callback
  answers 303 to the fixed app link with ``code`` and nothing else, sets no session cookie and issues no
  session; the code is kept as a hash on a pending ``oidc`` row for a minute. The app exchanges it with its
  verifier for an ``amd_`` device token of kind ``oidc`` that resolves as the person, in their one workspace,
  with the plain ``user`` role, audited ``device:paired``.
* **The code** is single use: a second exchange, a wrong verifier (which spends it) and an expired code are each
  the one 401. Signing in again on the same phone keeps its device and replaces its token.
* **Refusals reach the app**: a person in no workspace, in several, or at the device cap is sent to the app link
  with the error's code, and no code is kept. A browser's sign-in is as it was.
* **Nothing from the provider reaches the phone**: no code, ID or access token, claim, subject or address of
  the provider is in the redirect to the app or in the exchange's answer.
* **Devices of kind ``oidc``** hold with pairing on or off (a paired device is refused while it is off),
  rotate as paired ones do, are revoked by another edition but not by pairing switched off, and stop with the
  person's deactivation, their access ended, their membership ended, and the device's own revoke. As a device
  they make no pairing code, decide no push question and pass no step-up.
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
from api.widgets.rate_limit import RateLimitStore  # noqa: E402
from config import config  # noqa: E402
from core import editions  # noqa: E402
from core.auth import device, hybrid, oidc_provider, people, session_lane, sign_in_flow, step_up  # noqa: E402
from core.auth.oidc_pending import code_challenge  # noqa: E402
from core.auth.pairing import hash_secret, mint_token  # noqa: E402
from core.auth.oidc_sign_in import OidcSignIn  # noqa: E402
from core.auth.phone_sign_in import APP_LINKS, DEVICE_LIMIT_REACHED, NO_WORKSPACE, SEVERAL_WORKSPACES  # noqa: E402
from core.auth.principal import claimed_user_pk  # noqa: E402
from core.auth.sign_in_methods import SignInRegistry  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from services.mobile import device_audit, devices, edition_change  # noqa: E402
from tests.helpers_fake_oidc_provider import INSTALL, FakeIdentityProvider, Person, point_install_at  # noqa: E402

START = "/api/v1/auth/oidc/start"
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
    app.include_router(auth_oidc.router)
    app.include_router(mobile_devices.router)
    app.dependency_overrides[get_db] = lambda: db_session
    yield SimpleNamespace(idp=idp, db=db_session, workspace_id=workspace_id,
                          client=lambda: TestClient(app, base_url=INSTALL))
    idp.close()
    oidc_provider.forget_provider()


class Phone:
    """One phone: its install id, its PKCE verifier and its own system browser."""

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

    def exchange(self, code, verifier=None):
        return self.browser.post(EXCHANGE, json={"code": code, "code_verifier": verifier or self.verifier,
                                                 "name": "A phone", "app_version": "1.0.0"})


def _app_answer(answer) -> dict:
    """The app link a callback sent the phone to, as its query; asserts it is the fixed link."""
    assert answer.status_code == 303, answer.text
    location = urlsplit(answer.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == APP_LINKS["go"]
    return {name: values[0] for name, values in parse_qs(location.query).items()}


def _signed_in(phone) -> SimpleNamespace:
    """A phone signed in and exchanged: the device's id, token and the person."""
    code = _app_answer(phone.sign_in())["code"]
    answer = phone.exchange(code)
    assert answer.status_code == 200, answer.text
    body = answer.json()
    return SimpleNamespace(id=uuid.UUID(body["device_id"]), token=body["token"], body=body, code=code)


def _person(db, subject) -> int:
    return db.execute(text("SELECT user_id FROM user_identities WHERE subject = :s"), {"s": subject}).scalar_one()


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


# --------------------------------------------------------------------------- the flow


@pytest.mark.asyncio
async def test_a_phone_signs_in_and_exchanges_its_code_for_a_device_token_of_its_person(install):
    install.idp.sign_in_as(Person(subject="phone-person"))
    phone = Phone(install)

    answer = phone.sign_in()

    query = _app_answer(answer)
    assert set(query) == {"code"} and len(query["code"]) >= 43
    assert config.SESSION_COOKIE_NAME not in answer.cookies
    assert not any(value.startswith(config.SESSION_COOKIE_NAME) for value in answer.headers.get_list("set-cookie"))
    person = _person(install.db, "phone-person")
    assert install.db.execute(text("SELECT count(*) FROM auth_sessions WHERE user_id = :u"),
                              {"u": person}).scalar_one() == 0
    pending = install.db.query(MobileDevice).filter(MobileDevice.user_id == person).one()
    assert (pending.kind, pending.status, pending.install_id) == ("oidc", "pending", None)
    assert pending.pairing_code_hash == hash_secret(query["code"]) and pending.sign_in_install_id == phone.install_id
    lives = pending.pairing_expires_at.astimezone(timezone.utc) - datetime.now(timezone.utc)
    assert timedelta(seconds=50) < lives <= timedelta(seconds=60)

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
    actions = install.db.execute(text(
        "SELECT action, details FROM audit_logs WHERE workspace_id = CAST(:ws AS uuid) AND user_id = :u "
        "AND (action LIKE 'auth:%' OR action LIKE 'device:%') ORDER BY id"),
        {"ws": str(install.workspace_id), "u": person}).all()
    assert [action for action, _details in actions] == ["auth:identity_linked", device_audit.DEVICE_PAIRED]
    assert actions[-1][1] == {"platform": "ios", "kind": "oidc"}


def test_the_phones_start_is_kept_on_its_sign_in_and_the_provider_sees_a_browsers_start(install):
    phone = Phone(install)

    started = phone.browser.get(START, params=phone.params(), follow_redirects=False)

    sent = {name: values[0] for name, values in parse_qs(urlsplit(started.headers["location"]).query).items()}
    assert sent["code_challenge"] != code_challenge(phone.verifier) and "install_id" not in sent
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


# --------------------------------------------------------------------------- the code


def test_a_code_is_used_once(install):
    phone = Phone(install)
    signed_in = _signed_in(phone)

    again = phone.exchange(signed_in.code)

    assert (again.status_code, again.json()) == (401, {"detail": {"code": mobile_devices.INVALID_SIGN_IN_CODE}})


def test_a_wrong_verifier_spends_the_code(install):
    phone = Phone(install)
    code = _app_answer(phone.sign_in())["code"]

    assert phone.exchange(code, verifier=secrets.token_urlsafe(48)).status_code == 401
    assert phone.exchange(code).status_code == 401
    assert install.db.query(MobileDevice).filter(MobileDevice.kind == "oidc",
                                                 MobileDevice.workspace_id == install.workspace_id).count() == 0


def test_an_expired_code_is_refused_and_gone(install):
    phone = Phone(install)
    code = _app_answer(phone.sign_in())["code"]
    install.db.execute(text("UPDATE mobile_devices SET pairing_expires_at = now() - interval '1 second' "
                            "WHERE sign_in_install_id = :i"), {"i": phone.install_id})
    install.db.commit()

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


# --------------------------------------------------------------------------- refusals reach the app


def test_a_person_in_no_workspace_is_told_so_on_the_app_link(install, monkeypatch):
    monkeypatch.setattr(config, "ENTERPRISE_DEFAULT_ROLE", "none")
    install.idp.sign_in_as(Person(subject="nowhere"))

    assert _app_answer(Phone(install).sign_in()) == {"error": NO_WORKSPACE}
    assert install.db.query(MobileDevice).filter(MobileDevice.user_id == _person(install.db, "nowhere")).count() == 0


def test_a_person_in_several_workspaces_is_told_so_on_the_app_link(install, seed_workspace):
    install.idp.sign_in_as(Person(subject="everywhere"))
    _signed_in(Phone(install))
    person = _person(install.db, "everywhere")
    install.db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :u, 'editor', true)"), {"ws": seed_workspace(), "u": person})
    install.db.commit()

    assert _app_answer(Phone(install).sign_in()) == {"error": SEVERAL_WORKSPACES}


def test_a_person_holding_the_most_devices_is_told_so_and_no_code_is_kept(install):
    install.idp.sign_in_as(Person(subject="many-phones"))
    _signed_in(Phone(install))
    person = _person(install.db, "many-phones")
    for _ in range(devices.MAX_ACTIVE_DEVICES_PER_PERSON - 1):
        _signed_in(Phone(install))

    assert _app_answer(Phone(install).sign_in()) == {"error": DEVICE_LIMIT_REACHED}
    assert install.db.query(MobileDevice).filter(MobileDevice.user_id == person,
                                                 MobileDevice.status == "pending").count() == 0


def test_a_refused_identity_is_told_so_on_the_app_link(install):
    install.idp.misbehave(signature="wrong_key")
    assert _app_answer(Phone(install).sign_in()) == {"error": sign_in_flow.REFUSED}


# --------------------------------------------------------------------------- nothing from the provider


def test_nothing_from_the_identity_provider_reaches_the_phone(install):
    install.idp.sign_in_as(Person(subject="marked-" + MARKER, email=f"{MARKER}@example.test",
                                  extra={"groups": [MARKER], "custom": MARKER}))
    phone = Phone(install)

    answer = phone.sign_in()
    exchanged = phone.exchange(_app_answer(answer)["code"])

    assert exchanged.status_code == 200
    sent_to_phone = answer.headers["location"] + exchanged.text + str(dict(exchanged.headers))
    assert install.idp.issued and not [value for value in install.idp.issued if value in sent_to_phone]
    for leak in (MARKER, install.idp.issuer, install.idp.origin, install.idp.client_id):
        assert leak not in sent_to_phone


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


def _admin(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(text("INSERT INTO users (username, email, is_active, system_role) "
                           "VALUES (:u, :e, true, 'admin') RETURNING id"),
                      {"u": f"prd277-w6b-{tag}", "e": f"prd277-w6b-{tag}@example.test"}).scalar_one()


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
