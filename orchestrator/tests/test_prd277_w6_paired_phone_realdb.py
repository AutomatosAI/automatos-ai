"""PRD-277 US-601 against Postgres: a paired phone in enterprise is its person, and only while they are.

* **Who a device is:** the person in its row (``users.id``), with the plain ``user`` role and never
  ``super_admin``, even when that person's email is ``LOCAL_OPERATOR_EMAIL`` or their install role is
  ``super_admin``; its workspace role is that person's membership, read again at each request.
* **Losing access:** deactivating the person (which revokes their devices), a ``users`` row made inactive,
  and leaving the device's workspace each refuse the device at its next request: ``device.py``'s own
  checks, here in enterprise.
* **Pairing:** a signed-in person makes a code, which pairs a phone to them (the row's ``user_id``); the code,
  the pairing and a revoke are each one ``audit_logs`` row in enterprise (a second revoke of the same device
  none), none in ``local``; in enterprise no ``LOCAL_REMOTE_ACCESS`` is needed. The pair link names the
  install's configured address whatever the request's host. A person holds one pending code (a new one
  replaces it) and makes none while they hold the most active devices. A code exchanged with an install id
  another person already paired leaves that person's row as it was.
* **The edition change:** the first start as another edition, or with pairing switched off, revokes every
  paired device, pending codes included, and only paired ones, with one ``device:revoked`` row each and the
  reason for its cause; a token issued before is then ``401 device_revoked``; both directions; switching
  pairing back on brings none back; the same start, a second run and an unmarked ``local`` install revoke
  nothing.

Each test runs in a transaction rolled back at the end; ``commit`` releases a savepoint.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
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

from api import mobile_devices  # noqa: E402
from api.widgets.rate_limit import RateLimitStore  # noqa: E402
from config import config  # noqa: E402
from core import editions  # noqa: E402
from core.auth import device, hybrid, people  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.pairing import mint_token  # noqa: E402
from core.auth.principal import AUTH_SESSION_CLAIM_SOURCE, LOCAL_OPERATOR_CLAIM_SOURCE, claimed_user_pk  # noqa: E402
from core.auth.roles import caller_is_admin  # noqa: E402
from core.auth.session_lane import SESSION_AUTH_TYPE  # noqa: E402
from core.auth.workspace_admin import may_see_own_workspace_health  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from services.mobile import device_audit, devices, edition_change  # noqa: E402

INSTALL = "https://automatos.example.test"
PAIRING_CODES = "/api/v1/mobile/pairing-codes"
PAIR = "/api/v1/mobile/pair"


class _BorrowedSession:
    """The test's session, lent to a lane: ``close()`` is counted, not done."""

    def __init__(self, session):
        self._session = session
        self.closes = 0

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        self.closes += 1


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


def _seed_person(db, *, role=None) -> SimpleNamespace:
    tag = uuid.uuid4().hex[:12]
    email = f"prd277-w6-{tag}@example.test"
    pk = db.execute(
        text("INSERT INTO users (username, email, is_active, system_role) VALUES (:u, :e, true, :r) RETURNING id"),
        {"u": f"prd277-w6-{tag}", "e": email, "r": role},
    ).scalar_one()
    return SimpleNamespace(pk=pk, email=email)


def _member(db, workspace_id, user_pk, role="member") -> None:
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :uid, :role, true)"),
               {"ws": str(workspace_id), "uid": user_pk, "role": role})


def _set_edition(monkeypatch, edition, pairing="true") -> None:
    for name, value in {"AUTH_EDITION": edition, "ENTERPRISE_DEVICE_PAIRING": pairing,
                        "LOCAL_REMOTE_ACCESS": "paired"}.items():
        monkeypatch.setattr(config, name, value)


@pytest.fixture
def install(db_session, seed_workspace, monkeypatch):
    """An enterprise install whose one person is also named the operator and is super_admin, in one workspace."""
    workspace_id = uuid.UUID(seed_workspace())
    person = _seed_person(db_session, role="super_admin")
    _member(db_session, workspace_id, person.pk)
    db_session.execute(text("DELETE FROM system_settings WHERE category = :c AND key = :k"),
                       {"c": edition_change.MARKER_CATEGORY, "k": edition_change.MARKER_KEY})
    db_session.commit()
    _set_edition(monkeypatch, "enterprise")
    for name, value in {"REQUIRE_AUTH": True, "FRONTEND_URL": INSTALL, "PUBLIC_API_HOST": INSTALL,
                        "DEFAULT_WORKSPACE_ID": str(workspace_id), "LOCAL_OPERATOR_EMAIL": person.email}.items():
        monkeypatch.setattr(config, name, value)
    borrowed = _BorrowedSession(db_session)
    monkeypatch.setattr(device, "SessionLocal", lambda: borrowed)
    monkeypatch.setattr(edition_change, "SessionLocal", lambda: borrowed)
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    monkeypatch.setattr(mobile_devices, "_rate_store", RateLimitStore(redis_factory=lambda: None))
    return SimpleNamespace(workspace_id=workspace_id, person=person.pk, email=person.email)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _device(db, install, *, kind="paired", status="active", person=None):
    token, digest = mint_token("amd_", 32)
    row = MobileDevice(workspace_id=install.workspace_id, user_id=person or install.person, kind=kind, status=status,
                       install_id=f"prd277-w6-{uuid.uuid4().hex[:12]}", platform="ios",
                       token_hash=digest if kind == "paired" else None,  # a signed-in phone holds no token
                       paired_at=_now(), events=[])
    db.add(row)
    db.commit()
    return SimpleNamespace(id=row.id, token=token)


def _request(token):
    return SimpleNamespace(method="GET", headers={"authorization": "Bearer " + token}, query_params={},
                           state=SimpleNamespace())


async def _resolve(token) -> RequestContext:
    return await hybrid.get_request_context_hybrid(_request(token))


async def _refused(token) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        await _resolve(token)
    return refused.value


def _audits(db, workspace_id) -> list:
    return db.execute(text(
        "SELECT action, user_id, resource_type, resource_id, details FROM audit_logs "
        "WHERE workspace_id = CAST(:ws AS uuid) AND action LIKE 'device:%' ORDER BY id"),
        {"ws": str(workspace_id)}).all()


def _status_of(db, device_id) -> str:
    db.expire_all()
    return db.get(MobileDevice, device_id).status


# --------------------------------------------------------------------------- who a device is


@pytest.mark.asyncio
async def test_an_enterprise_device_is_its_person_never_the_operator_or_super_admin(db_session, install):
    phone = _device(db_session, install)

    ctx = await _resolve(phone.token)

    assert ctx.auth_type == device.DEVICE_AUTH_TYPE and ctx.workspace_id == install.workspace_id
    assert claimed_user_pk(ctx) == install.person and ctx.user.email == install.email
    assert ctx.user.system_role == "user" and not caller_is_admin(ctx.user)
    assert ctx.admin_all_workspaces is False


@pytest.mark.asyncio
async def test_a_devices_workspace_role_is_its_persons_membership_at_each_request(db_session, install):
    phone = _device(db_session, install)
    assert may_see_own_workspace_health(db_session, await _resolve(phone.token)) is False

    db_session.execute(text("UPDATE workspace_members SET role = 'admin' WHERE workspace_id = CAST(:ws AS uuid) "
                            "AND user_id = :uid"), {"ws": str(install.workspace_id), "uid": install.person})
    db_session.commit()

    assert may_see_own_workspace_health(db_session, await _resolve(phone.token)) is True


# --------------------------------------------------------------------------- losing access


@pytest.mark.asyncio
async def test_deactivating_the_person_refuses_their_device(db_session, install):
    phone = _device(db_session, install)
    admin = _seed_person(db_session, role="admin")

    people.deactivate_person(db_session, install.person, people.Requester(admin.pk, install.workspace_id))

    refused = await _refused(phone.token)
    assert (refused.status_code, refused.detail) == (401, {"code": device.DEVICE_REVOKED})


@pytest.mark.asyncio
async def test_a_person_made_inactive_is_refused_even_before_their_device_is_revoked(db_session, install):
    phone = _device(db_session, install)
    db_session.execute(text("UPDATE users SET is_active = false WHERE id = :id"), {"id": install.person})
    db_session.commit()

    assert (await _refused(phone.token)).status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("removal", [
    "UPDATE workspace_members SET is_active = false WHERE workspace_id = CAST(:ws AS uuid) AND user_id = :uid",
    "DELETE FROM workspace_members WHERE workspace_id = CAST(:ws AS uuid) AND user_id = :uid",
], ids=["membership ended", "membership removed"])
async def test_leaving_the_devices_workspace_refuses_it(db_session, install, removal):
    phone = _device(db_session, install)
    await _resolve(phone.token)
    db_session.execute(text(removal), {"ws": str(install.workspace_id), "uid": install.person})
    db_session.commit()

    assert (await _refused(phone.token)).status_code == 401


# --------------------------------------------------------------------------- pairing


def _session_ctx(install) -> RequestContext:
    user = UserContext(id=install.email, email=install.email, system_role="user",
                       raw_claims={"source": AUTH_SESSION_CLAIM_SOURCE, "user_id": install.person, "auth_time": 0})
    return RequestContext(workspace_id=install.workspace_id, user=user, auth_type=SESSION_AUTH_TYPE)


def _operator_ctx(install) -> RequestContext:
    user = UserContext(id="operator@local", email="operator@local", system_role="super_admin",
                       raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": install.person})
    return RequestContext(workspace_id=install.workspace_id, user=user)


def _client(db_session, ctx) -> TestClient:
    app = FastAPI()
    app.include_router(mobile_devices.router)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    return TestClient(app)


def _pair_with(client, code):
    return client.post(PAIR, json={"code": code, "platform": "ios", "install_id": f"prd277w6{uuid.uuid4().hex[:12]}"})


@pytest.mark.asyncio
async def test_a_signed_in_person_pairs_their_own_phone_and_each_step_is_recorded(db_session, install):
    client = _client(db_session, _session_ctx(install))

    minted = client.post(PAIRING_CODES)
    assert minted.status_code == 200 and minted.headers["cache-control"] == "no-store"
    pending = db_session.query(MobileDevice).filter(MobileDevice.workspace_id == install.workspace_id).one()
    assert (pending.user_id, pending.kind) == (install.person, "paired")

    paired = _pair_with(client, minted.json()["code"])
    assert paired.status_code == 200
    assert claimed_user_pk(await _resolve(paired.json()["token"])) == install.person

    assert client.delete(f"/api/v1/mobile/devices/{paired.json()['device_id']}").status_code == 200
    refused = await _refused(paired.json()["token"])
    assert (refused.status_code, refused.detail) == (401, {"code": device.DEVICE_REVOKED})

    device_id = paired.json()["device_id"]
    assert [(a, uid, rtype) for a, uid, rtype, _r, _d in _audits(db_session, install.workspace_id)] == [
        (device_audit.PAIRING_CODE_CREATED, install.person, device_audit.DEVICE_RESOURCE),
        (device_audit.DEVICE_PAIRED, install.person, device_audit.DEVICE_RESOURCE),
        (device_audit.DEVICE_REVOKED, install.person, device_audit.DEVICE_RESOURCE),
    ]
    assert [r for _a, _u, _t, r, _d in _audits(db_session, install.workspace_id)][1:] == [device_id, device_id]


def test_in_enterprise_pairing_needs_no_remote_access_setting(db_session, install, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    client = _client(db_session, _session_ctx(install))
    assert _pair_with(client, client.post(PAIRING_CODES).json()["code"]).status_code == 200


def test_in_enterprise_the_pair_link_names_the_configured_address_whatever_the_request_says(db_session, install,
                                                                                            monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_API_HOST", "api.elsewhere.example")
    client = _client(db_session, _session_ctx(install))

    minted = client.post(PAIRING_CODES, headers={"host": "attacker.example", "x-forwarded-proto": "http"})

    link = urlsplit(minted.json()["qr_payload"])
    assert f"{link.scheme}://{link.netloc}" == INSTALL and parse_qs(link.query)["api"] == [INSTALL]
    assert "attacker.example" not in minted.text and "http://" not in minted.text


def _pending(db, install, person=None) -> list:
    db.expire_all()
    return db.query(MobileDevice).filter(MobileDevice.workspace_id == install.workspace_id,
                                         MobileDevice.user_id == (person or install.person),
                                         MobileDevice.status == "pending").all()


def test_in_enterprise_a_new_code_replaces_the_persons_pending_one(db_session, install):
    client = _client(db_session, _session_ctx(install))
    first = client.post(PAIRING_CODES).json()["code"]
    _device(db_session, install, status="pending")  # an older one, whatever its time

    second = client.post(PAIRING_CODES).json()["code"]

    assert len(_pending(db_session, install)) == 1
    assert _pair_with(client, first).status_code == 401
    assert _pair_with(client, second).status_code == 200


def test_in_enterprise_no_code_is_made_while_the_person_holds_the_most_devices(db_session, install):
    for _ in range(devices.MAX_ACTIVE_DEVICES_PER_PERSON):
        _device(db_session, install)
    client = _client(db_session, _session_ctx(install))

    refused = client.post(PAIRING_CODES)

    assert refused.status_code == 409
    assert refused.json()["detail"] == {"code": devices.DEVICE_LIMIT_REACHED, "message": devices.DEVICE_LIMIT_MESSAGE}
    assert _pending(db_session, install) == []


def test_a_device_revoked_twice_is_recorded_once(db_session, install):
    client = _client(db_session, _session_ctx(install))
    phone = _device(db_session, install)

    for _ in range(2):
        assert client.delete(f"/api/v1/mobile/devices/{phone.id}").status_code == 200

    assert [action for action, *_rest in _audits(db_session, install.workspace_id)] == [device_audit.DEVICE_REVOKED]


@pytest.mark.asyncio
async def test_a_code_exchanged_with_another_persons_install_id_leaves_their_row_alone(db_session, install):
    other = _seed_person(db_session)
    _member(db_session, install.workspace_id, other.pk)
    theirs = _device(db_session, install, person=other.pk)
    before = db_session.get(MobileDevice, theirs.id)
    kept = (before.user_id, before.status, before.token_hash, before.install_id)
    client = _client(db_session, _session_ctx(install))

    paired = client.post(PAIR, json={"code": client.post(PAIRING_CODES).json()["code"], "platform": "ios",
                                     "install_id": kept[3]})

    assert paired.status_code == 200 and paired.json()["device_id"] != str(theirs.id)
    db_session.expire_all()
    after = db_session.get(MobileDevice, theirs.id)
    assert (after.user_id, after.status, after.token_hash, after.install_id) == kept
    assert claimed_user_pk(await _resolve(theirs.token)) == other.pk
    assert claimed_user_pk(await _resolve(paired.json()["token"])) == install.person


def test_local_pairs_as_before_and_records_nothing(db_session, install, monkeypatch):
    _set_edition(monkeypatch, "local")
    client = _client(db_session, _operator_ctx(install))

    assert _pair_with(client, client.post(PAIRING_CODES).json()["code"]).status_code == 200
    assert _audits(db_session, install.workspace_id) == []


# --------------------------------------------------------------------------- the edition change


def _check(db, edition) -> int:
    """The boot check, starting as ``edition`` (the running one, which ``_set_edition`` set)."""
    assert editions.current_name() == edition
    return edition_change.revoke_if_edition_changed(db, _now())


@pytest.mark.asyncio
@pytest.mark.parametrize("before, after", [("local", "enterprise"), ("enterprise", "local")])
async def test_a_device_paired_under_one_edition_does_not_survive_the_other(db_session, install, monkeypatch,
                                                                             before, after):
    _set_edition(monkeypatch, before)
    _check(db_session, before)  # the install has started as `before`
    phone = _device(db_session, install)
    pending = _device(db_session, install, status="pending")
    signed_in = _device(db_session, install, kind="clerk")
    assert claimed_user_pk(await _resolve(phone.token)) == install.person

    _set_edition(monkeypatch, after)
    assert _check(db_session, after) >= 2  # install-wide: ours, and any other test left committed

    refused = await _refused(phone.token)
    assert (refused.status_code, refused.detail) == (401, {"code": device.DEVICE_REVOKED})
    assert (_status_of(db_session, pending.id), _status_of(db_session, signed_in.id)) == ("revoked", "active")
    assert edition_change.stored_edition(db_session) == STARTED_AS[after]
    _assert_revoked_rows(db_session, install, {phone.id, pending.id}, reason=device_audit.REVOKED_EDITION_CHANGED,
                         before=STARTED_AS[before], after=STARTED_AS[after])
    assert _check(db_session, after) == 0


# What each start is recorded as (editions.started_as), with ENTERPRISE_DEVICE_PAIRING on.
STARTED_AS = {"local": "local", "saas": "saas", "enterprise": "enterprise:pairing_on"}
PAIRING_OFF = "enterprise:pairing_off"


def _assert_revoked_rows(db, install, device_ids, *, reason, before, after) -> None:
    """One ``device:revoked`` row per device, by the install, naming the cause and both starts."""
    rows = _audits(db, install.workspace_id)
    assert {(a, uid, r) for a, uid, _t, r, _d in rows} == {(device_audit.DEVICE_REVOKED, None, str(i))
                                                            for i in device_ids}
    assert {tuple(sorted(d.items())) for *_rest, d in rows} == {tuple(sorted({
        "reason": reason, "person": install.person, "from_edition": before, "to_edition": after}.items()))}


@pytest.mark.parametrize("edition, pairing", [
    ("local", "true"), ("saas", "true"), ("enterprise", "true"), ("enterprise", "false"), ("local", "false"),
])
def test_an_install_that_starts_as_it_did_revokes_nothing(db_session, install, monkeypatch, edition, pairing):
    _set_edition(monkeypatch, edition, pairing)
    edition_change._store_edition(db_session, editions.started_as())
    phone = _device(db_session, install)

    assert _check(db_session, edition) == 0
    assert _status_of(db_session, phone.id) == "active" and _audits(db_session, install.workspace_id) == []


@pytest.mark.asyncio
async def test_switching_pairing_off_ends_every_phone_and_switching_it_on_brings_none_back(db_session, install,
                                                                                         monkeypatch):
    _set_edition(monkeypatch, "enterprise")
    _check(db_session, "enterprise")  # the install has started with pairing on
    phone = _device(db_session, install)
    pending = _device(db_session, install, status="pending")
    assert claimed_user_pk(await _resolve(phone.token)) == install.person

    _set_edition(monkeypatch, "enterprise", pairing="false")
    assert _check(db_session, "enterprise") >= 2 and edition_change.stored_edition(db_session) == PAIRING_OFF
    _assert_revoked_rows(db_session, install, {phone.id, pending.id},
                         reason=device_audit.REVOKED_PAIRING_SWITCHED_OFF,
                         before=STARTED_AS["enterprise"], after=PAIRING_OFF)
    rows = len(_audits(db_session, install.workspace_id))

    _set_edition(monkeypatch, "enterprise")
    _check(db_session, "enterprise")  # install-wide; nothing of ours is live to revoke
    assert edition_change.stored_edition(db_session) == STARTED_AS["enterprise"]
    assert len(_audits(db_session, install.workspace_id)) == rows
    refused = await _refused(phone.token)
    assert (refused.status_code, refused.detail) == (401, {"code": device.DEVICE_REVOKED})
    assert _status_of(db_session, pending.id) == "revoked"


def test_an_unmarked_install_is_taken_to_have_been_local(db_session, install, monkeypatch):
    phone = _device(db_session, install)
    _set_edition(monkeypatch, "local")
    assert _check(db_session, "local") == 0 and _status_of(db_session, phone.id) == "active"

    db_session.execute(text("DELETE FROM system_settings WHERE category = :c AND key = :k"),
                       {"c": edition_change.MARKER_CATEGORY, "k": edition_change.MARKER_KEY})
    _set_edition(monkeypatch, "enterprise")
    assert _check(db_session, "enterprise") >= 1 and _status_of(db_session, phone.id) == "revoked"


def test_the_boot_check_runs_on_a_session_of_its_own_and_closes_it(db_session, install, monkeypatch):
    monkeypatch.setattr(edition_change, "_status", edition_change.NOT_RUN)
    lent = _BorrowedSession(db_session)
    monkeypatch.setattr(edition_change, "SessionLocal", lambda: lent)
    phone = _device(db_session, install)

    assert edition_change.confirm_edition() is True
    assert edition_change.needs_confirming() is False and lent.closes == 1
    assert _status_of(db_session, phone.id) == "revoked"  # unmarked, so taken to have been local
