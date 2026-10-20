"""PRD-266 US-003c — the device routes: ``api/mobile_devices.py`` over ``services/mobile/devices.py``.

* ``POST /devices``: a device caller (its real ``Bearer amd_…``, through the real lane)
  updates its own row; a Clerk caller upserts one ``clerk`` row by (workspace, person,
  ``install_id``) that never holds a token or code hash (§8). ``push_key`` is in the answer on
  the first registration only (and again after signing out or a new pairing); a clerk row an
  admin revoked stays revoked; an Expo token another row holds is ``409 push_token_in_use``;
  the body is validated (platform, IANA time zone, ``HH:MM``, locale length, Expo's token
  form). Any other lane is 403.
* ``GET /devices``: a member sees their own devices, a workspace owner/admin (and the local
  operator) the workspace's; never a pending code, never a secret.
* ``PATCH``: the device's person only, an active device only, three fields, anything else 422.
* ``DELETE``: the person or an owner/admin; the token hash is kept and the next call is
  ``401 device_revoked``; the push state is cleared.
* ``rotate``: the device lane on its own device with its current token; the old token
  works for ten minutes, then 401 (a frozen clock, through the real lane).
* Tenancy: another person's device is 404 to a device; another workspace's is 404 to
  everyone, its admin and the device's own person included.
"""
from __future__ import annotations

import inspect
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
DEVICES = "/api/v1/mobile/devices"
T0 = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)
TOKEN_BYTES = 32
OVERLAP = timedelta(minutes=10)


def _one(device_id) -> str:
    return f"{DEVICES}/{device_id}"


def _rotate(device_id) -> str:
    return f"{DEVICES}/{device_id}/rotate"


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _register_body(install_id: str, **overrides) -> dict:
    return {"install_id": install_id, "platform": "ios", "app_version": "1.0.0", "locale": "en-IE",
            "timezone": "Europe/Dublin", "expo_push_token": f"ExponentPushToken[{install_id}]",
            "digest_time": "07:30", **overrides}


def _install() -> str:
    return f"prd266-devices-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- fixtures


class _BorrowedSession:
    """The test's transactional session, lent to the device lane: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


class _Clock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here).

    A 409 rolls the route's work back (``_write``); on conftest's session that rollback
    would take the outer transaction, and the test's seed, with it. Here it rolls back to
    the last savepoint only, and every seed helper commits (``_settle``), which releases
    the savepoint its rows were written in, so no route's rollback can take them.
    """
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def clock(monkeypatch):
    from core.auth import device
    from services.mobile import devices

    frozen = _Clock(T0)
    monkeypatch.setattr(device, "_now", frozen)
    monkeypatch.setattr(devices, "_now", frozen)
    return frozen


@pytest.fixture
def local(monkeypatch, db_session, clock):
    """The local edition, the real device lane on the test's session."""
    from config import config
    from core.auth import device, hybrid

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(device, "SessionLocal", lambda: _BorrowedSession(db_session))
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    return config


@pytest.fixture
def make_client(db_session, local):
    """A client on the router: ``ctx`` stands in for the request context, or (None) the real lane."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import mobile_devices
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    def _make(ctx=None):
        app = FastAPI()
        app.include_router(mobile_devices.router)
        app.dependency_overrides[get_db] = lambda: db_session
        if ctx is not None:
            app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
        return TestClient(app)

    return _make


@pytest.fixture
def lane(make_client):
    """A client whose calls carry a real device token through the real lane."""
    return make_client()


@pytest.fixture
def ws(seed_workspace) -> uuid.UUID:
    return uuid.UUID(seed_workspace())


def _settle(db) -> None:
    """Release the seed's savepoint (``db_session``): a route's rollback never reaches it."""
    db.commit()


def _person(db, clerk: bool = False) -> tuple:
    """A users row: ``(pk, email, clerk_user_id or None)``."""
    tag = uuid.uuid4().hex[:12]
    email, clerk_id = f"prd266-devices-{tag}@t.test", (f"user_prd266{tag}" if clerk else None)
    pk = db.execute(
        text("INSERT INTO users (username, email, is_active, clerk_user_id) "
             "VALUES (:u, :e, true, :c) RETURNING id"),
        {"u": f"prd266-devices-{tag}", "e": email, "c": clerk_id},
    ).scalar()
    _settle(db)
    return pk, email, clerk_id


def _clerk(db, workspace_id, role=None):
    """A Clerk caller in ``workspace_id`` (a ``workspace_members`` row when ``role``): ``(ctx, pk)``."""
    from core.auth.dependencies import RequestContext, UserContext

    pk, email, clerk_id = _person(db, clerk=True)
    if role:
        db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                        "VALUES (CAST(:ws AS uuid), :user, :role, TRUE)"),
                   {"ws": str(workspace_id), "user": pk, "role": role})
        _settle(db)
    user = UserContext(id=clerk_id, email=email, clerk_user_id=clerk_id)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="clerk"), pk


def _operator(db, workspace_id):
    """The anonymous local operator (hybrid's local lane): super_admin, its PK on raw_claims."""
    from core.auth.dependencies import RequestContext, UserContext

    pk, email, _ = _person(db)
    user = UserContext(id=email, email=email, system_role="super_admin",
                       raw_claims={"source": "local_operator", "user_id": pk})
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="anonymous"), pk


def _paired(db, workspace_id, user_pk, *, install_id=None):
    """An active paired row and its token: ``(row, token)``."""
    from core.auth.bearer_registry import DEVICE_TOKEN_PREFIX
    from core.auth.pairing import mint_token
    from core.models.mobile_devices import MobileDevice

    token, digest = mint_token(DEVICE_TOKEN_PREFIX, TOKEN_BYTES)
    row = MobileDevice(workspace_id=workspace_id, user_id=user_pk, kind="paired", status="active",
                       install_id=install_id or _install(), platform="ios", token_hash=digest,
                       paired_at=T0, events=[])
    db.add(row)
    _settle(db)
    return row, token


def _row(db, device_id):
    from core.models.mobile_devices import MobileDevice

    db.expire_all()
    return db.get(MobileDevice, device_id)


def _listed_ids(response) -> set:
    assert response.status_code == 200, response.text
    return {device["id"] for device in response.json()["devices"]}


# --------------------------------------------------------------------------- POST /devices


def test_a_devices_first_registration_returns_its_push_key_and_the_second_does_not(db_session, ws, lane):
    pk, _, _ = _person(db_session)
    row, token = _paired(db_session, ws, pk)

    first = lane.post(DEVICES, json=_register_body(row.install_id), headers=_bearer(token))
    second = lane.post(DEVICES, json=_register_body(row.install_id, app_version="1.1.0"), headers=_bearer(token))

    assert first.status_code == 200, first.text
    push_key = first.json()["push_key"]
    assert first.json()["device_id"] == str(row.id) and len(push_key) >= 32
    assert first.headers["Cache-Control"] == "no-store"
    assert second.json() == {"device_id": str(row.id), "push_key": None}
    stored = _row(db_session, row.id)
    assert (stored.kind, stored.push_key, stored.app_version) == ("paired", push_key, "1.1.0")
    assert (stored.locale, stored.timezone, stored.digest_time) == ("en-IE", "Europe/Dublin", "07:30")
    assert stored.expo_push_token == f"ExponentPushToken[{row.install_id}]"


def test_an_optional_field_left_out_keeps_its_value_and_a_null_clears_it(db_session, ws, lane):
    pk, _, _ = _person(db_session)
    row, token = _paired(db_session, ws, pk)
    body = _register_body(row.install_id)
    lane.post(DEVICES, json=body, headers=_bearer(token))

    lane.post(DEVICES, json={k: v for k, v in body.items() if k != "digest_time"}, headers=_bearer(token))
    assert _row(db_session, row.id).digest_time == "07:30"
    lane.post(DEVICES, json={**body, "digest_time": None}, headers=_bearer(token))
    assert _row(db_session, row.id).digest_time is None


def test_a_clerk_registration_upserts_one_clerk_row_with_no_token_hash(db_session, ws, make_client):
    from core.models.mobile_devices import MobileDevice

    ctx, pk = _clerk(db_session, ws, role="editor")
    client, install_id = make_client(ctx), _install()

    first = client.post(DEVICES, json=_register_body(install_id))
    second = client.post(DEVICES, json=_register_body(install_id, app_version="2.0.0"))

    assert first.status_code == 200, first.text
    assert first.json()["push_key"] and second.json()["push_key"] is None
    assert first.json()["device_id"] == second.json()["device_id"]
    rows = db_session.query(MobileDevice).filter(MobileDevice.workspace_id == ws, MobileDevice.user_id == pk).all()
    assert len(rows) == 1
    row = rows[0]
    assert (row.kind, row.status, row.install_id, row.app_version) == ("clerk", "active", install_id, "2.0.0")
    assert (row.token_hash, row.prev_token_hash, row.pairing_code_hash) == (None, None, None)
    assert row.push_key == first.json()["push_key"]
    assert [event["event"] for event in row.events] == ["enrolled"]


def test_a_revoked_clerk_row_registers_again_active_with_a_new_push_key(db_session, ws, make_client):
    ctx, _ = _clerk(db_session, ws)
    client, install_id = make_client(ctx), _install()
    first = client.post(DEVICES, json=_register_body(install_id)).json()
    assert client.delete(_one(first["device_id"])).status_code == 200

    again = client.post(DEVICES, json=_register_body(install_id)).json()

    assert again["device_id"] == first["device_id"]
    assert again["push_key"] and again["push_key"] != first["push_key"]
    row = _row(db_session, uuid.UUID(first["device_id"]))
    assert (row.status, row.revoked_at, row.revoked_by) == ("active", None, None)
    assert [event["event"] for event in row.events] == ["enrolled", "revoked", "enrolled"]


def test_a_phone_paired_again_gets_a_new_push_key_on_its_next_registration(db_session, ws, lane, monkeypatch):
    from config import HOSTED_PUBLIC_API_HOST, config
    from services.mobile import pairing

    monkeypatch.setattr(config, "FRONTEND_URL", "https://studio.prd266.test")
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    pk, install_id = _person(db_session)[0], _install()

    def _pair_and_register() -> str:
        code = pairing.create_pairing_code(db_session, ws, pk, request_base_url="http://testserver").code
        token = pairing.pair_device(db_session, code, "Phone", "ios", "1.0.0", install_id).token
        return lane.post(DEVICES, json=_register_body(install_id), headers=_bearer(token)).json()["push_key"]

    first, again = _pair_and_register(), _pair_and_register()

    assert first and again and again != first


def test_a_device_token_naming_another_install_is_409(db_session, ws, lane):
    pk, _, _ = _person(db_session)
    row, token = _paired(db_session, ws, pk)

    response = lane.post(DEVICES, json=_register_body(_install()), headers=_bearer(token))

    assert (response.status_code, response.json()) == (409, {"detail": {"code": "install_id_mismatch"}})
    assert _row(db_session, row.id).push_key is None


def test_a_clerk_registration_over_a_paired_install_is_409(db_session, ws, make_client):
    ctx, pk = _clerk(db_session, ws)
    row, _ = _paired(db_session, ws, pk)

    response = make_client(ctx).post(DEVICES, json=_register_body(row.install_id))

    assert (response.status_code, response.json()) == (409, {"detail": {"code": "install_id_paired"}})
    assert _row(db_session, row.id).kind == "paired"


def test_an_expo_token_another_row_holds_is_409_until_that_row_lets_it_go(db_session, ws, make_client):
    first, _ = _clerk(db_session, ws)
    second, _ = _clerk(db_session, ws)
    held, other = _install(), _install()
    device_id = make_client(first).post(DEVICES, json=_register_body(held)).json()["device_id"]
    taken = _register_body(other, expo_push_token=f"ExponentPushToken[{held}]")

    clash = make_client(second).post(DEVICES, json=taken)
    assert (clash.status_code, clash.json()) == (409, {"detail": {"code": "push_token_in_use"}})
    assert make_client(first).delete(_one(device_id)).status_code == 200
    assert make_client(second).post(DEVICES, json=taken).status_code == 200


def test_an_admins_revoke_of_a_clerk_row_stands_against_its_persons_next_sign_in(db_session, ws, make_client):
    member, _ = _clerk(db_session, ws, role="editor")
    admin, admin_pk = _clerk(db_session, ws, role="admin")
    install_id = _install()
    device_id = make_client(member).post(DEVICES, json=_register_body(install_id)).json()["device_id"]
    assert make_client(admin).delete(_one(device_id)).status_code == 200

    again = make_client(member).post(DEVICES, json=_register_body(install_id))

    assert (again.status_code, again.json()) == (409, {"detail": {"code": "device_revoked"}})
    row = _row(db_session, uuid.UUID(device_id))
    assert (row.status, row.revoked_by, row.push_key) == ("revoked", admin_pk, None)


def test_only_a_device_or_a_signed_in_caller_registers(db_session, ws, make_client):
    from core.auth.dependencies import RequestContext, UserContext
    from core.models.mobile_devices import MobileDevice

    operator, _ = _operator(db_session, ws)
    api_key = RequestContext(workspace_id=ws, user=UserContext(id="api_key", system_role="admin"), auth_type="api_key")

    for ctx in (operator, api_key):
        assert make_client(ctx).post(DEVICES, json=_register_body(_install())).status_code == 403
    assert db_session.query(MobileDevice).filter(MobileDevice.workspace_id == ws).count() == 0


@pytest.mark.parametrize("overrides", [
    {"platform": "windows"},
    {"timezone": "Mars/Olympus_Mons"},
    {"timezone": "../../etc/passwd"},
    {"timezone": ""},
    {"digest_time": "24:00"},
    {"digest_time": "7:30"},
    {"digest_time": "07:60"},
    {"locale": "x" * 36},
    {"locale": ""},
    {"expo_push_token": "not-an-expo-token"},
    {"expo_push_token": "ExponentPushToken[]"},
    {"expo_push_token": "ExponentPushToken[" + "a" * 240 + "]"},
    {"install_id": "short"},
    {"app_version": None},
])
def test_a_malformed_registration_is_422(db_session, ws, make_client, overrides):
    ctx, _ = _clerk(db_session, ws)

    assert make_client(ctx).post(DEVICES, json=_register_body(_install(), **overrides)).status_code == 422


@pytest.mark.parametrize("platform", ["ios", "android", "web"])
def test_every_platform_registers(db_session, ws, make_client, platform):
    ctx, _ = _clerk(db_session, ws)

    assert make_client(ctx).post(DEVICES, json=_register_body(_install(), platform=platform)).status_code == 200


# --------------------------------------------------------------------------- GET /devices


def _workspace_of_three(db, ws, make_client):
    """A member's, an admin's and someone's paired device, plus a pending code: who is who."""
    from core.auth.pairing import hash_secret
    from core.models.mobile_devices import MobileDevice

    member, _ = _clerk(db, ws, role="editor")
    admin, _ = _clerk(db, ws, role="admin")
    mine = make_client(member).post(DEVICES, json=_register_body(_install())).json()["device_id"]
    theirs = make_client(admin).post(DEVICES, json=_register_body(_install())).json()["device_id"]
    paired, _ = _paired(db, ws, _person(db)[0])
    db.add(MobileDevice(workspace_id=ws, user_id=paired.user_id, kind="paired", status="pending",
                        pairing_code_hash=hash_secret("PENDING1"), pairing_expires_at=T0 + OVERLAP, events=[]))
    _settle(db)
    return member, admin, {mine, theirs, str(paired.id)}, mine


def test_a_member_lists_their_own_devices_and_an_admin_the_workspaces(db_session, ws, make_client):
    member, admin, everyone, mine = _workspace_of_three(db_session, ws, make_client)

    assert _listed_ids(make_client(member).get(DEVICES)) == {mine}
    assert _listed_ids(make_client(admin).get(DEVICES)) == everyone


def test_the_local_operator_lists_the_workspaces_devices(db_session, ws, make_client):
    _, _, everyone, _ = _workspace_of_three(db_session, ws, make_client)
    operator, _ = _operator(db_session, ws)

    assert _listed_ids(make_client(operator).get(DEVICES)) == everyone


def test_the_list_carries_no_secret(db_session, ws, make_client):
    from core.models.mobile_devices import SECRET_COLUMNS

    _workspace_of_three(db_session, ws, make_client)
    operator, _ = _operator(db_session, ws)

    listed = make_client(operator).get(DEVICES).json()["devices"]

    assert listed and not any(set(SECRET_COLUMNS) & set(device) for device in listed)
    assert not any("expo_push_token" in device for device in listed)


# --------------------------------------------------------------------------- tenancy


def test_a_device_cannot_read_patch_rotate_or_revoke_another_persons_device(db_session, ws, lane):
    mine, token = _paired(db_session, ws, _person(db_session)[0])
    theirs, _ = _paired(db_session, ws, _person(db_session)[0])
    before = theirs.token_hash

    assert _listed_ids(lane.get(DEVICES, headers=_bearer(token))) == {str(mine.id)}
    assert lane.patch(_one(theirs.id), json={"push_enabled": True}, headers=_bearer(token)).status_code == 404
    assert lane.post(_rotate(theirs.id), headers=_bearer(token)).status_code == 404
    assert lane.delete(_one(theirs.id), headers=_bearer(token)).status_code == 404
    row = _row(db_session, theirs.id)
    assert (row.status, row.token_hash, row.push_enabled) == ("active", before, False)


def test_another_workspaces_device_is_404_to_everyone_here(db_session, ws, seed_workspace, lane, make_client):
    other_ws = uuid.UUID(seed_workspace())
    pk = _person(db_session)[0]
    _, token = _paired(db_session, ws, pk)
    elsewhere, _ = _paired(db_session, other_ws, pk)  # the same person's device, in another workspace
    operator, _ = _operator(db_session, ws)
    admin, _ = _clerk(db_session, ws, role="owner")

    assert lane.patch(_one(elsewhere.id), json={"push_enabled": True}, headers=_bearer(token)).status_code == 404
    assert lane.delete(_one(elsewhere.id), headers=_bearer(token)).status_code == 404
    for ctx in (operator, admin):
        assert make_client(ctx).delete(_one(elsewhere.id)).status_code == 404
        assert str(elsewhere.id) not in _listed_ids(make_client(ctx).get(DEVICES))
    assert _row(db_session, elsewhere.id).status == "active"


# --------------------------------------------------------------------------- PATCH


def test_patch_changes_the_three_fields(db_session, ws, lane):
    row, token = _paired(db_session, ws, _person(db_session)[0])
    changes = {"push_enabled": True, "digest_time": "18:45", "timezone": "America/New_York"}

    response = lane.patch(_one(row.id), json=changes, headers=_bearer(token))
    cleared = lane.patch(_one(row.id), json={"timezone": None}, headers=_bearer(token))

    assert response.status_code == 200, response.text
    assert {k: response.json()[k] for k in changes} == changes
    assert cleared.json()["timezone"] is None and cleared.json()["push_enabled"] is True


@pytest.mark.parametrize("body", [
    {"platform": "android"},
    {"push_enabled": True, "name": "Renamed"},
    {"push_enabled": None},
    {"digest_time": "25:00"},
    {"timezone": "Not/AZone"},
])
def test_patch_refuses_an_unknown_field_or_a_bad_value(db_session, ws, lane, body):
    row, token = _paired(db_session, ws, _person(db_session)[0])

    assert lane.patch(_one(row.id), json=body, headers=_bearer(token)).status_code == 422
    assert _row(db_session, row.id).platform == "ios"


def test_a_revoked_device_cannot_be_patched(db_session, ws, make_client):
    ctx, _ = _clerk(db_session, ws)
    client = make_client(ctx)
    device_id = client.post(DEVICES, json=_register_body(_install())).json()["device_id"]
    assert client.delete(_one(device_id)).status_code == 200

    assert client.patch(_one(device_id), json={"push_enabled": True}).status_code == 404
    assert _row(db_session, uuid.UUID(device_id)).push_enabled is False


def test_patch_is_the_devices_person_only_not_an_admin(db_session, ws, make_client):
    member, _ = _clerk(db_session, ws, role="viewer")
    admin, _ = _clerk(db_session, ws, role="admin")
    device_id = make_client(member).post(DEVICES, json=_register_body(_install())).json()["device_id"]

    assert make_client(admin).patch(_one(device_id), json={"push_enabled": True}).status_code == 404
    assert make_client(member).patch(_one(device_id), json={"push_enabled": True}).status_code == 200


# --------------------------------------------------------------------------- rotation


def test_rotation_the_old_token_works_for_ten_minutes_then_401(db_session, ws, lane, clock):
    from config import config
    from core.auth.pairing import hash_secret

    row, old = _paired(db_session, ws, _person(db_session)[0])

    response = lane.post(_rotate(row.id), headers=_bearer(old))

    assert response.status_code == 200, response.text
    body, new = response.json(), response.json()["token"]
    assert new.startswith("amd_") and new != old and response.headers["Cache-Control"] == "no-store"
    assert config.MOBILE_TOKEN_ROTATION_OVERLAP_SECONDS == 600
    assert body["previous_token_valid_until"] == (T0 + OVERLAP).isoformat()
    stored = _row(db_session, row.id)
    assert (stored.token_hash, stored.prev_token_hash) == (hash_secret(new), hash_secret(old))
    assert stored.token_rotated_at == T0 and stored.events[-1]["event"] == "rotated"

    clock.now = T0 + OVERLAP - timedelta(seconds=1)
    assert lane.get(DEVICES, headers=_bearer(old)).status_code == 200
    clock.now = T0 + OVERLAP
    refused = lane.get(DEVICES, headers=_bearer(old))
    assert refused.status_code == 401 and refused.json()["detail"] != {"code": "device_revoked"}
    assert lane.get(DEVICES, headers=_bearer(new)).status_code == 200


def test_a_rotated_out_token_may_not_rotate_again(db_session, ws, lane):
    from core.auth.pairing import hash_secret

    row, old = _paired(db_session, ws, _person(db_session)[0])
    new = lane.post(_rotate(row.id), headers=_bearer(old)).json()["token"]

    again = lane.post(_rotate(row.id), headers=_bearer(old))

    assert (again.status_code, again.json()) == (409, {"detail": {"code": "token_superseded"}})
    assert _row(db_session, row.id).token_hash == hash_secret(new)


def test_only_the_device_lane_rotates_and_only_its_own_device(db_session, ws, lane, make_client):
    pk = _person(db_session)[0]
    phone, token = _paired(db_session, ws, pk)
    browser, _ = _paired(db_session, ws, pk)  # the same person's other device
    operator, _ = _operator(db_session, ws)

    assert lane.post(_rotate(browser.id), headers=_bearer(token)).status_code == 404
    assert make_client(operator).post(_rotate(phone.id)).status_code == 404
    assert _row(db_session, phone.id).prev_token_hash is None


# --------------------------------------------------------------------------- revoke


def test_revoke_keeps_the_token_hash_and_the_next_call_is_device_revoked(db_session, ws, lane):
    from core.auth.pairing import hash_secret

    pk = _person(db_session)[0]
    row, token = _paired(db_session, ws, pk)

    response = lane.delete(_one(row.id), headers=_bearer(token))

    assert response.status_code == 200, response.text
    assert (response.json()["status"], response.json()["revoked_by"]) == ("revoked", pk)
    stored = _row(db_session, row.id)
    assert stored.token_hash == hash_secret(token) and stored.revoked_at == T0
    assert stored.events[-1]["event"] == "revoked"
    after = lane.get(DEVICES, headers=_bearer(token))
    assert (after.status_code, after.json()) == (401, {"detail": {"code": "device_revoked"}})


def test_revoke_clears_the_push_state(db_session, ws, lane):
    from core.auth.pairing import hash_secret

    row, token = _paired(db_session, ws, _person(db_session)[0])
    lane.post(DEVICES, json=_register_body(row.install_id), headers=_bearer(token))
    lane.patch(_one(row.id), json={"push_enabled": True}, headers=_bearer(token))

    body = lane.delete(_one(row.id), headers=_bearer(token)).json()

    assert (body["push_enabled"], body["push_registered"], body["push_disabled_reason"]) == \
        (False, False, "device_revoked")
    stored = _row(db_session, row.id)
    assert (stored.expo_push_token, stored.push_key, stored.token_hash) == (None, None, hash_secret(token))


def test_an_admin_revokes_a_members_device_and_a_member_cannot_revoke_anothers(db_session, ws, make_client):
    member, _ = _clerk(db_session, ws, role="editor")
    admin, admin_pk = _clerk(db_session, ws, role="admin")
    members = make_client(member).post(DEVICES, json=_register_body(_install())).json()["device_id"]
    admins = make_client(admin).post(DEVICES, json=_register_body(_install())).json()["device_id"]

    assert make_client(member).delete(_one(admins)).status_code == 404
    revoked = make_client(admin).delete(_one(members))
    again = make_client(admin).delete(_one(members))

    assert revoked.json()["revoked_by"] == admin_pk and again.status_code == 200
    events = [event["event"] for event in _row(db_session, uuid.UUID(members)).events]
    assert events == ["enrolled", "revoked"]
    assert _row(db_session, uuid.UUID(admins)).status == "active"


# --------------------------------------------------------------------------- wiring and secrets


def test_the_device_routes_are_plain_def_and_listed():
    from api import mobile_devices

    for handler in (mobile_devices.register_device, mobile_devices.list_devices, mobile_devices.update_device,
                    mobile_devices.revoke_device, mobile_devices.rotate_device_token):
        assert not inspect.iscoroutinefunction(handler)
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    for method, path in (("POST", DEVICES), ("GET", DEVICES), ("PATCH", f"{DEVICES}/{{device_id}}"),
                         ("DELETE", f"{DEVICES}/{{device_id}}"), ("POST", f"{DEVICES}/{{device_id}}/rotate")):
        assert {"method": method, "path": path} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])


def test_no_log_line_carries_a_token_a_hash_or_a_push_key(db_session, ws, lane, caplog):
    from core.auth.pairing import hash_secret

    row, old = _paired(db_session, ws, _person(db_session)[0])
    with caplog.at_level(logging.DEBUG):
        push_key = lane.post(DEVICES, json=_register_body(row.install_id), headers=_bearer(old)).json()["push_key"]
        new = lane.post(_rotate(row.id), headers=_bearer(old)).json()["token"]
        lane.post(_rotate(row.id), headers=_bearer(old))
        lane.delete(_one(row.id), headers=_bearer(new))

    lines = [r.getMessage() for r in caplog.records
             if r.name.startswith(("api.mobile_devices", "services.mobile", "core.auth"))]
    for secret in (old, new, push_key, hash_secret(old), hash_secret(new)):
        assert not any(secret in line for line in lines)
