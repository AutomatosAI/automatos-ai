"""PRD-277 US-601 against Postgres: in enterprise a device token past its maximum age is refused.

Through the real device lane (``core.auth.hybrid`` to ``core.auth.device``), on the fixtures of
``test_prd277_w6_paired_phone_realdb.py``:

* a device, of either kind that holds a token, whose phone last paired or signed in
  ``ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS`` ago is ``401 device_token_expired``, revoked by the install, and
  recorded once (``device:revoked``, reason ``token_max_age_reached``); its next request is
  ``device_revoked`` and records nothing more;
* a rotation does not restart the count, and the rotate route issues no new token for such a device;
  enrolling the same phone again restarts it; a row with no pairing time counts from its creation;
* a device inside its maximum age is served, and the setting decides the maximum;
* a revoke whose commit fails leaves the device active and the request refused; the next one revokes it;
* push fan-out leaves such a device out in enterprise, and reaches it in ``local``;
* ``local`` never reads the setting: a phone paired years ago keeps working there; ``saas`` neither
  refuses nor revokes one for its age.

Each test runs in a transaction rolled back at the end; ``commit`` releases a savepoint.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import mobile_devices  # noqa: E402
from config import config  # noqa: E402
from core.auth import device  # noqa: E402
from core.auth.pairing import mint_token  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from services.mobile import device_audit  # noqa: E402
from services.push import fanout  # noqa: E402
from tests.test_prd277_w6_paired_phone_realdb import (  # noqa: E402,F401  (fixtures and helpers)
    PAIR,
    PAIRING_CODES,
    _audits,
    _client,
    _refused,
    _resolve,
    _session_ctx,
    _set_edition,
    _status_of,
    db_session,
    install,
)

SETTING = "ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _push_fields() -> dict:
    """What makes an active device one push can reach."""
    return {"push_enabled": True, "expo_push_token": f"ExponentPushToken[prd277-age-{os.urandom(6).hex()}]",
            "push_key": os.urandom(16).hex()}


def _phone(db, install, *, kind="paired", paired_days_ago=None, created_days_ago=None, rotated=False,
           pushable=False):
    """An active device holding a token, paired (or signed in on) ``paired_days_ago`` days ago."""
    token, digest = mint_token("amd_", 32)
    row = MobileDevice(workspace_id=install.workspace_id, user_id=install.person, kind=kind, status="active",
                       install_id=f"prd277-age-{os.urandom(6).hex()}", platform="ios", token_hash=digest,
                       paired_at=None if paired_days_ago is None else _now() - timedelta(days=paired_days_ago),
                       token_rotated_at=_now() if rotated else None, events=[],
                       **(_push_fields() if pushable else {}))
    if created_days_ago is not None:
        row.created_at = _now() - timedelta(days=created_days_ago)
    db.add(row)
    db.commit()
    return SimpleNamespace(id=row.id, token=token, install_id=row.install_id)


def _row(db, device_id) -> MobileDevice:
    db.expire_all()
    return db.get(MobileDevice, device_id)


def _revocations(db, install) -> list:
    return [row for row in _audits(db, install.workspace_id) if row[0] == device_audit.DEVICE_REVOKED]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["paired", "oidc"])
async def test_a_device_past_its_maximum_age_is_refused_revoked_and_recorded_once(db_session, install, kind):
    phone = _phone(db_session, install, kind=kind, paired_days_ago=91)

    first = await _refused(phone.token)
    assert (first.status_code, first.detail) == (401, {"code": device.DEVICE_TOKEN_EXPIRED})
    assert _status_of(db_session, phone.id) == "revoked"
    rows = _revocations(db_session, install)
    assert [(action, user_id, resource_id) for action, user_id, _type, resource_id, _d in rows] == [
        (device_audit.DEVICE_REVOKED, None, str(phone.id))]
    assert rows[0][4] == {"reason": device_audit.REVOKED_TOKEN_MAX_AGE, "person": install.person,
                          "max_age_days": 90}

    again = await _refused(phone.token)
    assert (again.status_code, again.detail) == (401, {"code": device.DEVICE_REVOKED})
    assert len(_revocations(db_session, install)) == 1


@pytest.mark.asyncio
async def test_the_refusal_is_told_apart_from_a_token_nobody_holds(db_session, install):
    unknown = await _refused(mint_token("amd_", 32)[0])
    assert unknown.status_code == 401 and unknown.detail != {"code": device.DEVICE_TOKEN_EXPIRED}


@pytest.mark.asyncio
async def test_a_rotation_does_not_restart_the_count(db_session, install):
    phone = _phone(db_session, install, paired_days_ago=91, rotated=True)

    refused = await _refused(phone.token)
    assert refused.detail == {"code": device.DEVICE_TOKEN_EXPIRED}


@pytest.mark.asyncio
async def test_a_device_with_no_pairing_time_counts_from_its_creation(db_session, install):
    old = _phone(db_session, install, created_days_ago=91)
    new = _phone(db_session, install, created_days_ago=1)

    assert (await _refused(old.token)).detail == {"code": device.DEVICE_TOKEN_EXPIRED}
    assert (await _resolve(new.token)).auth_type == device.DEVICE_AUTH_TYPE


@pytest.mark.asyncio
async def test_a_device_inside_its_maximum_age_is_served(db_session, install):
    phone = _phone(db_session, install, kind="oidc", paired_days_ago=89)

    assert (await _resolve(phone.token)).auth_type == device.DEVICE_AUTH_TYPE
    assert _status_of(db_session, phone.id) == "active" and not _revocations(db_session, install)


@pytest.mark.asyncio
async def test_the_setting_decides_the_maximum_age(db_session, install, monkeypatch):
    monkeypatch.setattr(config, SETTING, 1)
    phone = _phone(db_session, install, paired_days_ago=2)

    refused = await _refused(phone.token)
    assert refused.detail == {"code": device.DEVICE_TOKEN_EXPIRED}
    assert _revocations(db_session, install)[0][4]["max_age_days"] == 1


@pytest.mark.asyncio
async def test_local_never_reads_the_setting(db_session, install, monkeypatch):
    phone = _phone(db_session, install, paired_days_ago=3650)
    _set_edition(monkeypatch, "local")
    monkeypatch.setattr(config, SETTING, "not a number")

    assert (await _resolve(phone.token)).auth_type == device.DEVICE_AUTH_TYPE
    assert _status_of(db_session, phone.id) == "active" and not _revocations(db_session, install)


def _device_lane_client(db_session) -> TestClient:
    """The device routes behind the real device lane: no request context is stood in."""
    app = FastAPI()
    app.include_router(mobile_devices.router)
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)


def test_the_rotate_route_issues_no_token_to_a_device_past_its_maximum_age(db_session, install):
    phone = _phone(db_session, install, paired_days_ago=91)
    held = _row(db_session, phone.id).token_hash

    rotated = _device_lane_client(db_session).post(f"/api/v1/mobile/devices/{phone.id}/rotate",
                                                   headers={"authorization": "Bearer " + phone.token})

    assert rotated.status_code == 401 and rotated.json()["detail"] == {"code": device.DEVICE_TOKEN_EXPIRED}
    assert "token" not in rotated.json()
    row = _row(db_session, phone.id)
    assert (row.status, row.token_hash, row.token_rotated_at, row.prev_token_hash) == ("revoked", held, None, None)


@pytest.mark.asyncio
async def test_enrolling_the_same_phone_again_restarts_the_count(db_session, install):
    phone = _phone(db_session, install, paired_days_ago=91)
    client = _client(db_session, _session_ctx(install))

    code = client.post(PAIRING_CODES).json()["code"]
    again = client.post(PAIR, json={"code": code, "platform": "ios", "install_id": phone.install_id})

    assert again.status_code == 200 and again.json()["device_id"] == str(phone.id)
    assert (await _resolve(again.json()["token"])).auth_type == device.DEVICE_AUTH_TYPE
    assert _status_of(db_session, phone.id) == "active" and not _revocations(db_session, install)


@pytest.mark.asyncio
async def test_a_pairing_time_in_the_future_counts_as_now(db_session, install):
    phone = _phone(db_session, install, paired_days_ago=-400)

    assert (await _resolve(phone.token)).auth_type == device.DEVICE_AUTH_TYPE
    assert _status_of(db_session, phone.id) == "active" and not _revocations(db_session, install)


@pytest.mark.asyncio
async def test_a_revoke_whose_commit_fails_leaves_the_device_active_and_the_request_refused(db_session, install,
                                                                                           monkeypatch):
    phone = _phone(db_session, install, paired_days_ago=91)
    committed = db_session.commit

    def failing_commit():
        raise OperationalError("COMMIT", {}, Exception("the database went away"))

    monkeypatch.setattr(db_session, "commit", failing_commit)
    refused = await _refused(phone.token)
    monkeypatch.setattr(db_session, "commit", committed)

    assert (refused.status_code, refused.detail) == (401, {"code": device.DEVICE_TOKEN_EXPIRED})
    assert _status_of(db_session, phone.id) == "active" and not _revocations(db_session, install)

    assert (await _refused(phone.token)).detail == {"code": device.DEVICE_TOKEN_EXPIRED}
    assert _status_of(db_session, phone.id) == "revoked" and len(_revocations(db_session, install)) == 1


def test_push_fan_out_leaves_out_a_device_past_its_maximum_age_in_enterprise(db_session, install):
    _phone(db_session, install, paired_days_ago=91, pushable=True)
    assert fanout.has_eligible_device(db_session, install.workspace_id) is False

    _phone(db_session, install, paired_days_ago=89, pushable=True)
    assert fanout.has_eligible_device(db_session, install.workspace_id) is True


def test_push_fan_out_in_local_reaches_a_device_whatever_its_age(db_session, install, monkeypatch):
    _phone(db_session, install, paired_days_ago=3650, pushable=True)
    _set_edition(monkeypatch, "local")

    assert fanout.has_eligible_device(db_session, install.workspace_id) is True


@pytest.mark.asyncio
async def test_saas_neither_refuses_nor_revokes_a_device_for_its_age(db_session, install, monkeypatch):
    phone = _phone(db_session, install, paired_days_ago=3650)
    _set_edition(monkeypatch, "saas")
    monkeypatch.setattr(config, SETTING, "not a number")

    refused = await _refused(phone.token)

    assert refused.status_code == 401 and refused.detail != {"code": device.DEVICE_TOKEN_EXPIRED}
    assert _status_of(db_session, phone.id) == "active" and not _revocations(db_session, install)
