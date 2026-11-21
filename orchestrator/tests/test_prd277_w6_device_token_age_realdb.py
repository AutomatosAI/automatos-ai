"""PRD-277 US-601 against Postgres: in enterprise a device token past its maximum age is refused.

Through the real device lane (``core.auth.hybrid`` to ``core.auth.device``), on the fixtures of
``test_prd277_w6_paired_phone_realdb.py``:

* a device, of either kind that holds a token, whose phone last paired or signed in
  ``ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS`` ago is ``401 device_token_expired``, revoked by the install, and
  recorded once (``device:revoked``, reason ``token_max_age_reached``); its next request is
  ``device_revoked`` and records nothing more;
* a rotation does not restart the count; a row with no pairing time counts from its creation;
* a device inside its maximum age is served, and the setting decides the maximum;
* ``local`` never reads the setting: a phone paired years ago keeps working there.

Each test runs in a transaction rolled back at the end; ``commit`` releases a savepoint.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.auth import device  # noqa: E402
from core.auth.pairing import mint_token  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from services.mobile import device_audit  # noqa: E402
from tests.test_prd277_w6_paired_phone_realdb import (  # noqa: E402,F401  (fixtures and helpers)
    _audits,
    _refused,
    _resolve,
    _set_edition,
    _status_of,
    db_session,
    install,
)

SETTING = "ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _phone(db, install, *, kind="paired", paired_days_ago=None, created_days_ago=None, rotated=False):
    """An active device holding a token, paired (or signed in on) ``paired_days_ago`` days ago."""
    token, digest = mint_token("amd_", 32)
    row = MobileDevice(workspace_id=install.workspace_id, user_id=install.person, kind=kind, status="active",
                       install_id=f"prd277-age-{os.urandom(6).hex()}", platform="ios", token_hash=digest,
                       paired_at=None if paired_days_ago is None else _now() - timedelta(days=paired_days_ago),
                       token_rotated_at=_now() if rotated else None, events=[])
    if created_days_ago is not None:
        row.created_at = _now() - timedelta(days=created_days_ago)
    db.add(row)
    db.commit()
    return SimpleNamespace(id=row.id, token=token)


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
