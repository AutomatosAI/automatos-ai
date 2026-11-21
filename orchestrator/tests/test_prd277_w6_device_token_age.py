"""PRD-277 US-601 (open question 11, decided): in enterprise a device token has a maximum age.

* **The setting** (``ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS``): 90 when unset; in enterprise start-up takes a
  whole number from 1 to 3650 and refuses anything else, naming the setting and never the value. There is
  no "off". ``local`` and ``saas`` never read it: no value stops them.
* **The age** counts from the phone's last pairing or sign-in on the phone (``paired_at``), never from a
  rotation, from the row's creation where no pairing time was recorded. A pairing time later than now is read
  as now, and the device ages out one maximum age after that time. A device is past its age from the
  moment the age reaches the maximum.
* **One rule, three readers**: push fan-out leaves a device past its age out in enterprise and builds
  exactly today's filters elsewhere; the device list shows such a device as ``expired`` in enterprise only.

The device lane's refusal, the revoke and its one audit row are proved against Postgres
(``test_prd277_w6_device_token_age_realdb.py``).
"""
from __future__ import annotations

import importlib
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from sqlalchemy import select  # noqa: E402

from config import config  # noqa: E402
from core import edition_start_up, editions  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from services.mobile import token_age  # noqa: E402
from services.push import fanout  # noqa: E402
from tests.test_prd277_enterprise_start_up import (  # noqa: E402,F401  (the fixture restores config after a reload)
    ENV_KEYS,
    LOCAL_WS,
    OIDC_ENV,
    STAND_IN_CLERK_SECRET,
    _restore_config_module,
)

SETTING = editions.DEVICE_TOKEN_MAX_AGE_SETTING
NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
# What config.py hands start-up: the number, or the text as it was when it is not one.
ACCEPTED = [1, 30, 90, 3650]
REFUSED = [0, -1, 3651, 100_000, "abc", "", "1.5", "ninety", True, False, None]


def _fresh(**env):
    """Reload config.py with only ``env`` among the start-up names and the setting; its singleton."""
    keys = (*ENV_KEYS, SETTING)
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    os.environ.update({key: value for key, value in env.items() if value is not None})
    try:
        import config as config_mod

        importlib.reload(config_mod)
        return config_mod.config
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------- the setting


def test_the_setting_is_90_days_when_unset_and_an_enterprise_install_starts():
    cfg = _fresh(AUTH_EDITION="enterprise", **OIDC_ENV)
    assert getattr(cfg, SETTING) == 90
    cfg.validate_auth_edition()  # must not raise


@pytest.mark.parametrize("value, read", [("1", 1), (" 30 ", 30), ("3650", 3650)])
def test_an_enterprise_install_starts_with_a_whole_number_from_1_to_3650(value, read):
    cfg = _fresh(AUTH_EDITION="enterprise", **OIDC_ENV, **{SETTING: value})
    assert getattr(cfg, SETTING) == read
    cfg.validate_auth_edition()  # must not raise


@pytest.mark.parametrize("value", ["0", "3651", "-5", "ninety", "", "1.5", "off"])
def test_start_up_refuses_any_other_value_naming_the_setting_never_the_value(value):
    cfg = _fresh(AUTH_EDITION="enterprise", **OIDC_ENV, **{SETTING: value})
    with pytest.raises(edition_start_up.EditionNotReady) as refused:
        cfg.validate_auth_edition()
    message = str(refused.value)
    assert SETTING in message and edition_start_up.DEVICE_TOKEN_MAX_AGE_PROBLEM in message
    if value.strip():
        assert value not in message.replace(SETTING, "").replace(edition_start_up.DEVICE_TOKEN_MAX_AGE_PROBLEM, "")


@pytest.mark.parametrize("days", ACCEPTED)
def test_the_check_takes_a_whole_number_of_days_from_1_to_3650(days):
    edition_start_up.check_device_token_max_age(editions.CAPABILITIES["enterprise"], {SETTING: days})


@pytest.mark.parametrize("days", REFUSED)
def test_the_check_refuses_every_other_value(days):
    with pytest.raises(edition_start_up.EditionNotReady) as refused:
        edition_start_up.check_device_token_max_age(editions.CAPABILITIES["enterprise"], {SETTING: days})
    assert SETTING in str(refused.value)


@pytest.mark.parametrize("edition", ["local", "saas"])
@pytest.mark.parametrize("days", REFUSED)
def test_local_and_saas_never_read_the_setting(edition, days):
    assert editions.CAPABILITIES[edition].device_tokens_age_out is False
    edition_start_up.check_device_token_max_age(editions.CAPABILITIES[edition], {SETTING: days})


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_local_and_saas_start_whatever_the_setting_says(edition):
    env = {"DEFAULT_WORKSPACE_ID": LOCAL_WS} if edition == "local" else {
        "CLERK_JWKS_URL": "https://clerk.example.test/.well-known/jwks.json", "CLERK_SECRET_KEY": STAND_IN_CLERK_SECRET}
    _fresh(AUTH_EDITION=edition, **env, **{SETTING: "ninety"}).validate_auth_edition()  # must not raise


def test_the_setting_is_a_known_config_name():
    import json
    from pathlib import Path

    surface = json.loads((Path(__file__).resolve().parents[1] / "reports" / "config-surface.json")
                         .read_text(encoding="utf-8"))["settings"]
    assert SETTING in surface and hasattr(config, SETTING)
    assert edition_start_up.DEVICE_TOKEN_MAX_AGE_SETTING_NAMES == (SETTING,)


# --------------------------------------------------------------------------- the age


def _row(**columns):
    values = {"paired_at": None, "created_at": NOW - timedelta(days=400), "token_rotated_at": None, **columns}
    return SimpleNamespace(**values)


def test_the_age_counts_from_the_last_pairing_or_sign_in_never_from_a_rotation():
    row = _row(paired_at=NOW - timedelta(days=10), token_rotated_at=NOW - timedelta(minutes=1))
    assert token_age.issued_at(row, NOW) == NOW - timedelta(days=10)


def test_a_row_with_no_pairing_time_counts_from_its_creation():
    assert token_age.issued_at(_row(), NOW) == NOW - timedelta(days=400)


def test_a_pairing_time_without_a_zone_reads_as_utc():
    naive = (NOW - timedelta(days=3)).replace(tzinfo=None)
    assert token_age.issued_at(_row(paired_at=naive), NOW) == NOW - timedelta(days=3)


def test_a_device_is_past_its_maximum_age_from_the_moment_its_age_reaches_it(monkeypatch):
    monkeypatch.setattr(config, SETTING, 90)
    limit = timedelta(days=90)
    assert token_age.is_past_max_age(_row(paired_at=NOW - limit), NOW) is True
    assert token_age.is_past_max_age(_row(paired_at=NOW - limit + timedelta(seconds=1)), NOW) is False


def test_a_rotated_token_of_a_phone_paired_long_ago_is_past_its_maximum_age(monkeypatch):
    monkeypatch.setattr(config, SETTING, 90)
    row = _row(paired_at=NOW - timedelta(days=91), token_rotated_at=NOW - timedelta(days=1))
    assert token_age.is_past_max_age(row, NOW) is True


def test_the_maximum_age_follows_the_setting(monkeypatch):
    monkeypatch.setattr(config, SETTING, 1)
    assert token_age.max_age() == timedelta(days=1)
    assert token_age.is_past_max_age(_row(paired_at=NOW - timedelta(days=2)), NOW) is True


def test_a_pairing_time_later_than_now_is_read_as_now_and_ages_from_that_time(monkeypatch):
    monkeypatch.setattr(config, SETTING, 1)
    row = _row(paired_at=NOW + timedelta(days=30))
    assert token_age.issued_at(row, NOW) == NOW
    assert token_age.is_past_max_age(row, NOW) is False
    # Read as now at every moment before it, so the device is never past its age early...
    assert token_age.is_past_max_age(row, NOW + timedelta(days=30)) is False
    # ...and one maximum age after that pairing time, it is.
    assert token_age.is_past_max_age(row, NOW + timedelta(days=31)) is True


# --------------------------------------------------------------------------- push fan-out and the device list


def _fanout_sql(monkeypatch, edition) -> str:
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    query = select(MobileDevice.id).where(*fanout._device_filters("00000000-0000-0000-0000-0000000277f1"))
    return str(query.compile())


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_fan_out_builds_todays_filters_where_tokens_never_age_out(monkeypatch, edition):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    assert token_age.within_max_age_filters(NOW) == ()
    assert len(fanout._device_filters("00000000-0000-0000-0000-0000000277f1")) == 7
    assert "coalesce" not in _fanout_sql(monkeypatch, edition).lower()


def test_fan_out_in_enterprise_keeps_only_devices_inside_their_maximum_age(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    (kept,) = token_age.within_max_age_filters(NOW)
    sql = str(kept.compile(compile_kwargs={"literal_binds": False})).lower()
    assert "coalesce(mobile_devices.paired_at, mobile_devices.created_at) >" in sql
    assert kept.right.value == token_age.cutoff(NOW)
    assert "coalesce" in _fanout_sql(monkeypatch, "enterprise").lower()


@pytest.mark.parametrize("edition, shown", [("enterprise", token_age.EXPIRED_STATUS), ("local", "active"),
                                             ("saas", "active")])
def test_the_device_list_shows_an_active_device_past_its_age_as_expired_in_enterprise_only(
        monkeypatch, edition, shown):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, SETTING, 90)
    old = _row(paired_at=NOW - timedelta(days=91), status="active")
    assert token_age.view_status(old, NOW) == shown
    assert token_age.view_status(_row(paired_at=NOW - timedelta(days=1), status="active"), NOW) == "active"
    assert token_age.view_status(_row(paired_at=NOW - timedelta(days=91), status="revoked"), NOW) == "revoked"
