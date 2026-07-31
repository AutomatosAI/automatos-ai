"""PRD-222 W1·S9 (US-004) — trial config + grant at provisioning.

Pure tests: no Postgres. The grant decision is a pure function; the DB-touching
helpers run against a fake session that returns canned rows and records the
``UPDATE``. Real-Postgres coverage of the provisioning path is CI's job (the
``hybrid.py`` seam is exercised by the auth integration suite on push).
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from config import config
from services.trial_ledger import (
    TRIAL_ACTIVE,
    daily_spend_key,
    decide_trial_grant,
    get_daily_trial_spend,
    grant_trial_at_provisioning,
    user_has_held_trial,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeDB:
    """Session stand-in: dispatches SELECTs by target table, records UPDATEs."""

    def __init__(self, *, has_trial=False, daily_spend_raw=None):
        self._has_trial = has_trial
        self._daily_spend_raw = daily_spend_raw
        self.updates = []  # (sql, params) for every UPDATE
        self.selects = []  # (kind, params)
        self.commits = 0

    def execute(self, clause, params=None):
        upper = str(clause).strip().upper()
        if upper.startswith("SELECT") and "SYSTEM_SETTINGS" in upper:
            self.selects.append(("daily", params))
            row = (self._daily_spend_raw,) if self._daily_spend_raw is not None else None
            return _FakeResult(row)
        if upper.startswith("SELECT") and "WORKSPACES" in upper:
            self.selects.append(("has_trial", params))
            return _FakeResult((1,) if self._has_trial else None)
        if upper.startswith("UPDATE") and "WORKSPACES" in upper:
            self.updates.append((str(clause), params))
            return _FakeResult(None)
        return _FakeResult(None)

    def commit(self):
        self.commits += 1


# --------------------------------------------------------------------------- #
# AC1 — config keys read via config.py with the stated defaults; no os.getenv
# in the new service file.
# --------------------------------------------------------------------------- #


def test_trial_config_defaults():
    # Guard each assertion on its env being unset so a CI override never fails it.
    if not os.getenv("TRIAL_ENABLED"):
        assert config.TRIAL_ENABLED is True
    if not os.getenv("TRIAL_CREDIT_USD"):
        assert config.TRIAL_CREDIT_USD == 5.00
    if not os.getenv("TRIAL_GLOBAL_DAILY_USD"):
        assert config.TRIAL_GLOBAL_DAILY_USD == 25.00
    # All four are the right types.
    assert isinstance(config.TRIAL_ENABLED, bool)
    assert isinstance(config.TRIAL_CREDIT_USD, float)
    assert isinstance(config.TRIAL_GLOBAL_DAILY_USD, float)
    assert isinstance(config.TRIAL_MODEL_ALLOWLIST, str)


def test_trial_model_allowlist_reuses_budget_models():
    # Reuse the platform's existing economical model comma-list — no new id.
    if not os.getenv("TRIAL_MODEL_ALLOWLIST"):
        assert config.TRIAL_MODEL_ALLOWLIST == config.BUDGET_MODELS
    assert config.TRIAL_MODEL_ALLOWLIST  # non-empty
    assert "," in config.TRIAL_MODEL_ALLOWLIST or config.TRIAL_MODEL_ALLOWLIST.strip()


def test_trial_ledger_uses_no_os_getenv():
    # No os.getenv outside config.py — the service reads config.TRIAL_* only.
    src = (Path(__file__).resolve().parents[1] / "services" / "trial_ledger.py").read_text()
    assert "os.getenv" not in src
    assert "import os" not in src


# --------------------------------------------------------------------------- #
# Daily counter — read side (US-005 writes it).
# --------------------------------------------------------------------------- #


def test_daily_spend_key_format():
    assert daily_spend_key(date(2026, 7, 31)) == "trial_spend_2026-07-31"


def test_get_daily_trial_spend_reads_value():
    assert get_daily_trial_spend(_FakeDB(daily_spend_raw="2.50")) == 2.5


def test_get_daily_trial_spend_unset_is_zero():
    assert get_daily_trial_spend(_FakeDB()) == 0.0


def test_get_daily_trial_spend_unparseable_is_zero():
    assert get_daily_trial_spend(_FakeDB(daily_spend_raw="garbage")) == 0.0


# --------------------------------------------------------------------------- #
# Pure grant decision — every branch.
# --------------------------------------------------------------------------- #


def test_decide_grant_happy():
    trial, reason = decide_trial_grant(
        enabled=True, already_held=False, daily_spend=0.0, daily_cap=25.0, credit_usd=5.0
    )
    assert reason == "granted"
    assert trial == {"granted_usd": 5.0, "spent_usd": 0, "state": TRIAL_ACTIVE}


def test_decide_grant_disabled():
    trial, reason = decide_trial_grant(
        enabled=False, already_held=False, daily_spend=0.0, daily_cap=25.0, credit_usd=5.0
    )
    assert trial is None and reason == "disabled"


def test_decide_grant_already_held():
    trial, reason = decide_trial_grant(
        enabled=True, already_held=True, daily_spend=0.0, daily_cap=25.0, credit_usd=5.0
    )
    assert trial is None and reason == "already_held"


def test_decide_grant_daily_cap_reached_at_boundary():
    # >= cap pauses (boundary inclusive).
    trial, reason = decide_trial_grant(
        enabled=True, already_held=False, daily_spend=25.0, daily_cap=25.0, credit_usd=5.0
    )
    assert trial is None and reason == "daily_cap_reached"


def test_decide_grant_rounds_credit():
    trial, _ = decide_trial_grant(
        enabled=True, already_held=False, daily_spend=0.0, daily_cap=25.0, credit_usd=5.005
    )
    assert trial["granted_usd"] == 5.0


# --------------------------------------------------------------------------- #
# One-per-user check.
# --------------------------------------------------------------------------- #


def test_user_has_held_trial_true():
    assert user_has_held_trial(_FakeDB(has_trial=True), 42) is True


def test_user_has_held_trial_false():
    assert user_has_held_trial(_FakeDB(has_trial=False), 42) is False


def test_user_has_held_trial_none_db_or_owner():
    assert user_has_held_trial(None, 42) is False
    assert user_has_held_trial(_FakeDB(has_trial=True), None) is False


# --------------------------------------------------------------------------- #
# grant_trial_at_provisioning — end to end against the fake session.
# --------------------------------------------------------------------------- #


def test_grant_writes_trial_for_fresh_user():
    db = _FakeDB(has_trial=False, daily_spend_raw="0")
    trial = grant_trial_at_provisioning(db, "ws-1", owner_id=7)
    assert trial == {"granted_usd": 5.0, "spent_usd": 0, "state": TRIAL_ACTIVE}
    assert len(db.updates) == 1
    # The UPDATE writes the exact trial JSON to the trial key.
    _sql, params = db.updates[0]
    assert json.loads(params["trial"]) == trial
    assert params["ws_id"] == "ws-1"


def test_grant_skipped_when_user_already_held_trial():
    db = _FakeDB(has_trial=True)
    assert grant_trial_at_provisioning(db, "ws-2", owner_id=7) is None
    assert db.updates == []  # nothing written for a second workspace


def test_grant_skipped_when_kill_switch_off(monkeypatch):
    monkeypatch.setattr(config, "TRIAL_ENABLED", False)
    db = _FakeDB(has_trial=False, daily_spend_raw="0")
    assert grant_trial_at_provisioning(db, "ws-3", owner_id=8) is None
    assert db.updates == []


def test_grant_paused_when_daily_cap_reached(monkeypatch):
    monkeypatch.setattr(config, "TRIAL_GLOBAL_DAILY_USD", 25.0)
    db = _FakeDB(has_trial=False, daily_spend_raw="25.00")
    assert grant_trial_at_provisioning(db, "ws-4", owner_id=9) is None
    assert db.updates == []  # grant PAUSED — nothing written


def test_grant_uses_configured_credit_amount(monkeypatch):
    monkeypatch.setattr(config, "TRIAL_CREDIT_USD", 3.0)
    db = _FakeDB(has_trial=False, daily_spend_raw="0")
    trial = grant_trial_at_provisioning(db, "ws-5", owner_id=10)
    assert trial["granted_usd"] == 3.0


# --------------------------------------------------------------------------- #
# The granted trial round-trips through US-002's public snapshot.
# --------------------------------------------------------------------------- #


def test_granted_trial_surfaces_through_public_snapshot():
    from services.onboarding_state import public_snapshot

    trial, _ = decide_trial_grant(
        enabled=True, already_held=False, daily_spend=0.0, daily_cap=25.0, credit_usd=5.0
    )

    class _WS:
        onboarding = {"stage": "not_started", "trial": trial}

    snap = public_snapshot(_WS())
    assert snap["trial"] == {"granted_usd": 5.0, "spent_usd": 0, "state": TRIAL_ACTIVE}
