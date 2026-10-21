"""PRD-222 W1·S9 (US-004) — trial config + the pure grant decision.

PRD-265 W1 replaced the $5 trial ledger (``services.trial_ledger``, a JSONB
blob on the workspace plus a ``system_settings`` daily-spend counter) with the
credit ledger's one-time trial grant (``services.credits.trial``, backed by
``credit_ledger`` / ``credit_balances``). The grant decision stays a pure
function here, as it always was; everything that needs the ledger's SQL (the
daily spend read, the one-per-person check, ``grant_trial`` end to end) needs
real Postgres now and moved to ``test_prd265_credit_ledger_realdb.py`` —
removed from this file, not duplicated.
"""
from __future__ import annotations

import os
from pathlib import Path

from config import config
from services.credits.trial import (
    ALREADY_HELD,
    DAILY_CAP_REACHED,
    DISABLED,
    GRANTED,
    decide_trial_grant,
)


# --------------------------------------------------------------------------- #
# Config keys read via config.py with the stated defaults; no os.getenv in
# the service file.
# --------------------------------------------------------------------------- #


def test_trial_config_defaults():
    # Guard each assertion on its env being unset so a CI override never fails it.
    if not os.getenv("TRIAL_ENABLED"):
        assert config.TRIAL_ENABLED is True
    if not os.getenv("TRIAL_CREDITS"):
        assert config.TRIAL_CREDITS == 500
    if not os.getenv("TRIAL_GLOBAL_DAILY_CREDITS"):
        assert config.TRIAL_GLOBAL_DAILY_CREDITS == 2500
    # All three are the right types — credits are integers now, never a float USD amount.
    assert isinstance(config.TRIAL_ENABLED, bool)
    assert isinstance(config.TRIAL_CREDITS, int)
    assert isinstance(config.TRIAL_GLOBAL_DAILY_CREDITS, int)


def test_credits_trial_uses_no_os_getenv():
    # No os.getenv outside config.py — the service reads config.TRIAL_* only.
    src = (Path(__file__).resolve().parents[1] / "services" / "credits" / "trial.py").read_text()
    assert "os.getenv" not in src
    assert "import os" not in src


# --------------------------------------------------------------------------- #
# Pure grant decision — every branch (decide_trial_grant returns the reason
# string directly now, not a (trial_dict, reason) tuple — there is no USD
# trial dict to build; the ledger writes the credits on GRANTED).
# --------------------------------------------------------------------------- #


def test_decide_grant_happy():
    assert decide_trial_grant(enabled=True, already_held=False, spent_today=0, daily_cap=2500) == GRANTED


def test_decide_grant_disabled():
    assert decide_trial_grant(enabled=False, already_held=False, spent_today=0, daily_cap=2500) == DISABLED


def test_decide_grant_already_held():
    assert decide_trial_grant(enabled=True, already_held=True, spent_today=0, daily_cap=2500) == ALREADY_HELD


def test_decide_grant_daily_cap_reached_at_boundary():
    # >= cap pauses (boundary inclusive).
    assert decide_trial_grant(enabled=True, already_held=False, spent_today=2500, daily_cap=2500) == DAILY_CAP_REACHED


def test_decide_grant_already_held_wins_over_daily_cap():
    # Order of checks (services.credits.trial.decide_trial_grant): a person who
    # already holds their trial is never re-paused by today's platform-wide spend.
    assert decide_trial_grant(enabled=True, already_held=True, spent_today=99999, daily_cap=2500) == ALREADY_HELD


def test_decide_grant_under_cap_is_granted():
    assert decide_trial_grant(enabled=True, already_held=False, spent_today=2499, daily_cap=2500) == GRANTED
