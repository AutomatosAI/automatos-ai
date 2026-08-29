"""PRD-222 W1·S9 (US-005) — trial enforcement at the LLM key-resolution seam.

Pure tests: no Postgres, no live LLM. The routing gate is a pure function; the
accrual runs against a fake session with the pricing call monkeypatched (the real
``modules.policy.pricing`` path is CI's job). The choke-point wiring is asserted
by source grep — the BYOK bypass is `not resolved.is_byok` at the seam.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import config
from services import trial_ledger
from services.trial_ledger import (
    ACTION_BLOCKED,
    ACTION_BYOK,
    ACTION_PASSTHROUGH,
    ACTION_PLATFORM_TRIAL,
    TRIAL_ACTIVE,
    TRIAL_CONVERTED,
    TRIAL_EXHAUSTED,
    TRIAL_EXHAUSTED_CODE,
    TRIAL_WARNED,
    TrialExhaustedError,
    compute_trial_state,
    decide_trial_grant,
    is_trial_active_workspace,
    record_trial_spend,
    resolve_trial_routing,
)

REPO = Path(__file__).resolve().parents[1]


class _WS:
    """Duck-typed Workspace row carrying an onboarding JSONB doc."""

    def __init__(self, trial=None, stage="questions"):
        self.onboarding = {"stage": stage, "stages": {}, "segment": {}}
        if trial is not None:
            self.onboarding["trial"] = trial


def _trial(state, *, granted=5.0, spent=0.0):
    return {"granted_usd": granted, "spent_usd": spent, "state": state}


# --------------------------------------------------------------------------- #
# resolve_trial_routing — the pure gate (AC2 + AC3 model pinning)
# --------------------------------------------------------------------------- #


def test_byok_bypasses_trial_routing():
    # Even WITH an active trial, a BYOK-resolved call is never trial-routed.
    r = resolve_trial_routing(_WS(_trial(TRIAL_ACTIVE)), "any-model", is_byok=True)
    assert r.action == ACTION_BYOK


def test_no_trial_passes_through():
    assert resolve_trial_routing(_WS(None), "m", is_byok=False).action == ACTION_PASSTHROUGH


def test_converted_trial_passes_through():
    r = resolve_trial_routing(_WS(_trial(TRIAL_CONVERTED)), "m", is_byok=False)
    assert r.action == ACTION_PASSTHROUGH


def test_active_trial_routes_to_platform(monkeypatch):
    monkeypatch.setattr(config, "TRIAL_MODEL_ALLOWLIST", "modelA,modelB")
    r = resolve_trial_routing(_WS(_trial(TRIAL_ACTIVE)), "modelA", is_byok=False)
    assert r.action == ACTION_PLATFORM_TRIAL
    assert r.model == "modelA"  # allowlisted request kept as-is


def test_offlist_model_is_substituted(monkeypatch):
    monkeypatch.setattr(config, "TRIAL_MODEL_ALLOWLIST", "modelA,modelB")
    r = resolve_trial_routing(_WS(_trial(TRIAL_WARNED)), "expensive-gpt", is_byok=False)
    assert r.action == ACTION_PLATFORM_TRIAL
    assert r.model == "modelA"  # off-list → substituted to the first allowlisted


def test_exhausted_trial_is_blocked_with_typed_code():
    r = resolve_trial_routing(_WS(_trial(TRIAL_EXHAUSTED)), "m", is_byok=False)
    assert r.action == ACTION_BLOCKED
    assert r.error_code == TRIAL_EXHAUSTED_CODE == "trial_exhausted"


def test_trial_exhausted_error_carries_stable_code():
    err = TrialExhaustedError()
    assert err.error_code == "trial_exhausted"
    assert str(err)  # human message present


# --------------------------------------------------------------------------- #
# compute_trial_state — the 80% / 100% thresholds (AC3)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spent,granted,expected",
    [
        (0.0, 5.0, TRIAL_ACTIVE),
        (3.99, 5.0, TRIAL_ACTIVE),   # 79.8% — still active
        (4.0, 5.0, TRIAL_WARNED),    # exactly 80% — warned
        (4.5, 5.0, TRIAL_WARNED),
        (5.0, 5.0, TRIAL_EXHAUSTED), # exactly 100% — exhausted
        (6.0, 5.0, TRIAL_EXHAUSTED),
        (0.0, 0.0, TRIAL_EXHAUSTED), # nothing to spend
    ],
)
def test_compute_trial_state_thresholds(spent, granted, expected):
    assert compute_trial_state(spent, granted) == expected


# --------------------------------------------------------------------------- #
# is_trial_active_workspace — the "no background burn" set (AC5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("state", [TRIAL_ACTIVE, TRIAL_WARNED, TRIAL_EXHAUSTED])
def test_on_trial_states_are_skipped(state):
    assert is_trial_active_workspace(_WS(_trial(state))) is True


@pytest.mark.parametrize("ws", [_WS(None), _WS(_trial(TRIAL_CONVERTED))])
def test_converted_and_untried_workspaces_run(ws):
    assert is_trial_active_workspace(ws) is False


def test_none_workspace_is_not_a_trial():
    assert is_trial_active_workspace(None) is False


# --------------------------------------------------------------------------- #
# record_trial_spend — accrual, state flip, daily counter (AC3 + AC4)
# --------------------------------------------------------------------------- #


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeQuery:
    def __init__(self, ws):
        self._ws = ws

    def get(self, _id):
        return self._ws


class _FakeSpendDB:
    def __init__(self, ws, *, daily_raw=None):
        self._ws = ws
        self._daily_raw = daily_raw
        self.trial_writes = []   # trial dicts passed to _write_trial
        self.daily_writes = []   # new daily totals written
        self.commits = 0

    def query(self, _model):
        return _FakeQuery(self._ws)

    def execute(self, clause, params=None):
        s = str(clause).strip().upper()
        if s.startswith("UPDATE") and "WORKSPACES" in s:
            self.trial_writes.append(json.loads(params["trial"]))
            return _FakeResult(None)
        if "VALUE FROM SYSTEM_SETTINGS" in s:          # get_daily_trial_spend read
            return _FakeResult((self._daily_raw,) if self._daily_raw is not None else None)
        if "ID FROM SYSTEM_SETTINGS" in s:             # _increment row-exists probe
            return _FakeResult(None)                   # no row → INSERT path
        if "SYSTEM_SETTINGS" in s and s.startswith(("INSERT", "UPDATE")):
            self.daily_writes.append(params.get("v"))
            return _FakeResult(None)
        return _FakeResult(None)

    def commit(self):
        self.commits += 1


def test_accrual_flips_active_to_warned(monkeypatch):
    monkeypatch.setattr(trial_ledger, "_price_request", lambda *a, **k: 4.0)
    ws = _WS(_trial(TRIAL_ACTIVE, granted=5.0, spent=0.0))
    db = _FakeSpendDB(ws, daily_raw="0")

    state = record_trial_spend("ws-1", model_id="m",
                               input_tokens=10, output_tokens=10, db=db)

    assert state == TRIAL_WARNED
    assert db.trial_writes[0]["spent_usd"] == 4.0
    assert db.trial_writes[0]["state"] == TRIAL_WARNED
    assert db.daily_writes == ["4.0"]     # daily counter incremented (AC4)
    assert db.commits == 1


def test_accrual_flips_warned_to_exhausted(monkeypatch):
    monkeypatch.setattr(trial_ledger, "_price_request", lambda *a, **k: 2.0)
    ws = _WS(_trial(TRIAL_WARNED, granted=5.0, spent=4.0))
    db = _FakeSpendDB(ws, daily_raw="10")

    state = record_trial_spend("ws-2", model_id="m", input_tokens=1, output_tokens=1, db=db)

    assert state == TRIAL_EXHAUSTED
    assert db.trial_writes[0]["spent_usd"] == 6.0
    assert db.daily_writes == ["12.0"]    # 10 + 2
    assert db.commits == 1


def test_no_accrual_for_converted_workspace(monkeypatch):
    monkeypatch.setattr(trial_ledger, "_price_request", lambda *a, **k: 3.0)
    ws = _WS(_trial(TRIAL_CONVERTED))
    db = _FakeSpendDB(ws)
    assert record_trial_spend("ws-3", model_id="m", input_tokens=1, output_tokens=1, db=db) is None
    assert db.trial_writes == [] and db.commits == 0


def test_zero_cost_does_not_accrue(monkeypatch):
    monkeypatch.setattr(trial_ledger, "_price_request", lambda *a, **k: 0.0)
    ws = _WS(_trial(TRIAL_ACTIVE))
    db = _FakeSpendDB(ws)
    assert record_trial_spend("ws-4", model_id="m", input_tokens=0, output_tokens=0, db=db) is None
    assert db.trial_writes == []


def test_accrual_never_raises_on_bad_db():
    # A broken session must never break the LLM response path.
    class _Boom:
        def query(self, _m):
            raise RuntimeError("db down")

    assert record_trial_spend("ws-5", model_id="m", input_tokens=1, output_tokens=1, db=_Boom()) is None


# --------------------------------------------------------------------------- #
# Daily cap pause ties back to US-004's grant (AC4)
# --------------------------------------------------------------------------- #


def test_daily_increment_pushes_grant_into_pause():
    # After accrual pushes the day's spend to/over the cap, a fresh grant pauses.
    daily_after = 24.0 + 2.0  # a $2 request on a day already at $24
    trial, reason = decide_trial_grant(
        enabled=True, already_held=False, daily_spend=daily_after, daily_cap=25.0, credit_usd=5.0
    )
    assert trial is None and reason == "daily_cap_reached"


# --------------------------------------------------------------------------- #
# Choke-point wiring — BYOK bypass + gate presence asserted by source grep (AC2)
# --------------------------------------------------------------------------- #


def test_gate_wired_at_choke_point_and_bypasses_byok():
    # PRD-230 US-001 refactor: the gate is now a shared helper
    # (_resolve_trial_decision) called by BOTH the mission path
    # (_create_llm_manager) and the chat path (activate_agent). BYOK still
    # provably bypasses — the helper short-circuits on the is_byok flag, which
    # each seam passes as resolved.is_byok.
    src = (REPO / "modules" / "agents" / "factory" / "agent_factory.py").read_text()
    assert "resolve_trial_routing" in src              # gate logic at the seam
    assert "_resolve_trial_decision(" in src           # shared gate invoked
    assert "if not workspace_id or is_byok" in src     # BYOK provably bypasses it
    assert "resolved.is_byok" in src                   # the BYOK flag flows into the gate
    assert "TrialExhaustedError" in src                # exhausted → typed error


def test_accrual_wired_at_usage_seam():
    src = (REPO / "core" / "llm" / "manager.py").read_text()
    assert "record_trial_spend" in src                 # accrual at the usage seam
    assert 'self._tracking_ctx.get("trial")' in src    # gated to trial calls only


def test_no_background_burn_guards_present():
    for rel in [
        "services/heartbeat_service.py",
        "services/scheduled_task_service.py",
        "services/playbook_scheduler.py",
    ]:
        src = (REPO / rel).read_text()
        assert "is_trial_active_workspace" in src, f"missing trial skip guard in {rel}"
