"""PRD-230 US-001 (W0, fix/prd-222) — chat trial metering.

The trial gate previously lived ONLY in ``AgentFactory._create_llm_manager`` (the
mission factory path). The chatbot activates Auto through ``activate_agent``, which
builds the ``LLMManager`` directly and never called the gate — so CHAT, the primary
surface, ran unmetered (no ``record_trial_spend``), unpinned (any model), and could
never exhaust. Worse, Auto the system agent carries no ``workspace_id`` of its own,
so even the BYOK lookup no-oped.

This wave threads the CONVERSATION's workspace into ``activate_agent`` and shares ONE
gate (``_resolve_trial_decision``) across both surfaces. These are PURE tests: the
decision helper is exercised through a real ``AgentFactory`` over a fake session
(``resolve_trial_routing`` is pure), and the activation→accrual wiring is locked by
source grep — the established PRD-222 style (``test_prd222_trial_enforcement``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config import config
from services.trial_ledger import (
    TRIAL_ACTIVE,
    TRIAL_CONVERTED,
    TRIAL_EXHAUSTED,
    TRIAL_WARNED,
    TrialExhaustedError,
)

REPO = Path(__file__).resolve().parents[1]
WS_ID = "11111111-1111-1111-1111-111111111111"


# --------------------------------------------------------------------------- #
# Fakes — a Workspace row carrying an onboarding JSONB doc + a session that
# returns it from ``.query(Workspace).get(id)`` (all the gate reads).
# --------------------------------------------------------------------------- #


class _WS:
    def __init__(self, trial=None, stage="questions"):
        self.onboarding = {"stage": stage, "stages": {}, "segment": {}}
        if trial is not None:
            self.onboarding["trial"] = trial


def _trial(state, *, granted=5.0, spent=0.0):
    return {"granted_usd": granted, "spent_usd": spent, "state": state}


class _FakeQuery:
    def __init__(self, ws):
        self._ws = ws

    def get(self, _id):
        return self._ws


class _FakeSession:
    def __init__(self, ws):
        self._ws = ws

    def query(self, _model):
        return _FakeQuery(self._ws)


def _factory(ws):
    from modules.agents.factory.agent_factory import AgentFactory

    return AgentFactory(db_session=_FakeSession(ws))


# --------------------------------------------------------------------------- #
# The shared gate on the chat path — active trial meters + pins (AC2)
# --------------------------------------------------------------------------- #


def test_active_trial_chat_turn_meters_and_pins_offlist_model(monkeypatch):
    # An active-trial workspace: the request is trial-routed (so record_trial_spend
    # fires at the usage seam) and an off-allowlist model is substituted.
    monkeypatch.setattr(config, "TRIAL_MODEL_ALLOWLIST", "trial-cheap-a,trial-cheap-b")
    f = _factory(_WS(_trial(TRIAL_ACTIVE)))
    trial_routed, pinned = f._resolve_trial_decision(WS_ID, "expensive-gpt", is_byok=False)
    assert trial_routed is True
    assert pinned == "trial-cheap-a"  # off-list → pinned to TRIAL_MODEL_ALLOWLIST[0]


def test_active_trial_allowlisted_model_is_not_repinned(monkeypatch):
    monkeypatch.setattr(config, "TRIAL_MODEL_ALLOWLIST", "trial-cheap-a,trial-cheap-b")
    f = _factory(_WS(_trial(TRIAL_ACTIVE)))
    trial_routed, pinned = f._resolve_trial_decision(WS_ID, "trial-cheap-a", is_byok=False)
    assert trial_routed is True  # still metered
    assert pinned is None        # already allowlisted → no substitution


def test_warned_trial_still_meters(monkeypatch):
    monkeypatch.setattr(config, "TRIAL_MODEL_ALLOWLIST", "trial-cheap-a")
    f = _factory(_WS(_trial(TRIAL_WARNED)))
    trial_routed, pinned = f._resolve_trial_decision(WS_ID, "expensive-gpt", is_byok=False)
    assert trial_routed is True and pinned == "trial-cheap-a"


# --------------------------------------------------------------------------- #
# BYOK bypasses untouched (AC2) — no db read, no metering, no pin
# --------------------------------------------------------------------------- #


def test_byok_chat_turn_bypasses_trial_untouched():
    # BYOK short-circuits BEFORE any workspace read (a Boom session proves it).
    class _Boom:
        def query(self, _m):
            raise AssertionError("BYOK must not read the workspace")

    from modules.agents.factory.agent_factory import AgentFactory

    f = AgentFactory(db_session=_Boom())
    assert f._resolve_trial_decision(WS_ID, "any-model", is_byok=True) == (False, None)


# --------------------------------------------------------------------------- #
# Exhausted trial → the typed error on the chat path (AC3)
# --------------------------------------------------------------------------- #


def test_exhausted_trial_raises_typed_error_on_chat_path():
    f = _factory(_WS(_trial(TRIAL_EXHAUSTED)))
    with pytest.raises(TrialExhaustedError) as ei:
        f._resolve_trial_decision(WS_ID, "any-model", is_byok=False)
    assert ei.value.error_code == "trial_exhausted"


# --------------------------------------------------------------------------- #
# No-op set — non-trial / converted / system calls run exactly as before
# --------------------------------------------------------------------------- #


def test_no_trial_workspace_passes_through():
    assert _factory(_WS(None))._resolve_trial_decision(WS_ID, "m", is_byok=False) == (False, None)


def test_converted_trial_passes_through():
    f = _factory(_WS(_trial(TRIAL_CONVERTED)))
    assert f._resolve_trial_decision(WS_ID, "m", is_byok=False) == (False, None)


def test_system_call_without_workspace_is_noop():
    # workspace_id None (a system/non-chat call) → no read, no metering.
    assert _factory(None)._resolve_trial_decision(None, "m", is_byok=False) == (False, None)


# --------------------------------------------------------------------------- #
# Wiring — the chat activation path is gated and the manager is tagged (AC1/AC2)
# --------------------------------------------------------------------------- #


def test_activate_agent_applies_gate_and_tags_manager_for_accrual():
    src = (REPO / "modules" / "agents" / "factory" / "agent_factory.py").read_text()
    assert "_resolve_trial_decision" in src        # the chat path runs the gate
    assert "trial=trial_routed" in src             # LLMManager tagged → accrual fires
    assert "workspace_id=effective_ws_id" in src   # conversation workspace threaded
    assert "except TrialExhaustedError" in src     # typed error not swallowed to None


def test_chatbot_threads_conversation_workspace_into_activation():
    src = (REPO / "consumers" / "chatbot" / "service.py").read_text()
    assert "activate_agent(" in src
    assert "workspace_id=self.workspace_id" in src  # grep-proof call site (AC1)


def test_trial_flag_drives_spend_accrual_at_the_usage_seam():
    # Closes the loop: the flag activate_agent sets is exactly what manager.py
    # gates record_trial_spend on — so a metered chat turn accrues.
    mgr = (REPO / "core" / "llm" / "manager.py").read_text()
    assert 'self._tracking_ctx.get("trial")' in mgr
    assert "record_trial_spend" in mgr


def test_gate_is_shared_not_duplicated():
    # _create_llm_manager (mission path) and activate_agent (chat path) must call
    # the SAME helper — no parallel trial mechanism to drift.
    src = (REPO / "modules" / "agents" / "factory" / "agent_factory.py").read_text()
    assert src.count("def _resolve_trial_decision") == 1
    assert src.count("_resolve_trial_decision(") >= 2  # defined once, called by both
