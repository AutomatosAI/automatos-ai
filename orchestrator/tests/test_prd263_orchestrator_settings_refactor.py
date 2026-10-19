"""PRD-263 PR-1E-a: the workspace orchestrator save, split into helpers, does what it did.

``PUT /api/workspaces/current/orchestrator`` was one 230-line function. Its
validation, merge and response view moved into helpers in api/workspaces.py, and its
Auto agent writes into services/orchestrator_auto_agent.py, with no behaviour
change. These tests pin that behaviour on the helpers: the merge is key by key, the
first invalid field wins in the old order, and the Auto agent and the global
``orchestrator_llm`` system settings get the same writes. No DB.

HARNESS is the one part PR-1E-b (US-016) changed: the mode (report_only, propose,
disabled) is the only switch and weekly the only schedule; the retired
``enabled``/``disabled`` keys are neither validated, written nor returned, and the
view reads the mode through services.harness_mode.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import api.workspaces as ws
import services.orchestrator_auto_agent as auto
from core.llm.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER


def _detail(payload):
    with pytest.raises(HTTPException) as caught:
        ws._validate_orchestrator_payload(payload)
    return caught.value.status_code, caught.value.detail


def test_view_merges_stored_fields_over_the_defaults_key_by_key():
    stored = {"orchestrator": {
        "proactive_level": "silent",
        "heartbeat": {"enabled": True},
        "harness": {"enabled": True, "schedule": "monthly", "mode": "manual"},
    }}
    view = ws._orchestrator_view(stored)

    assert list(view) == list(ws._ORCHESTRATOR_DEFAULTS)
    assert view["proactive_level"] == "silent"
    assert view["heartbeat"] == {**ws._ORCHESTRATOR_DEFAULTS["heartbeat"], "enabled": True}
    # PR-1E-b: a stored manual reads report_only, the one schedule, no retired key.
    assert view["harness"] == {"schedule": "weekly", "mode": "report_only"}
    assert ws._orchestrator_view(None) == ws._ORCHESTRATOR_DEFAULTS
    assert ws._ORCHESTRATOR_DEFAULTS["harness"] == {"schedule": "weekly", "mode": "report_only"}


def test_merge_writes_the_six_fields_and_merges_heartbeat_and_harness():
    existing = {"custom_soul": "kept", "heartbeat": {"interval_minutes": 60}, "harness": {"enabled": False}}
    payload = {
        "personality_mode": "technical", "llm": {"model_id": "m"}, "unknown": 1,
        "heartbeat": {"enabled": True}, "harness": {"enabled": True, "disabled": False, "mode": "propose"},
    }
    merged = ws._merge_orchestrator(existing, payload)

    # PR-1E-b: the payload's retired keys are not written; the stored one is not rewritten.
    assert merged == {
        "custom_soul": "kept",
        "heartbeat": {"interval_minutes": 60, "enabled": True},
        "harness": {"enabled": False, "mode": "propose"},
        "personality_mode": "technical",
    }
    assert existing == {"custom_soul": "kept", "heartbeat": {"interval_minutes": 60}, "harness": {"enabled": False}}


def test_validation_accepts_the_modes_that_exist_and_the_weekly_schedule():
    ws._validate_orchestrator_payload({
        "personality_mode": "custom", "preferred_channel": "channel:abc",
        "heartbeat": {"interval_minutes": 30, "active_hours_start": "08:00", "active_hours_end": "23:59"},
        "harness": {"schedule": "weekly", "mode": "propose"},
    })
    for mode in ("report_only", "disabled"):
        ws._validate_orchestrator_payload({"harness": {"mode": mode}})
    # The retired keys are not validated (nor written, see the merge test).
    ws._validate_orchestrator_payload({"harness": {"enabled": "anything", "disabled": 7}})


def test_validation_rejects_with_the_same_status_and_message():
    assert _detail({"harness": {"mode": "full_auto"}}) == (
        400, "harness.mode must be one of ['report_only', 'propose', 'disabled']")
    assert _detail({"harness": {"schedule": "biweekly"}}) == (400, "harness.schedule must be one of ['weekly']")
    assert _detail({"preferred_channel": 7}) == (400, "preferred_channel must be a string")
    assert _detail({"heartbeat": {"active_hours_end": "24:00"}}) == (
        400, "heartbeat.active_hours_end must be HH:MM format")
    status, detail = _detail({"thinking_level": "max"})
    assert (status, detail.split(" must be one of ")[0]) == (400, "thinking_level")


def test_the_first_invalid_field_wins_in_the_old_order():
    _, detail = _detail({"harness": {"mode": "x"}, "heartbeat": {"interval_minutes": 7}, "personality_mode": "x"})
    assert detail.startswith("personality_mode must be one of")
    _, detail = _detail({"harness": {"mode": "x"}, "heartbeat": {"interval_minutes": 7}})
    assert detail.startswith("heartbeat.interval_minutes must be one of")


def test_auto_llm_view_fills_the_same_defaults():
    assert auto.auto_llm_view({}) == {
        "provider": DEFAULT_LLM_PROVIDER, "model_id": DEFAULT_LLM_MODEL, "temperature": 0.7, "max_tokens": 4000,
        "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "stop": None, "timeout": None,
        "fallback_model_id": None,
    }


@pytest.fixture
def flagged(monkeypatch):
    calls = []
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda obj, key: calls.append(key))
    return calls


def test_sync_auto_llm_writes_model_config_and_the_global_orchestrator_llm_settings(flagged):
    agent = SimpleNamespace(model_config={"provider": "old", "extra": 1})
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    auto.sync_auto_llm(db, agent, {"provider": "anthropic", "model_id": "claude", "bogus": 2})

    assert agent.model_config == {"provider": "anthropic", "extra": 1, "model_id": "claude"}
    added = [call.args[0] for call in db.add.call_args_list]
    assert [(row.category, row.key, row.value, row.default_value) for row in added] == [
        ("orchestrator_llm", "provider", "anthropic", "anthropic"),
        ("orchestrator_llm", "model", "claude", "claude"),
    ]
    assert flagged == ["model_config"]


def test_sync_auto_llm_updates_an_existing_setting_and_skips_no_llm(flagged):
    row = SimpleNamespace(value="old")
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = row
    auto.sync_auto_llm(db, SimpleNamespace(model_config=None), {"model_id": "m"})
    assert row.value == "m"
    db.add.assert_not_called()

    untouched = SimpleNamespace(model_config={"a": 1})
    auto.sync_auto_llm(MagicMock(), untouched, None)
    assert untouched.model_config == {"a": 1}
    assert flagged == ["model_config"]


def test_sync_auto_configuration_writes_personality_mode_first(flagged):
    agent = SimpleNamespace(configuration={"thinking_level": "low", "other": True})
    auto.sync_auto_configuration(agent, {"thinking_level": "high", "personality_mode": "custom", "custom_soul": "x"})

    assert agent.configuration == {"thinking_level": "high", "other": True, "personality_mode": "custom"}
    fresh = SimpleNamespace(configuration=None)
    auto.sync_auto_configuration(fresh, {"communication_style": "concise", "personality_mode": "friendly"})
    assert list(fresh.configuration) == ["personality_mode", "communication_style"]

    unchanged = SimpleNamespace(configuration={"a": 1})
    auto.sync_auto_configuration(unchanged, {"personality_mode": "", "custom_soul": "x"})
    assert unchanged.configuration == {"a": 1}
    assert flagged == ["configuration", "configuration"]
