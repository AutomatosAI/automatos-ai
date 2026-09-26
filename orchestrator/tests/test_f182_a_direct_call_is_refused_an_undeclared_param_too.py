"""F182 (night 6) — a direct platform_* call is held to platform_execute's rule.

5245efa18 made platform_execute refuse a key its action does not take. A model
also calls an action directly, as its own tool, and there nothing checked the
keys: at 02:13:59 Auto moved ticket #1093 back to the inbox with a "reason",
which update_task_status does not take (it takes blocked_reason), and the
reason was dropped without a word. A direct call now gets the same alias fill
and the same refusal, logged the same way.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
# Night 6, 02:13:59 (chats.jsonl 24).
REASON = {"status": "inbox", "reason": "Re-drafting emails due to brand voice and data access issues.",
          "task_id": 1093}


@pytest.fixture
def direct(monkeypatch):
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran = []

    async def _run(self, action_name, params, **kwargs):
        ran.append((action_name, params))
        return {"success": True}

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_platform_action", _run)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    executor = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    executor.composio_actions, executor.db = {}, None

    def call(action, params):
        return asyncio.run(executor.execute_tool(action, params, agent_id=322, workspace_id=WS, trace_id="t-f182"))
    return SimpleNamespace(call=call, ran=ran)


def test_the_reason_night_6_sent_is_refused_not_dropped(direct, caplog):
    with caplog.at_level(logging.INFO, logger="modules.tools.execution.unified_executor"):
        result = direct.call("platform_update_task_status", REASON)

    assert result["success"] is False and direct.ran == []
    assert result["error"].startswith("'platform_update_task_status' does not take ['reason'], so nothing was done.")
    assert "blocked_reason (string)" in result["error"]
    assert [r.getMessage() for r in caplog.records if r.getMessage().startswith("[F182]")] == [
        "[F182] direct call refused param 'reason' for platform_update_task_status (trace t-f182)"]


def test_the_corrected_call_is_shown_the_way_it_was_made(direct):
    result = direct.call("platform_get_playbook", {"playbook_idd": 102})

    assert 'Call it like this: platform_get_playbook({"playbook_id": 102})' in result["error"]


def test_a_direct_call_takes_a_required_param_under_its_other_name(direct):
    """F027-C's aliases reach a direct call too; before, its handler said the prompt was missing."""
    result = direct.call("platform_add_playbook_step", {"playbook_id": 102, "prompt": "Draft the welcome email"})

    assert result == {"success": True}
    ((action, params),) = direct.ran
    assert action == "platform_add_playbook_step" and params["prompt_template"] == "Draft the welcome email"


def test_night_6s_direct_create_playbook_still_runs(direct):
    params = {"tags": ["onboarding"], "name": "New Cafe Onboarding", "description": "Welcome a new café."}
    assert direct.call("platform_create_playbook", params) == {"success": True}
    assert direct.ran == [("platform_create_playbook", params)]
