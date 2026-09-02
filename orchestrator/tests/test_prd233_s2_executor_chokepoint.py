"""PRD-233 S2 — ONE Composio availability seam for every caller.

The router refuses routed calls (test_prd233_s2_composio_degrade); this covers
the executor chokepoint that catches the two production callers which bypass
the router (agent_factory direct execution, approval-grant replays).
"""
from __future__ import annotations

import pytest

from modules.tools.execution.unified_executor import UnifiedToolExecutor


def _bare_executor() -> UnifiedToolExecutor:
    # No constructor: the seam sits before every sub-executor is touched.
    ex = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    ex.composio_actions = {"SLACK_SEND_MESSAGE": {"app": "slack"}}
    ex.db = None
    return ex


@pytest.fixture
def no_policy_gate(monkeypatch):
    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)


@pytest.mark.asyncio
async def test_direct_composio_call_is_refused_when_unavailable(monkeypatch, no_policy_gate):
    monkeypatch.setattr("core.composio.client.composio_available", lambda: False)
    ex = _bare_executor()
    for name, params in (("composio_execute", {"action": "SLACK_SEND_MESSAGE"}), ("composio_slack_send", {}), ("SLACK_SEND_MESSAGE", {})):
        res = await ex.execute_tool(name, params, agent_id=1, trace_id="t-chokepoint")
        assert res["success"] is False, name
        assert res.get("error_code") == "integrations_unavailable", (name, res)


@pytest.mark.asyncio
async def test_native_tools_pass_the_seam(monkeypatch, no_policy_gate):
    monkeypatch.setattr("core.composio.client.composio_available", lambda: False)
    ex = _bare_executor()
    # platform_execute without an action reaches the dispatcher's own validation —
    # proof the seam let a native tool through.
    res = await ex.execute_tool("platform_execute", {}, agent_id=1, trace_id="t-native")
    assert res.get("error_code") != "integrations_unavailable"
    assert "action" in (res.get("error") or "")


@pytest.mark.asyncio
async def test_with_a_key_the_seam_is_a_no_op(monkeypatch, no_policy_gate):
    monkeypatch.setattr("core.composio.client.composio_available", lambda: True)
    ex = _bare_executor()
    res = await ex.execute_tool("composio_execute", {"action": "SLACK_SEND_MESSAGE"}, agent_id=1, trace_id="t-key")
    # Past the seam the bare executor has no sub-executors — whatever it returns,
    # it is NOT the unavailable refusal.
    assert res.get("error_code") != "integrations_unavailable"
