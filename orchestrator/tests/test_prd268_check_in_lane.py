"""PRD-268 US-C02b (§6.2.3, FR-1, K8) — a check-in can only read.

A check-in's calls carry ``caller_context["lane"] = "check_in"``. In that lane the
platform executor's gates (``PlatformActionExecutor.clear``) refuse every call that is
not positively ``read``, an unknown action included, before any other gate runs; only
``platform_suggest_next_step`` is let through besides. The unified executor refuses any
tool that is not a platform action (Composio, a plugin, a workspace tool, the
``platform_execute`` dispatcher) before anything routes it. With no lane, or any other
lane, both executors run exactly as before.

Pure: no database. The platform tests read a stub registry; the unified tests stub the
routes behind the lane, as test_f182_a does.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from modules.tools.discovery.check_in_lane import CHECK_IN_REFUSAL

WS = "5b0c7a1e-2f43-4a8e-9c61-0d7e2b9f4a10"
IN_THE_LANE = {"lane": "check_in"}
REFUSED = {"success": False, "error": CHECK_IN_REFUSAL}

READ = "platform_lane_test_read"
WRITE = "platform_lane_test_write"
DESTROY = "platform_lane_test_destroy"
SUPER_WRITE = "platform_lane_test_super_write"
SUGGEST = "platform_suggest_next_step"
UNKNOWN = "platform_lane_test_unknown"
PLAYBOOK = {"tags": ["onboarding"], "name": "New Cafe Onboarding", "description": "Welcome a new café."}


def test_the_refusal_says_what_a_check_in_can_do():
    assert CHECK_IN_REFUSAL == "Not available in a check-in, which can only read. Suggest it as a next step instead."


# ── PlatformActionExecutor.clear ────────────────────────────────────────────

def _action(name, level, **flags):
    from modules.tools.discovery.action_registry import ActionDefinition

    return ActionDefinition(name=name, description=f"{name} (lane test)", category="test",
                            parameters={"type": "object", "properties": {}}, permission_level=level, **flags)


class _Registry:
    def __init__(self, actions):
        self._actions = {action.name: action for action in actions}

    def get(self, name):
        return self._actions.get(name)


@pytest.fixture
def registry(monkeypatch):
    import modules.tools.discovery as discovery

    stub = _Registry([
        _action(READ, "read"),
        _action(WRITE, "write"),
        _action(DESTROY, "destructive", requires_confirmation=True),
        _action(SUPER_WRITE, "write", super_admin_only=True),
        _action(SUGGEST, "write"),
    ])
    monkeypatch.setattr(discovery, "get_action_registry", lambda: stub)
    return stub


def _executor():
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(None, uuid4())
    executor._full_autonomy = lambda: False
    return executor


@pytest.mark.parametrize("name", [WRITE, DESTROY, UNKNOWN])
def test_in_the_lane_a_write_a_destructive_and_an_unknown_action_are_refused(registry, name):
    assert _executor().clear(name, {"id": 7}, IN_THE_LANE) == REFUSED


@pytest.mark.parametrize("name", [READ, SUGGEST])
def test_in_the_lane_a_read_and_platform_suggest_next_step_clear(registry, name):
    from modules.tools.discovery.platform_executor import Cleared

    cleared = _executor().clear(name, {}, IN_THE_LANE)
    assert isinstance(cleared, Cleared) and cleared.action_def.name == name


def test_the_lane_refuses_before_any_other_gate(registry, monkeypatch):
    """A destructive call never reaches the confirmation gate (no grant is consumed and
    no card raised), and a super-admin-only write gets the lane's refusal, not the tier's."""
    from modules.tools.execution import tool_grants

    asked = []
    monkeypatch.setattr(tool_grants, "consume_tool_grant", lambda *a, **k: asked.append("grant"))
    monkeypatch.setattr(tool_grants, "attach_ask_grant", lambda *a, **k: asked.append("card"))
    executor = _executor()

    assert executor.clear(DESTROY, {"id": 7}, IN_THE_LANE) == REFUSED
    assert executor.clear(SUPER_WRITE, {}, IN_THE_LANE) == REFUSED
    assert asked == []


def test_in_the_lane_a_registry_that_cannot_be_read_refuses(monkeypatch):
    import modules.tools.discovery as discovery

    def _down():
        raise RuntimeError("registry down")

    monkeypatch.setattr(discovery, "get_action_registry", _down)
    assert _executor().clear(READ, {}, IN_THE_LANE) == REFUSED


def test_in_the_lane_execute_never_runs_a_writes_handler(registry):
    executor = _executor()
    handler = AsyncMock(return_value={"success": True, "handler": "ran"})
    executor._handlers = {**executor._handlers, WRITE: handler, UNKNOWN: handler}

    assert asyncio.run(executor.execute(WRITE, {}, IN_THE_LANE)) == REFUSED
    assert asyncio.run(executor.execute(UNKNOWN, {}, IN_THE_LANE)) == REFUSED
    handler.assert_not_called()


@pytest.mark.parametrize("context", [None, {}, {"lane": "heartbeat"}])
@pytest.mark.parametrize("name", [WRITE, READ])
def test_outside_the_lane_a_write_and_a_read_clear_as_before(registry, name, context):
    from modules.tools.discovery.platform_executor import Cleared, PlatformActionExecutor

    executor = _executor()
    cleared = executor.clear(name, {}, context)
    before = PlatformActionExecutor.clear.__wrapped__(executor, name, {}, context)

    assert isinstance(cleared, Cleared) and cleared.action_def.name == name
    assert cleared == before


def test_outside_the_lane_the_other_gates_answer_as_before(registry):
    reply = _executor().clear(SUPER_WRITE, {}, {"lane": "heartbeat"})
    assert reply["permission_denied"] is True and reply["error"] != CHECK_IN_REFUSAL


def test_the_lane_reads_the_registered_level():
    from modules.tools.discovery.check_in_lane import may_clear_in_a_check_in

    assert may_clear_in_a_check_in("platform_list_agents") is True
    assert may_clear_in_a_check_in("platform_create_task") is False
    assert may_clear_in_a_check_in("platform_field_inject") is False
    assert may_clear_in_a_check_in("platform_delete_agent") is False
    assert may_clear_in_a_check_in("platform_no_such_action") is False
    assert may_clear_in_a_check_in(SUGGEST) is True


# ── UnifiedToolExecutor.execute_tool ────────────────────────────────────────

@pytest.fixture
def tools(monkeypatch):
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran, gated = [], []

    async def _run(self, action_name, params, **kwargs):
        ran.append((action_name, kwargs.get("caller_context")))
        return {"success": True}

    def _gate(self, tool_name, *args, **kwargs):
        gated.append(tool_name)

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", _gate)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_platform_action", _run)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    executor = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    executor.composio_actions, executor.db = {"GMAIL_SEND_EMAIL": "gmail"}, None

    def call(tool, params, context):
        return asyncio.run(executor.execute_tool(tool, params, agent_id=7, workspace_id=WS, trace_id="t-c02b",
                                                 caller_context=context))
    return NS(call=call, ran=ran, gated=gated, executor=executor)


@pytest.mark.parametrize("tool,params", [
    ("GMAIL_SEND_EMAIL", {"recipient_email": "owner@harbourline.test", "body": "hello"}),
    ("composio_execute", {"action": "GMAIL_SEND_EMAIL", "params": {}}),
    ("workspace_write_file", {"path": "notes.md", "content": "x"}),
    ("platform_execute", {"action": "platform_list_agents", "params": {}}),
    ("search_codebase", {"query": "lane"}),
    ("search_documents", {"query": "price"}),
])
def test_in_the_lane_a_tool_that_is_not_a_platform_action_is_refused(tools, tool, params):
    assert tools.call(tool, params, IN_THE_LANE) == {**REFUSED, "tool": tool}
    assert tools.ran == [] and tools.gated == []


def test_in_the_lane_a_composio_call_is_refused_when_its_context_comes_by_position(tools):
    result = asyncio.run(tools.executor.execute_tool("GMAIL_SEND_EMAIL", {}, 7, None, WS, "t-c02b", IN_THE_LANE))
    assert result == {**REFUSED, "tool": "GMAIL_SEND_EMAIL"}


def test_in_the_lane_a_platform_action_goes_on_with_its_lane(tools):
    """The platform executor's own lane check (above) is what refuses a write here."""
    assert tools.call("platform_list_agents", {}, IN_THE_LANE) == {"success": True}
    assert tools.ran == [("platform_list_agents", IN_THE_LANE)]


@pytest.mark.parametrize("context", [None, {"lane": "heartbeat"}])
@pytest.mark.parametrize("tool,params", [("platform_list_agents", {}), ("platform_create_playbook", PLAYBOOK)])
def test_outside_the_lane_a_read_and_a_write_run_as_before(tools, tool, params, context):
    assert tools.call(tool, params, context) == {"success": True}
    assert tools.ran == [(tool, context)] and tools.gated == [tool]
