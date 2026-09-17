"""PRD-245 W3 — a session runs a connected app's action through Automatos.

OPS's 2026-09-17 ticket is the case: asked to review the calendar, holding a LIVE
Google Calendar connection, and unable to reach it — the session had no
``composio_execute`` and its skill's every example called one. Now it does, under
exactly the name and calling convention the skills already document, and the
credential stays in the backend: the session sends an action and parameters, and
Automatos makes the call.

What this file pins: the SHAPE the session must send (an action is required, and
top-level parameters are folded in rather than dropped), and the DISPATCH —
``composio_execute`` is a tool name the executor routes itself, not a platform
action to put through the PRD-64 dispatcher, which would fail the registry lookup.
"""
from __future__ import annotations

import asyncio

import pytest

from services import session_tools as st

CTX = st.SessionContext(task_id=118, agent_id=267, agent_name="OPS", workspace_id="ws-c1")


def test_the_tool_keeps_the_name_and_the_calling_convention_the_skills_use():
    tool = st.get_tool("composio_execute")
    assert tool is not None and tool.action == "composio_execute"
    assert set(tool.input_schema["properties"]) == {"action", "params"}
    assert tool.input_schema["required"] == ["action"]
    # a skill body that says "use composio_execute" is now telling the truth
    assert st.equivalent_of("composio_execute") == "composio_execute"
    assert "GOOGLECALENDAR_FIND_EVENT" in tool.description      # the convention, by example
    assert "never see a key" in tool.description


def test_it_is_dispatched_as_a_tool_name_not_as_a_platform_action():
    """The executor routes ``composio_execute`` to the Composio router itself.
    Sending it through ``platform_execute`` would fail the action-registry
    lookup with 'Unknown platform action'."""
    tool = st.get_tool("composio_execute")
    assert tool.dispatch == st.DISPATCH_TOOL_NAME
    for other in st.SESSION_TOOLS:
        if other.name not in ("composio_execute",):
            assert other.dispatch == st.DISPATCH_PLATFORM_ACTION, other.name


def test_the_action_is_required_and_said_plainly_when_missing():
    tool = st.get_tool("composio_execute")
    for refused in ({}, {"action": "  "}, {"params": {"calendar_id": "primary"}}):
        with pytest.raises(st.SessionToolRefused) as raised:
            st.resolve_parameters(tool, refused, CTX)
        assert "GMAIL_FETCH_EMAILS" in str(raised.value)      # the shape, by example


def test_parameters_the_session_put_at_the_top_level_are_folded_in_not_dropped():
    """Every skill body invites this mistake by writing the action and its
    parameters at one level; dropping them would look like a broken tool."""
    tool = st.get_tool("composio_execute")
    nested = st.resolve_parameters(tool, {"action": "GOOGLECALENDAR_FIND_EVENT",
                                          "params": {"calendar_id": "primary"}}, CTX)
    assert nested == {"action": "GOOGLECALENDAR_FIND_EVENT", "params": {"calendar_id": "primary"}}

    flat = st.resolve_parameters(tool, {"action": "GOOGLECALENDAR_FIND_EVENT",
                                        "calendar_id": "primary", "time_min": "2026-09-18T00:00:00Z"}, CTX)
    assert flat == {"action": "GOOGLECALENDAR_FIND_EVENT",
                    "params": {"calendar_id": "primary", "time_min": "2026-09-18T00:00:00Z"}}

    both = st.resolve_parameters(tool, {"action": "GMAIL_FETCH_EMAILS",
                                        "params": {"query": "is:unread"}, "max_results": 10}, CTX)
    assert both == {"action": "GMAIL_FETCH_EMAILS", "params": {"query": "is:unread", "max_results": 10}}
    # what the call nested WINS over the same key at the top level
    clash = st.resolve_parameters(tool, {"action": "A", "params": {"x": "nested"}, "x": "top"}, CTX)
    assert clash["params"]["x"] == "nested"


def test_the_call_reaches_the_executor_under_its_own_tool_name(monkeypatch):
    seen = {}

    class _Executor:
        def __init__(self, db):
            seen["db"] = db

        async def execute_tool(self, **kwargs):
            seen.update(kwargs)
            return {"success": True, "result": {"events": []}}

    import modules.tools.execution.unified_executor as ue
    monkeypatch.setattr(ue, "UnifiedToolExecutor", _Executor)

    tool = st.get_tool("composio_execute")
    params = st.resolve_parameters(tool, {"action": "GOOGLECALENDAR_FIND_EVENT",
                                          "params": {"calendar_id": "primary"}}, CTX)
    out = asyncio.run(st.call_tool(object(), tool, params, CTX))
    assert out["success"] is True
    # the tool NAME, with the action's own parameters — not the platform dispatcher
    assert seen["tool_name"] == "composio_execute"
    assert seen["parameters"] == {"action": "GOOGLECALENDAR_FIND_EVENT", "params": {"calendar_id": "primary"}}
    # …as the ticket's agent, in the ticket's workspace, with no user context
    assert seen["agent_id"] == 267 and seen["workspace_id"] == "ws-c1"
    assert seen["caller_context"] is None
    assert seen["trace_id"] == "session:118:composio_execute"


def test_a_platform_tool_still_goes_through_the_dispatcher(monkeypatch):
    seen = {}

    class _Executor:
        def __init__(self, db):
            pass

        async def execute_tool(self, **kwargs):
            seen.update(kwargs)
            return {"success": True, "result": {}}

    import modules.tools.execution.unified_executor as ue
    monkeypatch.setattr(ue, "UnifiedToolExecutor", _Executor)

    tool = st.get_tool("board_summary")
    asyncio.run(st.call_tool(object(), tool, {}, CTX))
    assert seen["tool_name"] == st.PLATFORM_DISPATCHER
    assert seen["parameters"] == {"action": "platform_board_summary", "params": {}}
