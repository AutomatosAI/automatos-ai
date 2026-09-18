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
import json

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


# ---------------------------------------------------------------------------
# The two clauses of this story's test plan that had no test — and the fake that
# was hiding a signature drift.
# ---------------------------------------------------------------------------

def _record(monkeypatch):
    """A fake executor that records the call and answers whatever it is told to."""
    seen: dict = {}

    def _make(answer):
        class _Executor:
            def __init__(self, db):
                pass

            async def execute_tool(self, **kwargs):
                seen.update(kwargs)
                return answer

        import modules.tools.execution.unified_executor as ue
        monkeypatch.setattr(ue, "UnifiedToolExecutor", _Executor)
        return seen

    return _make


def test_the_recorded_call_binds_to_the_REAL_executor_signature(monkeypatch):
    """The fakes in this file take ``**kwargs``, so they accept a call the real
    executor would reject — a renamed or newly-required parameter passes every
    test here and raises TypeError in production. Bind the recorded call against
    the real signature so drift fails in CI instead.
    """
    import inspect

    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    seen = _record(monkeypatch)({"success": True, "result": {}})
    tool = st.get_tool("composio_execute")
    params = st.resolve_parameters(tool, {"action": "GMAIL_FETCH_EMAILS"}, CTX)
    asyncio.run(st.call_tool(object(), tool, params, CTX))

    signature = inspect.signature(UnifiedToolExecutor.execute_tool)
    signature.bind(UnifiedToolExecutor, **seen)          # raises if a name moved


def test_an_action_for_an_unconnected_app_is_refused_and_said_plainly(monkeypatch):
    """OPS's case if the calendar were not linked. The router refuses by name;
    the session must read WHY and be able to act on it, not see a dead tool."""
    from services import session_tools_rpc as rpc

    refusal = {"success": False,
               "error": "GOOGLECALENDAR is not connected in this workspace. "
                        "Connect it on the Tools page, then retry."}
    _record(monkeypatch)(refusal)

    tool = st.get_tool("composio_execute")
    params = st.resolve_parameters(tool, {"action": "GOOGLECALENDAR_FIND_EVENT"}, CTX)
    out = asyncio.run(st.call_tool(object(), tool, params, CTX))
    assert out["success"] is False

    rendered = rpc.render_result(out)
    assert rendered["isError"] is True
    body = rendered["content"][0]["text"]
    assert "GOOGLECALENDAR is not connected" in body and "Tools page" in body


def test_the_key_is_never_in_the_request_or_the_response(monkeypatch):
    """The whole point of routing Composio through the backend: the session says
    WHICH action, Automatos holds the credential. Nothing the session sends can
    carry one, and nothing it gets back may either."""
    from services import session_tools_rpc as rpc

    secret = "ak_live_0123456789abcdef"
    answer = {"success": True, "result": {"events": [{"summary": "Standup"}]},
              # a router that leaked one would look like this
              "connection": {"toolkit": "GOOGLECALENDAR", "status": "ACTIVE"}}
    seen = _record(monkeypatch)(answer)

    tool = st.get_tool("composio_execute")
    params = st.resolve_parameters(
        tool, {"action": "GOOGLECALENDAR_FIND_EVENT", "calendar_id": "primary"}, CTX)
    out = asyncio.run(st.call_tool(object(), tool, params, CTX))

    # the request carries an action and its parameters, and no credential field
    sent = seen["parameters"]
    assert set(sent) == {"action", "params"}
    assert secret not in json.dumps(sent)
    for forbidden in ("api_key", "apiKey", "auth_config", "connected_account_id", "bearer", "token"):
        assert forbidden not in json.dumps(sent).lower(), forbidden

    # and neither the tool's schema nor what comes back invites one
    assert set(tool.input_schema["properties"]) == {"action", "params"}
    assert secret not in json.dumps(rpc.render_result(out))
