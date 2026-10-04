"""F321-B (night 9b, build 15) — a platform action's params sent as JSON text
run as the object they hold; anything else is refused in plain words.

4 Oct 15:32Z: run #0084 of "Monday green stock" (exec-84c41d2630d1, board_tasks
1941) failed on "Step 2 failed: its last tool call, platform_submit_report,
failed: Tool platform_execute failed: platform_submit_report's params must be an
object of its parameters, e.g. {...}, not str." Runs #0085 (exec-76879400dd8d)
and #0102 (exec-45d8ac862a79) lost their reports the same way: six refused
platform_submit_report calls each (the backend log, 15:50:01–15:51:09Z, tool
traces d0368fa1e5f3 …), then the agent saved the report to the scratchpad. A
playbook step parses its own tool calls, so F181's decoding in the chat's tool
loop never reached it. The decoding now lives in the executor every caller uses.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

WS = "febae41b-374b-4580-a5ef-f698bdd382e4"
STOCK = [("Brazil Cerrado", 540.0), ("Guji Shakiso", 118.0), ("Huila La Esperanza", 212.0),
         ("Kirinyaga AA", 41.0), ("Nariño Buesaco", 300.0), ("Sumatra Gayo", 176.0),
         ("Swiss Water Decaf", 64.0), ("Yirgacheffe Konga", 19.1)]
CONTENT = "# Green stock\n" + "\n".join(
    f"- {name}: {kg} kg{' (under 50 kg)' if kg < 50 else ''}" for name, kg in STOCK)
REPORT = {"title": "Green stock report", "content": CONTENT, "report_type": "stock_report"}


@pytest.fixture
def dispatch(monkeypatch):
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran = []

    async def _platform(self, action_name, params, **kwargs):
        ran.append((action_name, params))
        return {"success": True, "report_id": "r1"}

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_platform_action", _platform)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    executor = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    executor.composio_actions, executor.db = {}, None

    def call(name, arguments):
        return asyncio.run(executor.execute_tool(name, arguments, agent_id=342, workspace_id=WS,
                                                 trace_id="t-f321"))
    return SimpleNamespace(call=call, ran=ran)


def test_a_playbook_steps_report_sent_as_json_text_is_submitted_whole(dispatch):
    # The step's shape: the tool call's arguments parsed once, params still text.
    out = dispatch.call("platform_execute", {"action": "platform_submit_report", "params": json.dumps(REPORT)})

    assert out["success"] is True
    assert dispatch.ran == [("platform_submit_report", REPORT)]
    assert "Yirgacheffe Konga: 19.1 kg" in dispatch.ran[0][1]["content"]


def test_a_report_whose_text_has_raw_line_breaks_is_still_read():
    from modules.tools.execution.params_text import params_object

    # A model writing a long report often leaves the newlines unescaped.
    raw = '{"title": "Green stock report", "content": "' + CONTENT + '"}'

    assert params_object(raw) == {"title": "Green stock report", "content": CONTENT}
    assert params_object(json.dumps(json.dumps(REPORT))) == REPORT        # sent twice-encoded


def test_params_that_are_not_an_object_are_refused_in_plain_words(dispatch):
    out = dispatch.call("platform_execute", {"action": "platform_submit_report", "params": "stock report, 8 coffees"})
    listed = dispatch.call("platform_execute", {"action": "platform_submit_report", "params": "[1, 2]"})

    assert out["success"] is False and listed["success"] is False and dispatch.ran == []
    assert "came as text that is not a JSON object, so nothing ran: send params as an object" in out["error"]
    assert "not str" not in out["error"]


def test_composio_params_text_is_read_or_refused_never_run_empty():
    from modules.tools.execution.params_text import decodes_nested_params

    seen = []

    async def _execute(self, tool_name, parameters, *args, **kwargs):
        seen.append(parameters)
        return {"success": True}

    wrapped = decodes_nested_params(_execute)
    as_text = {"action": "GMAIL_SEND_EMAIL", "params": json.dumps({"to": "a@b.co"})}
    readable = asyncio.run(wrapped(None, "composio_execute", as_text))
    refused = asyncio.run(wrapped(None, "composio_execute", {"action": "GMAIL_SEND_EMAIL", "params": "to a@b.co"}))

    assert readable["success"] is True and seen == [{"action": "GMAIL_SEND_EMAIL", "params": {"to": "a@b.co"}}]
    assert refused["success"] is False and "GMAIL_SEND_EMAIL's params must be an object" in refused["error"]
    assert len(seen) == 1                                                   # the unreadable call never ran
