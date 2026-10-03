"""F181 (night 6, in chat) — Auto counts an uploaded spreadsheet with code, or
says how much of it it read.

2 Oct 14:21Z (chat fce973f3): asked for the numbers in the club list, Auto read
the sheet's workspace copy from read_document and called workspace_exec three
times, python3 and then awk. Each call failed on "'str' object has no attribute
'get'": Gemini sent platform_execute's ``params`` as a JSON string, and the
dispatcher read it as an object. With no code to run, Auto read pages by eye
and answered "313, the exact numbers" (501 was right). On a board card, where
the agent's code ran, the same file counted 440 rows correctly.

Now the tool loop decodes a ``params`` sent as JSON text, the dispatcher refuses
a ``params`` that is no object instead of crashing, and read_document tells the
model to say how many rows it read whenever code cannot run.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
COUNT = {"command": "python3 -c \"import csv; print(sum(1 for _ in csv.DictReader(open('documents/club-list.csv'))))\""}


def _call(name, arguments):
    return {"id": "call_1", "function": {"name": name, "arguments": arguments}}


@pytest.fixture
def dispatch(monkeypatch):
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran = []

    async def _workspace(self, action_name, params, **kwargs):
        ran.append((action_name, params))
        return {"success": True, "stdout": "501\n", "exit_code": 0}

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_workspace_action", _workspace)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    executor = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    executor.composio_actions, executor.db = {}, None

    def call(arguments):
        return asyncio.run(executor.execute_tool("platform_execute", arguments,
                                                 agent_id=1, workspace_id=WS, trace_id="t-f181"))
    return SimpleNamespace(call=call, ran=ran)


def test_a_count_sent_as_json_text_runs_whole(dispatch):
    from modules.tools.execution.tool_loop import _tc_args

    # Night 6, as Gemini sent it: params is a string holding the object.
    sent = _call("platform_execute", json.dumps({"action": "workspace_exec", "params": json.dumps(COUNT)}))
    out = dispatch.call(_tc_args(sent))

    assert out["success"] is True and dispatch.ran == [("workspace_exec", COUNT)]


def test_params_that_are_no_object_are_refused_not_crashed_on(dispatch):
    out = dispatch.call({"action": "workspace_exec", "params": "python3 count.py"})

    assert out["success"] is False and dispatch.ran == []                    # night 6: AttributeError
    assert "workspace_exec's params must be an object of its parameters" in out["error"]


def test_only_the_dispatchers_params_and_only_an_object_are_decoded():
    from modules.tools.execution.tool_loop import _tc_args

    other = _call("composio_execute", json.dumps({"action": "X", "params": "{\"a\": 1}"}))
    not_json = _call("platform_execute", json.dumps({"action": "workspace_exec", "params": "python3 count.py"}))
    a_list = _call("platform_execute", json.dumps({"action": "workspace_exec", "params": "[1, 2]"}))
    as_dict = _call("platform_execute", {"action": "workspace_exec", "params": json.dumps(COUNT)})

    assert _tc_args(other)["params"] == "{\"a\": 1}"
    assert _tc_args(not_json)["params"] == "python3 count.py" and _tc_args(a_list)["params"] == "[1, 2]"
    assert _tc_args(as_dict)["params"] == COUNT                               # arguments already an object
    assert _tc_args(_call("platform_execute", "{not json")) == {}


def test_a_spreadsheet_read_says_how_much_was_read_when_code_cannot_run():
    from services.spreadsheet_workspace import COUNT_WITH_CODE

    assert "say how many of its row_count rows you read" in COUNT_WITH_CODE
    assert "call no figure exact" in COUNT_WITH_CODE
