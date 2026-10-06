"""F369 (night 10c, chat 16bb619c 18:11): platform_execute reads params Auto sends as a Python dict.

"It's the one you just made: 'Design Brand Kit' … My board calls it 2147." Auto called
platform_execute {action: platform_update_task, params: "{'task_id': 2147, 'tags':
['sim-night-2026-10-06']}"}: a Python dict written out as text, not JSON. The call was refused and
the ticket never got its tag. Text that is no JSON is now read as a Python literal
(``ast.literal_eval``: values only, nothing runs) and kept only when it is a dict; anything else is
refused as before and never runs.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from modules.tools.discovery.session_ticket import session_ticket_params
from modules.tools.execution.params_text import params_object

WS = "00000000-0000-0000-0000-0000000000c1"
AT_18_11 = "{'task_id': 2147, 'tags': ['sim-night-2026-10-06']}"
TAGGED = {"task_id": 2147, "tags": ["sim-night-2026-10-06"]}
REFUSED = "came as text that is not a JSON object, so nothing ran: send params as an object"


@pytest.fixture
def dispatch(monkeypatch):
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran = []

    async def _platform(self, action_name, params, **kwargs):
        ran.append((action_name, params))
        return {"success": True}

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_platform_action", _platform)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    executor = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    executor.composio_actions, executor.db = {}, None

    def call(params):
        return asyncio.run(executor.execute_tool("platform_execute", {"action": "platform_update_task", "params": params},
                                                 agent_id=1, workspace_id=WS, trace_id="t-f369"))
    return SimpleNamespace(call=call, ran=ran)


def test_the_18_11_call_tags_the_ticket(dispatch):
    out = dispatch.call(AT_18_11)

    assert out["success"] is True
    assert dispatch.ran == [("platform_update_task", TAGGED)]


def test_json_text_still_runs_as_its_object(dispatch):
    assert dispatch.call(json.dumps(TAGGED))["success"] is True
    assert dispatch.ran == [("platform_update_task", TAGGED)]


@pytest.mark.parametrize("params", [
    "[2147, 'sim-night-2026-10-06']",                 # a literal, but a list
    "(2147,)",                                        # a tuple
    "'task 2147'",                                    # a string
    "tag ticket 2147 please",                         # junk
    "{'task_id': 2147, 'tags': {'a', 'b'}}",          # a set: no JSON carries it
    "__import__('os').getcwd()",                      # code: never run
    "{'task_id': __import__('os').getpid()}",
])
def test_anything_but_a_dict_is_refused_and_never_runs(dispatch, params):
    out = dispatch.call(params)

    assert out["success"] is False and REFUSED in out["error"]
    assert dispatch.ran == []


def test_the_decoder_reads_a_python_dict_as_json_would():
    assert params_object(AT_18_11) == TAGGED
    assert params_object("{'steps': (1, 2), 'done': True, 'note': None}") == {"steps": [1, 2], "done": True,
                                                                              "note": None}
    assert params_object(json.dumps(AT_18_11)) == TAGGED                 # the Python text, JSON-encoded once more
    assert params_object("[1, 2]") == "[1, 2]" and params_object("junk") == "junk"


def test_a_sessions_ticket_is_still_injected_over_python_dict_text():
    params = session_ticket_params("platform_render_preview", "{'template': 'brand_board'}",
                                   {"session_task_id": 2147})
    assert params == {"template": "brand_board", "_session_task_id": 2147}
