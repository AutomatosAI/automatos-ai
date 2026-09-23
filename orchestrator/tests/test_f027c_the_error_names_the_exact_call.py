"""F027-C (night 3) — a missing-params error names the exact call to make.

69 of night 3's 98 failed platform_execute calls came back "Missing required
params … Pass them inside params={...}". Every one carried exactly
["action", "params"]: the model (gemini-2.5-flash, 10 output tokens each) sent
params={} and retried the same after the hint. The error now spells out the
call — every required key with its type, from the action's schema — and the
playbook-step actions accept the names their arguments arrived under. No
schema the model sees changes.
"""
from __future__ import annotations

import inspect

import pytest

from modules.tools.execution.unified_executor import _fill_required_from_aliases

ACTIONS = ["platform_read_document", "platform_query_graph", "platform_search_documents",
           "platform_ask_human", "platform_submit_report"]


def _schema(action):
    from modules.tools.discovery import get_action_registry

    return get_action_registry().get(action).parameters


@pytest.mark.parametrize("action", ACTIONS)
def test_the_error_names_every_required_key_in_the_exact_call(action):
    from modules.tools.execution.unified_executor import missing_params_error

    schema = _schema(action)
    required = list(schema.get("required") or [])
    assert required
    error = missing_params_error(action, schema, required, {})
    call = error.splitlines()[1]
    assert call.startswith('Call it exactly like this: {"action": "' + action + '", "params": {')
    assert all(f'"{key}": ' in call for key in required)
    assert "Your params was empty" in error
    assert error.startswith(f"Missing required params for '{action}': {required}")


def test_each_type_is_written_the_way_the_call_takes_it():
    from modules.tools.execution.unified_executor import missing_params_error

    schema = {"properties": {"document_id": {"type": "integer"}, "question": {"type": "string"},
                             "tags": {"type": "array"}, "mode": {"type": "string", "enum": ["fast", "full"]}},
              "required": ["document_id", "question", "tags", "mode"]}
    error = missing_params_error("platform_x", schema, ["document_id"], {"question": "Which café?"})
    assert error.splitlines()[1] == ('Call it exactly like this: {"action": "platform_x", "params": '
                                     '{"document_id": <integer>, "question": "<string>", "tags": [...], '
                                     '"mode": "<one of: fast | full>"}}')
    assert "Your params was empty" not in error            # it was not


def test_the_dispatcher_answers_with_the_exact_call():
    from modules.tools.execution import unified_executor

    source = inspect.getsource(unified_executor.UnifiedToolExecutor.execute_tool)
    assert "missing_params_error(action_name, action_def.parameters, missing, action_params)" in source


# ── the playbook-step aliases ───────────────────────────────────────────────

@pytest.mark.parametrize("name", ["prompt", "instructions", "template", "step_prompt"])
def test_add_playbook_step_takes_its_prompt_under_another_name(name):
    filled = _fill_required_from_aliases({"playbook_id": 12, name: "Draft the café reminder"},
                                         ["playbook_id", "prompt_template"])
    assert filled["prompt_template"] == "Draft the café reminder"


def test_update_playbook_step_takes_its_playbook_and_step_under_other_names():
    filled = _fill_required_from_aliases({"recipe_id": 12, "step": 3, "order": 1}, ["playbook_id", "step_index"])
    assert (filled["playbook_id"], filled["step_index"], filled["order"]) == (12, 3, 1)


def test_order_is_never_read_as_the_step_index():
    """update_playbook_step's `order` is where the step moves to, not which step."""
    filled = _fill_required_from_aliases({"playbook_id": 12, "order": 2}, ["playbook_id", "step_index"])
    assert "step_index" not in filled
