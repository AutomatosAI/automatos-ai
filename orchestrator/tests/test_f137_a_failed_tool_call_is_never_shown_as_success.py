"""F137 (night 4) — a step's tool summary says a call failed when it failed.

B22: `GMAIL_CREATE_EMAIL_DRAFT (success)` four times over, where every result
was "Tool composio_execute failed: Action not allowed for this intent: Unknown
action: execute". The summary guessed from the text, looking for "error" in its
first 100 characters, and that failure never says "error". The router's envelope
carried `success: False` all along; the step's record dropped it at every append
site, and so did the normaliser. Now each call records its own flag, and the
summary reads it. A record from before the flag keeps the old guess.
"""
from __future__ import annotations

import ast
from pathlib import Path

from api import recipe_executor as rex
from tests.helpers_playbook_run import run_playbook

B22 = "Tool composio_execute failed: Action not allowed for this intent: Unknown action: execute"


def test_a_failed_call_is_summarised_as_an_error():
    compact = rex._build_compact_step_result({"tool_calls": [
        {"action": "GMAIL_CREATE_EMAIL_DRAFT", "result": B22, "success": False},
        {"action": "platform_search_documents", "result": "3 hits", "success": True},
    ]})
    assert compact["tool_calls_summary"] == ["GMAIL_CREATE_EMAIL_DRAFT (error)", "platform_search_documents (success)"]


def test_the_normaliser_keeps_the_flag():
    [call] = rex._normalize_tool_calls([{"action": "GMAIL_CREATE_EMAIL_DRAFT", "result": B22, "success": False}])
    assert call["success"] is False


def test_a_record_without_the_flag_keeps_the_old_guess():
    compact = rex._build_compact_step_result({"tool_calls": [
        {"action": "old_a", "result": "ERROR: timed out"}, {"action": "old_b", "result": "done"}]})
    assert compact["tool_calls_summary"] == ["old_a (error)", "old_b (success)"]


def test_the_run_record_shows_the_failed_drafts_as_errors(monkeypatch):
    step = {"step_id": "s1", "order": 1, "agent_id": 7, "error_handling": "stop", "max_retries": 0,
            "prompt_template": "Draft the four café emails."}
    outcome = {"status": "success", "result": "Drafted the four emails.", "execution": {
        "tokens_used": 900,
        "tool_calls": [{"action": "GMAIL_CREATE_EMAIL_DRAFT", "result": B22, "success": False}] * 4,
    }}
    execution, _card = run_playbook(monkeypatch, outcomes=[outcome], step_seconds=5, exec_config={}, steps=[step])
    assert execution.step_results[0]["tool_calls_summary"] == ["GMAIL_CREATE_EMAIL_DRAFT (error)"] * 4


def test_every_tool_call_the_step_records_carries_its_flag():
    """The scratchpad tools, the LinkedIn workaround, the Composio spine, the router,
    and F140's ask to the owner and the calls not run after it."""
    tree = ast.parse(Path(rex.__file__).read_text())
    records = [
        node.args[0] for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "all_tool_calls"
        and node.args and isinstance(node.args[0], ast.Dict)
    ]
    assert len(records) == 7
    for record in records:
        keys = {k.value for k in record.keys if isinstance(k, ast.Constant)}
        assert "success" in keys, f"line {record.lineno}: {sorted(keys)}"
