"""P256-FIX-N17-10 (F449, night 17): a refused call says why in the owner's words.

Night 17 (455 asks): 141 refused receipts, each with the reason written for the model: 66 call-shape
("'platform_board_snapshot' does not take ['status'] … Fix the call and send it again", "Missing
required params"), 3 unknown actions (ask 229 platform_list_questions; ask 411 search_knowledge through
platform_execute), 10 intent-gate refusals, 6 psycopg2 traces and 6 "Validation failed" printing the
workspace's table list. Each refusal now carries ``owner_reason`` where it is made, and the receipt reads
it; the model's ``error`` (F182/F027-C) and the router's ``llm_context`` are untouched. Shapes are the
night's; the night's data stays local.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from consumers.chatbot.receipts import REFUSED, WRITE, build_receipts, honesty_lines, receipt
from modules.tools.execution.owner_reasons import (
    DATA_FAILED, NO_SUCH_ACTION, OWNER_REASON, PLATFORM_FAILED, SENT_WRONG, Refusal,
)
from modules.tools.execution.refused_calls import in_the_owners_words
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
OWNER = {"workspace_role": "owner"}
# What a reason the owner reads never carries: the text written for the model.
MODEL_WORDS = ("does not take", "Fix the call", "params={", "platform_", "Use one of")
TABLES = "Validation failed: relation \"green_lot\" does not exist. Tables: green_lots, roast_logs, wholesale_orders"
PSYCOPG = ("Execution error: (psycopg2.errors.UndefinedColumn) column \"kg\" does not exist\n"
           "LINE 1: SELECT kg FROM green_lots")


class _NoTools:
    """A tool registry with no tools of its own: the executor's routes are the tools."""

    def get_tool(self, name):
        return None

    def get_all_tools(self):
        return []


@pytest.fixture
def executor(monkeypatch):
    """The unified executor, its platform actions recorded and never run (F182's dispatch fixture)."""
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran = []

    async def _run(self, action_name, params, **kwargs):
        ran.append((action_name, params))
        return {"success": True}

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_platform_action", _run)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    made = UnifiedToolExecutor(None, registry=_NoTools())

    def call(tool, params):
        return asyncio.run(made.execute_tool(tool, params, agent_id=322, workspace_id=WS, trace_id="t-f449"))
    return SimpleNamespace(call=call, ran=ran)


def _reason(tool, params, result):
    """The reason the turn's receipt gives for the call, as the loop records it."""
    tracker = ToolExecutionTracker()
    tracker.record_outcome(tool, params, result)
    (only,) = build_receipts(tracker)
    assert only["status"] == REFUSED, only
    return only["reason"]


def _in_owner_words(reason):
    assert not any(words in reason for words in MODEL_WORDS), reason


def test_an_action_the_platform_does_not_have_is_said_so(executor):
    params = {"action": "platform_list_questions", "params": {}}        # ask 229
    result = executor.call("platform_execute", params)

    assert result["error"].startswith("Unknown platform action: 'platform_list_questions'. Use one of:")
    assert _reason("platform_execute", params, result) == NO_SUCH_ACTION and executor.ran == []


def test_a_real_tool_through_the_dispatcher_was_sent_wrong(executor):
    params = {"action": "search_knowledge", "params": {"query": "Guji quote"}}     # ask 411
    result = executor.call("platform_execute", params)

    assert result["error"].startswith("Unknown platform action: 'search_knowledge'")
    assert _reason("platform_execute", params, result) == SENT_WRONG


@pytest.mark.parametrize("tool, params, model_text", [
    ("platform_board_snapshot", {"status": "review"}, "'platform_board_snapshot' does not take ['status']"),
    ("platform_execute", {"action": "platform_recommend_agent", "params": {}},
     "Missing required params for 'platform_recommend_agent': ['objective']"),
    ("platform_execute", {}, "Missing required field: action"),
    ("platform_execute", {"action": "platform_list_tasks", "params": "not json {"},
     "platform_list_tasks's params must be an object of its parameters"),
], ids=["direct-board-snapshot-status", "recommend-agent-empty", "no-action", "params-not-json"])
def test_a_call_sent_in_the_wrong_form_says_so_and_the_model_keeps_its_text(executor, tool, params, model_text):
    result = executor.call(tool, params)

    assert result["success"] is False and model_text in result["error"] and executor.ran == []
    reason = _reason(tool, params, result)
    assert reason == SENT_WRONG
    _in_owner_words(reason)


def test_composio_params_that_are_no_object_were_sent_wrong(executor):
    params = {"action": "GMAIL_SEND_EMAIL", "params": "send it to Maya"}
    result = executor.call("composio_execute", params)

    assert "params must be an object of its parameters" in result["error"]
    assert result[OWNER_REASON] == SENT_WRONG


def test_the_platform_executor_says_its_own_refusals_in_the_owners_words():
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(MagicMock(), uuid.uuid4())
    nope = asyncio.run(executor.execute("platform_nope", {}, OWNER))
    text = asyncio.run(executor.execute("platform_list_tasks", "not json {", OWNER))

    assert (nope["error"], nope[OWNER_REASON]) == ("Unknown platform action: platform_nope", NO_SUCH_ACTION)
    assert (text["error"], text[OWNER_REASON]) == ("Invalid params format: expected dict, got string", SENT_WRONG)


def test_the_intent_gate_names_the_app_and_the_model_still_reads_its_refusal():
    from modules.tools import tool_router
    from modules.tools.tool_router import ToolRouter

    sent = {"action": "GMAIL_SEND_EMAIL", "params": {"recipient_email": "maya@tidewater.example"}}
    with patch.object(tool_router, "composio_available", lambda: True), \
            patch.object(tool_router, "validate_action_for_intent", lambda **kw: (False, "a send is not a read")), \
            patch.object(tool_router, "execute_tool", new=AsyncMock()) as ran, \
            patch.object(ToolRouter, "_record_tool_signal", lambda *a, **k: None):
        envelope = asyncio.run(ToolRouter().execute_and_format("composio_execute", sent, agent_id=1,
                                                               original_intent="read my roast log"))

    assert envelope["llm_context"].startswith("Tool composio_execute failed: Action not allowed for this intent")
    assert _reason("composio_execute", sent, {"success": False, "llm_context": envelope["llm_context"],
                                              "raw_result": envelope["raw_result"]}) == \
        "the Gmail action isn't allowed for this request"
    ran.assert_not_called()


@pytest.mark.parametrize("action, error, said", [
    ("platform_query_data", PSYCOPG, DATA_FAILED),
    ("platform_query_data", TABLES, DATA_FAILED),
    ("query_database", "Validation failed: no such column", DATA_FAILED),
    ("platform_create_blog_post", "Failed to create blog mission: (psycopg2.errors.NotNullViolation) null value",
     PLATFORM_FAILED),
    ("platform_update_agent", "sqlalchemy.exc.OperationalError: server closed the connection", PLATFORM_FAILED),
])
def test_a_driver_or_validator_failure_is_said_plainly_without_the_tables(action, error, said):
    reason = receipt(action, {"question": "how much Guji is left?"}, {"success": False, "error": error})["reason"]
    assert reason == said and "green_lots" not in reason


def test_a_handlers_own_words_are_kept():
    refused = receipt("platform_get_agent", {"agent_id": 99}, {"success": False, "error": "No agent #99 in this workspace."})
    assert refused["reason"] == "No agent #99 in this workspace."


def test_an_agent_name_clash_says_how_many_and_never_the_calls_agent_id(monkeypatch):
    from modules.tools.discovery import agent_binding, agent_refs

    twins = [SimpleNamespace(id=211, name="BEANCOUNTER", job_title="Bookkeeper"),
             SimpleNamespace(id=267, name="BEANCOUNTER", job_title="Accountant")]
    monkeypatch.setattr(agent_binding, "named_agents", lambda db, ws, said: twins)
    monkeypatch.setattr(agent_refs, "active_named", lambda db, ws, said: twins)
    said = "2 agents are called BEANCOUNTER, so it wasn't clear which one you meant"

    _agent, bound = agent_binding.one_named(None, WS, "BEANCOUNTER", "platform_update_agent")
    _agent, clash = agent_refs.resolve_active_agent(None, WS, "BEANCOUNTER")
    from_a_handler = in_the_owners_words({"success": False, "error": clash}, "platform_create_task", {}, None)

    assert bound[OWNER_REASON] == said and "agent_id" in bound["error"]
    assert isinstance(clash, Refusal) and from_a_handler[OWNER_REASON] == said
    for result in (bound, from_a_handler):
        assert _reason("platform_execute", {"action": "platform_update_agent", "params": {}}, result) == said
        assert "agent_id" not in result[OWNER_REASON]


def test_a_line_names_the_real_failure_after_a_call_sent_wrong():
    """Ask 38: the call was sent wrong, then sent right and failed inside the platform."""
    first = {"success": False, "error": "'platform_create_blog_post' does not take ['title']", OWNER_REASON: SENT_WRONG}
    then = {"success": False, "error": "Failed to create blog mission: (psycopg2.errors.UndefinedTable) missions"}
    tracker = ToolExecutionTracker()
    for result in (first, then):
        tracker.record_outcome("platform_create_blog_post", {"topic": "Guji harvest"}, result)

    (line,) = honesty_lines(build_receipts(tracker), "Here's the plan.")
    assert line.startswith("I tried to ") and line.endswith(f": {PLATFORM_FAILED}.")


def test_the_same_line_is_said_once():
    """Ask 370: two refusals that read the same are one line."""
    gated = "the Gmail action isn't allowed for this request"
    writes = [{"action": slug, "kind": WRITE, "status": REFUSED, "subject": "", "reason": gated}
              for slug in ("GMAIL_REPLY_TO_THREAD", "GMAIL_FORWARD_MESSAGE")]

    assert honesty_lines(writes, "") == [f"I tried to do that and it didn't go through: {gated}."]
