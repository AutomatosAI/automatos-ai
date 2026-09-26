"""F131 (night 4, the persona's fix-first #3) — a step that ran to its end but
failed ends its run as failed, never "Playbook complete".

The run's status came from the loop finishing, never from what the steps did.
B33, B25: a step's tool call failed in words while the turn ended normally, so
its 'stop' never fired and the run reported complete, "ok". B47: a session step
whose MCP client never reached Automatos wrote "No Automatos tools this session"
and the bell said "Playbook complete". Two deterministic signals now fail the
step and go through its own error handling: its LAST tool call failed (F137's
flag; a call it recovered from does not count), or its session was offered the
Automatos tools and never connected. That second one is stamped by the backend
itself, when the session's MCP client sends `initialize`; it is never read from
the agent's own words, which vary. Whether an answer MEANS failure is the PRD-204
watch's job (Gerard's wiring).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

from api import recipe_executor as rex
from services import cli_host_service as svc
from tests.helpers_playbook_run import run_playbook

B22 = "Tool composio_execute failed: Action not allowed for this intent: Unknown action: execute"
STOP = {"step_id": "s1", "order": 1, "agent_id": 7, "error_handling": "stop", "max_retries": 0,
        "prompt_template": "Draft the four café payment chasers."}


def _outcome(tool_calls, output="Drafted the four emails.", **extra):
    return {"status": "success", "result": output,
            "execution": {"tokens_used": 900, "tool_calls": tool_calls}, **extra}


def _run(monkeypatch, outcome, step=STOP):
    return run_playbook(monkeypatch, outcomes=[outcome], step_seconds=5, exec_config={}, steps=[step])


# ── a failed last tool call ─────────────────────────────────────────────────

def test_a_step_whose_last_tool_call_failed_fails_the_run(monkeypatch):
    execution, card = _run(monkeypatch, _outcome([{"action": "GMAIL_CREATE_EMAIL_DRAFT", "result": B22,
                                                   "success": False}]))
    assert execution.status == "failed"
    assert execution.error_message.startswith(
        "Step 1 failed: its last tool call, GMAIL_CREATE_EMAIL_DRAFT, failed: Tool composio_execute failed")
    assert card.status != "done"


def test_a_failed_call_the_step_recovered_from_does_not_fail_it(monkeypatch):
    execution, card = _run(monkeypatch, _outcome([
        {"action": "platform_query_graph", "result": "Missing required params", "success": False},
        {"action": "platform_query_data", "result": "3 rows", "success": True},
    ]))
    assert execution.status == "completed" and card.status == "done"


def test_skip_still_skips_a_step_that_failed_this_way(monkeypatch):
    execution, _card = _run(monkeypatch, _outcome([{"action": "GMAIL_CREATE_EMAIL_DRAFT", "result": B22,
                                                    "success": False}]), step={**STOP, "error_handling": "skip"})
    assert execution.status == "completed"
    assert execution.step_results[0]["status"] == "failed"


# ── a session that never reached Automatos ──────────────────────────────────

B47 = ("The Automatos server didn't connect. It timed out after 30 seconds… No Automatos tools this session.")


def test_a_session_that_never_reached_automatos_fails_the_step(monkeypatch):
    execution, _card = _run(monkeypatch, _outcome([], output=B47, runtime="cli", session_connected=False))
    assert execution.status == "failed"
    assert execution.error_message == (
        "Step 1 failed: the session never reached Automatos, so it ran without any of its tools")


def test_an_unknown_connection_is_not_read_as_a_failure(monkeypatch):
    execution, _card = _run(monkeypatch, _outcome([], output="Newsletter drafted.", runtime="cli",
                                                  session_connected=None))
    assert execution.status == "completed"


def test_the_ticket_says_whether_its_session_reached_automatos():
    from services.cli_ticket_lane import exec_result_for

    def ticket(ref):
        return NS(id=763, status="done", result="Newsletter drafted.", runtime_ref=ref)

    offered = {svc.SESSION_TOOLS_OFFERED_KEY: True}
    assert exec_result_for(ticket(offered))["session_connected"] is False
    assert exec_result_for(ticket({**offered, svc.SESSION_CONNECTED_KEY: "2026-09-23 09:27:12"}))[
        "session_connected"] is True
    assert exec_result_for(ticket({}))["session_connected"] is None      # a ticket from before the stamp


def test_the_session_endpoint_stamps_the_ticket_when_the_session_connects():
    """The writer and the reader share svc.SESSION_CONNECTED_KEY: a rename cannot split them."""
    from api import session_tools

    sql = []

    class _Db:
        def execute(self, statement, params=None):
            sql.append((" ".join(str(statement).split()), params))
            return NS(first=lambda: None)

        def commit(self):
            pass

        def rollback(self):
            pass

        def expire(self, obj, attrs=None):
            pass

    class _Request:
        async def json(self):
            return {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                               "clientInfo": {"name": "claude-code", "version": "2"}}}

    reply = asyncio.run(session_tools.session_tools_mcp(_Request(), (NS(id=763, runtime_ref={}), None), _Db()))
    assert reply.status_code == 200
    stamps = [params for statement, params in sql if statement.startswith("UPDATE board_tasks SET runtime_ref")]
    assert stamps == [{"path": "{" + svc.SESSION_CONNECTED_KEY + "}", "task_id": 763}]


def test_step_failure_reads_only_the_two_signals():
    assert rex.step_failure([], {"status": "success", "result": "I stopped: no roast log."}) is None
    assert rex.step_failure([{"action": "x", "success": None}], {}) is None
