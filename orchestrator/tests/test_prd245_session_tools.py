"""PRD-245 W1 — the Automatos tools a ticket session may call, and the MCP wire.

Three things this pins, none of which needs a database or a client:

* the LIST is fixed and stable (Claude Code's prompt cache breaks if the text
  moves between sessions of one agent) and every entry is a legal tool
  definition;
* the SCOPE is forced from the ticket, never taken from the call: a session
  reports on its own ticket, moves its own ticket, and cannot close it;
* the WIRE answers what a client actually sends, and a tool that refused comes
  back as tool OUTPUT (``isError``) — the model must be able to read why —
  while a broken request is a JSON-RPC error.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from services import session_tools as st
from services import session_tools_rpc as rpc

CTX = st.SessionContext(task_id=119, agent_id=268, agent_name="TRACKER", workspace_id="ws-c1")


def _run(coro):
    return asyncio.run(coro)


async def _ok(tool, arguments, ctx):
    return {"success": True, "result": {"todo": 3, "review": 1}}


def _recorder():
    seen = []

    async def call(tool, arguments, ctx):
        seen.append((tool.name, st.resolve_parameters(tool, arguments, ctx)))
        return {"success": True, "result": "done"}

    return seen, call


# ── the list ────────────────────────────────────────────────────────────────

def test_the_tool_list_is_the_one_definition_and_is_stable():
    names = st.tool_names()
    assert names == ("board_summary", "list_tasks", "update_ticket", "submit_report",
                     "ask_human", "composio_execute", "search_knowledge")
    assert st.tool_names() == names                      # stable per process
    first = json.dumps(st.definitions(), sort_keys=True)
    assert json.dumps(st.definitions(), sort_keys=True) == first   # byte-stable (prompt cache)
    for definition in st.definitions():
        assert set(definition) == {"name", "description", "inputSchema"}
        assert definition["description"].strip() and definition["inputSchema"]["type"] == "object"
        assert isinstance(definition["inputSchema"].get("properties"), dict)


def test_every_tool_runs_a_platform_action_that_exists():
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    for tool in st.SESSION_TOOLS:
        assert registry.get(tool.action) is not None, f"{tool.name} → unknown action {tool.action}"


def test_a_skill_that_names_the_api_spelling_is_pointed_at_the_session_one():
    assert st.equivalent_of("platform_submit_report") == "submit_report"
    assert st.equivalent_of("platform_search_memory") == "search_knowledge"
    assert st.equivalent_of("submit_report") == "submit_report"
    assert st.equivalent_of("composio_execute") == "composio_execute"   # W3: the skills' own name
    assert st.equivalent_of("") is None and st.equivalent_of(None) is None


# ── the scope ───────────────────────────────────────────────────────────────

def test_a_session_reports_on_its_own_ticket_only():
    tool = st.get_tool("submit_report")
    params = st.resolve_parameters(tool, {"title": "t", "content": "c", "linked_task_ids": [999, 7]}, CTX)
    assert params["linked_task_ids"] == [119]
    assert set(params) <= {"title", "content", "summary", "report_type", "recommendations", "action_items", "linked_task_ids"}


def test_a_session_moves_its_own_ticket_and_cannot_close_it():
    tool = st.get_tool("update_ticket")
    assert st.resolve_parameters(tool, {"status": "review"}, CTX) == {"task_id": 119, "status": "review"}
    blocked = st.resolve_parameters(tool, {"status": "blocked", "note": "no notes to write from"}, CTX)
    assert blocked == {"task_id": 119, "status": "blocked", "blocked_reason": "no notes to write from"}
    # 'blocked' without a reason still carries one — the action requires it
    assert st.resolve_parameters(tool, {"status": "blocked"}, CTX)["blocked_reason"]
    for refused in ("done", "failed", "cancelled", "assigned", "", "DONE "):
        with pytest.raises(st.SessionToolRefused):
            st.resolve_parameters(tool, {"status": refused}, CTX)


def test_a_filter_tool_passes_only_the_fields_it_documents():
    params = st.resolve_parameters(st.get_tool("list_tasks"),
                                   {"status": "review", "limit": 50, "workspace_id": "somebody-elses"}, CTX)
    assert params == {"status": "review", "limit": 50}
    assert st.resolve_parameters(st.get_tool("search_knowledge"),
                                 {"query": "ledger", "agent_id": 1}, CTX) == {"query": "ledger"}


# ── the wire ────────────────────────────────────────────────────────────────

def test_initialize_echoes_a_version_it_knows_and_ours_otherwise():
    for asked in rpc.SUPPORTED_PROTOCOL_VERSIONS:
        reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                         "params": {"protocolVersion": asked}}, CTX,
                                        server_version="1.0", call=_ok))
        assert reply["result"]["protocolVersion"] == asked
    for unknown in ("1999-01-01", "", None):
        reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                         "params": {"protocolVersion": unknown}}, CTX,
                                        server_version="1.0", call=_ok))
        assert reply["result"]["protocolVersion"] == rpc.LATEST_PROTOCOL_VERSION
    result = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, CTX,
                                     server_version="1.0", call=_ok))["result"]
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["serverInfo"]["name"] == "automatos" and result["serverInfo"]["version"] == "1.0"


def test_a_notification_is_never_answered():
    for message in ({"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}},
                    {"jsonrpc": "2.0", "method": "ping"}):   # no id ⇒ a notification
        assert _run(rpc.handle_message(message, CTX, server_version="1.0", call=_ok)) is None
    assert _run(rpc.handle_payload([{"jsonrpc": "2.0", "method": "notifications/x"}], CTX,
                                   server_version="1.0", call=_ok)) is None


def test_tools_list_is_what_the_registry_says():
    reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, CTX,
                                    server_version="1.0", call=_ok))
    assert [t["name"] for t in reply["result"]["tools"]] == list(st.tool_names())


def test_tools_call_runs_the_tool_with_the_tickets_scope():
    seen, call = _recorder()
    reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                     "params": {"name": "submit_report",
                                                "arguments": {"title": "t", "content": "c"}}},
                                    CTX, server_version="1.0", call=call))
    assert reply["result"]["isError"] is False
    assert seen == [("submit_report", {"title": "t", "content": "c", "linked_task_ids": [119]})]


def test_a_refusal_a_failure_and_a_crash_all_come_back_as_readable_output():
    refused = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                       "params": {"name": "update_ticket", "arguments": {"status": "done"}}},
                                      CTX, server_version="1.0", call=_ok))
    assert "error" not in refused and refused["result"]["isError"] is True
    assert "not yours to record" in refused["result"]["content"][0]["text"]

    async def failing(tool, arguments, ctx):
        return {"success": False, "error": "the board is unavailable"}

    failed = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                      "params": {"name": "board_summary"}}, CTX, server_version="1.0", call=failing))
    assert "error" not in failed and failed["result"]["isError"] is True
    assert "board is unavailable" in failed["result"]["content"][0]["text"]

    async def crashing(tool, arguments, ctx):
        raise RuntimeError("secret internal detail")

    crashed = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                                       "params": {"name": "board_summary"}}, CTX, server_version="1.0", call=crashing))
    assert crashed["result"]["isError"] is True
    assert "secret internal detail" not in crashed["result"]["content"][0]["text"]


def test_a_tool_we_never_offered_is_named_not_guessed():
    reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                     "params": {"name": "delete_workspace"}}, CTX, server_version="1.0", call=_ok))
    text = reply["result"]["content"][0]["text"]
    assert reply["result"]["isError"] is True and "delete_workspace" in text
    for name in st.tool_names():
        assert name in text          # the model is told what it DOES have


def test_the_allowance_refuses_in_words_the_model_can_act_on():
    reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                                     "params": {"name": "board_summary"}}, CTX, server_version="1.0",
                                    call=_ok, on_call=lambda name: "This ticket has used its 200 Automatos tool calls."))
    assert reply["result"]["isError"] is True and "200 Automatos tool calls" in reply["result"]["content"][0]["text"]


def test_an_unknown_method_and_a_broken_request_are_json_rpc_errors():
    for method in ("nonsense", "logging/setLevel", "completion/complete", rpc.DISCOVERY_PROBE_METHOD):
        reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 9, "method": method}, CTX,
                                        server_version="1.0", call=_ok))
        assert reply["error"]["code"] == rpc.METHOD_NOT_FOUND, method
    assert _run(rpc.handle_message("not an object", CTX, server_version="1.0", call=_ok))["error"]["code"] == rpc.INVALID_REQUEST
    assert _run(rpc.handle_payload([], CTX, server_version="1.0", call=_ok))["error"]["code"] == rpc.INVALID_REQUEST


def test_a_batch_answers_every_request_and_no_notification():
    replies = _run(rpc.handle_payload([
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ], CTX, server_version="1.0", call=_ok))
    assert [r["id"] for r in replies] == [1, 2]


def test_a_long_result_is_truncated_not_dropped():
    async def huge(tool, arguments, ctx):
        return {"success": True, "result": "x" * (st.MAX_TOOL_RESULT_CHARS + 500)}

    reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                                     "params": {"name": "board_summary"}}, CTX, server_version="1.0", call=huge))
    text = reply["result"]["content"][0]["text"]
    assert len(text) <= st.MAX_TOOL_RESULT_CHARS + 32 and text.endswith("(truncated)")


# ── what the real client does, in the order it does it ──────────────────────

def test_the_capability_probe_before_initialize_is_answered_not_refused():
    """Claude Code 2.1.267's v2 runtime probes ``server/discover`` FIRST. Method
    not found over HTTP 200 is what classifies us as a legacy server and lets the
    handshake proceed; a hang or a 4xx loses the connection before it starts."""
    reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 0, "method": rpc.DISCOVERY_PROBE_METHOD,
                                     "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}}},
                                    CTX, server_version="1.0", call=_ok))
    assert reply["error"]["code"] == rpc.METHOD_NOT_FOUND
    assert "result" not in reply


def test_we_never_echo_a_version_the_client_would_refuse():
    """The client accepts back only its pre-2026 list; ``2026-07-28`` — the
    version its own probe names — fails the handshake."""
    assert "2026-07-28" not in rpc.SUPPORTED_PROTOCOL_VERSIONS
    assert rpc.LATEST_PROTOCOL_VERSION == "2025-11-25"      # what 2.1.267 asks for
    for asked in ("2026-07-28", "2027-01-01", "nonsense"):
        reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                         "params": {"protocolVersion": asked}}, CTX,
                                        server_version="1.0", call=_ok))
        assert reply["result"]["protocolVersion"] in rpc.SUPPORTED_PROTOCOL_VERSIONS


def test_a_list_method_for_a_capability_we_never_declared_returns_empty():
    """Cheaper than being reported broken: the client's discovery step asks for
    prompts and resources alongside tools."""
    for method, key in (("prompts/list", "prompts"), ("resources/list", "resources"),
                        ("resources/templates/list", "resources")):
        reply = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 2, "method": method}, CTX,
                                        server_version="1.0", call=_ok))
        assert reply["result"] == {key: []}, (method, reply)


def test_every_tool_schema_is_one_a_client_will_accept():
    """A tool whose input schema fails the client's check is dropped from the
    session SILENTLY (the reason only reaches the CLI's own log): property names
    must be 1–64 chars of ASCII letters, digits, ``_``, ``.`` or ``-``."""
    import re

    allowed = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
    for definition in st.definitions():
        assert allowed.match(definition["name"]), definition["name"]
        schema = definition["inputSchema"]
        assert schema["type"] == "object" and isinstance(schema.get("required", []), list)
        for prop, spec in (schema.get("properties") or {}).items():
            assert allowed.match(prop), (definition["name"], prop)
            assert spec.get("type") in ("string", "integer", "number", "boolean", "array", "object"), (prop, spec)
            if spec.get("type") == "array":
                assert isinstance(spec.get("items"), dict), (definition["name"], prop)
        for name in schema.get("required", []):
            assert name in (schema.get("properties") or {}), (definition["name"], name)


def test_the_tool_name_the_model_sees_stays_short_and_plain():
    """``mcp__automatos__<tool>`` is what the model, the permission rules and the
    hook payloads all match on."""
    for name in st.tool_names():
        wire = f"mcp__automatos__{name}"
        assert wire.replace("_", "").isalnum() and len(wire) <= 64, wire
