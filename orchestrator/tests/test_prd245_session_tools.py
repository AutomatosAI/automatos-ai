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
                     "ask_human", "composio_execute", "search_knowledge",
                     "search_documents", "record_memory",       # night-1 95f261a1f: document + memory parity
                     "read_step_file")                          # F161: a later step reads an earlier step's file
    assert st.tool_names() == names                      # stable per process
    first = json.dumps(st.definitions(), sort_keys=True)
    assert json.dumps(st.definitions(), sort_keys=True) == first   # byte-stable (prompt cache)
    for definition in st.definitions():
        assert set(definition) == {"name", "description", "inputSchema"}
        assert definition["description"].strip() and definition["inputSchema"]["type"] == "object"
        assert isinstance(definition["inputSchema"].get("properties"), dict)


def test_every_platform_tool_runs_an_action_that_exists():
    """A tool dispatched through ``platform_execute`` must name an action the
    registry knows, or the dispatcher refuses it at run time with "Unknown
    platform action". A tool dispatched by NAME (``composio_execute``) is routed
    by the executor itself and is deliberately not in that registry."""
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    for tool in st.SESSION_TOOLS:
        if tool.dispatch != st.DISPATCH_PLATFORM_ACTION:
            continue
        assert registry.get(tool.action) is not None, f"{tool.name} → unknown action {tool.action}"
    # …and the ones dispatched by name are names the executor really handles
    by_name = [t.action for t in st.SESSION_TOOLS if t.dispatch == st.DISPATCH_TOOL_NAME]
    assert by_name == ["composio_execute"], by_name


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


def test_a_session_notes_progress_on_its_own_ticket_and_moves_it_nowhere():
    """A session may not move its ticket out of ``in_progress`` at all.

    ``blocked`` and ``review`` were allowed at first and turned out to be worse
    than ``done``, not milder: ``apply_result`` returns early for a ticket that
    is no longer ``in_progress``, so the moment a session set either one its own
    turn was discarded — no deliverables, no report, no result text, no usage.
    """
    tool = st.get_tool("update_ticket")
    # The note goes through a RUNNER. Dispatching it to platform_update_task_status
    # was a no-op that reported success: for a ticket already ``in_progress`` —
    # which every running session's is — that action takes its atomic-claim
    # branch, whose ``UPDATE … WHERE status <> 'in_progress'`` matches no row,
    # and it never reads ``blocked_reason`` at all. The note vanished and the
    # model was told it had landed.
    assert tool.runner is not None
    assert sorted(tool.input_schema["properties"]) == ["note"]
    assert tool.input_schema["required"] == ["note"]

    assert st.resolve_parameters(tool, {"note": "reading the brief"}, CTX) == {"note": "reading the brief"}
    assert st.resolve_parameters(tool, {"note": "x", "status": "in_progress"}, CTX) == {"note": "x"}
    # the ticket is always ITS ticket, whatever the call says
    assert st.resolve_parameters(tool, {"note": "y", "task_id": 7}, CTX) == {"note": "y"}
    assert len(st.resolve_parameters(tool, {"note": "z" * 900}, CTX)["note"]) == st.MAX_NOTE_CHARS

    for refused in ("blocked", "review", "done", "failed", "cancelled", "assigned", "DONE "):
        with pytest.raises(st.SessionToolRefused) as caught:
            st.resolve_parameters(tool, {"status": refused, "note": "x"}, CTX)
        assert "ask_human" in str(caught.value)          # where a stuck session should go
    with pytest.raises(st.SessionToolRefused):
        st.resolve_parameters(tool, {}, CTX)             # a note is the whole point


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
    assert "cannot move its ticket" in refused["result"]["content"][0]["text"]
    # a scope refusal reaches the model as tool output whatever the tool
    empty = _run(rpc.handle_message({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                     "params": {"name": "ask_human", "arguments": {"question": "  "}}},
                                    CTX, server_version="1.0", call=_ok))
    assert empty["result"]["isError"] is True and "one decision" in empty["result"]["content"][0]["text"]

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


# ---------------------------------------------------------------------------
# The endpoint module itself: the bearer, the 401 shape, the per-ticket cap.
# Pure — no client, no database, no router mounted.
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

import api.session_tools as api_st  # noqa: E402


class _Req:
    def __init__(self, **headers):
        self.headers = {k.replace("_", "-"): v for k, v in headers.items()}


class _CountingDb:
    """A database that counts in one statement, the way the real one does."""

    def __init__(self, fail=False):
        self.commits = 0
        self.fail = fail
        self.value = 0
        self.statements = []

    def execute(self, statement, params=None):
        if self.fail:
            raise RuntimeError("bookkeeping is down")
        self.statements.append(str(statement))
        self.value += 1
        return SimpleNamespace(first=lambda: (self.value,))

    def commit(self):
        if self.fail:
            raise RuntimeError("bookkeeping is down")
        self.commits += 1

    def rollback(self):
        pass

    def expire(self, obj, attrs=None):
        self.expired = list(attrs or [])


def test_the_bearer_is_read_from_either_header_and_trimmed():
    assert api_st.bearer_token(_Req(Authorization="Bearer tok-1")) == "tok-1"
    assert api_st.bearer_token(_Req(Authorization="bearer  tok-2 ")) == "tok-2"   # case + padding
    assert api_st.bearer_token(_Req(**{"X-Session-Token": " tok-3 "})) == "tok-3"
    assert api_st.bearer_token(_Req()) == ""
    # a scheme we do not speak is not a token
    assert api_st.bearer_token(_Req(Authorization="Basic tok-4")) == ""


def test_a_refused_token_401s_without_a_challenge(monkeypatch):
    """No ``WWW-Authenticate``: that header is what starts a client's OAuth
    discovery, and this endpoint has none. With one, the operator would watch the
    CLI try a browser flow instead of reporting a dead token."""
    from fastapi import HTTPException

    monkeypatch.setattr(api_st, "_require_cli_runtime", lambda: None)
    monkeypatch.setattr(api_st.svc, "resolve_session_token", lambda db, token: None)

    with pytest.raises(HTTPException) as caught:
        _run(api_st.require_session(_Req(Authorization="Bearer nope"), db=None))
    assert caught.value.status_code == 401
    assert not (caught.value.headers or {})


def test_require_session_builds_the_identity_from_the_ticket_not_the_call(monkeypatch):
    task = SimpleNamespace(id=119, assigned_agent_id=268, workspace_id="ws-c1", runtime_ref={})
    agent = SimpleNamespace(name="TRACKER")
    monkeypatch.setattr(api_st, "_require_cli_runtime", lambda: None)
    monkeypatch.setattr(api_st.svc, "resolve_session_token", lambda db, token: (task, agent))

    resolved_task, ctx = _run(api_st.require_session(_Req(Authorization="Bearer t"), db=None))
    assert resolved_task is task
    assert (ctx.task_id, ctx.agent_id, ctx.agent_name, ctx.workspace_id) == (119, 268, "TRACKER", "ws-c1")


def test_the_allowance_counts_on_the_ticket_and_refuses_in_words(monkeypatch):
    monkeypatch.setattr(api_st.config, "SESSION_TOOLS_MAX_CALLS_PER_TICKET", 3)
    api_st._UNPERSISTED_CALLS.clear()
    task = SimpleNamespace(id=119, runtime_ref={})
    db = _CountingDb()

    assert api_st.call_allowance(db, task) is None                    # 1
    assert api_st.call_allowance(db, task) is None                    # 2
    assert api_st.call_allowance(db, task) is None                    # 3 — the cap itself passes
    assert db.value == 3                                              # counted on the ROW
    refusal = api_st.call_allowance(db, task)                         # 4 — over
    assert refusal and "3" in refusal
    # the model has to be able to act on it: what happened, and what to do now
    assert "finish your turn" in refusal
    assert db.value == 4 and db.commits == 4
    # the ORM attribute is expired for a reload, never assigned — an assignment
    # marks the row dirty and the tool's own commit would flush the whole
    # document over what the host wrote since
    assert db.expired == ["runtime_ref"]
    assert task.runtime_ref == {}
    # ONE key, not the whole document: the host writes pending_permissions onto
    # this same row while the session runs, and that list decides whether a held
    # command sends the ticket to review.
    assert db.statements and all("jsonb_set" in stmt for stmt in db.statements)
    assert all("runtime_ref = jsonb_set" in stmt for stmt in db.statements)


def test_the_counter_does_not_write_back_the_whole_ref(monkeypatch):
    monkeypatch.setattr(api_st.config, "SESSION_TOOLS_MAX_CALLS_PER_TICKET", 10)
    api_st._UNPERSISTED_CALLS.clear()
    before = {"runtime": "cli", "pending_permissions": [{"request_id": "r1"}]}
    task = SimpleNamespace(id=119, runtime_ref=dict(before))
    db = _CountingDb()
    api_st.call_allowance(db, task)
    # ONE key in ONE statement, and the in-memory document is not touched at all
    assert db.statements and all("jsonb_set" in stmt for stmt in db.statements)
    assert task.runtime_ref == before


def test_the_sql_uses_the_casts_this_repo_can_bind(monkeypatch):
    """SQLAlchemy ``text()`` mis-parses ``:param::type`` (a known trap here), so
    every cast in the statement is ``CAST(… AS …)`` and no bind is followed by
    ``::``. The statement has never run in CI against Postgres — the tests use a
    fake — so its text is checked for the shapes that would fail there."""
    monkeypatch.setattr(api_st.config, "SESSION_TOOLS_MAX_CALLS_PER_TICKET", 10)
    api_st._UNPERSISTED_CALLS.clear()
    db = _CountingDb()
    api_st.call_allowance(db, SimpleNamespace(id=119, runtime_ref={}))
    stmt = db.statements[0]
    import re
    assert not re.search(r":\w+::", stmt), "a bind followed by :: — SQLAlchemy text() will not parse it"
    assert "CAST(:path AS text[])" in stmt                  # jsonb_set wants text[], not text
    assert "CAST(runtime_ref ->> :field AS int)" in stmt
    assert "RETURNING" in stmt


def test_a_broken_counter_still_counts(monkeypatch):
    """Fail-open on the WRITE, not on the rule.

    The cap is the only bound on this endpoint. Returning early when the persist
    fails used to remove it for that call — and a database that is unhappy once
    is usually unhappy for the rest of the run, so the bound quietly disappeared
    exactly when things were going wrong. The count now lives in the request too.
    """
    monkeypatch.setattr(api_st.config, "SESSION_TOOLS_MAX_CALLS_PER_TICKET", 2)
    api_st._UNPERSISTED_CALLS.clear()
    task = SimpleNamespace(id=1, runtime_ref={})
    broken = _CountingDb(fail=True)
    assert api_st.call_allowance(broken, task) is None                # 1
    assert api_st.call_allowance(broken, task) is None                # 2
    assert api_st.call_allowance(broken, task) is not None            # 3 — still refused
    assert api_st._UNPERSISTED_CALLS[1] == 3                          # counted in-process
    assert task.runtime_ref == {}                                     # and never assigned

    # a persisted count that is higher wins, so a recovering database cannot
    # reset the count below what the row already says
    recovering = SimpleNamespace(id=3, runtime_ref={api_st.CALLS_KEY: 5})
    assert api_st.call_allowance(_CountingDb(fail=True), recovering) is not None   # 6 > 2

    # a cap of zero is "no cap", not "no calls"
    monkeypatch.setattr(api_st.config, "SESSION_TOOLS_MAX_CALLS_PER_TICKET", 0)
    open_task = SimpleNamespace(id=2, runtime_ref={})
    db = _CountingDb()
    for _ in range(50):
        assert api_st.call_allowance(db, open_task) is None


# ---------------------------------------------------------------------------
# What a session gets back, and what it cannot get back (review findings)
# ---------------------------------------------------------------------------

def _list_handler_row_keys():
    """The keys the REAL list handler puts on each task — read from its source,
    so this test cannot pass on a shape the handler does not produce."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "modules" / "tools" / "discovery" / "handlers_board_tasks.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    # By NAME: ``list_board_tasks`` is what ``platform_list_tasks`` runs. A looser
    # match (any dict carrying description + error_message) found a DIFFERENT
    # handler's row first, one with raw_prompt and review_mode on it.
    handler = next((n for n in ast.walk(tree)
                    if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == "list_board_tasks"), None)
    assert handler is not None, "list_board_tasks was renamed — re-verify what list_tasks returns"
    for node in ast.walk(handler):
        if isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if "assigned_agent" in keys and "title" in keys:
                return keys
    raise AssertionError("the list handler's row literal was not found")


def test_list_tasks_returns_only_the_fields_it_advertises():
    """The board action hands back every field of every ticket in the workspace,
    including each one's full description — operator-written text that routinely
    carries paths, hostnames and pasted credentials. One call would put every
    other ticket's brief in front of a session an injected page may be steering.

    The shape below is the handler's own: ``tasks`` at the TOP level (the
    executor returns the handler's dict as is — there is no ``result`` key) and
    the agent under ``assigned_agent``. The first version of this test fed the
    projection an imagined shape and passed while the projection did nothing.
    """
    handler_keys = _list_handler_row_keys()
    assert set(st.LIST_TASKS_FIELDS) <= handler_keys, "the projection names a key the handler never emits"
    assert {"description", "error_message"} <= handler_keys       # i.e. there is something to strip

    raw = {
        "success": True, "total": 2,
        "tasks": [
            {"id": 1, "title": "A", "description": "ssh deploy@10.0.0.4 — password hunter2",
             "status": "todo", "priority": "high", "tags": ["ops"], "assigned_agent": "OPS",
             "created_at": "2026-09-18", "started_at": None, "completed_at": None,
             "error_message": "boom at /srv/app"},
            {"id": 2, "title": "B", "description": "internal notes", "status": "done",
             "priority": "low", "tags": [], "assigned_agent": "unassigned",
             "created_at": "2026-09-18", "started_at": None, "completed_at": None,
             "error_message": None},
        ],
    }
    assert set(raw["tasks"][0]) == handler_keys                      # this IS the handler's row
    out = st._project_list_tasks(raw)
    tasks = out["tasks"]
    assert out["total"] == 2                                          # the envelope survives
    assert [t["id"] for t in tasks] == [1, 2]
    assert [t["assigned_agent"] for t in tasks] == ["OPS", "unassigned"]
    for task in tasks:
        assert set(task) == set(st.LIST_TASKS_FIELDS)
    dumped = json.dumps(out)
    assert "hunter2" not in dumped and "10.0.0.4" not in dumped and "/srv/app" not in dumped
    # and what the session reads is the narrowed set, end to end through the wire
    body = rpc.render_result(out)["content"][0]["text"]
    assert "hunter2" not in body and "OPS" in body


def test_a_failure_is_only_projected_when_it_succeeded():
    failure = {"success": False, "error": "no"}
    assert st._project_list_tasks(failure) is failure


def test_only_the_advertised_progress_status_is_accepted():
    """A session that set its ticket to blocked or review used to throw away its
    OWN turn: apply_result returns early for a ticket that is not in_progress, so
    the deliverables, the report, the result text and the usage were all lost."""
    assert st.SESSION_TICKET_STATUSES == ("in_progress",)
    for refused in ("blocked", "review", "done", "failed", "cancelled", "assigned"):
        with pytest.raises(st.SessionToolRefused) as caught:
            st.resolve_parameters(st.get_tool("update_ticket"), {"status": refused}, CTX)
        assert "ask_human" in str(caught.value)              # where a stuck session should go


def test_a_ticket_with_no_agent_runs_no_tools():
    """agent_id 0 is not a harmless placeholder: for composio_execute an agent
    with no explicit assignments inherits every app the workspace has connected."""
    orphan = st.SessionContext(task_id=119, agent_id=None, agent_name=None, workspace_id="ws-c1")
    with pytest.raises(st.SessionToolRefused):
        _run(st.call_tool(None, st.get_tool("board_summary"), {}, orphan))


def test_an_internal_fault_is_not_explained_to_the_session():
    """A catch-all executor sets ``error`` to ``str(exc)``. A refusal we wrote is
    for the model to act on; a driver traceback is not."""
    leaky = rpc.render_result({"success": False, "error":
        'psycopg2.errors.UndefinedColumn: column board_tasks.foo does not exist\nSQL: SELECT ...'})
    body = leaky["content"][0]["text"]
    assert leaky["isError"] is True
    assert "psycopg2" not in body and "SELECT" not in body
    assert "carry on with what you can do" in body
    # a refusal we wrote still reaches the model verbatim
    ours = rpc.render_result({"success": False, "error": "GOOGLECALENDAR is not connected in this workspace"})
    assert "GOOGLECALENDAR is not connected in this workspace" in ours["content"][0]["text"]


def test_a_batch_is_bounded_and_an_unknown_name_still_costs_a_call():
    charged = []

    async def _call(tool, params, ctx):
        return {"success": True, "result": {}}

    over = [{"jsonrpc": "2.0", "id": i, "method": "ping"} for i in range(rpc.MAX_BATCH_MESSAGES + 1)]
    reply = _run(rpc.handle_payload(over, CTX, server_version="1", call=_call))
    assert reply["error"]["code"] == rpc.INVALID_REQUEST
    assert str(rpc.MAX_BATCH_MESSAGES) in reply["error"]["message"]

    # exactly at the cap is fine
    ok = [{"jsonrpc": "2.0", "id": i, "method": "ping"} for i in range(rpc.MAX_BATCH_MESSAGES)]
    assert len(_run(rpc.handle_payload(ok, CTX, server_version="1", call=_call))) == rpc.MAX_BATCH_MESSAGES

    # an unknown tool name is charged before it is judged — otherwise the
    # allowance is avoidable by calling a name that does not exist.
    _run(rpc.handle_message(
        {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
        CTX, server_version="1", call=_call, on_call=lambda name: charged.append(name) or None))
    assert charged == ["nope"]
