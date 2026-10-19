"""P256-FIX-N17-9 (night 17, F448) (C) — a slow read answers in words before the session's client gives up.

WRITER's two search_documents calls on #1756 ended in "The operation timed out.": the
session's own client giving up, while the search waited behind Auto's chat retrieval on a
4-thread executor and the session tool slots, with no deadline of ours. A document or
memory read now has SESSION_TOOL_READ_DEADLINE_S from the moment the call arrived (the
allowance and the wait for a slot count), and answers in words that say what to do
instead. A write is never cut off. Through the route's own functions, as F330's burst test
drives them: ``require_session`` then ``session_tools_mcp``.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

import api.session_tools as api_st
from services import session_tools as st
from services.session_read_deadline import DEADLINED_TOOLS
from services.session_tool_slots import session_tool_slot

DEADLINE_S = 0.2
SLOW_S = 1.0
ANSWERED_WITHIN_S = 0.6
WRITES = ("composio_execute", "generate_document", "run_playbook", "submit_report", "ask_human",
          "propose_brand_kit", "save_approved_brand_kit", "record_memory", "update_ticket",
          "create_template", "update_template")
QUESTION = {"query": "harbourline green coffee price list"}


class _Request:
    """One MCP ``tools/call`` from WRITER's session on #1756."""

    def __init__(self, tool: str, arguments: dict):
        self.headers = {"Authorization": "Bearer ticket-3044"}
        self.state = SimpleNamespace()
        self._body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": tool, "arguments": arguments}}

    async def json(self):
        return self._body


@pytest.fixture
def route(monkeypatch):
    """The route as it runs, with the ticket lookup and the counter faked; ``tool_s`` is how
    long the tool takes, ``ran`` what it ran."""
    seen = {"tool_s": SLOW_S, "ran": []}
    task = SimpleNamespace(id=3044, assigned_agent_id=268, workspace_id="ws-f448", runtime_ref={},
                           orchestration_run_id=None)
    agent = SimpleNamespace(name="WRITER", configuration={"runtime": "cli"})

    async def tool(db, chosen, params, ctx):
        seen["ran"].append(chosen.name)
        await asyncio.sleep(seen["tool_s"])
        return {"success": True, "result": {"passages": ["Harbour Blend £22.00"]}}

    monkeypatch.setattr(api_st, "_require_cli_runtime", lambda: None)
    monkeypatch.setattr(api_st.svc, "resolve_session_token", lambda db, token: (task, agent))
    monkeypatch.setattr(api_st, "_count_call", lambda db, row: 1)
    monkeypatch.setattr(st, "call_tool", tool)
    monkeypatch.setattr(api_st.config, "SESSION_TOOL_READ_DEADLINE_S", DEADLINE_S, raising=False)
    return seen


async def _through_the_route(name: str, arguments: dict):
    request = _Request(name, arguments)
    session = await api_st.require_session(request, db=None)
    response = await api_st.session_tools_mcp(request, session=session, db=None)
    return json.loads(response.body)["result"]


def _timed(name: str, arguments: dict):
    started = time.monotonic()
    result = asyncio.run(_through_the_route(name, arguments))
    return result, time.monotonic() - started


def test_a_slow_document_search_answers_in_words_naming_read_document(route):
    result, elapsed = _timed("search_documents", QUESTION)

    assert result["isError"] is True
    assert "did not answer in time" in result["content"][0]["text"]
    assert "read_document" in result["content"][0]["text"]
    assert elapsed < ANSWERED_WITHIN_S


def test_a_read_still_waiting_for_its_slot_fails_within_the_budget_and_frees_the_slot(route, monkeypatch):
    monkeypatch.setattr(api_st.config, "SESSION_TOOL_CONCURRENCY", 1, raising=False)

    async def scenario():
        loop = asyncio.get_running_loop()
        async with session_tool_slot():                      # another call holds the only slot
            started = loop.time()
            result = await _through_the_route("read_document", {"filename": "x.csv"})
            waited = loop.time() - started
        async with asyncio.timeout(DEADLINE_S):                # the slot is free again at once
            async with session_tool_slot():
                pass
        return result, waited

    result, waited = asyncio.run(scenario())

    assert result["isError"] is True and "did not answer in time" in result["content"][0]["text"]
    assert waited < ANSWERED_WITHIN_S and route["ran"] == []


def test_the_budget_counts_from_the_calls_arrival(route, monkeypatch):
    allowance_s = 0.15
    route["tool_s"] = 0.1                                       # in time on its own, late after the allowance

    def slow_allowance(db, task):
        time.sleep(allowance_s)

    monkeypatch.setattr(api_st, "call_allowance", slow_allowance)
    result, elapsed = _timed("search_knowledge", QUESTION)

    assert result["isError"] is True and "Memory search did not answer in time" in result["content"][0]["text"]
    assert elapsed < allowance_s + route["tool_s"] + 0.05


def test_a_read_that_answers_in_time_is_untouched(route):
    route["tool_s"] = 0.01

    result, _elapsed = _timed("search_documents", QUESTION)

    assert result["isError"] is False and "Harbour Blend" in result["content"][0]["text"]


def test_only_reads_have_a_deadline():
    assert set(DEADLINED_TOOLS) == {"search_documents", "search_knowledge", "read_document"}
    assert all(st.get_tool(name).reads_only for name in DEADLINED_TOOLS)
    assert not set(DEADLINED_TOOLS) & set(WRITES)
    assert all(not st.get_tool(name).reads_only for name in WRITES)


def test_a_slow_write_is_never_cut_off(route, monkeypatch):
    monkeypatch.setattr(api_st.config, "SESSION_TOOL_READ_DEADLINE_S", 0.05, raising=False)
    route["tool_s"] = 0.2

    result, _elapsed = _timed("generate_document", {"title": "Quote", "format": "pdf", "data": {"total": 1}})

    assert result["isError"] is False and route["ran"] == ["generate_document"]
