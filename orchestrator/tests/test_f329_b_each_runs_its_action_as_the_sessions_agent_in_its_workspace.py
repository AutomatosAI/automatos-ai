"""F329 (B): each data tool runs the API agents' own action, through the executor's
dispatcher, as the session's agent in the session's workspace.

Whatever workspace, agent or extra field a call sends is not what runs: the
executor is handed the ticket's ``workspace_id`` and ``agent_id`` (it mints the
action's ``_agent_id`` from that, which is also what scopes the graph to the
agent's team), and only the fields the tool's schema declares are forwarded.
"""
from __future__ import annotations

import asyncio

import pytest

from services import session_tools as st
from services import session_tools_rpc as rpc

CTX = st.SessionContext(task_id=1995, agent_id=268, agent_name="Business Analyst", workspace_id="ws-shop")
SMUGGLED = {"workspace_id": "ws-somebody-else", "agent_id": 1, "_agent_id": 1, "_workspace_id": "x",
            "task_id": 7}


class _Executor:
    """The API agents' executor, recording what it was asked to run."""

    calls: list = []
    answer: dict = {"success": True, "answer": "ok"}

    def __init__(self, db):
        self.db = db

    async def execute_tool(self, **kwargs):
        type(self).calls.append(kwargs)
        return dict(type(self).answer)


@pytest.fixture
def executor(monkeypatch):
    from modules.tools.execution import unified_executor

    monkeypatch.setattr(_Executor, "calls", [])
    monkeypatch.setattr(unified_executor, "UnifiedToolExecutor", _Executor)
    return _Executor


def _call(name, arguments):
    tool = st.get_tool(name)
    return asyncio.run(st.call_tool(None, tool, st.resolve_parameters(tool, arguments, CTX), CTX))


def _ran(executor):
    (only,) = executor.calls
    assert only["tool_name"] == st.PLATFORM_DISPATCHER                 # the one dispatcher, no second path
    assert only["workspace_id"] == "ws-shop" and only["agent_id"] == 268  # the ticket's, never the call's
    assert only["caller_context"] is None                               # an agent, not a human operator
    return only["parameters"]


def test_query_database_asks_the_question_of_the_sessions_own_database(executor):
    _call("query_database", {"question": "How many active subscribers?", **SMUGGLED})
    assert _ran(executor) == {"action": "platform_query_data",
                              "params": {"question": "How many active subscribers?"}}
    assert executor.calls[0]["trace_id"] == "session:1995:query_database"


def test_a_named_database_is_passed_as_the_actions_own_field(executor):
    _call("query_database", {"question": "Top products?", "database": "harbourline_shop"})
    assert _ran(executor)["params"] == {"question": "Top products?", "database_id": "harbourline_shop"}


@pytest.mark.parametrize("arguments, action, params", [
    ({"question": "who supplies the coffee?"}, "platform_query_graph", {"question": "who supplies the coffee?"}),
    ({"concept": "House Blend"}, "platform_graph_neighbors", {"concept": "House Blend"}),
    ({"concept": "House Blend", "relation": "part_of"}, "platform_graph_neighbors",
     {"concept": "House Blend", "relation_filter": "part_of"}),
    ({"from": "Supplier A", "to": "House Blend"}, "platform_graph_path",
     {"source": "Supplier A", "target": "House Blend"}),
])
def test_query_graph_runs_the_lookup_it_was_given(executor, arguments, action, params):
    _call("query_graph", {**arguments, **SMUGGLED})
    assert _ran(executor) == {"action": action, "params": params}
    assert executor.calls[0]["trace_id"] == "session:1995:query_graph"


@pytest.mark.parametrize("name, arguments, words", [
    ("query_database", {}, "question in plain words"),
    ("query_database", {"question": "   ", "sql": "SELECT 1"}, "question in plain words"),
    ("query_graph", {}, "ONE of"),
    ("query_graph", {"question": "suppliers?", "concept": "House Blend"}, "ONE of"),
    ("query_graph", {"from": "Supplier A"}, "both 'from' and 'to'"),
])
def test_a_call_it_cannot_run_is_refused_in_words_and_runs_nothing(executor, name, arguments, words):
    with pytest.raises(st.SessionToolRefused) as refused:
        st.resolve_parameters(st.get_tool(name), arguments, CTX)
    assert words in str(refused.value)
    assert executor.calls == []


def test_over_the_wire_the_answer_reaches_the_session(executor, monkeypatch):
    monkeypatch.setattr(_Executor, "answer", {"success": True, "answer": "active | 383", "row_count": 1})

    async def call(tool, arguments, ctx):
        return await st.call_tool(None, tool, arguments, ctx)

    reply = asyncio.run(rpc.handle_message(
        {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
         "params": {"name": "query_database", "arguments": {"question": "How many active?", **SMUGGLED}}},
        CTX, server_version="1.0", call=call))
    assert reply["result"]["isError"] is False
    assert "active | 383" in reply["result"]["content"][0]["text"]
    assert _ran(executor)["params"] == {"question": "How many active?"}


def test_a_ticket_with_no_agent_runs_neither(executor):
    nobody = st.SessionContext(task_id=1995, agent_id=None, agent_name=None, workspace_id="ws-shop")
    for name, arguments in (("query_database", {"question": "q"}), ("query_graph", {"concept": "c"})):
        tool = st.get_tool(name)
        with pytest.raises(st.SessionToolRefused):
            asyncio.run(st.call_tool(None, tool, st.resolve_parameters(tool, arguments, nobody), nobody))
    assert executor.calls == []
