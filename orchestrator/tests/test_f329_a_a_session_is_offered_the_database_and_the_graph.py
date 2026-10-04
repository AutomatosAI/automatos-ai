"""F329 (A): a ticket session is offered the owner's database and the Knowledge Graph.

Smoke ticket #1995 (B6 → Business Analyst) ran as a Claude Code session and answered
"There is no database query tool here": PRD-245's fixed list had neither tool, so no
data-night ticket could run on a session agent. The list now ends with
``query_database`` and ``query_graph``, in a fixed order with fixed text (Claude
Code's prompt cache), and everything that tells a session its tools says so.
"""
from __future__ import annotations

import asyncio
import json

from services import session_tools as st
from services import session_tools_rpc as rpc
from services.cli_session_prompt import SESSION_TOOLS_AVAILABLE, session_system_prompt
from tests import test_prd239_session_prompt as prompt_tests

CTX = st.SessionContext(task_id=1995, agent_id=268, agent_name="Business Analyst", workspace_id="ws-shop")
EARLIER = ("board_summary", "list_tasks", "update_ticket", "submit_report", "ask_human",
           "composio_execute", "search_knowledge", "search_documents", "record_memory",
           "read_step_file")


async def _unused(tool, arguments, ctx):
    raise AssertionError("no tool runs in this test")


def test_the_list_ends_with_the_two_data_tools_after_every_earlier_one():
    assert st.tool_names() == EARLIER + ("query_database", "query_graph")
    assert st.get_tool("query_database").action == "platform_query_data"
    assert st.get_tool("query_graph").action == "platform_query_graph"


def test_their_text_is_the_same_on_every_read():
    first = json.dumps(st.definitions(), sort_keys=True)
    assert json.dumps(st.definitions(), sort_keys=True) == first
    by_name = {d["name"]: d for d in st.definitions()}
    database, graph = by_name["query_database"], by_name["query_graph"]
    # what each one is, and when to use it, in words the model reads
    assert "business database" in database["description"] and "plain words" in database["description"]
    assert "schema" in database["description"] and "read-only" in database["description"]
    assert "Knowledge Graph" in graph["description"] and "who supplies, buys" in graph["description"]
    assert database["inputSchema"]["required"] == ["question"]
    assert set(graph["inputSchema"]["properties"]) == {"question", "concept", "relation", "from", "to"}


def test_the_wire_lists_them_and_says_what_the_tools_reach():
    listed = asyncio.run(rpc.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, CTX,
                                            server_version="1.0", call=_unused))
    names = [t["name"] for t in listed["result"]["tools"]]
    assert names[-2:] == ["query_database", "query_graph"]
    hello = asyncio.run(rpc.handle_message({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}},
                                           CTX, server_version="1.0", call=_unused))
    assert "the owner's database" in hello["result"]["instructions"]


def test_a_skill_that_names_the_api_actions_is_pointed_at_the_session_tools():
    assert st.equivalent_of("platform_query_data") == "query_database"
    assert st.equivalent_of("smart_query_database") == "query_database"
    for graph_action in ("platform_query_graph", "platform_graph_neighbors", "platform_graph_path"):
        assert st.equivalent_of(graph_action) == "query_graph", graph_action


def test_the_session_prompt_offers_them_and_drops_the_old_gap_line():
    assert {"query_database", "query_graph"} <= set(SESSION_TOOLS_AVAILABLE)
    body = ("Answer from the shop: call platform_query_data with the question, then "
            "platform_graph_neighbors on the product to see who supplies it.")
    analyst = prompt_tests._agent(skills=[prompt_tests._skill(5, "shop-analysis", "Reads the shop.", body)])
    text = session_system_prompt(analyst)
    tools_line = next(line for line in text.splitlines() if line.startswith("- Automatos:"))
    assert "`query_database`" in tools_line and "`query_graph`" in tools_line
    assert "`query_database` instead of `platform_query_data`" in text
    assert "`query_graph` instead of `platform_graph_neighbors`" in text
    assert "Not available in a session: `platform_query_data`" not in text
