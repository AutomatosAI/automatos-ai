"""#942 (C): a session that calls a tool its agent's groups leave out is refused in
words it can act on, and nothing runs.

The list it was offered never named the tool, but a model can still try a name it
read in a skill. The refusal says which group holds the tool and that the owner
turns it on, and lists what the session does have; the call is not charged as a
run and never reaches the executor.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from services import session_tool_groups as groups
from services import session_tools as st
from services import session_tools_rpc as rpc

AGENT = SimpleNamespace(id=268, name="Business Analyst", configuration={"session_tool_groups": ["graph"]})
CTX = st.SessionContext(task_id=2001, agent_id=268, agent_name="Business Analyst", workspace_id="ws-shop",
                        offered=groups.agent_tool_names(AGENT))


def _call(name, arguments, ran):
    async def call(tool, scoped, ctx):
        ran.append(tool.name)
        return {"success": True, "answer": "ran"}

    reply = asyncio.run(rpc.handle_message(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        CTX, server_version="1.0", call=call))
    return reply["result"], ran


def test_a_tool_in_a_group_that_is_off_is_refused_and_named():
    result, ran = _call("query_database", {"question": "How many orders?"}, [])
    text = result["content"][0]["text"]
    assert result["isError"] is True and ran == []
    assert "not turned on for this agent" in text and "Data tool group" in text
    assert "query_graph" in text and "board_summary" in text       # what it does have
    assert "run_playbook" not in text                              # never offers what it lacks


def test_a_tool_in_a_group_that_is_on_runs():
    result, ran = _call("query_graph", {"concept": "House Blend"}, [])
    assert result["isError"] is False and ran == ["query_graph"]


def test_a_core_tool_always_runs():
    result, ran = _call("board_summary", {}, [])
    assert result["isError"] is False and ran == ["board_summary"]


def test_a_name_no_group_has_keeps_the_old_answer():
    result, _ran = _call("delete_workspace", {}, [])
    assert result["isError"] is True
    assert result["content"][0]["text"].startswith("'delete_workspace' is not a tool this session has.")


def test_the_manifest_and_tools_list_never_name_a_tool_that_is_off():
    names = [d["name"] for d in groups.offered_definitions(CTX)]
    assert "query_graph" in names and "query_database" not in names and "generate_document" not in names
