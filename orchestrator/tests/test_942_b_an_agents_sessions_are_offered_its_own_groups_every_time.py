"""#942 (B): what a session is offered is the agent's own list, the same every time.

``configuration.session_tool_groups`` picks the groups; absent is the fixed default
(all six), ``[]`` is the core tools only. The list is core first, then the groups'
tools in display order, whatever order the owner saved them in, so two sessions of
one agent advertise byte-identical tools (Claude Code's prompt cache) and the list
moves only when the owner changes the agent. The claim's names (the host's MCP
config and gate) and ``tools/list`` read the same list.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from services import session_tool_groups as groups
from services import session_tools as st
from services import session_tools_rpc as rpc

CORE = ("board_summary", "list_tasks", "update_ticket", "submit_report", "ask_human", "composio_execute",
        "search_knowledge", "search_documents", "record_memory", "read_step_file")


def _agent(**config):
    return SimpleNamespace(id=268, name="Business Analyst", configuration={"runtime": "cli", **config})


def _session(agent, task_id):
    return st.SessionContext(task_id=task_id, agent_id=agent.id, agent_name=agent.name, workspace_id="ws-shop",
                             offered=groups.agent_tool_names(agent))


def _listed(ctx):
    async def unused(tool, arguments, context):
        raise AssertionError("nothing runs")

    reply = asyncio.run(rpc.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, ctx,
                                           server_version="1.0", call=unused))
    return reply["result"]["tools"]


def test_the_catalogue_is_the_agreed_six_and_core_is_the_agreed_ten():
    assert groups.GROUP_IDS == ("data", "graph", "documents", "playbooks", "reports", "missions")
    assert groups.DEFAULT_SESSION_TOOL_GROUPS == groups.GROUP_IDS
    assert groups.CORE_TOOLS == CORE
    # every session tool is core or in exactly one group
    grouped = [name for g in groups.SESSION_TOOL_GROUPS for name in g.tools]
    assert sorted(CORE + tuple(grouped)) == sorted(st.tool_names()) and len(set(grouped)) == len(grouped)


def test_absent_is_every_group_and_empty_is_core_only():
    assert groups.agent_tool_names(_agent()) == st.tool_names()
    assert groups.agent_tool_names(_agent(session_tool_groups=[])) == CORE
    assert groups.agent_tool_names(_agent(session_tool_groups=None)) == st.tool_names()


def test_the_order_is_core_then_display_order_whatever_was_saved():
    agent = _agent(session_tool_groups=["missions", "data"])
    assert groups.effective_groups(agent) == ["data", "missions"]
    assert groups.agent_tool_names(agent) == CORE + ("query_database", "list_missions", "get_mission",
                                                     "search_mission_findings")
    assert [t.name for t in groups.session_tools_for_agent(agent)] == list(groups.agent_tool_names(agent))


def test_two_sessions_of_one_agent_are_offered_the_identical_list():
    agent = _agent(session_tool_groups=["data", "graph", "reports"])
    first, second = _listed(_session(agent, 2001)), _listed(_session(agent, 2002))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [t["name"] for t in first] == list(CORE) + ["query_database", "query_graph", "get_latest_report"]


def test_the_list_changes_when_the_owner_changes_the_agents_groups():
    before = _listed(_session(_agent(session_tool_groups=["data"]), 2001))
    after = _listed(_session(_agent(session_tool_groups=["data", "playbooks"]), 2002))
    assert [t["name"] for t in after] == [t["name"] for t in before] + ["list_playbooks", "get_playbook",
                                                                         "run_playbook"]


def test_the_claim_carries_the_agents_own_names(monkeypatch):
    from services import cli_host_service as svc

    agent = _agent(session_tool_groups=["graph"])
    task = SimpleNamespace(id=2001, workspace_id="ws-shop", assigned_agent_id=268, title="t", review_mode="auto",
                           attachment_ids=[])
    ref = {"provider": "claude", "model": "sonnet", "cwd": "/tmp/x", "session_id": "s", "attempt": 1}
    monkeypatch.setattr(svc, "_session_system_prompt", lambda a: "")
    payload = svc._claim_payload(task, agent, agent.configuration, ref, "do it", "tok", ("auto", False))
    assert payload["session_tools"] == list(CORE) + ["query_graph"]


def test_settings_show_the_default_agents_list():
    assert [d["name"] for d in groups.default_definitions()] == list(st.tool_names())
