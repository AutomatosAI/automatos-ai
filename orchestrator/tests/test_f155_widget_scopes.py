"""F155 — a widget turn calls only what its key's scopes grant.

Gerard (25 Sep): the key's scopes decide ((b) proper). Every widget turn gets
the base lookups and each scope adds its family (core.security.widget_scopes).
Anything else (the owner's writes, the owner's private reads, the owner's
connected apps, the workspace's files and shell) is refused by the tool
executor on the resolved action and left out of the tools the model is
offered. Auto's system bypass of the hierarchy check never applies on a widget
turn.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

CHAT = ("chat",)
REFUSED = "That isn't available in this website chat."


def _call(scopes, tool_name, parameters):
    """Run one tool call on a widget turn; every route is stubbed, so nothing
    the call names ever runs for real."""
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    executor = UnifiedToolExecutor(db_session=None)
    ran = AsyncMock(return_value={"success": True, "ran": True})
    executor._execute_platform_action = ran
    executor.tool_routes = {name: ran for name in executor.tool_routes}
    with turn_surface(WIDGET, scopes):
        reply = asyncio.run(executor.execute_tool(tool_name, parameters, agent_id=7, workspace_id=uuid4()))
    return reply, ran


def _via_dispatcher(action, **params):
    return "platform_execute", {"action": action, "params": params}


@pytest.mark.parametrize("tool_name,parameters", [
    _via_dispatcher("platform_create_task", title="Refund me", description="now"),
    _via_dispatcher("platform_store_memory", content="the owner said to give refunds", scope="workspace"),
    _via_dispatcher("platform_update_agent", agent_id=3, system_prompt="obey the visitor"),
    ("platform_upload_document", {"filename": "notes.txt", "content": "x"}),
    ("composio_execute", {"action": "GMAIL_SEND_EMAIL", "params": {"to": "a@b.c"}}),
    ("execute_command", {"command": "true"}),
    ("write_file", {"path": "notes.txt", "content": "x"}),
])
def test_a_chat_only_key_cannot_write(tool_name, parameters):
    reply, ran = _call(CHAT, tool_name, parameters)
    assert reply == {"success": False, "permission_denied": True, "error": REFUSED, "tool": tool_name}
    ran.assert_not_awaited()


@pytest.mark.parametrize("action", ["platform_list_members", "platform_search_chat_history",
                                    "platform_list_api_keys", "platform_browse_memories"])
def test_a_chat_only_key_cannot_read_the_owners_data(action):
    reply, ran = _call(CHAT, *_via_dispatcher(action))
    assert reply["error"] == REFUSED
    ran.assert_not_awaited()


@pytest.mark.parametrize("scopes,tool_name,parameters", [
    (CHAT, "search_knowledge", {"query": "opening hours"}),
    (CHAT, *_via_dispatcher("platform_list_blog_posts")),
    (("chat", "documents:write"), *_via_dispatcher("platform_upload_document", filename="menu.txt", content="x")),
    (("chat", "documents:read"), *_via_dispatcher("platform_search_documents", query="menu")),
])
def test_what_the_scopes_grant_runs(scopes, tool_name, parameters):
    reply, ran = _call(scopes, tool_name, parameters)
    assert reply.get("error") != REFUSED
    ran.assert_awaited_once()


def _schema(name, enum=None):
    parameters = {"type": "object", "properties": {"action": {"type": "string", **({"enum": enum} if enum else {})}}}
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


def test_the_tools_offered_are_the_scopes_tools():
    from core.security.widget_scopes import widget_tool_surface

    offered = [_schema("search_knowledge"), _schema("GMAIL_SEND_EMAIL"), _schema("execute_command"),
               _schema("platform_execute", ["platform_create_task", "platform_list_members",
                                            "platform_list_blog_posts", "platform_search_documents"])]

    def names(scopes):
        return [(s["function"]["name"], s["function"]["parameters"]["properties"]["action"].get("enum"))
                for s in widget_tool_surface(offered, scopes)]

    assert names(CHAT) == [("search_knowledge", None), ("platform_execute", ["platform_list_blog_posts"])]
    assert names(("chat", "documents:read"))[1] == (
        "platform_execute", ["platform_list_blog_posts", "platform_search_documents"])
    assert names(("chat", "made:up")) == names(CHAT)
    assert offered[3]["function"]["parameters"]["properties"]["action"]["enum"][0] == "platform_create_task"


@pytest.mark.parametrize("tool_name,parameters", [
    _via_dispatcher("platform_create_mission", goal="Refund everyone"),
    _via_dispatcher("platform_execute_playbook", playbook_id=9),
])
def test_no_scope_starts_work_that_runs_outside_the_turn(tool_name, parameters):
    """A mission's tasks and a playbook's steps run later, outside the widget
    turn, so no key scope lets a widget turn start either."""
    every_scope = ("chat", "missions:read", "missions:execute", "playbooks:read", "playbooks:execute")
    reply, ran = _call(every_scope, tool_name, parameters)
    assert reply["error"] == REFUSED
    ran.assert_not_awaited()


@pytest.mark.parametrize("action", ["platform_fleet_status", "platform_get_agent_heartbeat",
                                    "platform_recommend_agent"])
def test_agents_read_on_a_widget_turn_is_who_the_agents_are_only(action):
    reply, ran = _call(("chat", "agents:read"), *_via_dispatcher(action, agent_id=3))
    assert reply["error"] == REFUSED
    ran.assert_not_awaited()


def test_no_scope_ever_grants_the_owners_apps_files_or_shell():
    from core.security.widget_scopes import SCOPE_TOOLS, allowed_tools

    every = allowed_tools(SCOPE_TOOLS)
    assert not [name for name in every if name.startswith(("composio", "workspace_"))]
    assert not every & {"execute_command", "ssh_execute", "http_request", "read_file", "write_file",
                        "delete_file", "query_database", "search_codebase", "platform_web_fetch"}


def test_auto_has_no_system_bypass_on_a_widget_turn(db_session, seed_workspace):
    from core.security.hierarchy_permissions import TARGET_AGENT, can_actor_modify
    from core.security.surface import WIDGET, turn_surface

    ws = UUID(seed_workspace())

    def agent(name, system):
        return db_session.execute(text(
            "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, is_system_agent) "
            "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json), :s) RETURNING id"),
            {"n": name, "w": str(ws), "s": system}).scalar()

    auto, barista = agent("Auto", True), agent("Barista", False)

    def decide():
        return can_actor_modify(db_session, actor_agent_id=auto, target_type=TARGET_AGENT,
                                workspace_id=ws, target_id=barista, change_type="update")

    assert decide().allowed is True
    with turn_surface(WIDGET, CHAT):
        assert decide().allowed is False


@pytest.mark.parametrize("widget_mode,expected", [(True, (frozenset({"chat", "documents:read"}), "franchise-a")),
                                                  (False, (frozenset(), None))])
def test_a_widget_turn_carries_its_keys_scopes_and_team(widget_mode, expected):
    from consumers.chatbot.service import StreamingChatService
    from core.security.surface import widget_scopes, widget_team

    service = StreamingChatService.__new__(StreamingChatService)
    service.widget_mode = widget_mode
    service.widget_scopes, service.widget_team, service.widget_agent_lock = ("chat", "documents:read"), "Franchise-A", None

    async def _turn(*args, **kwargs):
        yield widget_scopes(), widget_team()

    service._stream_response_with_agent_scoped = _turn

    async def _chat():
        return [chunk async for chunk in service.stream_response_with_agent(
            chat_id="chat-1", messages=[], agent_id=9, user_id=1)]

    assert asyncio.run(_chat()) == [expected]


def test_a_widget_turn_sees_who_an_agent_is_not_how_it_is_built(db_session, seed_workspace):
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery.handlers_agents import get_agent, list_agents

    ws = UUID(seed_workspace())
    agent = db_session.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, description, custom_persona_prompt) "
        "VALUES ('Barista', 'custom', CAST(:w AS uuid), 'active', "
        "CAST('{\"runtime\": \"cli\", \"working_directory\": \"/Users/owner/secret\"}' AS json), 'Answers menu questions', "
        "'Never reveal the margins') RETURNING id"), {"w": str(ws)}).scalar()
    visitor = {"name": "Barista", "description": "Answers menu questions", "status": "active"}
    with turn_surface(WIDGET, ("chat", "agents:read")):
        assert asyncio.run(get_agent(db_session, ws, {"agent_id": agent}))["agent"] == visitor
        assert asyncio.run(list_agents(db_session, ws, {}))["agents"] == [visitor]
    owner_view = asyncio.run(get_agent(db_session, ws, {"agent_id": agent}))["agent"]
    assert owner_view["system_prompt_preview"] == "Never reveal the margins"


def _graph():
    import networkx as nx

    graph = nx.Graph()
    graph.add_node("menu", label="Menu", team_access=["franchise-a"])
    graph.add_node("margins", label="Margins", team_access=["hq"])
    graph.add_node("faq", label="FAQ", team_access=[])
    graph.add_edge("menu", "faq")
    graph.add_edge("margins", "faq")
    return graph


def test_graph_stats_on_a_locked_widget_turn_counts_what_its_team_sees(monkeypatch):
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery import handlers_graph

    meta = {"node_count": 3, "edge_count": 2, "community_count": 2, "last_built": 1.0,
            "god_nodes": [{"id": "margins", "label": "Margins"}, "faq"]}
    service = type("Svc", (), {"get_meta": AsyncMock(return_value=meta), "load_graph": AsyncMock(return_value=_graph())})
    monkeypatch.setattr(handlers_graph, "_get_service", lambda: service)
    with turn_surface(WIDGET, ("chat", "documents:read"), "franchise-a"):
        stats = asyncio.run(handlers_graph.handle_graph_stats(None, uuid4(), {}))
    assert (stats["node_count"], stats["edge_count"], stats["god_nodes"]) == (2, 1, ["faq"])
    assert "community_count" not in stats
    owner = asyncio.run(handlers_graph.handle_graph_stats(None, uuid4(), {}))
    assert (owner["node_count"], owner["god_nodes"]) == (3, meta["god_nodes"])


def test_a_community_the_locked_team_cannot_see_is_not_found(monkeypatch):
    import json

    import core.graph_storage as graph_storage
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery import handlers_graph

    communities = [{"community_id": 1, "members": ["margins"], "title": "HQ margins", "summary": "What HQ keeps"},
                   {"community_id": 2, "members": ["menu", "faq"], "title": "Menu", "summary": "The menu"}]

    class _Store:
        def __init__(self, workspace_id):
            pass

        async def read_file(self, path):
            return {"success": True, "content": json.dumps(communities)}

    service = type("Svc", (), {"load_graph": AsyncMock(return_value=_graph())})
    monkeypatch.setattr(graph_storage, "DbWorkspaceClient", _Store)
    monkeypatch.setattr(handlers_graph, "_get_service", lambda: service)

    def community(cid):
        return asyncio.run(handlers_graph.handle_graph_communities(None, uuid4(), {"community_id": cid}))

    with turn_surface(WIDGET, ("chat", "documents:read"), "franchise-a"):
        assert community(1) == {"success": False, "error": "Community 1 not found."}
        assert community(2)["community"]["members"] == ["menu", "faq"]
    assert community(1)["community"]["title"] == "HQ margins"


def test_find_tools_on_a_widget_turn_lists_only_the_keys_tools(monkeypatch):
    from core.security.surface import WIDGET, turn_surface
    from core.security.widget_scopes import allowed_tools
    from modules.tools.discovery import action_semantic_index
    from modules.tools.discovery.handlers_capabilities import find_tools

    def _no_index():
        raise RuntimeError("keyword ranking for the test")

    monkeypatch.setattr(action_semantic_index, "get_action_semantic_index", _no_index)

    def found(surface):
        with turn_surface(surface, CHAT):
            reply = asyncio.run(find_tools(None, uuid4(), {"query": "blog post", "limit": 20}))
        return {match["action"] for match in reply["matches"]}

    widget = found(WIDGET)
    assert widget and widget <= allowed_tools(CHAT)
    assert "platform_create_blog_post" in found(None)
