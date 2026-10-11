"""F302 / F312 (night 9) — every chat turn in a workspace with a database holds platform_query_data,
and one with a Knowledge Graph holds platform_query_graph.

Night 9: the prompt sent number questions to smart_query_database, which the chat surface kept
only when the words matched the router's data patterns. "Were any Harvest Club boxes late going
out in September?" and "Have we got enough Guji for October's club boxes?" matched none: Auto had
no data tool, called the mission field, a Shopify sync or an invented action, or asked the owner
for "the exact names of the fields". The same question went two ways in two fresh chats.
platform_query_graph was never called: it was only an enum entry.

These drive Auto's chat tool loading (ToolsSection, FILTERED, as the chat runs it) over a
surface shaped like the one the chat builds, with the night's own questions, against a real
``database_knowledge_sources`` row; the Knowledge Graph and the plan tier are the only fakes.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

import networkx as nx
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.context.sections.tools import ToolLoadingStrategy, ToolsSection

NIGHT_9_QUESTIONS = (
    "Were any Harvest Club boxes late going out in September?",              # L66: no data tool, then Shopify
    "Have we got enough Guji for October's club boxes?",                      # L92: asked the owner for numbers
    "How many Harvest Club members cancelled between April and September, and what's the most common reason?",
    "If Meridian's Brazil Cerrado is late, which cafés' orders are at risk?",  # L68: asked which system
    "Yes, check the shop system.",                                            # L91: the Shopify sync
)


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"{name} tool",
                                             "parameters": {"type": "object", "properties": {}}}}


def _dispatcher():
    schema = _tool("platform_execute")
    schema["function"]["parameters"]["properties"]["action"] = {
        "type": "string", "enum": ["platform_query_data", "platform_query_graph", "platform_list_agents"]}
    return schema


SURFACE = [_dispatcher(), _tool("search_knowledge"), _tool("smart_query_database"), _tool("query_database"),
           _tool("platform_find_tools"), _tool("platform_field_query")]


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text("DROP TABLE IF EXISTS pg_temp.database_knowledge_sources"))
        session.execute(text("CREATE TEMP TABLE database_knowledge_sources "
                             "(LIKE public.database_knowledge_sources INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        session.execute(text("DROP TABLE IF EXISTS pg_temp.database_knowledge_sources"))
        session.commit()
        session.close()


@pytest.fixture
def graph(monkeypatch):
    """The workspace's Knowledge Graph: three nodes unless a test empties it."""
    held = {"graph": nx.Graph()}
    held["graph"].add_edges_from([("brazil-cerrado", "harbour-blend"), ("harbour-blend", "kiln-bakehouse")])

    async def _load(workspace_id):
        return held["graph"]

    monkeypatch.setattr("modules.knowledge.graph_service.get_graph_service", lambda: NS(load_graph=_load))
    return held


def _connect(db, ws, name="harbourline_shop", active=True):
    db.execute(text("INSERT INTO database_knowledge_sources "
                    "(id, workspace_id, tenant_id, name, credential_id, dialect, is_active, created_at) "
                    "VALUES (36, CAST(:ws AS uuid), 1, :name, 1, 'postgresql', :active, now())"),
               {"ws": ws, "name": name, "active": active})


def _surface(db, ws, question):
    tools, _choice = asyncio.run(ToolsSection().load_tools(
        agent_id=None, workspace_id=ws, strategy=ToolLoadingStrategy.FILTERED, db_session=db,
        query=question, prebuilt_tools=[dict(t) for t in SURFACE],
    ))
    return [t["function"]["name"] for t in tools]


@pytest.mark.parametrize("question", NIGHT_9_QUESTIONS)
def test_every_shop_question_holds_the_database_route_and_only_that_one(db, graph, question):
    ws = str(uuid4())
    _connect(db, ws)
    names = _surface(db, ws, question)
    assert names.count("platform_query_data") == 1
    assert "smart_query_database" not in names and "query_database" not in names
    assert "platform_query_graph" in names


def test_without_a_database_the_surface_is_what_the_router_kept(db, graph):
    ws = str(uuid4())
    _connect(db, ws, active=False)                       # a switched-off source is no database
    question = "How many Harvest Club members cancelled between April and September?"
    names = _surface(db, ws, question)
    assert "platform_query_data" not in names
    assert "smart_query_database" in names               # the router's data tool, untouched


def test_an_empty_graph_attaches_no_graph_route(db, graph):
    ws = str(uuid4())
    _connect(db, ws)
    graph["graph"] = nx.Graph()
    names = _surface(db, ws, "Were any Harvest Club boxes late going out in September?")
    assert "platform_query_graph" not in names and "platform_query_data" in names


def test_a_tier_without_the_data_family_keeps_its_surface(db, graph, monkeypatch):
    """The route clears the plan tier's families like its enum entry: none attached, and the
    surface keeps the tool it had."""
    ws = str(uuid4())
    _connect(db, ws)
    monkeypatch.setattr("modules.tools.tool_router._apply_tier_exposure",
                        lambda session, workspace_id, tools, trace: [
                            t for t in tools if t["function"]["name"] != "platform_query_data"])
    names = _surface(db, ws, "How many Harvest Club members cancelled between April and September?")
    assert "platform_query_data" not in names
    assert "smart_query_database" in names


def test_the_route_tells_the_model_how_to_ask_and_never_to_ask_for_fields(db, graph):
    ws = str(uuid4())
    _connect(db, ws)
    tools, _choice = asyncio.run(ToolsSection().load_tools(
        agent_id=None, workspace_id=ws, strategy=ToolLoadingStrategy.FILTERED, db_session=db,
        query="Have we got enough Guji for October's club boxes?", prebuilt_tools=[dict(t) for t in SURFACE],
    ))
    [route] = [t for t in tools if t["function"]["name"] == "platform_query_data"]
    description = route["function"]["description"]
    assert "read the schema it returns" in description
    assert "never ask the user for table, column or schema names" in description
    assert "one figure per call" in description and "a group's count is never the total" in description
    assert route["function"]["parameters"]["required"] == ["question"]


def test_the_lightweight_lane_holds_the_routes_too(db, graph, monkeypatch):
    """Night 9's 5b284d48 ran in the ATOM lane with the dispatcher alone: ten calls over 55 s,
    platform_field_query three times and an invented platform_smart_query_database first."""
    import consumers.chatbot.service as svc

    ws = str(uuid4())
    _connect(db, ws)

    async def _no_memory(*_args, **_kwargs):
        return ""

    monkeypatch.setattr(svc, "atom_memory_block", _no_memory)
    monkeypatch.setattr(svc, "atom_system_prompt", lambda *_args, **_kwargs: "atom prompt")
    monkeypatch.setattr("modules.context.sections.product_facts.product_facts", lambda _db, _ws: "")
    lane = NS(workspace_id=ws, db=db, widget_mode=False,
              prompt_analyzer=NS(convert_to_llm_messages=lambda messages, **_kwargs: list(messages)))
    question = [{"role": "user", "content": "Have we got enough Guji for October's club boxes?"}]
    _messages, tools, _none = asyncio.run(svc.StreamingChatService._prepare_atom_path(
        lane, question, NS(agent_id=7, metadata={}), NS(orchestrator=None, get_user_name=lambda: "Gerard"),
        atom_tools=[_dispatcher()],
    ))
    # PRD-256 US-006: the lane carries the pinned first-class tools too (platform_query_data is one of
    # them); the routes are each there once, beside the dispatcher.
    from modules.tools.first_class_tools import pinned_first_class

    names = [t["function"]["name"] for t in tools]
    pinned = [s["function"]["name"] for s in pinned_first_class(ws, db)]
    assert names[0] == "platform_execute" and names[-1] == "platform_query_graph"
    assert names == ["platform_execute", *pinned, "platform_query_graph"] and "platform_query_data" in pinned
    assert len(names) == len(set(names))
