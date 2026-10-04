"""F312 (night 9) — Auto is told its workspace has a Knowledge Graph, and which questions go to it.

Night 9: nobody called platform_query_graph all night. G2 "If Meridian's Brazil Cerrado arrives
late, which cafés' orders are at risk?" was answered from one document every time and missed
Kiln Bakehouse (L8, L26, L108). The graph section injects an excerpt only when a node's label
covers 30% of the message's words, which a long question rarely does, and nothing else said the
graph existed; the tool was only a dispatcher enum entry (now attached first-class:
test_f302_b), with a description that pointed numbers at a tool Auto did not hold.

Now, on Auto's chat turns, the section opens with a line naming the graph and the questions
that are its own, whether or not an excerpt follows; and the tool's description says the same.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

import networkx as nx
import pytest

from core.security.surface import WIDGET, turn_surface
from modules.context.sections.base import SectionContext
from modules.context.sections.graph_context import GRAPH_ROUTE_LINE, GraphSection

G2 = "If Meridian's Brazil Cerrado arrives late, which cafés' orders are at risk?"


@pytest.fixture
def graph(monkeypatch):
    """Four things the documents name; the BFS and its text are the service's, faked."""
    held = nx.Graph()
    held.add_node("brazil-cerrado", label="Brazil Cerrado")
    held.add_node("harbour-blend", label="Harbour Blend")
    held.add_node("kiln-bakehouse", label="Kiln Bakehouse")
    held.add_node("meridian", label="Meridian Bean Traders")
    held.add_edges_from([("brazil-cerrado", "harbour-blend"), ("harbour-blend", "kiln-bakehouse"),
                         ("brazil-cerrado", "meridian")])

    async def _load(workspace_id):
        return held

    async def _bfs(whole, node_id, depth):
        return {"nodes": {node_id, *whole.neighbors(node_id)}, "edges": list(whole.edges(node_id))}

    async def _text(whole, nodes, edges, budget):
        return "Brazil Cerrado -> Harbour Blend -> Kiln Bakehouse"

    monkeypatch.setattr("modules.knowledge.graph_service.get_graph_service",
                        lambda: NS(load_graph=_load, bfs=_bfs, subgraph_to_text=_text))
    monkeypatch.setattr("modules.knowledge.graph_service.team_filtered_view", lambda whole, team: whole)
    return held


def _render(message, mode="chatbot"):
    ctx = SectionContext(agent=NS(id=7, team=None), workspace_id=str(uuid4()), context_mode=mode,
                         messages=[{"role": "user", "content": message}])
    return asyncio.run(GraphSection().render(ctx))


def test_a_long_relationship_question_is_told_the_graph_is_there(graph):
    rendered = _render(G2)
    assert rendered.startswith("## Business Context (Knowledge Graph)\n\n")
    assert GRAPH_ROUTE_LINE.format(nodes=4) in rendered
    assert "call platform_query_graph with the question beside search_knowledge" in rendered
    assert "check the live orders or figures it points to with platform_query_data" in rendered


def test_the_line_comes_first_and_an_excerpt_follows_it(graph):
    rendered = _render("Brazil Cerrado blend")             # a node's label covers the message
    route = rendered.index(GRAPH_ROUTE_LINE.format(nodes=4))
    assert route < rendered.index("Brazil Cerrado -> Harbour Blend -> Kiln Bakehouse")


def test_an_agents_task_turn_gets_no_line(graph):
    assert _render(G2, mode="task_execution") == ""
    assert "platform_query_graph" not in _render("Brazil Cerrado blend", mode="task_execution")


def test_a_widget_turn_gets_no_line_even_with_documents_read(graph):
    with turn_surface(WIDGET, ("chat", "documents:read"), None):
        assert _render(G2) == ""


def test_an_empty_graph_says_nothing(monkeypatch):
    async def _load(workspace_id):
        return nx.Graph()

    monkeypatch.setattr("modules.knowledge.graph_service.get_graph_service", lambda: NS(load_graph=_load))
    assert _render(G2) == ""


def test_the_tool_says_which_questions_are_the_graphs():
    from modules.tools.discovery import get_action_registry

    graph_tool = get_action_registry().get("platform_query_graph")
    description = graph_tool.description
    assert "which customers, orders or products are affected if something is late or changes" in description
    assert "who supplies or buys what" in description
    assert "platform_query_data" in description and "smart_query_database" not in description
    assert "which customers are affected if a supplier's delivery is late?" in graph_tool.examples
