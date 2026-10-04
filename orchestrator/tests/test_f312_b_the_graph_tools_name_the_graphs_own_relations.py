"""F312 (night 9) — the graph tools take and follow the relations the Knowledge Graph holds.

platform_graph_neighbors offered a relation_filter of "depends_on, implements, triggers, measures,
constrained_by, semantically_similar_to, conceptually_related_to", and platform_graph_impact
walked only depends_on, implements, constrained_by, triggers, measures, semantically_similar_to
and conflicts_with. Extraction snaps every edge to its own vocabulary (part_of, supplies, buys,
uses, ...: FIXER's fix/f312-graph-relations adds supplies, buys, responsible_for and
substitutes_for), so the filter named relations no edge has, and "what is affected if Brazil
Cerrado is late?" followed none of "Brazil Cerrado part_of Harbour Blend" or "Kiln Bakehouse buys
Harbour Blend".
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

import networkx as nx

from modules.tools.discovery import get_action_registry
from modules.tools.discovery.actions_data_routes import GRAPH_RELATIONS
from modules.tools.discovery.handlers_graph import handle_graph_impact

FIXERS_RELATIONS = ("supplies", "buys", "responsible_for", "substitutes_for")


def _extraction_vocabulary():
    """The relations extraction snaps an edge to, wherever this build keeps them."""
    try:
        from modules.knowledge.graph_relations import CANONICAL_RELATIONS
    except ImportError:  # before fix/f312-graph-relations moved the vocabulary out
        from modules.knowledge.graph_extraction import CANONICAL_RELATIONS
    return CANONICAL_RELATIONS


def test_the_neighbors_filter_takes_exactly_the_graphs_relations():
    neighbors = get_action_registry().get("platform_graph_neighbors")
    relation_filter = neighbors.parameters["properties"]["relation_filter"]
    assert relation_filter["enum"] == list(GRAPH_RELATIONS)
    assert set(_extraction_vocabulary()) <= set(GRAPH_RELATIONS)
    assert set(FIXERS_RELATIONS) <= set(GRAPH_RELATIONS)
    text = relation_filter["description"] + neighbors.description
    for stale in ("implements", "constrained_by", "semantically_similar_to", "conceptually_related_to"):
        assert stale not in text


def test_the_impact_tool_names_the_relations_it_follows():
    impact = get_action_registry().get("platform_graph_impact")
    assert "part_of" in impact.description and "buys" in impact.description
    assert "implements" not in impact.description and "constrained_by" not in impact.description
    assert "which customers are affected if this delivery is late?" in impact.examples


def test_a_late_coffee_reaches_the_blend_and_the_cafe_that_buys_it(monkeypatch):
    graph = nx.Graph()
    for node, label in (("cerrado", "Brazil Cerrado"), ("harbour", "Harbour Blend"), ("kiln", "Kiln Bakehouse"),
                        ("note", "Cafe notes"), ("meridian", "Meridian Bean Traders")):
        graph.add_node(node, label=label)
    graph.add_edge("cerrado", "harbour", relation="part_of")
    graph.add_edge("kiln", "harbour", relation="buys")
    graph.add_edge("meridian", "cerrado", relation="supplies")
    graph.add_edge("note", "kiln", relation="references")        # a mention carries no effect

    async def _load(workspace_id):
        return graph

    monkeypatch.setattr("modules.tools.discovery.handlers_graph._get_service", lambda: NS(load_graph=_load))
    monkeypatch.setattr("modules.tools.discovery.handlers_graph._get_filtered_graph", lambda whole, team: whole)
    result = asyncio.run(handle_graph_impact(None, uuid4(), {"concept": "Brazil Cerrado"}))
    assert result["success"] is True
    reached = {node["label"]: layer["depth"] for layer in result["impact_layers"] for node in layer["nodes"]}
    assert reached == {"Harbour Blend": 1, "Meridian Bean Traders": 1, "Kiln Bakehouse": 2}
