"""F312 (build 14) — every edge a reader hands out reads the way it was extracted.

The workspace graph is an undirected ``nx.Graph``: a stored link's ``source`` /
``target`` are networkx's storage order, and the direction extraction gave it is
in ``_src``/``_tgt``. On the build-14 graph TESTER read "harbour_blend -buys->
crane_kitchen" for "Crane Kitchen buys Harbour Blend", and the graph tools gave
each neighbour as a bare "target" with no direction.

The graph here is loaded from the stored graph.json shape exactly as
``load_graph`` loads it, with every link stored the wrong way round.
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402

import networkx as nx  # noqa: E402

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from modules.knowledge.graph_direction import edge_payload  # noqa: E402
from modules.knowledge.graph_service import GraphifyService  # noqa: E402
from modules.tools.discovery import handlers_graph  # noqa: E402

STORED = {
    "directed": False, "multigraph": False, "graph": {},
    "nodes": [
        {"id": "harbour_blend", "label": "Harbour Blend"},
        {"id": "crane_kitchen", "label": "Crane Kitchen"},
        {"id": "brazil_cerrado", "label": "Brazil Cerrado"},
    ],
    "links": [
        {"source": "harbour_blend", "target": "crane_kitchen", "relation": "buys",
         "relation_label": "Thursday: Harbour Blend", "_src": "crane_kitchen", "_tgt": "harbour_blend"},
        {"source": "harbour_blend", "target": "brazil_cerrado", "relation": "depends_on",
         "relation_label": "the blend needs the Brazil", "_src": "brazil_cerrado", "_tgt": "harbour_blend"},
    ],
}


def _stored_graph() -> nx.Graph:
    import copy

    data = GraphifyService._normalize_node_link_data(copy.deepcopy(STORED))
    return nx.node_link_graph(data)


def _svc() -> GraphifyService:
    return GraphifyService.__new__(GraphifyService)


def _serve(monkeypatch, graph: nx.Graph) -> None:
    class _FakeSvc:
        async def load_graph(self, _ws):
            return graph

    monkeypatch.setattr(handlers_graph, "_get_service", lambda: _FakeSvc())
    monkeypatch.setattr(handlers_graph, "_resolve_agent_team", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_graph, "_get_filtered_graph", lambda g, _team: g)


def test_the_graph_neighbors_tool_says_crane_kitchen_buys_the_blend(monkeypatch):
    _serve(monkeypatch, _stored_graph())

    res = asyncio.run(handlers_graph.handle_graph_neighbors(None, "ws", {"concept": "Harbour Blend"}))

    assert res["success"] is True
    buys = next(n for n in res["neighbors"] if n["relation"] == "buys")
    assert (buys["source"], buys["target"]) == ("crane_kitchen", "harbour_blend")
    assert buys["statement"] == "Crane Kitchen buys Harbour Blend"
    assert buys["direction"] == "incoming"
    assert buys["neighbor"] == "crane_kitchen"


def test_the_impact_tool_reads_each_hop_as_extracted(monkeypatch):
    _serve(monkeypatch, _stored_graph())

    res = asyncio.run(handlers_graph.handle_graph_impact(None, "ws", {"concept": "Harbour Blend"}))

    # COPILOT's F312 (night 9) also walks buys/supplies, so the first layer holds the buyer
    # too ("Crane Kitchen buys Harbour Blend"): find the Cerrado hop, wherever it sits.
    hop = next(n for n in res["impact_layers"][0]["nodes"] if n["source"] == "brazil_cerrado")
    assert hop["statement"] == "Brazil Cerrado depends_on Harbour Blend"
    assert (hop["source"], hop["target"]) == ("brazil_cerrado", "harbour_blend")


def test_subgraph_links_for_the_ui_point_from_the_buyer():
    data = asyncio.run(_svc().community_subgraph(_stored_graph(), ["harbour_blend", "crane_kitchen"]))

    (link,) = data["links"]
    assert (link["source"], link["relation"], link["target"]) == ("crane_kitchen", "buys", "harbour_blend")


def test_the_graph_text_agents_read_says_crane_kitchen_buys_the_blend():
    graph = _stored_graph()

    text = asyncio.run(_svc().subgraph_to_text(
        graph, {"harbour_blend", "crane_kitchen"}, [("harbour_blend", "crane_kitchen")], 500,
    ))

    (edge_line,) = [line for line in text.splitlines() if line.startswith("EDGE")]
    assert edge_line.startswith("EDGE Crane Kitchen --buys")
    assert edge_line.endswith("Harbour Blend")


def test_an_edge_without_a_stamped_direction_keeps_its_order():
    link = edge_payload("product_1", "vendor_1", {"relation": "by_vendor"})

    assert (link["source"], link["target"]) == ("product_1", "vendor_1")
