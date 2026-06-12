"""PRD-165 S2 — cluster-first drill-in primitives + platform_graph_path tool.

Pure graph logic over a tiny in-memory NetworkX graph: no DB, no network, no
graph build. Exercises the server-side subgraph extraction (community subgraph,
1-hop expand, shortest path, label search) added to GraphifyService and the
platform_graph_path handler that agents call.
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio

import networkx as nx
import pytest

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from modules.knowledge.graph_service import GraphifyService  # noqa: E402
from modules.tools.discovery import handlers_graph  # noqa: E402


def _svc() -> GraphifyService:
    # Bypass __init__ — the S2 methods only use nx + pure helpers, no instance
    # state (same pattern as test_prd154_s12's CodeGraphService.__new__).
    return GraphifyService.__new__(GraphifyService)


def _graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("a", label="Alpha", file_type="concept", community=1, source_file="doc1.md")
    g.add_node("b", label="Beta", file_type="concept", community=1)
    g.add_node("e", label="Epsilon", file_type="concept", community=1)
    g.add_node("c", label="Gamma", file_type="metric", community=2)
    g.add_node("d", label="Delta", file_type="rule", community=2)
    g.add_edge("a", "b", relation="depends_on", confidence_score=0.9)
    g.add_edge("a", "e", relation="related_to", confidence_score=0.5)
    g.add_edge("b", "c", relation="measures", confidence_score=0.8)
    g.add_edge("c", "d", relation="implements", confidence_score=0.7)
    return g


# ---------------------------------------------------------------------------
# shortest_path
# ---------------------------------------------------------------------------

def test_shortest_path_found_returns_ordered_trail_and_subgraph():
    res = asyncio.run(_svc().shortest_path(_graph(), "a", "d"))
    assert res["found"] is True
    assert res["length"] == 3
    assert [n["label"] for n in res["path"]] == ["Alpha", "Beta", "Gamma", "Delta"]
    # The {nodes, links} subgraph carries exactly the path nodes.
    assert {n["id"] for n in res["nodes"]} == {"a", "b", "c", "d"}
    assert res["nodes"][0]["source_file"] == "doc1.md"  # provenance survives


def test_shortest_path_missing_node():
    res = asyncio.run(_svc().shortest_path(_graph(), "a", "zzz"))
    assert res["found"] is False
    assert "not found" in res["error"].lower()


def test_shortest_path_disconnected_returns_no_path():
    g = _graph()
    g.add_node("island", label="Island")  # no edges
    res = asyncio.run(_svc().shortest_path(g, "a", "island"))
    assert res["found"] is False
    assert "no path" in res["error"].lower()


# ---------------------------------------------------------------------------
# community_subgraph — induced subgraph + honest truncation
# ---------------------------------------------------------------------------

def test_community_subgraph_is_induced():
    data = asyncio.run(_svc().community_subgraph(_graph(), ["a", "b", "e"], max_nodes=300))
    assert {n["id"] for n in data["nodes"]} == {"a", "b", "e"}
    # Only edges with BOTH endpoints in the set (a-b, a-e); b-c is excluded.
    pairs = {frozenset((l["source"], l["target"])) for l in data["links"]}
    assert pairs == {frozenset(("a", "b")), frozenset(("a", "e"))}
    assert data["truncated"] is False


def test_community_subgraph_caps_by_degree_and_flags_truncated():
    data = asyncio.run(_svc().community_subgraph(_graph(), ["a", "b", "e"], max_nodes=2))
    assert len(data["nodes"]) == 2
    assert data["truncated"] is True
    # 'a' (degree 2) survives the degree-ranked cap.
    assert "a" in {n["id"] for n in data["nodes"]}


# ---------------------------------------------------------------------------
# node_neighbors_subgraph — 'expand from here'
# ---------------------------------------------------------------------------

def test_node_neighbors_subgraph_one_hop():
    data = asyncio.run(_svc().node_neighbors_subgraph(_graph(), "a"))
    assert {n["id"] for n in data["nodes"]} == {"a", "b", "e"}


def test_node_neighbors_subgraph_missing_node_is_none():
    assert asyncio.run(_svc().node_neighbors_subgraph(_graph(), "zzz")) is None


# ---------------------------------------------------------------------------
# search_nodes — exact > prefix > substring, ties by degree
# ---------------------------------------------------------------------------

def test_search_nodes_ranks_prefix_over_substring():
    res = asyncio.run(_svc().search_nodes(_graph(), "a", limit=10))
    labels = [m["label"] for m in res]
    # 'Alpha' (prefix) ranks ahead of substring-only matches like 'Beta'/'Gamma'.
    assert labels[0] == "Alpha"
    assert "Beta" in labels  # 'a' is a substring of beta


def test_search_nodes_empty_query():
    assert asyncio.run(_svc().search_nodes(_graph(), "  ")) == []


# ---------------------------------------------------------------------------
# platform_graph_path tool handler
# ---------------------------------------------------------------------------

def test_handle_graph_path_resolves_labels_and_returns_trail(monkeypatch):
    g = _graph()

    class _FakeSvc:
        async def load_graph(self, _ws):
            return g
        async def shortest_path(self, graph, s, t):
            return await _svc().shortest_path(graph, s, t)

    monkeypatch.setattr(handlers_graph, "_get_service", lambda: _FakeSvc())
    monkeypatch.setattr(handlers_graph, "_resolve_agent_team", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_graph, "_get_filtered_graph", lambda graph, _team: graph)

    # Labels (not ids) — agents speak in labels; the handler resolves them.
    res = asyncio.run(handlers_graph.handle_graph_path(None, "ws", {"source": "Alpha", "target": "Delta"}))
    assert res["success"] is True
    assert res["hops"] == 3
    assert res["path"] == ["Alpha", "Beta", "Gamma", "Delta"]


def test_handle_graph_path_missing_source(monkeypatch):
    g = _graph()

    class _FakeSvc:
        async def load_graph(self, _ws):
            return g

    monkeypatch.setattr(handlers_graph, "_get_service", lambda: _FakeSvc())
    monkeypatch.setattr(handlers_graph, "_resolve_agent_team", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_graph, "_get_filtered_graph", lambda graph, _team: graph)
    res = asyncio.run(handlers_graph.handle_graph_path(None, "ws", {"source": "Nope", "target": "Delta"}))
    assert res["success"] is False
    assert "not found" in res["error"].lower()


def test_handle_graph_path_requires_both_args():
    res = asyncio.run(handlers_graph.handle_graph_path(None, "ws", {"source": "Alpha"}))
    assert res["success"] is False
    assert "required" in res["error"].lower()
