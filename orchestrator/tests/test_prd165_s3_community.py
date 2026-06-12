"""PRD-165 S3 — build-time community reports (titles/summaries/rank).

The LLM is injected (a fake), so titling is deterministic and offline. Covers:
ranking by size for every community, titling only the top-N, graceful
degradation when the LLM fails, tolerant JSON parsing, and the merge into the
exported communities.json shape.
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

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from modules.knowledge import community_reports  # noqa: E402
from modules.knowledge.community_reports import generate_community_reports, _parse_report  # noqa: E402
from modules.knowledge.graph_service import GraphifyService  # noqa: E402


class _FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    async def generate_response(self, _messages):
        self.calls += 1
        return type("R", (), {"content": self.content})()


def _graph_and_communities():
    g = nx.Graph()
    for i in range(6):
        g.add_node(f"n{i}", label=f"Node{i}")
    g.add_edge("n0", "n1")
    g.add_edge("n1", "n2")
    g.add_edge("n3", "n4")
    communities = {0: ["n0", "n1", "n2"], 1: ["n3", "n4"], 2: ["n5"]}
    return g, communities


def _no_db_settings(monkeypatch):
    # Keep titling offline — never touch system_settings / the DB.
    monkeypatch.setattr(community_reports, "_get_setting", lambda _k, default: default)


def test_ranks_every_community_titles_only_top_n(monkeypatch):
    _no_db_settings(monkeypatch)
    g, comms = _graph_and_communities()
    llm = _FakeLLM('{"title": "Cluster Name", "summary": "A short summary."}')

    reports = asyncio.run(generate_community_reports(g, comms, top_n=2, member_cap=10, llm=llm))

    # Rank by size, every community: 0(3) -> 0, 1(2) -> 1, 2(1) -> 2.
    assert reports[0]["rank"] == 0
    assert reports[1]["rank"] == 1
    assert reports[2]["rank"] == 2
    # Only the top-2 get an LLM title.
    assert reports[0]["title"] == "Cluster Name"
    assert reports[1]["title"] == "Cluster Name"
    assert "title" not in reports[2]
    assert llm.calls == 2


def test_top_n_zero_is_ranks_only(monkeypatch):
    _no_db_settings(monkeypatch)
    g, comms = _graph_and_communities()
    llm = _FakeLLM('{"title": "x"}')
    reports = asyncio.run(generate_community_reports(g, comms, top_n=0, member_cap=10, llm=llm))
    assert all("title" not in r for r in reports.values())
    assert llm.calls == 0


def test_degrades_to_ranks_on_llm_error(monkeypatch):
    _no_db_settings(monkeypatch)
    g, comms = _graph_and_communities()

    class _BoomLLM:
        async def generate_response(self, _m):
            raise RuntimeError("boom")

    reports = asyncio.run(generate_community_reports(g, comms, top_n=5, member_cap=10, llm=_BoomLLM()))
    assert reports[0]["rank"] == 0
    assert all("title" not in r for r in reports.values())  # never raised, just no titles


def test_parse_report_tolerant():
    assert _parse_report('```json\n{"title":"T","summary":"S"}\n```') == {"title": "T", "summary": "S"}
    assert _parse_report('Here: {"title":"T"} done')["title"] == "T"
    assert _parse_report("no json") is None
    # A response with no title is unusable.
    assert _parse_report('{"summary":"only summary"}') is None


def test_format_communities_merges_reports():
    comms = {0: ["a", "b"], 1: ["c"]}
    reports = {0: {"rank": 0, "title": "Top", "summary": "Sum"}, 1: {"rank": 1}}
    out = GraphifyService._format_communities(comms, reports)
    by_id = {c["community_id"]: c for c in out}
    assert by_id[0]["title"] == "Top"
    assert by_id[0]["summary"] == "Sum"
    assert by_id[0]["rank"] == 0
    assert "title" not in by_id[1]  # ranked but untitled
    assert by_id[1]["rank"] == 1


def test_format_communities_without_reports_is_unchanged():
    out = GraphifyService._format_communities({0: ["a", "b"]})
    assert out == [{"community_id": 0, "member_count": 2, "members": ["a", "b"]}]
