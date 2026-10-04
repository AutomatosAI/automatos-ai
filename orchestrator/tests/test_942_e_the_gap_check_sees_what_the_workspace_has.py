"""#942 (E): the agent page's gap check sees what the workspace has, not only what
the agent's skills say.

F329 (4 Oct): a Business Analyst session answered "There is no database query tool
here" in a workspace with a connected shop database, and its agent page reported no
gap at all. Now a connected database, a built Knowledge Graph (and templates,
playbooks, reports, missions) whose group is off for the agent each come back as a
``kind: "workspace"`` entry with the switch that closes it, for every agent,
including one still on the API runtime, so the page warns before the switch.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from api import agents as agents_api
from services import session_capability_gaps as gaps
from services.session_tool_groups import GROUP_IDS

WS = "ws-shop"


class _Row:
    pass


class _DB:
    """A workspace that has the tables named in ``present``; counts its reads."""

    def __init__(self, *present):
        self.present = set(present)
        self.reads = []
        self._table = ""

    def query(self, column):
        self._table = getattr(getattr(column, "class_", None), "__tablename__", "")
        return self

    def filter(self, *_clauses):
        return self

    def first(self):
        self.reads.append(self._table)
        return _Row() if self._table in self.present else None

    def execute(self, statement, params=None):
        self._table = "agent_reports" if "agent_reports" in str(statement) else ""
        assert params == {"ws": WS}                                  # workspace-scoped, bound
        return self

    def rollback(self):
        pass


@pytest.fixture
def graph(monkeypatch):
    """A Knowledge Graph that is built (``meta``) or not (``None``)."""
    from modules.knowledge import graph_service

    state = {"meta": None}

    class _Service:
        async def get_meta(self, workspace_id):
            assert workspace_id == WS
            return state["meta"]

    monkeypatch.setattr(graph_service, "get_graph_service", lambda: _Service())
    return state


def _gaps(db, enabled):
    return asyncio.run(gaps.workspace_gaps(db, WS, enabled))


def test_a_connected_database_with_data_off_is_flagged_with_its_fix(graph):
    found = _gaps(_DB("database_knowledge_sources"), ["graph"])
    assert found == [{
        "kind": "workspace", "capability": "database", "group": "data",
        "message": gaps.CAPABILITIES[0].message, "fix": {"enable_group": "data"},
    }]
    assert "connected database" in found[0]["message"]


def test_a_built_graph_with_graph_off_is_flagged(graph):
    graph["meta"] = {"node_count": 40}
    found = _gaps(_DB(), ["data"])
    assert [(g["capability"], g["fix"]) for g in found] == [("graph", {"enable_group": "graph"})]


def test_nothing_is_flagged_when_the_groups_are_on_and_nothing_is_read(graph):
    graph["meta"] = {"node_count": 40}
    db = _DB("database_knowledge_sources", "document_templates", "workflow_recipes", "agent_reports",
             "orchestration_runs")
    assert _gaps(db, list(GROUP_IDS)) == []
    assert db.reads == []                                            # a group that is on costs no query


def test_core_only_flags_everything_the_workspace_has_in_display_order(graph):
    graph["meta"] = {"node_count": 40}
    db = _DB("database_knowledge_sources", "workflow_recipes", "orchestration_runs")
    found = _gaps(db, [])
    assert [g["capability"] for g in found] == ["database", "graph", "playbooks", "missions"]
    assert all(g["fix"]["enable_group"] in GROUP_IDS for g in found)


def test_a_check_that_fails_is_left_out_not_raised(graph, monkeypatch):
    async def broken(db, workspace_id):
        raise RuntimeError("database is down")

    first = gaps.CAPABILITIES[0]
    monkeypatch.setattr(gaps, "CAPABILITIES", (gaps.Capability(first.capability, first.group, first.message, broken),))
    assert _gaps(_DB(), []) == []


def test_skill_gaps_are_marked_skill_and_come_for_an_api_agent_too():
    skill = SimpleNamespace(id=5, name="shop-analysis", description="Reads the shop.", is_active=True,
                            prompt_template="Call platform_query_data with the question.", content_hash=None,
                            tools_schema=None)
    api_agent = SimpleNamespace(id=268, name="Business Analyst", configuration={"runtime": "api"}, skills=[skill])
    assert agents_api._session_tool_gaps(api_agent) == [{
        "skill": "shop-analysis", "tools": [], "instead": {"platform_query_data": "query_database"}, "kind": "skill",
    }]
    # with Data off, the same skill names a tool its sessions would not have
    off = agents_api._session_tool_gaps(api_agent, ["graph"])
    assert off == [{"skill": "shop-analysis", "tools": ["platform_query_data"], "kind": "skill"}]
