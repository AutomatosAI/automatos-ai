"""F155 — a widget turn's prompt carries only what its key may read.

The key's scopes decided what a widget turn could call, but not what its
system prompt told the model. On every widget turn the prompt named up to ten
of the owner's document titles (with no documents:read and past the key's team
lock), counted the reports the agents saved, named the connected databases
(with no data:query), described the owner's connected apps, carried the
owner's onboarding script while the workspace was onboarding, drew the
knowledge graph under the agent's team rather than the key's lock, and listed
the whole platform-action catalog. Now, on a widget turn:

- the documents inventory names the documents only with documents:read (the
  key's team's and the shared ones, never the agents' reports) and the
  databases only with data:query;
- the graph section reads only with documents:read, under the key's lock;
- the connected-apps and onboarding sections are left out;
- the action catalog lists only the actions the key's scopes grant.
"""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.security.surface import WIDGET, turn_surface
from modules.context.sections.base import SectionContext

TABLES = ("documents", "database_knowledge_sources")


def _widget(*scopes, team=None):
    return turn_surface(WIDGET, ("chat", *scopes), team)


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        for table in TABLES:
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{table}"))
            session.execute(text(f"CREATE TEMP TABLE {table} (LIKE public.{table} INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        for table in TABLES:
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{table}"))
        session.commit()
        session.close()


def _doc(db, doc_id, ws, name, teams, *, report=False, day=1):
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, source_type, status, upload_date, team_access) "
                    "VALUES (:id, CAST(:ws AS uuid), :name, :st, 'completed', make_timestamp(2026, 9, :day, 12, 0, 0), "
                    "CAST(:teams AS varchar[]))"),
               {"id": doc_id, "ws": ws, "name": name, "st": "agent_output" if report else None, "day": day,
                "teams": "{" + ",".join(teams) + "}"})


def _database(db, ws, name):
    db.execute(text("INSERT INTO database_knowledge_sources "
                    "(id, workspace_id, tenant_id, name, credential_id, dialect, is_active, created_at) "
                    "VALUES (1, CAST(:ws AS uuid), 1, :name, 1, 'postgresql', true, now())"),
               {"ws": ws, "name": name})


def test_the_inventory_names_only_what_the_widget_key_may_read(db):
    from modules.context.sections.documents_inventory import documents_summary

    ws = str(uuid4())
    _doc(db, 1, ws, "price-list.pdf", [], day=1)
    _doc(db, 2, ws, "franchise-a-rota.pdf", ["franchise-a"], day=2)
    _doc(db, 3, ws, "franchise-b-payroll.xlsx", ["franchise-b"], day=3)
    _doc(db, 4, ws, "q3-board-report.md", [], report=True, day=4)
    _database(db, ws, "Shop orders")

    with _widget("documents:read", team="Franchise-A"):
        assert documents_summary(db, ws) == (
            "## Documents in this workspace\n"
            "This workspace holds 2 of the owner's documents (franchise-a-rota.pdf, price-list.pdf). For a question "
            "about the business, a document, or how the product works, search them with search_knowledge first "
            "and name the file you used.")
    with _widget("data:query", team="Franchise-A"):
        assert documents_summary(db, ws) == (
            "## Documents and data in this workspace\n"
            "This workspace has 1 connected database (Shop orders). For numbers about the business (counts, "
            "totals, rankings, trends), call smart_query_database with the question; with one database no name "
            "is needed.")
    with _widget():
        assert documents_summary(db, ws) is None

    owner = documents_summary(db, ws)
    assert "3 of the owner's documents (franchise-b-payroll.xlsx" in owner
    assert "1 report its agents saved" in owner and "(Shop orders)" in owner


def _graph_render(monkeypatch, ctx):
    import networkx as nx

    loads, teams = [], []
    graph = nx.Graph()
    graph.add_node("margin-target")

    async def _load(workspace_id):
        loads.append(workspace_id)
        return graph

    def _view(whole, team):
        teams.append(team)
        return nx.Graph()

    monkeypatch.setattr("modules.knowledge.graph_service.get_graph_service", lambda: NS(load_graph=_load))
    monkeypatch.setattr("modules.knowledge.graph_service.team_filtered_view", _view)
    from modules.context.sections.graph_context import GraphSection

    return asyncio.run(GraphSection().render(ctx)), loads, teams


def test_the_graph_is_read_only_with_documents_read_and_under_the_keys_lock(monkeypatch):
    ctx = SectionContext(agent=NS(id=7, team="support"), workspace_id=str(uuid4()),
                         messages=[{"role": "user", "content": "What is the margin target?"}])
    with _widget(team="Franchise-A"):
        rendered, loads, _teams = _graph_render(monkeypatch, ctx)
    assert (rendered, loads) == ("", [])
    with _widget("documents:read", team="Franchise-A"):
        _rendered, _loads, teams = _graph_render(monkeypatch, ctx)
    assert teams == ["franchise-a"]
    _rendered, _loads, teams = _graph_render(monkeypatch, ctx)
    assert teams == ["support"]


def test_the_owners_connected_apps_are_not_described_on_a_widget_turn():
    from modules.context.sections.composio import ComposioSection

    db = MagicMock()
    ctx = SectionContext(agent=NS(id=7), workspace_id=str(uuid4()), db_session=db)
    with _widget("documents:read"):
        assert asyncio.run(ComposioSection().render(ctx)) == ""
    db.query.assert_not_called()
    asyncio.run(ComposioSection().render(ctx))
    db.query.assert_called()


def test_a_widget_visitor_is_never_onboarded(monkeypatch):
    from modules.context.sections.onboarding import OnboardingSection

    script = "## Getting started\nAsk what the business sells, then staff the team from the marketplace."
    monkeypatch.setattr(OnboardingSection, "_build", AsyncMock(return_value=script))
    ctx = SectionContext(agent=NS(id=7), workspace_id=str(uuid4()),
                         messages=[{"role": "user", "content": "Help me get started"}])
    with _widget():
        assert asyncio.run(OnboardingSection().render(ctx)) == ""
    assert asyncio.run(OnboardingSection().render(ctx)) == script


def test_a_widget_turns_catalog_lists_only_the_granted_actions():
    from core.security.widget_scopes import allowed_tools
    from modules.context.sections.platform_actions import PlatformActionsSection

    scopes = ("chat", "documents:read")
    ctx = SectionContext(agent=NS(id=7), workspace_id=str(uuid4()), context_mode="chatbot",
                         kwargs={"query": "Create an agent that answers refunds"})
    with turn_surface(WIDGET, scopes, None):
        catalog = asyncio.run(PlatformActionsSection().render(ctx))
    named = set(re.findall(r"`(platform_[a-z_]+)`", catalog)) - {"platform_execute"}  # the dispatcher itself
    assert "platform_list_documents" in named
    assert named <= allowed_tools(scopes)
