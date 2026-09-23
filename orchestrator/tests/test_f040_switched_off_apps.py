"""F040 — an app switched off for an agent stays off, for seeing AND executing.

Night 1: OPS (agent 267) had Gmail and Calendar switched off (is_active=False)
and a ticket of its still sent mail through Gmail. Every site asked "does the
agent have an ACTIVE assignment?" and, when it had none, inherited every app the
workspace had connected — so switching an agent's last app off handed it every
app instead. Live: OPS GMAIL(off) GOOGLECALENDAR(off), TRACKER GMAIL(off).
"""
from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.composio import agent_apps

OPS, NEVER_CONFIGURED, WRITER = 267, 900, 58
STATE = {
    OPS: (set(), {"GMAIL", "GOOGLECALENDAR"}),          # every app switched off
    NEVER_CONFIGURED: (set(), set()),                    # no rows at all
    WRITER: ({"COMPOSIO_SEARCH"}, set()),
}
CONNECTED = ["GMAIL", "GOOGLECALENDAR", "COMPOSIO_SEARCH", "SLACK"]


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(agent_apps, "assignment_state", lambda db, agent_id: STATE[agent_id])


@pytest.fixture
def connected(monkeypatch):
    from core.composio.entity_manager import EntityManager

    monkeypatch.setattr(EntityManager, "get_entity_by_workspace", lambda self, ws: {"id": 1})
    monkeypatch.setattr(EntityManager, "get_entity_connections",
                        lambda self, eid: [{"app_name": a, "status": "active"} for a in CONNECTED])


class _Query:
    def filter(self, *_a):
        return self

    def first(self):
        return None                                      # no ACTIVE assignment for the app asked

    def all(self):
        return []

    def get(self, _id):
        return None


class _Db:
    def query(self, *_a):
        return _Query()


def test_the_rule(state):
    assert agent_apps.inherits_workspace_apps(None, NEVER_CONFIGURED)
    assert not agent_apps.inherits_workspace_apps(None, OPS)        # configured — to nothing
    assert not agent_apps.inherits_workspace_apps(None, WRITER)
    assert agent_apps.switched_off(None, OPS, "gmail") and not agent_apps.switched_off(None, OPS, "SLACK")


# ── executing ───────────────────────────────────────────────────────────────

def _execute(monkeypatch, agent_id, app):
    from core.composio.entity_manager import EntityManager
    from core.composio.tool_executor import ComposioToolExecutor

    monkeypatch.setattr(EntityManager, "get_connected_apps", lambda self, ws: set())   # stop at the next gate
    ex = ComposioToolExecutor(db=_Db())
    return asyncio.run(ex.execute(f"{app}_DO_THING", {}, agent_id=agent_id, workspace_id=uuid4(), app_name=app))


def test_an_app_switched_off_for_the_agent_is_refused_with_that_reason(state, monkeypatch):
    out = _execute(monkeypatch, OPS, "GMAIL")
    assert out["success"] is False and out["error_type"] == "composio_app_switched_off"
    assert "switched off" in out["error"] and "GMAIL" in out["error"]


def test_an_agent_with_every_app_off_inherits_nothing(state, monkeypatch):
    out = _execute(monkeypatch, OPS, "SLACK")            # connected in the workspace, never assigned
    assert out["error_type"] == "composio_not_assigned"


def test_an_agent_never_configured_still_inherits(state, monkeypatch):
    out = _execute(monkeypatch, NEVER_CONFIGURED, "SLACK")
    assert out["error_type"] == "composio_not_connected"   # past the assignment gate, stopped at connection


# ── seeing ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("service_path", [
    "modules.tools.services.composio_tool_service.ComposioToolService",
    "modules.tools.services.composio_hint_service.ComposioHintService",
])
def test_discovery_shows_an_all_off_agent_nothing_and_a_new_agent_everything(state, connected, service_path):
    module, name = service_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    service = cls.__new__(cls)
    service.db = _Db()
    ws = uuid4()
    assert service._resolve_allowed_apps(OPS, ws) == []
    assert sorted(service._resolve_allowed_apps(NEVER_CONFIGURED, ws)) == sorted(CONNECTED)


def test_the_tool_registry_refuses_composio_to_an_all_off_agent(state, monkeypatch):
    from core.composio.entity_manager import EntityManager
    from modules.tools.registry.tool_registry import get_tool_registry

    monkeypatch.setattr(EntityManager, "get_connected_apps", lambda self, ws: set(CONNECTED))
    registry = get_tool_registry()
    ok, reason = registry.validate_tool_access(OPS, "composio_execute", db=_Db(), workspace_id=uuid4())
    assert (ok, reason) == (False, "Every app assigned to this agent is switched off")
    ok, _ = registry.validate_tool_access(NEVER_CONFIGURED, "composio_execute", db=_Db(), workspace_id=uuid4())
    assert ok is True


def test_the_router_does_not_describe_an_all_off_agent_with_workspace_apps(state, connected):
    from core.routing.engine import UniversalRouter

    router = UniversalRouter.__new__(UniversalRouter)
    router._db = _Db()
    ws = uuid4()
    ops = NS(id=OPS, name="OPS", description="", tags=[], workspace_id=ws)
    fresh = NS(id=NEVER_CONFIGURED, name="NEW", description="", tags=[], workspace_id=ws)
    described = {d["agent_id"]: d["apps"] for d in router._build_agent_descriptions([ops, fresh])}
    assert described[OPS] == [] and sorted(described[NEVER_CONFIGURED]) == sorted(CONNECTED)


# ── the query itself, on Postgres ───────────────────────────────────────────

_DROP_TEMP = "DROP TABLE IF EXISTS pg_temp.agent_app_assignments"


@pytest.fixture
def db(test_engine):
    # A temp table shadows the real one on this connection (pg_temp is searched
    # first); dropped schema-qualified on the way in and out.
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text(_DROP_TEMP))
        session.execute(text("CREATE TEMP TABLE agent_app_assignments (id serial, agent_id int, "
                             "app_name varchar, app_type varchar, is_active boolean)"))
        session.execute(text("INSERT INTO agent_app_assignments (agent_id, app_name, app_type, is_active) VALUES "
                             "(267, 'GMAIL', 'EXTERNAL', false), (267, 'googlecalendar', 'EXTERNAL', false), "
                             "(58, 'COMPOSIO_SEARCH', 'EXTERNAL', true), (58, 'GMAIL', 'EXTERNAL', false)"))
        yield session
        session.rollback()
        session.execute(text(_DROP_TEMP))
        session.commit()
        session.close()


def test_assignment_state_reads_on_and_off_rows_on_postgres(db):
    assert agent_apps.assignment_state(db, 267) == (set(), {"GMAIL", "GOOGLECALENDAR"})
    assert agent_apps.assignment_state(db, 58) == ({"COMPOSIO_SEARCH"}, {"GMAIL"})
    assert agent_apps.inherits_workspace_apps(db, 999) is True
    assert agent_apps.inherits_workspace_apps(db, 267) is False
    assert agent_apps.switched_off(db, 267, "GoogleCalendar") is True
