"""Auto sees the apps the workspace has connected, not only the ones assigned to an agent.

In production Auto told the owner "Auto can see three connected apps: COMPOSIO_SEARCH,
GOOGLE_ANALYTICS and GITHUB. Neither Instagram nor X appears…". The workspace had nine
active Composio connections, and its Socials module had published to Instagram through
one of them minutes earlier. ``platform_list_connected_apps`` listed the apps ASSIGNED to
some agent (``AgentAppAssignment``), so an app connected but assigned to no agent was
missing, and ``composio_execute``'s own description sends the model to that list.

It now reads the workspace's connections through ``EntityManager``, as Socials and
``composio_execute`` do: ``get_connected_apps`` decides which are connected (and settles a
pending row that holds its Composio account id), each app keeps the count of agents that
have it assigned, and a read that fails says so, and rolls the session back, instead of
reading as "nothing is connected". Mocked at the ``EntityManager`` boundary, as
``test_f040_switched_off_apps.py`` does; ``get_connected_apps`` itself runs for real.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from modules.tools.discovery.handlers_workspace import (
    CONNECTED_APPS_NOTE,
    CONNECTED_APPS_UNREAD,
    list_connected_apps,
)

WS = uuid4()
ENTITY = {"id": 1, "workspace_id": WS, "composio_entity_id": "workspace-entity", "created_at": None}
CONNECTED_AT = datetime(2026, 11, 3, 9, 30)
# The production workspace's nine active connections.
NINE = ("TWITTER", "INSTAGRAM", "GITHUB", "COMPOSIO_SEARCH", "GOOGLEDRIVE", "GOOGLESHEETS",
        "GOOGLE_ANALYTICS", "GOOGLECALENDAR", "GMAIL")


def _row(app, status="active", connection_id="ca_1", connected_at=CONNECTED_AT):
    """One connection as ``EntityManager.get_entity_connections`` returns it."""
    return {"id": 1, "entity_id": 1, "app_name": app, "status": status, "connected_at": connected_at,
            "connection_id": connection_id, "last_synced_at": None}


class _Query:
    def __init__(self, db):
        self.db = db

    def filter(self, *criteria):
        self.db.criteria.extend(criteria)
        return self

    join = filter

    def all(self):
        return list(self.db.assigned)


class _Db:
    """The session: its only query is the assignment count, which finds ``assigned``.
    It keeps the query's criteria and counts its rollbacks."""

    def __init__(self, assigned, rollback_fails=False):
        self.assigned, self.criteria, self.rolled_back, self.rollback_fails = assigned, [], 0, rollback_fails

    def query(self, *_a):
        return _Query(self)

    def rollback(self):
        self.rolled_back += 1
        if self.rollback_fails:
            raise RuntimeError("the connection is gone")


def _compared(criteria, column):
    """The values the query compared ``column`` ("table.column") with."""
    return [getattr(c.right, "value", None) for c in criteria if str(getattr(c, "left", "")) == column]


@pytest.fixture
def workspace(monkeypatch):
    """``run(rows, assigned=())`` lists ``WS``'s apps (``without_entity``: it has no Composio
    entity). ``settled`` holds each ``update_connection_status`` call ``get_connected_apps``
    made, ``asked`` each workspace id ``get_entity_by_workspace`` was given, ``db`` the session."""
    from core.composio.entity_manager import EntityManager

    state = NS(settled=[], asked=[], db=None)

    def entity_of(self, workspace_id, entity):
        state.asked.append(workspace_id)
        return entity

    def run(rows, assigned=(), *, without_entity=False):
        entity = None if without_entity else ENTITY
        monkeypatch.setattr(EntityManager, "get_entity_by_workspace", lambda self, ws: entity_of(self, ws, entity))
        monkeypatch.setattr(EntityManager, "get_entity_connections", lambda self, eid: [dict(r) for r in rows])
        monkeypatch.setattr(EntityManager, "update_connection_status",
                            lambda self, **kw: state.settled.append(kw) or True)
        state.db = _Db(assigned)
        return asyncio.run(list_connected_apps(state.db, WS, {}))
    state.run = run
    return state


def _by_app(result):
    return {app["app_name"]: app for app in result["connected_apps"]}


def _assigned(app, agent_id):
    return NS(app_name=app, agent_id=agent_id)


def test_every_connected_app_is_listed_though_no_agent_has_it_assigned(workspace):
    result = workspace.run([_row(app) for app in NINE])

    assert result["success"] is True
    assert sorted(_by_app(result)) == sorted(NINE)
    assert {app["status"] for app in result["connected_apps"]} == {"active"}
    assert {app["assigned_to_agents"] for app in result["connected_apps"]} == {0}
    assert _by_app(result)["INSTAGRAM"]["connected_at"] == CONNECTED_AT.isoformat()
    assert result["count"] == result["connected_count"] == 9
    assert result["note"] == CONNECTED_APPS_NOTE


def test_an_assigned_app_shows_how_many_agents_have_it(workspace):
    assigned = [_assigned("github", 7), _assigned("GITHUB", 7),   # one agent, both spellings: counted once
                _assigned("GITHUB", 8),
                _assigned("SLACK", 9)]                            # assigned, never connected

    apps = _by_app(workspace.run([_row("GITHUB"), _row("INSTAGRAM")], assigned=assigned))

    assert apps["GITHUB"]["assigned_to_agents"] == 2                       # agents 7 and 8
    assert apps["INSTAGRAM"]["assigned_to_agents"] == 0
    assert "SLACK" not in apps                                             # an assignment is not a connection


def test_only_this_workspaces_connections_and_agents_are_read(workspace):
    from core.models import Agent
    from core.models.composio_cache import AgentAppAssignment

    workspace.run([_row("GITHUB")], assigned=[_assigned("GITHUB", 7)])

    assert workspace.asked and set(workspace.asked) == {WS}               # the entity, and get_connected_apps's
    criteria = workspace.db.criteria
    assert _compared(criteria, str(Agent.__table__.c.workspace_id)) == [WS]
    assert _compared(criteria, str(AgentAppAssignment.__table__.c.app_type)) == ["EXTERNAL"]   # the executor's rows


def test_a_pending_row_whose_sign_in_finished_is_connected_and_settled(workspace):
    rows = [_row("INSTAGRAM", status="pending", connection_id="ca_instagram", connected_at=None),
            _row("TWITTER", status="pending", connection_id=None, connected_at=None)]

    result = workspace.run(rows)   # the second read still says pending: get_connected_apps decides

    apps = _by_app(result)
    assert apps["INSTAGRAM"]["status"] == "active"
    assert apps["TWITTER"]["status"] == "pending"                          # no account id: as recorded
    assert result["count"] == 2 and result["connected_count"] == 1
    assert [(s["app_name"], s["status"], s["connection_id"]) for s in workspace.settled] == [
        ("INSTAGRAM", "active", "ca_instagram")]


def test_in_a_check_in_a_finished_sign_in_is_connected_and_nothing_is_written(workspace):
    from core.read_only_call import reading_only

    with reading_only():
        result = workspace.run([_row("INSTAGRAM", status="pending", connection_id="ca_instagram", connected_at=None)])

    assert _by_app(result)["INSTAGRAM"]["status"] == "active"
    assert workspace.settled == []


def test_an_added_app_is_listed_with_its_status_so_auto_can_say_reconnect_it(workspace):
    rows = [_row("GMAIL"), _row("LINKEDIN", status="added", connection_id=None),
            _row("SHOPIFY", status="error")]                                # the Tools page hides an error row

    result = workspace.run(rows)

    apps = _by_app(result)
    assert apps["LINKEDIN"] == {"app_name": "LINKEDIN", "status": "added", "connected_at": None,
                                "assigned_to_agents": 0}
    assert "SHOPIFY" not in apps
    assert result["count"] == 2 and result["connected_count"] == 1


def test_a_workspace_with_no_composio_entity_has_nothing_connected(workspace):
    result = workspace.run([], assigned=[_assigned("GITHUB", 7)], without_entity=True)

    assert result["success"] is True
    assert result["connected_apps"] == [] and result["count"] == 0 and result["connected_count"] == 0


def test_the_tool_says_what_it_lists_and_never_that_any_agent_can_use_an_app():
    """The executor gates on assignment before connection (core/composio/tool_executor.py):
    an agent with apps assigned is refused an active app it does not have, so neither the
    description nor the note may say a connection is all an agent needs. A pending row is
    read as recorded, never checked with Composio as the Tools page checks it."""
    from modules.tools.discovery import get_action_registry

    description = get_action_registry().get("platform_list_connected_apps").description
    assert "active = connected for the whole workspace" in description
    assert "added = in the workspace but disconnected" in description and "assigned_to_agents" in description
    assert "not yet checked with Composio" in description and "Tools page" not in description
    assert "pending is as recorded here, not yet checked with Composio" in CONNECTED_APPS_NOTE
    assert "any agent" not in CONNECTED_APPS_NOTE.lower()
    assert "an agent with apps assigned can use only the active apps assigned and switched on" in CONNECTED_APPS_NOTE


def _unreadable(monkeypatch):
    from core.composio.entity_manager import EntityManager

    def unreadable(self, entity_id):
        raise RuntimeError("connection pool exhausted")

    monkeypatch.setattr(EntityManager, "get_entity_by_workspace", lambda self, ws: ENTITY)
    monkeypatch.setattr(EntityManager, "get_entity_connections", unreadable)


def test_a_read_that_fails_is_a_failure_never_an_empty_list_and_rolls_back(monkeypatch, caplog):
    _unreadable(monkeypatch)
    db = _Db([])

    with caplog.at_level(logging.ERROR):
        result = asyncio.run(list_connected_apps(db, WS, {}))

    assert result == {"success": False, "error": CONNECTED_APPS_UNREAD}
    assert any(record.exc_info for record in caplog.records)               # logged with its traceback
    assert db.rolled_back == 1                                             # the caller's session is usable again


def test_a_rollback_that_fails_too_is_logged_and_the_answer_is_still_the_failure(monkeypatch, caplog):
    _unreadable(monkeypatch)
    db = _Db([], rollback_fails=True)

    with caplog.at_level(logging.ERROR):
        result = asyncio.run(list_connected_apps(db, WS, {}))

    assert result == {"success": False, "error": CONNECTED_APPS_UNREAD}
    assert db.rolled_back == 1
    assert sum(1 for record in caplog.records if record.exc_info) == 2    # the read, then the rollback
