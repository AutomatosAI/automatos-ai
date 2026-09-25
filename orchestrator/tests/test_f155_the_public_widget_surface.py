"""F155 — what a public widget key reaches.

(h) The widget data plane runs no caller SQL: only the natural-language query,
which the NL2SQL service scopes to the key's workspace, remains.
(d-1) A key's team lock scopes its chat's documents, as it already scopes
/search and /docs; without one, the answering agent's team does.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from sqlalchemy import text


def test_the_widget_data_plane_runs_no_caller_sql():
    from api.widgets import data

    assert sorted(route.path for route in data.router.routes) == ["/data/query"]


def test_the_keys_team_lock_scopes_widget_chat(db_session, seed_workspace):
    from api.widgets.chat import _retrieval_team

    agent = db_session.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration, team) "
                                    "VALUES ('Barista', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json), 'hq') "
                                    "RETURNING id"), {"w": seed_workspace()}).scalar()
    assert _retrieval_team(db_session, NS(team="franchise-a"), agent) == "franchise-a"
    assert _retrieval_team(db_session, NS(team=None), agent) == "hq"
