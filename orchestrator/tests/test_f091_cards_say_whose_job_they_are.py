"""F091-E1 (night 3) — every card says whose job it belongs to.

Night 3's cards read "Agent #294 · board_task:612" or a bare tool name; the
persona could not tell which agent was asking or which ticket an answer would
move. Each listed grant now carries ``owner`` — the agent with its name and the
ticket with its title — looked up once for the whole list.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from services.grant_owners import grant_owners

_TABLES = ("agents", "board_tasks")


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        for t in _TABLES:
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{t}"))
            session.execute(text(f"CREATE TEMP TABLE {t} (LIKE public.{t} INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        for t in _TABLES:
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{t}"))
        session.commit()
        session.close()


def _seed(db, ws):
    db.execute(text("INSERT INTO agents (id, name, agent_type, workspace_id, status) "
                    "VALUES (294, 'Scout', 'custom', CAST(:ws AS uuid), 'active')"), {"ws": ws})
    db.execute(text("INSERT INTO board_tasks (id, workspace_id, title, status, priority, source_type, attempts) "
                    "VALUES (612, CAST(:ws AS uuid), 'Cafe questions', 'review', 'medium', 'user', 0)"),
               {"ws": ws})


def _grant(gid, **kw):
    return NS(**{"id": gid, "subject_type": "tool_call", "subject_id": "k", "asked_by_agent_id": None,
                 "agent_id": None, "details": {}, **kw})


def test_a_question_and_a_gated_call_name_their_agent_and_ticket(db):
    ws = str(uuid4())
    _seed(db, ws)
    owners = grant_owners(db, ws, [
        _grant(1, subject_type="board_task", subject_id="612", asked_by_agent_id=294),
        _grant(2, agent_id=294, details={"board_task_id": 612}),
        _grant(3),
    ])
    named = {"agent": {"id": 294, "name": "Scout"}, "ticket": {"id": 612, "title": "Cafe questions"}}
    assert owners[1] == named and owners[2] == named
    assert owners[3] == {"agent": None, "ticket": None}


def test_another_workspaces_rows_are_never_named(db):
    ws = str(uuid4())
    _seed(db, ws)
    owners = grant_owners(db, str(uuid4()), [_grant(1, subject_type="board_task", subject_id="612",
                                                    asked_by_agent_id=294)])
    assert owners[1] == {"agent": {"id": 294, "name": None}, "ticket": {"id": 612, "title": None}}


def test_the_list_route_carries_the_owner():
    from api import approval_grants

    source = inspect.getsource(approval_grants.list_grants)
    assert "grant_owners(db, ctx.workspace_id, rows)" in source and '"owner": owners.get(g.id)' in source
