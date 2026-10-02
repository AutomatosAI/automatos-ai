"""F225 — the board shows every open ticket, however old.

Refresh 7: the board asked for the newest 200 tickets of any status. In the
owner's workspace 33 tickets waited in Review, but only 9 were among the newest
200, so the Review column showed 9 while Needs you counted 33: "a review I can't
find". With ``finished_limit`` the list returns every open ticket and windows
only Done, Cancelled and Closed.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine, text

NOW = datetime.now(timezone.utc)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the board-list tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def board(engine, new_session):
    """Two old tickets in Review, then three newer finished ones."""
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f225')"), {"id": ws})
    ids = {}
    for title, status, age in (("Old price list", "review", 40), ("Old rota", "review", 39),
                               ("Menu", "done", 3), ("Flyer", "cancelled", 2), ("Signs", "done", 1)):
        at = NOW - timedelta(days=age)
        ids[title] = s.execute(text(
            "INSERT INTO board_tasks (workspace_id, title, status, created_at, completed_at) "
            "VALUES (CAST(:w AS uuid), :t, :s, :at, :at) RETURNING id"),
            {"w": ws, "t": title, "s": status, "at": at}).scalar()
    s.commit()
    yield NS(ws=uuid.UUID(ws), ids=ids)
    s = new_session.sweep()
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _list(board, new_session, **over):
    from api.board_tasks import list_tasks

    params = dict(status=None, agent_id=None, priority=None, search=None, parent_task_id=None,
                  limit=100, offset=0, finished_limit=None)
    params.update(over)
    out = list_tasks(ctx=NS(workspace_id=board.ws), db=new_session(), **params)
    return [t["title"] for t in out["tasks"]], out["total"]


def test_the_board_gets_every_open_ticket_and_the_newest_finished(board, new_session):
    titles, total = _list(board, new_session, finished_limit=2)

    assert {"Old price list", "Old rota"} <= set(titles)       # before: past the window, off the board
    assert [t for t in titles if t in ("Menu", "Flyer", "Signs")] == ["Signs", "Flyer"]
    assert total == 5


def test_a_plain_list_still_pages_by_age(board, new_session):
    titles, total = _list(board, new_session, limit=2)

    assert titles == ["Signs", "Flyer"] and total == 5
