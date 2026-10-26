"""PRD-128 — a notice fits the bell's columns, and a failed one never takes down the work it reports.

Production, 26 Oct 2026: a Mission step finished and the coordinator announced it
(``mission_step_complete``, title ``'Mission step: ' + the step's title``). The title was over
300 characters; ``notifications.title`` is VARCHAR(255), so the in-app INSERT failed with
StringDataRightTruncation. That INSERT ran on the coordinator's own session with no savepoint,
so the error aborted the coordinator's transaction: the step's result and the ticket's move into
Review were lost, the run errored, and the Review push never went out. Proven here:

* ``clamp`` cuts to the limit and ends the cut in '…'; the bounded column lengths are the ones
  the prd128 revision declares; every bounded value of an in-app row is cut to its column, and
  the TEXT columns are left whole.
* The insert's savepoint is taken on the caller's connection, never with
  ``Session.begin_nested()`` (which would flush the caller's pending work). An insert that fails
  (a ``DataError``) fails inside it, is logged, and the push is still handed over, without the
  row's id.
* On Postgres, through the coordinator's own announcement: a step with a 300-character title
  rings the bell with a 255-character title ending in '…'; and an insert that fails in Postgres
  (the 26 Oct title with the cut turned off, or a step agent that is gone) leaves the caller's
  transaction usable: its next write and its commit land.

CI's test database has no ``notifications`` table (no model), so the real-DB fixture runs the
prd128 revision's DDL inside the test's transaction, rolled back with everything else.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DataError

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.services import notification_dispatcher as nd  # noqa: E402
from core.services.notification_columns import ELLIPSIS, NOTIFICATION_COLUMN_CHARS, clamp  # noqa: E402
from core.services.notification_dispatcher import NotificationDispatcher  # noqa: E402

ORCH = Path(__file__).resolve().parents[1]
NOTIFICATIONS_REVISION = ORCH / "alembic" / "versions" / "prd128_notifications.py"
HAND_OVER = "services.push.fanout.hand_over"
LONG_STEP_TITLE = "R" * 300
FAILED_IN_APP = "Failed to deliver event_type={event} destination=in_app"


def _revision_statements(path: Path) -> list:
    """The DDL a revision's ``upgrade()`` runs, collected instead of executed."""
    spec = importlib.util.spec_from_file_location(f"revision_{path.stem}", path)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    statements: list = []
    revision.op = SimpleNamespace(execute=statements.append)
    revision.upgrade()
    return statements


# --------------------------------------------------------------------------- the cut


def test_clamp_leaves_a_short_value_and_none_alone_and_cuts_a_long_one_to_the_limit():
    assert clamp("Weekly numbers", 20) == "Weekly numbers"
    assert clamp("x" * 20, 20) == "x" * 20
    assert clamp(None, 20) is None
    cut = clamp("x" * 21, 20)
    assert len(cut) == 20 and cut == "x" * 19 + ELLIPSIS


def test_the_bounded_column_lengths_are_the_ones_the_prd128_revision_declares():
    (table,) = [s for s in _revision_statements(NOTIFICATIONS_REVISION)
                if "CREATE TABLE IF NOT EXISTS notifications " in s]
    declared = {name: int(n) for name, n in re.findall(r"^\s*(\w+)\s+VARCHAR\((\d+)\)", table, re.M)}

    assert declared == dict(NOTIFICATION_COLUMN_CHARS)


def test_every_bounded_value_of_an_in_app_row_is_cut_to_its_column_and_text_is_left_whole():
    db = MagicMock()
    long = "y" * 400
    NotificationDispatcher(db, uuid.uuid4())._insert_in_app(
        user_id=None, event_type=long, title=long, message=long, link_type=long, link_id=long,
        agent_id=None, agent_name=long, status=long,
    )
    params = db.execute.call_args.args[1]

    for column, limit in NOTIFICATION_COLUMN_CHARS.items():
        assert len(params[column]) == limit and params[column].endswith(ELLIPSIS), column
    assert params["message"] == long and params["link_id"] == long   # TEXT columns
    db.connection.return_value.begin_nested.assert_called_once()   # the savepoint, on the connection
    db.begin_nested.assert_not_called()                             # never the Session's: no flush of the caller's work


def test_a_failed_in_app_insert_fails_in_its_savepoint_is_logged_and_still_hands_the_push_over(caplog):
    db = MagicMock()
    preferences = MagicMock()
    preferences.fetchall.return_value = []
    db.execute.side_effect = [preferences, DataError("INSERT INTO notifications", {}, Exception("too long"))]
    handed = MagicMock(return_value=True)

    with patch.object(NotificationDispatcher, "_load_auto_reporting", lambda self: {}), \
            patch(HAND_OVER, new=handed), patch.object(nd, "send_workspace_notification", new=AsyncMock()), \
            caplog.at_level(logging.ERROR, logger=nd.__name__):
        result = asyncio.run(NotificationDispatcher(db, uuid.uuid4()).dispatch(
            event_type="question_pending", title="Auto asks", link_type="question", link_id="9"))

    assert result == {"dispatched_to": ["push"]}
    savepoint = db.connection.return_value.begin_nested.return_value
    assert savepoint.__exit__.call_args.args[0] is DataError        # the failure ended its savepoint
    assert handed.call_args.args[1].notification_id is None
    assert handed.call_args.args[1].link_id == "9"
    assert any(FAILED_IN_APP.format(event="question_pending") in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- on Postgres


@pytest.fixture
def db_session(test_engine):
    """The coordinator's session: one connection in one transaction, rolled back after the test.

    Its commit releases a savepoint of its own, a statement Postgres refuses on an aborted
    transaction (a real COMMIT there would quietly roll the work back instead).
    """
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    outer = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    outer.rollback()
    connection.close()


@pytest.fixture
def workspace(db_session, seed_workspace) -> tuple:
    """The bell's tables as production has them, a workspace and its agent: ``(ws, agent id)``."""
    for statement in _revision_statements(NOTIFICATIONS_REVISION):
        db_session.execute(text(statement))
    ws = seed_workspace(name="Mission")
    agent_id = db_session.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status) "
        "VALUES ('Researcher', 'custom', CAST(:w AS uuid), 'active') RETURNING id"), {"w": ws}).scalar()
    db_session.commit()
    return ws, int(agent_id)


def _announce_step(db, ws: str, agent_id: int) -> None:
    """The coordinator announces a finished step whose title is 300 characters long."""
    from core.models.orchestration_enums import TaskState
    from services.coordinator_service import CoordinatorService

    run = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.UUID(ws), created_by=None)
    task = SimpleNamespace(id=1926, title=LONG_STEP_TITLE, state=TaskState.RUNNING.value)
    result = {"status": "error", "output": "Draft written"}
    asyncio.run(CoordinatorService.__new__(CoordinatorService)._announce_task_result(db, run, task, agent_id, result))


def _bell_titles(db, ws: str) -> list:
    return list(db.execute(text("SELECT title FROM notifications WHERE workspace_id = CAST(:w AS uuid)"),
                           {"w": ws}).scalars())


def _write(db, ws: str, words: str) -> None:
    """One of the caller's own writes on its session (it stands for the step's result, the ticket's move)."""
    db.execute(text("UPDATE workspaces SET name = name || :words WHERE id = CAST(:w AS uuid)"),
               {"words": words, "w": ws})


def _name(db, ws: str) -> str:
    return db.execute(text("SELECT name FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws}).scalar()


def test_a_mission_steps_300_character_title_rings_the_bell_cut_to_255_ending_in_an_ellipsis(workspace, db_session):
    ws, agent_id = workspace

    _announce_step(db_session, ws, agent_id)
    db_session.commit()

    (title,) = _bell_titles(db_session, ws)
    assert len(title) == NOTIFICATION_COLUMN_CHARS["title"] == 255
    assert title == f"Mission step: {LONG_STEP_TITLE}"[:254] + ELLIPSIS


def _gone_agent(db) -> int:
    return int(db.execute(text("SELECT COALESCE(MAX(id), 0) + 1 FROM agents")).scalar())


@pytest.mark.parametrize("failure", ["the 26 Oct title, the cut turned off", "a step agent that is gone"])
def test_an_insert_postgres_refuses_leaves_the_callers_transaction_usable(
    failure, workspace, db_session, monkeypatch, caplog,
):
    ws, agent_id = workspace
    if failure == "a step agent that is gone":
        agent_id = _gone_agent(db_session)                      # notifications.agent_id's foreign key
    else:
        monkeypatch.setattr(nd, "fit_columns", dict)            # the INSERT as it ran on 26 Oct

    _write(db_session, ws, ": step recorded")                  # before the announcement, uncommitted
    with caplog.at_level(logging.ERROR, logger=nd.__name__):
        _announce_step(db_session, ws, agent_id)
    _write(db_session, ws, ", ticket in Review")               # the caller's next write
    db_session.commit()

    assert _name(db_session, ws) == "Mission: step recorded, ticket in Review"
    assert _bell_titles(db_session, ws) == []
    assert any(FAILED_IN_APP.format(event="mission_step_complete") in r.getMessage() for r in caplog.records)
