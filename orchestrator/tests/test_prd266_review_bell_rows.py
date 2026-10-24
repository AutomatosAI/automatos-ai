"""PRD-266 W2 post-merge review — the bell's rows, on Postgres, through the real dispatcher.

* One row per event: a watch verdict and a Mission's end each put exactly one row in their
  creator's bell (``watch_verdict``, ``mission_complete``), while their line still lands in the
  person's chat. Neither origin is announced as ``auto_message`` any more, so no second
  "Auto messaged you" row joins it (``auto_message.ANNOUNCED_ORIGINS``; the notices
  ``send_soon`` scheduled are awaited, so a regression would show).
* A row that persists: ``approval_pending`` from a tool's grant, a board task's grant and a
  social post, a social post's publish outcome, a social plan's notice and "AI credit is back",
  each dispatched on a session of its own, is still in the bell after the helper returns.
  Before the fix each helper closed its session without a commit and the row was rolled back.

Each helper's "session of its own" is a real ``Session`` on the test's connection, joined by
SAVEPOINT: its ``close()`` without a commit rolls back what it wrote, as closing a
``SessionLocal`` session does in production, and its commit keeps it for the test to read.
CI's test database has no ``notifications`` or ``notification_preferences`` (no model), so
the fixture runs the real ``prd128_notifications`` revision's DDL inside the test's
transaction, which is rolled back with everything else.
"""
from __future__ import annotations

import asyncio
import importlib.util
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
PREFERENCES_REVISION = ORCH / "alembic" / "versions" / "prd128_notifications.py"
POST_ID = uuid.UUID("2f6d1c1e-4f7b-4a51-9a3e-5d1f0c2b7a10")
PLAN_ID = uuid.UUID("8c3e0b52-1d4a-4c7e-b0f2-6a9d3e1c5b77")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def connection(test_engine):
    """One connection in one transaction, rolled back after the test."""
    conn = test_engine.connect()
    transaction = conn.begin()
    yield conn
    transaction.rollback()
    conn.close()


def _savepoint_session(conn):
    from sqlalchemy.orm import Session

    return Session(bind=conn, join_transaction_mode="create_savepoint")


@pytest.fixture
def db_session(connection):
    """The test's session (overrides conftest's): a commit releases its savepoint only."""
    session = _savepoint_session(connection)
    yield session
    session.close()


@pytest.fixture
def own_sessions(connection, monkeypatch):
    """``SessionLocal`` as the code under test opens it: a session of its own on this connection."""
    from core.database import database

    def session_local():
        return _savepoint_session(connection)

    monkeypatch.setattr(database, "SessionLocal", session_local)
    return session_local


def _revision_statements(path: Path) -> list:
    spec = importlib.util.spec_from_file_location(f"revision_{path.stem}", path)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    statements: list = []
    revision.op = SimpleNamespace(execute=statements.append)
    revision.upgrade()
    return statements


@pytest.fixture(autouse=True)
def bell_tables(db_session):
    """``notifications`` and ``notification_preferences`` as production has them (prd128's DDL)."""
    for statement in _revision_statements(PREFERENCES_REVISION):
        db_session.execute(text(statement))
    db_session.commit()


@pytest.fixture
def ws(seed_workspace, db_session) -> str:
    workspace_id = seed_workspace()
    db_session.commit()
    return workspace_id


@pytest.fixture
def person(db_session) -> tuple:
    """A signed-in person: ``(clerk id, users.id)``."""
    clerk = f"user_prd266bell_{uuid.uuid4().hex[:10]}"
    pk = db_session.execute(text(
        "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
        {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    db_session.commit()
    return clerk, int(pk)


def _bell(db, workspace_id) -> list:
    """Every bell row in the workspace: ``(user_id, event_type, title, link_type, link_id)``."""
    return [tuple(row) for row in db.execute(text(
        "SELECT user_id, event_type, title, link_type, link_id FROM notifications "
        "WHERE workspace_id = CAST(:w AS uuid) ORDER BY created_at"), {"w": str(workspace_id)}).fetchall()]


def _chat_lines(db, workspace_id) -> list:
    """The background lines posted in the workspace's chats: ``(origin, text)``."""
    from core.models.core import Message

    rows = db.query(Message).filter(Message.workspace_id == uuid.UUID(str(workspace_id))).all()
    return [((row.source or {}).get("origin"), row.parts[0]["text"]) for row in rows]


async def _the_notices_run() -> None:
    """Every notice ``send_soon`` scheduled on this loop runs to its end."""
    from services.notification_events import soon

    loop = asyncio.get_running_loop()
    await asyncio.gather(*[task for task in list(soon._PENDING) if task.get_loop() is loop])


# --------------------------------------------------------------------------- one row per event


def test_a_watch_verdict_puts_one_row_in_its_creators_bell(ws, person, db_session, own_sessions):
    from services.watch_notifications import notify_watch_verdict

    clerk, pk = person
    watch = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.UUID(ws), created_by=clerk,
                            origin_chat_id=None, title="Lisbon deck", quality_threshold=0.7)

    async def verdict():
        sent = await notify_watch_verdict(db_session, watch, score=0.84, explanation="Passed.", passed=True)
        db_session.commit()
        await _the_notices_run()
        return sent

    assert asyncio.run(verdict()) is True

    assert _bell(db_session, ws) == [
        (pk, "watch_verdict", "Watch verdict (passed): Lisbon deck", "watch", str(watch.id))]
    assert [origin for origin, _ in _chat_lines(db_session, ws)] == ["watcher"]  # the chat still hears it


def test_a_missions_end_puts_one_row_in_its_creators_bell(ws, person, db_session, own_sessions):
    from core.models.orchestration_enums import RunState
    from services import coordinator_service

    clerk, pk = person
    run = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.UUID(ws), created_by=clerk, goal="Lisbon deck",
                          state=RunState.COMPLETED.value, config={}, plan={}, stop_detail=None, stop_reason=None)

    async def complete():
        await coordinator_service._dispatch_mission_event(
            db=db_session, run=run, event_type="mission_complete", title="Mission complete: Lisbon deck",
            message="3 tasks verified", status="ok")
        db_session.commit()
        coordinator_service._narrate_run_terminal(db_session, run)
        await _the_notices_run()

    asyncio.run(complete())

    assert _bell(db_session, ws) == [
        (pk, "mission_complete", "Mission complete: Lisbon deck", "mission", str(run.id))]
    assert _chat_lines(db_session, ws) == [("mission", "Mission complete: Lisbon deck")]


# --------------------------------------------------------------------------- a row that persists


def _tool_grant(workspace_id, _factory, _monkeypatch):
    from modules.tools.execution import tool_grants

    return tool_grants._dispatch_approval_pending(
        workspace_id, grant_id=7, tool_name="send_email", risk_tier="high", reason="Send the deck to Matt")


def _board_grant(workspace_id, _factory, _monkeypatch):
    from services import board_approval

    return board_approval._dispatch_approval_pending(
        workspace_id, grant_id=8, subject_id="42", risk_tier=None, reason="Over budget", estimated_cost_usd=1.5)


def _social_post(workspace_id, factory, _monkeypatch):
    from modules.socials import notify

    return notify.dispatch_approval_pending(workspace_id, POST_ID, "Launch week", session_factory=factory)


def _social_outcome(workspace_id, factory, _monkeypatch):
    from modules.socials import notify

    return notify.dispatch_publish_outcome(workspace_id, POST_ID, "Launch week", "published",
                                           session_factory=factory)


def _plan_notice(workspace_id, _factory, _monkeypatch):
    from modules.socials import plan_notify

    return plan_notify._send(workspace_id, plan_notify.PLAN_LINK_TYPE, PLAN_ID, plan_notify.READY, "Harvest Club")


def _credit_back(workspace_id, _factory, monkeypatch):
    from core.llm import credit

    monkeypatch.setattr(credit, "_back_to_tell", {workspace_id})  # the bell said credit ran out
    return credit._tell_credit_back(workspace_id, 0)


@pytest.mark.parametrize("helper, row", [
    (_tool_grant, ("approval_pending", "Approval needed: send_email", "approval_grant", "7")),
    (_board_grant, ("approval_pending", "Approval needed: board task 42", "approval_grant", "8")),
    (_social_post, ("approval_pending", "Social post needs approval: Launch week", "social_post", str(POST_ID))),
    (_social_outcome, ("social_post_published", "Social post published: Launch week", "social_post",
                       str(POST_ID))),
    (_plan_notice, ("social_plan_ready", "Today's posts are ready: Harvest Club", "social_plan", str(PLAN_ID))),
    (_credit_back, ("credit_restored", "AI credit is back", None, None)),
], ids=["tool-grant", "board-grant", "social-post", "social-outcome", "social-plan", "credit-back"])
def test_a_row_dispatched_on_a_session_of_its_own_is_still_there_after(ws, db_session, own_sessions, monkeypatch,
                                                                       helper, row):
    asyncio.run(helper(ws, own_sessions, monkeypatch))

    assert _bell(db_session, ws) == [(None, *row)]  # the workspace's, not one person's
