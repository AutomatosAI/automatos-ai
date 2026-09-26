"""F179 (review HIGH) — one approval is one publish, even for two calls at once.

A single-use approval was read GRANTED, checked, then marked spent in memory,
so two calls on two sessions could both read it GRANTED and both publish: two
public links from one yes. The approval is now claimed under a row lock that a
concurrent call skips (it gets the ask); a call whose transaction rolls back
leaves the approval for the next.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text

ACTION = "workspace_get_public_url"
PARAMS = {"path": "content/social/instagram/launch.png"}


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the grant race tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def approved(engine, new_session):
    from core.services.approval_grants import grant_grant
    from modules.tools.execution.tool_grants import issue_tool_grant

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f179-race')"), {"id": ws})
    grant = issue_tool_grant(s, ws, action=ACTION, params=PARAMS, permission_level="write",
                             description="Publish a workspace image", caller_context={"mission_id": "m-7"})
    grant_grant(grant, granted_by="user:1")
    s.commit()
    yield ws, grant.id
    s = new_session.sweep()
    s.execute(text("DELETE FROM approval_grants WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _consume(session, ws):
    from modules.tools.execution.tool_grants import consume_tool_grant

    return consume_tool_grant(session, ws, action=ACTION, params=PARAMS, permission_level="write")


def test_two_calls_at_once_on_one_approval_publish_once(approved, new_session):
    ws, grant_id = approved
    first, second = new_session(), new_session()
    won = _consume(first, ws)
    lost = _consume(second, ws)                     # while the first call still holds it
    assert won is not None and won.id == grant_id
    assert lost is None                             # it gets the ask, not a second link
    first.commit()
    assert _consume(new_session(), ws) is None      # and once spent, it stays spent


def test_a_call_that_rolled_back_leaves_the_approval(approved, new_session):
    ws, grant_id = approved
    first = new_session()
    assert _consume(first, ws) is not None
    first.rollback()
    again = _consume(new_session(), ws)
    assert again is not None and again.id == grant_id
