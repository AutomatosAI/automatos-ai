"""P256-FIX-N17-3 (F439, night 17): a click on a card past its expiry runs nothing and says so.

Night 17 clicked #2607 thirteen hours after it expired: /grant checked only that it was pending,
granted it and resumed the call, whose gate found the yes expired and asked again; the chat was
told "Your click ran: … it did not go through: no reason given".

Now /grant refuses a call's card past its ``expires_at`` with a 422 naming when it expired; the
row is closed (expired, ``system:expired``) and committed, nothing is resumed and no decision is
remembered. A card a newer one replaced says so. An agent's send card and a board ticket's card
keep today's path.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text

from api.approval_grant_lapses import EXPIRED, REPLACED
from modules.tools.discovery.my_actions_reads import at_said
from modules.tools.execution.stale_cards import EXPIRED_BY

ACTION = "platform_execute_playbook"
RUN_NOW = {"playbook_id": 81}
QUESTION = "Run a playbook 'Monday green-coffee reorder' (playbook #81):"


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the expired-click tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def ws(engine, new_session, monkeypatch):
    import modules.tools.execution.tool_grants as tool_grants

    monkeypatch.setattr(tool_grants, "_notify_approval_pending", lambda grant, workspace_id: None)
    workspace = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f439')"), {"id": workspace})
    s.commit()
    yield workspace
    s = new_session.sweep()
    s.execute(text("DELETE FROM approval_grants WHERE workspace_id = CAST(:w AS uuid)"), {"w": workspace})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": workspace})
    s.commit()


@pytest.fixture
def route(monkeypatch):
    """The /grant route with what runs after the yes recorded: the resume and the remembered click."""
    import api.approval_grants as ag
    import services.click_memory as click_memory

    remembered = AsyncMock(return_value=None)
    monkeypatch.setattr(click_memory, "remember_the_click", remembered)
    monkeypatch.setattr(ag, "_audit", lambda *a, **k: None)
    return NS(ag=ag, remembered=remembered)


def _chat(conversation=None):
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": conversation or str(uuid.uuid4()),
            "turn_id": "t-1"}


def _card(new_session, workspace, ctx, *, expired_hours=None, details=None):
    """A run-now card, committed: raised for ``ctx``, past its expiry by ``expired_hours``."""
    from modules.tools.execution.tool_grants import issue_tool_grant

    s = new_session()
    grant = issue_tool_grant(s, workspace, action=ACTION, params=RUN_NOW, permission_level="write",
                             description="Run a playbook", caller_context=ctx, question_md=QUESTION)
    if details:
        grant.details = {**grant.details, **details}
    if expired_hours is not None:
        grant.expires_at = datetime.now(timezone.utc) - timedelta(hours=expired_hours)
    s.commit()
    return grant.id, grant.expires_at


def _click(route, new_session, workspace, grant_id):
    ctx = NS(workspace_id=UUID(workspace), user=NS(id=7, clerk_user_id=None, email="owner@harbourline.test"))
    return asyncio.run(route.ag.grant_approval(grant_id, ctx=ctx, db=new_session()))


def _status(new_session, grant_id):
    row = new_session().execute(text("SELECT status, revoked_by FROM approval_grants WHERE id = :id"),
                                {"id": grant_id}).one()
    return row[0], row[1]


def test_a_click_on_an_expired_card_is_refused_with_its_time_and_runs_nothing(ws, new_session, route, monkeypatch):
    resumed = []

    async def _resume(db, grant):
        resumed.append(grant.id)

    monkeypatch.setattr(route.ag, "_resume_tool_call", _resume)
    grant_id, expired_at = _card(new_session, ws, _chat(), expired_hours=13)

    with pytest.raises(HTTPException) as refused:
        _click(route, new_session, ws, grant_id)

    assert refused.value.status_code == 422
    assert refused.value.detail == EXPIRED.format(at=at_said(expired_at))
    assert "nothing ran" in refused.value.detail and "at " in refused.value.detail
    assert _status(new_session, grant_id) == ("expired", EXPIRED_BY)     # committed, though the route raised
    assert resumed == []
    route.remembered.assert_not_awaited()


def test_a_second_click_on_it_says_the_same_never_just_not_pending(ws, new_session, route):
    grant_id, expired_at = _card(new_session, ws, _chat(), expired_hours=13)
    with pytest.raises(HTTPException):
        _click(route, new_session, ws, grant_id)

    with pytest.raises(HTTPException) as again:
        _click(route, new_session, ws, grant_id)

    assert again.value.status_code == 422 and again.value.detail == EXPIRED.format(at=at_said(expired_at))


def test_a_click_on_a_card_a_newer_one_replaced_says_so(ws, new_session, route):
    first, _ = _card(new_session, ws, _chat())
    second, _ = _card(new_session, ws, _chat())          # another chat: the first card is replaced
    assert second != first

    with pytest.raises(HTTPException) as refused:
        _click(route, new_session, ws, first)

    assert refused.value.status_code == 422 and refused.value.detail == REPLACED
    assert "replaced by a newer card" in refused.value.detail
    route.remembered.assert_not_awaited()


def test_an_agents_expired_send_card_keeps_todays_path(ws, new_session, route, monkeypatch):
    import modules.tools.discovery.agent_sends as agent_sends

    sent = AsyncMock(return_value=True)
    monkeypatch.setattr(agent_sends, "sends_on_the_click", sent)
    marker = {agent_sends.AGENT_SEND: {"lane": "session", "task_id": 41, "context": {"session_task_id": 41}}}
    grant_id, _ = _card(new_session, ws, {"session_task_id": 41}, expired_hours=13, details=marker)

    out = _click(route, new_session, ws, grant_id)

    assert out["grant"]["id"] == grant_id
    sent.assert_awaited_once()
    route.remembered.assert_awaited_once()


def test_a_board_tickets_expired_card_keeps_todays_path(ws, new_session, route, monkeypatch):
    requeued = []
    monkeypatch.setattr(route.ag, "_requeue_blocked_task",
                        lambda db, workspace_id, task_id: requeued.append(task_id) or True)
    grant_id, _ = _card(new_session, ws, {"board_task_id": 5}, expired_hours=13)

    out = _click(route, new_session, ws, grant_id)

    assert out["grant"]["id"] == grant_id and requeued == [5]
    assert out["grant"]["details"]["executed_result"]["resumed_via"] == "board_task_requeue"


def test_a_close_that_another_click_beat_says_what_the_card_is_now(ws, new_session, route, monkeypatch):
    """Two admins click an expired card at once: the one whose close matched nothing is told the
    card's status now, never 'expired' over a card the other click decided."""
    import api.approval_grant_lapses as lapses

    grant_id, _ = _card(new_session, ws, _chat(), expired_hours=13)

    def beaten(db, workspace_id, grant_ids, *, revoked_by, now=None):
        other = new_session()
        other.execute(text("UPDATE approval_grants SET status = 'granted' WHERE id = :id"), {"id": grant_id})
        other.commit()
        return 0

    monkeypatch.setattr(lapses, "expire_pending_grants", beaten)

    with pytest.raises(HTTPException) as refused:
        _click(route, new_session, ws, grant_id)

    assert refused.value.status_code == 422 and refused.value.detail == "Grant is not pending (status: granted)"
    route.remembered.assert_not_awaited()


def test_a_live_card_still_runs_on_the_click(ws, new_session, route, monkeypatch):
    resumed = []

    async def _resume(db, grant):
        resumed.append(grant.id)

    monkeypatch.setattr(route.ag, "_resume_tool_call", _resume)
    grant_id, _ = _card(new_session, ws, _chat())

    _click(route, new_session, ws, grant_id)

    assert resumed == [grant_id] and _status(new_session, grant_id)[0] == "granted"
