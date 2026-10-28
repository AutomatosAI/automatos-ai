"""PRD-268 US-C06b (§6.2.5, FR-8) — started work is watched.

``POST /api/proactive/suggestions/{id}/start`` on a ``board_task`` suggestion files its ticket
and, in the same transaction, a PRD-204 watch on it through ``WatchService`` (target
``board_task``, the kind the ASSIGN lane's ticket watches already have); ``started_ref``
records ``{board_task_id, watch_id}``. On Postgres. Proven here:

* Start creates one live watch on the ticket: run_and_report, ``watching``, the ticket's title,
  asking what its brief asks, owned by the proposed agent, with its ``created`` event;
  ``started_ref`` (stored and answered) holds both ids;
* the watch records the person who pressed Start (their Clerk id, or from a paired phone their
  email, which resolves to them) and no origin chat, so its verdict goes to that person's Auto
  thread; a ticket in the Inbox is watched too, with no owning agent;
* a chat Start, a Start the spend ceiling refuses and a second Start make no watch;
* a watch that cannot be written fails the Start whole: no ticket, the suggestion still open,
  no audit row.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

SUGGESTIONS = "/api/proactive/suggestions"
TITLE = "Ticket #0846 has been blocked for two days"
DETAIL = "VECTOR waits on the owner's answer."
EVIDENCE = [{"type": "ticket", "id": "0846"}]


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here):
    the Start route commits, and neither its commit nor a failed Start's rollback takes the seed."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(autouse=True)
def saas(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "SPEND_CEILING_USD", 0)


def _person(db, tag: str) -> NS:
    clerk = f"user_prd268watch_{tag}_{uuid.uuid4().hex[:10]}"
    pk = db.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                    {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    return NS(pk=int(pk), clerk=clerk)


@pytest.fixture
def team(db_session) -> NS:
    """A workspace with its owner, an editor and an agent."""
    db = db_session
    owner, editor = _person(db, "owner"), _person(db, "editor")
    ws = uuid.uuid4()
    db.execute(text("INSERT INTO workspaces (id, name, owner_id) VALUES (CAST(:id AS uuid), 'prd268 watch', :o)"),
               {"id": str(ws), "o": owner.pk})
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :uid, 'editor', true)"), {"ws": str(ws), "uid": editor.pk})
    agent = int(db.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('VECTOR', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar())
    db.commit()
    return NS(db=db, ws=ws, owner=owner, editor=editor, agent=agent)


def _suggestion(team, *, kind: str = "board_task", payload=None) -> uuid.UUID:
    from core.models.proactive import ProactiveSuggestion

    now = datetime.now(timezone.utc)
    row = ProactiveSuggestion(
        workspace_id=team.ws, user_id=team.owner.pk, source_type="check_in", title=TITLE, detail=DETAIL,
        evidence=EVIDENCE, subject_key=f"title:{uuid.uuid4().hex}", status="open", last_seen_at=now,
        created_at=now, proposed_action={"kind": kind, "payload": {"agent_id": team.agent} if payload is None
                                         else payload})
    team.db.add(row)
    team.db.commit()
    return row.id


def _post(team, ctx, sid):
    import api.proactive as proactive
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(proactive.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: team.db
    return TestClient(app).post(f"{SUGGESTIONS}/{sid}/start")


def _start(team, person, sid):
    from core.auth.dependencies import RequestContext, UserContext

    user = UserContext(id=person.clerk, clerk_user_id=person.clerk, system_role="user")
    return _post(team, RequestContext(workspace_id=team.ws, user=user, auth_type="clerk"), sid)


def _start_from_the_phone(team, person, sid):
    """A paired phone's Start: the person is their email, with no Clerk id."""
    from core.auth.dependencies import RequestContext, UserContext

    user = UserContext(id=f"{person.clerk}@t.test", system_role="user", raw_claims={
        "source": "mobile_device", "device_id": str(uuid.uuid4()), "user_id": person.pk})
    return _post(team, RequestContext(workspace_id=team.ws, user=user, auth_type="device"), sid)


def _row(team, sid):
    from core.models.proactive import ProactiveSuggestion

    team.db.expire_all()
    return team.db.get(ProactiveSuggestion, sid)


def _tickets(team, sid) -> list:
    from core.models.core import BoardTask

    team.db.expire_all()
    return team.db.query(BoardTask).filter(BoardTask.source_type == "suggestion",
                                           BoardTask.source_id == str(sid)).all()


def _watches(team) -> list:
    from core.models.watches import Watch

    team.db.expire_all()
    return team.db.query(Watch).filter(Watch.workspace_id == team.ws).all()


def _events(team, watch) -> list:
    from core.models.watches import WatchEvent

    return [e.event_type for e in team.db.query(WatchEvent).filter(WatchEvent.watch_id == watch.id).all()]


def test_start_creates_the_tickets_watch_and_started_ref_holds_both_ids(team):
    sid = _suggestion(team)
    response = _start(team, team.owner, sid)
    assert response.status_code == 200, response.text
    [task] = _tickets(team, sid)
    [watch] = _watches(team)
    assert (watch.watch_type, watch.target_type, watch.target_id) == ("board_task", "board_task", str(task.id))
    assert (watch.status, watch.policy, watch.title) == ("watching", "run_and_report", TITLE)
    assert (watch.success_criteria, watch.owner_agent_id) == (task.description, team.agent)
    assert _events(team, watch) == ["created"]
    expected = {"board_task_id": task.id, "watch_id": str(watch.id)}
    assert _row(team, sid).started_ref == expected
    assert response.json()["suggestion"]["started_ref"] == expected


def test_the_verdict_goes_to_the_person_who_started_it(team):
    sid = _suggestion(team)
    assert _start(team, team.editor, sid).status_code == 200
    [watch] = _watches(team)
    assert (watch.created_by, watch.origin_chat_id) == (team.editor.clerk, None)


def test_a_start_from_the_phone_records_the_persons_email_and_resolves_to_them(team):
    from core.auth.actor import resolve_recorded_person

    sid = _suggestion(team)
    assert _start_from_the_phone(team, team.owner, sid).status_code == 200
    [watch] = _watches(team)
    assert watch.created_by == f"{team.owner.clerk}@t.test"
    assert resolve_recorded_person(team.db, watch.created_by) == team.owner.pk


def test_a_ticket_in_the_inbox_is_watched_with_no_owning_agent(team):
    sid = _suggestion(team, payload={})
    assert _start(team, team.owner, sid).status_code == 200
    [task] = _tickets(team, sid)
    [watch] = _watches(team)
    assert (task.status, watch.target_id, watch.owner_agent_id) == ("inbox", str(task.id), None)
    assert _row(team, sid).started_ref == {"board_task_id": task.id, "watch_id": str(watch.id)}


def test_a_chat_start_makes_no_watch(team):
    sid = _suggestion(team, kind="chat")
    assert _start(team, team.owner, sid).status_code == 200
    assert _watches(team) == [] and _row(team, sid).started_ref is None


def test_a_refused_or_second_start_makes_no_watch(team, monkeypatch):
    from services import daily_spend_guard

    refused, started = _suggestion(team), _suggestion(team)
    with monkeypatch.context() as ceiling:
        ceiling.setattr(daily_spend_guard, "ceiling_usd", lambda db: 1.0)
        ceiling.setattr(daily_spend_guard, "spent_today_usd", lambda db, workspace_id: 2.5)
        assert _start(team, team.owner, refused).json()["code"] == "spend_ceiling"
    assert _watches(team) == []
    assert _start(team, team.owner, started).status_code == 200
    assert _start(team, team.editor, started).json()["code"] == "already_started"
    assert len(_watches(team)) == 1


def test_a_watch_that_cannot_be_written_fails_the_start_whole(team, monkeypatch):
    from sqlalchemy.exc import OperationalError

    from services.watch_service import WatchService

    def down(*args, **kwargs):
        raise OperationalError("INSERT INTO watches ...", {}, Exception("the database went away"))

    monkeypatch.setattr(WatchService, "create_watch", staticmethod(down))
    sid = _suggestion(team)
    response = _start(team, team.owner, sid)
    assert response.status_code == 503 and response.json()["code"] == "unavailable"
    row = _row(team, sid)
    assert (row.status, row.started_ref, row.decided_at) == ("open", None, None)
    assert _tickets(team, sid) == [] and _watches(team) == []
    assert team.db.execute(text("SELECT count(*) FROM audit_logs WHERE resource_type = 'proactive_suggestion' "
                                "AND resource_id = :sid"), {"sid": str(sid)}).scalar() == 0
