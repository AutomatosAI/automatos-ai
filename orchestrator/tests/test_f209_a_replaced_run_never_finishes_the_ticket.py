"""F209 — a run that no longer holds its ticket never finishes it.

Since 19 Sep, 3 of 892 tickets got two or three task_complete notices, each with
a different result (#296, #484, #1093): the same ticket ran more than once. Run
Now read "not running" without a lock, reset the ticket under a run the
dispatcher had just claimed, and the dispatcher claimed it again; each run then
finalized it, and the later write won. Every start now stamps a run id on the
ticket, a redispatch takes the row lock and clears it, and finalize (and the
lease heartbeat) acts only for the run that holds the ticket.

Review (26 Sep): the locks never wait (a sync wait on the event loop froze every
request, F105), a drag re-reads the ticket under its lock before it starts a run,
and a Claude Code ticket's claims are numbered so a replaced session's result is
refused.
"""
from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the run-id tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def ticket(engine, new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f209')"), {"id": ws})
    agent = s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status) "
        "VALUES ('Club Secretary', 'custom', CAST(:w AS uuid), 'active') RETURNING id"), {"w": ws}).scalar()
    task = s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, assigned_agent_id, source_type, attempts) "
        "VALUES (CAST(:w AS uuid), 'Draft the October club email', 'assigned', :a, 'user', 0) RETURNING id"),
        {"w": ws, "a": agent}).scalar()
    s.commit()
    yield NS(id=task, ws=ws, agent=agent, new=new_session)
    s = new_session.sweep()
    for table, col in (("board_tasks", "workspace_id"), ("agents", "workspace_id"), ("workspaces", "id")):
        s.execute(text(f"DELETE FROM {table} WHERE {col} = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _row(ticket):
    return ticket.new().execute(text(
        "SELECT status, result, runtime_ref->>'run_id' AS run_id FROM board_tasks WHERE id = :i"),
        {"i": ticket.id}).first()


def _claim(ticket):
    """The dispatcher's claim; the run id it stamped (None on code that stamps none)."""
    from services import board_dispatcher

    s = ticket.new()
    board_dispatcher.claim_tasks(s, worker_id="w-f209", limit=50, lease_seconds=600, workspace_id=ticket.ws)
    s.close()
    return _row(ticket).run_id


def _supports_run_id(fn):
    return "run_id" in inspect.signature(fn).parameters


def _finalize(ticket, run_id, result):
    from api import board_tasks as bt

    extra = {"run_id": run_id} if _supports_run_id(bt.finalize_board_task_run) else {}
    return asyncio.run(bt.finalize_board_task_run(
        ticket.new(), task_id=ticket.id, workspace_id=ticket.ws, agent_id=ticket.agent,
        exec_result={"status": "success", "result": result}, **extra))


def test_a_replaced_run_never_writes_over_the_run_that_holds_the_ticket(ticket):
    from api import board_tasks as bt
    from core.models.core import BoardTask

    first = _claim(ticket)
    lapse = ticket.new()                               # its worker looks dead: the lease has lapsed
    lapse.execute(text("UPDATE board_tasks SET lease_until = now() - interval '1 minute' WHERE id = :i"),
                  {"i": ticket.id})
    lapse.commit()
    redo = ticket.new()
    assert bt._redispatch_task(redo, redo.get(BoardTask, ticket.id)) is not False   # Run Now
    second = _claim(ticket)
    told = []

    async def _told(db, workspace_id, task):
        told.append(task.result)

    async def _no_report(*a, **k):
        return None

    with patch("api.board_tasks._dispatch_task_complete", _told), \
            patch("api.board_tasks._auto_create_task_report", _no_report):
        replaced = _finalize(ticket, first, "Draft 1, from the run that was replaced")
        current = _finalize(ticket, second, "Draft 2")

    row = _row(ticket)
    assert (row.status, row.result) == ("done", "Draft 2")          # night: the replaced run's draft stood
    assert told == ["Draft 2"] and replaced is None and current == "done"


def test_run_now_cannot_reset_a_ticket_a_run_has_just_claimed(ticket):
    from api import board_tasks as bt
    from core.models.core import BoardTask

    stale = ticket.new()                                # in progress, lease lapsed: nothing holds it
    stale.execute(text("UPDATE board_tasks SET status = 'in_progress', lease_until = now() - interval '1 minute' "
                       "WHERE id = :i"), {"i": ticket.id})
    stale.commit()
    run_now = ticket.new()
    seen = run_now.get(BoardTask, ticket.id)            # Run Now reads it: not running
    assert not bt._running_now(run_now, seen)
    live = ticket.new()                                 # meanwhile a run claims it and holds a live lease
    live.execute(text("UPDATE board_tasks SET lease_until = now() + interval '10 minutes', "
                      "runtime_ref = '{\"run_id\": \"live\"}'::jsonb WHERE id = :i"), {"i": ticket.id})
    live.commit()

    reset = bt._redispatch_task(run_now, seen)

    row = _row(ticket)
    assert (row.status, row.run_id) == ("in_progress", "live")      # night: reset to assigned under the run
    assert reset is False


def test_a_replaced_runs_heartbeat_keeps_nothing_alive(ticket):
    from services.board_dispatcher import renew_lease

    run = _claim(ticket)
    s = ticket.new()
    stale = {"run_id": "a-run-that-was-replaced"} if _supports_run_id(renew_lease) else {}

    assert renew_lease(s, ticket.id, lease_seconds=600, **stale) is False     # night: any run renewed it
    assert renew_lease(s, ticket.id, lease_seconds=600, run_id=run) is True


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def test_a_drag_into_in_progress_launches_the_run_it_stamped(monkeypatch):
    from uuid import UUID

    from api import board_tasks as bt
    from core.models.core import BoardTask
    from tests.test_board_task_handlers import _FakeSession

    launched = []
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: launched.append(kw))
    monkeypatch.setattr(bt, "record_operator_consent", lambda *a, **k: None)
    monkeypatch.setattr(bt, "notify_board_event", lambda *a, **k: None)
    ws = UUID("00000000-0000-0000-0000-0000000000c1")
    task = BoardTask(id=51, workspace_id=ws, title="Weekly numbers", status="assigned", assigned_agent_id=5,
                     source_type="user", review_mode="auto", planning_data={}, runtime_ref={"session_id": "s-1"})

    asyncio.run(bt.update_task_status(51, _Req({"status": "in_progress"}), ctx=NS(
        workspace_id=ws, user=NS(id=1, clerk_user_id="u1", email="owner@cafe.test")),
        db=_FakeSession(agent=NS(id=5), task=task)))

    assert launched and launched[0].get("run_id") == task.runtime_ref.get("run_id") is not None   # night: no run id
    assert task.runtime_ref["session_id"] == "s-1"


# ── Review: the locks that close the race never wait on the event loop ──────────

def _held(ticket):
    """Another writer holding the ticket's row (a run finishing, a claim) until the
    caller rolls it back."""
    s = ticket.new()
    s.execute(text("SELECT id FROM board_tasks WHERE id = :i FOR UPDATE"), {"i": ticket.id})
    return s


def _impatient(ticket):
    """A request's session whose lock waits give up after 3 s, so code that waits
    fails the test instead of hanging it."""
    s = ticket.new()
    s.execute(text("SET LOCAL lock_timeout = '3s'"))
    return s


def _owner(ticket):
    return NS(workspace_id=uuid.UUID(ticket.ws), user=NS(id=1, clerk_user_id="u1", email="owner@club.test"))


def test_run_now_on_a_ticket_being_finished_answers_at_once(ticket):
    from api import board_tasks as bt
    from core.models.core import BoardTask

    failed = ticket.new()
    failed.execute(text("UPDATE board_tasks SET status = 'failed', attempts = 2 WHERE id = :i"), {"i": ticket.id})
    failed.commit()
    finishing = _held(ticket)
    run_now = _impatient(ticket)
    began = time.monotonic()
    try:
        reset = bt._redispatch_task(run_now, run_now.get(BoardTask, ticket.id))
    finally:
        finishing.rollback()

    assert reset is False and time.monotonic() - began < 1.0     # 51ae84860: waited on the row (a frozen loop)
    assert _row(ticket).status == "failed"


@pytest.mark.parametrize("route", ["update_task_status", "update_task"])
def test_a_drag_that_lands_as_the_dispatcher_claims_starts_no_second_run(ticket, route, monkeypatch):
    from api import board_tasks as bt

    launched, claimed = [], {}
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: launched.append(kw))
    monkeypatch.setattr(bt, "record_operator_consent", lambda *a, **k: None)

    class _BodyAfterTheClaim:                 # the body arrives after the route read the ticket
        async def json(self):
            claimed["run"] = _claim(ticket)   # ...and the dispatcher claimed it in that gap
            return {"status": "in_progress"}

    asyncio.run(getattr(bt, route)(ticket.id, _BodyAfterTheClaim(), ctx=_owner(ticket), db=ticket.new()))

    row = _row(ticket)
    assert claimed["run"] is not None and (row.status, row.run_id) == ("in_progress", claimed["run"])
    assert launched == []                     # 51ae84860: a second run, under a run id of its own


@pytest.mark.parametrize("route", ["update_task_status", "update_task"])
def test_a_drag_onto_a_ticket_being_finished_is_told_to_retry_at_once(ticket, route, monkeypatch):
    from fastapi import HTTPException

    from api import board_tasks as bt

    launched = []
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: launched.append(kw))
    monkeypatch.setattr(bt, "record_operator_consent", lambda *a, **k: None)
    finishing = _held(ticket)
    began = time.monotonic()
    try:
        with pytest.raises(HTTPException) as refused:
            asyncio.run(getattr(bt, route)(ticket.id, _Req({"status": "in_progress"}), ctx=_owner(ticket),
                                           db=_impatient(ticket)))
    finally:
        finishing.rollback()

    assert refused.value.status_code == 409 and time.monotonic() - began < 1.0   # 51ae84860: waited
    assert launched == [] and _row(ticket).status == "assigned"


# ── Review: a Claude Code ticket's replaced session never finishes it ───────────

@pytest.fixture
def cli_ticket(engine, new_session, monkeypatch):
    """A Claude Code agent's ticket and a paired host (the realdb CLI suite's shape)."""
    from api import board_tasks as bt
    from services import cli_host_service as svc

    async def _quiet(*a, **k):
        return None

    monkeypatch.setattr(bt, "_board_task_blocked_pending_approval", lambda *a, **k: False)
    monkeypatch.setattr(bt, "_dispatch_task_complete", _quiet)
    monkeypatch.setattr(bt, "_dispatch_task_failed", _quiet)
    monkeypatch.setattr(bt, "_auto_create_task_report", _quiet)
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f209-cli')"), {"id": ws})
    agent = s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) VALUES "
        "('Club Secretary', 'custom', CAST(:w AS uuid), 'active', "
        "CAST('{\"runtime\": \"cli\", \"provider\": \"claude\"}' AS json)) RETURNING id"), {"w": ws}).scalar()
    task = s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, assigned_agent_id, source_type, attempts) "
        "VALUES (CAST(:w AS uuid), 'Draft the October club email', 'assigned', :a, 'user', 0) RETURNING id"),
        {"w": ws, "a": agent}).scalar()
    s.commit()
    host, _token = svc.pair_host(s, svc.create_pairing_code(s, uuid.UUID(ws), "club-laptop")[1])
    yield NS(id=task, ws=ws, agent=agent, host=host, new=new_session)
    s = new_session.sweep()
    for table, col in (("llm_usage", "workspace_id"), ("board_tasks", "workspace_id"), ("cli_hosts", "workspace_id"),
                       ("agents", "workspace_id"), ("workspaces", "id")):
        s.execute(text(f"DELETE FROM {table} WHERE {col} = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def test_a_replaced_claude_code_session_never_finishes_the_ticket(cli_ticket):
    from api import board_tasks as bt
    from core.models.core import BoardTask
    from services import cli_host_service as svc

    t = cli_ticket
    first = svc.claim_for_host(t.new(), t.host, 1)["tasks"][0]
    away = t.new()                                     # the laptop slept: the lease lapsed
    away.execute(text("UPDATE board_tasks SET lease_until = now() - interval '1 minute' WHERE id = :i"), {"i": t.id})
    away.commit()
    redo = t.new()
    assert bt._redispatch_task(redo, redo.get(BoardTask, t.id)) is not False      # Run Now
    second = svc.claim_for_host(t.new(), t.host, 1)["tasks"][0]

    def _result(claim, text_):
        return asyncio.run(svc.apply_result(t.new(), t.host, t.id, {
            "attempt": claim["attempt"], "status": "success", "result_text": text_}))

    replaced = _result(first, "Draft 1, from the session that was replaced")
    current = _result(second, "Draft 2")

    row = _row(t)
    assert (row.status, row.result) == ("done", "Draft 2")          # 51ae84860: Draft 1 stood
    assert replaced["applied"] is False and current["applied"] is True
    assert row.run_id is not None                                    # the claim kept its run
