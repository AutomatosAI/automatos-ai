"""PRD-274 Wave 1 follow-up: a callback a restart left pending is sent again.

Delivery attempts run in-process (#1462), so a restart mid-retry leaves the row ``pending``. A job
on the shared scheduler re-sends, every ten minutes, each callback still pending five minutes or
more after its ticket ended; a ticket still running, or one that just ended, is left to the normal
delivery. Real database; delivery itself is a fake.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from services import run_callbacks as rc


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.run_callbacks import RunCallback

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    RunCallback.__table__.create(eng, checkfirst=True)
    yield eng
    eng.dispose()


@pytest.fixture
def tickets(new_session):
    """Five tickets with pending callbacks: how long ago each ended (None: still running)."""
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1-resend')"), {"id": ws})
    made = {}
    for name, status, ended_ago, callback in (("cut_off", "done", 900, "pending"), ("failed_run", "failed", 900, "pending"),
                                              ("just_ended", "done", 30, "pending"), ("running", "in_progress", None, "pending"),
                                              ("delivered", "done", 900, "delivered")):
        task_id = s.execute(text(
            "INSERT INTO board_tasks (workspace_id, title, status, completed_at) VALUES (CAST(:w AS uuid), :t, :st, "
            "CASE WHEN CAST(:ago AS integer) IS NULL THEN NULL ELSE now() - make_interval(secs => :ago) END) RETURNING id"),
            {"w": ws, "t": name, "st": status, "ago": ended_ago}).scalar()
        s.execute(text("INSERT INTO run_callbacks (id, workspace_id, task_id, url, status) VALUES "
                       "(gen_random_uuid(), CAST(:w AS uuid), :t, 'https://hooks.example.com/runs', :st)"),
                  {"w": ws, "t": task_id, "st": callback})
        made[name] = task_id
    s.commit()
    s.close()
    yield made
    s = new_session.sweep()
    for table in ("run_callbacks", "board_tasks"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def test_only_callbacks_cut_off_after_their_ticket_ended_are_sent_again(new_session, tickets, monkeypatch):
    sent = []

    async def deliver(task_id, run_id=None):
        sent.append(task_id)
        return "delivered"

    monkeypatch.setattr(rc, "deliver_run_callback", deliver)
    asyncio.run(rc.resend_pending_callbacks())
    mine = set(tickets.values())
    assert {t for t in sent if t in mine} == {tickets["cut_off"], tickets["failed_run"]}


def test_the_shared_scheduler_runs_the_resend_once_it_starts():
    added = []
    scheduler = SimpleNamespace(add_job=lambda fn, trigger, **kw: added.append((fn, trigger, kw)))

    @rc.schedules_the_resend
    def start(self):
        self._scheduler = scheduler

    start(SimpleNamespace())
    ((fn, trigger, kw),) = added
    assert fn is rc.resend_pending_callbacks and trigger == "interval"
    assert kw["seconds"] == rc.RESEND_EVERY_SECONDS and kw["id"] == rc.RESEND_JOB_ID and kw["replace_existing"]


def test_the_unified_scheduler_carries_it():
    from services.scheduler import UnifiedScheduler

    assert UnifiedScheduler.start.__code__.co_name == "started"
