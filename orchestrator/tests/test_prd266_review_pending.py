"""PRD-266 US-012a (§4.4, D22, K10) — 'Ready for review': one notice each time a ticket ENTERS Review.

On Postgres with committing sessions (the listener acts on a real commit), ``NotificationDispatcher
.dispatch`` recording each call, and ``send_soon`` running inline (a sync test runs no loop, so
``dispatch_entries`` runs inside the commit, on a session of the test's own). Proven here:

* Every ORM writer: in_progress → review committed raises one ``review_pending`` linked to the
  ticket (its title, its label, the workspace's agent); review → review raises none; review →
  done → review raises two in all; a ticket filed in Review raises one; a move rolled back, or
  undone before the flush, raises none; two tickets in one commit raise two; a long title is cut;
  a ticket expired by its commit (production's sessions) still counts; a session closed without
  a commit forgets its entry.
* Savepoints: a savepoint's rollback keeps an earlier entry; a released savepoint waits for the
  outer commit, and an outer rollback, or a rollback with a savepoint still open, drops it.
* A batch: one ticket's failed write takes back only its own notice.
* Raw SQL: the sweeper's delivered tickets are announced one each; ``_reopen_review`` is no entry.
* The re-read: a ticket out of Review by the time its notice runs, and a Mission's own card, are
  not announced; a Mission step's card is.
* A dispatcher or an ``announce`` that raises never reaches the commit; the listener is
  registered once; the event is valid and a push type; ``send_soon`` tracks its task on a running
  loop, runs inline without one, and never raises.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event, text

DELIVERED_REF = '{"deliverables": [{"path": "sessions/rota.md"}]}'


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the review_pending tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(engine, new_session):
    """A throwaway workspace with one agent: ``(workspace id, agent id)``. Swept on teardown."""
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd266-review')"), {"id": ws})
    agent_id = s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status) "
        "VALUES ('Reviewer Rae', 'custom', CAST(:w AS uuid), 'active') RETURNING id"), {"w": ws}).scalar()
    s.commit()
    yield ws, agent_id
    s = new_session.sweep()
    for table in ("board_tasks", "workspace_ticket_counters", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


@pytest.fixture
def told(monkeypatch, new_session):
    """Every dispatch, recorded; ``dispatch_entries`` opens the test's own sessions."""
    from core.services.notification_dispatcher import NotificationDispatcher
    from services.notification_events import review_pending

    calls = []

    async def record(self, **kwargs):
        calls.append({"workspace_id": self.workspace_id, **kwargs})
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", record)
    monkeypatch.setattr(review_pending, "_open_session", new_session)
    return calls


def _mine(calls, ws):
    """The calls for this workspace (the sweeper sweeps every workspace's tickets)."""
    return [c for c in calls if c["workspace_id"] == ws and c["event_type"] == "review_pending"]


def _file(new_session, ws, status="in_progress", title="Check the rota", **over):
    """A ticket filed and committed on a session of its own: ``(session, ticket)``."""
    from core.models.core import BoardTask

    s = new_session()
    task = BoardTask(workspace_id=ws, title=title, status=status, **over)
    s.add(task)
    s.commit()
    return s, task


def _status(new_session, task_id):
    return new_session().execute(text("SELECT status FROM board_tasks WHERE id = :i"), {"i": task_id}).scalar()


# --------------------------------------------------------------------------- every ORM writer


def test_a_ticket_moved_into_review_raises_one_notice_linked_to_it(workspace, told, new_session):
    ws, agent_id = workspace
    s, task = _file(new_session, ws, assigned_agent_id=agent_id)

    task.status = "review"
    s.commit()

    assert _mine(told, ws) == [{
        "workspace_id": ws, "event_type": "review_pending", "title": "Ready for review: Check the rota",
        "message": f"Ticket #{task.workspace_seq:04d}", "link_type": "task", "link_id": str(task.id),
        "agent_id": agent_id, "agent_name": "Reviewer Rae", "status": "action_required",
    }]


def test_review_to_review_raises_nothing(workspace, told, new_session):
    ws, _ = workspace
    s, task = _file(new_session, ws, status="review")
    told.clear()

    task.status = "review"
    task.title = "Check the rota again"          # the ticket is dirty; its status did not move
    s.commit()

    assert _mine(told, ws) == []


def test_review_then_done_then_review_raises_two_in_all(workspace, told, new_session):
    ws, _ = workspace
    s, task = _file(new_session, ws)

    for status in ("review", "done", "review"):
        task.status = status
        s.commit()

    assert [c["link_id"] for c in _mine(told, ws)] == [str(task.id)] * 2


def test_a_ticket_filed_in_review_raises_one(workspace, told, new_session):
    ws, _ = workspace

    _, task = _file(new_session, ws, status="review")

    assert [c["link_id"] for c in _mine(told, ws)] == [str(task.id)]


def test_a_move_rolled_back_raises_nothing(workspace, told, new_session):
    ws, _ = workspace
    s, task = _file(new_session, ws)

    task.status = "review"
    s.flush()
    s.rollback()
    task.title = "Check the rota tomorrow"       # a later commit carries no forgotten entry
    s.commit()

    assert _mine(told, ws) == [] and _status(new_session, task.id) == "in_progress"


def test_a_move_undone_before_the_flush_raises_nothing(workspace, told, new_session):
    ws, _ = workspace
    s, task = _file(new_session, ws)

    task.status = "review"
    task.status = "done"
    s.commit()

    assert _mine(told, ws) == []


def test_two_tickets_in_one_commit_raise_two(workspace, told, new_session):
    from core.models.core import BoardTask

    ws, _ = workspace
    s = new_session()
    s.add_all([BoardTask(workspace_id=ws, title=title, status="review") for title in ("Rota", "Menu")])
    s.commit()

    assert sorted(c["title"] for c in _mine(told, ws)) == ["Ready for review: Menu", "Ready for review: Rota"]


def test_a_ticket_expired_by_its_commit_still_counts_its_move(workspace, told, new_session):
    """Production sessions expire on commit: setting the status loads the stored value first."""
    from core.models.core import BoardTask

    ws, _ = workspace
    s = new_session(expire_on_commit=True)
    task = BoardTask(workspace_id=ws, title="Check the rota", status="in_progress")
    s.add(task)
    s.commit()

    task.status = "review"
    s.commit()

    assert [c["link_id"] for c in _mine(told, ws)] == [str(task.id)]


def test_a_session_closed_without_a_commit_forgets_its_entry(workspace, told, new_session):
    ws, _ = workspace
    s, task = _file(new_session, ws)
    task.status = "review"
    s.flush()
    s.close()                                    # rolled back; the session can be used again
    other = new_session()
    moved = other.get(type(task), task.id)
    moved.status = "review"
    other.commit()                               # announced once, by the writer that moved it

    s.execute(text("SELECT 1"))
    s.commit()                                   # the reused session has nothing to announce

    assert [c["link_id"] for c in _mine(told, ws)] == [str(task.id)]


def test_a_long_title_is_cut_to_200_characters(workspace, told, new_session):
    ws, _ = workspace

    _file(new_session, ws, status="review", title="Rota " * 80)

    [call] = _mine(told, ws)
    assert len(call["title"]) == 200
    assert call["title"].startswith("Ready for review: Rota") and call["title"].endswith("…")


# --------------------------------------------------------------------------- savepoints


def test_a_savepoints_rollback_keeps_an_earlier_entry(workspace, told, new_session):
    from core.models.core import BoardTask

    ws, _ = workspace
    s = new_session()
    first, second = (BoardTask(workspace_id=ws, title=title, status="in_progress") for title in ("Rota", "Menu"))
    s.add_all([first, second])
    s.commit()

    first.status = "review"
    s.flush()
    savepoint = s.begin_nested()
    second.status = "review"
    s.flush()
    savepoint.rollback()                         # takes back only the second move
    s.commit()

    assert [c["link_id"] for c in _mine(told, ws)] == [str(first.id)]


def test_a_released_savepoint_waits_for_the_outer_commit(workspace, told, new_session):
    ws, _ = workspace
    s, task = _file(new_session, ws)

    with s.begin_nested():
        task.status = "review"
    released = list(_mine(told, ws))
    s.commit()

    assert released == [] and [c["link_id"] for c in _mine(told, ws)] == [str(task.id)]


def test_an_outer_rollback_drops_a_released_savepoints_entry(workspace, told, new_session):
    ws, _ = workspace
    s, task = _file(new_session, ws)

    with s.begin_nested():
        task.status = "review"
    s.rollback()
    s.commit()

    assert _mine(told, ws) == []


def test_a_rollback_with_a_savepoint_still_open_drops_every_entry(workspace, told, new_session):
    ws, _ = workspace
    s, task = _file(new_session, ws)

    task.status = "review"
    s.flush()
    s.begin_nested()
    s.rollback()                                 # the savepoint, then the session's transaction
    s.commit()

    assert _mine(told, ws) == []


# --------------------------------------------------------------------------- raw SQL


def test_the_sweepers_delivered_tickets_are_announced_one_each(workspace, told, new_session, tmp_path,
                                                              monkeypatch):
    from config import config
    from services import board_dispatcher

    ws, _ = workspace
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    ids = [_file(new_session, ws, title=title, attempts=2)[1].id for title in ("Rota", "Menu")]
    s = new_session()
    s.execute(text(
        "UPDATE board_tasks SET lease_until = :past, runtime_ref = CAST(:ref AS jsonb) WHERE id = ANY(:ids)"),
        {"past": datetime.now(timezone.utc) - timedelta(minutes=5), "ref": DELIVERED_REF, "ids": ids})
    s.commit()

    out = board_dispatcher.requeue_expired_leases(s, max_attempts=2)

    assert set(ids) <= set(out["delivered"])
    assert sorted(int(c["link_id"]) for c in _mine(told, ws)) == sorted(ids)


def test_reopening_review_after_a_failed_approval_is_not_an_entry(workspace, told, new_session):
    from api.board_tasks import _reopen_review

    ws, _ = workspace
    decided = datetime.now(timezone.utc).replace(microsecond=0)
    s, task = _file(new_session, ws, status="done", completed_at=decided)

    _reopen_review(s, task.id, decided_at=decided, finished_before=None)

    assert _status(new_session, task.id) == "review" and _mine(told, ws) == []


# --------------------------------------------------------------------------- the re-read


def test_a_ticket_out_of_review_when_its_notice_runs_is_not_announced(workspace, told, new_session, monkeypatch):
    from services.notification_events import review_pending

    ws, _ = workspace
    held = []
    monkeypatch.setattr(review_pending, "send_soon", lambda coro, *, what: held.append(coro))
    s, task = _file(new_session, ws)
    task.status = "review"
    s.commit()
    task.status = "done"
    s.commit()

    asyncio.run(held[0])

    assert len(held) == 1 and _mine(told, ws) == []


def test_a_missions_own_card_is_not_announced_and_a_steps_card_is(workspace, told, new_session):
    ws, _ = workspace
    card_session, card = _file(new_session, ws, title="Winter menu", source_type="orchestration")
    step_session, step = _file(new_session, ws, title="Draft the menu", source_type="orchestration_task",
                               parent_task_id=card.id)

    card.status = "review"                       # its Mission awaits plan approval: already announced
    card_session.commit()
    step.status = "review"
    step_session.commit()

    assert [(c["link_id"], c["message"]) for c in _mine(told, ws)] == [
        (str(step.id), f"Ticket #{card.workspace_seq:04d}.1")]


# --------------------------------------------------------------------------- never into the commit


def test_a_dispatcher_that_raises_never_reaches_the_commit(workspace, told, new_session, monkeypatch):
    from core.services.notification_dispatcher import NotificationDispatcher

    async def bell_down(self, **kwargs):
        raise RuntimeError("the bell is down")

    monkeypatch.setattr(NotificationDispatcher, "dispatch", bell_down)
    ws, _ = workspace
    s, task = _file(new_session, ws)

    task.status = "review"
    s.commit()

    assert _status(new_session, task.id) == "review"


def test_one_tickets_failed_write_leaves_the_batchs_other_notices(workspace, told, new_session, monkeypatch):
    """A failed statement aborts only that ticket's savepoint: the next ticket is still read and told."""
    from core.models.core import BoardTask
    from core.services.notification_dispatcher import NotificationDispatcher

    async def fails_for_the_rota(self, **kwargs):
        if kwargs["title"] == "Ready for review: Rota":
            self.db.execute(text("SELECT * FROM prd266_no_such_table"))
        told.append({"workspace_id": self.workspace_id, **kwargs})
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", fails_for_the_rota)
    ws, _ = workspace
    s = new_session()
    s.add_all([BoardTask(workspace_id=ws, title=title, status="review") for title in ("Rota", "Menu")])
    s.commit()

    assert [c["title"] for c in _mine(told, ws)] == ["Ready for review: Menu"]


def test_an_announce_that_raises_never_reaches_the_commit(workspace, told, new_session, monkeypatch):
    from services.notification_events import review_pending

    def broken(entries):
        raise RuntimeError("no announcer")

    monkeypatch.setattr(review_pending, "announce", broken)
    ws, _ = workspace
    s, task = _file(new_session, ws)

    task.status = "review"
    s.commit()

    assert _status(new_session, task.id) == "review"


def test_the_listener_is_registered_once(workspace, told, new_session):
    from sqlalchemy.orm import Session

    from core.models.core import BoardTask

    models = importlib.import_module("core.models")
    importlib.import_module("core.models")
    entries = importlib.import_module("core.models.review_entries")
    assert models.review_entries is entries
    assert event.contains(BoardTask.status, "set", entries._flag_entry)
    assert event.contains(Session, "after_commit", entries._announce_entries)
    ws, _ = workspace
    s, task = _file(new_session, ws)

    task.status = "review"
    s.commit()

    assert len(_mine(told, ws)) == 1


# --------------------------------------------------------------------------- the event, send_soon


def test_review_pending_is_a_valid_event_and_a_push_type():
    from core.services.notification_dispatcher import VALID_EVENT_TYPES
    from services.push.payload import PUSH_TYPE_EVENTS

    assert "review_pending" in VALID_EVENT_TYPES and PUSH_TYPE_EVENTS["review"] == ("review_pending",)


def test_send_soon_runs_inline_without_a_loop_and_tracks_a_task_on_one():
    from services.notification_events import soon

    ran = []

    async def note(tag):
        ran.append(tag)

    soon.send_soon(note("inline"), what="test")
    assert ran == ["inline"]
    before = set(soon._PENDING)

    async def on_a_loop():
        soon.send_soon(note("task"), what="test")
        tracked = len(soon._PENDING - before)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return tracked

    assert asyncio.run(on_a_loop()) == 1
    assert ran == ["inline", "task"] and not soon._PENDING - before


def test_send_soon_never_raises_and_closes_what_it_cannot_run(monkeypatch):
    from services.notification_events import soon

    async def fails():
        raise RuntimeError("down")

    soon.send_soon(fails(), what="test")         # the notice raised inline; send_soon did not

    async def never_run():
        return None

    coro = never_run()

    def no_loop(_coro):
        raise RuntimeError("cannot start a loop")

    monkeypatch.setattr(soon.asyncio, "run", no_loop)
    soon.send_soon(coro, what="test")

    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
