"""F241 (night 7): Auto acts on the ticket the owner names, only that ticket, and the
change is recorded on it.

Auto's calls carried "#0175" as 175 or "0175", and every one was read as an id
("Task 175 not found"). By the owner's count it found 1 card in 11 by number,
reassigned none and cancelled none. Asked to cancel #0156 by its title, it moved
#0014 (last week's, closed) to Done, and nothing on #0014 said so.
"""
from __future__ import annotations

import asyncio
import uuid

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
        pytest.skip(f"the F241 tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def shop(engine, new_session, monkeypatch):
    """A workspace with Auto, an Analyst, and the board's notices switched off."""
    import modules.tools.discovery.handlers_board_tasks as handlers

    for quiet in ("_notify_board_safe", "_notify_dispatch_safe", "_consent_for_chat_filed"):
        monkeypatch.setattr(handlers, quiet, lambda *a, **k: None)
    ws = uuid.uuid4()
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f241')"), {"id": str(ws)})
    agents = {}
    for name in ("Auto", "Analyst"):
        agents[name] = s.execute(text(
            "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
            "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json), 'workspace') RETURNING id"),
            {"n": name, "w": str(ws)}).scalar()
    s.commit()
    yield type("Shop", (), {"ws": ws, "agents": agents, "handlers": handlers})
    s = new_session.sweep()
    for table in ("board_tasks", "workspace_ticket_counters", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:w AS uuid)"), {"w": str(ws)})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": str(ws)})
    s.commit()


def _file(new_session, ws, title, **over):
    from core.models.core import BoardTask

    s = new_session()
    task = BoardTask(workspace_id=ws, title=title, status=over.pop("status", "inbox"), **over)
    s.add(task)
    s.commit()
    return task


def _row(new_session, task_id):
    return new_session().execute(text("SELECT status, assigned_agent_id, runtime_ref FROM board_tasks WHERE id = :i"),
                                 {"i": task_id}).first()


def _call(handler, new_session, ws, params):
    return asyncio.run(handler(new_session(), ws, params))


def test_a_number_without_its_hash_is_the_number(shop, new_session):
    _file(new_session, shop.ws, "Price list")
    labels = _file(new_session, shop.ws, "Christmas gift box labels")
    card = _file(new_session, shop.ws, "Winter menu", source_type="orchestration")
    step = _file(new_session, shop.ws, "Draft it", source_type="orchestration_task", parent_task_id=card.id)

    for said in ("#0002", "0002", "2", 2):              # night 7: "#0175" arrived as 175 and as "0175"
        out = _call(shop.handlers.get_board_task, new_session, shop.ws, {"task_id": said})
        assert out["success"] is True and out["task"]["id"] == labels.id, said
    for said in ("#0003.1", "0003.1", "3.1"):            # a mission step, its '#' gone
        out = _call(shop.handlers.get_board_task, new_session, shop.ws, {"task_id": said})
        assert out["task"]["id"] == step.id, said


def test_a_bare_number_that_is_two_tickets_is_refused_naming_both(shop, new_session):
    one = _file(new_session, shop.ws, "Price list")
    two = _file(new_session, shop.ws, "Roast schedule")
    s = new_session()
    s.execute(text("UPDATE board_tasks SET workspace_seq = :n WHERE id = :i"), {"n": one.id, "i": two.id})
    s.commit()

    out = _call(shop.handlers.get_board_task, new_session, shop.ws, {"task_id": one.id})

    assert out["success"] is False
    assert f"#{one.id:04d} ('Roast schedule')" in out["error"] and "#0001 ('Price list')" in out["error"]
    named = _call(shop.handlers.get_board_task, new_session, shop.ws, {"task_id": f"#{one.id}"})
    assert named["task"]["id"] == two.id                 # with its '#', it is the number


def test_a_ticket_that_is_neither_says_its_number(shop, new_session):
    _file(new_session, shop.ws, "Price list")
    out = _call(shop.handlers.update_board_task_status, new_session, shop.ws, {"task_id": "175", "status": "cancelled"})
    assert out["success"] is False and "No ticket #0175" in out["error"]


def test_a_closed_ticket_is_never_moved(shop, new_session):
    """Night 7: asked to cancel #0156, Auto moved last week's #0014 from Closed to Done."""
    old = _file(new_session, shop.ws, "Christmas gift box labels - 40 of them", status="closed")
    mine = _file(new_session, shop.ws, "Christmas gift box labels - 40 of them (wait for me)")

    out = _call(shop.handlers.update_board_task_status, new_session, shop.ws, {"task_id": "#0001", "status": "done"})
    bulk = _call(shop.handlers.update_board_task_status, new_session, shop.ws,
                 {"task_ids": ["#0002", "#0001"], "status": "cancelled"})

    assert out["success"] is False and "Ticket #0001 ('Christmas gift box labels - 40 of them') is closed" in out["error"]
    assert bulk["success"] is False and "Nothing was changed" in bulk["error"]
    assert _row(new_session, old.id).status == "closed" and _row(new_session, mine.id).status == "inbox"


def test_each_change_is_noted_on_its_ticket_with_who_made_it(shop, new_session):
    from core.llm.usage_context import LANE_CHAT, usage_scope

    task = _file(new_session, shop.ws, "Margin per bag for last week's coffees")
    with usage_scope(request_type=LANE_CHAT, agent_id=shop.agents["Auto"]):
        given = _call(shop.handlers.assign_board_task, new_session, shop.ws,
                      {"task_id": "0001", "agent_name": "Analyst"})
        edited = _call(shop.handlers.update_board_task, new_session, shop.ws,
                       {"task_id": "#0001", "title": "Margin per bag, week 40"})
        cancelled = _call(shop.handlers.update_board_task_status, new_session, shop.ws,
                          {"task_id": 1, "status": "cancelled"})

    assert given["success"] and edited["success"] and cancelled["success"]
    row = _row(new_session, task.id)
    notes = [(n["by"], n["note"]) for n in row.runtime_ref["session_notes"]]
    assert notes == [("Auto", "Gave this to Analyst, in chat."), ("Auto", "Changed its title, in chat."),
                     ("Auto", "Moved this from Assigned to Cancelled, in chat.")]
    assert row.status == "cancelled" and row.assigned_agent_id == shop.agents["Analyst"]


def test_a_call_that_changes_nothing_leaves_no_note(shop, new_session):
    task = _file(new_session, shop.ws, "Price list")
    out = _call(shop.handlers.update_board_task_status, new_session, shop.ws, {"task_id": "#0001", "status": "inbox"})
    assert out["success"] is True and "session_notes" not in (_row(new_session, task.id).runtime_ref or {})


def test_auto_is_told_how_to_cancel_a_ticket():
    from modules.tools.discovery.action_registry import get_action_registry

    status = get_action_registry().get("platform_update_task_status")
    assert "To cancel a ticket set 'cancelled', never 'done'" in status.description
    assert "platform_cancel_scheduled_task is for timers, not tickets" in status.description
