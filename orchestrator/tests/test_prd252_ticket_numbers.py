"""PRD-252 R4 — tickets you can tell apart: a number in the workspace, #0042.

Night 6: two tickets with the same title could not be told apart, and Auto named
tickets by their place in a list ("tasks 2 and 3"). Every ticket but a mission
step now takes its workspace's next number when it is inserted, and keeps it; a
step shows its mission card's number and its step (#0051.3, D5). Auto's tools
take the number and answer with it.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

NOW = datetime.now(timezone.utc)
MIGRATION = pathlib.Path(__file__).resolve().parents[1] / "alembic" / "versions" / "prd252_ticket_numbers.py"


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the ticket-number tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspaces(engine, new_session):
    made = []

    def make() -> uuid.UUID:
        ws = uuid.uuid4()
        s = new_session()
        s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd252-r4')"), {"id": str(ws)})
        s.commit()
        made.append(str(ws))
        return ws

    yield make
    s = new_session.sweep()
    for ws in made:
        for table in ("board_tasks", "workspace_ticket_counters"):
            s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _file(new_session, ws, title, **over):
    from core.models.core import BoardTask

    s = new_session()
    task = BoardTask(workspace_id=ws, title=title, status="inbox", **over)
    s.add(task)
    s.commit()
    return task


def test_a_new_ticket_takes_its_workspaces_next_number(workspaces, new_session):
    ws = workspaces()

    seqs = [_file(new_session, ws, f"Rota {n}").workspace_seq for n in range(3)]

    assert seqs == [1, 2, 3]


def test_a_number_is_never_given_twice_even_after_a_delete(workspaces, new_session):
    ws = workspaces()
    _file(new_session, ws, "Price list")
    gone = _file(new_session, ws, "Price list")             # the same title: told apart by number
    s = new_session()
    s.execute(text("DELETE FROM board_tasks WHERE id = :i"), {"i": gone.id})
    s.commit()

    assert _file(new_session, ws, "Supplier check").workspace_seq == 3


def test_each_workspace_counts_on_its_own(workspaces, new_session):
    a, b = workspaces(), workspaces()
    _file(new_session, a, "One")

    assert _file(new_session, b, "One").workspace_seq == 1 and _file(new_session, a, "Two").workspace_seq == 2


def test_a_mission_step_shows_its_cards_number_and_its_step(workspaces, new_session):
    from services.ticket_numbers import ticket_numbers

    ws = workspaces()
    _file(new_session, ws, "Earlier ticket")
    card = _file(new_session, ws, "Launch the autumn blend", source_type="orchestration")
    step = _file(new_session, ws, "Write the menu post", source_type="orchestration_task",
                 parent_task_id=card.id, planning_data={"sequence_number": 3})

    assert step.workspace_seq is None                       # D5: a step takes no number of its own
    numbers = ticket_numbers(new_session(), ws, [card, step])
    assert (numbers[card.id], numbers[step.id]) == ("#0002", "#0002.3")


def test_auto_can_name_a_ticket_by_its_number(workspaces, new_session):
    from services.ticket_numbers import is_number_ref, resolve_ticket_ref

    ws = workspaces()
    first = _file(new_session, ws, "Price list")
    card = _file(new_session, ws, "Winter menu", source_type="orchestration")
    step = _file(new_session, ws, "Draft it", source_type="orchestration_task", parent_task_id=card.id,
                 planning_data={"sequence_number": 1})
    db = new_session()

    assert resolve_ticket_ref(db, ws, "#0001") == resolve_ticket_ref(db, ws, "#1") == first.id
    assert resolve_ticket_ref(db, ws, "#0002.1") == step.id
    assert resolve_ticket_ref(db, ws, "#0099") is None
    assert not is_number_ref("42") and not is_number_ref(42)    # an id stays an id


def test_the_backfill_numbers_old_tickets_in_the_order_they_were_made(workspaces, new_session):
    """Raw inserts bypass the listener, as rows from before the migration did."""
    spec = importlib.util.spec_from_file_location("prd252_ticket_numbers", MIGRATION)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    ws = workspaces()
    s = new_session()
    ids = {}
    for title, age, source in (("Newest", 1, "user"), ("Oldest", 9, "user"), ("A step", 5, "orchestration_task"),
                               ("Middle", 5, "user")):
        ids[title] = s.execute(text(
            "INSERT INTO board_tasks (workspace_id, title, status, source_type, created_at) "
            "VALUES (CAST(:w AS uuid), :t, 'done', :src, :at) RETURNING id"),
            {"w": str(ws), "t": title, "src": source, "at": NOW - timedelta(days=age)}).scalar()
    s.execute(text(migration.BACKFILL))
    s.execute(text(migration.COUNTERS))
    s.commit()

    seq = dict(new_session().execute(text(
        "SELECT title, workspace_seq FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": str(ws)}).all())
    assert (seq["Oldest"], seq["Middle"], seq["Newest"], seq["A step"]) == (1, 2, 3, None)
    assert _file(new_session, ws, "After the migration").workspace_seq == 4


def test_the_board_serves_each_tickets_number(workspaces, new_session):
    from services.board_task_view import enrich_with_agents

    ws = workspaces()
    tasks = [_file(new_session, ws, "Rota"), _file(new_session, ws, "Rota")]

    served = enrich_with_agents(tasks, new_session(), ws)

    assert [t["number"] for t in served] == ["#0001", "#0002"]


def test_autos_tools_take_a_number_and_answer_with_numbers(workspaces, new_session):
    from services.ticket_refs import by_ticket_number

    ws = workspaces()
    first = _file(new_session, ws, "Price list")
    seen = {}

    @by_ticket_number
    async def handler(db, workspace_id, params):
        seen.update(params)
        return {"success": True, "task_id": params["task_id"], "tasks": [{"id": first.id, "title": "Price list"}]}

    out = asyncio.run(handler(new_session(), ws, {"task_id": "#0001"}))
    assert seen["task_id"] == first.id
    assert out["number"] == "#0001" and out["tasks"][0]["number"] == "#0001"

    missing = asyncio.run(handler(new_session(), ws, {"task_id": "#0099"}))
    assert missing["success"] is False and "#0099" in missing["error"]


def test_a_widget_visitor_never_sees_a_number(workspaces, new_session):
    """F155: a number would tell a visitor how many tickets the business has."""
    from core.security.surface import WIDGET, turn_surface
    from services.ticket_refs import by_ticket_number

    ws = workspaces()
    first = _file(new_session, ws, "Price list")

    @by_ticket_number
    async def handler(db, workspace_id, params):
        return {"success": True, "task": {"id": first.id, "title": "Price list", "status": "inbox"}}

    with turn_surface(WIDGET, ("chat", "tasks:read"), None):
        out = asyncio.run(handler(new_session(), ws, {}))
    assert "number" not in out["task"]


def test_the_migration_merges_both_heads_and_survives_create_all():
    source = MIGRATION.read_text(encoding="utf-8")

    assert 'down_revision = ("prd251_wave2", "document_chunks_ingestion_columns")' in source
    assert "ADD COLUMN IF NOT EXISTS workspace_seq" in source and "CREATE TABLE IF NOT EXISTS" in source
    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_board_tasks_workspace_seq" in source
    assert json.dumps(source).count("workspace_seq IS NULL") >= 1     # the backfill numbers only unnumbered rows
