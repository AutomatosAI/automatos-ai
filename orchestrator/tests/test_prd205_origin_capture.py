"""PRD-205 S4 -- origin_chat_id capture at watch/schedule creation.

Covers:
- the pure executor stamp (``stamp_origin_context``): injects for exactly
  the watch/mission/playbook/schedule creation actions, ALWAYS strips a
  caller/LLM-supplied ``_origin_chat_id`` (anti-spoof -- injection
  overwrites), captures ``_created_by`` for platform_schedule_task, and
  never mutates the input params;
- WatchService.create_watch persists origin_chat_id (garbage coerces to
  NULL, never a failed create);
- the platform_create_watch handler passes the injected param through
  (target resolution stubbed);
- auto_create_watch (the mission/playbook launch path) passes it through;
- ScheduledTaskService.create_task INSERTs origin_chat_id + created_by
  (mock-db; the table is migration-only, so the SQL/bind surface is the
  contract here).

Watch DB tests are live-Postgres (PRD-204 stage-1 pattern) with clean
skip; no notification dispatch on any path exercised here.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

from core.database.database import get_database_url


# ---------------------------------------------------------------------------
# Pure: the executor stamp
# ---------------------------------------------------------------------------


def _stamp():
    from modules.tools.discovery.platform_executor import stamp_origin_context

    return stamp_origin_context


STAMPED_ACTIONS = (
    "platform_create_watch",
    "platform_create_mission",
    "platform_execute_playbook",
    "platform_schedule_task",
)


@pytest.mark.parametrize("action", STAMPED_ACTIONS)
def test_stamp_injects_conversation_for_creating_actions(action):
    stamp = _stamp()
    chat_id = str(uuid.uuid4())
    out = stamp(action, {"goal": "x"}, {"conversation_id": chat_id, "user_id": "user_1"})
    assert out["_origin_chat_id"] == chat_id
    assert out["goal"] == "x"


def test_stamp_leaves_other_actions_untouched():
    stamp = _stamp()
    params = {"target_type": "mission", "_origin_chat_id": "spoofed"}
    out = stamp("platform_list_watches", params, {"conversation_id": str(uuid.uuid4())})
    assert out is params  # not a stamped action -- passthrough


@pytest.mark.parametrize("action", STAMPED_ACTIONS)
def test_stamp_overwrites_llm_supplied_value(action):
    """An LLM tool arg can never point delivery at someone else's chat."""
    stamp = _stamp()
    real = str(uuid.uuid4())
    out = stamp(
        action,
        {"_origin_chat_id": str(uuid.uuid4())},  # spoof attempt
        {"conversation_id": real},
    )
    assert out["_origin_chat_id"] == real


@pytest.mark.parametrize(
    "caller_context", [None, {}, {"user_id": "user_1"}]
)
@pytest.mark.parametrize("action", STAMPED_ACTIONS)
def test_stamp_strips_spoof_when_no_trusted_conversation(action, caller_context):
    stamp = _stamp()
    out = stamp(action, {"_origin_chat_id": "spoofed"}, caller_context)
    assert "_origin_chat_id" not in out


def test_stamp_never_mutates_input_params():
    stamp = _stamp()
    params = {"_origin_chat_id": "spoofed", "goal": "x"}
    stamp("platform_create_mission", params, {"conversation_id": str(uuid.uuid4())})
    assert params == {"_origin_chat_id": "spoofed", "goal": "x"}


def test_stamp_captures_creator_for_schedule_task_only():
    stamp = _stamp()
    ctx = {"conversation_id": str(uuid.uuid4()), "user_id": "user_gerard"}

    sched = stamp("platform_schedule_task", {"_created_by": "spoofed"}, ctx)
    assert sched["_created_by"] == "user_gerard"

    # No creator in context -> the spoof is stripped, not kept.
    sched2 = stamp("platform_schedule_task", {"_created_by": "spoofed"}, None)
    assert "_created_by" not in sched2

    # Watch create does NOT gain _created_by from this stamp (the
    # _MISSION_ATTRIBUTED block owns mission attribution).
    watch = stamp("platform_create_watch", {}, ctx)
    assert "_created_by" not in watch


# ---------------------------------------------------------------------------
# ScheduledTaskService.create_task INSERT surface (mock db -- table is
# migration-only, absent from any ORM/create_all schema)
# ---------------------------------------------------------------------------


def test_create_task_binds_origin_and_creator():
    from services.scheduled_task_service import ScheduledTaskService

    ws_id = uuid.uuid4()
    db = MagicMock()
    agents_result = MagicMock()
    agents_result.fetchall.return_value = [
        SimpleNamespace(id=11, name="Auto"),
        SimpleNamespace(id=11, name="Auto"),
    ]
    count_result = MagicMock()
    count_result.scalar.return_value = 0
    insert_result = MagicMock()
    insert_result.fetchone.return_value = SimpleNamespace(
        id=42, created_at=datetime(2026, 7, 17)
    )
    db.execute.side_effect = [agents_result, count_result, insert_result]

    origin = str(uuid.uuid4())
    svc = ScheduledTaskService(db, ws_id)
    out = asyncio.run(
        svc.create_task(
            created_by_agent_id=11,
            target_agent_id=11,
            task_type="one_shot",
            description="daily digest",
            schedule="2030-01-01T09:00:00Z",
            origin_chat_id=origin,
            created_by="user_gerard",
        )
    )
    assert out["success"] is True, out

    insert_sql, insert_params = db.execute.call_args_list[2][0]
    assert "origin_chat_id" in str(insert_sql)
    assert "created_by" in str(insert_sql)
    assert insert_params["origin_chat"] == origin
    assert insert_params["created_by_user"] == "user_gerard"


def test_create_task_drops_garbage_origin_and_null_creator():
    from services.scheduled_task_service import ScheduledTaskService

    db = MagicMock()
    agents_result = MagicMock()
    agents_result.fetchall.return_value = [SimpleNamespace(id=11, name="Auto")]
    count_result = MagicMock()
    count_result.scalar.return_value = 0
    insert_result = MagicMock()
    insert_result.fetchone.return_value = SimpleNamespace(
        id=43, created_at=datetime(2026, 7, 17)
    )
    db.execute.side_effect = [agents_result, count_result, insert_result]

    svc = ScheduledTaskService(db, uuid.uuid4())
    out = asyncio.run(
        svc.create_task(
            created_by_agent_id=11,
            target_agent_id=11,
            task_type="one_shot",
            description="digest",
            schedule="2030-01-01T09:00:00Z",
            origin_chat_id="not-a-uuid",
        )
    )
    assert out["success"] is True, out
    _, insert_params = db.execute.call_args_list[2][0]
    assert insert_params["origin_chat"] is None  # degraded, not failed
    assert insert_params["created_by_user"] is None


# ---------------------------------------------------------------------------
# DB fixtures (PRD-204 stage-1 pattern; skip cleanly without Postgres)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT origin_chat_id FROM watches LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"prd205 S4 suite needs a migrated Postgres: {exc}")
    yield eng
    eng.dispose()


# ``new_session`` comes from tests/conftest.py -- the shared tracking
# factory (leaked-session guard); teardown sweeps run via new_session.sweep().


@pytest.fixture
def workspace(new_session):
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd205-s4"},
    )
    s.commit()
    s.close()

    yield ws_id

    s = new_session.sweep()
    for stmt in (
        "DELETE FROM watch_events WHERE watch_id IN "
        "(SELECT id FROM watches WHERE workspace_id = CAST(:w AS uuid))",
        "DELETE FROM watches WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
    ):
        s.execute(text(stmt), {"w": ws_id})
    s.commit()
    s.close()


def test_watch_service_persists_origin_chat_id(new_session, workspace):
    from services.watch_service import WatchService

    origin = uuid.uuid4()
    s = new_session()
    try:
        watch = WatchService.create_watch(
            s,
            workspace_id=workspace,
            watch_type="mission",
            target_type="mission",
            target_id=str(uuid.uuid4()),
            title="origin persist",
            origin_chat_id=str(origin),
        )
        s.commit()
        stored = s.execute(
            text("SELECT origin_chat_id FROM watches WHERE id = CAST(:i AS uuid)"),
            {"i": str(watch.id)},
        ).scalar()
        assert str(stored) == str(origin)
    finally:
        s.close()


def test_watch_service_coerces_garbage_origin_to_null(new_session, workspace):
    from services.watch_service import WatchService

    s = new_session()
    try:
        watch = WatchService.create_watch(
            s,
            workspace_id=workspace,
            watch_type="mission",
            target_type="mission",
            target_id=str(uuid.uuid4()),
            title="garbage origin",
            origin_chat_id="chat-42-not-a-uuid",
        )
        s.commit()
        stored = s.execute(
            text("SELECT origin_chat_id FROM watches WHERE id = CAST(:i AS uuid)"),
            {"i": str(watch.id)},
        ).scalar()
        assert stored is None
    finally:
        s.close()


def test_create_watch_handler_passes_injected_origin(new_session, workspace, monkeypatch):
    import modules.tools.discovery.handlers_watches as hw

    monkeypatch.setattr(
        hw, "_resolve_target", lambda db, ws, tt, ti: {"title": "t", "criteria": "c"}
    )
    origin = str(uuid.uuid4())
    s = new_session()
    try:
        out = asyncio.run(
            hw.create_watch(
                s,
                uuid.UUID(workspace),
                {
                    "target_type": "mission",
                    "target_id": str(uuid.uuid4()),
                    "_origin_chat_id": origin,  # executor-injected by S4
                },
            )
        )
        assert out["success"] is True, out
        s.commit()
        stored = s.execute(
            text("SELECT origin_chat_id FROM watches WHERE id = CAST(:i AS uuid)"),
            {"i": out["watch"]["id"]},
        ).scalar()
        assert str(stored) == origin
    finally:
        s.close()


def test_auto_create_watch_passes_origin(new_session, workspace):
    from modules.tools.discovery.handlers_watches import auto_create_watch

    origin = str(uuid.uuid4())
    s = new_session()
    try:
        watch = auto_create_watch(
            s,
            uuid.UUID(workspace),
            target_type="mission",
            target_id=str(uuid.uuid4()),
            title="launch watch",
            success_criteria="it completes",
            created_by="user_gerard",
            origin_chat_id=origin,
        )
        assert watch is not None
        s.commit()
        stored = s.execute(
            text("SELECT origin_chat_id FROM watches WHERE id = CAST(:i AS uuid)"),
            {"i": str(watch.id)},
        ).scalar()
        assert str(stored) == origin
    finally:
        s.close()
