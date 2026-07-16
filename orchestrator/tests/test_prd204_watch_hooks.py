"""PRD-204 S3 -- fail-soft terminal hooks.

Proves the two contract halves:
1. every producer terminal path reports into the watch registry, and
2. a RAISING watch service never breaks the producer (mission transition,
   playbook executor, board-task dispatch all complete normally).

Mission-path tests are DB-backed (real Postgres, skip cleanly when absent);
executor/board hook tests run against mocks -- the hook seam is what is under
test, not those modules' heavy internals.
"""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.database.database import get_database_url
from core.models.orchestration import OrchestrationRun
from core.models.orchestration_enums import ActorType, RunState
from core.models.watches import WatchEvent
from core.models.watch_enums import WatchStatus
from services.orchestration_state import transition_run
from services.watch_hooks import watch_ingest_terminal
from services.watch_service import WatchService


# ---------------------------------------------------------------------------
# Unit: the hook itself is fail-soft
# ---------------------------------------------------------------------------


def test_hook_swallows_raising_watch_service(monkeypatch):
    """watch_ingest_terminal must never raise, whatever the service does."""

    def _boom(*args, **kwargs):
        raise RuntimeError("watch registry down")

    monkeypatch.setattr(
        "services.watch_service.WatchService.ingest_terminal", _boom
    )
    # Must not raise.
    watch_ingest_terminal(
        MagicMock(),
        workspace_id=str(uuid.uuid4()),
        target_type="mission",
        target_id=str(uuid.uuid4()),
        terminal_state="completed",
    )


def test_hook_noop_without_live_watch_uses_service(monkeypatch):
    """The hook passes through to WatchService.ingest_terminal once."""
    calls = []

    def _record(db, **kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(
        "services.watch_service.WatchService.ingest_terminal",
        staticmethod(_record),
    )
    watch_ingest_terminal(
        MagicMock(),
        workspace_id="ws-1",
        target_type="playbook_execution",
        target_id="exec-1",
        terminal_state="failed",
        summary="boom",
        cost_snapshot={"total_tokens": 5},
    )
    assert len(calls) == 1
    assert calls[0]["target_type"] == "playbook_execution"
    assert calls[0]["target_id"] == "exec-1"
    assert calls[0]["terminal_state"] == "failed"
    assert calls[0]["cost_snapshot"] == {"total_tokens": 5}


# ---------------------------------------------------------------------------
# Mission choke point (DB-backed): transition_run covers every fail/cancel/
# complete call site because they all flow through it.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM watches LIMIT 1"))
            c.execute(text("SELECT 1 FROM orchestration_runs LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"watch hooks suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def workspace(new_session):
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd204-watch-hooks"},
    )
    s.commit()
    s.close()

    yield ws_id

    s = new_session()
    s.execute(
        text(
            "DELETE FROM watch_events WHERE watch_id IN "
            "(SELECT id FROM watches WHERE workspace_id = CAST(:w AS uuid))"
        ),
        {"w": ws_id},
    )
    s.execute(
        text("DELETE FROM watches WHERE workspace_id = CAST(:w AS uuid)"),
        {"w": ws_id},
    )
    s.execute(
        text(
            "DELETE FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid)"
        ),
        {"w": ws_id},
    )
    s.execute(
        text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws_id}
    )
    s.commit()
    s.close()


def _seed_run_and_watch(s, ws_id: str):
    run = OrchestrationRun(
        workspace_id=ws_id,
        goal="hooks test mission",
        state=RunState.PLANNING.value,
        created_by="user_test",
    )
    s.add(run)
    s.flush()
    watch = WatchService.create_watch(
        s,
        workspace_id=ws_id,
        watch_type="mission",
        target_type="mission",
        target_id=str(run.id),
        title="Watch: hooks test mission",
    )
    s.commit()
    return run, watch


def test_mission_failed_transition_ingests_watch(workspace, new_session):
    """S10 semantics: the producer's terminal is RECORDED at the choke point
    and the watch is handed to the decision step (next check pulled to now)
    -- the close now belongs to the decider, not the sync hook."""
    s = new_session()
    run, watch = _seed_run_and_watch(s, workspace)

    transition_run(
        db=s,
        run=run,
        new_state=RunState.FAILED,
        actor_type=ActorType.COORDINATOR,
        actor_id="coordinator",
        reason="Plan validation failed",
        stop_reason="coordinator_error",
        stop_detail="Plan validation failed after all retries",
    )
    s.commit()

    s.refresh(watch)
    assert watch.status == WatchStatus.WATCHING.value  # deferred to the decider
    assert watch.closed_at is None
    # Pulled forward for the tick: well inside the original +300s cadence.
    assert watch.next_check_at is not None
    assert (watch.next_check_at - watch.created_at).total_seconds() < 60
    events = (
        s.query(WatchEvent)
        .filter(WatchEvent.watch_id == watch.id, WatchEvent.event_type == "terminal")
        .all()
    )
    assert len(events) == 1
    assert events[0].snapshot["terminal_state"] == "failed"
    assert "Plan validation failed" in (events[0].summary or "")
    s.close()


def test_mission_cancelled_transition_ingests_watch(workspace, new_session):
    s = new_session()
    run, watch = _seed_run_and_watch(s, workspace)

    transition_run(
        db=s,
        run=run,
        new_state=RunState.CANCELLED,
        actor_type=ActorType.HUMAN,
        actor_id="user_test",
        stop_reason="human_cancelled",
    )
    s.commit()

    s.refresh(watch)
    assert watch.status == WatchStatus.CANCELLED.value
    s.close()


def test_non_terminal_transition_does_not_touch_watch(workspace, new_session):
    s = new_session()
    run, watch = _seed_run_and_watch(s, workspace)

    transition_run(
        db=s,
        run=run,
        new_state=RunState.RUNNING,
        actor_type=ActorType.COORDINATOR,
        actor_id="coordinator",
    )
    s.commit()

    s.refresh(watch)
    assert watch.status == WatchStatus.WATCHING.value
    terminal_events = (
        s.query(WatchEvent)
        .filter(WatchEvent.watch_id == watch.id, WatchEvent.event_type == "terminal")
        .count()
    )
    assert terminal_events == 0
    s.close()


def test_producer_completes_when_watch_service_raises(
    workspace, new_session, monkeypatch
):
    """THE fail-soft assertion: a raising watch service must not fail the
    mission transition (the producer)."""

    def _boom(*args, **kwargs):
        raise RuntimeError("watch registry exploded")

    monkeypatch.setattr(
        "services.watch_service.WatchService.ingest_terminal", _boom
    )

    s = new_session()
    run, watch = _seed_run_and_watch(s, workspace)

    transition_run(
        db=s,
        run=run,
        new_state=RunState.FAILED,
        actor_type=ActorType.COORDINATOR,
        actor_id="coordinator",
        stop_reason="coordinator_error",
    )
    s.commit()

    s.refresh(run)
    assert run.state == RunState.FAILED.value  # producer landed its terminal
    s.refresh(watch)
    assert watch.status == WatchStatus.WATCHING.value  # ingest was lost, softly
    s.close()


# ---------------------------------------------------------------------------
# Playbook executor seam (mocked executor internals; real hook seam)
# ---------------------------------------------------------------------------


def _fake_execution(**overrides):
    execution = MagicMock()
    execution.execution_id = overrides.get("execution_id", "exec-prd204")
    execution.workspace_id = overrides.get("workspace_id", str(uuid.uuid4()))
    execution.output_data = overrides.get(
        "output_data",
        {"final_output": "report text", "total_tokens": 42, "total_duration_ms": 1500},
    )
    return execution


def test_playbook_terminal_helper_reports_completed(monkeypatch):
    from api.recipe_executor import _ingest_playbook_terminal_watch

    calls = []
    monkeypatch.setattr(
        "services.watch_hooks.watch_ingest_terminal",
        lambda db, **kw: calls.append(kw),
    )

    execution = _fake_execution()
    _ingest_playbook_terminal_watch(
        MagicMock(), execution, terminal_state="completed", summary="3 steps"
    )

    assert len(calls) == 1
    kw = calls[0]
    assert kw["target_type"] == "playbook_execution"
    assert kw["target_id"] == "exec-prd204"
    assert kw["terminal_state"] == "completed"
    assert kw["cost_snapshot"] == {"total_tokens": 42, "total_duration_ms": 1500}
    assert kw["output_pointer"] == (
        "recipe_execution:exec-prd204:output_data.final_output"
    )


def test_playbook_terminal_helper_reports_failed_without_output(monkeypatch):
    from api.recipe_executor import _ingest_playbook_terminal_watch

    calls = []
    monkeypatch.setattr(
        "services.watch_hooks.watch_ingest_terminal",
        lambda db, **kw: calls.append(kw),
    )

    execution = _fake_execution(output_data=None)
    _ingest_playbook_terminal_watch(
        MagicMock(), execution, terminal_state="failed", summary="boom"
    )

    kw = calls[0]
    assert kw["terminal_state"] == "failed"
    assert kw["output_pointer"] is None
    assert kw["summary"] == "boom"


def test_playbook_terminal_helper_is_fail_soft(monkeypatch):
    """A raising watch service does not raise out of the executor helper."""
    from api.recipe_executor import _ingest_playbook_terminal_watch

    def _boom(*args, **kwargs):
        raise RuntimeError("registry down")

    monkeypatch.setattr(
        "services.watch_service.WatchService.ingest_terminal", _boom
    )
    # Must not raise (watch_ingest_terminal is the fail-soft wrapper).
    _ingest_playbook_terminal_watch(
        MagicMock(), _fake_execution(), terminal_state="completed"
    )


# ---------------------------------------------------------------------------
# Board-task seam
# ---------------------------------------------------------------------------


def _mock_board_db():
    db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = None
    db.query.return_value = q
    exec_result = MagicMock()
    exec_result.fetchall.return_value = []
    db.execute.return_value = exec_result
    return db


def _fake_board_task():
    task = MagicMock()
    task.id = 7
    task.title = "hooks board task"
    task.assigned_agent_id = None
    task.result = "done ok"
    task.description = None
    task.error_message = "it broke"
    return task


def test_board_task_complete_reports_watch(monkeypatch):
    from api.board_tasks import _dispatch_task_complete

    calls = []
    monkeypatch.setattr(
        "services.watch_hooks.watch_ingest_terminal",
        lambda db, **kw: calls.append(kw),
    )

    asyncio.run(
        _dispatch_task_complete(_mock_board_db(), str(uuid.uuid4()), _fake_board_task())
    )

    assert len(calls) == 1
    assert calls[0]["target_type"] == "board_task"
    assert calls[0]["target_id"] == "7"
    assert calls[0]["terminal_state"] == "completed"


def test_board_task_failed_reports_watch(monkeypatch):
    from api.board_tasks import _dispatch_task_failed

    calls = []
    monkeypatch.setattr(
        "services.watch_hooks.watch_ingest_terminal",
        lambda db, **kw: calls.append(kw),
    )

    asyncio.run(
        _dispatch_task_failed(_mock_board_db(), str(uuid.uuid4()), _fake_board_task())
    )

    assert len(calls) == 1
    assert calls[0]["target_type"] == "board_task"
    assert calls[0]["terminal_state"] == "failed"
    assert calls[0]["summary"] == "it broke"


def test_board_task_dispatch_survives_raising_watch_service(monkeypatch):
    """Producer helper completes even when the watch service raises."""
    from api.board_tasks import _dispatch_task_complete

    def _boom(*args, **kwargs):
        raise RuntimeError("registry down")

    monkeypatch.setattr(
        "services.watch_service.WatchService.ingest_terminal", _boom
    )
    # Must not raise.
    asyncio.run(
        _dispatch_task_complete(_mock_board_db(), str(uuid.uuid4()), _fake_board_task())
    )
