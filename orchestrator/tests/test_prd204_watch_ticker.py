"""PRD-204 S5 -- watcher tick: sweep fallback, missed-cron detection with a
frozen clock, benched recording, expiry, and the no-noise rule.

DB-backed (real Postgres; skips cleanly when absent). All clocks are
injected (``tick_once(db, now=...)``) -- no wall-clock dependence.
Notifications are captured at the ticker's dispatch seam (the
``notifications`` table itself is migration-only and not present in the
create_all test schema; the row shape is covered by the S4 mock suite).
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.database.database import get_database_url
from core.models.core import RecipeExecution, WorkflowTemplate
from core.models.orchestration import OrchestrationRun
from core.models.orchestration_enums import RunState
from core.models.watches import WatchEvent
from core.models.watch_enums import WatchEventType, WatchStatus
from services.watch_service import WatchService
from services.watch_ticker import WatchTicker


FROZEN_NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM watches LIMIT 1"))
            c.execute(text("SELECT 1 FROM recipe_executions LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"watch ticker suite needs a reachable Postgres with schema: {exc}")
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
        {"id": ws_id, "n": "prd204-watch-ticker"},
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
            "DELETE FROM recipe_executions WHERE workspace_id = CAST(:w AS uuid)"
        ),
        {"w": ws_id},
    )
    s.execute(
        text(
            "DELETE FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)"
        ),
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


@pytest.fixture
def ticker(monkeypatch):
    """A WatchTicker whose notification seam records instead of dispatching."""
    t = WatchTicker()
    t.dispatched = []

    async def _record(self, db, watch, *, event_type, title, message, status="ok"):
        self.dispatched.append(
            {"watch_id": str(watch.id), "event_type": event_type, "status": status}
        )

    monkeypatch.setattr(WatchTicker, "_dispatch_watch_event", _record)
    return t


def _tick(ticker, db, now):
    return asyncio.run(ticker.tick_once(db, now=now))


def _make_due(s, watch, now):
    s.execute(
        text("UPDATE watches SET next_check_at = :d WHERE id = CAST(:i AS uuid)"),
        {"d": now - timedelta(seconds=1), "i": str(watch.id)},
    )
    s.commit()


def _seed_run(s, ws_id: str, state: str) -> OrchestrationRun:
    run = OrchestrationRun(
        workspace_id=ws_id,
        goal="ticker test mission",
        state=state,
        created_by="user_test",
    )
    s.add(run)
    s.commit()
    return run


def _seed_recipe(s, ws_id: str, cron: str = "0 9 * * *") -> WorkflowTemplate:
    recipe = WorkflowTemplate(
        template_id=f"prd204-{uuid.uuid4().hex[:10]}",
        name="ticker test playbook",
        description="prd204 watch ticker test",
        workspace_id=ws_id,
        template_definition={"steps": []},
        steps=[{"step_id": "s1", "order": 1}],
        schedule_config={"type": "cron", "cron_expression": cron},
        created_by="user_test",  # NOT NULL on workflow_recipes
    )
    s.add(recipe)
    s.commit()
    return recipe


def _watch_events(s, watch, event_type=None):
    q = s.query(WatchEvent).filter(WatchEvent.watch_id == watch.id)
    if event_type:
        q = q.filter(WatchEvent.event_type == event_type)
    return q.all()


# ---------------------------------------------------------------------------
# Sweep fallback: a terminal state the hooks missed
# ---------------------------------------------------------------------------


def test_sweep_ingests_missed_terminal_and_notifies_once(
    workspace, new_session, ticker
):
    s = new_session()
    # Run already terminal in the DB -- simulates a hook that never fired
    # (crash between the terminal write and the hook's transaction).
    run = _seed_run(s, workspace, state=RunState.FAILED.value)
    watch = WatchService.create_watch(
        s,
        workspace_id=workspace,
        watch_type="mission",
        target_type="mission",
        target_id=str(run.id),
        title="Watch: sweep fallback",
        now=FROZEN_NOW - timedelta(minutes=30),
    )
    s.commit()
    _make_due(s, watch, FROZEN_NOW)

    processed = _tick(ticker, s, FROZEN_NOW)
    assert processed >= 1

    s.refresh(watch)
    assert watch.status == WatchStatus.FAILED.value
    assert len(_watch_events(s, watch, "terminal")) == 1

    verdicts = [d for d in ticker.dispatched if d["watch_id"] == str(watch.id)]
    assert [d["event_type"] for d in verdicts] == ["watch_verdict"]
    assert verdicts[0]["status"] == "error"

    # Tick idempotence: the closed watch is never claimed again -- two ticks,
    # one event, one notification.
    _tick(ticker, s, FROZEN_NOW + timedelta(seconds=600))
    assert len(_watch_events(s, watch, "terminal")) == 1
    assert len([d for d in ticker.dispatched if d["watch_id"] == str(watch.id)]) == 1
    s.close()


# ---------------------------------------------------------------------------
# No-noise: running -> running writes nothing
# ---------------------------------------------------------------------------


def test_running_target_writes_nothing(workspace, new_session, ticker):
    s = new_session()
    run = _seed_run(s, workspace, state=RunState.RUNNING.value)
    watch = WatchService.create_watch(
        s,
        workspace_id=workspace,
        watch_type="mission",
        target_type="mission",
        target_id=str(run.id),
        title="Watch: no-noise",
        now=FROZEN_NOW - timedelta(minutes=30),
    )
    s.commit()
    _make_due(s, watch, FROZEN_NOW)

    _tick(ticker, s, FROZEN_NOW)

    s.refresh(watch)
    assert watch.status == WatchStatus.WATCHING.value
    # Only the creation event exists -- the sweep added no noise.
    assert [e.event_type for e in _watch_events(s, watch)] == [
        WatchEventType.CREATED.value
    ]
    assert [d for d in ticker.dispatched if d["watch_id"] == str(watch.id)] == []
    # And the claim rescheduled the next check.
    assert watch.next_check_at == FROZEN_NOW + timedelta(
        seconds=watch.check_interval_seconds
    )
    s.close()


# ---------------------------------------------------------------------------
# Missed-cron detection (frozen clock) + idempotence
# ---------------------------------------------------------------------------


def test_missed_cron_detected_with_frozen_clock(workspace, new_session, ticker):
    s = new_session()
    recipe = _seed_recipe(s, workspace, cron="0 9 * * *")
    watch = WatchService.create_watch(
        s,
        workspace_id=workspace,
        watch_type="scheduled_playbook",
        target_type="scheduled_playbook",
        target_id=str(recipe.id),
        title="Watch: nightly playbook",
        policy="persistent",
        now=FROZEN_NOW - timedelta(hours=12),
    )
    s.commit()
    # Baseline = watch.created_at (no executions exist): 00:00 -> expected
    # fire 09:00; now is 12:00 -> 3h overdue, no execution row -> MISSED.
    s.execute(
        text("UPDATE watches SET created_at = :c WHERE id = CAST(:i AS uuid)"),
        {"c": FROZEN_NOW - timedelta(hours=12), "i": str(watch.id)},
    )
    s.commit()
    _make_due(s, watch, FROZEN_NOW)

    _tick(ticker, s, FROZEN_NOW)

    missed = _watch_events(s, watch, WatchEventType.MISSED_RUN.value)
    assert len(missed) == 1
    assert missed[0].requires_attention is True
    assert "09:00" in missed[0].snapshot["expected_fire"]

    escalations = [
        d
        for d in ticker.dispatched
        if d["watch_id"] == str(watch.id) and d["event_type"] == "watch_escalation"
    ]
    assert len(escalations) == 1

    # Second tick at the same frozen instant: the event_key (expected fire
    # time) dedupes -- still exactly one event, one notification.
    _make_due(s, watch, FROZEN_NOW)
    _tick(ticker, s, FROZEN_NOW)
    assert len(_watch_events(s, watch, WatchEventType.MISSED_RUN.value)) == 1
    assert (
        len(
            [
                d
                for d in ticker.dispatched
                if d["watch_id"] == str(watch.id)
                and d["event_type"] == "watch_escalation"
            ]
        )
        == 1
    )
    # The watch stays live -- a missed run is attention-worthy, not terminal.
    s.refresh(watch)
    assert watch.status == WatchStatus.WATCHING.value
    s.close()


def test_recent_run_within_grace_is_not_missed(workspace, new_session, ticker):
    s = new_session()
    recipe = _seed_recipe(s, workspace, cron="0 9 * * *")
    # An execution actually started at 09:00 today.
    s.add(
        RecipeExecution(
            execution_id=f"prd204-{uuid.uuid4().hex[:10]}",
            recipe_id=recipe.id,
            workspace_id=workspace,
            status="completed",
            input_data={},
            started_at=FROZEN_NOW.replace(hour=9, minute=0).replace(tzinfo=None),
            triggered_by="cron_scheduler",
        )
    )
    s.commit()
    watch = WatchService.create_watch(
        s,
        workspace_id=workspace,
        watch_type="scheduled_playbook",
        target_type="scheduled_playbook",
        target_id=str(recipe.id),
        title="Watch: healthy playbook",
        now=FROZEN_NOW - timedelta(hours=12),
    )
    s.commit()
    _make_due(s, watch, FROZEN_NOW)

    _tick(ticker, s, FROZEN_NOW)

    # Baseline is the 09:00 run -> next expected fire is tomorrow -> nothing.
    assert _watch_events(s, watch, WatchEventType.MISSED_RUN.value) == []
    assert [d for d in ticker.dispatched if d["watch_id"] == str(watch.id)] == []
    s.close()


# ---------------------------------------------------------------------------
# Benched: recorded on the watch, NOT notified by the tick (S4 scheduler owns
# the playbook_benched notification)
# ---------------------------------------------------------------------------


def test_open_breaker_records_benched_event_once(workspace, new_session, ticker):
    s = new_session()
    recipe = _seed_recipe(s, workspace, cron="0 9 * * *")
    # Three consecutive failures trip the breaker (threshold default 3).
    for i in range(3):
        s.add(
            RecipeExecution(
                execution_id=f"prd204-fail-{i}-{uuid.uuid4().hex[:8]}",
                recipe_id=recipe.id,
                workspace_id=workspace,
                status="failed",
                input_data={},
                started_at=(
                    FROZEN_NOW.replace(hour=8, minute=50) + timedelta(minutes=i)
                ).replace(tzinfo=None),
                triggered_by="cron_scheduler",
            )
        )
    s.commit()
    watch = WatchService.create_watch(
        s,
        workspace_id=workspace,
        watch_type="scheduled_playbook",
        target_type="scheduled_playbook",
        target_id=str(recipe.id),
        title="Watch: benched playbook",
        now=FROZEN_NOW - timedelta(minutes=30),
    )
    s.commit()
    # 09:01 -> the 09:00 expected fire is only 60s overdue (inside the
    # missed-run grace window), so ONLY the benched path is exercised.
    now = FROZEN_NOW.replace(hour=9, minute=1)
    _make_due(s, watch, now)

    _tick(ticker, s, now)

    benched = _watch_events(s, watch, WatchEventType.BENCHED.value)
    assert len(benched) == 1
    assert benched[0].requires_attention is True
    # The tick records; the SCHEDULER notifies (once per open period). No
    # tick-side dispatch for benched.
    assert [d for d in ticker.dispatched if d["watch_id"] == str(watch.id)] == []

    # Same open period -> same latest-terminal key -> still one event.
    _make_due(s, watch, now)
    _tick(ticker, s, now)
    assert len(_watch_events(s, watch, WatchEventType.BENCHED.value)) == 1
    s.close()


# ---------------------------------------------------------------------------
# Deadline expiry
# ---------------------------------------------------------------------------


def test_deadline_expires_watch_and_escalates(workspace, new_session, ticker):
    s = new_session()
    run = _seed_run(s, workspace, state=RunState.RUNNING.value)
    watch = WatchService.create_watch(
        s,
        workspace_id=workspace,
        watch_type="mission",
        target_type="mission",
        target_id=str(run.id),
        title="Watch: deadline",
        deadline_at=FROZEN_NOW - timedelta(minutes=5),
        now=FROZEN_NOW - timedelta(hours=2),
    )
    s.commit()
    _make_due(s, watch, FROZEN_NOW)

    _tick(ticker, s, FROZEN_NOW)

    s.refresh(watch)
    assert watch.status == WatchStatus.EXPIRED.value
    assert watch.closed_at is not None
    assert len(_watch_events(s, watch, WatchEventType.EXPIRED.value)) == 1

    escalations = [
        d
        for d in ticker.dispatched
        if d["watch_id"] == str(watch.id) and d["event_type"] == "watch_escalation"
    ]
    assert len(escalations) == 1
    s.close()


# ---------------------------------------------------------------------------
# Missing target parks the watch for a human
# ---------------------------------------------------------------------------


def test_missing_target_parks_watch(workspace, new_session, ticker):
    s = new_session()
    watch = WatchService.create_watch(
        s,
        workspace_id=workspace,
        watch_type="mission",
        target_type="mission",
        target_id=str(uuid.uuid4()),  # no such run
        title="Watch: lost target",
        now=FROZEN_NOW - timedelta(minutes=30),
    )
    s.commit()
    _make_due(s, watch, FROZEN_NOW)

    _tick(ticker, s, FROZEN_NOW)

    s.refresh(watch)
    assert watch.status == WatchStatus.NEEDS_ATTENTION.value
    events = _watch_events(s, watch, WatchEventType.TARGET_MISSING.value)
    assert len(events) == 1
    s.close()
