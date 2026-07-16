"""PRD-204 S2 -- WatchService: transition guard, idempotent ingest, budget
hard-stop, lineage repoint, SKIP LOCKED claim.

DB-backed (real Postgres) -- savepoint-swallowed unique violations, the
partial unique index and FOR UPDATE SKIP LOCKED are Postgres semantics.
Skips cleanly when no DB is reachable (test_board_dispatch.py precedent).

PRD-158 lesson: workspaces seeded FIRST for every FK-touching test.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.database.database import get_database_url
from core.models.watches import Watch, WatchEvent
from core.models.watch_enums import WatchEventType, WatchStatus
from services.orchestration_state import InvalidTransitionError
from services.watch_service import WatchAlreadyExistsError, WatchService


FROZEN_NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM watches LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"watch service suite needs a reachable Postgres with schema: {exc}")
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
        {"id": ws_id, "n": "prd204-watch-service"},
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
        text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws_id}
    )
    s.commit()
    s.close()


def _create(db, ws_id: str, **overrides) -> Watch:
    params = dict(
        workspace_id=ws_id,
        watch_type="mission",
        target_type="mission",
        target_id=str(uuid.uuid4()),
        title="Watch: service test",
        success_criteria="produce the report",
        now=FROZEN_NOW,
    )
    params.update(overrides)
    return WatchService.create_watch(db, **params)


# ---------------------------------------------------------------------------
# create + duplicate guard
# ---------------------------------------------------------------------------


def test_create_watch_writes_created_event_and_lineage(workspace, new_session):
    s = new_session()
    watch = _create(s, workspace)
    s.commit()

    assert watch.status == WatchStatus.WATCHING.value
    assert watch.next_check_at == FROZEN_NOW + timedelta(seconds=300)
    assert len(watch.lineage) == 1
    assert watch.lineage[-1]["target_id"] == watch.target_id

    events = s.query(WatchEvent).filter(WatchEvent.watch_id == watch.id).all()
    assert [e.event_type for e in events] == [WatchEventType.CREATED.value]
    s.close()


def test_create_duplicate_live_watch_rejected(workspace, new_session):
    s = new_session()
    watch = _create(s, workspace)
    s.commit()

    with pytest.raises(WatchAlreadyExistsError):
        _create(s, workspace, target_id=watch.target_id)
    s.rollback()
    s.close()


# ---------------------------------------------------------------------------
# transition guard
# ---------------------------------------------------------------------------


def test_transition_guard_allows_and_blocks(workspace, new_session):
    s = new_session()
    watch = _create(s, workspace)

    WatchService.transition(s, watch, WatchStatus.ACTING)
    assert watch.status == WatchStatus.ACTING.value

    WatchService.transition(s, watch, WatchStatus.PASSED)
    assert watch.status == WatchStatus.PASSED.value
    assert watch.closed_at is not None

    # Terminal is terminal (except the escalated renewal).
    with pytest.raises(InvalidTransitionError):
        WatchService.transition(s, watch, WatchStatus.WATCHING)
    s.rollback()
    s.close()


def test_escalated_can_renew_to_watching(workspace, new_session):
    s = new_session()
    watch = _create(s, workspace)
    WatchService.transition(s, watch, WatchStatus.ESCALATED)
    assert watch.closed_at is not None

    WatchService.transition(s, watch, WatchStatus.WATCHING)
    assert watch.status == WatchStatus.WATCHING.value
    assert watch.closed_at is None
    s.commit()
    s.close()


# ---------------------------------------------------------------------------
# idempotent ingest
# ---------------------------------------------------------------------------


def test_double_ingest_swallowed_and_transaction_survives(workspace, new_session):
    s = new_session()
    watch = _create(s, workspace)

    first = WatchService.ingest(
        s, watch, event_type="terminal", event_key="k-dup", summary="one"
    )
    second = WatchService.ingest(
        s, watch, event_type="terminal", event_key="k-dup", summary="two"
    )
    assert first is not None
    assert second is None

    # The savepoint swallow must leave the outer transaction usable.
    third = WatchService.ingest(
        s, watch, event_type="status_change", event_key="k-other"
    )
    assert third is not None
    s.commit()

    keys = [
        e.event_key
        for e in s.query(WatchEvent).filter(WatchEvent.watch_id == watch.id).all()
    ]
    assert sorted(keys) == ["created", "k-dup", "k-other"]
    s.close()


# ---------------------------------------------------------------------------
# ingest_terminal
# ---------------------------------------------------------------------------


def test_ingest_terminal_closes_watch_by_outcome(workspace, new_session):
    s = new_session()
    watch = _create(s, workspace)
    s.commit()

    event = WatchService.ingest_terminal(
        s,
        workspace_id=workspace,
        target_type="mission",
        target_id=watch.target_id,
        terminal_state="completed",
        summary="All tasks verified",
        cost_snapshot={"tokens_used": 900},
        output_pointer=f"mission:{watch.target_id}:output_summary",
    )
    s.commit()

    assert event is not None
    assert event.snapshot["cost"]["tokens_used"] == 900
    s.refresh(watch)
    assert watch.status == WatchStatus.PASSED.value
    assert watch.closed_at is not None
    assert "completed" in (watch.final_verdict or "")

    # Second delivery of the same terminal is a no-op.
    again = WatchService.ingest_terminal(
        s,
        workspace_id=workspace,
        target_type="mission",
        target_id=watch.target_id,
        terminal_state="completed",
    )
    assert again is None
    s.close()


@pytest.mark.parametrize(
    "terminal_state,expected_status",
    [
        ("failed", WatchStatus.FAILED.value),
        ("cancelled", WatchStatus.CANCELLED.value),
    ],
)
def test_ingest_terminal_outcome_mapping(
    workspace, new_session, terminal_state, expected_status
):
    s = new_session()
    watch = _create(s, workspace)
    s.commit()

    WatchService.ingest_terminal(
        s,
        workspace_id=workspace,
        target_type="mission",
        target_id=watch.target_id,
        terminal_state=terminal_state,
    )
    s.commit()
    s.refresh(watch)
    assert watch.status == expected_status
    s.close()


def test_ingest_terminal_recovers_lost_close(workspace, new_session):
    """Self-healing idempotency: if a prior attempt committed the terminal
    EVENT but lost the CLOSE (version conflict), the next delivery still
    closes the watch instead of being blocked by the duplicate event."""
    from core.models.watches import WatchEvent

    s = new_session()
    watch = _create(s, workspace)
    # Simulate the half-landed prior attempt: event row exists, watch open.
    s.add(
        WatchEvent(
            watch_id=watch.id,
            event_type=WatchEventType.TERMINAL.value,
            event_key=f"terminal:mission:{watch.target_id}",
            summary="first attempt",
        )
    )
    s.commit()

    result = WatchService.ingest_terminal(
        s,
        workspace_id=workspace,
        target_type="mission",
        target_id=watch.target_id,
        terminal_state="completed",
    )
    s.commit()

    assert result is None  # not the first writer of the event...
    s.refresh(watch)
    assert watch.status == WatchStatus.PASSED.value  # ...but the close lands
    assert watch.closed_at is not None
    s.close()


def test_ingest_terminal_without_live_watch_is_noop(workspace, new_session):
    s = new_session()
    result = WatchService.ingest_terminal(
        s,
        workspace_id=workspace,
        target_type="mission",
        target_id=str(uuid.uuid4()),
        terminal_state="completed",
    )
    assert result is None
    s.close()


# ---------------------------------------------------------------------------
# action budget hard-stop
# ---------------------------------------------------------------------------


def test_record_action_hard_stops_at_budget(workspace, new_session):
    s = new_session()
    watch = _create(s, workspace, action_budget=2)

    _, ok1 = WatchService.record_action(s, watch, action="rerun")
    _, ok2 = WatchService.record_action(s, watch, action="tweak")
    assert (ok1, ok2) == (True, True)
    assert watch.actions_taken == 2
    assert watch.status == WatchStatus.ACTING.value

    _, ok3 = WatchService.record_action(s, watch, action="rerun")
    assert ok3 is False
    assert watch.actions_taken == 2  # refused action not counted
    assert watch.status == WatchStatus.NEEDS_ATTENTION.value
    s.commit()

    exhausted = (
        s.query(WatchEvent)
        .filter(
            WatchEvent.watch_id == watch.id,
            WatchEvent.event_type == WatchEventType.BUDGET_EXHAUSTED.value,
        )
        .all()
    )
    assert len(exhausted) == 1
    assert exhausted[0].requires_attention is True
    s.close()


def test_zero_budget_refuses_first_action(workspace, new_session):
    s = new_session()
    watch = _create(s, workspace, action_budget=0)
    _, ok = WatchService.record_action(s, watch, action="rerun")
    assert ok is False
    assert watch.status == WatchStatus.NEEDS_ATTENTION.value
    s.commit()
    s.close()


# ---------------------------------------------------------------------------
# lineage repoint
# ---------------------------------------------------------------------------


def test_follow_repoints_target_and_appends_lineage(workspace, new_session):
    s = new_session()
    watch = _create(s, workspace)
    original_target = watch.target_id
    new_target = str(uuid.uuid4())

    WatchService.follow(
        s,
        watch,
        new_target_type="mission",
        new_target_id=new_target,
        reason="rerun after failure",
        now=FROZEN_NOW + timedelta(minutes=10),
    )
    s.commit()

    assert watch.target_id == new_target
    assert len(watch.lineage) == 2
    assert watch.lineage[0]["target_id"] == original_target
    assert watch.lineage[-1]["target_id"] == new_target
    assert watch.next_check_at == FROZEN_NOW + timedelta(minutes=10)

    # Terminal ingest for the NEW target closes the same watch.
    WatchService.ingest_terminal(
        s,
        workspace_id=workspace,
        target_type="mission",
        target_id=new_target,
        terminal_state="completed",
    )
    s.commit()
    s.refresh(watch)
    assert watch.status == WatchStatus.PASSED.value
    s.close()


def test_follow_on_closed_watch_raises(workspace, new_session):
    s = new_session()
    watch = _create(s, workspace)
    WatchService.transition(s, watch, WatchStatus.CANCELLED)
    with pytest.raises(InvalidTransitionError):
        WatchService.follow(
            s, watch, new_target_type="mission", new_target_id=str(uuid.uuid4())
        )
    s.rollback()
    s.close()


# ---------------------------------------------------------------------------
# SKIP LOCKED claim
# ---------------------------------------------------------------------------


def test_claim_due_watches_claims_and_reschedules(workspace, new_session):
    s = new_session()
    due_a = _create(s, workspace)
    due_b = _create(s, workspace)
    future = _create(s, workspace)
    parked = _create(s, workspace)
    WatchService.transition(s, parked, WatchStatus.NEEDS_ATTENTION)
    s.commit()

    now = FROZEN_NOW + timedelta(seconds=600)
    s.execute(
        text(
            "UPDATE watches SET next_check_at = :due "
            "WHERE id IN (CAST(:a AS uuid), CAST(:b AS uuid), CAST(:p AS uuid))"
        ),
        {
            "due": now - timedelta(seconds=1),
            "a": str(due_a.id),
            "b": str(due_b.id),
            "p": str(parked.id),
        },
    )
    s.execute(
        text("UPDATE watches SET next_check_at = :f WHERE id = CAST(:i AS uuid)"),
        {"f": now + timedelta(hours=1), "i": str(future.id)},
    )
    s.commit()
    s.close()

    s = new_session()
    claimed = WatchService.claim_due_watches(s, now=now)
    claimed_ids = {str(w.id) for w in claimed}
    # Both due watches claimed; the future one and the parked
    # (needs_attention) one are not.
    assert {str(due_a.id), str(due_b.id)} <= claimed_ids
    assert str(future.id) not in claimed_ids
    assert str(parked.id) not in claimed_ids

    for w in claimed:
        if str(w.id) in {str(due_a.id), str(due_b.id)}:
            assert w.last_checked_at == now
            assert w.next_check_at == now + timedelta(
                seconds=w.check_interval_seconds
            )

    # Idempotence: an immediate second claim at the same instant finds nothing
    # (the first claim already rescheduled next_check_at).
    reclaimed = WatchService.claim_due_watches(s, now=now)
    assert {str(w.id) for w in reclaimed} & {str(due_a.id), str(due_b.id)} == set()
    s.close()
