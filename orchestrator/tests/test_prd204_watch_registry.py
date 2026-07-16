"""PRD-204 S1 -- watch registry schema: model round-trip, dedup, FK integrity.

DB-backed (real Postgres): the partial unique index and ON CONFLICT semantics
are Postgres-only, so this suite follows the test.yml Postgres pattern
(test_board_dispatch.py precedent) and skips cleanly when no DB is reachable.

PRD-158 lesson: every test touching FK'd tables seeds ``workspaces`` FIRST.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from core.database.database import get_database_url
from core.models.watches import Watch, WatchEvent
from core.models.watch_enums import WatchStatus


@pytest.fixture(scope="module")
def engine():
    """Real Postgres engine; skip the whole module cleanly when none is up."""
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
            c.execute(text("SELECT 1 FROM watches LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"watch registry suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    """A sessionmaker that hands out independent, committing sessions."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def workspace(engine, new_session):
    """Throwaway workspace (seeded FIRST -- PRD-158). Yields workspace_id str."""
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd204-watch-registry"},
    )
    s.commit()
    s.close()

    yield ws_id

    s = new_session()
    # watch_events cascade off watches; watches cascade off workspaces --
    # explicit deletes keep teardown independent of CASCADE behaviour.
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
        text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"),
        {"w": ws_id},
    )
    s.commit()
    s.close()


def _make_watch(ws_id: str, **overrides) -> Watch:
    defaults = dict(
        workspace_id=ws_id,
        created_by="user_test",
        watch_type="mission",
        target_type="mission",
        target_id=str(uuid.uuid4()),
        title="Watch: test mission",
        success_criteria="The mission produces a report",
    )
    defaults.update(overrides)
    return Watch(**defaults)


# ---------------------------------------------------------------------------
# Model round-trip + defaults
# ---------------------------------------------------------------------------


def test_watch_round_trip_and_defaults(workspace, new_session):
    s = new_session()
    watch = _make_watch(workspace)
    s.add(watch)
    s.commit()
    watch_id = watch.id
    s.close()

    s = new_session()
    loaded = s.query(Watch).filter(Watch.id == watch_id).one()
    assert str(loaded.workspace_id) == workspace
    assert loaded.status == WatchStatus.WATCHING.value
    assert loaded.quality_threshold == pytest.approx(0.8)
    assert loaded.check_interval_seconds == 300
    assert loaded.policy == "run_and_report"
    assert loaded.action_budget == 2
    assert loaded.actions_taken == 0
    assert loaded.lineage == []
    assert loaded.version_id == 1
    assert loaded.closed_at is None
    assert loaded.created_at is not None
    s.close()


def test_watch_event_round_trip(workspace, new_session):
    s = new_session()
    watch = _make_watch(workspace)
    s.add(watch)
    s.commit()

    event = WatchEvent(
        watch_id=watch.id,
        event_type="terminal",
        summary="mission completed",
        snapshot={"tokens_used": 1200, "output_pointer": "mission:x:output_summary"},
        event_key="terminal:mission:abc",
    )
    s.add(event)
    s.commit()
    event_id = event.id
    s.close()

    s = new_session()
    loaded = s.query(WatchEvent).filter(WatchEvent.id == event_id).one()
    assert loaded.event_type == "terminal"
    assert loaded.snapshot["tokens_used"] == 1200
    assert loaded.requires_attention is False
    assert loaded.score is None
    s.close()


# ---------------------------------------------------------------------------
# Dedup indexes
# ---------------------------------------------------------------------------


def test_one_live_watch_per_target(workspace, new_session):
    """Partial unique index: a second NON-terminal watch on the same target
    is rejected; once the first closes, a new watch on the target is fine."""
    target = str(uuid.uuid4())

    s = new_session()
    s.add(_make_watch(workspace, target_id=target))
    s.commit()

    s.add(_make_watch(workspace, target_id=target))
    with pytest.raises(IntegrityError):
        s.commit()
    s.rollback()

    # Close the live watch -> the partial index no longer applies to it.
    s.execute(
        text(
            "UPDATE watches SET status = 'passed', closed_at = NOW() "
            "WHERE workspace_id = CAST(:w AS uuid) AND target_id = :t"
        ),
        {"w": workspace, "t": target},
    )
    s.commit()

    s.add(_make_watch(workspace, target_id=target))
    s.commit()  # must not raise

    live = s.execute(
        text(
            "SELECT COUNT(*) FROM watches "
            "WHERE workspace_id = CAST(:w AS uuid) AND target_id = :t"
        ),
        {"w": workspace, "t": target},
    ).scalar()
    assert live == 2
    s.close()


def test_event_key_unique_per_watch(workspace, new_session):
    """UNIQUE(watch_id, event_key): the same observation lands exactly once,
    but the same key on a DIFFERENT watch is allowed."""
    s = new_session()
    w1 = _make_watch(workspace)
    w2 = _make_watch(workspace)
    s.add_all([w1, w2])
    s.commit()

    s.add(WatchEvent(watch_id=w1.id, event_type="terminal", event_key="k1"))
    s.commit()

    s.add(WatchEvent(watch_id=w1.id, event_type="terminal", event_key="k1"))
    with pytest.raises(IntegrityError):
        s.commit()
    s.rollback()

    # Same key, different watch -> fine.
    s.add(WatchEvent(watch_id=w2.id, event_type="terminal", event_key="k1"))
    s.commit()
    s.close()


# ---------------------------------------------------------------------------
# FK integrity
# ---------------------------------------------------------------------------


def test_watch_requires_existing_workspace(new_session):
    s = new_session()
    s.add(_make_watch(str(uuid.uuid4())))  # workspace never seeded
    with pytest.raises(IntegrityError):
        s.commit()
    s.rollback()
    s.close()


def test_workspace_delete_cascades_watches_and_events(engine, new_session):
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n)"),
        {"id": ws_id, "n": "prd204-cascade"},
    )
    s.commit()

    watch = _make_watch(ws_id)
    s.add(watch)
    s.commit()
    s.add(WatchEvent(watch_id=watch.id, event_type="created", event_key="created"))
    s.commit()
    watch_id = str(watch.id)

    s.execute(
        text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws_id}
    )
    s.commit()

    remaining_watches = s.execute(
        text("SELECT COUNT(*) FROM watches WHERE id = CAST(:i AS uuid)"),
        {"i": watch_id},
    ).scalar()
    remaining_events = s.execute(
        text("SELECT COUNT(*) FROM watch_events WHERE watch_id = CAST(:i AS uuid)"),
        {"i": watch_id},
    ).scalar()
    assert remaining_watches == 0
    assert remaining_events == 0
    s.close()


# ---------------------------------------------------------------------------
# Deadline / scheduling columns behave as timestamps
# ---------------------------------------------------------------------------


def test_schedule_columns_round_trip(workspace, new_session):
    now = datetime.now(timezone.utc)
    s = new_session()
    watch = _make_watch(
        workspace,
        next_check_at=now + timedelta(seconds=300),
        deadline_at=now + timedelta(hours=4),
    )
    s.add(watch)
    s.commit()
    loaded = s.query(Watch).filter(Watch.id == watch.id).one()
    assert loaded.next_check_at > now
    assert loaded.deadline_at > loaded.next_check_at
    s.close()
