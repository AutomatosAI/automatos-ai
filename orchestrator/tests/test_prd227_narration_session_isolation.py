"""PRD-227 P227-RVW-1 — mission narration must NEVER commit or roll back the
coordinator's SHARED session.

The review finding: US-002 narration fired the PRD-205 chat write on the
coordinator's own mid-transaction session. ``ChatService.save_message``
hard-commits (consumers/chatbot/service.py) and ``deliver_background_message``
rolls back on failure (services/chat_messenger.py); on the coordinator's session
— narration fires MID-transaction at approve_plan / _record_task_result /
cancel_mission / the tick terminal observer / approval-expiry, before the caller
commits — a transient chat-write failure would roll back the caller's uncommitted
transition (RUNNING + queued tasks) and silently strand the mission, while a
success would commit half-built state early.

The fix (``_narrate_mission``) routes the delivery through an INDEPENDENT
short-lived ``SessionLocal()``, confining any commit AND any rollback to the
message insert alone — the coordinator transaction is untouched on both paths,
matching US-001's ``notify_board_event`` (pg_notify-only, never commits the
caller's session).

Two layers:
  * PURE (run everywhere): the coordinator session is never committed/rolled
    back by narration; delivery gets a fresh session; and the messenger's
    contract for the OTHER producers (watch/scheduled) is unchanged — they still
    own their commit on their own session.
  * ``@integration`` (skips cleanly without Postgres; CI test.yml is the gate):
    drives the REAL post_background_message / ChatService.save_message against a
    session whose message-write commit RAISES, and proves a caller run flushed to
    RUNNING survives and persists; and that REAL approve_plan persists RUNNING
    with a no-op messenger (narration is not load-bearing for durability).
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

import services.coordinator_service as cs
from core.models.orchestration_enums import RunState, StateType


def _run(**kw):
    """Minimal OrchestrationRun-shaped object carrying the attrs narration reads."""
    import types

    base = dict(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        goal="Ship the Q3 report",
        created_by="user_abc",
        state=RunState.RUNNING.value,
        config={"origin_chat_id": str(uuid.uuid4())},
        plan={"tasks": [{"id": 1}, {"id": 2}]},
        stop_detail=None,
        stop_reason=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


# ===========================================================================
# PURE — session isolation invariant (run everywhere; no DB)
# ===========================================================================

def test_narration_delivery_runs_on_independent_session(monkeypatch):
    """_narrate_mission delivers on a FRESH SessionLocal(), never the caller's
    session — and never commits or rolls back the caller's session."""
    import core.database.database as dbmod
    import services.chat_messenger as cm

    narration_db = MagicMock(name="narration_db")
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: narration_db)

    seen = {}
    monkeypatch.setattr(cm, "deliver_background_message",
                        lambda db, **kw: seen.__setitem__("db", db))

    caller = MagicMock(name="coordinator_db")
    cs._narrate_mission(caller, _run(), "Mission approved", level="run", event="run_started")

    # Delivery ran on the independent session, NOT the coordinator's.
    assert seen["db"] is narration_db
    assert seen["db"] is not caller
    # The coordinator session was neither committed nor rolled back by narration.
    caller.commit.assert_not_called()
    caller.rollback.assert_not_called()
    # PRD-227 P227-RVW-5: the trailing chat_changed pg_notify is flushed by
    # committing the INDEPENDENT session on the success path — else close would
    # roll it back (reset_on_return='rollback') and the live SSE frame would never
    # fire. The commit lands on the narration session (never the caller's) and
    # happens BEFORE the close.
    narration_db.commit.assert_called_once()
    ordered = [c[0] for c in narration_db.method_calls if c[0] in ("commit", "close")]
    assert ordered == ["commit", "close"]
    # The short-lived session was closed.
    narration_db.close.assert_called_once()


def test_narration_failure_leaves_coordinator_session_untouched(monkeypatch):
    """A raising delivery must not touch (commit/rollback) the coordinator's
    session, must not propagate, and must still close the narration session."""
    import core.database.database as dbmod
    import services.chat_messenger as cm

    narration_db = MagicMock(name="narration_db")
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: narration_db)

    def _boom(db, **kw):
        raise RuntimeError("transient chat-write failure")

    monkeypatch.setattr(cm, "deliver_background_message", _boom)

    caller = MagicMock(name="coordinator_db")
    # Must not raise — narration is best-effort.
    cs._narrate_mission(caller, _run(), "x", level="run", event="run_started")

    caller.commit.assert_not_called()
    caller.rollback.assert_not_called()
    narration_db.close.assert_called_once()


def test_deliver_background_message_uses_callers_session(monkeypatch):
    """AC4 guard: the PRD-205 seam still writes on the CALLER's own session —
    the watch_notifications / scheduled_task producers own their own commit. The
    RVW-1 fix lives in the coordinator's narration path ONLY; it must not turn
    deliver_background_message into a self-sessioning call (that would break
    those producers' top-level transaction ownership)."""
    import services.chat_messenger as cm

    seen = {}
    monkeypatch.setattr(cm, "post_background_message",
                        lambda db, **kw: seen.setdefault("db", db))

    sentinel = object()
    cm.deliver_background_message(sentinel, workspace_id=uuid.uuid4(), text="x", source={})
    assert seen["db"] is sentinel


# ===========================================================================
# @integration — real Postgres round-trip (skips cleanly without a DB;
# CI test.yml is the gate). Mirrors the test_prd204_watch_service.py idiom.
# ===========================================================================

@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text

    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM orchestration_runs LIMIT 1"))
            c.execute(text("SELECT 1 FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"narration isolation suite needs a reachable Postgres with schema: {exc}"
        )
    yield eng
    eng.dispose()


@pytest.fixture
def seeded(new_session):
    """Workspace + creator user + the creator's origin chat, committed so a
    SEPARATE narration session can see them (committed rows cross connections).

    Yields ``(ws_id, clerk_id, chat_id)``. The chat is owned by the creator
    (``clerk_id``) — deliberately valid under the P227-RVW-2 owner check too, so
    this durability suite keeps passing once that fix lands. PRD-158 lesson:
    workspaces seeded FIRST for every FK-touching test.
    """
    from sqlalchemy import text

    from core.models.core import Chat, User

    ws_id = str(uuid.uuid4())
    clerk_id = f"user_rvw1_{uuid.uuid4().hex[:8]}"

    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd227-rvw1"},
    )
    s.commit()

    user = User(username=clerk_id, email=f"{clerk_id}@t.test", clerk_user_id=clerk_id)
    s.add(user)
    s.commit()

    chat = Chat(user_id=user.id, workspace_id=uuid.UUID(ws_id), title="origin")
    s.add(chat)
    s.commit()
    chat_id = str(chat.id)
    s.close()

    yield ws_id, clerk_id, chat_id

    s = new_session.sweep()
    s.execute(text("DELETE FROM messages WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id})
    s.execute(
        text(
            "DELETE FROM orchestration_events WHERE run_id IN "
            "(SELECT id FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid))"
        ),
        {"w": ws_id},
    )
    s.execute(
        text(
            "DELETE FROM orchestration_tasks WHERE run_id IN "
            "(SELECT id FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid))"
        ),
        {"w": ws_id},
    )
    s.execute(text("DELETE FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id})
    s.execute(text("DELETE FROM chats WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id})
    s.execute(text("DELETE FROM users WHERE clerk_user_id = :c"), {"c": clerk_id})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws_id})
    s.commit()
    s.close()


def test_narration_commit_failure_preserves_caller_run_running(seeded, new_session, engine, monkeypatch):
    """AC2 — REAL post_background_message / ChatService.save_message, message-write
    commit RAISES: a run flushed to RUNNING on the caller session SURVIVES the
    narration failure and persists on the caller's own commit."""
    from sqlalchemy.orm import sessionmaker

    from core.models.orchestration import OrchestrationRun

    ws_id, clerk_id, chat_id = seeded

    # A raising-commit session stands in for a transient message-write failure
    # (dropped connection / serialization error) at save_message's commit. Only
    # commit() raises; query/rollback/close stay REAL, so the genuine
    # post_background_message path runs and its fail-soft rollback is exercised —
    # on THIS session, never the caller's.
    raw_maker = sessionmaker(bind=engine)

    def _raising_session():
        s = raw_maker()

        def _boom():
            raise RuntimeError("simulated transient commit failure")

        s.commit = _boom
        return s

    monkeypatch.setattr("core.database.database.SessionLocal", _raising_session)

    # The caller (coordinator) flushes a run to RUNNING but does NOT commit —
    # exactly the mid-transaction window every narration call site fires in.
    caller = new_session()
    run = OrchestrationRun(
        workspace_id=uuid.UUID(ws_id),
        goal="RVW-1 durability",
        state=RunState.RUNNING.value,
        state_type=StateType.ACTIVE.value,
        created_by=clerk_id,
        config={"origin_chat_id": chat_id},
        plan={"tasks": [{"id": 1}]},
    )
    caller.add(run)
    caller.flush()  # written to the caller's tx, still UNCOMMITTED
    run_id = run.id

    # Drive the REAL messenger. save_message's message-write commit raises;
    # deliver_background_message rolls back the NARRATION session and swallows.
    cs._narrate_mission(caller, run, "Mission approved — starting", level="run", event="run_started")

    # Coordinator transaction intact: still RUNNING, and still uncommitted —
    # narration neither committed nor rolled back the caller's tx.
    assert run.state == RunState.RUNNING.value
    probe = new_session()
    assert probe.query(OrchestrationRun).get(run_id) is None, \
        "narration must NOT have committed the caller's uncommitted run"
    probe.close()

    # The caller's OWN commit still persists RUNNING — durability never depended
    # on narration.
    caller.commit()
    caller.close()

    verify = new_session()
    persisted = verify.query(OrchestrationRun).get(run_id)
    assert persisted is not None and persisted.state == RunState.RUNNING.value
    verify.close()


def test_approve_plan_persists_running_with_noop_messenger(seeded, new_session, engine, monkeypatch):
    """AC3 — narration is not load-bearing: REAL approve_plan followed by the
    caller's commit persists RUNNING even when narration delivers nothing."""
    from core.models.orchestration import OrchestrationRun

    ws_id, clerk_id, _chat_id = seeded

    # Disabled / no-op messenger: narration "delivers nothing".
    monkeypatch.setattr(
        "services.chat_messenger.deliver_background_message", lambda db, **kw: None
    )

    caller = new_session()
    run = OrchestrationRun(
        workspace_id=uuid.UUID(ws_id),
        goal="RVW-1 approve durability",
        state=RunState.AWAITING_APPROVAL.value,
        state_type=StateType.BLOCKED.value,
        created_by=clerk_id,
        config={},
        plan={"tasks": []},
    )
    caller.add(run)
    caller.commit()
    run_id = run.id

    result = cs.CoordinatorService().approve_plan(caller, run_id, clerk_id)
    assert result.state == RunState.RUNNING.value  # reached RUNNING despite no-op narration

    caller.commit()
    caller.close()

    verify = new_session()
    persisted = verify.query(OrchestrationRun).get(run_id)
    assert persisted is not None and persisted.state == RunState.RUNNING.value
    verify.close()


def test_narration_notify_reaches_board_events_listener(seeded, new_session, engine):
    """AC3 — a raw LISTEN on the 'board_events' channel RECEIVES a ``chat_changed``
    frame for a mission-narration line driven through the REAL ``_narrate_mission``
    → deliver_background_message → post_background_message path.

    Proves the P227-RVW-5 fix delivers the live push end-to-end: the independent
    narration session COMMITS its trailing ``pg_notify`` (before RVW-5 it was
    rolled back on close and Postgres never delivered it), so the message row AND
    the NOTIFY are both present. Uses the REAL ``SessionLocal`` (unpatched) so the
    genuine commit fires. Mirrors ``board_events._SSEListener``'s raw LISTEN;
    skips cleanly without Postgres (CI test.yml is the gate)."""
    import json as _json
    import select as _select
    import time as _time
    import types

    from core.models.core import Message
    from services.board_events import NOTIFY_CHANNEL

    ws_id, clerk_id, chat_id = seeded

    # Raw LISTEN on the SSE lane's channel, mirroring _SSEListener: autocommit so
    # the LISTEN registers immediately (a LISTEN inside a tx only takes on commit).
    raw = engine.raw_connection()
    try:
        raw.connection.autocommit = True
        cur = raw.cursor()
        cur.execute(f"LISTEN {NOTIFY_CHANNEL}")

        # Run whose origin is the creator's OWN chat (the owner check honours it),
        # narrated through the REAL messenger on the REAL independent SessionLocal.
        run = types.SimpleNamespace(
            id=uuid.uuid4(),
            workspace_id=uuid.UUID(ws_id),
            goal="RVW-5 live push",
            created_by=clerk_id,
            state=RunState.RUNNING.value,
            config={"origin_chat_id": chat_id},
            plan={"tasks": [{"id": 1}]},
            stop_detail=None,
            stop_reason=None,
        )

        caller = new_session()
        cs._narrate_mission(
            caller, run, "Mission approved — starting", level="run", event="run_started"
        )
        caller.close()  # caller session owns nothing here; narration used its own

        # Drain notifications for a short window (the NOTIFY fired on the narration
        # commit, which already returned above).
        got = []
        end = _time.time() + 5.0
        while _time.time() < end and not got:
            if _select.select([raw.connection], [], [], 1.0)[0]:
                raw.connection.poll()
                while raw.connection.notifies:
                    got.append(raw.connection.notifies.pop(0).payload)

        frames = [_json.loads(p) for p in got]
        chat_frames = [
            d for d in frames
            if d.get("event") == "chat_changed" and d.get("workspace_id") == ws_id
        ]
        assert chat_frames, f"no chat_changed NOTIFY received from narration: {got}"
        assert any(d.get("chat_id") == chat_id for d in chat_frames), (
            f"chat_changed NOTIFY did not target the origin chat {chat_id}: {chat_frames}"
        )
    finally:
        # Return the connection to the pool CLEAN (the _SSEListener lesson: a leaked
        # autocommit connection silently auto-commits the next SessionLocal caller).
        try:
            raw.connection.autocommit = False
        except Exception:  # noqa: BLE001
            pass
        raw.close()

    # The message row persisted too (save_message committed it) — the live NOTIFY
    # is not firing on an empty write.
    verify = new_session()
    msgs = verify.query(Message).filter(Message.chat_id == uuid.UUID(chat_id)).all()
    assert len(msgs) >= 1
    verify.close()
