"""PRD-180 S1 (F090) — the board SSE is a real LISTEN/NOTIFY stream, not a ping.

Two layers, mirroring the dispatch suite (``test_board_dispatch.py``):

* **DB-backed (real Postgres):** ``LISTEN/NOTIFY`` is Postgres-only, so the
  end-to-end tests use committed cross-connection sessions and skip cleanly when
  no DB is up. They assert (a) ``notify_board_event`` publishes a JSON payload a
  raw LISTENer receives on the ``board_events`` channel, and (b) the
  ``board_event_stream`` async generator yields a ``board_changed`` frame
  carrying the changed task's status when a NOTIFY for that workspace arrives —
  i.e. driven by a real event, not a timed ping.
* **Pure (no DB):** the SSE generator's workspace isolation + frame formatting,
  driven by injecting notifications straight onto its internal queue (the PG
  NOTIFY channel mocked at the boundary), so cross-tenant events never leak and
  the heartbeat is a comment, not a refresh.
"""
from __future__ import annotations

import asyncio
import json
import select
import sys
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ``board_events`` imports its DB connection lazily (inside the listener thread),
# so importing it does NOT touch Postgres — the pure tests below run with no DB.
# ``get_database_url`` (which the app evaluates at import) is imported lazily in
# the ``engine`` fixture instead, so a missing DB only skips the DB-backed tests
# rather than crashing collection of the pure ones.
_ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCH_ROOT))

from services import board_events  # noqa: E402


# ─────────────────────────── DB-backed (real Postgres) ──────────────────────

@pytest.fixture(scope="module")
def engine():
    """Real Postgres engine; skip the DB-backed tests cleanly when none is up.

    ``get_database_url`` is imported here (not at module top) so a dev box with
    no DB creds skips these tests instead of failing collection of the pure ones.
    """
    try:
        from core.database.database import get_database_url

        eng = create_engine(get_database_url(), pool_pre_ping=True, pool_size=6, max_overflow=4)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"board SSE suite needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    """A sessionmaker handing out independent, committing sessions."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_notify_board_event_reaches_a_raw_listener(new_session, engine):
    """``notify_board_event`` publishes a JSON payload on the board_events channel
    that a raw LISTEN connection receives within 1s (the NOTIFY leg works)."""
    raw = engine.raw_connection()
    try:
        raw.connection.autocommit = True
        cur = raw.cursor()
        cur.execute(f"LISTEN {board_events.NOTIFY_CHANNEL}")

        ws_id = str(uuid.uuid4())
        s = new_session()
        board_events.notify_board_event(
            s, workspace_id=ws_id, task_id=777, status="in_progress",
            event="task_claimed",
        )
        s.commit()
        s.close()

        deadline = time.monotonic() + 1.0
        payload = None
        while time.monotonic() < deadline and payload is None:
            if select.select([raw.connection], [], [], deadline - time.monotonic())[0]:
                raw.connection.poll()
                while raw.connection.notifies:
                    note = raw.connection.notifies.pop(0)
                    if note.channel == board_events.NOTIFY_CHANNEL:
                        payload = note.payload
        assert payload is not None, "no board_events NOTIFY received within 1s"
        data = json.loads(payload)
        assert data["workspace_id"] == ws_id
        assert data["task_id"] == 777
        assert data["status"] == "in_progress"
        assert data["event"] == "task_claimed"
    finally:
        # Reset autocommit before returning the pooled connection — otherwise it
        # goes back to the pool ``autocommit=True`` and poisons the next
        # ``SessionLocal()`` (SQLAlchemy's reset-on-return does not clear psycopg2
        # autocommit). Mirrors the production fix in ``board_events`` and stops
        # this test from tripping the W2-S5 idle-in-transaction regression.
        try:
            raw.connection.autocommit = False
        except Exception:
            pass
        raw.close()


@pytest.mark.asyncio
async def test_board_sse_listen_notify(new_session):
    """The SSE stream yields an event when a board task changes status (the NOTIFY
    path), not merely a timed ping.

    The generator LISTENs board_events; we fire ``notify_board_event`` for its
    workspace and assert the next yielded frame is a ``board_changed`` carrying
    the changed task's new status. The heartbeat is set far higher than the test
    window so any frame we get is NOTIFY-driven, never a ping.
    """
    ws_id = str(uuid.uuid4())
    # Big heartbeat: within the test window, only a real NOTIFY can produce a frame.
    stream = board_events.board_event_stream(ws_id, heartbeat_seconds=30.0)

    # First frame is the initial connect sync (renders current state immediately).
    first = await asyncio.wait_for(stream.__anext__(), timeout=5.0)
    assert first.startswith("event: board_changed")

    # Give the listener thread a moment to run its LISTEN before we NOTIFY.
    await asyncio.sleep(0.3)

    s = new_session()
    board_events.notify_board_event(
        s, workspace_id=ws_id, task_id=4321, status="done", event="status_changed",
    )
    s.commit()
    s.close()

    # The next frame must be the pushed event, arriving well inside the heartbeat.
    frame = await asyncio.wait_for(stream.__anext__(), timeout=5.0)
    assert frame.startswith("event: board_changed"), f"expected a pushed event frame, got {frame!r}"
    data = json.loads(frame.split("data: ", 1)[1].strip())
    assert data["task_id"] == 4321
    assert data["status"] == "done"
    assert data["workspace_id"] == ws_id

    await stream.aclose()


@pytest.mark.asyncio
async def test_sse_stream_drops_other_workspace_events(new_session):
    """A NOTIFY for a DIFFERENT workspace never reaches this tenant's stream, and
    the heartbeat comment fires instead (no cross-tenant leak)."""
    my_ws = str(uuid.uuid4())
    other_ws = str(uuid.uuid4())
    # Short heartbeat so we can observe the heartbeat comment quickly.
    stream = board_events.board_event_stream(my_ws, heartbeat_seconds=0.5)

    first = await asyncio.wait_for(stream.__anext__(), timeout=5.0)
    assert first.startswith("event: board_changed")  # initial sync

    await asyncio.sleep(0.3)
    s = new_session()
    board_events.notify_board_event(
        s, workspace_id=other_ws, task_id=999, status="done", event="status_changed",
    )
    s.commit()
    s.close()

    # The other-workspace event is dropped; the next frame is a heartbeat comment.
    frame = await asyncio.wait_for(stream.__anext__(), timeout=5.0)
    assert frame.startswith(": hb"), f"other-workspace event must not leak; got {frame!r}"

    await stream.aclose()


# ─────────────────────────── Pure (PG NOTIFY mocked) ─────────────────────────

def test_parse_payload_rejects_malformed():
    """Malformed / non-dict payloads parse to None (never crash the stream)."""
    assert board_events._parse_payload("not json") is None
    assert board_events._parse_payload("[1,2,3]") is None
    assert board_events._parse_payload('{"workspace_id":"w","task_id":1}') == {
        "workspace_id": "w",
        "task_id": 1,
    }


def test_frame_formats_a_valid_sse_event():
    """``_frame`` emits a well-formed ``event:``/``data:`` SSE frame."""
    frame = board_events._frame("board_changed", {"task_id": 7, "status": "done"})
    assert frame.startswith("event: board_changed\ndata: ")
    assert frame.endswith("\n\n")
    body = frame.split("data: ", 1)[1].strip()
    assert json.loads(body) == {"task_id": 7, "status": "done"}


def test_frame_for_payload_isolates_and_frames(monkeypatch):
    """The pure routing gate (PG NOTIFY channel mocked = payloads fed directly):
    a matching-workspace payload becomes a frame; a mismatching one, a malformed
    one, and a non-dict one are all dropped (``None``)."""
    my_ws = "ws-match"
    match = json.dumps({"workspace_id": my_ws, "task_id": 5, "status": "in_progress"})
    other = json.dumps({"workspace_id": "ws-other", "task_id": 6, "status": "done"})

    frame = board_events.frame_for_payload(match, my_ws)
    assert frame is not None and frame.startswith("event: board_changed")
    data = json.loads(frame.split("data: ", 1)[1].strip())
    assert data["task_id"] == 5 and data["status"] == "in_progress"

    # Cross-tenant, malformed, and non-dict payloads are all dropped.
    assert board_events.frame_for_payload(other, my_ws) is None
    assert board_events.frame_for_payload("not json", my_ws) is None
    assert board_events.frame_for_payload("[1,2]", my_ws) is None
