"""F408 (night 13) — the signal recorder's batch write never blocks the event loop.

At 03:01 UTC the event loop stood still for 6 minutes. The recorder's drain task
ran its UPDATE tool_routing_edges synchronously on the loop and waited on a row
lock held by another session of the same process, left idle in transaction after
its own INSERT; that session's coroutine could not resume to commit while the
loop was blocked. /health timed out until the backend was terminated by hand.

The batch now runs in a worker thread, in one short transaction under a
transaction-local lock_timeout (TOOL_SIGNAL_FLUSH_LOCK_TIMEOUT_MS): a lock not had
in time drops that batch with one warning, and the transaction is rolled back.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

import pytest
from sqlalchemy import create_engine, text

from modules.tools.discovery.signal_recorder import ToolSignal, ToolSignalRecorder

LOCK_TIMEOUT_MS = 500   # the flush gives up a row lock after this
TICK = 0.02             # the /health stand-in answers this often while the loop runs
WORST_STALL = 0.25      # well under the lock wait: the loop never waits on the lock
SLACK = 2.0             # a dropped batch returns within the lock_timeout plus this


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM tool_routing_edges LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the recorder's lock wait needs a reachable Postgres with tool_routing_edges: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def actions(new_session):
    """A (from, to) action pair no other test writes, swept afterwards."""
    prefix = f"f408-{uuid.uuid4().hex[:8]}"
    yield f"{prefix}-a", f"{prefix}-b"
    s = new_session.sweep()
    s.execute(text("DELETE FROM tool_routing_edges WHERE from_action LIKE :p"), {"p": f"{prefix}%"})
    s.execute(text("DELETE FROM tool_routing_affinities WHERE action_name LIKE :p"), {"p": f"{prefix}%"})
    s.commit()


@pytest.fixture
def short_lock_timeout(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "TOOL_SIGNAL_FLUSH_LOCK_TIMEOUT_MS", LOCK_TIMEOUT_MS)


def _edge(new_session, a, b):
    return new_session().execute(text(
        "SELECT sample_count, weight FROM tool_routing_edges WHERE from_action = :a AND to_action = :b "
        "AND edge_type = 'used_after' AND workspace_id IS NULL AND agent_id IS NULL"), {"a": a, "b": b}).one()


async def _flush_beside_the_health_check(recorder, batch):
    """Flush while a concurrent coroutine answers every TICK: how long the
    flush took, the loop's worst stall, and how often the coroutine ran."""
    stalls, ticks, stop = [0.0], [0], asyncio.Event()

    async def _health():
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(TICK)
            now = time.monotonic()
            stalls.append(now - last)
            ticks[0] += 1
            last = now

    health = asyncio.create_task(_health())
    started = time.monotonic()
    try:
        await recorder._flush(batch)
    finally:
        took = time.monotonic() - started
        stop.set()
        await health
    return took, max(stalls), ticks[0]


def _idle_in_transaction_on(engine, action, except_pid):
    """Backends left idle in transaction whose last statement named ``action``
    (psycopg2 binds client-side, so the value is in the statement text)."""
    with engine.connect() as c:
        return c.execute(text(
            "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
            "AND state LIKE 'idle in transaction%' AND pid <> :pid AND query LIKE :named"),
            {"pid": except_pid, "named": f"%{action}%"}).scalar()


def test_a_normal_batch_upserts_its_edges(actions, new_session, short_lock_timeout):
    a, b = actions
    recorder = ToolSignalRecorder()
    signal = ToolSignal(b, True, prior_action=a)

    asyncio.run(recorder._flush([signal, signal]))   # inserts the edge with 2
    asyncio.run(recorder._flush([signal]))           # increments it

    edge = _edge(new_session, a, b)
    assert edge.sample_count == 3 and edge.weight == 3
    assert recorder.stats()["flushes"] == 2
    assert recorder.stats()["flush_errors"] == 0 and recorder.stats()["lock_drops"] == 0


def test_a_batch_waiting_on_a_held_row_lock_never_stops_the_loop(
        actions, engine, new_session, short_lock_timeout, caplog):
    a, b = actions
    recorder = ToolSignalRecorder()
    signal = ToolSignal(b, True, prior_action=a)
    asyncio.run(recorder._flush([signal]))  # the edge exists, sample_count 1

    holder = engine.connect()  # another session holds the edge's row lock, uncommitted
    try:
        holder_pid = holder.execute(text("SELECT pg_backend_pid()")).scalar()
        holder.execute(text("SELECT id FROM tool_routing_edges WHERE from_action = :a FOR UPDATE"), {"a": a})
        with caplog.at_level(logging.WARNING, logger="modules.tools.discovery.signal_recorder"):
            took, worst_stall, ticks = asyncio.run(_flush_beside_the_health_check(recorder, [signal]))
        stuck = _idle_in_transaction_on(engine, a, holder_pid)
    finally:
        holder.rollback()
        holder.close()

    assert took < LOCK_TIMEOUT_MS / 1000 + SLACK, f"the dropped batch took {took:.2f}s"
    assert worst_stall < WORST_STALL, f"the event loop stalled {worst_stall:.2f}s"
    assert ticks >= (LOCK_TIMEOUT_MS / 1000) / TICK / 2   # the concurrent coroutine ran meanwhile
    assert recorder.stats()["lock_drops"] == 1 and recorder.stats()["flush_errors"] == 0
    warnings = [r for r in caplog.records if "row lock was not had" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert stuck == 0, "the failed batch left a connection idle in transaction"
    assert _edge(new_session, a, b).sample_count == 1   # the batch was dropped, not half-written

    asyncio.run(recorder._flush([signal]))  # the lock released: the next batch writes
    assert _edge(new_session, a, b).sample_count == 2
    assert recorder.stats()["flushes"] == 2
