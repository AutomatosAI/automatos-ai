"""
PRD-232 US-009 — turn the signal recorder on (decision LOCKED §6.1).
====================================================================

C3: the write side of the learning loop was dark — TOOL_SIGNAL_RECORDER_ENABLED
defaulted false, so only the nightly edge_builder learned and harness stats read
all-zero. US-009 flips the default ON and hardens the clean-shutdown path so a
graceful stop loses no queued signals (PRD-142 W4-S9), while keeping PRD-141
US-019's contract: ONE DB session per flush, never a session/task per call.

Covered:
- default flip (config + _enabled());
- load-shape: N records collapse to ceil(N/batch) flushes, exactly one session
  each (the US-019 invariant, extended not duplicated);
- flush-on-stop: stop() flushes the in-flight batch AND the queue remainder, so
  every queued signal is persisted exactly once (no loss, no double-write).
"""
from __future__ import annotations

import asyncio
import math
from contextlib import contextmanager

from modules.tools.discovery.signal_recorder import ToolSignal, ToolSignalRecorder


# ── AC1: default ON ──────────────────────────────────────────────────────────
def test_recorder_enabled_by_default():
    from config import config
    assert config.TOOL_SIGNAL_RECORDER_ENABLED is True, "US-009 (§6.1): default must be ON"
    assert ToolSignalRecorder._enabled() is True


# ── AC3: load-shape — ceil(N/batch) flushes, one session each ────────────────
class _SessionFactory:
    """Counts session opens. Yields a minimal db (only .flush() is used once the
    upserts are stubbed) — the invariant under test is 'one session per flush'."""

    def __init__(self):
        self.opens = 0

    @contextmanager
    def __call__(self):
        self.opens += 1
        yield type("_DB", (), {"flush": lambda self: None})()


def test_load_shape_one_session_per_flush(monkeypatch):
    import core.database.database as dbmod

    factory = _SessionFactory()
    monkeypatch.setattr(dbmod, "get_db_session", factory)
    # isolate the batching/session invariant from the upsert SQL (tested elsewhere)
    monkeypatch.setattr(ToolSignalRecorder, "_upsert_edge", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(ToolSignalRecorder, "_upsert_affinity", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(ToolSignalRecorder, "_batch_size", staticmethod(lambda: 10))
    monkeypatch.setattr(ToolSignalRecorder, "_interval_seconds", staticmethod(lambda: 5.0))

    N, B = 30, 10

    async def _run():
        r = ToolSignalRecorder()
        r._queue = asyncio.Queue()
        for i in range(N):
            # every signal yields an affinity, so each non-empty batch opens a session
            r._queue.put_nowait(ToolSignal(f"act{i % 3}", i % 2 == 0, agent_id=1, workspace_id="ws"))
        flushes = 0
        while not r._queue.empty():
            batch = await r._collect_batch()      # fills to batch_size by size (all pre-queued)
            await r._flush(batch)
            flushes += 1
        return flushes, r.stats()

    flushes, stats = asyncio.run(_run())
    assert flushes == math.ceil(N / B)            # 3 flushes for 30 @ batch 10
    assert flushes <= math.ceil(N / B)            # AC bound (<=)
    assert factory.opens == flushes               # exactly ONE session per flush
    assert stats["flushes"] == flushes


# ── AC2: flush-on-shutdown — no queued signal is lost ────────────────────────
def test_stop_flushes_queued_signals_no_loss(monkeypatch):
    flushed = []

    async def spy_flush(self, batch):
        flushed.extend(batch)
        self._stats["flushes"] += 1

    monkeypatch.setattr(ToolSignalRecorder, "_flush", spy_flush)
    monkeypatch.setattr(ToolSignalRecorder, "_batch_size", staticmethod(lambda: 50))
    monkeypatch.setattr(ToolSignalRecorder, "_interval_seconds", staticmethod(lambda: 0.02))

    async def _run():
        r = ToolSignalRecorder()
        r._ensure_started(asyncio.get_running_loop())   # queue + single drain task
        for i in range(3):
            r._queue.put_nowait(ToolSignal(f"a{i}", True, agent_id=1, workspace_id="ws"))
        await r.stop()                                   # must flush all 3 before returning
        return r

    r = asyncio.run(_run())
    names = sorted(s.action_name for s in flushed)
    assert names == ["a0", "a1", "a2"], f"lost or duplicated signals: {names}"
    assert r._queue.empty()
    assert r._drain_task is None


def test_stop_is_safe_when_never_started():
    """stop() on a recorder that never ran (no queue/task) is a no-op, not a crash."""
    r = ToolSignalRecorder()
    asyncio.run(r.stop())
    assert r._drain_task is None


def test_stop_flushes_in_flight_batch_even_if_drain_blocked(monkeypatch):
    """The drain loop dequeues 'first' and blocks waiting for more (long interval).
    stop() must still flush that in-flight signal — the sentinel wakes it."""
    flushed = []

    async def spy_flush(self, batch):
        flushed.extend(batch)

    monkeypatch.setattr(ToolSignalRecorder, "_flush", spy_flush)
    monkeypatch.setattr(ToolSignalRecorder, "_batch_size", staticmethod(lambda: 50))
    monkeypatch.setattr(ToolSignalRecorder, "_interval_seconds", staticmethod(lambda: 30.0))

    async def _run():
        r = ToolSignalRecorder()
        r._ensure_started(asyncio.get_running_loop())
        r._queue.put_nowait(ToolSignal("solo", True, agent_id=1, workspace_id="ws"))
        await asyncio.sleep(0.01)   # let the drain loop dequeue 'first' and block for more
        await r.stop()
        return r

    asyncio.run(_run())
    assert [s.action_name for s in flushed] == ["solo"], "in-flight batch lost on stop"
