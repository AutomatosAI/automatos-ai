"""F105 (night 3) — a best-effort write never freezes the event loop.

Night 3: with the connection pool dry (10 + 20 connections, 30 s wait), each
usage row, telemetry row, heartbeat finding and NL2SQL audit waited for a
connection ON the event loop. The whole backend froze for up to 30 s at a time
and unrelated turns' calls "timed out" together. These writes now run on
threads of their own; a turn that has nothing to do with them is not delayed.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from core import best_effort

STALL_S = 1.5          # how long the dry pool keeps a writer waiting
TURN_SLACK_S = 0.25    # the most an unrelated turn may be held up


class _DryPool:
    """A session whose checkout waits like a dry pool, then works."""
    sessions: list = []

    def __init__(self):
        time.sleep(STALL_S)
        self.added, self.executed = [], []
        _DryPool.sessions.append(self)

    def add(self, row):
        self.added.append(row)

    def execute(self, statement, params=None):
        self.executed.append(params)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _fresh_pool():
    _DryPool.sessions = []


async def _worst_stall(ticks: int = 60, tick: float = 0.01) -> float:
    """An unrelated turn: short awaits back to back; returns the longest it
    was kept waiting beyond each one."""
    worst, last = 0.0, time.monotonic()
    for _ in range(ticks):
        await asyncio.sleep(tick)
        now = time.monotonic()
        worst, last = max(worst, now - last - tick), now
    return worst


def test_a_stalled_telemetry_write_does_not_delay_an_unrelated_turn():
    from modules.tools.execution.telemetry import fire_telemetry

    async def main():
        fire_telemetry(tool_name="platform_search_documents", parameters={"query": "moreish"}, agent_id=None,
                       workspace_id=uuid4(), result={"success": True}, execution_time_ms=12,
                       session_factory=_DryPool)
        return await _worst_stall()

    worst = asyncio.run(main())
    assert worst < TURN_SLACK_S, f"an unrelated turn waited {worst:.2f}s behind a telemetry write"
    assert best_effort.drain(5) and _DryPool.sessions[0].added      # and the row was still written


def test_a_usage_row_waiting_for_a_connection_does_not_hold_the_loop(monkeypatch):
    import core.database.database as database
    from core.llm.usage_tracker import UsageTracker

    monkeypatch.setattr(database, "SessionLocal", _DryPool)

    async def main():
        turn = asyncio.ensure_future(_worst_stall())                   # an unrelated turn, under way
        await asyncio.sleep(0.05)
        UsageTracker.track(workspace_id=uuid4(), model_id="qwen/qwen3-embedding-8b", provider="openrouter",
                           input_tokens=12, output_tokens=0, request_type="embedding", cost_override=(0.0, 0.0))
        return await turn

    assert asyncio.run(main()) < TURN_SLACK_S
    assert best_effort.drain(5)
    assert _DryPool.sessions[0].added[0].request_type == "embedding"


def test_a_heartbeat_finding_waiting_for_a_connection_does_not_hold_the_loop(monkeypatch):
    import core.database.database as database
    from services.heartbeat_service import emit_primitive_finding

    monkeypatch.setattr(database, "SessionLocal", _DryPool)

    async def main():
        turn = asyncio.ensure_future(_worst_stall())                   # an unrelated turn, under way
        await asyncio.sleep(0.05)
        written = emit_primitive_finding(str(uuid4()), "chat", "green", "turn ok")
        return await turn, written

    worst, written = asyncio.run(main())
    assert worst < TURN_SLACK_S
    assert written is None                                            # handed over, not written inline
    assert best_effort.drain(5) and _DryPool.sessions[0].executed


def test_the_nl2sql_audit_waits_for_its_row_without_holding_the_loop(monkeypatch):
    import core.database.database as database
    from modules.nl2sql.service import DatabaseKnowledgeService

    monkeypatch.setattr(database, "SessionLocal", _DryPool)
    svc = DatabaseKnowledgeService.__new__(DatabaseKnowledgeService)

    async def main():
        audit = asyncio.ensure_future(svc.write_nl_audit(
            source_id=36, nl_query="how many boxes sold?", result={"success": True, "sql": "SELECT 1"}))
        worst = await _worst_stall()
        await audit
        return worst

    assert asyncio.run(main()) < TURN_SLACK_S
    assert _DryPool.sessions[0].added[0].source_id == 36


def test_a_write_that_cannot_get_a_connection_is_dropped_within_its_bound(monkeypatch, caplog):
    """The pool stays full: the write gives up after its own short wait, says
    so, and frees its thread — it never queues for the pool's 30 s."""
    import core.database.database as database
    from config import config

    monkeypatch.setattr(config, "BEST_EFFORT_POOL_WAIT_S", 0.2)
    monkeypatch.setattr(database, "engine", SimpleNamespace(pool=SimpleNamespace(
        checkedin=lambda: 0, overflow=lambda: 20, _max_overflow=20)))
    written = []

    @best_effort.off_loop
    def write():
        written.append(True)

    async def main():
        write()

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="core.best_effort"):
        asyncio.run(main())
        assert best_effort.drain(5)
    assert written == [] and time.monotonic() - started < 1.5
    assert "best-effort write dropped: no connection free within 0.2s" in caplog.text


def test_off_the_loop_a_writer_runs_inline_as_before():
    ran_on = []

    @best_effort.off_loop
    def write():
        ran_on.append(threading.current_thread().name)
        return True

    assert write() is True and ran_on == [threading.current_thread().name]


def test_a_handed_over_write_keeps_the_callers_context_and_never_raises(caplog):
    workspace = contextvars.ContextVar("workspace", default=None)
    seen = []

    @best_effort.off_loop
    def write():
        seen.append(workspace.get())

    @best_effort.off_loop
    def broken():
        raise RuntimeError("db down")

    async def main():
        workspace.set("c1")
        write()
        broken()                                                      # must not raise here

    with caplog.at_level(logging.WARNING, logger="core.best_effort"):
        asyncio.run(main())
        assert best_effort.drain(5)
    assert seen == ["c1"]
    assert "best-effort write failed: RuntimeError('db down')" in caplog.text


def test_drain_waits_for_a_failed_writes_log_line(monkeypatch, caplog):
    """A write's future tells its waiters it is done before its callback runs,
    and the callback logs the failure. drain() waited on the futures, so the
    test above read an empty log in a shared run (1 in 3 with the threads
    already started). A slow callback makes that order certain."""
    finished = best_effort._finished

    def slow_finished(future):
        time.sleep(0.2)
        finished(future)

    monkeypatch.setattr(best_effort, "_finished", slow_finished)

    @best_effort.off_loop
    def broken():
        raise RuntimeError("db down")

    async def main():
        broken()

    with caplog.at_level(logging.WARNING, logger="core.best_effort"):
        asyncio.run(main())
        assert best_effort.drain(5)
        assert "best-effort write failed: RuntimeError('db down')" in caplog.text
