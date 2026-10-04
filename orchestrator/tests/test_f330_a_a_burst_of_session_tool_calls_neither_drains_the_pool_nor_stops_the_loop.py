"""F330 (night 9c) — a burst of session tool calls neither drains the pool nor stops the loop.

Night 9c, build 18 (4 Oct, ~20:20Z): 26 cards were filed at once with no
``--max-sessions`` cap on the CLI host, so up to 26 Claude Code sessions called
``query_database`` and the other session tools together. Twelve minutes in, the
pool (10 + 20) was gone: 29 connections "idle in transaction" for 12+ minutes,
the event loop stuck in ``pool._do_get``, /health timing out. Each call kept the
connection its token lookup opened while the tool ran (seconds of model calls),
and the tool then opened a session of its own, on the event loop: once every
connection belonged to a call waiting for one more, nothing could move.

Here 40 calls go at once against a pool of 6 (real Postgres, its own engine).
Each tool does what the real ones do: it reads on the request's session (the
executor's agent row), waits on a model, then opens a session of its own on the
loop (NL2SQL's source lookup). Every call must be answered, in a bounded time,
and a heartbeat on the loop must keep beating throughout.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import api.session_tools as api_st
from services import session_tools as st

CALLS = 40                    # far more calls than the pool has connections
POOL_SIZE = 6
SLOTS = 2                     # SESSION_TOOL_CONCURRENCY here: two calls hold at most four
POOL_WAIT_S = 1.0             # the real pool waits 30 s; this one gives up sooner
MODEL_S = 0.1                 # the tool's model call
BURST_BOUND_S = 20.0          # 40 calls, two at a time, take about 2 s
LOOP_STALL_S = 0.5            # a loop that waited on the pool stood still for POOL_WAIT_S
HEARTBEAT_S = 0.01
QUESTION = {"question": "How many orders shipped last week?"}


@pytest.fixture
def engine(test_db_url):
    eng = create_engine(test_db_url, pool_size=POOL_SIZE, max_overflow=0, pool_timeout=POOL_WAIT_S)
    yield eng
    eng.dispose()


class _Request:
    """One MCP ``tools/call`` from one ticket's session."""

    def __init__(self, number: int):
        self.number = number
        self.headers = {"Authorization": f"Bearer ticket-{number}"}

    async def json(self):
        return {"jsonrpc": "2.0", "id": self.number, "method": "tools/call",
                "params": {"name": "query_database", "arguments": dict(QUESTION)}}


@pytest.fixture
def platform(engine, monkeypatch):
    """The route as it runs, with the ticket lookup, the counter and the tool
    faked only as far as their database use: each reads or writes for real."""
    in_flight = {"now": 0, "peak": 0}

    def resolve(db, token):                            # the ticket and its agent row
        db.execute(text("SELECT 1")).all()
        number = int(token.rsplit("-", 1)[1])
        return (SimpleNamespace(id=number, assigned_agent_id=268, workspace_id="ws-f330", runtime_ref={}),
                SimpleNamespace(name="ANALYST"))

    def count(db, task):                               # the counter's UPDATE and commit
        db.execute(text("SELECT 1")).all()
        db.commit()
        return 1

    async def tool(db, chosen, params, ctx):
        in_flight["now"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        try:
            db.execute(text("SELECT 1")).all()         # the executor's own reads
            await asyncio.sleep(MODEL_S)               # the model writing the SQL
            with Session(engine) as own:               # the handler's own session, on the loop
                own.execute(text("SELECT 412")).all()
            return {"success": True, "result": {"orders": 412, "ticket": ctx.task_id}}
        finally:
            in_flight["now"] -= 1

    monkeypatch.setattr(api_st, "_require_cli_runtime", lambda: None)
    monkeypatch.setattr(api_st.svc, "resolve_session_token", resolve)
    monkeypatch.setattr(api_st, "_count_call", count)
    monkeypatch.setattr(st, "call_tool", tool)
    # raising=False: run against the code before F330, the burst itself fails, not this line.
    monkeypatch.setattr(api_st.config, "SESSION_TOOL_CONCURRENCY", SLOTS, raising=False)
    return in_flight


async def _one_call(engine, number: int):
    db = Session(engine)                                # what get_db hands the request
    try:
        request = _Request(number)
        session = await api_st.require_session(request, db=db)
        response = await api_st.session_tools_mcp(request, session=session, db=db)
        return json.loads(response.body)
    finally:
        db.close()


async def _heartbeat(beats, stop):
    while not stop.is_set():
        beats.append(time.monotonic())
        await asyncio.sleep(HEARTBEAT_S)


async def _burst(engine):
    beats, stop = [], asyncio.Event()
    beating = asyncio.create_task(_heartbeat(beats, stop))
    started = time.monotonic()
    try:
        replies = await asyncio.wait_for(
            asyncio.gather(*(_one_call(engine, n) for n in range(CALLS)), return_exceptions=True),
            BURST_BOUND_S,
        )
    finally:
        stop.set()
        await beating
    return replies, time.monotonic() - started, beats


def _answered(reply, number: int) -> bool:
    if not isinstance(reply, dict) or reply.get("id") != number:
        return False
    result = reply.get("result") or {}
    return not result.get("isError") and '"orders": 412' in result["content"][0]["text"]


def test_forty_calls_on_a_pool_of_six_are_all_answered_and_the_loop_keeps_beating(engine, platform):
    replies, elapsed, beats = asyncio.run(_burst(engine))

    unanswered = [(n, repr(r)[:200]) for n, r in enumerate(replies) if not _answered(r, n)]
    assert unanswered == []
    assert elapsed < BURST_BOUND_S
    longest_stall = max(later - earlier for earlier, later in zip(beats, beats[1:]))
    assert longest_stall < LOOP_STALL_S, f"the event loop stood still for {longest_stall:.2f}s"
    assert platform["peak"] <= SLOTS                   # the burst queued for a slot, not for the pool
    assert engine.pool.checkedout() == 0                # and nothing was left held
