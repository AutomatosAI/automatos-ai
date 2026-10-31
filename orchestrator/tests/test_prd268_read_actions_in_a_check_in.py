"""PRD-268 P268W1-RVW-1 (§6.2.3, FR-1, K17): ``read`` means read in the check-in lane.

The lane clears every action labelled ``read``, and some of them kept a record of their own
as they read, on a path a check-in reaches with a real workspace's data:
``platform_search_packages`` stamped the onboarding funnel; ``platform_codegraph_search``
and ``platform_codegraph_get_symbol`` logged a ``codegraph_query_logs`` row;
``platform_read_document`` wrote a spreadsheet's copy into the workspace;
``platform_query_data`` wrote its audit row, a training example, a primitive heartbeat and
the use count of each few-shot example it was handed;
``platform_search_documents`` stamped ``last_accessed`` and the substrate tile's event (the
memory searches write that event too); ``platform_render_preview`` writes its picture into a
session's folder once a ticket is injected.

None is relabelled (K17 allows either way): each stays ``read`` and writes nothing in a
check-in. The lane runs every check-in call but the suggestion inside
``core.read_only_call.reading_only()``, and each record is skipped there. Relabelling them
``write`` would have made Strict ask before a document, a data or a code read in chat, made
"what Auto did" count them as changes, and spent the write rate limit on reads.

- each seeded action (``tests/helpers_check_in_seeds.py``), run through
  ``PlatformActionExecutor.execute`` with ``caller_context`` lane ``check_in``, writes
  nothing on its session, on one it opens itself or over HTTP, and answers as it should;
- the lane marks every check-in call but the suggestion, and nothing outside the lane;
- the mark reaches the work a call starts (a task, a thread);
- the records a seed does not reach on their own are skipped too: the NL2SQL primitive
  heartbeat, the training store (no example stored, no use counted) and the substrate tile's
  event (both calls).

The sweep (``test_prd268_read_actions_write_nothing.py``) runs the same seeds handler by handler,
and outside a check-in, to show each seed reaches its write.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from core.read_only_call import only_reading, reading_only
from modules.tools.discovery.check_in_lane import CHECK_IN_REFUSAL, SUGGEST_NEXT_STEP
from tests.helpers_check_in_seeds import SEEDS
from tests.helpers_write_spies import WriteLog, own_sessions, recording_http, settle, spy_session

IN_THE_LANE = {"lane": "check_in", "run_id": "0b8e6c1a-5d2f-4e7a-9c3b-1f4a6d8e2b70"}
PROBE = "platform_list_agents"


def _executor(db=None):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(db, uuid4())
    executor._full_autonomy = lambda: False
    return executor


async def _probe(db, workspace_id, params):
    return {"success": True, "reading_only": only_reading()}


# ── the mark ────────────────────────────────────────────────────────────────

def test_reading_only_marks_the_block_and_the_work_it_starts():
    async def seen():
        async def in_a_task():
            return only_reading()

        return only_reading(), await asyncio.create_task(in_a_task()), await asyncio.to_thread(only_reading)

    assert only_reading() is False
    with reading_only():
        assert asyncio.run(seen()) == (True, True, True)
    assert only_reading() is False
    assert asyncio.run(seen()) == (False, False, False)


@pytest.mark.parametrize("context,marked", [
    (IN_THE_LANE, True),
    (None, False),
    ({}, False),
    ({"lane": "heartbeat"}, False),
])
def test_the_lane_runs_each_check_in_call_reading_only(context, marked):
    executor = _executor()
    executor._handlers = {**executor._handlers, PROBE: _probe}

    result = asyncio.run(executor.execute(PROBE, {}, context))

    assert result["reading_only"] is marked
    assert only_reading() is False


def test_the_suggestion_is_the_one_check_in_call_not_reading_only():
    executor = _executor()
    executor._handlers = {**executor._handlers, SUGGEST_NEXT_STEP: _probe}

    result = asyncio.run(executor.execute(SUGGEST_NEXT_STEP, {}, IN_THE_LANE))

    assert result["reading_only"] is False


# ── each read that kept a record, through the executor, in the lane ─────────

@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: seed.action)
def test_each_made_read_action_writes_nothing_through_the_executor_in_a_check_in(seed, monkeypatch, tmp_path):
    session_for = seed.plant(monkeypatch, tmp_path)
    log = WriteLog()

    async def run():
        with recording_http(log), own_sessions(log):
            before = asyncio.all_tasks()
            result = await _executor(session_for(log)).execute(
                seed.action, dict(seed.params), {**IN_THE_LANE, **seed.context})
            await settle(asyncio.all_tasks() - before)
            return result

    result = asyncio.run(run())

    assert log.writes == [], f"{seed.action} wrote in a check-in: {'; '.join(log.writes)}"
    assert result.get("error") != CHECK_IN_REFUSAL
    assert {key: result.get(key) for key in seed.answers} == seed.answers


# ── the records a seed does not reach ───────────────────────────────────────

def _emits(monkeypatch):
    from modules.nl2sql import primitive_heartbeat

    emitted = []
    monkeypatch.setattr(primitive_heartbeat, "emit_primitive_finding", lambda *args, **kwargs: emitted.append(args))
    return primitive_heartbeat, emitted


def test_a_check_in_query_emits_no_nl2sql_heartbeat(monkeypatch):
    primitive_heartbeat, emitted = _emits(monkeypatch)

    with reading_only():
        primitive_heartbeat._emit_nl2sql_primitive("ws-1", success=True, detail="1 rows")
    assert emitted == []

    primitive_heartbeat._emit_nl2sql_primitive("ws-1", success=True, detail="1 rows")
    assert emitted == [("ws-1", "nl2sql", "green", "1 rows")]


def _bare_store():
    from modules.nl2sql.training.example_store import SQLExampleStore

    return SQLExampleStore.__new__(SQLExampleStore)


def test_a_check_in_query_stores_no_training_example(monkeypatch):
    from modules.nl2sql.training.example_store import SQLExampleStore

    opened = []
    monkeypatch.setattr(SQLExampleStore, "_get_db_session", lambda self: opened.append("session"))

    with reading_only():
        stored = asyncio.run(_bare_store().add_example("how many orders", "SELECT 1", "7", "ws-1"))

    assert (stored, opened) == ("", [])


def _example(usage_count):
    return NS(id=1, question="how many orders this month", sql="SELECT count(*) FROM orders", tables_used=["orders"],
              embedding=[1.0, 0.0], is_verified=True, usage_count=usage_count, last_used_at=None)


@pytest.mark.parametrize("rank", [
    lambda store, examples, db: store._keyword_similarity("how many orders", examples, 5, db),
    lambda store, examples, db: store._embedding_similarity([1.0, 0.0], examples, 5, 0.3, db),
])
def test_a_check_in_query_counts_no_use_of_an_example(rank):
    example, log = _example(2), WriteLog()

    with reading_only():
        handed = rank(_bare_store(), [example], spy_session(log))
    assert [one["id"] for one in handed] == [1]
    assert (example.usage_count, example.last_used_at, log.writes) == (2, None, [])

    rank(_bare_store(), [example], spy_session(log))
    assert example.usage_count == 3 and example.last_used_at is not None and log.writes == ["commit"]


def _substrate(monkeypatch):
    from core.observability import substrate_metrics

    inserted = []
    monkeypatch.setattr(substrate_metrics, "_insert_event_sync", lambda *args: inserted.append(args[0]))
    monkeypatch.setattr(substrate_metrics, "fire_retrieval_score", lambda **kwargs: inserted.append("score"))
    return substrate_metrics, inserted


def test_a_check_in_search_records_no_substrate_event(monkeypatch):
    substrate_metrics, inserted = _substrate(monkeypatch)
    event = {"seam": substrate_metrics.SEAM_MEMORY, "workspace_id": "ws-1", "candidates": 1, "latency_ms": 2.0,
             "status": substrate_metrics.STATUS_HIT, "query": "cafe hours"}

    async def search(nowait):
        before = asyncio.all_tasks()
        if nowait:
            substrate_metrics.record_substrate_search_nowait(**event)
        else:
            await substrate_metrics.record_substrate_search(**event)
        await settle(asyncio.all_tasks() - before)

    with reading_only():
        asyncio.run(search(nowait=True))
        asyncio.run(search(nowait=False))
    assert inserted == []

    asyncio.run(search(nowait=True))
    asyncio.run(search(nowait=False))
    assert inserted == ["memory", "score", "memory", "score"]
