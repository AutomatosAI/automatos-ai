"""PRD-187 S2 — playbook write-side de-spam (memory J2 / C.2, P2-06).

The old writer double-wrote every run: an unconditional playbook-level row
plus an unconditional per-step row, both ``playbook_summary``, no dedup — so
the daily 402-failing cron memorised the SAME failure twice per run, forever
(~87% of L2 was this chatter). These tests pin the new contract:

1. The same playbook failure written twice yields ONE memory whose recurrence
   count climbs (the count is the signal), not two rows.
2. A first-time / notable success still writes — the gate suppresses only
   repeats, mirroring ``tool_outcome_capture``'s posture.
3. A single run no longer emits both a playbook-level and a per-step record
   for the SAME failure; a step failing with a DIFFERENT error class still
   gets its own record (that's real attribution, not spam).

All pure — records are planned from plain dicts and written through a fake
unified service; no DB, no Qdrant.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.services.playbook_memory_service import (  # noqa: E402
    PlaybookMemoryService,
    plan_execution_records,
    record_recurrence,
    reset_recurrence_registry,
    _signature_hash,
)

WS = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
PB = "tpl-daily-cron"


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_recurrence_registry()
    yield
    reset_recurrence_registry()


def _make_service() -> tuple[PlaybookMemoryService, MagicMock]:
    svc = PlaybookMemoryService.__new__(PlaybookMemoryService)
    unified = MagicMock()
    unified.store_long_term_messages = AsyncMock(return_value={"success": True})
    unified.store_short_term = AsyncMock(return_value="l2-row")
    svc._unified = unified
    return svc, unified


def _failed_run(execution_id: str):
    return plan_execution_records(
        workspace_id=WS, playbook_id=PB, execution_id=execution_id,
        status="failed", error_message="Payment Required: 402 from provider",
        learnings=None, quality_data=None,
        step_results=[{"status": "failed", "error": "402 payment required", "agent_id": 7}],
        step_agent_ids=[7],
    )


async def _write(svc, records, execution_id):
    return await svc.write_records(
        records, workspace_id=WS, playbook_id=PB, execution_id=execution_id,
    )


# ---------------------------------------------------------------------------
# 1. Same failure twice → ONE memory with a recurrence count
# ---------------------------------------------------------------------------

def test_playbook_failure_deduped():
    svc, unified = _make_service()

    first = asyncio.run(_write(svc, _failed_run("exec-day1"), "exec-day1"))
    second = asyncio.run(_write(svc, _failed_run("exec-day2"), "exec-day2"))

    # Day 1: exactly one L3 write (the run-level failure; the step failure is
    # the same error class, so it never became a second record).
    assert first["stored_memories"] == 1
    assert unified.store_long_term_messages.await_count == 1

    # Day 2: nothing new is written — the repeat is suppressed WITH its count.
    assert second["stored_memories"] == 0
    assert unified.store_long_term_messages.await_count == 1
    assert second["suppressed"] == [
        {"scope": "playbook", "recurrence": 2, "error_class": "schema"},
    ]


def test_recurrence_counter_climbs():
    h = _signature_hash(WS, PB, "fail:rate_limit")
    assert record_recurrence(h) == 1
    assert record_recurrence(h) == 2
    assert record_recurrence(h) == 3


# ---------------------------------------------------------------------------
# 2. First-time / notable success still records
# ---------------------------------------------------------------------------

def test_playbook_notable_success_still_recorded():
    svc, unified = _make_service()
    records = plan_execution_records(
        workspace_id=WS, playbook_id=PB, execution_id="exec-ok",
        status="completed", error_message=None,
        learnings={"performance_metrics": {"total_duration_ms": 8000, "success_rate": 1.0}},
        quality_data={"quality_score": 92, "grade": "A"},
        step_results=[{"status": "completed", "agent_id": 7}],
        step_agent_ids=[7],
    )
    result = asyncio.run(_write(svc, records, "exec-ok"))

    assert result["stored_memories"] == 1
    fact = unified.store_long_term_messages.await_args.kwargs["messages"][1]["content"]
    assert "quality score was 92" in fact

    # An identical success repeating is a trivial repeat — suppressed.
    repeat = asyncio.run(_write(svc, records, "exec-ok-2"))
    assert repeat["stored_memories"] == 0
    assert repeat["suppressed"][0]["recurrence"] == 2


# ---------------------------------------------------------------------------
# 3. No unconditional playbook-level + per-step double write
# ---------------------------------------------------------------------------

def test_playbook_write_no_unconditional_double_write():
    # Old behavior: 1 playbook record + 1 per-step record for the SAME 402.
    records = _failed_run("exec-x")
    assert [r["scope"] for r in records] == ["playbook"], (
        "a step failing with the run's own error class must fold into the "
        "run-level record, not double-write"
    )


def test_distinct_step_failure_class_still_attributed():
    records = plan_execution_records(
        workspace_id=WS, playbook_id=PB, execution_id="exec-y",
        status="failed", error_message="Payment Required: 402 from provider",
        learnings=None, quality_data=None,
        step_results=[
            {"status": "failed", "error": "402 payment required", "agent_id": 7},
            {"status": "failed", "error": "rate limit exceeded (429)", "agent_id": 9},
        ],
        step_agent_ids=[7, 9],
    )
    scopes = [(r["scope"], r["metadata"].get("error_class")) for r in records]
    assert scopes == [("playbook", "schema"), ("step", "rate_limit")]
    assert records[1]["agent_id"] == 9


def test_success_steps_never_emit_step_records():
    records = plan_execution_records(
        workspace_id=WS, playbook_id=PB, execution_id="exec-z",
        status="completed", error_message=None,
        learnings=None, quality_data=None,
        step_results=[{"status": "completed", "agent_id": 7}, {"status": "completed", "agent_id": 9}],
        step_agent_ids=[7, 9],
    )
    assert [r["scope"] for r in records] == ["playbook"]


# ---------------------------------------------------------------------------
# L2 side: one row per first-occurrence record, typed + counted
# ---------------------------------------------------------------------------

def test_l2_row_carries_type_and_recurrence():
    async def run():
        svc, unified = _make_service()
        await _write(svc, _failed_run("exec-l2"), "exec-l2")
        # let the fire-and-forget L2 task run
        await asyncio.sleep(0)
        return unified

    unified = asyncio.run(run())
    assert unified.store_short_term.await_count == 1
    kwargs = unified.store_short_term.await_args.kwargs
    assert kwargs["content_type"] == "playbook_summary"
    assert kwargs["metadata"]["recurrence"] == 1
    assert kwargs["importance"] == 0.6
