"""F321-C against real Postgres (``@integration``): the report rollup's SQL reads
a run's own llm_usage rows, splits them by step, and counts a subscription
session's calls as plan usage. Rows have the shapes night 9b wrote (run
exec-45d8ac862a79; Auto's chat dd0b6649… beside it). Skips cleanly when no
Postgres is reachable (CI runs it).
"""
from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from sqlalchemy import create_engine, text

from core.database.database import get_database_url
from core.models.core import LLMUsage

pytestmark = pytest.mark.integration

RUN = "exec-45d8ac862a79"
CHAT = "chat:dd0b6649-ffd8-4c82-a9e5-c84745c194d1"
STEPS = [{"order": 1, "started_at": "2026-10-04T15:49:36.446524+00:00"},
         {"order": 2, "started_at": "2026-10-04T15:49:48.533323+00:00"}]


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT tier, cache_read_tokens FROM llm_usage LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the F321 rollup suite needs a reachable Postgres with llm_usage: {exc}")
    yield eng
    eng.dispose()


def _row(ws, at, **kw):
    base = dict(workspace_id=ws, model_id="anthropic/claude-sonnet-4", provider="openrouter", tier="aggregator",
                agent_id=342, execution_id=RUN, request_type="recipe", input_tokens=13_075, output_tokens=331,
                total_tokens=13_406, cache_read_tokens=0, cache_write_tokens=0, input_cost=0.0098,
                output_cost=0.00069835, total_cost=0.01049835, is_byok=False, latency_ms=4000, status="success",
                created_at=at)
    base.update(kw)
    return LLMUsage(**base)


@pytest.fixture
def seeded(engine, new_session):
    ws = uuid.uuid4()
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
              {"id": str(ws), "n": "f321-rollup"})
    s.add_all([
        _row(ws, datetime(2026, 10, 4, 15, 49, 48, 493481)),                                     # step 1
        _row(ws, datetime(2026, 10, 4, 15, 51, 26, 674756), input_tokens=21_779, output_tokens=109,
             total_tokens=21_888, total_cost=0.01015935),                                         # step 2
        _row(ws, datetime(2026, 10, 4, 15, 50, 34, 312922), model_id="typesafe/jev-1.13-20260917",
             request_type="decision", input_tokens=2025, output_tokens=585, total_tokens=2610,
             total_cost=0.00008505),                                                              # step 2's routing
        _row(ws, datetime(2026, 10, 4, 15, 49, 28, 965962), model_id="google/gemini-2.5-flash", agent_id=346,
             execution_id=CHAT, request_type="chat", input_tokens=33_096, output_tokens=33,
             total_tokens=33_129, total_cost=0.0100113),                                          # Auto's own turn
        _row(ws, datetime(2026, 10, 3, 9, 7, 37), model_id="claude-sonnet-5-5", provider="claude_code",
             tier="subscription", agent_id=301, execution_id="board_task:1413", request_type="board_task",
             input_tokens=5_344_489, output_tokens=158_244, total_tokens=5_502_733, input_cost=0,
             output_cost=0, total_cost=0),                                                        # a session ticket
    ])
    s.commit()
    s.close()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM llm_usage WHERE workspace_id = CAST(:id AS uuid)"), {"id": str(ws)})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": str(ws)})
    s.commit()
    s.close()


def test_a_runs_rollup_reads_its_own_rows_and_splits_them_by_step(seeded, new_session):
    from services.report_metrics import compute_execution_metrics, step_tokens

    db = new_session()
    metrics = compute_execution_metrics(db, seeded, execution_id=RUN)
    per_step = step_tokens(db, seeded, RUN, STEPS)

    assert (metrics["llm_calls"], metrics["input_tokens"], metrics["output_tokens"], metrics["tokens_used"]) == (
        3, 36_879, 1_025, 37_904)
    assert round(metrics["cost_usd"], 6) == round(0.01049835 + 0.01015935 + 0.00008505, 6)
    assert metrics["model"] == "anthropic/claude-sonnet-4" and metrics["subscription_calls"] == 0
    assert per_step == {1: 13_406, 2: 24_498}


def test_a_session_tickets_rollup_is_plan_usage_not_zero_dollars(seeded, new_session):
    from services.report_metrics import compute_execution_metrics, cost_text

    metrics = compute_execution_metrics(new_session(), seeded, agent_id=301,
                                        started_at=datetime(2026, 10, 3, 9, 0, 1),
                                        completed_at=datetime(2026, 10, 3, 9, 7, 38))

    assert metrics["llm_calls"] == 1 and metrics["subscription_calls"] == 1
    assert cost_text(metrics) == "plan usage (subscription), no dollar figure"
