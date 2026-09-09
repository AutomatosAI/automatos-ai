"""Analytics cost tracking (2026-09-09) — the LLM analytics endpoints against
a real Postgres: routes, providers, cache tokens, executions, and the
activity-sync rows that must never be summed with the per-call rows.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database.database import get_database_url  # noqa: E402
from core.models.core import LLMUsage  # noqa: E402
from api import llm_analytics as api  # noqa: E402
from api import documents as docs_api  # noqa: E402
from core.models.core import Document  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT cache_read_tokens FROM llm_usage LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"analytics suite needs Postgres with prd240_llm_usage_cache_tokens applied: {exc}")
    yield eng
    eng.dispose()


def _row(ws, **kw):
    base = dict(
        workspace_id=ws, model_id="moonshotai/kimi-k3", provider="openrouter", tier="aggregator",
        agent_id=57, execution_id="chat:c1", request_type="chat", input_tokens=1000, output_tokens=100,
        total_tokens=1100, cache_read_tokens=0, cache_write_tokens=0, input_cost=0.003, output_cost=0.0015,
        total_cost=0.0045, is_byok=True, latency_ms=900, status="success", created_at=datetime.utcnow(),
    )
    base.update(kw)
    return LLMUsage(**base)


@pytest.fixture
def seeded(engine, new_session):
    ws = uuid.uuid4()
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
              {"id": str(ws), "n": "analytics-realdb"})
    s.add_all([
        _row(ws),                                                                          # paid kimi via OpenRouter
        _row(ws, provider="nvidia", tier="hosted_open", input_cost=0, output_cost=0, total_cost=0,
             execution_id="board_task:9", request_type="board_task", agent_id=57),         # free kimi via NVIDIA
        _row(ws, model_id="claude-fable-5", provider="claude_code", tier="subscription", agent_id=15,
             execution_id="board_task:97", request_type="board_task", input_tokens=236_734, output_tokens=666,
             total_tokens=237_400, cache_read_tokens=117_716, cache_write_tokens=119_012,
             input_cost=0, output_cost=0, total_cost=0),                                    # Claude Code session
        _row(ws, model_id="google/gemini-2.5-flash", execution_id="mission:m1", request_type="mission",
             agent_id=1, total_cost=0.02, input_cost=0.015, output_cost=0.005, status="error"),
        _row(ws, request_type="activity_sync", execution_id="openrouter_sync_x", total_cost=99.0,
             input_cost=0, output_cost=0, agent_id=None),                                   # the reconciliation copy
        _row(ws, created_at=datetime.utcnow() - timedelta(days=40), total_cost=5.0),        # outside the period
    ])
    s.commit()
    s.close()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM llm_usage WHERE workspace_id = CAST(:id AS uuid)"), {"id": str(ws)})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": str(ws)})
    s.commit()
    s.close()


def _ctx(ws):
    return SimpleNamespace(workspace_id=ws)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_usage_by_route_separates_free_paid_and_subscription(seeded, new_session):
    db = new_session()
    try:
        rows = _run(api.get_usage(period="30d", group_by="route", ctx=_ctx(seeded), db=db))
    finally:
        db.close()
    by_key = {r.key: r for r in rows}
    assert "moonshotai/kimi-k3@openrouter" in by_key and "moonshotai/kimi-k3@nvidia" in by_key
    assert by_key["moonshotai/kimi-k3@nvidia"].billing == "free" and by_key["moonshotai/kimi-k3@nvidia"].total_cost == 0
    sub = by_key["claude-fable-5@claude_code"]
    assert sub.billing == "subscription" and sub.provider_label == "Claude Code" and sub.cache_read_tokens == 117_716
    assert "moonshotai/kimi-k3@openrouter" in by_key and all(r.request_count for r in rows)
    assert not any(r.key.startswith("openrouter_sync") for r in rows)
    total = sum(r.total_cost for r in rows)
    assert total == pytest.approx(0.0045 + 0.02)  # never the 99.0 sync copy, never the 40-day-old row


def test_summary_carries_providers_cache_and_no_sync_dollars(seeded, new_session):
    db = new_session()
    try:
        summary = _run(api.get_summary(period="30d", ctx=_ctx(seeded), db=db))
    finally:
        db.close()
    assert summary.total_requests == 4 and summary.total_cost == pytest.approx(0.0245)
    assert summary.cache_read_tokens == 117_716 and summary.error_rate == 0.25
    providers = {p.provider: p for p in summary.by_provider}
    assert providers["claude_code"].billing == "subscription" and providers["nvidia"].billing == "free"
    assert providers["openrouter"].request_count == 2 and providers["openrouter"].error_count == 1
    assert summary.top_models[0]["key"] == "google/gemini-2.5-flash@openrouter"


def test_usage_by_execution_and_agent_join_back_to_the_spender(seeded, new_session):
    db = new_session()
    try:
        by_exec = {r.key: r for r in _run(api.get_usage(period="30d", group_by="execution", ctx=_ctx(seeded), db=db))}
        by_agent = {r.key: r for r in _run(api.get_usage(period="30d", group_by="agent", ctx=_ctx(seeded), db=db))}
    finally:
        db.close()
    assert by_exec["mission:m1"].total_cost == pytest.approx(0.02) and by_exec["board_task:97"].total_tokens == 237_400
    assert by_agent["15"].total_cost == 0 and by_agent["15"].total_tokens == 237_400
    assert by_agent["57"].request_count == 2
    # the route each agent actually used most this period rides on the group
    assert by_agent["15"].model_id == "claude-fable-5" and by_agent["15"].provider == "claude_code"
    assert by_agent["15"].billing == "subscription" and by_agent["57"].provider in ("openrouter", "nvidia")


def test_daily_series_is_keyed_by_route_with_facts(seeded, new_session):
    db = new_session()
    try:
        out = _run(api.get_daily_costs_by_model(period="30d", ctx=_ctx(seeded), db=db))
    finally:
        db.close()
    assert set(out["models"]) == {r["key"] for r in out["routes"]}
    facts = {r["key"]: r for r in out["routes"]}
    assert facts["claude-fable-5@claude_code"]["billing"] == "subscription"
    assert out["series"] and all(k in out["series"][-1] for k in out["models"])


def test_projections_and_comparison_are_route_aware(seeded, new_session):
    db = new_session()
    try:
        proj = _run(api.get_cost_projections(period="30d", ctx=_ctx(seeded), db=db))
        cmp_ = _run(api.get_model_comparison(model_ids="moonshotai/kimi-k3@nvidia,moonshotai/kimi-k3",
                                             period="30d", ctx=_ctx(seeded), db=db))
    finally:
        db.close()
    assert proj.current_period_cost == pytest.approx(0.0245)
    keys = {p.key for p in proj.projected_by_model}
    assert "moonshotai/kimi-k3@nvidia" in keys and any(p.billing == "subscription" for p in proj.projected_by_model)
    assert {p.provider for p in proj.projected_by_provider} >= {"openrouter", "nvidia", "claude_code"}
    assert cmp_[0].provider == "nvidia" and cmp_[0].billing == "free" and cmp_[0].total_cost == 0
    assert cmp_[1].provider in ("openrouter", "nvidia")


# ── Documents: the knowledge base is what the user uploaded, never agent outputs ──

@pytest.fixture
def seeded_documents(engine, new_session):
    ws = uuid.uuid4()
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
              {"id": str(ws), "n": "analytics-docs-realdb"})
    s.add_all([
        Document(filename="Business_Plan.pdf", file_type="pdf", file_size=4000, status="processed",
                 chunk_count=12, workspace_id=ws, source_type=None, tags=[]),
        Document(filename="notes.txt", file_type="text", file_size=100, status="processed",
                 chunk_count=1, workspace_id=ws, source_type="upload", tags=[]),
        Document(filename="2026-09-09_081500_heartbeat.md", file_type="md", file_size=330, status="processed",
                 chunk_count=1, workspace_id=ws, source_type="agent_output", tags=["agent_output", "report"]),
        Document(filename="mission-output.md", file_type="md", file_size=900, status="failed",
                 chunk_count=0, workspace_id=ws, source_type="agent_output", tags=["agent_output"]),
    ])
    s.commit()
    s.close()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM documents WHERE workspace_id = CAST(:id AS uuid)"), {"id": str(ws)})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": str(ws)})
    s.commit()
    s.close()


def test_document_analytics_counts_only_the_users_rag_documents(seeded_documents, new_session):
    db = new_session()
    try:
        out = _run(docs_api.get_document_analytics(ctx=_ctx(seeded_documents), db=db))
    finally:
        db.close()
    assert out["total_documents"] == 2 and out["agent_outputs"] == 2
    assert out["total_storage_bytes"] == 4100 and out["total_chunks"] == 13
    assert out["status_distribution"] == {"processed": 2}          # the failed agent output is not a failed upload
    assert set(out["file_type_distribution"]) == {"pdf", "text"}


def test_document_list_can_leave_agent_outputs_out(seeded_documents, new_session):
    db = new_session()
    try:
        everything = _run(docs_api.list_documents(ctx=_ctx(seeded_documents), skip=0, limit=100, team=None, db=db))
        knowledge = _run(docs_api.list_documents(ctx=_ctx(seeded_documents), skip=0, limit=100, team=None,
                                                 exclude_source_type="agent_output", db=db))
    finally:
        db.close()
    assert len(everything) == 4 and len(knowledge) == 2
    assert {d.filename for d in knowledge} == {"Business_Plan.pdf", "notes.txt"}
    assert all(d.source_type != "agent_output" for d in knowledge)
