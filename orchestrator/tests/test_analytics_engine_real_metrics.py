"""PRD-142 Wave 0 US-003 — analytics_engine must report real numbers, not placeholders.

These tests bypass ``AnalyticsEngine.__init__`` (which opens a DB session + Redis)
via ``__new__`` and inject a mock session, so they exercise the metric methods in
isolation without external services.
"""
from unittest.mock import MagicMock

import pytest

from core.services.analytics_engine import AnalyticsEngine


def _engine_with_db(db) -> AnalyticsEngine:
    """Construct an AnalyticsEngine without running __init__ (no DB/Redis)."""
    engine = AnalyticsEngine.__new__(AnalyticsEngine)
    engine.db = db
    return engine


@pytest.mark.asyncio
async def test_agent_metrics_no_placeholders():
    """_get_agent_metrics returns only real, DB-derived counts — no fake fields."""
    db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.count.return_value = 4
    db.query.return_value = q

    metrics = await _engine_with_db(db)._get_agent_metrics()

    # Real values survive
    assert metrics["activeAgents"] == 4
    assert metrics["totalAgents"] == 4

    # The PRD-142 §Wave0 fake placeholders are gone
    assert "successRate" not in metrics
    assert "avgExecutionTime" not in metrics
    assert "totalTokensUsed" not in metrics
    assert "recentExecutions" not in metrics

    # Belt-and-suspenders: the literal fakes never appear as a value
    assert 85.0 not in metrics.values()
    assert 2.5 not in metrics.values()


@pytest.mark.asyncio
async def test_success_rate_matches_orchestration_runs():
    """Mission success rate is computed from orchestration data, never hardcoded.

    With 0 legacy workflow executions and 10 orchestration tasks of which 8 are
    verified, the combined success rate must be 80.0 — derived, not the old 85.0.
    """
    # (bare .count(), filtered .filter()....count()) per model
    specs = {
        "Workflow": (0, 0),
        "WorkflowExecution": (0, 0),
        "OrchestrationRun": (0, 0),
        "OrchestrationTask": (10, 8),
    }

    def query_side_effect(model):
        total, subset = specs.get(getattr(model, "__name__", ""), (0, 0))
        q = MagicMock()
        filtered = MagicMock()
        filtered.filter.return_value = filtered
        filtered.count.return_value = subset
        q.filter.return_value = filtered
        q.count.return_value = total
        return q

    db = MagicMock()
    db.query.side_effect = query_side_effect

    metrics = await _engine_with_db(db)._get_workflow_metrics()

    assert metrics["successRate"] == 80.0
    assert metrics["successRate"] != 85.0
