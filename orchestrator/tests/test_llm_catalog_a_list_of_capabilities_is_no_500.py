"""23 Oct 2026 — one malformed row took the hosted LLM catalogue down.

A production ``llm_models`` row held ``capabilities = ["chat"]``: a list of names,
written outside the seeds and syncs. ``LLMModelOut.capabilities`` is a dict of
ratings, so building that one card raised a ValidationError and
``GET /api/marketplace/llm/catalog`` answered 500. The Marketplace's LLMs tab then
showed "0 models" and only its fallback OpenRouter and NVIDIA tabs, though the table
held 550 active routes. The analytics comparison builds the same field the same way.

What is pinned: only a dict is read as ratings, anything else reads as {} (a list
of names carries no ratings), and the catalogue and the comparison both answer with
the row in them.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import text

MODEL_ID = "claude-capabilities-list-20991231"   # dated, so the test never meets a seeded row


def _route(**kw):
    base = dict(
        id=1, provider="anthropic", serving_provider="anthropic", model_id=MODEL_ID,
        display_name="Claude Capabilities List", model_family="claude", description="d",
        context_window=200_000, max_output_tokens=64_000, input_cost_per_1k_tokens=0.001,
        output_cost_per_1k_tokens=0.005, capabilities=["chat"], recommended_for=[],
        supports_functions=True, supports_vision=True, supports_streaming=True, status="active",
        sourcing="direct", category=None, tags=[], is_featured=False, is_default=False,
        requires_plan=None, install_count=0,
    )
    return NS(**{**base, **kw})


@pytest.mark.parametrize("stored, shown", [
    (["chat"], {}),
    ("chat", {}),
    (None, {}),
    ({}, {}),
    ({"reasoning": "excellent", "coding": "good"}, {"reasoning": "excellent", "coding": "good"}),
])
def test_a_card_reads_ratings_only_from_a_dict(stored, shown):
    from api.llm_marketplace import _model_to_out

    assert _model_to_out(_route(capabilities=stored)).capabilities == shown


@pytest.fixture
def catalog(db_session):
    """The real llm_models with one list-shaped row; the sync-job table as a temp table (#829's fixture)."""
    from sqlalchemy.schema import CreateTable

    from core.models.core import LLMModel
    from core.models.openrouter_cache import OpenRouterSyncJob

    ddl = str(CreateTable(OpenRouterSyncJob.__table__).compile(dialect=db_session.bind.dialect))
    db_session.execute(text(ddl.replace("CREATE TABLE", "CREATE TEMP TABLE", 1)))
    db_session.add(LLMModel(
        provider="anthropic", serving_provider="anthropic", model_id=MODEL_ID,
        display_name="Claude Capabilities List", context_window=200_000, max_output_tokens=64_000,
        capabilities=["chat"], status="active",
    ))
    db_session.flush()
    return db_session


def test_the_catalogue_answers_with_a_row_whose_capabilities_are_a_list(catalog):
    from api.llm_marketplace import browse_catalog

    out = asyncio.run(browse_catalog(
        provider=None, vendor=None, category=None, price_tier=None, min_context=None, max_cost=None,
        capability=None, supports_tools=None, supports_vision=None, search=MODEL_ID,
        sort_by="popularity", limit=500, offset=0, ctx=NS(workspace_id=uuid4()), db=catalog,
    ))

    assert [(m.model_id, m.capabilities) for m in out.models] == [(MODEL_ID, {})]
    assert out.providers["anthropic"] >= 1
    assert "anthropic" in out.syncable


def test_the_comparison_answers_with_a_row_whose_capabilities_are_a_list(catalog):
    from api.llm_analytics import get_model_comparison

    out = asyncio.run(get_model_comparison(
        model_ids=f"{MODEL_ID}@anthropic", period="30d", ctx=NS(workspace_id=uuid4()), db=catalog,
    ))

    assert [(c.model_id, c.capabilities) for c in out] == [(MODEL_ID, {})]
