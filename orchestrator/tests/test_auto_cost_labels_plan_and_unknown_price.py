"""Auto's wishlist (9 Oct): two prices that read wrong.

1. ``platform_get_cost_breakdown`` showed a Claude Code session's calls at $0.00 and
   Auto called them free. They are booked at $0 because the subscription plan pays for
   them; each row now says how it was billed (plan, metered, or both).
2. Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 came into the catalogue with no price
   (the Anthropic Models API publishes none and OpenRouter has no twin yet), and the
   model dropdown called them free. An unpriced route is now "price unknown", never
   free, and Opus 5.5 gets its list price ($4 / $20 per million tokens) from the
   platform's price map: new rows on insert, unpriced rows on the next sync, and
   calls through the usage tracker's estimate.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from modules.tools.discovery import handlers_analytics as ha

OPUS_55 = "claude-opus-5-5-20991231"   # dated, so the test never meets a seeded row; the price map is undated
SONNET_55 = "claude-sonnet-5-5-20991231"


# --- 1. the cost breakdown says how each row was billed ---------------------------------


@pytest.mark.parametrize("requests, plan_requests, billing", [
    (4, 0, ha.BILLING_METERED), (4, 4, ha.BILLING_PLAN), (4, 1, ha.BILLING_MIXED),
])
def test_a_row_is_billed_by_plan_metered_or_both(requests, plan_requests, billing):
    assert ha._billing(requests, plan_requests) == billing


@pytest.fixture
def usage(db_session, seed_workspace):
    from core.models.core import LLMUsage

    ws = UUID(seed_workspace())

    def call(model, tier, provider, cost, agent_id):
        return LLMUsage(workspace_id=ws, model_id=model, provider=provider, tier=tier, agent_id=agent_id,
                        input_tokens=1000, output_tokens=100, total_tokens=1100, input_cost=cost * 0.8,
                        output_cost=cost * 0.2, total_cost=cost, status="success", created_at=datetime.utcnow())

    db_session.add_all([
        call("claude-sonnet-5", "subscription", "claude_code", 0.0, 301),
        call("claude-sonnet-5", "subscription", "claude_code", 0.0, 301),
        call("google/gemini-2.5-flash", "aggregator", "openrouter", 0.01, 346),
        call("google/gemini-2.5-flash", "subscription", "claude_code", 0.0, 301),
    ])
    db_session.flush()
    return NS(db=db_session, ws=ws)


def _breakdown(usage, **params):
    return asyncio.run(ha.get_cost_breakdown(usage.db, usage.ws, {"days": 1, **params}))


def test_a_subscription_sessions_calls_read_as_plan_not_free(usage):
    out = _breakdown(usage)
    rows = {r["model"]: r for r in out["breakdown"]}
    assert rows["claude-sonnet-5"]["total_cost"] == 0.0
    assert (rows["claude-sonnet-5"]["billing"], rows["claude-sonnet-5"]["plan_requests"]) == ("plan", 2)
    assert (rows["google/gemini-2.5-flash"]["billing"], rows["google/gemini-2.5-flash"]["plan_requests"]) == (
        "plan and metered", 1)
    assert out["plan_requests"] == 3 and out["total_cost"] == pytest.approx(0.01)
    assert "Never call them free" in out["note"]


def test_by_agent_the_session_agent_reads_as_plan(usage):
    rows = {r["agent"]: r for r in _breakdown(usage, group_by="agent")["breakdown"]}
    assert rows["301"]["billing"] == "plan" and rows["346"]["billing"] == "metered"


def test_a_workspace_with_no_plan_calls_gets_no_plan_note(db_session, seed_workspace):
    out = asyncio.run(ha.get_cost_breakdown(db_session, UUID(seed_workspace()), {"days": 1}))
    assert out["plan_requests"] == 0 and "note" not in out and out["breakdown"] == []


# --- 2. an unpriced route is "price unknown", never free --------------------------------


def _route(**kw):
    base = dict(
        id=1, provider="anthropic", serving_provider="anthropic", model_id=OPUS_55,
        display_name="Claude Opus 5.5", model_family="claude", description="d", context_window=1_000_000,
        max_output_tokens=128_000, input_cost_per_1k_tokens=None, output_cost_per_1k_tokens=None,
        capabilities={}, recommended_for=[], supports_functions=True, supports_vision=True,
        supports_streaming=True, status="active", sourcing="direct", category=None, tags=["anthropic"],
        is_featured=False, is_default=False, requires_plan=None, install_count=0,
    )
    return NS(**{**base, **kw})


def test_an_unpriced_route_is_not_free():
    from api.llm_marketplace import PRICE_TIER_UNKNOWN, _model_to_out

    out = _model_to_out(_route())
    assert out.is_free is False and out.price_known is False and out.price_tier == PRICE_TIER_UNKNOWN


def test_an_explicit_zero_price_is_still_free_and_a_priced_route_is_not():
    from api.llm_marketplace import _model_to_out

    free = _model_to_out(_route(input_cost_per_1k_tokens=0.0, output_cost_per_1k_tokens=0.0))
    assert free.is_free is True and free.price_known is True and free.price_tier == "free"
    priced = _model_to_out(_route(input_cost_per_1k_tokens=0.004, output_cost_per_1k_tokens=0.02))
    assert priced.is_free is False and priced.price_known is True and priced.price_tier == "premium"


def test_a_provider_that_bills_nothing_is_free():
    from api.llm_marketplace import _model_to_out

    nvidia = _model_to_out(_route(serving_provider="nvidia", model_id="moonshotai/kimi-k3",
                                  input_cost_per_1k_tokens=0, output_cost_per_1k_tokens=0))
    assert nvidia.is_free is True


# --- 3. Opus 5.5 gets its list price ----------------------------------------------------


@pytest.mark.parametrize("model_id, price", [
    ("claude-opus-5-5", (0.004, 0.020)),
    (OPUS_55, (0.004, 0.020)),           # a dated id reads its undated entry
    ("claude-opus-4-8", None),            # "claude-opus-4" is in the map, but only an exact entry counts
    (SONNET_55, None),                    # no published price
])
def test_the_list_price_is_the_maps_exact_entry_only(model_id, price):
    from core.services.anthropic_catalog_sync import list_price

    assert list_price(model_id) == price


def test_a_new_row_with_no_openrouter_twin_takes_its_list_price():
    from core.services.anthropic_catalog_sync import new_row_defaults

    opus = new_row_defaults({"id": OPUS_55, "display_name": "Claude Opus 5.5"}, None)
    assert (opus["input_cost_per_1k_tokens"], opus["output_cost_per_1k_tokens"]) == (0.004, 0.020)
    assert opus["pricing_updated_at"] is not None
    sonnet = new_row_defaults({"id": SONNET_55, "display_name": "Claude Sonnet 5.5"}, None)
    assert sonnet["input_cost_per_1k_tokens"] is None and sonnet["output_cost_per_1k_tokens"] is None


def _insert_route(db, model_id, price_in, price_out):
    db.execute(text(
        "INSERT INTO llm_models (provider, serving_provider, model_id, display_name, context_window, "
        "max_output_tokens, input_cost_per_1k_tokens, output_cost_per_1k_tokens, status) "
        "VALUES ('anthropic', 'anthropic', :m, :m, 1000000, 128000, :i, :o, 'active')"),
        {"m": model_id, "i": price_in, "o": price_out})


def _price(db, model_id):
    return db.execute(text(
        "SELECT input_cost_per_1k_tokens AS i, output_cost_per_1k_tokens AS o FROM llm_models "
        "WHERE serving_provider = 'anthropic' AND model_id = :m"), {"m": model_id}).mappings().first()


def test_an_unpriced_existing_row_is_priced_on_the_next_sync_and_a_set_price_is_kept(db_session):
    from core.services.anthropic_catalog_sync import price_unpriced_row

    _insert_route(db_session, OPUS_55, None, None)
    _insert_route(db_session, SONNET_55, None, None)
    assert price_unpriced_row(db_session, OPUS_55) == 1
    assert price_unpriced_row(db_session, SONNET_55) == 0
    assert dict(_price(db_session, OPUS_55)) == {"i": 0.004, "o": 0.020}
    assert dict(_price(db_session, SONNET_55)) == {"i": None, "o": None}

    db_session.execute(text("UPDATE llm_models SET input_cost_per_1k_tokens = 0, output_cost_per_1k_tokens = 0 "
                            "WHERE serving_provider = 'anthropic' AND model_id = :m"), {"m": OPUS_55})
    assert price_unpriced_row(db_session, OPUS_55) == 0       # an explicit price, 0 included, stays
    assert dict(_price(db_session, OPUS_55)) == {"i": 0, "o": 0}


@pytest.fixture
def catalog(db_session):
    """The real llm_models; an empty OpenRouter cache as a temp table (#829's fixture)."""
    from sqlalchemy.schema import CreateTable

    from core.models.openrouter_cache import OpenRouterModelCache

    ddl = str(CreateTable(OpenRouterModelCache.__table__).compile(dialect=db_session.bind.dialect))
    db_session.execute(text(ddl.replace("CREATE TABLE", "CREATE TEMP TABLE", 1)))
    return db_session


def test_a_call_on_an_unpriced_opus_55_route_is_priced_at_its_list_price(catalog):
    from core.llm.usage_tracker import resolve_price

    _insert_route(catalog, OPUS_55, None, None)
    price = resolve_price(catalog, OPUS_55, "anthropic")
    assert price["source"] == "estimate"
    assert (price["input_per_1k"], price["output_per_1k"]) == pytest.approx((0.004, 0.020))
