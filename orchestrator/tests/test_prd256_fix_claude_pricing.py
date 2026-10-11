"""P256-FIX-T2 (F399): the Claude 5.5 models are priced at list, and a call's cost
uses the model's own cache-read multiplier and prompt-length tier.

Night 12's benchmark found claude-opus-5-5, claude-sonnet-5-5 and claude-haiku-5-5
in ``llm_models`` with no price (the pickers said "free"), the cost audit pricing
Opus 5.5 at Opus 5's $5/$25 (list $4/$20), every Claude cache read at 0.1x input
(list: Fable 5.1 0.025x, Opus 5.5 and Sonnet 5.5 0.05x), and Haiku 5.5's second
tier (a prompt over 100k tokens, cache included) not modelled. One table,
``core.llm.list_prices``, holds all of it. Source: platform.claude.com pricing,
read 9-10 Oct 2026.

What is pinned:

- the Fable 5.1 call 89,764 in (55,122 cache read) / 1,120 out costs $0.416 at
  list (recorded $0.458 before), from a priced route row and from the table;
- the Opus 5.5 call 16,959 in (16,733 cache write) / 18 out costs $0.0849;
- a Haiku 5.5 call of 120k prompt tokens pays the second tier, cache included;
  100,000 exactly is still the first;
- the cache-read multiplier is the model's on Anthropic, the provider's for any
  other model, and none on a route with no cache discount;
- the audit estimate prices the 5.5 models at list;
- the three 5.5 rows are not free: the seed prices them (insert if absent, fill
  a NULL price only), and an unpriced row shows the list price, never "free".
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm.list_prices import LIST_PRICES, cache_read_multiplier, list_price, list_rates  # noqa: E402
from core.llm.manager import estimate_cost_usd  # noqa: E402
from core.llm.usage_tracker import price_call, resolve_price  # noqa: E402

FABLE_5_1 = "claude-fable-5-1"
OPUS_5_5 = "claude-opus-5-5"
SONNET_5_5 = "claude-sonnet-5-5"
HAIKU_5_5 = "claude-haiku-5-5"


class _Rows:
    """A session whose every query answers ``row`` (None: the registry and the cache know nothing)."""

    def __init__(self, row=None):
        self.row = row

    def query(self, *_a, **_k):
        return self

    def filter(self, *_a, **_k):
        return self

    def first(self):
        return self.row


def _cost(model_id, provider="anthropic", row=None, **tokens):
    price = resolve_price(_Rows(row), model_id, provider)
    return sum(price_call(price, provider=provider, **tokens)), price


# ── the benchmark's two calls ──────────────────────────────────────────────

FABLE_CALL = dict(input_tokens=89_764, cache_read_tokens=55_122, output_tokens=1_120)


def test_the_fable_5_1_call_costs_list_price_from_a_priced_route_row():
    row = NS(input_cost_per_1k_tokens=0.010, output_cost_per_1k_tokens=0.050, sourcing="direct")
    total, price = _cost(FABLE_5_1, row=row, **FABLE_CALL)
    assert price["source"] == "route" and price["model_id"] == FABLE_5_1
    assert total == pytest.approx(0.416, abs=0.001)           # was 0.458 at 0.1x
    assert total != pytest.approx(0.458, abs=0.001)


def test_the_fable_5_1_call_costs_list_price_from_the_table_on_an_unknown_route():
    total, price = _cost(FABLE_5_1, **FABLE_CALL)
    assert price["source"] == "estimate"
    assert (price["input_per_1k"], price["output_per_1k"]) == (0.010, 0.050)
    assert total == pytest.approx(0.416, abs=0.001)


def test_the_opus_5_5_call_with_a_cache_write_costs_list_price():
    total, price = _cost(OPUS_5_5, input_tokens=16_959, cache_write_tokens=16_733, output_tokens=18)
    assert (price["input_per_1k"], price["output_per_1k"]) == (0.004, 0.020)   # not Opus 5's 0.005/0.025
    # 226 fresh + 16,733 × 1.25 written = 21,142.25 × $4/M, + 18 × $20/M
    assert total == pytest.approx(0.0849, abs=0.0001)


# ── Haiku 5.5's second tier ────────────────────────────────────────────────

def test_a_haiku_5_5_prompt_over_100k_pays_the_second_tier():
    total, _ = _cost(HAIKU_5_5, input_tokens=120_000, output_tokens=1_000)
    assert total == pytest.approx(120_000 * 0.50 / 1e6 + 1_000 * 2.50 / 1e6)


def test_a_haiku_5_5_prompt_of_exactly_100k_stays_on_the_first_tier():
    total, _ = _cost(HAIKU_5_5, input_tokens=100_000, output_tokens=1_000)
    assert total == pytest.approx(100_000 * 0.10 / 1e6 + 1_000 * 0.50 / 1e6)


def test_cache_reads_count_toward_the_haiku_5_5_threshold():
    total, _ = _cost(HAIKU_5_5, input_tokens=120_000, cache_read_tokens=100_000, output_tokens=0)
    # 20k fresh + 100k read at 0.1x, all at the second tier's $0.50/M
    assert total == pytest.approx((20_000 + 100_000 * 0.1) * 0.50 / 1e6)


def test_the_tier_scales_a_routes_own_price_so_a_free_route_stays_free():
    free = {"input_per_1k": 0.0, "output_per_1k": 0.0, "multiplier": 1.0, "model_id": HAIKU_5_5}
    assert price_call(free, provider="anthropic", input_tokens=150_000, output_tokens=500) == (0.0, 0.0)


def test_the_audit_estimate_uses_the_tier_too():
    assert estimate_cost_usd(HAIKU_5_5, 120_000, 1_000) == pytest.approx(0.0625)
    assert estimate_cost_usd(HAIKU_5_5, 50_000, 1_000) == pytest.approx(0.0055)


# ── one table: the cache-read multiplier per model ─────────────────────────

@pytest.mark.parametrize("model_id, multiplier", [
    (FABLE_5_1, 0.025),
    (OPUS_5_5, 0.05),
    (SONNET_5_5, 0.05),
    ("anthropic/claude-sonnet-5.5", 0.05),          # OpenRouter's spelling of the same model
    (HAIKU_5_5, 0.10),
    ("claude-opus-5", 0.10),
    ("claude-sonnet-4-5", 0.10),                    # not in the table: the provider's
])
def test_a_cache_read_is_priced_at_the_models_multiplier(model_id, multiplier):
    price = {"input_per_1k": 1.0, "output_per_1k": 0.0, "multiplier": 1.0, "model_id": model_id}
    cost_in, _ = price_call(price, provider="anthropic", input_tokens=1_000, cache_read_tokens=1_000, output_tokens=0)
    assert cost_in == pytest.approx(multiplier)


def test_other_providers_keep_their_own_cache_rule():
    openai = {"input_per_1k": 1.0, "output_per_1k": 0.0, "multiplier": 1.0, "model_id": "gpt-4o"}
    assert price_call(openai, provider="openai", input_tokens=1_000, cache_read_tokens=1_000, output_tokens=0)[0] \
        == pytest.approx(0.5)
    # a route with no cache discount on record is billed the full input price, whatever the model
    claude = dict(openai, model_id=FABLE_5_1)
    assert price_call(claude, provider="openrouter", input_tokens=1_000, cache_read_tokens=1_000, output_tokens=0)[0] \
        == pytest.approx(1.0)


def test_a_cache_write_is_1_25x_on_every_claude_model():
    for model_id in (FABLE_5_1, OPUS_5_5, SONNET_5_5, "claude-haiku-4-5"):
        price = {"input_per_1k": 1.0, "output_per_1k": 0.0, "multiplier": 1.0, "model_id": model_id}
        cost_in, _ = price_call(price, provider="anthropic", input_tokens=1_000, cache_write_tokens=1_000,
                                output_tokens=0)
        assert cost_in == pytest.approx(1.25)


def test_the_table_is_tried_longest_key_first():
    keys = list(LIST_PRICES)
    assert keys == sorted(keys, key=lambda k: -len(k))
    assert list_price("claude-opus-5-5-20261001") is LIST_PRICES[OPUS_5_5]
    assert list_price("claude-opus-5") is LIST_PRICES["claude-opus-5"]
    assert cache_read_multiplier("vendor/never-heard-of-it", 0.42) == 0.42


# ── the audit estimate prices the 5.5 models at list ───────────────────────

@pytest.mark.parametrize("model_id, per_1k", [
    (OPUS_5_5, (0.004, 0.020)),
    ("anthropic/claude-opus-5.5", (0.004, 0.020)),
    (SONNET_5_5, (0.002, 0.010)),
    (HAIKU_5_5, (0.0001, 0.0005)),
    (FABLE_5_1, (0.010, 0.050)),
    ("claude-opus-5", (0.005, 0.025)),               # Opus 5 keeps its own price
])
def test_the_audit_estimate_prices_each_model_at_list(model_id, per_1k):
    assert estimate_cost_usd(model_id, 1_000, 0) == pytest.approx(per_1k[0])
    assert estimate_cost_usd(model_id, 0, 1_000) == pytest.approx(per_1k[1])


# ── the 5.5 rows are not free ──────────────────────────────────────────────

@pytest.mark.parametrize("model_id", [OPUS_5_5, SONNET_5_5, HAIKU_5_5])
def test_an_unpriced_5_5_row_shows_its_list_price_not_free(model_id):
    from api.llm_marketplace import route_price

    row = NS(model_id=model_id, input_cost_per_1k_tokens=None, output_cost_per_1k_tokens=None)
    cost_in, cost_out, is_free = route_price(row)
    assert (cost_in, cost_out) == list_rates(model_id) and not is_free


def test_only_an_explicit_zero_is_free():
    from api.llm_marketplace import route_price

    assert route_price(NS(model_id="x/free", input_cost_per_1k_tokens=0, output_cost_per_1k_tokens=0))[2] is True
    unknown = route_price(NS(model_id="vendor/unpriced", input_cost_per_1k_tokens=None, output_cost_per_1k_tokens=None))
    assert unknown == (0.0, 0.0, False)
    priced = route_price(NS(model_id=OPUS_5_5, input_cost_per_1k_tokens=0.004, output_cost_per_1k_tokens=0.02))
    assert priced == (0.004, 0.02, False)


# ── the seed: insert if absent, price if NULL (real Postgres) ──────────────

def _route(db, model_id):
    from core.models.core import LLMModel

    return db.query(LLMModel).filter(LLMModel.serving_provider == "anthropic", LLMModel.model_id == model_id).first()


def test_the_seed_prices_the_three_5_5_routes(db_session):
    from api.llm_marketplace import route_price
    from core.seeds.seed_claude_list_prices import seed_claude_list_prices

    seed_claude_list_prices(db_session)
    for model_id in (OPUS_5_5, SONNET_5_5, HAIKU_5_5):
        row = _route(db_session, model_id)
        assert row is not None and row.input_cost_per_1k_tokens is not None
        assert row.output_cost_per_1k_tokens is not None
        assert route_price(row)[2] is False, f"{model_id} renders as free"


def test_the_seed_inserts_a_missing_route_and_fills_only_a_null_price(db_session, monkeypatch):
    from core.seeds import seed_claude_list_prices as seed

    opus, haiku = "claude-opus-5-5-t2seed", "claude-haiku-5-5-t2seed"   # the table matches both by key
    monkeypatch.setattr(seed, "SEEDED_ROUTES", ((opus, "Opus T2", 1_000_000, 128_000), (haiku, "Haiku T2", 0, 0)))

    assert seed.seed_claude_list_prices(db_session) == {"inserted": 2, "priced": 0}
    row = _route(db_session, opus)
    assert (row.input_cost_per_1k_tokens, row.output_cost_per_1k_tokens) == (0.004, 0.020)
    assert row.provider == "anthropic" and row.status == "active" and row.context_window == 1_000_000
    assert row.sourcing == "direct" and row.supports_functions is True

    row.input_cost_per_1k_tokens = None                 # a NULL beside an operator's output figure
    row.output_cost_per_1k_tokens = 0.033
    held = _route(db_session, haiku)
    held.input_cost_per_1k_tokens, held.output_cost_per_1k_tokens = 0.0, 0.0   # an explicit 0 is a price
    held.status = "deprecated"
    db_session.flush()

    assert seed.seed_claude_list_prices(db_session) == {"inserted": 0, "priced": 1}
    assert (row.input_cost_per_1k_tokens, row.output_cost_per_1k_tokens) == (0.004, 0.033)
    assert (held.input_cost_per_1k_tokens, held.output_cost_per_1k_tokens, held.status) == (0.0, 0.0, "deprecated")
