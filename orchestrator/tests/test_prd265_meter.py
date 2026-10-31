"""PRD-265 — the meter: a call costs what it cost (pure parts).

Credits are metered like OpenRouter (Gerard, 20 Oct 2026): credits =
ceil(the call's real cost in cents × ``CREDITS_MARKUP``); 1 credit = 1 US cent at
Automatos prices. What the platform paid for a call:

- the row's ``total_cost`` — the provider's own figure when it reported one, else
  the model's registered price (one price source, ``UsageTracker.track``);
- $0 on the customer's own key;
- $0 on a free route — NVIDIA's price multiplier is 0, and $0 is its real price;
- $0 for a subscription session (the person's own CLI plan pays);
- a paid route whose price is unknown booked $0: it is metered at the conservative
  estimate instead, so an unpriced model is never free on the platform's keys.
The database half (debit once, mark charged, alerts) is ``test_prd265_asks_realdb.py``.
"""
from __future__ import annotations

import pytest

from config import config
from services.credits.meter import PendingCall, billable_cost
from services.credits.routing import credits_for


def _call(**kw):
    base = dict(usage_id=1, workspace_id="ws-1", model_id="claude-haiku-5-5", provider="anthropic", tier="direct",
                request_type="chat", input_tokens=1000, output_tokens=200, total_cost=0.0123, is_byok=False)
    base.update(kw)
    return PendingCall(**base)


def test_the_recorded_cost_is_what_is_charged():
    assert billable_cost(_call()) == 0.0123


def test_the_customers_own_key_costs_us_nothing():
    assert billable_cost(_call(is_byok=True)) == 0.0


def test_nvidias_free_route_stays_free():
    # NVIDIA is registered free=True, price_multiplier=0.0 (core/llm/providers.py).
    assert billable_cost(_call(provider="nvidia", total_cost=0.0)) == 0.0


def test_a_subscription_session_stays_free():
    assert billable_cost(_call(provider="claude_code", tier="subscription", total_cost=0.0)) == 0.0


def test_an_unpriced_paid_route_is_metered_at_the_estimate():
    cost = billable_cost(_call(model_id="some-new-model-nobody-priced", total_cost=0.0))
    assert cost > 0  # never free on the platform's keys


def test_a_call_with_no_tokens_costs_nothing():
    assert billable_cost(_call(total_cost=0.0, input_tokens=0, output_tokens=0)) == 0.0


@pytest.mark.parametrize("cost,markup,credits", [
    (5.00, 1.0, 500),     # $5 at cost = 500 credits
    (5.00, 1.2, 600),     # the default markup
    (0.0123, 1.2, 2),     # 1.476 cents → rounded up
    (0.0001, 1.2, 1),     # never under-charged
    (0.0, 1.2, 0),
])
def test_credits_are_cents_times_the_markup_rounded_up(monkeypatch, cost, markup, credits):
    monkeypatch.setattr(config, "CREDITS_MARKUP", markup)
    assert credits_for(cost) == credits
