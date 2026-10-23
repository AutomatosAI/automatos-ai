"""PRD-265 W2 — which Stripe Price sells which plan (``services/billing/catalog.py``).

The Price IDs come from the environment only; :func:`price_for_plan` raises
:class:`StripeNotConfigured` when a plan's price isn't set, and
:func:`plan_for_price` / :func:`plan_for_prices` read a webhook's price back to
the plan it sells.
"""
from __future__ import annotations

import pytest

from services.billing.catalog import plan_for_price, plan_for_prices, price_for_plan, topup_price
from services.billing.stripe_api import StripeNotConfigured


@pytest.fixture
def priced(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "STRIPE_PRICE_BASIC", "price_basic")
    monkeypatch.setattr(config, "STRIPE_PRICE_PRO", "price_pro")
    monkeypatch.setattr(config, "STRIPE_PRICE_BUSINESS", "price_business")
    monkeypatch.setattr(config, "STRIPE_PRICE_TOPUP", "price_topup")


def test_price_for_plan_raises_not_configured_when_unset(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "STRIPE_PRICE_PRO", "")
    with pytest.raises(StripeNotConfigured):
        price_for_plan("pro")


def test_price_for_plan_returns_the_configured_price(priced):
    assert price_for_plan("pro") == "price_pro"
    assert price_for_plan("business") == "price_business"


def test_plan_for_price_maps_a_price_back_to_its_plan(priced):
    assert plan_for_price("price_pro") == "pro"
    assert plan_for_price("price_basic") == "basic"


def test_plan_for_price_of_an_unknown_price_is_none(priced):
    assert plan_for_price("price_unknown") is None


def test_plan_for_price_of_nothing_is_none(priced):
    assert plan_for_price(None) is None
    assert plan_for_price("") is None


def test_plan_for_prices_returns_the_first_price_that_sells_a_plan(priced):
    assert plan_for_prices(("price_unknown", "price_pro", "price_basic")) == "pro"


def test_plan_for_prices_of_nothing_we_sell_is_none(priced):
    assert plan_for_prices(("price_unknown", "price_other")) is None


def test_plan_for_prices_of_an_empty_tuple_is_none(priced):
    assert plan_for_prices(()) is None


def test_topup_price_raises_not_configured_when_unset(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "STRIPE_PRICE_TOPUP", "")
    with pytest.raises(StripeNotConfigured):
        topup_price()


def test_topup_price_returns_the_configured_price(priced):
    assert topup_price() == "price_topup"
