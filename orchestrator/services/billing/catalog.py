"""Which Stripe Price sells which plan (PRD-265 FR-9).

The Price IDs come from the environment only (``STRIPE_PRICE_*``, printed by
``scripts/stripe_setup.py``); the plans themselves stay in ``config.PLAN_TIERS``.
A webhook's invoice names a price; :func:`plan_for_price` reads the plan back.
"""
from __future__ import annotations

from typing import Optional

from services.billing.stripe_api import StripeNotConfigured


def _plan_prices() -> dict[str, str]:
    from config import config

    return {
        "basic": config.STRIPE_PRICE_BASIC,
        "pro": config.STRIPE_PRICE_PRO,
        "business": config.STRIPE_PRICE_BUSINESS,
    }


def price_for_plan(plan: str) -> str:
    """The Price that sells ``plan``; :class:`StripeNotConfigured` when unset."""
    price = _plan_prices().get(plan) or ""
    if not price:
        raise StripeNotConfigured(f"no Stripe price is configured for plan {plan!r}")
    return price


def plan_for_price(price_id: Optional[str]) -> Optional[str]:
    """The plan a Price sells, or None for a price that isn't one of ours."""
    if not price_id:
        return None
    for plan, price in _plan_prices().items():
        if price and price == price_id:
            return plan
    return None


def plan_for_prices(price_ids: tuple) -> Optional[str]:
    """The plan of the first price in ``price_ids`` that sells one."""
    for price_id in price_ids:
        plan = plan_for_price(price_id)
        if plan:
            return plan
    return None


def topup_price() -> str:
    from config import config

    if not config.STRIPE_PRICE_TOPUP:
        raise StripeNotConfigured("no Stripe price is configured for the top-up")
    return config.STRIPE_PRICE_TOPUP
