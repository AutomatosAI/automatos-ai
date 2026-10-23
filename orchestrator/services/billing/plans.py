"""What the Billing tab shows: the plans, the top-up and this workspace's subscription (PRD-265 US-011).

Prices and limits are read from ``config.PLAN_TIERS`` (the facts sheet's numbers),
so the tab, Auto's plan proposal and the operator console never disagree. Stripe's
own Prices are created from the same numbers (``scripts/stripe_setup.py``).
"""
from __future__ import annotations

from typing import Any, Optional

from services.billing.accounts import account_for_workspace
from services.billing.checkout import checkout_open_for, is_configured, topup_available

PLAN_ORDER = ("basic", "pro", "business", "enterprise")


def plan_card(key: str, tier: dict) -> dict:
    """One plan as the picker shows it. Pure."""
    families = tier.get("families") or {}
    return {
        "key": key,
        "name": tier.get("display_name") or key,
        "price_usd": tier.get("display_price_usd"),
        "monthly_credits": int(tier.get("monthly_credits") or 0),
        "platform_ai": bool(tier.get("platform_ai")),
        "self_serve": bool(tier.get("self_serve")) and not tier.get("coming_soon"),
        "coming_soon": bool(tier.get("coming_soon")),
        "seats": tier.get("seats"),
        "max_agents": tier.get("max_agents"),
        "render_minutes_month": tier.get("render_minutes_month"),
        "voice": bool(families.get("voice")),
    }


def _subscription(account: Any) -> dict:
    if account is None:
        return {"status": None, "cancel_at_period_end": False, "current_period_end": None,
                "has_billing_account": False}
    end = account.current_period_end or account.credited_period_end
    return {"status": account.subscription_status, "cancel_at_period_end": bool(account.cancel_at_period_end),
            "current_period_end": end.isoformat() if end else None, "has_billing_account": True}


def plans_view(db: Any, workspace: Any, *, user: Any, can_manage: bool, tiers: Optional[dict] = None) -> dict:
    """The Billing tab's data for one workspace and one caller."""
    from config import PLAN_TIERS, config

    resolved = tiers if tiers is not None else PLAN_TIERS
    account = account_for_workspace(db, workspace.id)
    return {
        "billing_open": bool(config.BILLING_OPEN),
        "checkout_available": is_configured() and checkout_open_for(user),
        "can_manage": bool(can_manage),
        "current_plan": workspace.plan,
        "subscription": _subscription(account),
        "plans": [plan_card(key, resolved[key]) for key in PLAN_ORDER if key in resolved],
        "topup": {"credits": int(config.TOPUP_CREDITS), "price_usd": int(config.TOPUP_PRICE_USD),
                  "valid_days": int(config.TOPUP_VALID_DAYS),
                  "available": topup_available(account, workspace.plan)},
    }
