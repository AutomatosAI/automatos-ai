"""PRD-265 W2 v2 — the Billing tab's data (``services/billing/plans.py``).

``plan_card`` is a pure projection of one ``config.PLAN_TIERS`` entry.
``plans_view`` assembles the tab for one workspace and caller: whether checkout
is available, whether the caller can manage billing, whether a top-up is on
offer, the publishable key and counsel's consent text the in-page checkout
needs, and whether a platform admin's Business offer stands for this
workspace (Business is ``self_serve`` only while it does).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from services.billing.plans import PLAN_ORDER, plan_card, plans_view


def _workspace(plan="basic"):
    return SimpleNamespace(id="ws-1", plan=plan)


def _account(**overrides):
    base = dict(subscription_id="sub_1", subscription_status="active", customer_id="cus_1",
               current_period_end=None, credited_period_end=None, cancel_at_period_end=False,
               scheduled_plan=None, scheduled_at=None, business_offer_until=None)
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# plan_card — pure, against the real plan tiers
# --------------------------------------------------------------------------- #


def test_the_four_plans_are_in_the_fixed_order():
    assert PLAN_ORDER == ("basic", "pro", "business", "enterprise")


def test_plan_card_basic_is_self_serve_and_not_platform_ai():
    from config import PLAN_TIERS

    card = plan_card("basic", PLAN_TIERS["basic"])
    assert card == {
        "key": "basic", "name": "Own keys", "price_usd": 9, "monthly_credits": 0,
        "platform_ai": False, "self_serve": True, "coming_soon": False,
        "seats": 5, "max_agents": 20, "render_minutes_month": 60, "voice": False,
    }


def test_plan_card_business_is_not_self_serve():
    from config import PLAN_TIERS

    card = plan_card("business", PLAN_TIERS["business"])
    assert card["self_serve"] is False
    assert card["platform_ai"] is True
    assert card["voice"] is True


def test_plan_card_enterprise_is_coming_soon_and_not_self_serve():
    from config import PLAN_TIERS

    card = plan_card("enterprise", PLAN_TIERS["enterprise"])
    assert card["coming_soon"] is True
    assert card["self_serve"] is False
    assert card["price_usd"] is None


def test_plan_card_a_tier_marked_self_serve_but_coming_soon_is_not_self_serve():
    # self_serve is never true while coming_soon, even if a tier sets both.
    card = plan_card("x", {"display_name": "X", "self_serve": True, "coming_soon": True})
    assert card["self_serve"] is False


# --------------------------------------------------------------------------- #
# plans_view
# --------------------------------------------------------------------------- #


@pytest.fixture
def _hosted(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "BILLING_OPEN", False)
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_123")
    monkeypatch.setattr(config, "STRIPE_PUBLISHABLE_KEY", "pk_test_123")
    monkeypatch.setattr(config, "BILLING_WITHDRAWAL_CONSENT_TEXT", "")


def test_plans_view_lists_all_four_plans_in_order(monkeypatch, _hosted):
    monkeypatch.setattr("services.billing.plans.account_for_workspace", lambda db, ws: None)
    view = plans_view(None, _workspace(), user=SimpleNamespace(system_role="user"), can_manage=True)
    assert [plan["key"] for plan in view["plans"]] == ["basic", "pro", "business", "enterprise"]


def test_plans_view_with_no_account_has_no_subscription_and_no_topup(monkeypatch, _hosted):
    monkeypatch.setattr("services.billing.plans.account_for_workspace", lambda db, ws: None)
    view = plans_view(None, _workspace(plan="basic"), user=SimpleNamespace(system_role="user"), can_manage=False)
    assert view["subscription"] == {"status": None, "cancel_at_period_end": False, "current_period_end": None,
                                    "has_billing_account": False, "scheduled_change": None}
    assert view["topup"]["available"] is False
    assert view["can_manage"] is False
    assert view["current_plan"] == "basic"
    assert view["business_offer_until"] is None


def test_plans_view_checkout_available_needs_a_stripe_key_and_an_open_or_admin_caller(monkeypatch, _hosted):
    monkeypatch.setattr("services.billing.plans.account_for_workspace", lambda db, ws: None)
    not_admin = plans_view(None, _workspace(), user=SimpleNamespace(system_role="user"), can_manage=False)
    assert not_admin["checkout_available"] is False
    admin = plans_view(None, _workspace(), user=SimpleNamespace(system_role="super_admin"), can_manage=True)
    assert admin["checkout_available"] is True


def test_plans_view_checkout_unavailable_without_a_stripe_key(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "BILLING_OPEN", True)
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "")
    monkeypatch.setattr("services.billing.plans.account_for_workspace", lambda db, ws: None)
    view = plans_view(None, _workspace(), user=SimpleNamespace(system_role="user"), can_manage=True)
    assert view["checkout_available"] is False


def test_plans_view_topup_is_available_only_on_a_platform_ai_plan_with_a_live_subscription(monkeypatch, _hosted):
    monkeypatch.setattr("services.billing.plans.account_for_workspace", lambda db, ws: _account())
    pro = plans_view(None, _workspace(plan="pro"), user=SimpleNamespace(system_role="user"), can_manage=True)
    assert pro["topup"]["available"] is True

    monkeypatch.setattr("services.billing.plans.account_for_workspace", lambda db, ws: _account())
    basic = plans_view(None, _workspace(plan="basic"), user=SimpleNamespace(system_role="user"), can_manage=True)
    assert basic["topup"]["available"] is False  # Own keys: top-up credits can't pay for platform AI

    monkeypatch.setattr("services.billing.plans.account_for_workspace",
                        lambda db, ws: _account(subscription_status="canceled"))
    canceled = plans_view(None, _workspace(plan="pro"), user=SimpleNamespace(system_role="user"), can_manage=True)
    assert canceled["topup"]["available"] is False  # no live subscription


def test_plans_view_carries_the_publishable_key_and_consent_text(monkeypatch, _hosted):
    from config import config

    monkeypatch.setattr(config, "BILLING_WITHDRAWAL_CONSENT_TEXT", "You may withdraw consent at any time.")
    monkeypatch.setattr("services.billing.plans.account_for_workspace", lambda db, ws: None)
    view = plans_view(None, _workspace(), user=SimpleNamespace(system_role="user"), can_manage=True)
    assert view["publishable_key"] == "pk_test_123"
    assert view["consent_text"] == "You may withdraw consent at any time."


def test_plans_view_consent_text_is_empty_when_unset(monkeypatch, _hosted):
    monkeypatch.setattr("services.billing.plans.account_for_workspace", lambda db, ws: None)
    view = plans_view(None, _workspace(), user=SimpleNamespace(system_role="user"), can_manage=True)
    assert view["consent_text"] == ""


def test_business_is_self_serve_only_while_the_offer_stands(monkeypatch, _hosted):
    offered_until = datetime.now(timezone.utc) + timedelta(days=5)
    monkeypatch.setattr("services.billing.plans.account_for_workspace",
                        lambda db, ws: _account(business_offer_until=offered_until))
    view = plans_view(None, _workspace(), user=SimpleNamespace(system_role="user"), can_manage=True)
    business = next(p for p in view["plans"] if p["key"] == "business")
    assert business["self_serve"] is True
    assert view["business_offer_until"] == offered_until.isoformat()


def test_business_is_not_self_serve_once_the_offer_has_lapsed(monkeypatch, _hosted):
    expired = datetime.now(timezone.utc) - timedelta(days=1)
    monkeypatch.setattr("services.billing.plans.account_for_workspace",
                        lambda db, ws: _account(business_offer_until=expired))
    view = plans_view(None, _workspace(), user=SimpleNamespace(system_role="user"), can_manage=True)
    business = next(p for p in view["plans"] if p["key"] == "business")
    assert business["self_serve"] is False
    assert view["business_offer_until"] is None


def test_subscription_carries_a_scheduled_downgrade(monkeypatch, _hosted):
    effective = datetime(2026, 12, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("services.billing.plans.account_for_workspace",
                        lambda db, ws: _account(scheduled_plan="basic", scheduled_at=effective))
    view = plans_view(None, _workspace(plan="pro"), user=SimpleNamespace(system_role="user"), can_manage=True)
    assert view["subscription"]["scheduled_change"] == {"plan": "basic", "effective_at": "2026-12-01T00:00:00+00:00"}


def test_subscription_has_no_scheduled_change_by_default(monkeypatch, _hosted):
    monkeypatch.setattr("services.billing.plans.account_for_workspace", lambda db, ws: _account())
    view = plans_view(None, _workspace(plan="pro"), user=SimpleNamespace(system_role="user"), can_manage=True)
    assert view["subscription"]["scheduled_change"] is None
