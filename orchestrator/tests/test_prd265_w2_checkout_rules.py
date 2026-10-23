"""PRD-265 W2 — who may buy what (``services/billing/checkout.py``).

Business and Enterprise are never self-serve; a workspace with a live
subscription changes plan through the portal instead of a second Checkout; a
top-up needs a live subscription on a plan whose credits pay for platform AI.
The provider is a fake (no network) that records what it was asked to do.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.billing.checkout import (
    BillingRefused,
    business_invite,
    checkout_open_for,
    open_portal,
    start_plan_checkout,
    start_topup_checkout,
)
from services.billing.provider import CheckoutLink, PlanChange


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.customers = []
        self.checkouts = []
        self.portals = []

    def ensure_customer(self, **kwargs):
        self.customers.append(kwargs)
        return "cus_new"

    def create_checkout(self, request):
        self.checkouts.append(request)
        return CheckoutLink(url="https://checkout.stripe.com/pay/1", session_id="cs_1")

    def portal_link(self, **kwargs):
        self.portals.append(kwargs)
        return "https://billing.stripe.com/session/1"


def _workspace(plan="basic"):
    return SimpleNamespace(id="ws-1", name="Barber & Co", plan=plan)


def _account(**overrides):
    base = dict(customer_id="cus_1", subscription_id="sub_1", subscription_status="active", plan="basic")
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture(autouse=True)
def _frontend_url(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "FRONTEND_URL", "https://app.automatos.ai")


# --------------------------------------------------------------------------- #
# start_plan_checkout
# --------------------------------------------------------------------------- #


def test_business_is_never_sold_self_serve(db, provider):
    with pytest.raises(BillingRefused) as exc_info:
        start_plan_checkout(db, provider, _workspace(), plan="business", email=None)
    assert exc_info.value.status == 400


def test_enterprise_is_never_sold_self_serve(db, provider):
    with pytest.raises(BillingRefused) as exc_info:
        start_plan_checkout(db, provider, _workspace(), plan="enterprise", email=None)
    assert exc_info.value.status == 400


def test_a_live_subscription_on_the_same_plan_is_refused(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace",
                        lambda db, ws: _account(plan="pro"))
    with pytest.raises(BillingRefused) as exc_info:
        start_plan_checkout(db, provider, _workspace(plan="pro"), plan="pro", email=None)
    assert exc_info.value.status == 409


def test_a_live_subscription_on_another_plan_opens_the_portals_plan_change(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace",
                        lambda db, ws: _account(plan="basic", customer_id="cus_1", subscription_id="sub_1"))
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_pro")
    result = start_plan_checkout(db, provider, _workspace(plan="basic"), plan="pro", email=None)
    assert result == {"url": "https://billing.stripe.com/session/1", "flow": "portal"}
    (call,) = provider.portals
    assert call["customer_id"] == "cus_1"
    assert call["change"] == PlanChange(subscription_id="sub_1", price_id="price_pro")
    assert provider.checkouts == []


def test_with_no_account_the_customer_is_created_once_and_checkout_opens(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_basic")
    result = start_plan_checkout(db, provider, _workspace(plan="basic"), plan="basic", email="owner@example.com")
    assert result == {"url": "https://checkout.stripe.com/pay/1", "flow": "checkout"}
    assert len(provider.customers) == 1
    (request,) = provider.checkouts
    assert request.success_url == "https://app.automatos.ai/settings?tab=billing&checkout=success"
    assert request.cancel_url == "https://app.automatos.ai/settings?tab=billing&checkout=cancelled"


def test_an_unknown_or_unsold_plan_is_refused(db, provider):
    with pytest.raises(BillingRefused) as exc_info:
        start_plan_checkout(db, provider, _workspace(), plan="made-up", email=None)
    assert exc_info.value.status == 400


# --------------------------------------------------------------------------- #
# start_topup_checkout
# --------------------------------------------------------------------------- #


def test_a_topup_with_no_live_subscription_is_refused(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    with pytest.raises(BillingRefused) as exc_info:
        start_topup_checkout(db, provider, _workspace(plan="pro"), email=None)
    assert exc_info.value.status == 409


def test_a_topup_on_own_keys_is_refused_even_with_a_live_subscription(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace",
                        lambda db, ws: _account(plan="basic"))
    with pytest.raises(BillingRefused) as exc_info:
        start_topup_checkout(db, provider, _workspace(plan="basic"), email=None)
    assert exc_info.value.status == 409


def test_a_topup_on_pro_with_a_live_subscription_opens_checkout(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace",
                        lambda db, ws: _account(plan="pro", customer_id="cus_pro"))
    monkeypatch.setattr("services.billing.checkout.topup_price", lambda: "price_topup")
    result = start_topup_checkout(db, provider, _workspace(plan="pro"), email=None)
    assert result == {"url": "https://checkout.stripe.com/pay/1", "flow": "checkout"}
    (request,) = provider.checkouts
    assert request.price_id == "price_topup" and request.customer_id == "cus_pro"


# --------------------------------------------------------------------------- #
# open_portal
# --------------------------------------------------------------------------- #


def test_open_portal_with_no_account_is_404(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    with pytest.raises(BillingRefused) as exc_info:
        open_portal(db, provider, _workspace())
    assert exc_info.value.status == 404


def test_open_portal_with_an_account_returns_the_portal_url(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace",
                        lambda db, ws: _account(customer_id="cus_1"))
    assert open_portal(db, provider, _workspace()) == "https://billing.stripe.com/session/1"
    assert provider.portals[0]["customer_id"] == "cus_1"


# --------------------------------------------------------------------------- #
# business_invite
# --------------------------------------------------------------------------- #


def test_business_invite_with_a_live_subscription_is_refused(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: _account())
    with pytest.raises(BillingRefused) as exc_info:
        business_invite(db, provider, _workspace())
    assert exc_info.value.status == 409


def test_business_invite_without_a_subscription_opens_checkout_for_a_day(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_business")
    before = datetime.now(timezone.utc)
    result = business_invite(db, provider, _workspace())
    after = datetime.now(timezone.utc)
    assert result["url"] == "https://checkout.stripe.com/pay/1"
    expires_at = datetime.fromisoformat(result["expires_at"])
    assert before + timedelta(hours=23) < expires_at < after + timedelta(hours=24)
    (request,) = provider.checkouts
    assert request.price_id == "price_business"


# --------------------------------------------------------------------------- #
# checkout_open_for
# --------------------------------------------------------------------------- #


def test_checkout_open_for_anyone_when_billing_is_open(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "BILLING_OPEN", True)
    assert checkout_open_for(SimpleNamespace(system_role="user")) is True
    assert checkout_open_for(None) is True


def test_checkout_open_for_platform_admins_only_while_closed(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "BILLING_OPEN", False)
    assert checkout_open_for(SimpleNamespace(system_role="user")) is False
    assert checkout_open_for(SimpleNamespace(system_role="admin")) is True
    assert checkout_open_for(SimpleNamespace(system_role="super_admin")) is True
    assert checkout_open_for(None) is False
