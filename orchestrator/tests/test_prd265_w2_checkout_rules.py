"""PRD-265 W2 v2 — who may buy what, in-page (``services/billing/checkout.py``).

Business is never self-serve unless a platform admin's offer stands
(``business_offer_until`` in the future); a workspace with a live subscription
changes plan through ``manage.py`` instead of a second checkout; a top-up needs
a live subscription on a plan whose credits pay for platform AI; counsel's
consent text (when set) must be ticked, and the acceptance is stamped on the
session's metadata. The provider is a fake (no network) that records what it
was asked to do.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.billing.checkout import (
    BillingRefused,
    checkout_open_for,
    offer_business,
    start_plan_checkout,
    start_topup_checkout,
)
from services.billing.provider import CheckoutSession
from services.billing.stripe_api import StripeNotConfigured


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.customers = []
        self.checkouts = []

    def ensure_customer(self, **kwargs):
        self.customers.append(kwargs)
        return "cus_new"

    def create_checkout(self, request):
        self.checkouts.append(request)
        return CheckoutSession(session_id="cs_1", client_secret="cs_1_secret_abc")


def _workspace(plan="basic"):
    return SimpleNamespace(id="ws-1", name="Barber & Co", plan=plan)


def _account(**overrides):
    base = dict(customer_id="cus_1", subscription_id="sub_1", subscription_status="active", plan="basic",
               business_offer_until=None)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture(autouse=True)
def _frontend_and_publishable(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "FRONTEND_URL", "https://app.automatos.ai")
    monkeypatch.setattr(config, "STRIPE_PUBLISHABLE_KEY", "pk_test_1")
    monkeypatch.setattr(config, "BILLING_WITHDRAWAL_CONSENT_TEXT", "")


# --------------------------------------------------------------------------- #
# start_plan_checkout
# --------------------------------------------------------------------------- #


def test_a_plan_checkout_returns_the_client_secret_and_the_publishable_key(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_basic")
    result = start_plan_checkout(db, provider, _workspace(plan="basic"), plan="basic", email="owner@example.com")
    assert result == {"client_secret": "cs_1_secret_abc", "publishable_key": "pk_test_1"}
    assert len(provider.customers) == 1
    (request,) = provider.checkouts
    assert request.return_url == "https://app.automatos.ai/settings?tab=billing&checkout=return"


def test_business_is_refused_without_an_offer(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    with pytest.raises(BillingRefused) as exc_info:
        start_plan_checkout(db, provider, _workspace(), plan="business", email=None)
    assert exc_info.value.status == 400


def test_business_is_refused_once_the_offer_has_lapsed(monkeypatch, db, provider):
    expired = datetime.now(timezone.utc) - timedelta(days=1)
    monkeypatch.setattr("services.billing.checkout.account_for_workspace",
                        lambda db, ws: _account(business_offer_until=expired))
    with pytest.raises(BillingRefused) as exc_info:
        start_plan_checkout(db, provider, _workspace(), plan="business", email=None)
    assert exc_info.value.status == 400


def test_business_opens_checkout_while_the_offer_stands(monkeypatch, db, provider):
    offered_until = datetime.now(timezone.utc) + timedelta(days=5)
    monkeypatch.setattr("services.billing.checkout.account_for_workspace",
                        lambda db, ws: _account(business_offer_until=offered_until, subscription_status="canceled",
                                                subscription_id=None))
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_business")
    result = start_plan_checkout(db, provider, _workspace(), plan="business", email=None)
    assert result["client_secret"] == "cs_1_secret_abc"


def test_a_live_subscription_is_refused_with_409(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace",
                        lambda db, ws: _account(plan="pro"))
    with pytest.raises(BillingRefused) as exc_info:
        start_plan_checkout(db, provider, _workspace(plan="pro"), plan="pro", email=None)
    assert exc_info.value.status == 409


def test_an_unknown_or_unsold_plan_is_refused(db, provider):
    with pytest.raises(BillingRefused) as exc_info:
        start_plan_checkout(db, provider, _workspace(), plan="made-up", email=None)
    assert exc_info.value.status == 400


# --------------------------------------------------------------------------- #
# terms / consent
# --------------------------------------------------------------------------- #


def test_accept_terms_false_with_consent_text_set_is_422(monkeypatch, db, provider):
    from config import config

    monkeypatch.setattr(config, "BILLING_WITHDRAWAL_CONSENT_TEXT", "You may withdraw consent at any time.")
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_basic")
    with pytest.raises(BillingRefused) as exc_info:
        start_plan_checkout(db, provider, _workspace(), plan="basic", email=None, accept_terms=False)
    assert exc_info.value.status == 422


def test_accept_terms_true_stamps_the_metadata(monkeypatch, db, provider):
    from config import config

    monkeypatch.setattr(config, "BILLING_WITHDRAWAL_CONSENT_TEXT", "You may withdraw consent at any time.")
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_basic")
    before = datetime.now(timezone.utc)
    start_plan_checkout(db, provider, _workspace(), plan="basic", email=None, accept_terms=True)
    after = datetime.now(timezone.utc)
    (request,) = provider.checkouts
    stamped = datetime.fromisoformat(request.metadata["terms_accepted_at"])
    assert before <= stamped <= after


def test_no_consent_text_needs_no_acceptance(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_basic")
    result = start_plan_checkout(db, provider, _workspace(), plan="basic", email=None, accept_terms=False)
    assert result["client_secret"] == "cs_1_secret_abc"
    (request,) = provider.checkouts
    assert "terms_accepted_at" not in request.metadata


# --------------------------------------------------------------------------- #
# start_topup_checkout
# --------------------------------------------------------------------------- #


def test_a_topup_with_no_live_subscription_is_refused(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    with pytest.raises(BillingRefused) as exc_info:
        start_topup_checkout(db, provider, _workspace(plan="pro"))
    assert exc_info.value.status == 409


def test_a_topup_on_own_keys_is_refused_even_with_a_live_subscription(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace",
                        lambda db, ws: _account(plan="basic"))
    with pytest.raises(BillingRefused) as exc_info:
        start_topup_checkout(db, provider, _workspace(plan="basic"))
    assert exc_info.value.status == 409


def test_a_topup_on_pro_with_a_live_subscription_opens_checkout(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace",
                        lambda db, ws: _account(plan="pro", customer_id="cus_pro"))
    monkeypatch.setattr("services.billing.checkout.topup_price", lambda: "price_topup")
    result = start_topup_checkout(db, provider, _workspace(plan="pro"))
    assert result["client_secret"] == "cs_1_secret_abc"
    (request,) = provider.checkouts
    assert request.price_id == "price_topup" and request.customer_id == "cus_pro"


# --------------------------------------------------------------------------- #
# offer_business
# --------------------------------------------------------------------------- #


def test_offer_business_with_a_live_subscription_is_409(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: _account())
    with pytest.raises(BillingRefused) as exc_info:
        offer_business(db, provider, _workspace())
    assert exc_info.value.status == 409


def test_offer_business_without_a_business_price_raises_not_configured(monkeypatch, db, provider):
    from config import config

    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr(config, "STRIPE_PRICE_BUSINESS", "")
    with pytest.raises(StripeNotConfigured):
        offer_business(db, provider, _workspace())
    assert provider.customers == []  # refused before the customer is ever created


def test_offer_business_sets_the_offer_until_about_business_offer_days_ahead(monkeypatch, db, provider):
    from config import config

    created = _account(business_offer_until=None)
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_business")
    monkeypatch.setattr("services.billing.checkout.ensure_customer", lambda db, provider, ws, email: created)
    before = datetime.now(timezone.utc) + timedelta(days=int(config.BUSINESS_OFFER_DAYS))
    until = offer_business(db, provider, _workspace())
    after = datetime.now(timezone.utc) + timedelta(days=int(config.BUSINESS_OFFER_DAYS))
    assert before - timedelta(seconds=5) <= until <= after + timedelta(seconds=5)
    assert created.business_offer_until == until


def test_offer_business_creates_the_customer_exactly_once(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_business")
    monkeypatch.setattr("services.billing.checkout.link_account",
                        lambda db, ws_id, customer_id, provider: _account(customer_id=customer_id))
    offer_business(db, provider, _workspace())
    assert len(provider.customers) == 1


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
