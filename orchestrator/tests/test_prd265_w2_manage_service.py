"""PRD-265 W2 v2 — managing a subscription inside Studio (``services/billing/manage.py``).

There is no portal: these services check the workspace's billing account, call
the provider, and keep the account in step at once. ``db`` and the provider are
fakes; every ``BillingAccount`` attribute the code reads is on a
``SimpleNamespace``, so nothing here touches a real database.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.billing.checkout import BillingRefused
from services.billing.manage import (
    cancel_change,
    cancel_plan,
    card_view,
    change_plan,
    finish_card_change,
    invoice_pdf,
    invoices_view,
    preview_change,
    resume_plan,
    start_card_change,
)
from services.billing.provider import CardOnFile, PlanChangePreview, PlanChangeResult


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.card = None
        self.finish_card_error = None
        self.finish_card_result = CardOnFile(brand="visa", last4="4242", exp_month=12, exp_year=2028)
        self.invoices_list = []
        self.invoice_pdf_error = None
        self.invoice_pdf_result = (b"%PDF-1.4", "ABCD0001.pdf")
        self.cancel_period_end_return = None
        self.preview = PlanChangePreview(amount_due_cents=1350, tax_cents=0, currency="usd")
        self.upgrade_result = PlanChangeResult(status="done")
        self.downgrade_result = PlanChangeResult(
            status="scheduled", schedule_id="sub_sched_1",
            effective_at=datetime(2026, 12, 1, tzinfo=timezone.utc))

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))

    def card_on_file(self, **kwargs):
        self._record("card_on_file", **kwargs)
        return self.card

    def start_card_change(self, **kwargs):
        self._record("start_card_change", **kwargs)
        return "seti_1_secret_abc"

    def finish_card_change(self, **kwargs):
        self._record("finish_card_change", **kwargs)
        if self.finish_card_error:
            raise self.finish_card_error
        return self.finish_card_result

    def invoices(self, **kwargs):
        self._record("invoices", **kwargs)
        return self.invoices_list

    def invoice_pdf(self, **kwargs):
        self._record("invoice_pdf", **kwargs)
        if self.invoice_pdf_error:
            raise self.invoice_pdf_error
        return self.invoice_pdf_result

    def set_cancel_at_period_end(self, **kwargs):
        self._record("set_cancel_at_period_end", **kwargs)
        return self.cancel_period_end_return

    def preview_plan_change(self, **kwargs):
        self._record("preview_plan_change", **kwargs)
        return self.preview

    def upgrade_now(self, **kwargs):
        self._record("upgrade_now", **kwargs)
        return self.upgrade_result

    def downgrade_at_period_end(self, **kwargs):
        self._record("downgrade_at_period_end", **kwargs)
        return self.downgrade_result

    def cancel_scheduled_change(self, **kwargs):
        self._record("cancel_scheduled_change", **kwargs)


def _workspace(plan="basic"):
    return SimpleNamespace(id="ws-1", name="Barber & Co", plan=plan)


def _account(**overrides):
    base = dict(customer_id="cus_1", subscription_id="sub_1", subscription_status="active", plan="pro",
               cancel_at_period_end=False, schedule_id=None, scheduled_plan=None, scheduled_at=None,
               current_period_end=None, credited_period_end=None, business_offer_until=None)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture(autouse=True)
def _publishable(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "STRIPE_PUBLISHABLE_KEY", "pk_test_1")


# --------------------------------------------------------------------------- #
# card_view / invoices_view — no account is a safe default, not an error
# --------------------------------------------------------------------------- #


def test_card_view_with_no_account_is_no_payment_method(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: None)
    assert card_view(db, provider, _workspace()) == {"payment_method": None}


def test_invoices_view_with_no_account_is_an_empty_list(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: None)
    assert invoices_view(db, provider, _workspace()) == {"items": []}


# --------------------------------------------------------------------------- #
# finish_card_change / invoice_pdf — 404 with no account; ValueError → 400/404
# --------------------------------------------------------------------------- #


def test_finish_card_change_404_with_no_account(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: None)
    with pytest.raises(BillingRefused) as exc_info:
        finish_card_change(db, provider, _workspace(), setup_id="seti_1")
    assert exc_info.value.status == 404


def test_finish_card_change_400_when_the_provider_raises_value_error(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: _account())
    provider.finish_card_error = ValueError("not finished")
    with pytest.raises(BillingRefused) as exc_info:
        finish_card_change(db, provider, _workspace(), setup_id="seti_1")
    assert exc_info.value.status == 400


def test_invoice_pdf_404_with_no_account(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: None)
    with pytest.raises(BillingRefused) as exc_info:
        invoice_pdf(db, provider, _workspace(), invoice_id="in_1")
    assert exc_info.value.status == 404


def test_invoice_pdf_404_when_the_provider_raises_value_error(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: _account())
    provider.invoice_pdf_error = ValueError("not this workspace's invoice")
    with pytest.raises(BillingRefused) as exc_info:
        invoice_pdf(db, provider, _workspace(), invoice_id="in_1")
    assert exc_info.value.status == 404


def test_invoice_pdf_returns_the_providers_bytes_and_name(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: _account())
    content, name = invoice_pdf(db, provider, _workspace(), invoice_id="in_1")
    assert (content, name) == (b"%PDF-1.4", "ABCD0001.pdf")


# --------------------------------------------------------------------------- #
# cancel_plan / resume_plan — 404 no account, 409 no live subscription
# --------------------------------------------------------------------------- #


def test_cancel_plan_404_with_no_account(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: None)
    with pytest.raises(BillingRefused) as exc_info:
        cancel_plan(db, provider, _workspace())
    assert exc_info.value.status == 404


def test_cancel_plan_409_with_no_live_subscription(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace",
                        lambda db, ws: _account(subscription_status="canceled"))
    with pytest.raises(BillingRefused) as exc_info:
        cancel_plan(db, provider, _workspace())
    assert exc_info.value.status == 409


def test_cancel_plan_drops_a_pending_schedule_then_cancels(monkeypatch, db, provider):
    account = _account(schedule_id="sub_sched_1", scheduled_plan="basic",
                       scheduled_at=datetime(2026, 12, 1, tzinfo=timezone.utc))
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: account)
    provider.cancel_period_end_return = datetime(2026, 12, 1, tzinfo=timezone.utc)
    result = cancel_plan(db, provider, _workspace())
    assert ("cancel_scheduled_change", {"schedule_id": "sub_sched_1"}) in provider.calls
    assert account.schedule_id is None and account.scheduled_plan is None and account.scheduled_at is None
    assert account.cancel_at_period_end is True
    assert result == {"cancel_at_period_end": True, "ends_at": "2026-12-01T00:00:00+00:00"}


def test_resume_plan_clears_cancel_at_period_end(monkeypatch, db, provider):
    account = _account(cancel_at_period_end=True)
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: account)
    result = resume_plan(db, provider, _workspace())
    assert result == {"cancel_at_period_end": False}
    assert account.cancel_at_period_end is False


# --------------------------------------------------------------------------- #
# preview_change / change_plan — direction, Business-only-while-offered, status
# --------------------------------------------------------------------------- #


def test_plan_change_409_on_the_same_plan(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace",
                        lambda db, ws: _account(plan="pro"))
    with pytest.raises(BillingRefused) as exc_info:
        preview_change(db, provider, _workspace(), plan="pro")
    assert exc_info.value.status == 409


def test_plan_change_409_while_cancelling(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace",
                        lambda db, ws: _account(plan="pro", cancel_at_period_end=True))
    with pytest.raises(BillingRefused) as exc_info:
        preview_change(db, provider, _workspace(), plan="basic")
    assert exc_info.value.status == 409


def test_business_is_refused_unless_offered(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace",
                        lambda db, ws: _account(plan="pro"))
    with pytest.raises(BillingRefused) as exc_info:
        preview_change(db, provider, _workspace(), plan="business")
    assert exc_info.value.status == 400


def test_business_is_allowed_while_the_offer_stands(monkeypatch, db, provider):
    offered_until = datetime.now(timezone.utc) + timedelta(days=3)
    monkeypatch.setattr("services.billing.manage.account_for_workspace",
                        lambda db, ws: _account(plan="pro", business_offer_until=offered_until))
    monkeypatch.setattr("services.billing.manage.price_for_plan", lambda plan: "price_business")
    preview = preview_change(db, provider, _workspace(), plan="business")
    assert preview["plan"] == "business"


def test_basic_to_pro_is_an_upgrade_and_pro_to_basic_is_a_downgrade(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.price_for_plan", lambda plan: f"price_{plan}")
    monkeypatch.setattr("services.billing.manage.account_for_workspace",
                        lambda db, ws: _account(plan="basic"))
    up = preview_change(db, provider, _workspace(), plan="pro")
    assert up["direction"] == "upgrade" and up["effective_at"] == "now"

    monkeypatch.setattr("services.billing.manage.account_for_workspace",
                        lambda db, ws: _account(plan="pro"))
    down = preview_change(db, provider, _workspace(), plan="basic")
    assert down["direction"] == "downgrade" and down["amount_due_cents"] == 0


def test_change_plan_upgrade_returns_done_and_clears_any_scheduled_change(monkeypatch, db, provider):
    account = _account(plan="basic", schedule_id="sub_sched_old", scheduled_plan="business",
                       scheduled_at=datetime.now(timezone.utc))
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: account)
    monkeypatch.setattr("services.billing.manage.price_for_plan", lambda plan: "price_pro")
    result = change_plan(db, provider, _workspace(), plan="pro")
    assert result == {"status": "done"}
    assert account.schedule_id is None and account.scheduled_plan is None and account.scheduled_at is None


def test_change_plan_upgrade_requires_action_carries_the_publishable_key(monkeypatch, db, provider):
    provider.upgrade_result = PlanChangeResult(status="requires_action", client_secret="pi_1_secret_xyz")
    monkeypatch.setattr("services.billing.manage.account_for_workspace",
                        lambda db, ws: _account(plan="basic"))
    monkeypatch.setattr("services.billing.manage.price_for_plan", lambda plan: "price_pro")
    result = change_plan(db, provider, _workspace(), plan="pro")
    assert result == {"status": "requires_action", "client_secret": "pi_1_secret_xyz",
                      "publishable_key": "pk_test_1"}


def test_change_plan_downgrade_stores_the_schedule_on_the_account(monkeypatch, db, provider):
    account = _account(plan="pro")
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: account)
    monkeypatch.setattr("services.billing.manage.price_for_plan", lambda plan: "price_basic")
    result = change_plan(db, provider, _workspace(), plan="basic")
    assert result == {"status": "scheduled", "effective_at": "2026-12-01T00:00:00+00:00"}
    assert account.schedule_id == "sub_sched_1"
    assert account.scheduled_plan == "basic"
    assert account.scheduled_at == datetime(2026, 12, 1, tzinfo=timezone.utc)


def test_cancel_change_undoes_a_scheduled_downgrade(monkeypatch, db, provider):
    account = _account(plan="pro", schedule_id="sub_sched_1", scheduled_plan="basic",
                       scheduled_at=datetime(2026, 12, 1, tzinfo=timezone.utc))
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: account)
    result = cancel_change(db, provider, _workspace())
    assert result == {"scheduled_change": None}
    assert account.schedule_id is None and account.scheduled_plan is None and account.scheduled_at is None
    assert ("cancel_scheduled_change", {"schedule_id": "sub_sched_1"}) in provider.calls


# --------------------------------------------------------------------------- #
# start_card_change — carries the publishable key
# --------------------------------------------------------------------------- #


def test_start_card_change_returns_the_providers_secret_and_the_publishable_key(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: _account())
    result = start_card_change(db, provider, _workspace(), email="owner@example.com")
    assert result == {"client_secret": "seti_1_secret_abc", "publishable_key": "pk_test_1"}


def test_a_card_change_needs_a_billing_account_and_never_creates_a_customer(monkeypatch, db, provider):
    monkeypatch.setattr("services.billing.manage.account_for_workspace", lambda db, ws: None)
    with pytest.raises(BillingRefused) as refused:
        start_card_change(db, provider, _workspace(), email="owner@example.com")
    assert refused.value.status == 404
    assert provider.calls == []
