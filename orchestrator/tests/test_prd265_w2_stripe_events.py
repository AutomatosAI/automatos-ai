"""PRD-265 W2 — Stripe events → provider-neutral facts (``services/billing/stripe_events.py``).

``translate`` reads both the pinned API version's shape (``2025-03-31.basil``)
and the older one a dashboard-made endpoint still sends, so the webhook never
breaks across a Stripe version bump. Only :data:`WEBHOOK_EVENTS` are acted on;
anything else becomes :class:`Ignored`.
"""
from __future__ import annotations

from services.billing import stripe_events as se
from services.billing.facts import (
    CheckoutCompleted,
    Ignored,
    InvoicePaymentPaid,
    PaymentDisputed,
    PaymentFailed,
    PaymentRefunded,
    PeriodPaid,
    SubscriptionChanged,
    SubscriptionEnded,
    TopupPaid,
    WebhookEvent,
)

WORKSPACE_A = "11111111-1111-1111-1111-111111111111"
WORKSPACE_B = "22222222-2222-2222-2222-222222222222"


def _event(event_type: str, data: dict) -> WebhookEvent:
    return WebhookEvent(provider="stripe", event_id="evt_1", event_type=event_type, livemode=False, data=data)


# --------------------------------------------------------------------------- #
# invoice.paid — basil and pre-basil shapes
# --------------------------------------------------------------------------- #


def test_invoice_paid_basil_shape_reads_subscription_price_and_payment():
    invoice = {
        "id": "in_basil1",
        "customer": "cus_1",
        "amount_paid": 2900,
        "parent": {"subscription_details": {"subscription": "sub_1", "metadata": {"workspace_id": WORKSPACE_A}}},
        "lines": {"data": [
            {"amount": 2900, "pricing": {"price_details": {"price": "price_pro"}}, "period": {"end": 1700003600}},
        ]},
        "payments": {"data": [{"payment": {"payment_intent": "pi_basil1"}}]},
    }
    fact = se.translate(_event("invoice.paid", invoice))
    assert isinstance(fact, PeriodPaid)
    assert fact.subscription_id == "sub_1"
    assert fact.price_ids == ("price_pro",)
    assert fact.payment_ref == "pi_basil1"
    assert fact.workspace_ref == WORKSPACE_A
    assert fact.amount_cents == 2900


def test_invoice_paid_pre_basil_shape_reads_subscription_price_and_payment():
    invoice = {
        "id": "in_prebasil1",
        "customer": "cus_2",
        "amount_paid": 900,
        "subscription": "sub_2",
        "subscription_details": {"metadata": {"workspace_id": WORKSPACE_B}},
        "lines": {"data": [{"amount": 900, "price": {"id": "price_basic"}, "period": {"end": 1700007200}}]},
        "payment_intent": "pi_prebasil1",
    }
    fact = se.translate(_event("invoice.paid", invoice))
    assert isinstance(fact, PeriodPaid)
    assert fact.subscription_id == "sub_2"
    assert fact.price_ids == ("price_basic",)
    assert fact.payment_ref == "pi_prebasil1"
    assert fact.workspace_ref == WORKSPACE_B


def test_invoice_paid_orders_prices_by_line_amount_desc_and_takes_the_max_period_end():
    invoice = {
        "id": "in_multi",
        "customer": "cus_1",
        "amount_paid": 3000,
        "subscription": "sub_1",
        "lines": {"data": [
            {"amount": 500, "pricing": {"price_details": {"price": "price_a"}}, "period": {"end": 1000}},
            {"amount": 1500, "pricing": {"price_details": {"price": "price_b"}}, "period": {"end": 3000}},
            {"amount": 1000, "pricing": {"price_details": {"price": "price_c"}}, "period": {"end": 2000}},
        ]},
    }
    fact = se.translate(_event("invoice.paid", invoice))
    assert fact.price_ids == ("price_b", "price_c", "price_a")
    assert fact.period_end.timestamp() == 3000


def test_invoice_paid_with_no_subscription_is_ignored_as_a_topup_receipt():
    invoice = {"id": "in_topup", "customer": "cus_1", "amount_paid": 1000, "lines": {"data": []}}
    fact = se.translate(_event("invoice.paid", invoice))
    assert isinstance(fact, Ignored)


def test_invoice_paid_with_no_priced_period_is_ignored():
    invoice = {"id": "in_bare", "customer": "cus_1", "subscription": "sub_1", "lines": {"data": []}}
    fact = se.translate(_event("invoice.paid", invoice))
    assert isinstance(fact, Ignored)


# --------------------------------------------------------------------------- #
# checkout.session.completed / async_payment_succeeded
# --------------------------------------------------------------------------- #


def test_checkout_session_completed_subscription():
    session = {"mode": "subscription", "customer": "cus_1", "subscription": "sub_1",
               "client_reference_id": WORKSPACE_A}
    fact = se.translate(_event("checkout.session.completed", session))
    assert fact == CheckoutCompleted(customer_id="cus_1", subscription_id="sub_1", workspace_ref=WORKSPACE_A)


def test_checkout_session_completed_payment_topup_paid():
    session = {"mode": "payment", "id": "cs_1", "customer": "cus_1", "payment_intent": "pi_1",
               "amount_total": 3000, "payment_status": "paid", "client_reference_id": WORKSPACE_A,
               "metadata": {"kind": "topup", "workspace_id": WORKSPACE_A, "quantity": "3"}}
    fact = se.translate(_event("checkout.session.completed", session))
    assert fact == TopupPaid(customer_id="cus_1", session_id="cs_1", workspace_ref=WORKSPACE_A, quantity=3,
                             payment_ref="pi_1", amount_cents=3000)


def test_checkout_session_completed_payment_unpaid_is_ignored():
    session = {"mode": "payment", "id": "cs_1", "payment_status": "unpaid",
               "metadata": {"kind": "topup", "quantity": "1"}}
    assert isinstance(se.translate(_event("checkout.session.completed", session)), Ignored)


def test_checkout_session_completed_payment_without_our_metadata_is_ignored():
    session = {"mode": "payment", "id": "cs_1", "payment_status": "paid", "metadata": {}}
    assert isinstance(se.translate(_event("checkout.session.completed", session)), Ignored)


def test_async_payment_succeeded_uses_the_same_reader_as_completed():
    session = {"mode": "subscription", "customer": "cus_1", "subscription": "sub_1",
               "client_reference_id": WORKSPACE_A}
    fact = se.translate(_event("checkout.session.async_payment_succeeded", session))
    assert isinstance(fact, CheckoutCompleted)


# --------------------------------------------------------------------------- #
# invoice_payment.paid
# --------------------------------------------------------------------------- #


def test_invoice_payment_paid_links_the_payment_to_its_invoice():
    payment = {"invoice": "in_1", "payment": {"payment_intent": "pi_1"}, "amount_paid": 2900}
    fact = se.translate(_event("invoice_payment.paid", payment))
    assert fact == InvoicePaymentPaid(invoice_id="in_1", payment_ref="pi_1", amount_cents=2900)


def test_invoice_payment_paid_without_a_payment_intent_is_ignored():
    payment = {"invoice": "in_1", "payment": {}}
    assert isinstance(se.translate(_event("invoice_payment.paid", payment)), Ignored)


# --------------------------------------------------------------------------- #
# subscriptions
# --------------------------------------------------------------------------- #


def test_subscription_updated_reads_current_period_end_from_items_when_basil():
    sub = {"id": "sub_1", "customer": "cus_1", "status": "active", "cancel_at_period_end": False,
           "items": {"data": [{"price": "price_pro", "current_period_end": 1700000000}]}}
    fact = se.translate(_event("customer.subscription.updated", sub))
    assert isinstance(fact, SubscriptionChanged)
    assert fact.current_period_end.timestamp() == 1700000000
    assert fact.price_ids == ("price_pro",)


def test_subscription_created_uses_the_same_reader():
    sub = {"id": "sub_1", "customer": "cus_1", "status": "trialing", "current_period_end": 1700000000,
           "items": {"data": []}}
    fact = se.translate(_event("customer.subscription.created", sub))
    assert isinstance(fact, SubscriptionChanged)


def test_subscription_deleted():
    sub = {"id": "sub_del", "customer": "cus_1"}
    fact = se.translate(_event("customer.subscription.deleted", sub))
    assert fact == SubscriptionEnded(customer_id="cus_1", subscription_id="sub_del")


# --------------------------------------------------------------------------- #
# charges, disputes, failed payments
# --------------------------------------------------------------------------- #


def test_charge_refunded():
    charge = {"id": "ch_1", "payment_intent": "pi_1", "amount": 1000, "amount_refunded": 400}
    fact = se.translate(_event("charge.refunded", charge))
    assert fact == PaymentRefunded(charge_id="ch_1", payment_ref="pi_1", amount_cents=1000,
                                   amount_refunded_cents=400)


def test_dispute_created_has_no_outcome_yet():
    dispute = {"id": "dp_1", "payment_intent": "pi_1", "amount": 500}
    fact = se.translate(_event("charge.dispute.created", dispute))
    assert fact == PaymentDisputed(dispute_id="dp_1", payment_ref="pi_1", amount_cents=500, outcome=None)


def test_dispute_closed_won():
    dispute = {"id": "dp_1", "payment_intent": "pi_1", "amount": 500, "status": "won"}
    fact = se.translate(_event("charge.dispute.closed", dispute))
    assert fact.outcome == "won"


def test_dispute_closed_lost():
    dispute = {"id": "dp_1", "payment_intent": "pi_1", "amount": 500, "status": "lost"}
    fact = se.translate(_event("charge.dispute.closed", dispute))
    assert fact.outcome == "lost"


def test_dispute_closed_warning_closed_is_ignored():
    dispute = {"id": "dp_1", "payment_intent": "pi_1", "amount": 500, "status": "warning_closed"}
    assert isinstance(se.translate(_event("charge.dispute.closed", dispute)), Ignored)


def test_invoice_payment_failed():
    invoice = {"subscription": "sub_1", "customer": "cus_1", "id": "in_1", "attempt_count": 2,
               "next_payment_attempt": 1700000000}
    fact = se.translate(_event("invoice.payment_failed", invoice))
    assert isinstance(fact, PaymentFailed)
    assert fact.customer_id == "cus_1" and fact.invoice_id == "in_1" and fact.attempt_count == 2
    assert fact.next_attempt.timestamp() == 1700000000


def test_invoice_payment_failed_without_a_subscription_is_ignored():
    invoice = {"customer": "cus_1", "id": "in_1"}
    assert isinstance(se.translate(_event("invoice.payment_failed", invoice)), Ignored)


# --------------------------------------------------------------------------- #
# the unsubscribed, and the contract between WEBHOOK_EVENTS and the readers
# --------------------------------------------------------------------------- #


def test_an_unsubscribed_event_type_is_ignored():
    fact = se.translate(_event("customer.updated", {"id": "cus_1"}))
    assert isinstance(fact, Ignored)
    assert "customer.updated" in fact.reason


def test_webhook_events_equals_the_readers_keys():
    assert set(se.WEBHOOK_EVENTS) == set(se._READERS.keys())
