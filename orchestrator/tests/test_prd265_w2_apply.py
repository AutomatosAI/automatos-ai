"""PRD-265 W2 — acting on one payment fact (``services/billing/apply.py``).

Only the dispatch and the pure branches are covered here (an unknown customer,
a fact whose price sells no plan, a late event for a replaced subscription, a
clawback with nothing to take back): the ledger math itself is the realdb
lane's job. ``db`` is a bare ``MagicMock`` and every DB-touching helper is
monkeypatched, so nothing here needs a database.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from services.billing import facts as f
from services.billing.apply import (
    IGNORED,
    NEEDS_REVIEW,
    PROCESSED,
    Outcome,
    apply_fact,
    on_dispute,
    on_payment_failed,
    on_refund,
    on_subscription_changed,
    on_subscription_ended,
)
from services.billing.grants import Granted


def _db():
    return MagicMock()


def _account(**overrides):
    base = dict(workspace_id="ws-1", customer_id="cus_1", subscription_id="sub_1",
               subscription_status="active", plan="pro", current_period_end=None,
               credited_period_end=None, cancel_at_period_end=False, period_credits=0)
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# apply_fact — dispatch
# --------------------------------------------------------------------------- #


def test_apply_fact_of_ignored_returns_ignored_with_its_reason():
    outcome = apply_fact(_db(), f.Ignored("not handled"), provider="stripe")
    assert outcome == Outcome(IGNORED, note="not handled")


def test_apply_fact_of_an_unmapped_type_is_ignored():
    outcome = apply_fact(_db(), object(), provider="stripe")
    assert outcome.status == IGNORED
    assert "no handler" in outcome.note


def test_apply_fact_dispatches_every_known_fact_type_to_its_own_handler():
    from services.billing import apply

    expected = {
        f.CheckoutCompleted: apply.on_checkout_completed,
        f.TopupPaid: apply.on_topup_paid,
        f.PeriodPaid: apply.on_period_paid,
        f.InvoicePaymentPaid: apply.on_invoice_payment,
        f.SubscriptionChanged: apply.on_subscription_changed,
        f.SubscriptionEnded: apply.on_subscription_ended,
        f.PaymentFailed: apply.on_payment_failed,
        f.PaymentRefunded: apply.on_refund,
        f.PaymentDisputed: apply.on_dispute,
    }
    assert apply._HANDLERS == expected


# --------------------------------------------------------------------------- #
# on_checkout_completed
# --------------------------------------------------------------------------- #


def test_checkout_completed_for_an_unknown_customer_needs_review(monkeypatch):
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: None)
    fact = f.CheckoutCompleted(customer_id="cus_missing", subscription_id="sub_1", workspace_ref="ws-1")
    outcome = apply_fact(_db(), fact, provider="stripe")
    assert outcome.status == NEEDS_REVIEW
    assert "cus_missing" in outcome.note


def test_checkout_completed_links_the_subscription_onto_the_account(monkeypatch):
    account = _account(subscription_id=None, subscription_status=None)
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    fact = f.CheckoutCompleted(customer_id="cus_1", subscription_id="sub_new", workspace_ref="ws-1")
    outcome = apply_fact(_db(), fact, provider="stripe")
    assert outcome == Outcome(PROCESSED, workspace_id="ws-1")
    assert account.subscription_id == "sub_new"
    assert account.subscription_status == "active"


# --------------------------------------------------------------------------- #
# on_period_paid
# --------------------------------------------------------------------------- #


def test_period_paid_for_an_invoice_that_sells_no_plan_needs_review(monkeypatch):
    account = _account()
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    monkeypatch.setattr("services.billing.apply.plan_for_prices", lambda price_ids: None)
    fact = f.PeriodPaid(customer_id="cus_1", invoice_id="in_1", subscription_id="sub_1",
                        price_ids=("price_unknown",), period_end=datetime.now(timezone.utc),
                        payment_ref="pi_1", amount_cents=2900)
    outcome = apply_fact(_db(), fact, provider="stripe")
    assert outcome.status == NEEDS_REVIEW
    assert "in_1" in outcome.note


def test_period_paid_for_the_scheduled_downgrade_clears_the_schedule(monkeypatch):
    # PRD-265 W2 v2: once the invoice that pays for the scheduled plan lands, the
    # downgrade has taken effect — nothing is left waiting for the period end.
    account = _account(plan="pro", schedule_id="sub_sched_1", scheduled_plan="basic",
                       scheduled_at=datetime.now(timezone.utc))
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    monkeypatch.setattr("services.billing.apply.plan_for_prices", lambda price_ids: "basic")
    monkeypatch.setattr("services.billing.apply.grant_period", lambda *a, **k: 0)
    monkeypatch.setattr("services.plan_tiers.assign_plan", lambda *a, **k: None)
    fact = f.PeriodPaid(customer_id="cus_1", invoice_id="in_1", subscription_id="sub_1",
                        price_ids=("price_basic",), period_end=datetime.now(timezone.utc),
                        payment_ref="pi_1", amount_cents=900)
    apply_fact(_db(), fact, provider="stripe")
    assert account.plan == "basic"
    assert account.schedule_id is None and account.scheduled_plan is None and account.scheduled_at is None


def test_period_paid_for_a_different_plan_than_scheduled_leaves_the_schedule(monkeypatch):
    account = _account(plan="pro", schedule_id="sub_sched_1", scheduled_plan="basic",
                       scheduled_at=datetime.now(timezone.utc))
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    monkeypatch.setattr("services.billing.apply.plan_for_prices", lambda price_ids: "pro")
    monkeypatch.setattr("services.billing.apply.grant_period", lambda *a, **k: 2000)
    monkeypatch.setattr("services.plan_tiers.assign_plan", lambda *a, **k: None)
    fact = f.PeriodPaid(customer_id="cus_1", invoice_id="in_1", subscription_id="sub_1",
                        price_ids=("price_pro",), period_end=datetime.now(timezone.utc),
                        payment_ref="pi_1", amount_cents=2900)
    apply_fact(_db(), fact, provider="stripe")
    assert account.schedule_id == "sub_sched_1"  # a renewal of the same plan: the downgrade still waits


def test_period_paid_for_a_known_price_grants_credits_and_moves_the_plan(monkeypatch):
    account = _account(plan="basic")
    moved = []
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    monkeypatch.setattr("services.billing.apply.plan_for_prices", lambda price_ids: "pro")
    monkeypatch.setattr("services.billing.apply.grant_period", lambda *a, **k: 2000)
    monkeypatch.setattr("services.plan_tiers.assign_plan", lambda *a, **k: moved.append(a))
    period_end = datetime.now(timezone.utc)
    fact = f.PeriodPaid(customer_id="cus_1", invoice_id="in_1", subscription_id="sub_1",
                        price_ids=("price_pro",), period_end=period_end, payment_ref="pi_1", amount_cents=2900)
    outcome = apply_fact(_db(), fact, provider="stripe")
    assert outcome.status == PROCESSED
    assert "granted 2000 monthly credits" in outcome.note
    assert account.plan == "pro"
    assert len(moved) == 1  # the plan is moved exactly once


# --------------------------------------------------------------------------- #
# on_subscription_changed / on_subscription_ended
# --------------------------------------------------------------------------- #


def test_subscription_changed_for_an_unknown_customer_needs_review(monkeypatch):
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: None)
    fact = f.SubscriptionChanged(customer_id="cus_x", subscription_id="sub_1", status="active",
                                 price_ids=(), current_period_end=None, cancel_at_period_end=False)
    assert on_subscription_changed(_db(), fact, "stripe").status == NEEDS_REVIEW


def test_a_late_event_for_a_replaced_subscription_is_ignored(monkeypatch):
    account = _account(subscription_id="sub_current")
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    fact = f.SubscriptionChanged(customer_id="cus_1", subscription_id="sub_old", status="active",
                                 price_ids=(), current_period_end=None, cancel_at_period_end=False)
    outcome = on_subscription_changed(_db(), fact, "stripe")
    assert outcome.status == IGNORED
    assert account.subscription_id == "sub_current"  # untouched


def test_a_subscription_ending_moves_the_workspace_back_to_entry_and_notifies(monkeypatch):
    account = _account(subscription_status="active", plan="pro")
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    monkeypatch.setattr("services.credits.expiry.expire_monthly", lambda *a, **k: 0)
    monkeypatch.setattr("services.plan_tiers.assign_plan", lambda *a, **k: None)
    fact = f.SubscriptionChanged(customer_id="cus_1", subscription_id="sub_1", status="canceled",
                                 price_ids=(), current_period_end=None, cancel_at_period_end=False)
    outcome = on_subscription_changed(_db(), fact, "stripe")
    assert outcome.status == PROCESSED
    assert account.plan == "basic"  # ENTRY_PLAN
    assert account.period_credits == 0
    assert outcome.notice.event_type == "billing_subscription_ended"


def test_a_subscription_ending_clears_any_scheduled_downgrade(monkeypatch):
    account = _account(subscription_status="active", plan="pro", schedule_id="sub_sched_1",
                       scheduled_plan="basic", scheduled_at=datetime.now(timezone.utc))
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    monkeypatch.setattr("services.credits.expiry.expire_monthly", lambda *a, **k: 0)
    monkeypatch.setattr("services.plan_tiers.assign_plan", lambda *a, **k: None)
    fact = f.SubscriptionChanged(customer_id="cus_1", subscription_id="sub_1", status="canceled",
                                 price_ids=(), current_period_end=None, cancel_at_period_end=False)
    on_subscription_changed(_db(), fact, "stripe")
    assert account.schedule_id is None and account.scheduled_plan is None and account.scheduled_at is None


def test_subscription_deleted_delegates_to_subscription_changed_as_canceled(monkeypatch):
    account = _account(subscription_status="active")
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    monkeypatch.setattr("services.credits.expiry.expire_monthly", lambda *a, **k: 0)
    monkeypatch.setattr("services.plan_tiers.assign_plan", lambda *a, **k: None)
    fact = f.SubscriptionEnded(customer_id="cus_1", subscription_id="sub_1")
    outcome = on_subscription_ended(_db(), fact, "stripe")
    assert outcome.status == PROCESSED
    assert account.subscription_status == "canceled"


# --------------------------------------------------------------------------- #
# on_payment_failed
# --------------------------------------------------------------------------- #


def test_payment_failed_for_an_unknown_customer_needs_review(monkeypatch):
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: None)
    fact = f.PaymentFailed(customer_id="cus_x", invoice_id="in_1", attempt_count=1, next_attempt=None)
    assert on_payment_failed(_db(), fact, "stripe").status == NEEDS_REVIEW


def test_payment_failed_marks_past_due_and_mentions_the_retry_date_when_known(monkeypatch):
    account = _account()
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    retry = datetime(2026, 11, 1, tzinfo=timezone.utc)
    fact = f.PaymentFailed(customer_id="cus_1", invoice_id="in_1", attempt_count=2, next_attempt=retry)
    outcome = on_payment_failed(_db(), fact, "stripe")
    assert account.subscription_status == "past_due"
    assert "01 Nov" in outcome.notice.message


def test_payment_failed_without_a_next_attempt_has_no_retry_sentence(monkeypatch):
    account = _account()
    monkeypatch.setattr("services.billing.apply.resolve_account", lambda *a, **k: account)
    fact = f.PaymentFailed(customer_id="cus_1", invoice_id="in_1", attempt_count=1, next_attempt=None)
    outcome = on_payment_failed(_db(), fact, "stripe")
    assert "Stripe will try again" not in outcome.notice.message


# --------------------------------------------------------------------------- #
# on_refund / on_dispute
# --------------------------------------------------------------------------- #


def test_refund_of_a_payment_that_granted_nothing_is_processed_without_a_clawback(monkeypatch):
    monkeypatch.setattr("services.billing.apply.granted_by_payment", lambda *a, **k: None)
    fact = f.PaymentRefunded(charge_id="ch_1", payment_ref="pi_missing", amount_cents=1000,
                             amount_refunded_cents=1000)
    outcome = on_refund(_db(), fact, "stripe")
    assert outcome.status == PROCESSED
    assert "no credits were granted" in outcome.note


def test_refund_claws_back_the_refunded_share(monkeypatch):
    granted = Granted(workspace_id="ws-1", bucket="monthly", credits=2000, amount_cents=2900)
    monkeypatch.setattr("services.billing.apply.granted_by_payment", lambda *a, **k: granted)
    monkeypatch.setattr("services.billing.apply.claw_back_refund", lambda *a, **k: 500)
    fact = f.PaymentRefunded(charge_id="ch_1", payment_ref="pi_1", amount_cents=2900, amount_refunded_cents=500)
    outcome = on_refund(_db(), fact, "stripe")
    assert outcome == Outcome(PROCESSED, workspace_id="ws-1", note="took back 500 credits")


def test_dispute_opened_claws_back_the_disputed_share(monkeypatch):
    granted = Granted(workspace_id="ws-1", bucket="monthly", credits=2000, amount_cents=2900)
    monkeypatch.setattr("services.billing.apply.granted_by_payment", lambda *a, **k: granted)
    monkeypatch.setattr("services.billing.apply.claw_back_dispute", lambda *a, **k: 2000)
    fact = f.PaymentDisputed(dispute_id="dp_1", payment_ref="pi_1", amount_cents=2900, outcome=None)
    outcome = on_dispute(_db(), fact, "stripe")
    assert outcome.note == "dispute opened: took back 2000"


def test_dispute_won_restores_the_credits(monkeypatch):
    granted = Granted(workspace_id="ws-1", bucket="monthly", credits=2000, amount_cents=2900)
    monkeypatch.setattr("services.billing.apply.granted_by_payment", lambda *a, **k: granted)
    monkeypatch.setattr("services.billing.apply.restore_dispute", lambda *a, **k: 2000)
    fact = f.PaymentDisputed(dispute_id="dp_1", payment_ref="pi_1", amount_cents=2900, outcome="won")
    outcome = on_dispute(_db(), fact, "stripe")
    assert outcome.note == "dispute won: gave back 2000"


def test_dispute_lost_leaves_the_credits_taken(monkeypatch):
    granted = Granted(workspace_id="ws-1", bucket="monthly", credits=2000, amount_cents=2900)
    monkeypatch.setattr("services.billing.apply.granted_by_payment", lambda *a, **k: granted)
    fact = f.PaymentDisputed(dispute_id="dp_1", payment_ref="pi_1", amount_cents=2900, outcome="lost")
    outcome = on_dispute(_db(), fact, "stripe")
    assert outcome.note == "dispute lost: credits stay taken"


def test_dispute_for_a_payment_that_granted_nothing_is_processed(monkeypatch):
    monkeypatch.setattr("services.billing.apply.granted_by_payment", lambda *a, **k: None)
    fact = f.PaymentDisputed(dispute_id="dp_1", payment_ref="pi_missing", amount_cents=500, outcome=None)
    outcome = on_dispute(_db(), fact, "stripe")
    assert "no credits were granted" in outcome.note
