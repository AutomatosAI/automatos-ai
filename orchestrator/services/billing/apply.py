"""Act on one payment fact: the plan, the billing account and the credits (PRD-265 US-010).

- A paid subscription invoice is the only thing that puts a workspace on a paid
  plan (FR-4): it assigns the plan the invoice paid for and grants its month.
- A subscription that ends (cancelled at period end, or unpaid after Stripe's
  retries) puts the workspace back on the entry plan (Own keys) and expires what
  is left of the month, so platform AI pauses; own-key calls keep working.
- A failed payment marks the account past due and tells the owner; Stripe retries.
- Refunds and disputes take credits back (``grants.py``).

Each handler returns an :class:`Outcome` that ``events.py`` stores on the event
row. Nothing here commits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from services.billing import facts as f
from services.billing.accounts import ENDED_STATUSES, has_live_subscription, resolve_account
from services.billing.catalog import plan_for_prices
from services.billing.grants import (
    SOURCE_INVOICE,
    SOURCE_TOPUP,
    claw_back_dispute,
    claw_back_refund,
    grant_period,
    grant_source,
    grant_topup,
    granted_by_payment,
    restore_dispute,
)

logger = logging.getLogger(__name__)

PROCESSED = "processed"
IGNORED = "ignored"
NEEDS_REVIEW = "needs_review"

NOTICE_PAYMENT_FAILED = "billing_payment_failed"
NOTICE_SUBSCRIPTION_ENDED = "billing_subscription_ended"


@dataclass(frozen=True)
class Notice:
    """A notification to send once the event's transaction has committed."""

    event_type: str
    title: str
    message: str


@dataclass(frozen=True)
class Outcome:
    status: str
    workspace_id: Any = None
    payment_ref: Optional[str] = None
    grant_source: Optional[str] = None
    amount_cents: Optional[int] = None
    note: Optional[str] = None
    notice: Optional[Notice] = None


def _review(note: str) -> Outcome:
    logger.error("[billing] event needs review: %s", note)
    return Outcome(NEEDS_REVIEW, note=note)


def _workspace(db: Any, workspace_id: Any) -> Any:
    from core.models.workspaces import Workspace

    return db.query(Workspace).filter(Workspace.id == workspace_id).first()


def _move_plan(db: Any, workspace_id: Any, plan: str) -> None:
    from services.plan_tiers import assign_plan

    workspace = _workspace(db, workspace_id)
    if workspace is not None and workspace.plan != plan:
        # No tier budget ceiling, as on the operator console: a plan change must
        # never hand a live workspace a spend throttle it didn't have.
        assign_plan(db, workspace, plan, commit=False, with_budget=False)


def on_checkout_completed(db: Any, fact: f.CheckoutCompleted, provider: str) -> Outcome:
    account = resolve_account(db, fact.customer_id, fact.workspace_ref, provider=provider)
    if account is None:
        return _review(f"checkout for unknown customer {fact.customer_id}")
    if fact.subscription_id:
        account.subscription_id = fact.subscription_id
        account.subscription_status = account.subscription_status or "active"
    return Outcome(PROCESSED, workspace_id=account.workspace_id)


def on_topup_paid(db: Any, fact: f.TopupPaid, provider: str) -> Outcome:
    account = resolve_account(db, fact.customer_id, fact.workspace_ref, provider=provider)
    if account is None:
        return _review(f"top-up {fact.session_id} for an unknown workspace")
    credits = grant_topup(db, account.workspace_id, session_id=fact.session_id, quantity=fact.quantity)
    return Outcome(PROCESSED, workspace_id=account.workspace_id, payment_ref=fact.payment_ref,
                   grant_source=grant_source(SOURCE_TOPUP, fact.session_id), amount_cents=fact.amount_cents,
                   note=f"granted {credits} top-up credits")


def on_period_paid(db: Any, fact: f.PeriodPaid, provider: str) -> Outcome:
    account = resolve_account(db, fact.customer_id, fact.workspace_ref, provider=provider)
    if account is None:
        return _review(f"invoice {fact.invoice_id} for unknown customer {fact.customer_id}")
    plan = plan_for_prices(fact.price_ids)
    if plan is None:
        return _review(f"invoice {fact.invoice_id} pays for no plan we sell ({', '.join(fact.price_ids)})")
    credits = grant_period(db, account, plan=plan, invoice_id=fact.invoice_id, period_end=fact.period_end)
    _move_plan(db, account.workspace_id, plan)
    account.plan = plan
    account.subscription_id = fact.subscription_id or account.subscription_id
    if account.subscription_status not in ("active", "trialing"):
        account.subscription_status = "active"
    return Outcome(PROCESSED, workspace_id=account.workspace_id, payment_ref=fact.payment_ref,
                   grant_source=grant_source(SOURCE_INVOICE, fact.invoice_id), amount_cents=fact.amount_cents,
                   note=f"{plan}: granted {credits} monthly credits")


def on_invoice_payment(db: Any, fact: f.InvoicePaymentPaid, provider: str) -> Outcome:
    # Only the link from payment to invoice is kept, for a later refund.
    return Outcome(PROCESSED, payment_ref=fact.payment_ref,
                   grant_source=grant_source(SOURCE_INVOICE, fact.invoice_id), amount_cents=fact.amount_cents)


def _end(db: Any, account: Any) -> Notice:
    from services.credits.expiry import expire_monthly
    from services.plan_tiers import ENTRY_PLAN, get_tier

    expire_monthly(db, account.workspace_id, account.credited_period_end)
    _move_plan(db, account.workspace_id, ENTRY_PLAN)
    ended = (get_tier(account.plan or "") or {}).get("display_name") or "paid"
    account.plan, account.period_credits = ENTRY_PLAN, 0
    return Notice(NOTICE_SUBSCRIPTION_ENDED, "Your subscription has ended",
                  f"Your {ended} plan has ended, so work on Automatos's AI keys is paused. "
                  "Choose a plan on the Billing tab, or add your own AI key in Settings.")


def on_subscription_changed(db: Any, fact: f.SubscriptionChanged, provider: str) -> Outcome:
    account = resolve_account(db, fact.customer_id, fact.workspace_ref, provider=provider)
    if account is None:
        return _review(f"subscription {fact.subscription_id} for unknown customer {fact.customer_id}")
    if has_live_subscription(account) and account.subscription_id != fact.subscription_id:
        # A late event for a subscription the workspace has since replaced.
        return Outcome(IGNORED, workspace_id=account.workspace_id, note="not the workspace's current subscription")
    was = account.subscription_status
    account.subscription_id = fact.subscription_id
    account.subscription_status = fact.status
    account.cancel_at_period_end = fact.cancel_at_period_end
    account.current_period_end = fact.current_period_end
    notice = _end(db, account) if fact.status in ENDED_STATUSES and was not in ENDED_STATUSES else None
    return Outcome(PROCESSED, workspace_id=account.workspace_id, note=f"status {fact.status}", notice=notice)


def on_subscription_ended(db: Any, fact: f.SubscriptionEnded, provider: str) -> Outcome:
    changed = f.SubscriptionChanged(customer_id=fact.customer_id, subscription_id=fact.subscription_id,
                                    status="canceled", price_ids=(), current_period_end=None,
                                    cancel_at_period_end=False)
    return on_subscription_changed(db, changed, provider)


def on_payment_failed(db: Any, fact: f.PaymentFailed, provider: str) -> Outcome:
    account = resolve_account(db, fact.customer_id, None, provider=provider)
    if account is None:
        return _review(f"failed invoice {fact.invoice_id} for unknown customer {fact.customer_id}")
    account.subscription_status = "past_due"
    retry = f" Stripe will try again on {fact.next_attempt:%d %b}." if fact.next_attempt else ""
    return Outcome(PROCESSED, workspace_id=account.workspace_id, note=f"attempt {fact.attempt_count}",
                   notice=Notice(NOTICE_PAYMENT_FAILED, "Your payment didn't go through",
                                 f"We couldn't take this month's payment.{retry} Update your card on the "
                                 "Billing tab (Manage payment) to keep your plan."))


def _granted_or_review(db: Any, payment_ref: Optional[str], what: str):
    granted = granted_by_payment(db, payment_ref) if payment_ref else None
    if granted is None:
        return None, Outcome(PROCESSED, payment_ref=payment_ref, note=f"{what}: no credits were granted by it")
    return granted, None


def on_refund(db: Any, fact: f.PaymentRefunded, provider: str) -> Outcome:
    granted, done = _granted_or_review(db, fact.payment_ref, f"refund of {fact.charge_id}")
    if done:
        return done
    taken = claw_back_refund(db, granted, payment_ref=fact.payment_ref, charge_cents=fact.amount_cents,
                             refunded_cents=fact.amount_refunded_cents)
    return Outcome(PROCESSED, workspace_id=granted.workspace_id, note=f"took back {taken} credits")


def on_dispute(db: Any, fact: f.PaymentDisputed, provider: str) -> Outcome:
    granted, done = _granted_or_review(db, fact.payment_ref, f"dispute {fact.dispute_id}")
    if done:
        return done
    if fact.outcome == "won":
        moved = restore_dispute(db, granted, payment_ref=fact.payment_ref, dispute_id=fact.dispute_id)
        return Outcome(PROCESSED, workspace_id=granted.workspace_id, note=f"dispute won: gave back {moved}")
    if fact.outcome == "lost":
        return Outcome(PROCESSED, workspace_id=granted.workspace_id, note="dispute lost: credits stay taken")
    taken = claw_back_dispute(db, granted, payment_ref=fact.payment_ref, dispute_id=fact.dispute_id,
                              disputed_cents=fact.amount_cents)
    return Outcome(PROCESSED, workspace_id=granted.workspace_id, note=f"dispute opened: took back {taken}")


_HANDLERS = {
    f.CheckoutCompleted: on_checkout_completed,
    f.TopupPaid: on_topup_paid,
    f.PeriodPaid: on_period_paid,
    f.InvoicePaymentPaid: on_invoice_payment,
    f.SubscriptionChanged: on_subscription_changed,
    f.SubscriptionEnded: on_subscription_ended,
    f.PaymentFailed: on_payment_failed,
    f.PaymentRefunded: on_refund,
    f.PaymentDisputed: on_dispute,
}


def apply_fact(db: Any, fact: Any, *, provider: str) -> Outcome:
    """Act on ``fact``. Does not commit."""
    if isinstance(fact, f.Ignored):
        return Outcome(IGNORED, note=fact.reason)
    handler = _HANDLERS.get(type(fact))
    if handler is None:
        return Outcome(IGNORED, note=f"no handler for {type(fact).__name__}")
    return handler(db, fact, provider)
