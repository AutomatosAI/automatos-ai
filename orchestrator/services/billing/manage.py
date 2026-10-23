"""The Billing tab's own management (PRD-265 W2): card, invoices, cancel, plan change.

There is no portal: the owner (or an admin) does all of it inside Studio. These
services check the workspace's billing account, call the provider, and keep the
account in step at once (the webhook confirms the same state moments later).

Plan changes:
- **Upgrade** (to a plan with a higher price) moves now and charges the prorated
  difference; when the bank wants 3-D Secure the answer carries the payment's
  client secret and the change waits until the customer confirms it in our tab.
  The credits follow from the paid invoice (``grants.grant_period``, prorated).
- **Downgrade** waits for the period end (a subscription schedule): nothing is
  charged or refunded now, and "keep my plan" undoes it.
- Business only while a platform admin's offer stands; never while the plan is set
  to end (resume it first).
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Optional

from services.billing.accounts import account_for_workspace, has_live_subscription
from services.billing.catalog import price_for_plan
from services.billing.checkout import BUSINESS_PLAN, BillingRefused, business_offered, publishable_key

# A preview's price holds for this long: the change charges what was shown.
PRORATION_DATE_MAX_AGE_SECONDS = 3600
PRORATION_DATE_SKEW_SECONDS = 60


def _account(db: Any, workspace: Any) -> Any:
    account = account_for_workspace(db, workspace.id)
    if account is None:
        raise BillingRefused(404, "There is nothing to manage yet: no plan has been bought here.")
    return account


def _subscribed(db: Any, workspace: Any) -> Any:
    account = _account(db, workspace)
    if not has_live_subscription(account):
        raise BillingRefused(409, "This workspace has no active plan.")
    return account


def card_view(db: Any, provider: Any, workspace: Any) -> dict:
    account = account_for_workspace(db, workspace.id)
    if account is None:
        return {"payment_method": None}
    card = provider.card_on_file(customer_id=account.customer_id, subscription_id=account.subscription_id)
    return {"payment_method": asdict(card) if card else None}


def start_card_change(db: Any, provider: Any, workspace: Any, *, email: Optional[str]) -> dict:
    """A setup session for a new card. Only for a workspace that has bought something:
    a customer is created by a (gated) checkout or a Business offer, never here."""
    account = _account(db, workspace)
    secret = provider.start_card_change(customer_id=account.customer_id, workspace_id=str(workspace.id))
    return {"client_secret": secret, "publishable_key": publishable_key()}


def finish_card_change(db: Any, provider: Any, workspace: Any, *, setup_id: str) -> dict:
    account = _account(db, workspace)
    try:
        card = provider.finish_card_change(customer_id=account.customer_id,
                                           subscription_id=account.subscription_id, setup_id=setup_id)
    except ValueError as exc:
        raise BillingRefused(400, "That card couldn't be saved. Enter it again.") from exc
    return {"payment_method": asdict(card)}


def invoices_view(db: Any, provider: Any, workspace: Any) -> dict:
    account = account_for_workspace(db, workspace.id)
    if account is None:
        return {"items": []}
    items = provider.invoices(customer_id=account.customer_id)
    return {"items": [{**asdict(i), "created": i.created.isoformat() if i.created else None} for i in items]}


def invoice_pdf(db: Any, provider: Any, workspace: Any, *, invoice_id: str) -> tuple[bytes, str]:
    account = _account(db, workspace)
    try:
        return provider.invoice_pdf(customer_id=account.customer_id, invoice_id=invoice_id)
    except ValueError as exc:
        raise BillingRefused(404, "Invoice not found.") from exc


def _drop_schedule(db: Any, provider: Any, account: Any) -> None:
    if account.schedule_id:
        provider.cancel_scheduled_change(schedule_id=account.schedule_id)
    account.schedule_id = account.scheduled_plan = account.scheduled_at = None


def cancel_plan(db: Any, provider: Any, workspace: Any) -> dict:
    """End the plan at the period end (a scheduled downgrade is dropped)."""
    account = _subscribed(db, workspace)
    _drop_schedule(db, provider, account)
    ends = provider.set_cancel_at_period_end(subscription_id=account.subscription_id, cancel=True)
    account.cancel_at_period_end = True
    db.commit()
    return {"cancel_at_period_end": True, "ends_at": ends.isoformat() if ends else None}


def resume_plan(db: Any, provider: Any, workspace: Any) -> dict:
    account = _subscribed(db, workspace)
    provider.set_cancel_at_period_end(subscription_id=account.subscription_id, cancel=False)
    account.cancel_at_period_end = False
    db.commit()
    return {"cancel_at_period_end": False}


def _list_price(plan: Optional[str]) -> float:
    from services.plan_tiers import get_tier

    return float((get_tier(plan or "") or {}).get("display_price_usd") or 0)


def _target(db: Any, workspace: Any, plan: str) -> tuple[Any, str, str]:
    """The account, the plan's price and the direction (``upgrade``/``downgrade``)."""
    from services.plan_tiers import is_self_serve

    account = _subscribed(db, workspace)
    if not (is_self_serve(plan) or (plan == BUSINESS_PLAN and business_offered(account))):
        raise BillingRefused(400, "That plan isn't sold here. Business is by invite: contact Automatos.")
    if account.plan == plan:
        raise BillingRefused(409, "You're already on this plan.")
    if account.cancel_at_period_end:
        raise BillingRefused(409, "Your plan is set to end. Resume it first, then change it.")
    direction = "upgrade" if _list_price(plan) > _list_price(account.plan) else "downgrade"
    return account, price_for_plan(plan), direction


def preview_change(db: Any, provider: Any, workspace: Any, *, plan: str) -> dict:
    account, price_id, direction = _target(db, workspace, plan)
    if direction == "downgrade":
        end = account.current_period_end or account.credited_period_end
        return {"plan": plan, "direction": direction, "amount_due_cents": 0, "tax_cents": 0, "currency": "usd",
                "effective_at": end.isoformat() if end else None}
    preview = provider.preview_plan_change(customer_id=account.customer_id,
                                           subscription_id=account.subscription_id, price_id=price_id)
    return {"plan": plan, "direction": direction, "amount_due_cents": preview.amount_due_cents,
            "tax_cents": preview.tax_cents, "currency": preview.currency, "effective_at": "now",
            "proration_date": preview.proration_date}


def fresh_proration_date(value: Optional[int], now: Optional[float] = None) -> Optional[int]:
    """A preview's proration moment when it is recent enough to honour, else None
    (Stripe then prorates as of now). Pure."""
    current = time.time() if now is None else now
    if value is None or not (current - PRORATION_DATE_MAX_AGE_SECONDS <= value <= current + PRORATION_DATE_SKEW_SECONDS):
        return None
    return int(value)


def change_plan(db: Any, provider: Any, workspace: Any, *, plan: str, proration_date: Optional[int] = None) -> dict:
    account, price_id, direction = _target(db, workspace, plan)
    if direction == "upgrade":
        # A waiting downgrade goes first, and the account says so at once: if the upgrade
        # then fails, the tab never shows a schedule Stripe no longer has.
        _drop_schedule(db, provider, account)
        db.commit()
        result = provider.upgrade_now(subscription_id=account.subscription_id, price_id=price_id,
                                      proration_date=fresh_proration_date(proration_date))
        if result.status == "requires_action":
            return {"status": result.status, "client_secret": result.client_secret,
                    "publishable_key": publishable_key()}
        return {"status": result.status}
    result = provider.downgrade_at_period_end(subscription_id=account.subscription_id, price_id=price_id)
    account.schedule_id, account.scheduled_plan, account.scheduled_at = result.schedule_id, plan, result.effective_at
    db.commit()
    return {"status": "scheduled", "effective_at": result.effective_at.isoformat() if result.effective_at else None}


def cancel_change(db: Any, provider: Any, workspace: Any) -> dict:
    """Keep the current plan: undo a downgrade waiting for the period end."""
    account = _subscribed(db, workspace)
    _drop_schedule(db, provider, account)
    db.commit()
    return {"scheduled_change": None}
