"""Stripe webhook events → provider-neutral facts (PRD-265 US-010).

Requests pin ``config.STRIPE_API_VERSION`` (2025-03-31.basil by default) and the
setup script creates the webhook endpoint at it, but the readers below accept the
older shapes too (``invoice.subscription`` vs ``invoice.parent.subscription_details``,
``line.price`` vs ``line.pricing.price_details``, ``subscription.current_period_end``
vs its items'), so an endpoint made at another version in the dashboard still works.

Only :data:`WEBHOOK_EVENTS` are subscribed; anything else becomes :class:`Ignored`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

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

WEBHOOK_EVENTS = (
    "checkout.session.completed",
    "checkout.session.async_payment_succeeded",
    "invoice.paid",
    "invoice_payment.paid",
    "invoice.payment_failed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
    "charge.refunded",
    "charge.dispute.created",
    "charge.dispute.closed",
)

# The metadata our checkouts carry (services/billing/checkout.py writes them).
META_WORKSPACE = "workspace_id"
META_KIND = "kind"
META_QUANTITY = "quantity"
KIND_TOPUP = "topup"


def _id(value: Any) -> Optional[str]:
    """An object's id whether Stripe sent the id or the expanded object."""
    if isinstance(value, dict):
        return value.get("id")
    return value or None


def _at(timestamp: Any) -> Optional[datetime]:
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc) if timestamp else None


def _dig(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _invoice_subscription(invoice: dict) -> Optional[str]:
    return _id(invoice.get("subscription")) or _id(_dig(invoice, "parent", "subscription_details", "subscription"))


def _invoice_workspace(invoice: dict) -> Optional[str]:
    meta = _dig(invoice, "subscription_details", "metadata") or _dig(
        invoice, "parent", "subscription_details", "metadata") or {}
    return meta.get(META_WORKSPACE)


def _line_price(line: dict) -> Optional[str]:
    return _id(line.get("price")) or _id(_dig(line, "pricing", "price_details", "price"))


def _invoice_payment_ref(invoice: dict) -> Optional[str]:
    direct = _id(invoice.get("payment_intent"))
    if direct:
        return direct
    for payment in _dig(invoice, "payments", "data") or []:
        ref = _id(_dig(payment, "payment", "payment_intent"))
        if ref:
            return ref
    return None


def _priced_lines(invoice: dict) -> list[dict]:
    lines = [line for line in (_dig(invoice, "lines", "data") or []) if _line_price(line)]
    return sorted(lines, key=lambda line: int(line.get("amount") or 0), reverse=True)


def invoice_paid(invoice: dict) -> Any:
    subscription_id = _invoice_subscription(invoice)
    if not subscription_id:
        return Ignored("not a subscription invoice (a top-up's receipt)")
    lines = _priced_lines(invoice)
    ends = [_dig(line, "period", "end") for line in lines if _dig(line, "period", "end")]
    if not lines or not ends:
        return Ignored("a subscription invoice with no priced period")
    return PeriodPaid(
        customer_id=_id(invoice.get("customer")), invoice_id=invoice["id"], subscription_id=subscription_id,
        price_ids=tuple(_line_price(line) for line in lines), period_end=_at(max(int(e) for e in ends)),
        payment_ref=_invoice_payment_ref(invoice), amount_cents=invoice.get("amount_paid"),
        workspace_ref=_invoice_workspace(invoice), period_start=_at(_dig(lines[0], "period", "start")),
    )


def invoice_payment_paid(payment: dict) -> Any:
    ref = _id(_dig(payment, "payment", "payment_intent"))
    invoice_id = _id(payment.get("invoice"))
    if not ref or not invoice_id:
        return Ignored("an invoice payment without a payment intent")
    return InvoicePaymentPaid(invoice_id=invoice_id, payment_ref=ref, amount_cents=payment.get("amount_paid"))


def checkout_completed(session: dict) -> Any:
    meta = session.get("metadata") or {}
    workspace_ref = session.get("client_reference_id") or meta.get(META_WORKSPACE)
    if session.get("mode") == "subscription":
        return CheckoutCompleted(customer_id=_id(session.get("customer")),
                                 subscription_id=_id(session.get("subscription")), workspace_ref=workspace_ref)
    if session.get("mode") != "payment" or meta.get(META_KIND) != KIND_TOPUP:
        return Ignored("a checkout that isn't ours")
    if session.get("payment_status") != "paid":
        return Ignored("checkout completed, payment still pending")
    return TopupPaid(customer_id=_id(session.get("customer")), session_id=session["id"],
                     workspace_ref=workspace_ref, quantity=max(1, int(meta.get(META_QUANTITY) or 1)),
                     payment_ref=_id(session.get("payment_intent")), amount_cents=session.get("amount_total"))


def subscription_changed(sub: dict) -> SubscriptionChanged:
    items = _dig(sub, "items", "data") or []
    ends = [item.get("current_period_end") for item in items if item.get("current_period_end")]
    period_end = sub.get("current_period_end") or (max(ends) if ends else None)
    return SubscriptionChanged(
        customer_id=_id(sub.get("customer")), subscription_id=sub["id"], status=str(sub.get("status") or ""),
        price_ids=tuple(p for p in (_id(item.get("price")) for item in items) if p),
        current_period_end=_at(period_end), cancel_at_period_end=bool(sub.get("cancel_at_period_end")),
        workspace_ref=(sub.get("metadata") or {}).get(META_WORKSPACE),
    )


def subscription_ended(sub: dict) -> SubscriptionEnded:
    return SubscriptionEnded(customer_id=_id(sub.get("customer")), subscription_id=sub["id"])


def charge_refunded(charge: dict) -> PaymentRefunded:
    return PaymentRefunded(charge_id=charge["id"], payment_ref=_id(charge.get("payment_intent")),
                           amount_cents=int(charge.get("amount") or 0),
                           amount_refunded_cents=int(charge.get("amount_refunded") or 0))


def dispute_opened(dispute: dict) -> PaymentDisputed:
    return PaymentDisputed(dispute_id=dispute["id"], payment_ref=_id(dispute.get("payment_intent")),
                           amount_cents=int(dispute.get("amount") or 0), outcome=None)


def dispute_closed(dispute: dict) -> Any:
    status = dispute.get("status")
    if status not in ("won", "lost"):
        return Ignored(f"dispute closed as {status}")
    return PaymentDisputed(dispute_id=dispute["id"], payment_ref=_id(dispute.get("payment_intent")),
                           amount_cents=int(dispute.get("amount") or 0), outcome=status)


def payment_failed(invoice: dict) -> Any:
    if not _invoice_subscription(invoice):
        return Ignored("a failed payment that isn't a subscription's")
    return PaymentFailed(customer_id=_id(invoice.get("customer")), invoice_id=invoice["id"],
                         attempt_count=int(invoice.get("attempt_count") or 0),
                         next_attempt=_at(invoice.get("next_payment_attempt")))


_READERS: dict[str, Callable[[dict], Any]] = {
    "checkout.session.completed": checkout_completed,
    "checkout.session.async_payment_succeeded": checkout_completed,
    "invoice.paid": invoice_paid,
    "invoice_payment.paid": invoice_payment_paid,
    "invoice.payment_failed": payment_failed,
    "customer.subscription.created": subscription_changed,
    "customer.subscription.updated": subscription_changed,
    "customer.subscription.deleted": subscription_ended,
    "charge.refunded": charge_refunded,
    "charge.dispute.created": dispute_opened,
    "charge.dispute.closed": dispute_closed,
}


def translate(event: WebhookEvent) -> Any:
    """The fact one verified Stripe event states (or :class:`Ignored`)."""
    reader = _READERS.get(event.event_type)
    if reader is None:
        return Ignored(f"{event.event_type} is not handled")
    return reader(event.data)
