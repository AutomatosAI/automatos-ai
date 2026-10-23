"""What a payment provider told us, in provider-neutral terms (PRD-265 W2).

The provider adapter (``stripe_events.py`` for Stripe) turns each verified
webhook into these facts; ``apply.py`` acts on them against the plan and the
credit ledger. A merchant of record added later writes its own translator and
reuses everything after it (US-009: "without touching the ledger").

Amounts are in the smallest currency unit (cents). Times are aware UTC datetimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class WebhookEvent:
    """A verified webhook: its id, type and the object it is about."""

    provider: str
    event_id: str
    event_type: str
    livemode: bool
    data: dict


@dataclass(frozen=True)
class CheckoutCompleted:
    """A subscription checkout finished: links the workspace to its subscription.
    The plan and the month's credits follow from the invoice it paid."""

    customer_id: str
    subscription_id: Optional[str]
    workspace_ref: Optional[str]


@dataclass(frozen=True)
class TopupPaid:
    """A one-off top-up was paid."""

    customer_id: Optional[str]
    session_id: str
    workspace_ref: Optional[str]
    quantity: int
    payment_ref: Optional[str]
    amount_cents: Optional[int]


@dataclass(frozen=True)
class PeriodPaid:
    """A subscription invoice was paid: the plan it pays for and the period it
    covers. ``price_ids`` are the invoice's subscription lines, largest first."""

    customer_id: str
    invoice_id: str
    subscription_id: Optional[str]
    price_ids: tuple
    period_end: datetime
    payment_ref: Optional[str]
    amount_cents: Optional[int]
    workspace_ref: Optional[str] = None


@dataclass(frozen=True)
class InvoicePaymentPaid:
    """Which payment settled which invoice (so a later refund finds its grant)."""

    invoice_id: str
    payment_ref: str
    amount_cents: Optional[int]


@dataclass(frozen=True)
class SubscriptionChanged:
    """A subscription's state: its status, price, period and cancellation flag."""

    customer_id: str
    subscription_id: str
    status: str
    price_ids: tuple
    current_period_end: Optional[datetime]
    cancel_at_period_end: bool
    workspace_ref: Optional[str] = None


@dataclass(frozen=True)
class SubscriptionEnded:
    customer_id: str
    subscription_id: str


@dataclass(frozen=True)
class PaymentRefunded:
    """A charge was refunded, in part or in full (``amount_refunded`` is the
    running total, so replays and later partial refunds stay idempotent)."""

    charge_id: str
    payment_ref: Optional[str]
    amount_cents: int
    amount_refunded_cents: int


@dataclass(frozen=True)
class PaymentDisputed:
    """A chargeback opened (``outcome`` None), or closed ``won`` / ``lost``."""

    dispute_id: str
    payment_ref: Optional[str]
    amount_cents: int
    outcome: Optional[str]


@dataclass(frozen=True)
class PaymentFailed:
    """A renewal (or first) payment failed; the provider retries on its schedule."""

    customer_id: str
    invoice_id: str
    attempt_count: int
    next_attempt: Optional[datetime]


@dataclass(frozen=True)
class Ignored:
    """An event we don't act on (stored, marked ``ignored``)."""

    reason: str
