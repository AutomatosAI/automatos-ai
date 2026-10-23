"""The payment provider interface (PRD-265 US-009).

Billing talks to its payment provider only through :class:`BillingProvider`. The
customer never leaves Studio (Gerard, 23 Oct 2026): a checkout is a session the
Billing tab renders with the provider's card fields inside our own form, and the
card, the invoices, cancelling and plan changes are calls our tab makes — there is
no hosted page and no portal. Stripe is the one implementation
(``stripe_provider.py`` + ``stripe_manage.py``); a merchant of record added later
implements the same methods and reuses the ledger, the plans and the webhook
storage unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol

from services.billing.facts import WebhookEvent

KIND_PLAN = "plan"
KIND_TOPUP = "topup"


@dataclass(frozen=True)
class CheckoutRequest:
    """One in-page checkout for one workspace."""

    workspace_id: str
    customer_id: str
    kind: str  # KIND_PLAN (a subscription) or KIND_TOPUP (a one-off payment)
    price_id: str
    return_url: str
    quantity: int = 1
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CheckoutSession:
    """What the Billing tab mounts the card form with."""

    session_id: str
    client_secret: str


@dataclass(frozen=True)
class CardOnFile:
    brand: str
    last4: str
    exp_month: Optional[int]
    exp_year: Optional[int]


@dataclass(frozen=True)
class InvoiceLine:
    """One invoice as the Billing tab lists it."""

    id: str
    number: Optional[str]
    status: Optional[str]
    total_cents: int
    currency: str
    created: Optional[datetime]


@dataclass(frozen=True)
class PlanChangePreview:
    """``proration_date`` is the moment the price was worked out at: the change sends
    it back, so the buyer is charged exactly what they were shown."""

    amount_due_cents: int
    tax_cents: int
    currency: str
    proration_date: Optional[int] = None


@dataclass(frozen=True)
class PlanChangeResult:
    """``done`` (paid), ``requires_action`` (the bank wants 3-D Secure:
    ``client_secret`` is the payment's) or ``scheduled`` (at ``effective_at``)."""

    status: str
    client_secret: Optional[str] = None
    effective_at: Optional[datetime] = None
    schedule_id: Optional[str] = None


class BillingProvider(Protocol):
    name: str

    def ensure_customer(self, *, workspace_id: str, name: str, email: Optional[str]) -> str:
        """The provider's customer id for this workspace (created once)."""

    def create_checkout(self, request: CheckoutRequest) -> CheckoutSession:
        """An in-page checkout session for ``request``."""

    def parse_webhook(self, payload: bytes, signature: str) -> WebhookEvent:
        """A verified webhook; raises when the signature doesn't hold."""

    def translate(self, event: WebhookEvent) -> Any:
        """The provider-neutral fact the event states (``facts.py``)."""

    def card_on_file(self, *, customer_id: str, subscription_id: Optional[str]) -> Optional[CardOnFile]:
        """The card the subscription (or else the customer) is charged on."""

    def start_card_change(self, *, customer_id: str, workspace_id: str) -> str:
        """A setup session's client secret for entering a new card in our tab."""

    def finish_card_change(self, *, customer_id: str, subscription_id: Optional[str],
                           setup_id: str) -> CardOnFile:
        """Make the card a finished setup saved the default; returns it."""

    def invoices(self, *, customer_id: str) -> list[InvoiceLine]:
        """The customer's invoices, newest first."""

    def invoice_pdf(self, *, customer_id: str, invoice_id: str) -> tuple[bytes, str]:
        """One of the customer's invoices as a PDF, and its file name."""

    def set_cancel_at_period_end(self, *, subscription_id: str, cancel: bool) -> Optional[datetime]:
        """Cancel at the period end (or resume); returns the period end."""

    def preview_plan_change(self, *, customer_id: str, subscription_id: str, price_id: str) -> PlanChangePreview:
        """What an upgrade to ``price_id`` charges now (prorated, with tax)."""

    def upgrade_now(self, *, subscription_id: str, price_id: str,
                    proration_date: Optional[int] = None) -> PlanChangeResult:
        """Move to ``price_id`` now and charge the prorated difference (as of ``proration_date``)."""

    def downgrade_at_period_end(self, *, subscription_id: str, price_id: str) -> PlanChangeResult:
        """Move to ``price_id`` when the paid period ends; nothing charged now."""

    def cancel_scheduled_change(self, *, schedule_id: str) -> None:
        """Undo a downgrade waiting for the period end."""
