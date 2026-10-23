"""The payment provider interface (PRD-265 US-009).

Billing talks to its payment provider only through :class:`BillingProvider`:
create a customer, open a checkout, open the customer portal, and verify and
translate a webhook into provider-neutral facts (``facts.py``). Stripe is the one
implementation (``stripe_provider.py``); a merchant of record added later
implements the same five methods and reuses the ledger, the plans and the
webhook storage unchanged.
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
    """One hosted checkout for one workspace."""

    workspace_id: str
    customer_id: str
    kind: str  # KIND_PLAN (a subscription) or KIND_TOPUP (a one-off payment)
    price_id: str
    success_url: str
    cancel_url: str
    quantity: int = 1
    metadata: dict = field(default_factory=dict)
    consent_text: str = ""
    expires_at: Optional[datetime] = None


@dataclass(frozen=True)
class CheckoutLink:
    url: str
    session_id: str
    expires_at: Optional[datetime] = None


@dataclass(frozen=True)
class PlanChange:
    """Move an existing subscription to another price (confirmed in the portal)."""

    subscription_id: str
    price_id: str


class BillingProvider(Protocol):
    name: str

    def ensure_customer(self, *, workspace_id: str, name: str, email: Optional[str]) -> str:
        """The provider's customer id for this workspace (created once)."""

    def create_checkout(self, request: CheckoutRequest) -> CheckoutLink:
        """A hosted checkout page for ``request``."""

    def portal_link(self, *, customer_id: str, return_url: str, change: Optional[PlanChange] = None) -> str:
        """The customer portal (invoices, card, cancel), or its plan-change confirmation."""

    def parse_webhook(self, payload: bytes, signature: str) -> WebhookEvent:
        """A verified webhook; raises when the signature doesn't hold."""

    def translate(self, event: WebhookEvent) -> Any:
        """The provider-neutral fact the event states (``facts.py``)."""
