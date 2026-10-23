"""Stripe as the billing provider (PRD-265 US-009).

The customer never leaves Studio (Gerard, 23 Oct 2026):

- **Checkout** is an in-page session (``ui_mode=elements``, API 2026-09-30.endive):
  the Billing tab mounts Stripe's card fields inside our own form with the
  session's ``client_secret``, and the bank's 3-D Secure step (if any) shows over
  our page. ``mode=subscription`` sells a plan (classic billing mode, so prorations
  read as they always have), ``mode=payment`` a top-up (with a VAT invoice). Every
  session has Stripe Tax on, takes a VAT number and a billing address, and carries
  the workspace in ``client_reference_id`` and metadata.
- **Card, invoices, cancelling and plan changes** are API calls our tab makes
  (``stripe_manage.py``). There is no hosted page and no portal.
- **Webhooks**: the signature is checked (``stripe_signature.py``) before the
  body is read as JSON, then ``stripe_events.translate`` states the fact.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from services.billing.facts import WebhookEvent
from services.billing.provider import KIND_PLAN, CheckoutRequest, CheckoutSession
from services.billing.stripe_api import StripeApi, stripe_api
from services.billing.stripe_events import translate
from services.billing.stripe_manage import StripeManagement
from services.billing.stripe_signature import SignatureError, verify

PROVIDER_STRIPE = "stripe"
UI_MODE = "elements"
# Prorations computed the classic way (the invoice lines our webhook readers know).
BILLING_MODE = "classic"


class StripeProvider(StripeManagement):
    """:class:`services.billing.provider.BillingProvider` on Stripe."""

    name = PROVIDER_STRIPE

    def __init__(self, api: StripeApi, *, webhook_secret: str, tolerance_seconds: int,
                 expect_livemode: Optional[bool] = None):
        self._api = api
        self._webhook_secret = webhook_secret
        self._tolerance = tolerance_seconds
        self._expect_livemode = expect_livemode

    def ensure_customer(self, *, workspace_id: str, name: str, email: Optional[str]) -> str:
        customer = self._api.post("/v1/customers", {
            "name": name, "email": email, "metadata": {"workspace_id": workspace_id},
        }, idempotency_key=f"automatos-customer-{workspace_id}")
        return customer["id"]

    def create_checkout(self, request: CheckoutRequest) -> CheckoutSession:
        session = self._api.post("/v1/checkout/sessions", checkout_params(request))
        return CheckoutSession(session_id=session["id"], client_secret=session["client_secret"])

    def parse_webhook(self, payload: bytes, signature: str) -> WebhookEvent:
        verify(payload, signature, self._webhook_secret, tolerance_seconds=self._tolerance)
        try:
            body = json.loads(payload)
            event = WebhookEvent(provider=PROVIDER_STRIPE, event_id=str(body["id"]), event_type=str(body["type"]),
                                 livemode=bool(body.get("livemode")), data=dict(body["data"]["object"]))
        except (ValueError, KeyError, TypeError) as exc:
            raise SignatureError("a signed body that isn't a Stripe event") from exc
        if self._expect_livemode is not None and event.livemode != self._expect_livemode:
            # A secret pointing the wrong mode at this endpoint: a test payment must never
            # grant real credits, nor a live one land on a test setup.
            raise SignatureError("the event's live/test mode doesn't match this environment's Stripe key")
        return event

    def translate(self, event: WebhookEvent) -> Any:
        return translate(event)


def checkout_params(request: CheckoutRequest) -> dict:
    """The in-page Checkout Session a :class:`CheckoutRequest` becomes. Pure.
    ``return_url`` is used only by a redirect-based payment method; cards stay on
    our page (the client confirms with ``redirect: 'if_required'``)."""
    params: dict[str, Any] = {
        "ui_mode": UI_MODE,
        "mode": "subscription" if request.kind == KIND_PLAN else "payment",
        "customer": request.customer_id,
        "client_reference_id": request.workspace_id,
        "line_items": [{"price": request.price_id, "quantity": request.quantity}],
        "return_url": request.return_url,
        "automatic_tax": {"enabled": True},
        "tax_id_collection": {"enabled": True},
        "customer_update": {"address": "auto", "name": "auto"},
        "billing_address_collection": "required",
        "metadata": request.metadata,
    }
    if request.kind == KIND_PLAN:
        params["subscription_data"] = {"metadata": request.metadata, "billing_mode": {"type": BILLING_MODE}}
    else:
        params["invoice_creation"] = {"enabled": True, "invoice_data": {"metadata": request.metadata}}
        params["payment_intent_data"] = {"metadata": request.metadata}
    return params


LIVE_KEY_PREFIXES = ("sk_live_", "rk_live_")
TEST_KEY_PREFIXES = ("sk_test_", "rk_test_")


def livemode_of(secret_key: str) -> Optional[bool]:
    """True for a live key, False for a test key, None when the prefix is unknown."""
    if secret_key.startswith(LIVE_KEY_PREFIXES):
        return True
    if secret_key.startswith(TEST_KEY_PREFIXES):
        return False
    return None


def stripe_provider() -> StripeProvider:
    """This environment's Stripe provider; ``StripeNotConfigured`` without a key."""
    from config import config

    return StripeProvider(stripe_api(), webhook_secret=config.STRIPE_WEBHOOK_SECRET,
                          tolerance_seconds=int(config.STRIPE_WEBHOOK_TOLERANCE_SECONDS),
                          expect_livemode=livemode_of(config.STRIPE_SECRET_KEY))
