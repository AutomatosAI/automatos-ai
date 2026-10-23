"""Stripe as the billing provider (PRD-265 US-009).

- **Checkout** (Stripe-hosted, so 3-D Secure is Stripe's): ``mode=subscription``
  for a plan, ``mode=payment`` (with a VAT invoice) for a top-up. Every session
  has Stripe Tax on, collects a VAT number and a billing address, carries the
  workspace in ``client_reference_id`` and metadata, and shows counsel's
  withdrawal-consent text when it is configured (D-4).
- **Portal**: invoices, card and cancellation; a plan change for an existing
  subscriber opens the portal's confirm-update page, so proration and any 3-D
  Secure step are Stripe's too.
- **Webhooks**: the signature is checked (``stripe_signature.py``) before the
  body is read as JSON, then ``stripe_events.translate`` states the fact.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from services.billing.facts import WebhookEvent
from services.billing.provider import KIND_PLAN, CheckoutLink, CheckoutRequest, PlanChange
from services.billing.stripe_api import StripeApi, stripe_api
from services.billing.stripe_events import translate
from services.billing.stripe_signature import SignatureError, verify

PROVIDER_STRIPE = "stripe"
_SUBSCRIPTION_ID = re.compile(r"^sub_[A-Za-z0-9]+$")


class StripeProvider:
    """:class:`services.billing.provider.BillingProvider` on Stripe."""

    name = PROVIDER_STRIPE

    def __init__(self, api: StripeApi, *, webhook_secret: str, tolerance_seconds: int,
                 portal_configuration: str = "", expect_livemode: Optional[bool] = None):
        self._api = api
        self._webhook_secret = webhook_secret
        self._tolerance = tolerance_seconds
        self._portal_configuration = portal_configuration
        self._expect_livemode = expect_livemode

    def ensure_customer(self, *, workspace_id: str, name: str, email: Optional[str]) -> str:
        customer = self._api.post("/v1/customers", {
            "name": name, "email": email, "metadata": {"workspace_id": workspace_id},
        }, idempotency_key=f"automatos-customer-{workspace_id}")
        return customer["id"]

    def create_checkout(self, request: CheckoutRequest) -> CheckoutLink:
        session = self._api.post("/v1/checkout/sessions", checkout_params(request))
        expires = session.get("expires_at")
        return CheckoutLink(url=session["url"], session_id=session["id"],
                            expires_at=datetime.fromtimestamp(int(expires), tz=timezone.utc) if expires else None)

    def portal_link(self, *, customer_id: str, return_url: str, change: Optional[PlanChange] = None) -> str:
        params: dict[str, Any] = {"customer": customer_id, "return_url": return_url}
        if self._portal_configuration:
            params["configuration"] = self._portal_configuration
        if change is not None:
            params["flow_data"] = self._plan_change_flow(change, return_url)
        return self._api.post("/v1/billing_portal/sessions", params)["url"]

    def _plan_change_flow(self, change: PlanChange, return_url: str) -> dict:
        if not _SUBSCRIPTION_ID.match(change.subscription_id or ""):
            raise ValueError("not a Stripe subscription id")
        subscription = self._api.get(f"/v1/subscriptions/{change.subscription_id}")
        item_id = subscription["items"]["data"][0]["id"]
        return {
            "type": "subscription_update_confirm",
            "subscription_update_confirm": {
                "subscription": change.subscription_id,
                "items": [{"id": item_id, "price": change.price_id, "quantity": 1}],
            },
            "after_completion": {"type": "redirect", "redirect": {"return_url": return_url}},
        }

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
    """The Checkout Session a :class:`CheckoutRequest` becomes. Pure."""
    params: dict[str, Any] = {
        "mode": "subscription" if request.kind == KIND_PLAN else "payment",
        "customer": request.customer_id,
        "client_reference_id": request.workspace_id,
        "line_items": [{"price": request.price_id, "quantity": request.quantity}],
        "success_url": request.success_url,
        "cancel_url": request.cancel_url,
        "automatic_tax": {"enabled": True},
        "tax_id_collection": {"enabled": True},
        "customer_update": {"address": "auto", "name": "auto"},
        "billing_address_collection": "required",
        "metadata": request.metadata,
    }
    if request.kind == KIND_PLAN:
        params["subscription_data"] = {"metadata": request.metadata}
    else:
        params["invoice_creation"] = {"enabled": True, "invoice_data": {"metadata": request.metadata}}
        params["payment_intent_data"] = {"metadata": request.metadata}
    if request.consent_text:
        params["consent_collection"] = {"terms_of_service": "required"}
        params["custom_text"] = {"terms_of_service_acceptance": {"message": request.consent_text}}
    if request.expires_at is not None:
        params["expires_at"] = int(request.expires_at.timestamp())
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
                          portal_configuration=config.STRIPE_PORTAL_CONFIGURATION_ID,
                          expect_livemode=livemode_of(config.STRIPE_SECRET_KEY))
