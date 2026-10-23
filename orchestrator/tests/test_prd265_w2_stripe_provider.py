"""PRD-265 W2 v2 — ``StripeProvider`` (``services/billing/stripe_provider.py``).

``parse_webhook`` checks the signature before anything else is trusted, then
turns the raw body into a :class:`WebhookEvent`. ``checkout_params`` is the
in-page Checkout Session request shape, pure: cards never leave Studio
(``ui_mode=elements``), so there is no success/cancel URL, no consent
collection and no expiry — only ``return_url``, used by a redirect-based
method, never a card. There is no portal any more: the constructor takes no
``portal_configuration``.
"""
from __future__ import annotations

import json
import time

import httpx
import pytest

from services.billing.facts import WebhookEvent
from services.billing.provider import KIND_PLAN, KIND_TOPUP, CheckoutRequest, CheckoutSession
from services.billing.stripe_api import StripeApi
from services.billing.stripe_provider import BILLING_MODE, UI_MODE, StripeProvider, checkout_params
from services.billing.stripe_signature import SignatureError, expected_signature

SECRET = "whsec_t1"


def _api(handler) -> StripeApi:
    return StripeApi("sk_test_1", api_version="2026-09-30.endive", timeout=5.0,
                     transport=httpx.MockTransport(handler), sleep=lambda s: None)


def _provider(handler=None) -> StripeProvider:
    handler = handler or (lambda request: httpx.Response(200, json={}))
    return StripeProvider(_api(handler), webhook_secret=SECRET, tolerance_seconds=300)


def _signed(body: dict) -> tuple[bytes, str]:
    payload = json.dumps(body).encode()
    now = int(time.time())  # signed when the test runs: parse_webhook refuses anything 5 minutes old
    sig = expected_signature(payload, now, SECRET)
    return payload, f"t={now},v1={sig}"


# --------------------------------------------------------------------------- #
# parse_webhook
# --------------------------------------------------------------------------- #


def test_parse_webhook_of_a_signed_valid_body_returns_the_object_as_data():
    body = {"id": "evt_1", "type": "invoice.paid", "livemode": False, "data": {"object": {"id": "in_1"}}}
    payload, header = _signed(body)
    event = _provider().parse_webhook(payload, header)
    assert event == WebhookEvent(provider="stripe", event_id="evt_1", event_type="invoice.paid",
                                 livemode=False, data={"id": "in_1"})


def test_parse_webhook_of_a_signed_body_that_isnt_an_event_raises_signature_error():
    payload, header = _signed({"not": "an event"})
    with pytest.raises(SignatureError):
        _provider().parse_webhook(payload, header)


def test_parse_webhook_with_a_bad_signature_raises_signature_error():
    body = {"id": "evt_1", "type": "invoice.paid", "data": {"object": {}}}
    payload = json.dumps(body).encode()
    with pytest.raises(SignatureError):
        _provider().parse_webhook(payload, "t=1,v1=deadbeef")


# --------------------------------------------------------------------------- #
# checkout_params
# --------------------------------------------------------------------------- #


def _request(**overrides) -> CheckoutRequest:
    base = dict(workspace_id="ws-1", customer_id="cus_1", kind=KIND_PLAN, price_id="price_pro",
               return_url="https://app.automatos.ai/settings?tab=billing&checkout=return")
    base.update(overrides)
    return CheckoutRequest(**base)


def test_checkout_params_is_an_in_page_elements_session_with_a_return_url():
    params = checkout_params(_request())
    assert params["ui_mode"] == UI_MODE == "elements"
    assert params["return_url"] == "https://app.automatos.ai/settings?tab=billing&checkout=return"


def test_checkout_params_never_carries_hosted_checkout_fields():
    params = checkout_params(_request(metadata={"workspace_id": "ws-1"}))
    for field in ("success_url", "cancel_url", "consent_collection", "custom_text", "expires_at"):
        assert field not in params


def test_checkout_params_for_a_plan_is_a_subscription_in_classic_billing_mode():
    params = checkout_params(_request(metadata={"workspace_id": "ws-1"}))
    assert params["mode"] == "subscription"
    assert params["subscription_data"]["metadata"] == {"workspace_id": "ws-1"}
    assert params["subscription_data"]["billing_mode"] == {"type": BILLING_MODE} == {"type": "classic"}
    assert "invoice_creation" not in params
    assert "payment_intent_data" not in params


def test_checkout_params_for_a_topup_is_a_payment_with_an_invoice_and_metadata_on_the_intent():
    params = checkout_params(_request(kind=KIND_TOPUP, metadata={"kind": "topup"}))
    assert params["mode"] == "payment"
    assert params["invoice_creation"] == {"enabled": True, "invoice_data": {"metadata": {"kind": "topup"}}}
    assert params["payment_intent_data"]["metadata"] == {"kind": "topup"}
    assert "subscription_data" not in params


def test_checkout_params_always_sets_tax_and_address_collection():
    params = checkout_params(_request())
    assert params["automatic_tax"] == {"enabled": True}
    assert params["tax_id_collection"] == {"enabled": True}
    assert params["customer_update"] == {"address": "auto", "name": "auto"}
    assert params["billing_address_collection"] == "required"
    assert params["client_reference_id"] == "ws-1"
    assert params["line_items"] == [{"price": "price_pro", "quantity": 1}]


# --------------------------------------------------------------------------- #
# create_checkout
# --------------------------------------------------------------------------- #


def test_create_checkout_returns_the_sessions_client_secret():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "cs_1", "client_secret": "cs_1_secret_abc"})

    session = _provider(handler).create_checkout(_request())
    assert session == CheckoutSession(session_id="cs_1", client_secret="cs_1_secret_abc")


def test_create_checkout_posts_to_the_checkout_sessions_endpoint():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "cs_1", "client_secret": "cs_1_secret"})

    _provider(handler).create_checkout(_request())
    (request,) = seen
    assert request.method == "POST" and request.url.path == "/v1/checkout/sessions"


# --------------------------------------------------------------------------- #
# the constructor takes no portal configuration — there is no portal
# --------------------------------------------------------------------------- #


def test_the_constructor_has_no_portal_configuration_parameter():
    import inspect

    params = inspect.signature(StripeProvider.__init__).parameters
    assert "portal_configuration" not in params
