"""PRD-265 W2 — ``StripeProvider`` (``services/billing/stripe_provider.py``).

``parse_webhook`` checks the signature before anything else is trusted, then
turns the raw body into a :class:`WebhookEvent`. ``checkout_params`` is Stripe
Checkout's own request shape, pure. ``portal_link`` with a :class:`PlanChange`
reads the subscription's item id before asking Stripe to open the confirm page.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import httpx
import pytest

from services.billing.facts import WebhookEvent
from services.billing.provider import KIND_PLAN, KIND_TOPUP, CheckoutRequest, PlanChange
from services.billing.stripe_api import StripeApi
from services.billing.stripe_provider import StripeProvider, checkout_params
from services.billing.stripe_signature import SignatureError, expected_signature

SECRET = "whsec_test"


def _api(handler) -> StripeApi:
    return StripeApi("sk_test", api_version="2025-03-31.basil", timeout=5.0,
                     transport=httpx.MockTransport(handler), sleep=lambda s: None)


def _provider(handler=None, **kwargs) -> StripeProvider:
    handler = handler or (lambda request: httpx.Response(200, json={}))
    return StripeProvider(_api(handler), webhook_secret=SECRET, tolerance_seconds=300, **kwargs)


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
               success_url="https://app/success", cancel_url="https://app/cancel")
    base.update(overrides)
    return CheckoutRequest(**base)


def test_checkout_params_for_a_plan_is_a_subscription_with_metadata_on_the_subscription():
    params = checkout_params(_request(metadata={"workspace_id": "ws-1"}))
    assert params["mode"] == "subscription"
    assert params["subscription_data"]["metadata"] == {"workspace_id": "ws-1"}
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


def test_checkout_params_adds_consent_text_only_when_given():
    with_consent = checkout_params(_request(consent_text="You may withdraw consent at any time."))
    assert with_consent["consent_collection"] == {"terms_of_service": "required"}
    assert with_consent["custom_text"]["terms_of_service_acceptance"]["message"] == (
        "You may withdraw consent at any time.")
    without_consent = checkout_params(_request())
    assert "consent_collection" not in without_consent
    assert "custom_text" not in without_consent


def test_checkout_params_sets_expires_at_as_a_unix_int_when_given():
    expires = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    params = checkout_params(_request(expires_at=expires))
    assert params["expires_at"] == int(expires.timestamp())
    assert "expires_at" not in checkout_params(_request())


# --------------------------------------------------------------------------- #
# portal_link
# --------------------------------------------------------------------------- #


def test_portal_link_with_a_plan_change_gets_the_subscription_then_sends_the_update_confirm_flow():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"items": {"data": [{"id": "si_1"}]}})
        return httpx.Response(200, json={"url": "https://billing.stripe.com/session/1"})

    change = PlanChange(subscription_id="sub_1", price_id="price_pro")
    url = _provider(handler).portal_link(customer_id="cus_1", return_url="https://app/return", change=change)
    assert url == "https://billing.stripe.com/session/1"
    get_request, post_request = seen
    assert get_request.method == "GET" and "/v1/subscriptions/sub_1" in str(get_request.url)
    body = post_request.content.decode()
    assert "flow_data%5Btype%5D=subscription_update_confirm" in body
    assert "si_1" in body and "price_pro" in body


def test_portal_link_with_a_malformed_subscription_id_raises_before_any_http_call():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    change = PlanChange(subscription_id="not-a-stripe-id", price_id="price_pro")
    with pytest.raises(ValueError):
        _provider(handler).portal_link(customer_id="cus_1", return_url="https://app/return", change=change)
    assert seen == []


def test_portal_link_sends_configuration_only_when_configured():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"url": "https://billing.stripe.com/session/1"})

    _provider(handler, portal_configuration="bpc_1").portal_link(customer_id="cus_1", return_url="https://app/return")
    assert "configuration=bpc_1" in seen[0].content.decode()

    seen.clear()
    _provider(handler).portal_link(customer_id="cus_1", return_url="https://app/return")
    assert "configuration" not in seen[0].content.decode()
