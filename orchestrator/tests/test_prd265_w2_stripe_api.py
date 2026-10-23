"""PRD-265 W2 — the Stripe REST client (``services/billing/stripe_api.py``).

``encode_form`` is Stripe's bracket form encoding, pure. ``StripeApi`` sends the
auth header, the pinned ``Stripe-Version``, and a form body; it retries a POST
only when it carries an idempotency key, retries a GET unconditionally, and
turns a 4xx into a :class:`StripeError` carrying Stripe's own message and code.
"""
from __future__ import annotations

from typing import Callable
from urllib.parse import parse_qsl

import httpx
import pytest

from services.billing.stripe_api import (
    StripeApi,
    StripeError,
    StripeNotConfigured,
    encode_form,
)

# --------------------------------------------------------------------------- #
# encode_form
# --------------------------------------------------------------------------- #


def test_encode_form_brackets_nested_dicts_and_lists():
    pairs = encode_form({"line_items": [{"price": "price_1", "quantity": 2}]})
    assert ("line_items[0][price]", "price_1") in pairs
    assert ("line_items[0][quantity]", "2") in pairs


def test_encode_form_renders_booleans_as_true_false():
    pairs = dict(encode_form({"automatic_tax": {"enabled": True}, "x": {"enabled": False}}))
    assert pairs["automatic_tax[enabled]"] == "true"
    assert pairs["x[enabled]"] == "false"


def test_encode_form_drops_none_and_stringifies_ints():
    pairs = dict(encode_form({"customer": None, "quantity": 3, "name": "Barber & Co"}))
    assert "customer" not in pairs
    assert pairs["quantity"] == "3"
    assert pairs["name"] == "Barber & Co"


def test_encode_form_of_none_is_empty():
    assert encode_form(None) == []


# --------------------------------------------------------------------------- #
# StripeApi — transport, headers, retries
# --------------------------------------------------------------------------- #


def _api(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> StripeApi:
    return StripeApi("sk_test_123", api_version="2025-03-31.basil", timeout=5.0,
                     transport=httpx.MockTransport(handler), sleep=lambda s: None, **kwargs)


def _ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body)


def test_a_missing_secret_key_raises_not_configured():
    with pytest.raises(StripeNotConfigured):
        StripeApi("", api_version="2025-03-31.basil", timeout=5.0)


def test_post_sends_bearer_auth_pinned_version_and_form_body():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"id": "cus_1"})

    _api(handler).post("/v1/customers", {"name": "Barber & Co"})
    (request,) = seen
    assert request.headers["authorization"] == "Bearer sk_test_123"
    assert request.headers["stripe-version"] == "2025-03-31.basil"
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert dict(parse_qsl(request.content.decode())) == {"name": "Barber & Co"}


def test_post_carries_the_idempotency_key_only_when_given():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"id": "cus_1"})

    api = _api(handler)
    api.post("/v1/customers", {}, idempotency_key="automatos-customer-1")
    api.post("/v1/customers", {})
    assert seen[0].headers["idempotency-key"] == "automatos-customer-1"
    assert "idempotency-key" not in seen[1].headers


def test_get_encodes_params_in_the_query_string():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"id": "sub_1"})

    _api(handler).get("/v1/subscriptions/sub_1", {"expand": ["items"]})
    assert "expand%5B0%5D=items" in str(seen[0].url) or "expand[0]=items" in str(seen[0].url)


def test_a_4xx_raises_stripe_error_with_stripes_message_and_is_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(402, json={"error": {"message": "Your card was declined.", "code": "card_declined"}})

    with pytest.raises(StripeError) as exc_info:
        _api(handler).post("/v1/customers", {}, idempotency_key="k-1")
    assert exc_info.value.status == 402
    assert exc_info.value.message == "Your card was declined."
    assert exc_info.value.code == "card_declined"
    assert len(calls) == 1


def test_a_503_on_a_post_with_a_key_is_retried_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "service unavailable"}})
        return _ok({"id": "cus_1"})

    result = _api(handler).post("/v1/customers", {}, idempotency_key="k-1")
    assert result == {"id": "cus_1"}
    assert len(calls) == 2


def test_a_503_on_a_post_without_a_key_is_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, json={"error": {"message": "service unavailable"}})

    with pytest.raises(StripeError):
        _api(handler).post("/v1/customers", {})
    assert len(calls) == 1


def test_a_transport_error_on_get_is_retried_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) < 2:
            raise httpx.ConnectError("boom")
        return _ok({"id": "sub_1"})

    result = _api(handler).get("/v1/subscriptions/sub_1")
    assert result == {"id": "sub_1"}
    assert len(calls) == 2


def test_a_transport_error_that_never_clears_is_raised_after_the_last_attempt():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("boom")

    with pytest.raises(httpx.TransportError):
        _api(handler).get("/v1/subscriptions/sub_1")
    assert len(calls) == 3  # MAX_ATTEMPTS


def test_a_409_is_retried_on_a_post_with_a_key_until_it_runs_out_of_attempts():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(409, json={"error": {"message": "lock contention"}})

    with pytest.raises(StripeError) as exc_info:
        _api(handler).post("/v1/customers", {}, idempotency_key="k-1")
    assert exc_info.value.status == 409
    assert len(calls) == 3  # MAX_ATTEMPTS, then raised


def test_stripe_api_factory_raises_not_configured_without_a_key(monkeypatch):
    from config import config
    from services.billing.stripe_api import stripe_api

    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "")
    with pytest.raises(StripeNotConfigured):
        stripe_api()
