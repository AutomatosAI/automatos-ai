"""PRD-265 W2 — the two guards in front of Stripe's unauthenticated webhook.

- Its body is counted on the wire (``main.PATH_BODY_LIMITS`` → ``CappedBodyMiddleware``):
  a chunked body with no ``Content-Length`` can't be buffered past 1 MB before the
  signature check runs.
- An event whose live/test mode doesn't match this environment's Stripe key is
  refused, even with a good signature: a test payment never grants real credits.
"""
from __future__ import annotations

import json
import time

import pytest

from services.billing.stripe_provider import StripeProvider, livemode_of
from services.billing.stripe_signature import SignatureError, expected_signature

SECRET = "whsec_g1"


def test_the_webhook_body_is_capped_on_the_wire():
    import main
    from api.billing_webhooks import STRIPE_WEBHOOK_MAX_BYTES, STRIPE_WEBHOOK_ROUTE

    assert STRIPE_WEBHOOK_ROUTE == "/api/v1/billing/webhooks/stripe"
    assert main.PATH_BODY_LIMITS[STRIPE_WEBHOOK_ROUTE] == STRIPE_WEBHOOK_MAX_BYTES == 1024 * 1024


@pytest.mark.parametrize("key, mode", [
    ("sk_live_abc", True), ("rk_live_abc", True), ("sk_test_abc", False), ("rk_test_abc", False), ("", None),
])
def test_the_key_says_which_mode_events_must_be(key, mode):
    assert livemode_of(key) is mode


def _signed(livemode: bool) -> tuple[bytes, str]:
    body = json.dumps({"id": "evt_guard", "type": "invoice.paid", "livemode": livemode,
                       "data": {"object": {"id": "in_guard"}}}).encode()
    now = int(time.time())
    return body, f"t={now},v1={expected_signature(body, now, SECRET)}"


def _provider(expect_livemode):
    return StripeProvider(api=None, webhook_secret=SECRET, tolerance_seconds=300, expect_livemode=expect_livemode)


def test_a_test_mode_event_is_refused_where_the_key_is_live():
    body, header = _signed(livemode=False)
    with pytest.raises(SignatureError):
        _provider(True).parse_webhook(body, header)


def test_a_live_event_is_refused_where_the_key_is_a_test_key():
    body, header = _signed(livemode=True)
    with pytest.raises(SignatureError):
        _provider(False).parse_webhook(body, header)


def test_an_event_in_the_right_mode_is_accepted():
    body, header = _signed(livemode=True)
    event = _provider(True).parse_webhook(body, header)
    assert event.event_id == "evt_guard" and event.livemode is True and event.data == {"id": "in_guard"}
