"""PRD-265 W2 — Stripe webhook signatures (``services/billing/stripe_signature.py``).

Stripe signs ``"<timestamp>.<raw body>"`` with HMAC-SHA256; the header carries
``t=<ts>,v1=<hex>[,v1=<hex>…]`` while a secret is being rolled. A webhook is
accepted only when one ``v1`` matches, within the tolerance, by a constant-time
comparison.
"""
from __future__ import annotations

import pytest

from services.billing.stripe_signature import SignatureError, expected_signature, verify

SECRET = "whsec_test_secret"
PAYLOAD = b'{"id": "evt_1", "type": "invoice.paid"}'
NOW = 1_700_000_000.0


def _header(*sigs: str, timestamp: float = NOW) -> str:
    parts = [f"t={int(timestamp)}"] + [f"v1={sig}" for sig in sigs]
    return ",".join(parts)


def test_a_correct_header_verifies():
    sig = expected_signature(PAYLOAD, int(NOW), SECRET)
    verify(PAYLOAD, _header(sig), SECRET, tolerance_seconds=300, now=NOW)  # does not raise


def test_a_wrong_secret_is_refused():
    sig = expected_signature(PAYLOAD, int(NOW), "whsec_other")
    with pytest.raises(SignatureError):
        verify(PAYLOAD, _header(sig), SECRET, tolerance_seconds=300, now=NOW)


def test_a_tampered_body_is_refused():
    sig = expected_signature(PAYLOAD, int(NOW), SECRET)
    with pytest.raises(SignatureError):
        verify(PAYLOAD + b"tampered", _header(sig), SECRET, tolerance_seconds=300, now=NOW)


def test_a_missing_header_is_refused():
    with pytest.raises(SignatureError):
        verify(PAYLOAD, "", SECRET, tolerance_seconds=300, now=NOW)


def test_a_header_with_no_v1_at_all_is_refused():
    with pytest.raises(SignatureError):
        verify(PAYLOAD, f"t={int(NOW)}", SECRET, tolerance_seconds=300, now=NOW)


def test_a_timestamp_older_than_the_tolerance_is_refused():
    sig = expected_signature(PAYLOAD, int(NOW) - 301, SECRET)
    with pytest.raises(SignatureError):
        verify(PAYLOAD, _header(sig, timestamp=NOW - 301), SECRET, tolerance_seconds=300, now=NOW)


def test_an_empty_secret_is_refused_even_with_a_correct_looking_signature():
    sig = expected_signature(PAYLOAD, int(NOW), "")
    with pytest.raises(SignatureError):
        verify(PAYLOAD, _header(sig), "", tolerance_seconds=300, now=NOW)


def test_a_header_with_two_v1_values_verifies_when_the_second_matches():
    wrong = expected_signature(PAYLOAD, int(NOW), "whsec_rolling_out")
    right = expected_signature(PAYLOAD, int(NOW), SECRET)
    verify(PAYLOAD, _header(wrong, right), SECRET, tolerance_seconds=300, now=NOW)  # does not raise


def test_expected_signature_is_deterministic_for_the_same_inputs():
    assert expected_signature(PAYLOAD, 123, SECRET) == expected_signature(PAYLOAD, 123, SECRET)
    assert expected_signature(PAYLOAD, 123, SECRET) != expected_signature(PAYLOAD, 124, SECRET)
