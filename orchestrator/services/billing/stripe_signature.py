"""Stripe webhook signatures (PRD-265 US-010, FR-6).

Stripe signs ``"<timestamp>.<raw body>"`` with HMAC-SHA256 under the endpoint's
signing secret and sends ``Stripe-Signature: t=<ts>,v1=<hex>[,v1=<hex>…]`` (two
``v1`` values while a secret is being rolled). A webhook is accepted only when one
``v1`` matches and its timestamp is within the tolerance, so a captured request
can't be replayed later. The comparison is constant-time.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional


class SignatureError(Exception):
    """The webhook is not from Stripe, or is too old to trust."""


def _parse(header: str) -> tuple[Optional[int], list[str]]:
    timestamp: Optional[int] = None
    signatures: list[str] = []
    for part in (header or "").split(","):
        key, _, value = part.strip().partition("=")
        if key == "t" and value.isdigit():
            timestamp = int(value)
        elif key == "v1" and value:
            signatures.append(value)
    return timestamp, signatures


def expected_signature(payload: bytes, timestamp: int, secret: str) -> str:
    """The ``v1`` value Stripe sends for ``payload`` at ``timestamp``."""
    signed = str(timestamp).encode() + b"." + payload
    return hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()


def verify(payload: bytes, header: str, secret: str, *, tolerance_seconds: int,
           now: Optional[float] = None) -> None:
    """Raise :class:`SignatureError` unless ``header`` signs ``payload`` recently."""
    if not secret:
        raise SignatureError("no webhook signing secret is configured")
    timestamp, signatures = _parse(header)
    if timestamp is None or not signatures:
        raise SignatureError("the Stripe-Signature header is missing or malformed")
    current = time.time() if now is None else now
    if abs(current - timestamp) > tolerance_seconds:
        raise SignatureError("the signature's timestamp is outside the tolerance")
    expected = expected_signature(payload, timestamp, secret)
    if not any(hmac.compare_digest(expected, candidate) for candidate in signatures):
        raise SignatureError("no signature matches")
