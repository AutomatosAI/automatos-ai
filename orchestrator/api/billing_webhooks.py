"""Stripe's webhook (PRD-265 US-010, FR-6).

``POST /api/v1/billing/webhooks/stripe`` is called by Stripe, not by a signed-in
person, so it carries no Clerk auth: its credential is the ``Stripe-Signature``
header, checked against ``STRIPE_WEBHOOK_SECRET`` over the raw body before
anything else is read. Answers:

- 200 once the event is stored and settled (or was already);
- 400 for a signature that doesn't hold (Stripe stops; an attacker learns nothing);
- 503 while billing isn't configured; 404 in the local edition;
- 500 when processing failed: the event is kept as ``failed`` and Stripe
  redelivers it (``services/billing/events.py``).

The route reads the raw body on the event loop, then does the database work in
the threadpool (``run_in_threadpool``), never blocking the loop.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

WEBHOOKS_PREFIX = "/api/v1/billing/webhooks"
STRIPE_PATH = "/stripe"
STRIPE_WEBHOOK_ROUTE = WEBHOOKS_PREFIX + STRIPE_PATH

router = APIRouter(prefix=WEBHOOKS_PREFIX, tags=["Billing"])

# A Stripe event is a few KB (at most a few hundred). The cap is counted on the wire
# (main.PATH_BODY_LIMITS → CappedBodyMiddleware), so an unauthenticated chunked body
# can't be buffered past it before the signature check runs.
STRIPE_WEBHOOK_MAX_BYTES = 1024 * 1024


def _receive_stripe(payload: bytes, signature: str) -> Dict[str, Any]:
    from config import config
    from core.database.database import SessionLocal
    from services.billing.events import receive
    from services.billing.stripe_api import StripeNotConfigured
    from services.billing.stripe_provider import stripe_provider
    from services.billing.stripe_signature import SignatureError

    try:
        provider = stripe_provider()
    except StripeNotConfigured as exc:
        raise HTTPException(status_code=503, detail="Billing isn't configured") from exc
    if not config.STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Billing isn't configured")
    db = SessionLocal()
    try:
        status = receive(db, provider, payload, signature)
    except SignatureError as exc:
        logger.warning("[billing] Stripe webhook refused: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid signature") from exc
    finally:
        db.close()
    return {"received": True, "status": status}


@router.post(STRIPE_PATH)
async def stripe_webhook(request: Request) -> Dict[str, Any]:
    """Stripe's events: verified, stored once, then applied to plans and credits."""
    from config import config

    if config.AUTH_EDITION == "local":
        raise HTTPException(status_code=404, detail="Not found")
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    return await run_in_threadpool(_receive_stripe, payload, signature)
