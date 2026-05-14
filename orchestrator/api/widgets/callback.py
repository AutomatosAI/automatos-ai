"""
Widget Callback Endpoint (PRD-008-A Feature B).

POST /api/widgets/callback — accepts a callback request from the
storefront chat widget, validates + dedups + rate-limits, writes a
``callback_requested`` event to ``widget_event_log``, and returns a
202 with the SLA phrase the widget surfaces to the shopper.

Destination delivery (email/Slack/CRM/Shopify-customer-note) is fanned
out asynchronously by the Phase 6 dispatcher, which consumes
``callback_requested`` events from a Redis queue. This endpoint never
blocks on destination latency.

GDPR: phone numbers are NEVER persisted in Automatos. A salted hash
goes into ``widget_event_log`` for the 5-minute idempotency window;
the plaintext is forwarded to merchant destinations only.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.database.database import get_db

from api.widgets.auth import WidgetAuthContext, widget_auth
from modules.widgets.telemetry import log_widget_event
from services.callback import (
    check_rate_limits,
    compute_eta_phrase,
    compute_phone_hash,
    find_recent_duplicate,
    is_valid_phone,
    new_request_id,
    normalise_phone,
)
from services.destinations.base import CallbackPayload
from services.destinations.dispatcher import enqueue_callback_dispatch
from services.sites import get_default_site

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Widget Callback"])


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------

class CallbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    phone: str = Field(..., max_length=32)
    name: str = Field(..., min_length=1, max_length=100)
    product_context: Optional[str] = Field(default=None, max_length=255)
    urgency: Optional[str] = Field(default=None, max_length=32)
    preferred_time: Optional[str] = Field(default=None, max_length=64)


class CallbackAccepted(BaseModel):
    accepted: bool
    request_id: str
    eta_phrase: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/callback", response_model=CallbackAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_callback(
    body: CallbackRequest,
    auth: WidgetAuthContext = Depends(widget_auth),
    db: Session = Depends(get_db),
) -> CallbackAccepted:
    """Accept a callback request from the storefront widget.

    Returns 202 + SLA phrase the widget shows to the shopper.
    Destination delivery happens asynchronously (Phase 6).
    """
    # --- 1. Resolve Site for this workspace (Phase 1 migration backfills) ---
    site = get_default_site(db, auth.workspace_id)
    if site is None:
        # Migration not yet run for this workspace — should not happen
        # post-deploy but kept defensive during the transition window.
        logger.error(
            "callback rejected: no Site for workspace=%s — migration not run?",
            auth.workspace_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Callback unavailable for this workspace — contact support.",
        )

    # --- 2. Validate phone format (E.164) -----------------------------------
    phone = normalise_phone(body.phone)
    if not is_valid_phone(phone):
        raise HTTPException(
            status_code=400,
            detail="phone must be in E.164 format (e.g. +447700900123)",
        )

    callback_settings = (site.settings or {}).get("callback", {})

    # --- 3. Feature-enabled gate -------------------------------------------
    if not callback_settings.get("enabled", False):
        raise HTTPException(
            status_code=403,
            detail="Callback feature not enabled for this Site.",
        )

    # --- 4. Compute phone hash (GDPR — never persist plaintext) ------------
    phone_hash = compute_phone_hash(phone, site.id)

    # --- 5. Idempotency check ----------------------------------------------
    existing_request_id = find_recent_duplicate(
        db,
        site_id=site.id,
        session_id=body.session_id,
        phone_hash=phone_hash,
    )
    if existing_request_id:
        # Same submission within 5 min — return same request_id, same phrase
        eta_phrase = compute_eta_phrase(
            callback_settings, product_context=body.product_context
        )
        return CallbackAccepted(
            accepted=True,
            request_id=existing_request_id,
            eta_phrase=eta_phrase,
        )

    # --- 6. Rate limit check -----------------------------------------------
    per_site_cap = int(
        callback_settings.get("rate_limit_per_hour", 100)
    )
    decision = check_rate_limits(
        db,
        site_id=site.id,
        session_id=body.session_id,
        per_site_hourly_cap=per_site_cap,
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: {decision.reason}",
            headers=(
                {"Retry-After": str(decision.retry_after_seconds)}
                if decision.retry_after_seconds
                else None
            ),
        )

    # --- 7. Allocate request_id + write telemetry --------------------------
    request_id = new_request_id()
    eta_phrase = compute_eta_phrase(
        callback_settings, product_context=body.product_context
    )

    await log_widget_event(
        db,
        site_id=site.id,
        event_type="callback_requested",
        session_id=body.session_id,
        event_data={
            "request_id": request_id,
            "phone_hash": phone_hash,
            "name": body.name,
            "product_context": body.product_context,
            "urgency": body.urgency,
            "preferred_time": body.preferred_time,
            "destinations_planned": [
                d.get("type") for d in callback_settings.get("destinations", [])
            ],
        },
    )

    # --- 8. Fan-out destination dispatch (Phase 6) -------------------------
    destinations = callback_settings.get("destinations", [])
    if destinations:
        enqueue_callback_dispatch(
            site=site,
            session_id=body.session_id,
            request_id=request_id,
            payload=CallbackPayload(
                request_id=request_id,
                name=body.name,
                phone=phone,
                product_context=body.product_context,
                urgency=body.urgency,
                preferred_time=body.preferred_time,
                site_display_name=site.display_name,
                site_external_id=site.external_id,
            ),
            destinations=destinations,
        )
        logger.info(
            "callback %s queued for site=%s — %d destination(s)",
            request_id, site.id, len(destinations),
        )
    else:
        logger.warning(
            "callback %s accepted for site=%s with NO destinations configured — "
            "request will be lost without merchant action",
            request_id, site.id,
        )

    return CallbackAccepted(
        accepted=True,
        request_id=request_id,
        eta_phrase=eta_phrase,
    )
