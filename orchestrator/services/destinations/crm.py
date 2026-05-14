"""
Generic CRM webhook destination dispatcher (PRD-008-A Phase 6).

POSTs the callback payload to a merchant-configured webhook URL.
Supports an optional Authorization header for the merchant's CRM
auth scheme (e.g. ``Bearer xyz``).

Native CRM connectors (HubSpot, Salesforce, Pipedrive) are out of
scope for v1 — generic webhook only. Merchants point at their own
integration (Zapier, Make.com, custom).
"""

from __future__ import annotations

import logging
import time

import httpx

from services.destinations.base import CallbackPayload, DispatchResult

logger = logging.getLogger(__name__)


def _build_crm_payload(payload: CallbackPayload) -> dict:
    """Stable JSON contract — merchants integrate against this shape."""
    return {
        "event": "automatos.callback_requested",
        "version": "1",
        "request_id": payload.request_id,
        "site": {
            "display_name": payload.site_display_name,
            "external_id": payload.site_external_id,
        },
        "lead": {
            "name": payload.name,
            "phone": payload.phone,
            "product_context": payload.product_context,
            "urgency": payload.urgency,
            "preferred_time": payload.preferred_time,
        },
    }


async def dispatch_crm_webhook(
    *,
    destination: dict,
    payload: CallbackPayload,
    http_client: httpx.AsyncClient | None = None,
) -> DispatchResult:
    started = time.monotonic()
    url = destination.get("url")
    if not url or not url.startswith("https://"):
        return DispatchResult(
            success=False,
            destination_type="crm_webhook",
            latency_ms=int((time.monotonic() - started) * 1000),
            error="destination missing or non-HTTPS 'url'",
            retryable=False,
        )

    headers = {"Content-Type": "application/json"}
    auth_header = destination.get("auth_header")
    if auth_header:
        headers["Authorization"] = auth_header

    body = _build_crm_payload(payload)

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=10.0)
    try:
        try:
            resp = await client.post(url, json=body, headers=headers)
        finally:
            if own_client:
                await client.aclose()
    except Exception as exc:  # noqa: BLE001
        return DispatchResult(
            success=False,
            destination_type="crm_webhook",
            latency_ms=int((time.monotonic() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
            retryable=True,
        )

    if 200 <= resp.status_code < 300:
        return DispatchResult(
            success=True,
            destination_type="crm_webhook",
            latency_ms=int((time.monotonic() - started) * 1000),
            extra={"status": resp.status_code},
        )

    return DispatchResult(
        success=False,
        destination_type="crm_webhook",
        latency_ms=int((time.monotonic() - started) * 1000),
        error=f"crm webhook returned {resp.status_code}: {resp.text[:200]}",
        # 401/403 = bad auth header (permanent); other 4xx = bad payload (permanent);
        # 5xx = retryable.
        retryable=resp.status_code >= 500,
    )
