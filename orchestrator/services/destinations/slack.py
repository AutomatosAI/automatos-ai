"""
Slack incoming-webhook destination dispatcher (PRD-008-A Phase 6).

Posts a callback notification to a merchant-configured Slack incoming
webhook URL. No Slack OAuth, no scopes — the merchant pastes a
webhook URL from Slack's UI; we POST to it.
"""

from __future__ import annotations

import logging
import time

import httpx

from services.destinations.base import CallbackPayload, DispatchResult

logger = logging.getLogger(__name__)


def _build_slack_payload(payload: CallbackPayload, channel_label: str | None) -> dict:
    """Build the Slack incoming-webhook JSON payload."""
    fields = [
        {"title": "Name", "value": payload.name, "short": True},
        {"title": "Phone", "value": payload.phone, "short": True},
    ]
    if payload.product_context:
        fields.append({"title": "Product", "value": payload.product_context, "short": False})
    if payload.urgency:
        fields.append({"title": "Urgency", "value": payload.urgency, "short": True})
    if payload.preferred_time:
        fields.append({"title": "Preferred time", "value": payload.preferred_time, "short": True})

    body: dict = {
        "text": f"📞 New callback from {payload.site_display_name}",
        "attachments": [
            {
                "color": "#36a64f",
                "fields": fields,
                "footer": f"Request {payload.request_id} · Automatos",
            }
        ],
    }
    if channel_label:
        body["channel"] = channel_label
    return body


async def dispatch_slack_webhook(
    *,
    destination: dict,
    payload: CallbackPayload,
    http_client: httpx.AsyncClient | None = None,
) -> DispatchResult:
    """POST the callback to ``destination['url']``."""
    started = time.monotonic()
    url = destination.get("url")
    if not url or not url.startswith("https://"):
        return DispatchResult(
            success=False,
            destination_type="slack_webhook",
            latency_ms=int((time.monotonic() - started) * 1000),
            error="destination missing or non-HTTPS 'url'",
            retryable=False,
        )

    body = _build_slack_payload(payload, destination.get("channel_label"))

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=10.0)
    try:
        try:
            resp = await client.post(url, json=body)
        finally:
            if own_client:
                await client.aclose()
    except Exception as exc:  # noqa: BLE001
        return DispatchResult(
            success=False,
            destination_type="slack_webhook",
            latency_ms=int((time.monotonic() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
            retryable=True,
        )

    if resp.status_code >= 200 and resp.status_code < 300:
        return DispatchResult(
            success=True,
            destination_type="slack_webhook",
            latency_ms=int((time.monotonic() - started) * 1000),
            extra={"status": resp.status_code},
        )

    # 4xx is permanent (bad webhook URL, payload rejected); 5xx is retryable.
    return DispatchResult(
        success=False,
        destination_type="slack_webhook",
        latency_ms=int((time.monotonic() - started) * 1000),
        error=f"slack returned {resp.status_code}: {resp.text[:200]}",
        retryable=resp.status_code >= 500,
    )
