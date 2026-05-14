"""
Shopify customer-note destination dispatcher (PRD-008-A Phase 6).

For Shopify Sites only. Looks up (or creates) a Shopify customer
record by phone number, then appends the callback payload to the
customer's note field. Lets merchants see callback context inside
their existing Shopify Admin UI without integrating anything else.

Uses the merchant's Shopify access token from ``site.secrets`` —
that's why this dispatcher is Shopify-Site-only.

Implementation note: v1 calls Shopify's Admin REST API directly. A
follow-up may route through Composio for consistency with other
Shopify reads, but the direct path keeps this dispatcher independent
of Composio's tool registration cycle.
"""

from __future__ import annotations

import logging
import time

import httpx

from services.destinations.base import CallbackPayload, DispatchResult

logger = logging.getLogger(__name__)


SHOPIFY_API_VERSION = "2024-04"


def _build_note_text(payload: CallbackPayload) -> str:
    parts = [
        f"[Automatos callback request — {payload.request_id}]",
        f"Phone: {payload.phone}",
    ]
    if payload.product_context:
        parts.append(f"Product: {payload.product_context}")
    if payload.urgency:
        parts.append(f"Urgency: {payload.urgency}")
    if payload.preferred_time:
        parts.append(f"Preferred time: {payload.preferred_time}")
    return " | ".join(parts)


async def dispatch_shopify_customer_note(
    *,
    destination: dict,  # always {"type": "shopify_customer_note"} — no params
    payload: CallbackPayload,
    shop_domain: str,
    access_token: str,
    http_client: httpx.AsyncClient | None = None,
) -> DispatchResult:
    """Append a note to the matching Shopify customer (or create one)."""
    started = time.monotonic()

    if not shop_domain or not access_token:
        return DispatchResult(
            success=False,
            destination_type="shopify_customer_note",
            latency_ms=int((time.monotonic() - started) * 1000),
            error="Shopify access_token or shop_domain not available on Site",
            retryable=False,
        )

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=10.0)
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json",
    }
    note_text = _build_note_text(payload)

    try:
        try:
            # 1. Look up customer by phone
            search_url = (
                f"https://{shop_domain}/admin/api/{SHOPIFY_API_VERSION}/"
                f"customers/search.json"
            )
            search_resp = await client.get(
                search_url,
                params={"query": f"phone:{payload.phone}"},
                headers=headers,
            )
            if search_resp.status_code == 401:
                return DispatchResult(
                    success=False,
                    destination_type="shopify_customer_note",
                    latency_ms=int((time.monotonic() - started) * 1000),
                    error="Shopify access token invalid (401)",
                    retryable=False,
                )
            if search_resp.status_code >= 500:
                return DispatchResult(
                    success=False,
                    destination_type="shopify_customer_note",
                    latency_ms=int((time.monotonic() - started) * 1000),
                    error=f"Shopify search returned {search_resp.status_code}",
                    retryable=True,
                )

            customers = (search_resp.json() or {}).get("customers", [])
            if customers:
                # Append to existing customer's note
                customer = customers[0]
                customer_id = customer["id"]
                existing_note = customer.get("note") or ""
                new_note = (existing_note + "\n\n" + note_text).strip()
                update_url = (
                    f"https://{shop_domain}/admin/api/{SHOPIFY_API_VERSION}/"
                    f"customers/{customer_id}.json"
                )
                update_resp = await client.put(
                    update_url,
                    json={"customer": {"id": customer_id, "note": new_note}},
                    headers=headers,
                )
                final_status = update_resp.status_code
                op = "updated"
            else:
                # Create a new customer with the note + first/last name parsed
                first, _, last = payload.name.partition(" ")
                create_url = (
                    f"https://{shop_domain}/admin/api/{SHOPIFY_API_VERSION}/"
                    f"customers.json"
                )
                create_resp = await client.post(
                    create_url,
                    json={
                        "customer": {
                            "first_name": first or payload.name,
                            "last_name": last or "",
                            "phone": payload.phone,
                            "note": note_text,
                            "tags": "automatos-callback",
                        }
                    },
                    headers=headers,
                )
                final_status = create_resp.status_code
                op = "created"

            if 200 <= final_status < 300:
                return DispatchResult(
                    success=True,
                    destination_type="shopify_customer_note",
                    latency_ms=int((time.monotonic() - started) * 1000),
                    extra={"customer_op": op, "status": final_status},
                )
            return DispatchResult(
                success=False,
                destination_type="shopify_customer_note",
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"Shopify {op} returned {final_status}",
                retryable=final_status >= 500,
            )
        finally:
            if own_client:
                await client.aclose()
    except Exception as exc:  # noqa: BLE001
        return DispatchResult(
            success=False,
            destination_type="shopify_customer_note",
            latency_ms=int((time.monotonic() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
            retryable=True,
        )
