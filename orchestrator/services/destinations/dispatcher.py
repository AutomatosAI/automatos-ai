"""
Destination dispatch orchestrator (PRD-008-A Phase 6).

Routes a single callback to all of a Site's configured destinations
in parallel, with retry-with-exponential-backoff on retryable
failures. Every attempt writes a row to ``widget_event_log`` so the
dashboard can surface delivery state.

For v1 the queue is in-process via ``asyncio.create_task`` (called
from the callback endpoint). The same orchestrator function will work
unchanged once a Redis-backed worker is wired in a follow-up — the
callsite swaps from ``create_task`` to ``enqueue_redis``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable
from uuid import UUID

from sqlalchemy.orm import Session

from core.database.database import SessionLocal
from core.models.sites import Site
from modules.widgets.telemetry import log_widget_event

from services.destinations.base import CallbackPayload, DispatchResult
from services.destinations.crm import dispatch_crm_webhook
from services.destinations.email import dispatch_email
from services.destinations.shopify_note import dispatch_shopify_customer_note
from services.destinations.slack import dispatch_slack_webhook

logger = logging.getLogger(__name__)


MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (0, 5, 15)  # before attempts 1, 2, 3


# Shopify customer-note dispatcher needs extra context (shop domain +
# access token), so its signature differs. We wrap it to fit the common
# DispatcherFn shape used by the orchestrator.

DispatcherFn = Callable[..., Awaitable[DispatchResult]]


def _resolve_dispatcher(
    destination_type: str,
    *,
    shop_domain: str | None = None,
    access_token: str | None = None,
) -> DispatcherFn | None:
    if destination_type == "email":
        return dispatch_email
    if destination_type == "slack_webhook":
        return dispatch_slack_webhook
    if destination_type == "crm_webhook":
        return dispatch_crm_webhook
    if destination_type == "shopify_customer_note":
        async def _wrapped(*, destination, payload):
            return await dispatch_shopify_customer_note(
                destination=destination,
                payload=payload,
                shop_domain=shop_domain or "",
                access_token=access_token or "",
            )
        return _wrapped
    return None


async def dispatch_one_destination(
    *,
    db: Session,
    site_id: UUID,
    session_id: str,
    request_id: str,
    destination: dict,
    payload: CallbackPayload,
    shop_domain: str | None = None,
    access_token: str | None = None,
) -> DispatchResult:
    """Try a single destination up to MAX_ATTEMPTS, with backoff between
    retryable failures. Writes a widget_event_log row for every attempt.

    Returns the final DispatchResult (success or terminal failure).
    """
    destination_type = destination.get("type") or "unknown"
    dispatcher = _resolve_dispatcher(
        destination_type, shop_domain=shop_domain, access_token=access_token
    )
    if dispatcher is None:
        result = DispatchResult(
            success=False,
            destination_type=destination_type,
            latency_ms=0,
            error=f"unknown destination type: {destination_type}",
            retryable=False,
        )
        await log_widget_event(
            db,
            site_id=site_id,
            event_type="callback_failed",
            session_id=session_id,
            event_data={
                "request_id": request_id,
                "destination_type": destination_type,
                "error": result.error,
                "attempt": 1,
                "permanent": True,
            },
        )
        return result

    last: DispatchResult | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            await asyncio.sleep(BACKOFF_SECONDS[attempt - 1])

        result = await dispatcher(destination=destination, payload=payload)
        last = result

        if result.success:
            await log_widget_event(
                db,
                site_id=site_id,
                event_type="callback_delivered",
                session_id=session_id,
                event_data={
                    "request_id": request_id,
                    "destination_type": destination_type,
                    "latency_ms": result.latency_ms,
                    "attempt": attempt,
                    **result.extra,
                },
            )
            return result

        # Failed — log this attempt
        await log_widget_event(
            db,
            site_id=site_id,
            event_type="callback_failed",
            session_id=session_id,
            event_data={
                "request_id": request_id,
                "destination_type": destination_type,
                "error": result.error,
                "attempt": attempt,
                "retryable": result.retryable,
                "permanent": (
                    not result.retryable or attempt == MAX_ATTEMPTS
                ),
            },
        )

        # Permanent failure — bail without burning more attempts
        if not result.retryable:
            return result

    # Exhausted attempts
    return last  # type: ignore[return-value]


async def dispatch_callback_for_site(
    *,
    site_id: UUID,
    session_id: str,
    request_id: str,
    payload: CallbackPayload,
    destinations: list[dict],
    shop_domain: str | None = None,
    access_token: str | None = None,
) -> list[DispatchResult]:
    """Dispatch a callback to all of a Site's destinations in parallel.

    Creates a fresh DB session — caller is the request lifecycle, which
    has already returned 202 to the widget. Never raises.
    """
    db: Session = SessionLocal()
    try:
        tasks = [
            dispatch_one_destination(
                db=db,
                site_id=site_id,
                session_id=session_id,
                request_id=request_id,
                destination=dest,
                payload=payload,
                shop_domain=shop_domain,
                access_token=access_token,
            )
            for dest in destinations
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    except Exception:  # noqa: BLE001 — fire-and-forget contract
        logger.exception(
            "dispatch_callback_for_site crashed for request_id=%s site_id=%s",
            request_id, site_id,
        )
        results = []
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001
            pass
    return results


def enqueue_callback_dispatch(
    *,
    site: Site,
    session_id: str,
    request_id: str,
    payload: CallbackPayload,
    destinations: list[dict],
) -> None:
    """Fire-and-forget submission point used by the callback endpoint.

    v1: in-process via ``asyncio.create_task``. v2 swap-in: push to a
    Redis queue + worker. The endpoint's contract doesn't change.
    """
    secrets = site.secrets or {}
    asyncio.create_task(
        dispatch_callback_for_site(
            site_id=site.id,
            session_id=session_id,
            request_id=request_id,
            payload=payload,
            destinations=destinations,
            shop_domain=site.external_id if site.type == "shopify" else None,
            access_token=secrets.get("shopify_access_token") if site.type == "shopify" else None,
        )
    )
