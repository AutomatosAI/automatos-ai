"""
Widget callback dispatch (PRD-008-A.1; durable since PRD-276 Wave 1b).

Routes a callback to a Site's configured destinations through
``channels.sender.send_to_channel``, the function heartbeats use to reach
Telegram / Slack / a webhook. No parallel destination zoo. Every attempt
writes a ``widget_event_log`` row so the dashboard can surface delivery state.

A shopper's callback is delivered by ``widget_destination.deliver`` jobs
(``services/destinations/delivery_jobs.py``), one per destination, retried
with backoff and never lost to a restart. The dashboard's test send
(``dispatch_callback_for_site``) makes one attempt per destination and
answers at once.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.database.database import SessionLocal
from core.utils.exception_telemetry import record_error
from modules.widgets.telemetry import log_widget_event

from services.destinations.base import CALLBACK_PLATFORMS, CallbackPayload, DispatchResult

logger = logging.getLogger(__name__)


def _resolve_platform(destination: dict) -> str:
    """Return the destination's platform, accepting both the new and
    legacy shapes:

    - ``{"platform": "telegram"}``                       — preferred
    - ``{"type": "channel_connection", "platform": …}``  — legacy
    """
    return str(destination.get("platform") or "").strip().lower()


def destination_type(destination: dict) -> str:
    """The destination's name in the event log and the error tile."""
    return _resolve_platform(destination) or destination.get("type") or "unknown"


def _render_callback_text(payload: CallbackPayload) -> str:
    """Format a callback for any text-based channel (Slack/Telegram/WA)."""
    lines = [
        f"🔔 Callback request — {payload.site_display_name}",
        "",
        f"Customer: {payload.name}",
        f"Phone: {payload.phone}",
    ]
    if payload.product_context:
        lines.append(f"Topic: {payload.product_context}")
    if payload.urgency:
        lines.append(f"Urgency: {payload.urgency}")
    if payload.preferred_time:
        lines.append(f"Preferred time: {payload.preferred_time}")
    lines.append("")
    lines.append(f"Request ID: {payload.request_id}")
    return "\n".join(lines)


def _target_for(platform: str, destination: dict) -> Tuple[Optional[str], Optional[str]]:
    """(target, error): the destination row's explicit target for the platform, if it names one.

    Slack accepts an explicit channel_id; Telegram an explicit chat_id (the
    workspace may never have received a /start, so the auto-captured default
    isn't available); a webhook needs its URL.
    """
    if not platform:
        return None, "destination missing 'platform' field"
    if platform not in CALLBACK_PLATFORMS:
        return None, f"unsupported platform {platform!r}"
    field = {"slack": "channel_id", "telegram": "chat_id", "webhook": "webhook_url"}.get(platform)
    target = (str(destination.get(field) or "").strip() or None) if field else None
    if platform == "webhook" and not target:
        return None, "webhook destination missing 'webhook_url'"
    return target, None


async def dispatch_via_channel(
    *,
    destination: dict,
    payload: CallbackPayload,
    db: Session,
    workspace_id: UUID,
    delivery_id: Optional[str] = None,
) -> DispatchResult:
    """One attempt to send a callback through the unified ``channels.sender``.

    Every per-platform detail (URLs, auth, error semantics) lives in the
    driver. This function shapes the input and output. ``delivery_id`` names
    the delivery across its retries; a webhook receives it as
    ``X-Automatos-Delivery``.

    Accepted destination shapes::

        {"platform": "telegram"}
        {"platform": "slack",   "channel_id":  "C01ABC..."}
        {"platform": "whatsapp"}
        {"platform": "webhook", "webhook_url": "https://…"}
    """
    started_ms = time.monotonic()
    platform = _resolve_platform(destination)
    target, refused = _target_for(platform, destination)
    if refused:
        return DispatchResult(success=False, destination_type=platform or "unknown",
                              latency_ms=int((time.monotonic() - started_ms) * 1000), error=refused, retryable=False)

    from channels.sender import send_to_channel

    result = await send_to_channel(db=db, workspace_id=workspace_id, platform=platform,
                                   text=_render_callback_text(payload), target=target, delivery_id=delivery_id)
    return DispatchResult(
        success=result.ok,
        destination_type=platform,
        latency_ms=result.latency_ms or int((time.monotonic() - started_ms) * 1000),
        error=result.error,
        retryable=result.retryable,
        extra={"platform": platform, "target": target} if result.ok else {},
    )


async def record_attempt(
    db: Session,
    *,
    site_id: UUID,
    workspace_id: UUID,
    session_id: str,
    request_id: str,
    destination: dict,
    result: DispatchResult,
    attempt: int,
    final: bool,
) -> None:
    """The attempt's ``widget_event_log`` row; a final failure is also recorded as an error, so
    the ERRORS-by-subsystem tile reflects undelivered callbacks."""
    kind = destination_type(destination)
    if result.success:
        data = {"request_id": request_id, "destination_type": kind, "latency_ms": result.latency_ms,
                "attempt": attempt, **result.extra}
        await log_widget_event(db, site_id=site_id, event_type="callback_delivered", session_id=session_id,
                               event_data=data)
        return
    data = {"request_id": request_id, "destination_type": kind, "error": result.error, "attempt": attempt,
            "retryable": result.retryable, "permanent": final}
    await log_widget_event(db, site_id=site_id, event_type="callback_failed", session_id=session_id, event_data=data)
    if final:
        record_error(subsystem="widget", operation="deliver_callback",
                     error=RuntimeError(result.error or "callback delivery failed"), workspace_id=workspace_id,
                     extra={"destination_type": kind, "site_id": str(site_id), "attempts": attempt,
                            "retryable": result.retryable})


async def dispatch_callback_for_site(
    *,
    site_id: UUID,
    workspace_id: UUID,
    session_id: str,
    request_id: str,
    payload: CallbackPayload,
    destinations: list[dict],
) -> list[DispatchResult]:
    """The dashboard's test send: one attempt to each of a Site's destinations, in parallel,
    each logged, the results answered at once. Never raises."""
    db: Session = SessionLocal()

    async def once(destination: dict) -> DispatchResult:
        result = await dispatch_via_channel(destination=destination, payload=payload, db=db,
                                            workspace_id=workspace_id)
        await record_attempt(db, site_id=site_id, workspace_id=workspace_id, session_id=session_id,
                             request_id=request_id, destination=destination, result=result, attempt=1, final=True)
        return result

    try:
        return list(await asyncio.gather(*(once(d) for d in destinations)))
    except Exception:  # noqa: BLE001 — logged; the test send answers with no results
        logger.exception("dispatch_callback_for_site crashed for request_id=%s site_id=%s", request_id, site_id)
        return []
    finally:
        db.close()
