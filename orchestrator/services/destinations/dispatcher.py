"""
Destination dispatch orchestrator (PRD-008-A.1).

Routes a callback to all of a Site's configured destinations in
parallel, with retry-with-exponential-backoff on retryable failures.
Every attempt writes a row to ``widget_event_log`` so the dashboard
can surface delivery state.

Destinations are platform-keyed (``{"platform": "telegram"}`` etc.) and
delivery goes through ``send_workspace_notification`` — the same
function heartbeats use to reach Telegram / Slack / Webhook. No
parallel destination zoo.

For v1 the queue is in-process via ``asyncio.create_task`` (called
from the callback endpoint). The same orchestrator function will work
unchanged once a Redis-backed worker is wired in a follow-up — the
callsite swaps from ``create_task`` to ``enqueue_redis``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import UUID

from sqlalchemy.orm import Session

from core.database.database import SessionLocal
from core.models.sites import Site
from core.utils.exception_telemetry import record_error
from modules.widgets.telemetry import log_widget_event

from services.destinations.base import CALLBACK_PLATFORMS, CallbackPayload, DispatchResult

logger = logging.getLogger(__name__)


MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (0, 5, 15)  # before attempts 1, 2, 3


def _resolve_platform(destination: dict) -> str:
    """Return the destination's platform, accepting both the new and
    legacy shapes:

    - ``{"platform": "telegram"}``                       — preferred
    - ``{"type": "channel_connection", "platform": …}``  — legacy
    """
    return str(destination.get("platform") or "").strip().lower()


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


async def dispatch_via_channel(
    *,
    destination: dict,
    payload: CallbackPayload,
    db: Session,
    workspace_id: UUID,
) -> DispatchResult:
    """Dispatch a callback through the unified ``channels.sender``.

    Every per-platform detail (URLs, auth, error semantics) lives in
    the driver. This function just shapes the input/output and
    enforces the validator's contract.

    Accepted destination shapes::

        {"platform": "telegram"}
        {"platform": "slack",   "channel_id":  "C01ABC..."}
        {"platform": "whatsapp"}
        {"platform": "webhook", "webhook_url": "https://…"}
    """
    started_ms = time.monotonic()
    platform = _resolve_platform(destination)

    if not platform:
        return DispatchResult(
            success=False,
            destination_type="unknown",
            latency_ms=int((time.monotonic() - started_ms) * 1000),
            error="destination missing 'platform' field",
            retryable=False,
        )
    if platform not in CALLBACK_PLATFORMS:
        return DispatchResult(
            success=False,
            destination_type=platform,
            latency_ms=int((time.monotonic() - started_ms) * 1000),
            error=f"unsupported platform {platform!r}",
            retryable=False,
        )

    text = _render_callback_text(payload)

    # Per-platform target overrides from the destination row. Slack
    # accepts an explicit channel_id; Telegram accepts an explicit
    # chat_id (the workspace may have never received a /start so the
    # auto-captured default isn't available); webhook needs the URL.
    target: str | None = None
    if platform == "slack":
        target = (destination.get("channel_id") or "").strip() or None
    elif platform == "telegram":
        target = (destination.get("chat_id") or "").strip() or None
    elif platform == "webhook":
        target = (destination.get("webhook_url") or "").strip() or None
        if not target:
            return DispatchResult(
                success=False,
                destination_type=platform,
                latency_ms=int((time.monotonic() - started_ms) * 1000),
                error="webhook destination missing 'webhook_url'",
                retryable=False,
            )

    from channels.sender import send_to_channel

    result = await send_to_channel(
        db=db,
        workspace_id=workspace_id,
        platform=platform,
        text=text,
        target=target,
    )

    return DispatchResult(
        success=result.ok,
        destination_type=platform,
        latency_ms=result.latency_ms or int((time.monotonic() - started_ms) * 1000),
        error=result.error,
        retryable=result.retryable,
        extra={"platform": platform, "target": target} if result.ok else {},
    )


async def dispatch_one_destination(
    *,
    db: Session,
    site_id: UUID,
    workspace_id: UUID,
    session_id: str,
    request_id: str,
    destination: dict,
    payload: CallbackPayload,
) -> DispatchResult:
    """Try a single destination up to MAX_ATTEMPTS, with backoff between
    retryable failures. Writes a widget_event_log row for every attempt.

    Returns the final DispatchResult (success or terminal failure).
    """
    destination_type = _resolve_platform(destination) or destination.get("type") or "unknown"

    last: DispatchResult | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            await asyncio.sleep(BACKOFF_SECONDS[attempt - 1])

        result = await dispatch_via_channel(
            destination=destination,
            payload=payload,
            db=db,
            workspace_id=workspace_id,
        )
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
            record_error(
                subsystem="widget",
                operation="deliver_callback",
                error=RuntimeError(result.error or "callback delivery failed"),
                workspace_id=workspace_id,
                extra={
                    "destination_type": destination_type,
                    "site_id": str(site_id),
                    "attempt": attempt,
                    "retryable": False,
                },
            )
            return result

    # Exhausted attempts — every retry failed; record the terminal failure so
    # the ERRORS-by-subsystem tile reflects undelivered callbacks.
    record_error(
        subsystem="widget",
        operation="deliver_callback",
        error=RuntimeError((last.error if last else None) or "callback delivery exhausted retries"),
        workspace_id=workspace_id,
        extra={
            "destination_type": destination_type,
            "site_id": str(site_id),
            "attempts": MAX_ATTEMPTS,
            "retryable": True,
        },
    )
    return last  # type: ignore[return-value]


async def dispatch_callback_for_site(
    *,
    site_id: UUID,
    workspace_id: UUID,
    session_id: str,
    request_id: str,
    payload: CallbackPayload,
    destinations: list[dict],
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
                workspace_id=workspace_id,
                session_id=session_id,
                request_id=request_id,
                destination=dest,
                payload=payload,
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
    asyncio.create_task(
        dispatch_callback_for_site(
            site_id=site.id,
            workspace_id=site.workspace_id,
            session_id=session_id,
            request_id=request_id,
            payload=payload,
            destinations=destinations,
        )
    )
