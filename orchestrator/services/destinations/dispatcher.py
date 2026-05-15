"""
Destination dispatch orchestrator (PRD-008-A.1).

Routes a callback to all of a Site's configured destinations in
parallel, with retry-with-exponential-backoff on retryable failures.
Every attempt writes a row to ``widget_event_log`` so the dashboard
can surface delivery state.

Destinations reference ``ChannelConnection`` rows (PRD-55) by id.
Dispatch loads the running adapter from ``ChannelManager`` and calls
``send_message(target, text)`` — same path heartbeats use.

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
from core.models.channels import ChannelConnection
from core.models.sites import Site
from modules.widgets.telemetry import log_widget_event

from services.destinations.base import CallbackPayload, DispatchResult

logger = logging.getLogger(__name__)


MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (0, 5, 15)  # before attempts 1, 2, 3


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
    """Dispatch a callback to a workspace's connected channel.

    `destination` shape:
        {"type": "channel_connection",
         "connection_id": "<uuid>",
         "target": "<channel/chat id or phone>",
         "platform": "<optional display hint>",
         "label": "<optional display label>"}
    """
    started_ms = time.monotonic()
    conn_id_raw = destination.get("connection_id")
    target = destination.get("target")

    # Static guards (permanent failures — no retry)
    if not conn_id_raw or not target:
        return DispatchResult(
            success=False,
            destination_type="channel_connection",
            latency_ms=int((time.monotonic() - started_ms) * 1000),
            error="missing connection_id or target",
            retryable=False,
        )
    try:
        conn_id = UUID(str(conn_id_raw))
    except (ValueError, TypeError):
        return DispatchResult(
            success=False,
            destination_type="channel_connection",
            latency_ms=int((time.monotonic() - started_ms) * 1000),
            error="connection_id is not a valid UUID",
            retryable=False,
        )

    # Workspace-scoped lookup — enforces tenant isolation even if a
    # malicious request slipped a foreign connection_id into Site
    # settings.
    conn = (
        db.query(ChannelConnection)
        .filter(
            ChannelConnection.id == conn_id,
            ChannelConnection.workspace_id == workspace_id,
        )
        .first()
    )
    if conn is None:
        return DispatchResult(
            success=False,
            destination_type="channel_connection",
            latency_ms=int((time.monotonic() - started_ms) * 1000),
            error="channel connection not found in this workspace",
            retryable=False,
        )
    if conn.status != "active":
        return DispatchResult(
            success=False,
            destination_type="channel_connection",
            latency_ms=int((time.monotonic() - started_ms) * 1000),
            error=f"channel connection status is {conn.status!r}; expected 'active'",
            retryable=False,
        )

    # Lazy import to avoid circular dep at module load
    from channels.manager import get_channel_manager

    manager = get_channel_manager()
    adapter = manager._adapters.get(str(conn_id))
    if adapter is None:
        # Adapter not running yet — could be a cold start. Mark retryable
        # so the next attempt gets a fresh chance after the worker has
        # had a moment to load.
        return DispatchResult(
            success=False,
            destination_type="channel_connection",
            latency_ms=int((time.monotonic() - started_ms) * 1000),
            error="channel adapter not loaded",
            retryable=True,
        )

    text = _render_callback_text(payload)
    try:
        ok = await adapter.send_message(str(target), text)
    except Exception as exc:  # noqa: BLE001 — adapter-specific transients
        logger.warning(
            "dispatch_via_channel raised for connection_id=%s: %s",
            conn_id, exc,
        )
        return DispatchResult(
            success=False,
            destination_type="channel_connection",
            latency_ms=int((time.monotonic() - started_ms) * 1000),
            error=f"adapter exception: {exc}",
            retryable=True,
        )

    latency_ms = int((time.monotonic() - started_ms) * 1000)
    if ok:
        return DispatchResult(
            success=True,
            destination_type="channel_connection",
            latency_ms=latency_ms,
            extra={"platform": conn.platform, "target": str(target)},
        )
    return DispatchResult(
        success=False,
        destination_type="channel_connection",
        latency_ms=latency_ms,
        error="adapter rejected target (returned False)",
        retryable=False,
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
    destination_type = destination.get("type") or "unknown"

    if destination_type != "channel_connection":
        result = DispatchResult(
            success=False,
            destination_type=destination_type,
            latency_ms=0,
            error=f"unsupported destination type: {destination_type!r}",
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
            return result

    # Exhausted attempts
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
