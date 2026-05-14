"""
Email destination dispatcher (PRD-008-A Phase 6).

Sends a callback notification to a merchant-nominated email address
via SMTP. Uses Automatos's relay credentials from env config; the
merchant doesn't have to configure SMTP themselves.

For v1 this is a system-level relay. Per-merchant SMTP credentials
(merchant brings their own SES / SendGrid) is a future enhancement.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import time
from email.mime.text import MIMEText

from config import config

from services.destinations.base import CallbackPayload, DispatchResult

logger = logging.getLogger(__name__)


def _build_email_body(payload: CallbackPayload) -> str:
    lines = [
        f"New callback request from your storefront chat widget.",
        f"",
        f"Site:      {payload.site_display_name}",
        f"Name:      {payload.name}",
        f"Phone:     {payload.phone}",
    ]
    if payload.product_context:
        lines.append(f"Product:   {payload.product_context}")
    if payload.urgency:
        lines.append(f"Urgency:   {payload.urgency}")
    if payload.preferred_time:
        lines.append(f"Preferred: {payload.preferred_time}")
    lines.extend([
        "",
        f"Request ID: {payload.request_id}",
        "",
        "— Automatos",
    ])
    return "\n".join(lines)


def _send_via_smtp(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
) -> None:
    """Sync SMTP send — called via asyncio.to_thread."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    with smtplib.SMTP(host, port, timeout=10) as smtp:
        smtp.starttls()
        if user:
            smtp.login(user, password)
        smtp.send_message(msg)


async def dispatch_email(
    *,
    destination: dict,
    payload: CallbackPayload,
    smtp_send_func=None,  # injectable for tests
) -> DispatchResult:
    """Send the callback notification to ``destination['address']``.

    Returns a DispatchResult — never raises.
    """
    started = time.monotonic()
    recipient = destination.get("address")
    if not recipient or "@" not in recipient:
        return DispatchResult(
            success=False,
            destination_type="email",
            latency_ms=int((time.monotonic() - started) * 1000),
            error="destination missing or malformed 'address'",
            retryable=False,
        )

    host = getattr(config, "SMTP_HOST", None) or ""
    port = int(getattr(config, "SMTP_PORT", 587) or 587)
    user = getattr(config, "SMTP_USER", "") or ""
    password = getattr(config, "SMTP_PASSWORD", "") or ""
    sender = (
        getattr(config, "SMTP_FROM", "") or "callbacks@automatos.app"
    )

    if not host:
        return DispatchResult(
            success=False,
            destination_type="email",
            latency_ms=int((time.monotonic() - started) * 1000),
            error="SMTP not configured (SMTP_HOST env var missing)",
            retryable=False,
        )

    subject = f"Callback request from {payload.site_display_name}"
    body = _build_email_body(payload)

    sender_func = smtp_send_func or _send_via_smtp

    try:
        await asyncio.to_thread(
            sender_func,
            host=host,
            port=port,
            user=user,
            password=password,
            sender=sender,
            recipient=recipient,
            subject=subject,
            body=body,
        )
    except Exception as exc:  # noqa: BLE001
        return DispatchResult(
            success=False,
            destination_type="email",
            latency_ms=int((time.monotonic() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
            retryable=True,
        )

    return DispatchResult(
        success=True,
        destination_type="email",
        latency_ms=int((time.monotonic() - started) * 1000),
        extra={"recipient": recipient},
    )
