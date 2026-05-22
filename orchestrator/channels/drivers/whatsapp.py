"""WhatsApp channel driver (Meta Cloud API).

Webhook-only: Meta POSTs incoming messages to a webhook URL the
merchant configures in the Meta Business Manager. Outbound goes through
the Graph API.

Configuration the merchant pastes:
- ``phone_number_id`` — the WhatsApp Business phone-number id
- ``access_token`` — a permanent system-user access token
- ``verify_token`` (optional) — the string Meta uses on the GET
  ``hub.verify_token`` handshake. The dashboard surfaces what value to
  paste into Meta after Connect.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

import httpx

from . import register_driver
from .base import (
    ChannelDriver,
    ConnectivityMode,
    DriverNotConfigured,
    SendResult,
    VerifyResult,
)

logger = logging.getLogger(__name__)

_API_BASE = "https://graph.facebook.com/v19.0"
_HTTP_TIMEOUT = 15.0


class WhatsAppDriver(ChannelDriver):
    display_name = "WhatsApp"
    supported_modes = (ConnectivityMode.WEBHOOK,)
    required_config = (
        ("phone_number_id", "Phone Number ID", "From Meta Business"),
        ("access_token", "Access Token", "Permanent system-user token"),
    )
    optional_config = (
        ("verify_token", "Verify Token", "Webhook GET handshake string"),
    )

    async def verify(self, *, workspace_id: str, config: Mapping[str, Any]) -> VerifyResult:
        phone_id = str(config.get("phone_number_id") or "").strip()
        token = str(config.get("access_token") or "").strip()
        if not phone_id or not token:
            return VerifyResult(
                ok=False, error="phone_number_id and access_token are required",
            )

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    f"{_API_BASE}/{phone_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"fields": "verified_name,display_phone_number"},
                )
        except Exception as exc:  # noqa: BLE001
            return VerifyResult(ok=False, error=f"network error: {exc}")

        if resp.status_code != 200:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            msg = ((data.get("error") or {}).get("message")) or f"HTTP {resp.status_code}"
            return VerifyResult(ok=False, error=f"whatsapp: {msg}")

        data = resp.json() or {}
        return VerifyResult(
            ok=True,
            identity=data.get("verified_name") or data.get("display_phone_number") or phone_id,
            metadata={
                "verified_name": data.get("verified_name"),
                "display_phone_number": data.get("display_phone_number"),
            },
        )

    async def send(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
        target: Optional[str],
        text: str,
    ) -> SendResult:
        phone_id = str(config.get("phone_number_id") or "").strip()
        token = str(config.get("access_token") or "").strip()
        if not phone_id or not token:
            raise DriverNotConfigured("WhatsApp driver requires phone_number_id and access_token")
        if not target:
            return SendResult(
                ok=False, latency_ms=0,
                error="whatsapp send requires a recipient phone number (E.164)",
                retryable=False,
            )

        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    f"{_API_BASE}/{phone_id}/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "messaging_product": "whatsapp",
                        "to": target,
                        "type": "text",
                        "text": {"body": text},
                    },
                )
        except Exception as exc:  # noqa: BLE001
            return SendResult(
                ok=False, latency_ms=int((time.monotonic() - started) * 1000),
                error=f"network error: {exc}", retryable=True,
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        if resp.status_code == 200:
            return SendResult(ok=True, latency_ms=latency_ms)

        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        err = ((data.get("error") or {}).get("message")) or f"HTTP {resp.status_code}"
        retryable = resp.status_code >= 500
        return SendResult(ok=False, latency_ms=latency_ms, error=f"whatsapp: {err}", retryable=retryable)

    async def install_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
        webhook_url: str,
    ) -> VerifyResult:
        # Meta's webhook configuration is app-side, not API-driven.
        return VerifyResult(ok=True, identity=webhook_url)

    async def uninstall_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
    ) -> bool:
        return True


register_driver("whatsapp", WhatsAppDriver)
