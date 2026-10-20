"""Generic outbound webhook driver.

For when a merchant wants callbacks delivered to a Zapier / n8n / their
own HTTPS endpoint instead of a chat platform. There's nothing to
verify (no platform API), so ``verify`` just checks the URL looks like
a URL. ``send`` POSTs JSON.

Inbound webhooks are NOT this driver's responsibility — incoming chat
traffic goes through ``/api/webhooks/ws/{workspace_id}`` which is
platform-aware.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

import httpx

from modules.policy import ai_disclosure as disclosure

from . import register_driver
from .base import (
    ChannelDriver,
    ConnectivityMode,
    DriverNotConfigured,
    SendResult,
    VerifyResult,
)

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 15.0


def _payload(text: str, ai_disclosure: Optional[str]) -> dict[str, Any]:
    """The JSON body: the text, the Art. 50(2) mark, and the notice when due."""
    body: dict[str, Any] = {"text": text, "message": text, disclosure.AI_GENERATED_FIELD: True}
    if ai_disclosure:
        return {**body, disclosure.AI_DISCLOSURE_FIELD: ai_disclosure}
    return body


class WebhookDriver(ChannelDriver):
    display_name = "Webhook URL"
    # PRD-264 EUAI-03: the receiver is a system, so the AI notice travels as
    # payload fields (ai_generated, ai_disclosure), not as a text prefix.
    AI_NOTICE_IN_PAYLOAD = True
    # Webhook here is a misnomer for "outbound HTTP POST" — we keep the
    # mode label so the UI treats it like the other webhook channels
    # (no polling to start).
    supported_modes = (ConnectivityMode.WEBHOOK,)
    required_config = (
        ("webhook_url", "Webhook URL", "https://hooks.example.com/your-endpoint"),
    )

    async def verify(self, *, workspace_id: str, config: Mapping[str, Any]) -> VerifyResult:
        url = str(config.get("webhook_url") or "").strip()
        if not url:
            return VerifyResult(ok=False, error="webhook_url is required")
        if not (url.startswith("https://") or url.startswith("http://")):
            return VerifyResult(ok=False, error="webhook_url must start with http(s)://")
        return VerifyResult(ok=True, identity=url)

    async def send(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
        target: Optional[str],
        text: str,
        ai_disclosure: Optional[str] = None,
    ) -> SendResult:
        """POST ``text`` as JSON, marked ``ai_generated: true``; the AI notice,
        when the channel sender says one is due, rides as ``ai_disclosure``."""
        url = str(target or config.get("webhook_url") or "").strip()
        if not url:
            raise DriverNotConfigured("Webhook driver requires webhook_url in config or target")

        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    url,
                    json=_payload(text, ai_disclosure),
                    headers={"Content-Type": "application/json"},
                )
        except Exception as exc:  # noqa: BLE001
            return SendResult(
                ok=False, latency_ms=int((time.monotonic() - started) * 1000),
                error=f"network error: {exc}", retryable=True,
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        if resp.status_code < 400:
            return SendResult(ok=True, latency_ms=latency_ms)
        return SendResult(
            ok=False, latency_ms=latency_ms,
            error=f"webhook returned HTTP {resp.status_code}",
            retryable=resp.status_code >= 500,
        )

    async def install_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
        webhook_url: str,
    ) -> VerifyResult:
        # Inbound from a generic webhook isn't part of this driver.
        return VerifyResult(ok=True, identity=webhook_url)

    async def uninstall_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
    ) -> bool:
        return True


register_driver("webhook", WebhookDriver)
