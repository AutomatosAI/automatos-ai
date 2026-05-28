"""Discord channel driver.

Discord doesn't expose a webhook model for inbound DMs/server messages —
bots use the Gateway (WebSocket) connection. That's a polling-style
adapter that requires the ``discord.py`` library as an optional dep,
identical in shape to the Telegram polling adapter.

For v1 the driver supports outbound message send via the REST API
(``channels/{id}/messages``) and exposes ``verify`` against
``users/@me`` so the Connect form can confirm the bot token works. The
gateway adapter is intentionally not wired up here — most workspaces
won't need inbound Discord. When they do, implement ``start_polling``
along the same lines as ``telegram.py`` and gate behind the import.
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

_API_BASE = "https://discord.com/api/v10"
_HTTP_TIMEOUT = 15.0


class DiscordDriver(ChannelDriver):
    display_name = "Discord"
    # Outbound-only for v1. POLLING (gateway) is on the roadmap once a
    # workspace genuinely needs it.
    supported_modes = (ConnectivityMode.WEBHOOK,)
    required_config = (
        ("bot_token", "Bot Token", "From Discord developer portal"),
    )

    async def verify(self, *, workspace_id: str, config: Mapping[str, Any]) -> VerifyResult:
        token = str(config.get("bot_token") or "").strip()
        if not token:
            return VerifyResult(ok=False, error="bot_token is required")

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    f"{_API_BASE}/users/@me",
                    headers={"Authorization": f"Bot {token}"},
                )
        except Exception as exc:  # noqa: BLE001
            return VerifyResult(ok=False, error=f"network error: {exc}")

        if resp.status_code != 200:
            return VerifyResult(ok=False, error=f"Discord API returned {resp.status_code}")
        data = resp.json() or {}
        return VerifyResult(
            ok=True,
            identity=data.get("username"),
            metadata={"bot_id": data.get("id"), "username": data.get("username")},
        )

    async def send(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
        target: Optional[str],
        text: str,
    ) -> SendResult:
        token = str(config.get("bot_token") or "").strip()
        if not token:
            raise DriverNotConfigured("Discord driver requires bot_token in config")
        if not target:
            return SendResult(
                ok=False, latency_ms=0,
                error="discord send requires a channel id",
                retryable=False,
            )

        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    f"{_API_BASE}/channels/{target}/messages",
                    headers={"Authorization": f"Bot {token}"},
                    json={"content": text},
                )
        except Exception as exc:  # noqa: BLE001
            return SendResult(
                ok=False, latency_ms=int((time.monotonic() - started) * 1000),
                error=f"network error: {exc}", retryable=True,
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        if resp.status_code in (200, 201):
            return SendResult(ok=True, latency_ms=latency_ms)

        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        err = data.get("message") or f"HTTP {resp.status_code}"
        retryable = resp.status_code >= 500 or resp.status_code == 429
        return SendResult(ok=False, latency_ms=latency_ms, error=f"discord: {err}", retryable=retryable)

    async def install_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
        webhook_url: str,
    ) -> VerifyResult:
        # Discord doesn't support pushing inbound to an HTTPS webhook.
        # The dashboard should pre-select POLLING when that's added.
        return VerifyResult(ok=True, identity="outbound-only")

    async def uninstall_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
    ) -> bool:
        return True


register_driver("discord", DiscordDriver)
