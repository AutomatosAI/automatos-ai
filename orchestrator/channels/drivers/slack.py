"""Slack channel driver.

Webhook-only: Slack apps run in Events API mode (Slack POSTs to us when
a message is sent in a channel the bot is in). Outbound replies use the
``chat.postMessage`` Web API with the bot's OAuth token.

Slack doesn't have a setWebhook equivalent — the events URL is
configured app-side in the Slack admin. ``install_webhook`` therefore
just returns the URL the merchant must paste; the driver doesn't push
it to Slack.
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

_API_BASE = "https://slack.com/api"
_HTTP_TIMEOUT = 15.0


class SlackDriver(ChannelDriver):
    display_name = "Slack"
    supported_modes = (ConnectivityMode.WEBHOOK,)
    required_config = (
        ("bot_token", "Bot Token", "xoxb-..."),
        ("signing_secret", "Signing Secret", "From OAuth & Permissions"),
    )
    optional_config = (
        ("default_channel", "Default Channel", "#sales-leads or C0123ABCDEF"),
    )

    async def verify(self, *, workspace_id: str, config: Mapping[str, Any]) -> VerifyResult:
        token = str(config.get("bot_token") or "").strip()
        if not token:
            return VerifyResult(ok=False, error="bot_token is required")

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    f"{_API_BASE}/auth.test",
                    headers={"Authorization": f"Bearer {token}"},
                )
        except Exception as exc:  # noqa: BLE001
            return VerifyResult(ok=False, error=f"network error: {exc}")

        if resp.status_code != 200:
            return VerifyResult(ok=False, error=f"HTTP {resp.status_code}")
        data = resp.json() or {}
        if not data.get("ok"):
            return VerifyResult(ok=False, error=data.get("error", "unknown slack error"))

        return VerifyResult(
            ok=True,
            identity=data.get("team") or data.get("user"),
            metadata={
                "team_id": data.get("team_id"),
                "team": data.get("team"),
                "user": data.get("user"),
                "user_id": data.get("user_id"),
                "bot_id": data.get("bot_id"),
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
        token = str(config.get("bot_token") or "").strip()
        if not token:
            raise DriverNotConfigured("Slack driver requires bot_token in config")

        channel = (target or "").strip() or str(config.get("default_channel") or "").strip()
        if not channel:
            return SendResult(
                ok=False, latency_ms=0,
                error="slack send requires a channel id or name (no default configured)",
                retryable=False,
            )

        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    f"{_API_BASE}/chat.postMessage",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "text": text},
                )
        except Exception as exc:  # noqa: BLE001
            return SendResult(
                ok=False, latency_ms=int((time.monotonic() - started) * 1000),
                error=f"network error: {exc}", retryable=True,
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        if resp.status_code == 200 and data.get("ok"):
            return SendResult(ok=True, latency_ms=latency_ms)

        err = data.get("error") or f"HTTP {resp.status_code}"
        # Slack uses ``channel_not_found``, ``not_in_channel``, ``invalid_auth`` —
        # all permanent. Transient: rate_limited (we won't see often) + 5xx.
        retryable = resp.status_code >= 500 or err == "ratelimited"
        return SendResult(ok=False, latency_ms=latency_ms, error=f"slack: {err}", retryable=retryable)

    async def install_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
        webhook_url: str,
    ) -> VerifyResult:
        # Slack apps configure the events URL in their app manifest /
        # admin UI; there's no API to push it from here. The dashboard
        # surfaces the URL we want them to paste, and we re-verify on
        # Test by checking events come in.
        return VerifyResult(ok=True, identity=webhook_url)

    async def uninstall_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
    ) -> bool:
        return True  # No-op for Slack — events URL is app-side.


register_driver("slack", SlackDriver)
