"""Telegram channel driver.

Two modes:

- **Webhook (default)** — uses raw ``httpx`` against the Bot API. No
  optional dependency required. Inbound messages POST to
  ``/api/webhooks/ws/{workspace_id}`` (we register the URL via
  ``setWebhook`` at Connect time). Replies are ``sendMessage`` POSTs.
- **Polling (optional)** — uses ``python-telegram-bot`` if installed.
  Lifecycle is managed via the module-level ``_RUNNING`` dict so the
  driver stays stateless from the caller's perspective.

The driver itself never touches the DB. Target resolution (the
shopper's chat_id, the merchant's default chat_id) is the sender's
job — by the time ``send`` is called the caller knows the target.
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


# Module-level adapter registry for polling mode. Keyed by
# connection_id so start/stop is idempotent. ``Any`` because the
# optional python-telegram-bot Application type isn't always
# importable.
_RUNNING: dict[str, Any] = {}


_API_BASE = "https://api.telegram.org"
_HTTP_TIMEOUT = 10.0


def _api(token: str, method: str) -> str:
    return f"{_API_BASE}/bot{token}/{method}"


class TelegramDriver(ChannelDriver):
    display_name = "Telegram"
    supported_modes = (ConnectivityMode.WEBHOOK, ConnectivityMode.POLLING)
    required_config = (
        ("bot_token", "Bot Token", "From @BotFather — e.g. 1234567890:AAF..."),
    )

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    async def verify(self, *, workspace_id: str, config: Mapping[str, Any]) -> VerifyResult:
        token = str(config.get("bot_token") or "").strip()
        if not token:
            return VerifyResult(ok=False, error="bot_token is required")
        if ":" not in token:
            # The most common Telegram-config failure mode. Catching
            # this here gives a clear error before we hit the API.
            return VerifyResult(
                ok=False,
                error=(
                    "bot_token is missing the '<bot_id>:' prefix. Paste the "
                    "full token from @BotFather (e.g. 1234567890:AAF...)."
                ),
            )

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(_api(token, "getMe"))
        except Exception as exc:  # noqa: BLE001 — surface to caller
            logger.warning("[telegram.verify] HTTP error for ws=%s: %s", workspace_id, exc)
            return VerifyResult(ok=False, error=f"network error: {exc}")

        if resp.status_code != 200:
            return VerifyResult(
                ok=False,
                error=f"Telegram API returned {resp.status_code}",
            )

        result = (resp.json() or {}).get("result") or {}
        username = result.get("username")
        bot_id = result.get("id")
        return VerifyResult(
            ok=True,
            identity=username,
            metadata={"bot_id": bot_id, "username": username},
        )

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

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
            raise DriverNotConfigured("Telegram driver requires bot_token in config")
        if not target:
            return SendResult(
                ok=False, latency_ms=0,
                error=(
                    "telegram send requires a chat_id target — none provided and "
                    "no default chat captured (send /start to the bot to capture one)"
                ),
                retryable=False,
            )

        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    _api(token, "sendMessage"),
                    json={"chat_id": target, "text": text},
                )
        except Exception as exc:  # noqa: BLE001
            return SendResult(
                ok=False,
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"network error: {exc}",
                retryable=True,
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        if resp.status_code == 200 and (resp.json() or {}).get("ok"):
            # PRD-225: echo the sent message_id + target so a Telegram reply can
            # be correlated back to a pending question (channel_refs).
            result_obj = (resp.json() or {}).get("result") or {}
            message_id = result_obj.get("message_id")
            return SendResult(
                ok=True,
                latency_ms=latency_ms,
                message_id=str(message_id) if message_id is not None else None,
                target=str(target),
            )

        # Telegram error shape: {"ok": false, "error_code": 400, "description": "Bad Request: chat not found"}
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        description = body.get("description") or f"HTTP {resp.status_code}"
        retryable = resp.status_code >= 500
        return SendResult(
            ok=False,
            latency_ms=latency_ms,
            error=f"telegram: {description}",
            retryable=retryable,
        )

    # ------------------------------------------------------------------
    # Webhook mode
    # ------------------------------------------------------------------

    async def install_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
        webhook_url: str,
    ) -> VerifyResult:
        token = str(config.get("bot_token") or "").strip()
        if not token:
            raise DriverNotConfigured("Telegram driver requires bot_token in config")

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    _api(token, "setWebhook"),
                    json={"url": webhook_url, "drop_pending_updates": False},
                )
        except Exception as exc:  # noqa: BLE001
            return VerifyResult(ok=False, error=f"network error: {exc}")

        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        if resp.status_code == 200 and body.get("ok"):
            return VerifyResult(ok=True, identity=webhook_url)
        return VerifyResult(
            ok=False,
            error=f"setWebhook failed: {body.get('description') or resp.status_code}",
        )

    async def uninstall_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
    ) -> bool:
        token = str(config.get("bot_token") or "").strip()
        if not token:
            return False
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(_api(token, "deleteWebhook"))
            return resp.status_code == 200 and (resp.json() or {}).get("ok", False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[telegram.uninstall_webhook] %s for ws=%s", exc, workspace_id)
            return False

    async def get_webhook_info(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
    ) -> Optional[str]:
        token = str(config.get("bot_token") or "").strip()
        if not token:
            return None
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(_api(token, "getWebhookInfo"))
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        url = ((resp.json() or {}).get("result") or {}).get("url")
        return url or None

    # ------------------------------------------------------------------
    # Polling mode (optional dep)
    # ------------------------------------------------------------------

    async def start_polling(
        self,
        *,
        connection_id: str,
        workspace_id: str,
        config: Mapping[str, Any],
    ) -> bool:
        if connection_id in _RUNNING:
            return True

        try:
            from telegram.ext import ApplicationBuilder  # type: ignore
        except ImportError:
            logger.warning(
                "[telegram.polling] python-telegram-bot not installed — install with "
                "`pip install python-telegram-bot` to enable polling mode. "
                "Webhook mode does not require this dependency."
            )
            return False

        token = str(config.get("bot_token") or "").strip()
        if not token:
            raise DriverNotConfigured("Telegram driver requires bot_token in config")

        # Webhook + polling are mutually exclusive on Telegram's side —
        # remove any existing webhook first so getUpdates doesn't 409.
        await self.uninstall_webhook(workspace_id=workspace_id, config=config)

        try:
            app = ApplicationBuilder().token(token).build()
            await app.initialize()
            await app.start()
            await app.updater.start_polling()
        except Exception as exc:  # noqa: BLE001 — runtime, log + return False
            logger.exception("[telegram.polling] start failed for conn=%s: %s", connection_id, exc)
            return False

        _RUNNING[connection_id] = app
        logger.info("[telegram.polling] started conn=%s", connection_id)
        return True

    async def stop_polling(self, *, connection_id: str) -> bool:
        app = _RUNNING.pop(connection_id, None)
        if app is None:
            return False
        try:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[telegram.polling] stop raised for conn=%s: %s", connection_id, exc)
            return False
        return True

    def is_polling_running(self, *, connection_id: str) -> bool:
        return connection_id in _RUNNING


register_driver("telegram", TelegramDriver)
