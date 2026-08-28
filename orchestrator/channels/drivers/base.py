"""Channel driver interface.

A ``ChannelDriver`` encapsulates everything needed to talk to a single
messaging platform: credential verification, message send, and (for
platforms that support it) webhook install / polling start. Higher-up
layers (``api.channels``, ``api.webhooks``, ``channels.sender``,
``services.destinations.dispatcher``) call methods on the driver and
never touch a platform's HTTP API directly.

Modes
-----
A driver declares the connectivity modes it supports:

- ``ConnectivityMode.WEBHOOK`` — platform POSTs inbound to our
  ``/api/webhooks/ws/{workspace_id}``; outbound replies go via HTTP.
  We register the webhook via ``install_webhook`` at Connect time.
- ``ConnectivityMode.POLLING`` — we run a long-poll loop against the
  platform (currently Telegram's ``getUpdates``, optionally Discord's
  gateway). Requires the platform SDK as an optional runtime dep.

Most platforms are webhook-only. Telegram supports both. Discord is
gateway-only. The driver tells the dashboard what's available so the
UI can render the right form.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnknownPlatform(LookupError):
    """Raised when no driver is registered for a requested platform."""


class DriverNotConfigured(RuntimeError):
    """Raised when a driver method is called on a row that's missing
    credentials the driver needs (e.g. Telegram driver with no
    ``bot_token``). Surfaces as a 400-ish error to API callers."""


# ---------------------------------------------------------------------------
# Mode + result types
# ---------------------------------------------------------------------------

class ConnectivityMode(str, enum.Enum):
    WEBHOOK = "webhook"
    POLLING = "polling"


@dataclass(frozen=True)
class VerifyResult:
    """Returned by ``ChannelDriver.verify``.

    Mirrors the JSON the dashboard's Test button consumes. ``identity``
    is a free-form display string (bot username, Slack team name, etc.)
    surfaced to the user on success.
    """
    ok: bool
    identity: Optional[str] = None
    error: Optional[str] = None
    # Free-form metadata stashed onto the row's ``metadata`` JSONB —
    # e.g. {"bot_id": 123, "team_id": "T0…"} captured during verify so
    # subsequent operations don't re-hit the platform API.
    metadata: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class SendResult:
    """Returned by ``ChannelDriver.send``."""
    ok: bool
    latency_ms: int
    error: Optional[str] = None
    # True if the failure mode is worth retrying (transient network /
    # 5xx) — False for permanent failures (invalid token, missing
    # chat_id, etc.). Mirrors ``DispatchResult.retryable``.
    retryable: bool = True
    # PRD-225: the platform's id for the sent message and the target it went to.
    # Drivers that can report them (Telegram) populate these so a reply can be
    # correlated back to a pending question. Optional — other drivers leave None.
    message_id: Optional[str] = None
    target: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract driver
# ---------------------------------------------------------------------------

class ChannelDriver(ABC):
    """Per-platform abstraction. Stateless — every method takes the
    workspace + config it needs. The driver never touches the DB."""

    #: Display name the dashboard shows (e.g. "Telegram", "Slack").
    display_name: str = ""

    #: Connectivity modes this driver supports, in preferred order.
    supported_modes: tuple[ConnectivityMode, ...] = ()

    #: Required config keys the connect form must collect, in order.
    #: Each entry is ``(key, label, placeholder)``. The dashboard
    #: renders them as inputs; backend validates presence at Connect.
    required_config: tuple[tuple[str, str, str], ...] = ()

    #: Optional config keys (no validation, shown after the required
    #: block). Same tuple shape.
    optional_config: tuple[tuple[str, str, str], ...] = ()

    # ------------------------------------------------------------------
    # Required lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def verify(self, *, workspace_id: str, config: Mapping[str, Any]) -> VerifyResult:
        """Hit the platform's identity endpoint and confirm the
        credentials in ``config`` work. Pure read — no side effects."""

    @abstractmethod
    async def send(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
        target: Optional[str],
        text: str,
    ) -> SendResult:
        """Deliver ``text`` to ``target`` (chat id, channel id, phone,
        URL, …) — whatever the platform's primitive is. When ``target``
        is None, drivers MAY use a sensible workspace default if one is
        known (e.g. Telegram's ``telegram_default_chat_id``)."""

    # ------------------------------------------------------------------
    # Webhook-mode (default = unsupported)
    # ------------------------------------------------------------------

    async def install_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
        webhook_url: str,
    ) -> VerifyResult:
        """Register ``webhook_url`` with the platform so inbound
        messages POST to us. Implemented only by drivers that declare
        ``ConnectivityMode.WEBHOOK`` in ``supported_modes``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support webhook mode"
        )

    async def uninstall_webhook(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
    ) -> bool:
        """Remove the webhook registration. Idempotent."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support webhook mode"
        )

    async def get_webhook_info(
        self,
        *,
        workspace_id: str,
        config: Mapping[str, Any],
    ) -> Optional[str]:
        """Return the currently-registered webhook URL on the platform,
        or None if no webhook is set. Used by status reconciliation."""
        return None

    # ------------------------------------------------------------------
    # Polling-mode (default = unsupported)
    # ------------------------------------------------------------------

    async def start_polling(
        self,
        *,
        connection_id: str,
        workspace_id: str,
        config: Mapping[str, Any],
    ) -> bool:
        raise NotImplementedError(
            f"{type(self).__name__} does not support polling mode"
        )

    async def stop_polling(
        self,
        *,
        connection_id: str,
    ) -> bool:
        raise NotImplementedError(
            f"{type(self).__name__} does not support polling mode"
        )

    def is_polling_running(self, *, connection_id: str) -> bool:
        return False

    # ------------------------------------------------------------------
    # Helpers — concrete drivers may override
    # ------------------------------------------------------------------

    def default_mode(self) -> ConnectivityMode:
        """The mode the dashboard pre-selects when this platform is
        chosen. Defaults to the first entry in ``supported_modes``."""
        if not self.supported_modes:
            raise RuntimeError(
                f"{type(self).__name__} declares no supported_modes — "
                f"override default_mode() or set supported_modes."
            )
        return self.supported_modes[0]

    def supports(self, mode: ConnectivityMode) -> bool:
        return mode in self.supported_modes
