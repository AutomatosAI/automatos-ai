"""
Base Channel Adapter (PRD-55 US-019)
=====================================
Abstract base class for all channel adapters.

Each platform (Telegram, Slack, Discord, etc.) subclasses
BaseChannelAdapter and implements the abstract methods to
translate platform-specific messages into RequestEnvelopes
and route them through the UniversalRouter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseChannelAdapter(ABC):
    """Abstract base for platform channel adapters."""

    def __init__(self, connection_id: str, workspace_id: str, config: Dict[str, Any]):
        self.connection_id = connection_id
        self.workspace_id = workspace_id
        self.config = config
        self.is_running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def start(self):
        """Start listening for messages on this channel."""
        ...

    @abstractmethod
    async def stop(self):
        """Stop the adapter gracefully."""
        ...

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    @abstractmethod
    async def send_message(self, channel_id: str, text: str, **kwargs) -> bool:
        """Send a message to a specific channel/conversation."""
        ...

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test if the connection credentials are valid.

        Returns a dict with at least ``{"ok": bool, "detail": str}``.
        """
        ...

    # ------------------------------------------------------------------
    # Ingest pipeline
    # ------------------------------------------------------------------

    async def handle_message(self, platform_message: Dict[str, Any]):
        """Process an incoming platform message through the
        ingest -> route -> execute -> respond pipeline.
        """
        try:
            envelope = self._to_envelope(platform_message)
            if not envelope:
                return

            # Lazy imports to avoid circular dependency at module load
            from core.routing.engine import UniversalRouter
            from core.database.database import SessionLocal

            db = SessionLocal()
            try:
                router = UniversalRouter(db=db)
                decision = await router.route(envelope)
            finally:
                db.close()

            # Send the response back to the originating channel
            if decision and getattr(decision, "response", None):
                reply_channel = (
                    platform_message.get("reply_channel_id")
                    or platform_message.get("channel_id")
                )
                if reply_channel:
                    await self.send_message(reply_channel, decision.response)

        except Exception as e:
            logger.error(
                "[Channel:%s] Failed to handle message: %s",
                self.connection_id,
                e,
            )

    @abstractmethod
    def _to_envelope(self, platform_message: Dict[str, Any]) -> Optional["RequestEnvelope"]:
        """Convert a platform-specific message dict to a RequestEnvelope.

        Return ``None`` to silently skip messages that should be ignored
        (e.g. bot's own messages).
        """
        ...
