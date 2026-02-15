"""
Base Channel Adapter (PRD-55: Autonomous Assistant Platform).

Abstract base class that all platform-specific channel adapters must implement.
Supports: Telegram, Slack, Discord, WhatsApp, Teams, Google Chat,
Signal, iMessage, IRC, Matrix, LINE.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class BaseChannelAdapter(ABC):
    """Abstract interface for a messaging-platform adapter.

    Subclasses must implement the abstract methods to connect, send, and
    translate platform-specific messages into the universal RequestEnvelope.
    """

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        self.connection_id = connection_id
        self.workspace_id = workspace_id
        self.config = config
        self._running = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Initialise the platform client and start listening for messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully disconnect and clean up resources."""

    @abstractmethod
    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        """Send a text message to the given platform channel/chat."""

    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Verify the adapter's credentials and return connection info."""

    @abstractmethod
    def _to_envelope(self, platform_message: Any) -> Any:
        """Convert a platform-native message into a ``RequestEnvelope``."""

    # ------------------------------------------------------------------
    # Concrete routing helper
    # ------------------------------------------------------------------

    async def handle_message(self, platform_message: Any) -> Optional[str]:
        """Convert *platform_message* to an envelope, route it, and return the response text.

        This is the main entry-point called by each adapter's native event
        handler.  It:
        1. Converts the platform message into a :class:`RequestEnvelope`.
        2. Routes via :class:`UniversalRouter`.
        3. Returns the response text so the adapter can post it back.
        """
        try:
            from core.models.routing import RequestEnvelope, ChannelSource
            from core.routing.engine import UniversalRouter
            from core.database.database import SessionLocal

            envelope = self._to_envelope(platform_message)

            db = SessionLocal()
            try:
                router = UniversalRouter(db=db)
                decision = await router.route(envelope)

                if decision is None:
                    return "I'm not sure how to handle that. Let me route you to a human."

                # The router returns a RoutingDecision; the actual execution
                # of the agent/workflow response is handled upstream.  For
                # channel adapters we return the routing summary so the caller
                # can relay it.
                return (
                    decision.reasoning
                    or f"Routed to {decision.route_type} (confidence {decision.confidence:.0%})"
                )
            finally:
                db.close()
        except Exception as exc:
            logger.error(
                "Error handling message in adapter %s: %s",
                self.connection_id,
                exc,
            )
            return "Sorry, something went wrong processing your message."

    @property
    def is_running(self) -> bool:
        return self._running
