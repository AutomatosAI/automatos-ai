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

        1. Convert platform message to RequestEnvelope
        2. Route via UniversalRouter -> RoutingDecision (agent_id)
        3. Execute via AgentFactory.execute_with_prompt()
        4. Send result back via send_message()
        5. Update activity stats on the channel connection
        """
        try:
            envelope = self._to_envelope(platform_message)
            if not envelope:
                return

            # Lazy imports to avoid circular dependency at module load
            from core.routing.engine import UniversalRouter
            from core.database.database import SessionLocal
            from modules.agents.factory.agent_factory import AgentFactory

            db = SessionLocal()
            try:
                # ── Route ──
                router = UniversalRouter(db=db)
                decision = await router.route(envelope)

                if not decision or not decision.agent_id:
                    logger.warning(
                        "[Channel:%s] No route found for message, sending fallback",
                        self.connection_id,
                    )
                    reply_channel = (
                        platform_message.get("reply_channel_id")
                        or platform_message.get("channel_id")
                    )
                    if reply_channel:
                        await self.send_message(
                            reply_channel,
                            "I'm not sure how to handle that request. Please try rephrasing.",
                        )
                    return

                # ── Execute ──
                factory = AgentFactory(db_session=db)
                result = await factory.execute_with_prompt(
                    agent=decision.agent_id,
                    prompt=envelope.content,
                    context={
                        "source": envelope.source.value if hasattr(envelope.source, "value") else str(envelope.source),
                        "workspace_id": str(envelope.workspace_id),
                        "connection_id": self.connection_id,
                    },
                )

                response_text = (result or {}).get("response", "")
                if not response_text:
                    response_text = (result or {}).get("content", "I processed your request but have no text to return.")

                # ── Respond ──
                reply_channel = (
                    platform_message.get("reply_channel_id")
                    or platform_message.get("channel_id")
                )
                if reply_channel and response_text:
                    await self.send_message(reply_channel, response_text)

                # ── Update activity stats ──
                await self._update_activity_stats(db)

            finally:
                db.close()

        except Exception as e:
            logger.error(
                "[Channel:%s] Failed to handle message: %s",
                self.connection_id,
                e,
            )

    async def _update_activity_stats(self, db) -> None:
        """Increment message_count and set last_activity_at on the connection row."""
        try:
            from sqlalchemy import text
            from datetime import datetime

            db.execute(
                text(
                    "UPDATE channel_connections "
                    "SET message_count = COALESCE(message_count, 0) + 1, "
                    "    last_activity_at = :now, "
                    "    updated_at = :now "
                    "WHERE id = :conn_id"
                ),
                {"conn_id": self.connection_id, "now": datetime.utcnow()},
            )
            db.commit()
        except Exception as e:
            logger.warning("[Channel:%s] Failed to update activity stats: %s", self.connection_id, e)

    @abstractmethod
    def _to_envelope(self, platform_message: Dict[str, Any]) -> Optional["RequestEnvelope"]:
        """Convert a platform-specific message dict to a RequestEnvelope.

        Return ``None`` to silently skip messages that should be ignored
        (e.g. bot's own messages).
        """
        ...
