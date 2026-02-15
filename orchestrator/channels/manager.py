"""
Channel Manager (PRD-55 US-019)
================================
Manages the lifecycle of all channel adapters.

On startup, ``start_all()`` loads every active ``ChannelConnection``
from the database and spins up the matching adapter.  Individual
adapters can be started / stopped at runtime (e.g. when a user
adds or removes a connection through the UI).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .base import BaseChannelAdapter

logger = logging.getLogger(__name__)


class ChannelManager:
    """Manages all active channel adapters."""

    def __init__(self):
        self._adapters: Dict[str, BaseChannelAdapter] = {}  # connection_id -> adapter

    # ------------------------------------------------------------------
    # Bulk lifecycle
    # ------------------------------------------------------------------

    async def start_all(self):
        """Load all active connections from DB and start their adapters."""
        from core.database.database import SessionLocal

        db = SessionLocal()
        try:
            from core.models.channels import ChannelConnection

            connections = (
                db.query(ChannelConnection)
                .filter(ChannelConnection.status == "active")
                .all()
            )

            for conn in connections:
                await self.start_adapter(
                    connection_id=str(conn.id),
                    workspace_id=str(conn.workspace_id),
                    platform=conn.platform,
                    config=conn.config or {},
                )
        except Exception as e:
            logger.error("[ChannelManager] Failed to load connections: %s", e)
        finally:
            db.close()

    async def stop_all(self):
        """Stop all running adapters."""
        for conn_id, adapter in list(self._adapters.items()):
            try:
                await adapter.stop()
                logger.info("[ChannelManager] Stopped adapter %s", conn_id)
            except Exception as e:
                logger.error("[ChannelManager] Error stopping %s: %s", conn_id, e)
        self._adapters.clear()

    # ------------------------------------------------------------------
    # Single-adapter lifecycle
    # ------------------------------------------------------------------

    async def start_adapter(
        self,
        connection_id: str,
        workspace_id: str,
        platform: str,
        config: dict,
    ):
        """Start (or restart) a specific adapter."""
        # Stop existing instance if already running
        if connection_id in self._adapters:
            await self._adapters[connection_id].stop()

        adapter = self._create_adapter(connection_id, workspace_id, platform, config)
        if adapter is None:
            return

        try:
            await adapter.start()
            self._adapters[connection_id] = adapter
            logger.info(
                "[ChannelManager] Started %s adapter for connection %s",
                platform,
                connection_id,
            )
        except Exception as e:
            logger.error("[ChannelManager] Failed to start %s: %s", platform, e)

    async def stop_adapter(self, connection_id: str):
        """Stop a specific adapter by connection ID."""
        adapter = self._adapters.pop(connection_id, None)
        if adapter:
            await adapter.stop()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def _create_adapter(
        self,
        connection_id: str,
        workspace_id: str,
        platform: str,
        config: dict,
    ) -> Optional[BaseChannelAdapter]:
        """Create a platform-specific adapter instance.

        Uses lazy imports so missing optional dependencies (e.g.
        python-telegram-bot) only surface when the adapter is actually
        requested.
        """
        try:
            if platform == "telegram":
                from .telegram_adapter import TelegramAdapter

                return TelegramAdapter(connection_id, workspace_id, config)
            elif platform == "slack":
                from .slack_adapter import SlackAdapter

                return SlackAdapter(connection_id, workspace_id, config)
            elif platform == "discord":
                from .discord_adapter import DiscordAdapter

                return DiscordAdapter(connection_id, workspace_id, config)
            else:
                logger.warning("[ChannelManager] Unknown platform: %s", platform)
                return None
        except ImportError as e:
            logger.warning(
                "[ChannelManager] %s adapter dependencies not installed: %s",
                platform,
                e,
            )
            return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a status snapshot of all managed adapters."""
        return {
            "total_adapters": len(self._adapters),
            "adapters": {
                conn_id: {
                    "platform": type(adapter).__name__,
                    "running": adapter.is_running,
                }
                for conn_id, adapter in self._adapters.items()
            },
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_channel_manager: Optional[ChannelManager] = None


def get_channel_manager() -> ChannelManager:
    """Return the global ChannelManager singleton."""
    global _channel_manager
    if _channel_manager is None:
        _channel_manager = ChannelManager()
    return _channel_manager
