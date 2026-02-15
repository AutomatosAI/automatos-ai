"""
Channel Manager (PRD-55: Autonomous Assistant Platform).

Manages the lifecycle of all channel adapters -- loading active
connections from the database, starting/stopping individual adapters,
and providing status introspection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)


class ChannelManager:
    """Registry and lifecycle manager for channel adapters."""

    def __init__(self) -> None:
        self._adapters: Dict[str, BaseChannelAdapter] = {}

    # ------------------------------------------------------------------
    # Bulk lifecycle
    # ------------------------------------------------------------------

    async def start_all(self) -> None:
        """Load active ChannelConnection records and start their adapters."""
        from core.database.database import SessionLocal
        from core.models.channels import ChannelConnection

        db = SessionLocal()
        try:
            connections = (
                db.query(ChannelConnection)
                .filter(ChannelConnection.status == "active")
                .all()
            )
            for conn in connections:
                try:
                    adapter = self._create_adapter(conn)
                    await adapter.start()
                    self._adapters[str(conn.id)] = adapter
                    logger.info(
                        "Started %s adapter for connection %s",
                        conn.platform,
                        conn.id,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to start adapter for connection %s: %s",
                        conn.id,
                        exc,
                    )
        finally:
            db.close()

    async def stop_all(self) -> None:
        """Stop every running adapter."""
        for cid, adapter in list(self._adapters.items()):
            try:
                await adapter.stop()
                logger.info("Stopped adapter %s", cid)
            except Exception as exc:
                logger.error("Error stopping adapter %s: %s", cid, exc)
        self._adapters.clear()

    # ------------------------------------------------------------------
    # Single-adapter lifecycle
    # ------------------------------------------------------------------

    async def start_adapter(self, connection_id: str) -> None:
        """Start (or restart) a single adapter by connection ID."""
        if connection_id in self._adapters:
            await self._adapters[connection_id].stop()

        from core.database.database import SessionLocal
        from core.models.channels import ChannelConnection

        db = SessionLocal()
        try:
            conn = (
                db.query(ChannelConnection)
                .filter(ChannelConnection.id == connection_id)
                .first()
            )
            if not conn:
                raise ValueError(f"Connection {connection_id} not found")

            adapter = self._create_adapter(conn)
            await adapter.start()
            self._adapters[connection_id] = adapter

            # Mark active in DB
            conn.status = "active"
            db.commit()
            logger.info("Started adapter for connection %s", connection_id)
        finally:
            db.close()

    async def stop_adapter(self, connection_id: str) -> None:
        """Stop a running adapter."""
        adapter = self._adapters.pop(connection_id, None)
        if adapter:
            await adapter.stop()

        from core.database.database import SessionLocal
        from core.models.channels import ChannelConnection

        db = SessionLocal()
        try:
            conn = (
                db.query(ChannelConnection)
                .filter(ChannelConnection.id == connection_id)
                .first()
            )
            if conn:
                conn.status = "disconnected"
                db.commit()
        finally:
            db.close()

        logger.info("Stopped adapter for connection %s", connection_id)

    # ------------------------------------------------------------------
    # Adapter factory
    # ------------------------------------------------------------------

    @staticmethod
    def _create_adapter(connection: Any) -> BaseChannelAdapter:
        """Instantiate the correct adapter subclass based on *platform*."""
        platform = connection.platform.lower()
        conn_id = str(connection.id)
        ws_id = str(connection.workspace_id)
        cfg = connection.config or {}

        _ADAPTER_MAP = {
            "telegram": ("channels.telegram_adapter", "TelegramAdapter"),
            "slack": ("channels.slack_adapter", "SlackAdapter"),
            "discord": ("channels.discord_adapter", "DiscordAdapter"),
            "whatsapp": ("channels.whatsapp_adapter", "WhatsAppAdapter"),
            "teams": ("channels.teams_adapter", "TeamsAdapter"),
            "google_chat": ("channels.google_chat_adapter", "GoogleChatAdapter"),
            "signal": ("channels.signal_adapter", "SignalAdapter"),
            "imessage": ("channels.imessage_adapter", "IMessageAdapter"),
            "irc": ("channels.irc_adapter", "IRCAdapter"),
            "matrix": ("channels.matrix_adapter", "MatrixAdapter"),
            "line": ("channels.line_adapter", "LINEAdapter"),
        }

        entry = _ADAPTER_MAP.get(platform)
        if not entry:
            raise ValueError(f"Unsupported channel platform: {platform}")

        module_path, class_name = entry
        import importlib
        mod = importlib.import_module(module_path)
        adapter_cls = getattr(mod, class_name)
        return adapter_cls(connection_id=conn_id, workspace_id=ws_id, config=cfg)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a dict of adapter statuses keyed by connection ID."""
        result: Dict[str, Any] = {}
        for cid, adapter in self._adapters.items():
            result[cid] = {
                "platform": adapter.__class__.__name__,
                "running": adapter.is_running,
                "workspace_id": adapter.workspace_id,
            }
        return result


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instance: Optional[ChannelManager] = None


def get_channel_manager() -> ChannelManager:
    """Return the singleton ChannelManager instance."""
    global _instance
    if _instance is None:
        _instance = ChannelManager()
    return _instance
