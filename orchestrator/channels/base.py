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
from typing import Any, Dict, List, Optional
from uuid import UUID
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
    # PRD-127: Attachment handling
    # ------------------------------------------------------------------

    async def upload_attachment(
        self,
        content: bytes,
        filename: str,
        mime_type: Optional[str] = None,
    ) -> Optional[str]:
        """Upload media bytes to AttachmentStore and return the attachment_id.

        Subclasses call this when they receive inbound images/files from the
        platform. Returns None on failure (logged, doesn't crash the pipeline).
        """
        try:
            from modules.attachments.store import get_attachment_store

            store = get_attachment_store()
            ref = await store.put(
                workspace_id=UUID(self.workspace_id),
                uploaded_by=f"channel:{self.connection_id}",
                filename=filename,
                content=content,
                declared_mime=mime_type,
            )
            logger.debug(
                "[Channel:%s] Uploaded attachment %s (%s bytes)",
                self.connection_id,
                ref.attachment_id,
                len(content),
            )
            return str(ref.attachment_id)
        except Exception as e:
            logger.warning(
                "[Channel:%s] Failed to upload attachment %s: %s",
                self.connection_id,
                filename,
                e,
            )
            return None

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

        PRD-127: If platform_message contains 'attachment_ids' (list of UUIDs),
        they are forwarded to execute_with_prompt for multimodal resolution.
        Subclass adapters should call upload_attachment() for inbound media
        and populate this field before calling handle_message().

        PRD-142 W3-S13: Emits the ``channels`` primitive heartbeat finding
        at the terminal boundary of the turn — green on a clean pipeline
        traversal, down on a caught exception. The emit is best-effort
        and NEVER raises (heartbeat must not break message handling).
        """
        try:
            envelope = self._to_envelope(platform_message)
            if not envelope:
                return

            # PRD-127: Extract attachment_ids from platform_message
            attachment_ids: List[str] = platform_message.get("attachment_ids", [])

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
                    self._emit_channel_heartbeat(success=True, detail="no_route_fallback")
                    return

                # ── Execute ──
                # PRD-234 S3: a Claude Code agent's mention becomes a board ticket the
                # paired host runs as the user's own session; the channel hears that
                # it is queued (the factory refuses cli agents by design).
                from services.cli_ticket_lane import file_cli_ticket, is_cli_agent, queued_line, source_id_for
                if is_cli_agent(db, decision.agent_id):
                    _text = (envelope.content or "").strip()
                    _first = _text.splitlines()[0][:80] if _text else "request"
                    _src = envelope.source.value if hasattr(envelope.source, "value") else str(envelope.source)
                    _msg_key = platform_message.get("message_id") or platform_message.get("ts") or platform_message.get("id") or str(id(envelope))
                    ticket = file_cli_ticket(
                        db, workspace_id=envelope.workspace_id, agent_id=decision.agent_id,
                        title=f"{_src}: {_first}", prompt=envelope.content,
                        source_type="channel", source_id=source_id_for("channel", f"{self.connection_id}:{_msg_key}"),
                    )
                    response_text = queued_line(ticket)
                else:
                    factory = AgentFactory(db_session=db)
                    result = await factory.execute_with_prompt(
                        agent=decision.agent_id,
                        prompt=envelope.content,
                        context={
                            "source": envelope.source.value if hasattr(envelope.source, "value") else str(envelope.source),
                            "workspace_id": str(envelope.workspace_id),
                            "connection_id": self.connection_id,
                        },
                        attachment_ids=attachment_ids if attachment_ids else None,  # PRD-127
                    )
                    response_text = (result or {}).get("result") or (result or {}).get("response") or (result or {}).get("content") or ""

                # ── Respond ──
                reply_channel = (
                    platform_message.get("reply_channel_id")
                    or platform_message.get("channel_id")
                )
                if reply_channel and response_text:
                    await self.send_message(reply_channel, response_text)

                # ── Update activity stats ──
                await self._update_activity_stats(db)

                self._emit_channel_heartbeat(success=True, detail="ok")

            finally:
                db.close()

        except Exception as e:
            logger.error(
                "[Channel:%s] Failed to handle message: %s",
                self.connection_id,
                e,
            )
            self._emit_channel_heartbeat(success=False, detail=f"{type(e).__name__}: {e}")

    def _emit_channel_heartbeat(self, *, success: bool, detail: str = "") -> None:
        """Emit the ``channels`` primitive finding for this adapter's
        workspace (PRD-142 W3-S13). Lazy import keeps the heartbeat
        wiring opt-in at module load and avoids a circular import."""
        try:
            from channels.primitive_heartbeat import _emit_channels_primitive

            _emit_channels_primitive(
                self.workspace_id,
                success=success,
                detail=detail,
            )
        except Exception:  # noqa: BLE001 — best-effort; never break handle_message
            logger.error(
                "[Channel:%s] heartbeat emit failed", self.connection_id, exc_info=True,
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
