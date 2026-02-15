"""
Matrix Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges Matrix protocol messages into the universal routing pipeline
using the matrix-nio async client library.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)

_MATRIX_MAX_LEN = 16000  # Matrix has no strict limit, but chunk large messages


class MatrixAdapter(BaseChannelAdapter):
    """Adapter for Matrix via matrix-nio."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._homeserver = config.get("homeserver", "https://matrix.org")
        self._user_id = config.get("user_id", "")
        self._access_token = config.get("access_token", "")
        self._client = None
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        from nio import AsyncClient, RoomMessageText

        if not self._access_token or not self._user_id:
            raise ValueError("Matrix adapter requires 'user_id' and 'access_token'")

        self._client = AsyncClient(self._homeserver, self._user_id)
        self._client.access_token = self._access_token

        self._client.add_event_callback(self._on_message, RoomMessageText)

        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info("Matrix adapter %s started (%s)", self.connection_id, self._homeserver)

    async def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Matrix adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        import httpx

        url = f"{self._homeserver}/_matrix/client/v3/account/whoami"
        headers = {"Authorization": f"Bearer {self._access_token}"}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            data = resp.json()

        if "user_id" not in data:
            raise ConnectionError(f"Matrix auth failed: {data.get('error', 'unknown')}")

        return {
            "platform": "matrix",
            "homeserver": self._homeserver,
            "user_id": data["user_id"],
            "device_id": data.get("device_id", ""),
        }

    # ------------------------------------------------------------------
    # Sync loop
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        """Continuously sync with the Matrix homeserver."""
        while self._running and self._client:
            try:
                resp = await self._client.sync(timeout=30000, full_state=False)
                if hasattr(resp, "transport_response"):
                    # Handle errors
                    pass
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Matrix sync error: %s", exc)
                await asyncio.sleep(5)

    async def _on_message(self, room: Any, event: Any) -> None:
        """Handle incoming Matrix room messages."""
        # Skip our own messages
        if event.sender == self._user_id:
            return

        platform_msg = {
            "sender": event.sender,
            "text": event.body,
            "room_id": room.room_id,
            "event_id": event.event_id,
            "room_name": room.display_name if hasattr(room, "display_name") else room.room_id,
        }

        response = await self.handle_message(platform_msg)
        if response:
            await self.send_message(room.room_id, response)

        self._increment_message_count()

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        if not self._client:
            return

        chunks = self._chunk_text(text, _MATRIX_MAX_LEN)
        for chunk in chunks:
            from nio import RoomSendError

            content = {
                "msgtype": "m.text",
                "body": chunk,
            }
            try:
                await self._client.room_send(
                    room_id=channel_id,
                    message_type="m.room.message",
                    content=content,
                )
            except Exception as exc:
                logger.error("Matrix send error: %s", exc)

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        msg = platform_message

        return RequestEnvelope(
            source=ChannelSource.MATRIX,
            content=msg.get("text", ""),
            raw_payload=msg,
            user=RequestUser(
                id=msg.get("sender", ""),
                name=msg.get("sender", ""),
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "matrix",
                "connection_id": self.connection_id,
                "room_id": msg.get("room_id", ""),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, max_len: int) -> List[str]:
        if len(text) <= max_len:
            return [text]
        chunks = []
        while text:
            chunks.append(text[:max_len])
            text = text[max_len:]
        return chunks

    def _increment_message_count(self) -> None:
        try:
            from core.database.database import SessionLocal
            from sqlalchemy import text as sa_text

            db = SessionLocal()
            try:
                db.execute(
                    sa_text(
                        "UPDATE channel_connections "
                        "SET message_count = message_count + 1, "
                        "last_activity_at = NOW() "
                        "WHERE id = :cid::uuid"
                    ),
                    {"cid": self.connection_id},
                )
                db.commit()
            finally:
                db.close()
        except Exception as exc:
            logger.debug("Failed to increment message count: %s", exc)
