"""
iMessage Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges iMessage via BlueBubbles REST API into the universal routing
pipeline. Requires a BlueBubbles server running on macOS.
See: https://bluebubbles.app/
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)

_IMSG_MAX_LEN = 10000  # iMessage has no hard limit, but chunk for sanity


class IMessageAdapter(BaseChannelAdapter):
    """Adapter for iMessage via BlueBubbles REST API."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._server_url = config.get("server_url", "http://localhost:1234")
        self._password = config.get("password", "")
        self._poll_task: Optional[asyncio.Task] = None
        self._last_message_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._server_url or not self._password:
            raise ValueError("iMessage adapter requires 'server_url' and 'password'")

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_messages())
        logger.info("iMessage adapter %s started (BlueBubbles: %s)", self.connection_id, self._server_url)

    async def stop(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        logger.info("iMessage adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        import httpx

        url = f"{self._server_url}/api/v1/server/info"
        params = {"password": self._password}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            data = resp.json()

        if data.get("status") != 200:
            raise ConnectionError(f"BlueBubbles error: {data.get('message', 'unknown')}")

        info = data.get("data", {})
        return {
            "platform": "imessage",
            "server_version": info.get("server_version", "unknown"),
            "os_version": info.get("os_version", "unknown"),
            "private_api": info.get("private_api", False),
        }

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_messages(self) -> None:
        """Poll BlueBubbles for new messages."""
        import httpx
        import time

        # Start from now
        self._last_timestamp = int(time.time() * 1000)

        while self._running:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    url = f"{self._server_url}/api/v1/message"
                    params = {
                        "password": self._password,
                        "after": self._last_timestamp,
                        "limit": 50,
                        "sort": "ASC",
                    }
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        messages = data.get("data", [])
                        for msg in messages:
                            if msg.get("isFromMe"):
                                continue
                            await self._process_message(msg)
                            ts = msg.get("dateCreated", self._last_timestamp)
                            if ts > self._last_timestamp:
                                self._last_timestamp = ts
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("BlueBubbles poll error: %s", exc)

            await asyncio.sleep(3)

    async def _process_message(self, msg: Dict[str, Any]) -> None:
        """Process a single BlueBubbles message."""
        text = msg.get("text", "")
        if not text:
            return

        handle = msg.get("handle", {})
        sender = handle.get("address", "") if isinstance(handle, dict) else ""

        platform_msg = {
            "text": text,
            "sender": sender,
            "guid": msg.get("guid", ""),
            "chat_guid": msg.get("chats", [{}])[0].get("guid", "") if msg.get("chats") else "",
            "date_created": msg.get("dateCreated"),
        }

        response = await self.handle_message(platform_msg)
        if response:
            chat_guid = platform_msg.get("chat_guid", "")
            if chat_guid:
                await self.send_message(chat_guid, response)

        self._increment_message_count()

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        """Send a message via BlueBubbles to a chat GUID."""
        import httpx

        url = f"{self._server_url}/api/v1/message/text"
        chunks = self._chunk_text(text, _IMSG_MAX_LEN)

        async with httpx.AsyncClient(timeout=30) as client:
            for chunk in chunks:
                payload = {
                    "chatGuid": channel_id,
                    "message": chunk,
                    "method": "private-api",
                }
                params = {"password": self._password}
                await client.post(url, json=payload, params=params)

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        msg = platform_message

        return RequestEnvelope(
            source=ChannelSource.IMESSAGE,
            content=msg.get("text", ""),
            raw_payload=msg,
            user=RequestUser(
                id=msg.get("sender", ""),
                name=msg.get("sender", ""),
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "imessage",
                "connection_id": self.connection_id,
                "chat_guid": msg.get("chat_guid", ""),
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
