"""
LINE Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges LINE Messaging API messages into the universal routing pipeline
using httpx for async HTTP calls to the LINE Platform API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)

_LINE_MAX_LEN = 5000


class LINEAdapter(BaseChannelAdapter):
    """Adapter for LINE Messaging API (webhook-based)."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._channel_access_token = config.get("channel_access_token", "")
        self._channel_secret = config.get("channel_secret", "")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._channel_access_token:
            raise ValueError("LINE adapter requires 'channel_access_token'")

        self._running = True
        logger.info("LINE adapter %s started (webhook mode)", self.connection_id)

    async def stop(self) -> None:
        self._running = False
        logger.info("LINE adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        import httpx

        url = "https://api.line.me/v2/bot/info"
        headers = {"Authorization": f"Bearer {self._channel_access_token}"}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            data = resp.json()

        if "userId" not in data and "botUserId" not in data:
            raise ConnectionError(f"LINE API error: {data.get('message', 'invalid token')}")

        return {
            "platform": "line",
            "bot_id": data.get("userId") or data.get("botUserId", ""),
            "display_name": data.get("displayName", ""),
            "basic_id": data.get("basicId", ""),
        }

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        """Send a text message via LINE push API."""
        import httpx

        url = "https://api.line.me/v2/bot/message/push"
        headers = {
            "Authorization": f"Bearer {self._channel_access_token}",
            "Content-Type": "application/json",
        }

        chunks = self._chunk_text(text, _LINE_MAX_LEN)
        messages = [{"type": "text", "text": chunk} for chunk in chunks[:5]]  # LINE max 5 messages per push

        payload = {
            "to": channel_id,
            "messages": messages,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code >= 400:
                logger.error("LINE push failed: %s", resp.text)

    async def _reply_message(self, reply_token: str, text: str) -> None:
        """Reply using a reply token (preferred — free, vs push which costs)."""
        import httpx

        url = "https://api.line.me/v2/bot/message/reply"
        headers = {
            "Authorization": f"Bearer {self._channel_access_token}",
            "Content-Type": "application/json",
        }

        chunks = self._chunk_text(text, _LINE_MAX_LEN)
        messages = [{"type": "text", "text": chunk} for chunk in chunks[:5]]

        payload = {
            "replyToken": reply_token,
            "messages": messages,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(url, json=payload, headers=headers)

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        event = platform_message
        source = event.get("source", {})

        return RequestEnvelope(
            source=ChannelSource.LINE,
            content=event.get("message", {}).get("text", ""),
            raw_payload=event,
            user=RequestUser(
                id=source.get("userId", ""),
                name=source.get("userId", ""),
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "line",
                "connection_id": self.connection_id,
                "reply_token": event.get("replyToken", ""),
                "source_type": source.get("type", ""),
                "group_id": source.get("groupId", ""),
            },
        )

    # ------------------------------------------------------------------
    # Webhook handler
    # ------------------------------------------------------------------

    async def handle_webhook(self, body: Dict[str, Any]) -> Optional[str]:
        """Process LINE webhook events."""
        events = body.get("events", [])

        for event in events:
            if event.get("type") != "message":
                continue
            if event.get("message", {}).get("type") != "text":
                continue

            response = await self.handle_message(event)
            if response:
                reply_token = event.get("replyToken")
                if reply_token:
                    await self._reply_message(reply_token, response)
                else:
                    user_id = event.get("source", {}).get("userId", "")
                    if user_id:
                        await self.send_message(user_id, response)

            self._increment_message_count()

        return "ok"

    # ------------------------------------------------------------------
    # Signature verification
    # ------------------------------------------------------------------

    def verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify LINE webhook signature using channel secret."""
        import hashlib
        import hmac
        import base64

        if not self._channel_secret:
            return True

        expected = hmac.new(
            self._channel_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).digest()

        return base64.b64encode(expected).decode("utf-8") == signature

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
