"""
Signal Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges Signal messenger messages into the universal routing pipeline
using signal-cli REST API (https://github.com/bbernhard/signal-cli-rest-api).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)

_SIGNAL_MAX_LEN = 4096


class SignalAdapter(BaseChannelAdapter):
    """Adapter for Signal via signal-cli REST API."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._api_url = config.get("api_url", "http://localhost:8080")
        self._phone_number = config.get("phone_number", "")
        self._poll_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._phone_number:
            raise ValueError("Signal adapter requires 'phone_number'")

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_messages())
        logger.info("Signal adapter %s started (polling %s)", self.connection_id, self._api_url)

    async def stop(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        logger.info("Signal adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        import httpx

        url = f"{self._api_url}/v1/about"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            data = resp.json()

        # Check registered accounts
        accts_url = f"{self._api_url}/v1/accounts"
        async with httpx.AsyncClient(timeout=10) as client:
            accts_resp = await client.get(accts_url)
            accounts = accts_resp.json() if accts_resp.status_code == 200 else []

        registered = self._phone_number in [a.get("number") for a in accounts] if isinstance(accounts, list) else False

        return {
            "platform": "signal",
            "api_version": data.get("version", "unknown"),
            "phone_number": self._phone_number,
            "registered": registered,
        }

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_messages(self) -> None:
        """Poll signal-cli REST API for incoming messages."""
        import httpx

        url = f"{self._api_url}/v1/receive/{self._phone_number}"
        while self._running:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        messages = resp.json()
                        for msg in messages:
                            await self._process_signal_message(msg)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Signal poll error: %s", exc)

            await asyncio.sleep(2)

    async def _process_signal_message(self, raw: Dict[str, Any]) -> None:
        """Process a single signal-cli message envelope."""
        envelope = raw.get("envelope", {})
        data_msg = envelope.get("dataMessage")
        if not data_msg or not data_msg.get("message"):
            return

        sender = envelope.get("source", "")
        text = data_msg.get("message", "")

        platform_msg = {
            "sender": sender,
            "text": text,
            "timestamp": envelope.get("timestamp"),
            "group": data_msg.get("groupInfo", {}).get("groupId"),
        }

        response = await self.handle_message(platform_msg)
        if response:
            recipient = platform_msg.get("group") or sender
            await self.send_message(recipient, response, is_group=bool(platform_msg.get("group")))

        self._increment_message_count()

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        import httpx

        is_group = kwargs.get("is_group", False)
        url = f"{self._api_url}/v2/send"

        chunks = self._chunk_text(text, _SIGNAL_MAX_LEN)
        async with httpx.AsyncClient(timeout=30) as client:
            for chunk in chunks:
                payload: Dict[str, Any] = {
                    "message": chunk,
                    "number": self._phone_number,
                }
                if is_group:
                    payload["recipients"] = []
                    payload["group_id"] = channel_id
                else:
                    payload["recipients"] = [channel_id]

                await client.post(url, json=payload)

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        msg = platform_message

        return RequestEnvelope(
            source=ChannelSource.SIGNAL,
            content=msg.get("text", ""),
            raw_payload=msg,
            user=RequestUser(
                id=msg.get("sender", ""),
                name=msg.get("sender", ""),
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "signal",
                "connection_id": self.connection_id,
                "reply_to": msg.get("sender", ""),
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
