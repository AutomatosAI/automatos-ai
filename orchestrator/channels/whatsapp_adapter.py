"""
WhatsApp Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges WhatsApp Business Cloud API messages into the universal routing
pipeline using httpx for async HTTP calls to Meta's Graph API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)

_WA_MAX_LEN = 4096
_GRAPH_API_BASE = "https://graph.facebook.com/v21.0"


class WhatsAppAdapter(BaseChannelAdapter):
    """Adapter for WhatsApp Business via Meta Cloud API."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._phone_number_id = config.get("phone_number_id", "")
        self._access_token = config.get("access_token", "")
        self._verify_token = config.get("verify_token", "")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """WhatsApp uses webhooks — mark adapter as running.

        The actual message receiving happens via the webhook endpoint
        registered in the FastAPI app. This adapter handles outbound
        messages and message-to-envelope conversion.
        """
        if not self._access_token or not self._phone_number_id:
            raise ValueError(
                "WhatsApp adapter requires 'access_token' and 'phone_number_id'"
            )
        self._running = True
        logger.info("WhatsApp adapter %s started (webhook mode)", self.connection_id)

    async def stop(self) -> None:
        self._running = False
        logger.info("WhatsApp adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        import httpx

        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}"
        headers = {"Authorization": f"Bearer {self._access_token}"}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            data = resp.json()

        if "error" in data:
            raise ConnectionError(
                f"WhatsApp API error: {data['error'].get('message', 'unknown')}"
            )

        return {
            "platform": "whatsapp",
            "phone_number_id": self._phone_number_id,
            "display_phone": data.get("display_phone_number", ""),
            "verified_name": data.get("verified_name", ""),
            "quality_rating": data.get("quality_rating", ""),
        }

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        """Send a text message to a WhatsApp user by phone number."""
        import httpx

        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        chunks = self._chunk_text(text, _WA_MAX_LEN)
        async with httpx.AsyncClient(timeout=30) as client:
            for chunk in chunks:
                payload = {
                    "messaging_product": "whatsapp",
                    "to": channel_id,
                    "type": "text",
                    "text": {"body": chunk},
                }
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code >= 400:
                    logger.error("WhatsApp send failed: %s", resp.text)

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        msg = platform_message
        sender = msg.get("from", "")
        text = ""
        if msg.get("type") == "text":
            text = msg.get("text", {}).get("body", "")

        contact = msg.get("_contact", {})

        return RequestEnvelope(
            source=ChannelSource.WHATSAPP,
            content=text,
            raw_payload=msg,
            user=RequestUser(
                id=sender,
                name=contact.get("profile", {}).get("name"),
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "whatsapp",
                "connection_id": self.connection_id,
                "reply_to": sender,
            },
        )

    # ------------------------------------------------------------------
    # Webhook handler (called from FastAPI webhook route)
    # ------------------------------------------------------------------

    async def handle_webhook(self, payload: Dict[str, Any]) -> Optional[str]:
        """Process an incoming WhatsApp webhook payload."""
        try:
            entry = payload.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            messages = value.get("messages", [])
            contacts = value.get("contacts", [])

            if not messages:
                return None

            for i, msg in enumerate(messages):
                contact = contacts[i] if i < len(contacts) else {}
                msg["_contact"] = contact
                response = await self.handle_message(msg)
                if response:
                    await self.send_message(msg.get("from", ""), response)
                self._increment_message_count()

            return "ok"
        except Exception as exc:
            logger.error("WhatsApp webhook error: %s", exc)
            return None

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
