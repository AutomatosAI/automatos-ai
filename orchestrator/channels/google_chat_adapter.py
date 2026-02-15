"""
Google Chat Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges Google Chat (Workspace) messages into the universal routing
pipeline using httpx for async HTTP calls to the Google Chat API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)

_GCHAT_MAX_LEN = 4096


class GoogleChatAdapter(BaseChannelAdapter):
    """Adapter for Google Chat via Chat API (webhook + service account)."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._service_account_json = config.get("service_account_json", "")
        self._credentials = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._service_account_json:
            raise ValueError("Google Chat adapter requires 'service_account_json'")

        await self._authenticate()
        self._running = True
        logger.info("Google Chat adapter %s started (webhook mode)", self.connection_id)

    async def stop(self) -> None:
        self._credentials = None
        self._running = False
        logger.info("Google Chat adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _authenticate(self) -> None:
        """Authenticate via service account."""
        import json
        import asyncio

        try:
            from google.oauth2 import service_account as sa
            from google.auth.transport.requests import Request

            info = json.loads(self._service_account_json)
            creds = sa.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/chat.bot"],
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: creds.refresh(Request()))
            self._credentials = creds
        except ImportError:
            logger.warning("google-auth not installed — Google Chat adapter limited to webhook mode")
        except Exception as exc:
            logger.error("Google Chat auth failed: %s", exc)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        await self._authenticate()
        return {
            "platform": "google_chat",
            "authenticated": self._credentials is not None,
            "service_account": bool(self._service_account_json),
        }

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        import httpx
        import asyncio

        if not self._credentials:
            logger.error("No credentials for Google Chat send")
            return

        url = f"https://chat.googleapis.com/v1/{channel_id}/messages"

        chunks = self._chunk_text(text, _GCHAT_MAX_LEN)
        async with httpx.AsyncClient(timeout=30) as client:
            for chunk in chunks:
                payload = {"text": chunk}
                for attempt in range(3):
                    headers = {
                        "Authorization": f"Bearer {self._credentials.token}",
                        "Content-Type": "application/json",
                    }
                    try:
                        resp = await client.post(url, json=payload, headers=headers)
                        if resp.status_code == 401:
                            logger.warning("Google Chat 401 — refreshing credentials")
                            await self._authenticate()
                            continue
                        if resp.status_code == 429 or resp.status_code >= 500:
                            wait = (attempt + 1) * 2
                            logger.warning(
                                "Google Chat %s for %s — retrying in %ds",
                                resp.status_code, channel_id, wait,
                            )
                            await asyncio.sleep(wait)
                            continue
                        if resp.status_code >= 400:
                            logger.error(
                                "Google Chat send failed (channel=%s, status=%s): %s",
                                channel_id, resp.status_code, resp.text,
                            )
                        break  # success or non-retryable error
                    except httpx.RequestError as exc:
                        logger.error(
                            "Google Chat request error (channel=%s): %s",
                            channel_id, exc,
                        )
                        break

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        msg = platform_message
        sender = msg.get("sender", {})

        return RequestEnvelope(
            source=ChannelSource.GOOGLE_CHAT,
            content=msg.get("argumentText", msg.get("text", "")),
            raw_payload=msg,
            user=RequestUser(
                id=sender.get("name"),
                name=sender.get("displayName"),
                email=sender.get("email"),
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "google_chat",
                "connection_id": self.connection_id,
                "space": msg.get("space", {}).get("name", ""),
                "thread": msg.get("thread", {}).get("name", ""),
            },
        )

    # ------------------------------------------------------------------
    # Webhook handler
    # ------------------------------------------------------------------

    async def handle_webhook(self, event: Dict[str, Any]) -> Optional[Dict]:
        """Process incoming Google Chat event, return response payload."""
        event_type = event.get("type", "")
        if event_type not in ("MESSAGE", "ADDED_TO_SPACE"):
            return None

        if event_type == "ADDED_TO_SPACE":
            return {"text": "Hello! I'm your Automatos assistant. How can I help?"}

        message = event.get("message", {})
        response = await self.handle_message(message)
        self._increment_message_count()

        if response:
            return {"text": response}
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
