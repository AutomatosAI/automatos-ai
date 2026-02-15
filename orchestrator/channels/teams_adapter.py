"""
Microsoft Teams Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges Microsoft Teams Bot Framework messages into the universal routing
pipeline using httpx for async HTTP calls to the Bot Framework REST API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)

_TEAMS_MAX_LEN = 28000  # Teams supports up to ~28KB per message


class TeamsAdapter(BaseChannelAdapter):
    """Adapter for Microsoft Teams via Bot Framework REST API."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._app_id = config.get("app_id", "")
        self._app_password = config.get("app_password", "")
        self._token: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._app_id or not self._app_password:
            raise ValueError("Teams adapter requires 'app_id' and 'app_password'")

        await self._refresh_token()
        self._running = True
        logger.info("Teams adapter %s started (webhook mode)", self.connection_id)

    async def stop(self) -> None:
        self._token = None
        self._running = False
        logger.info("Teams adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _refresh_token(self) -> None:
        import httpx

        url = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self._app_id,
            "client_secret": self._app_password,
            "scope": "https://api.botframework.com/.default",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, data=data)
            result = resp.json()

        if "access_token" not in result:
            raise ConnectionError(f"Teams auth failed: {result.get('error_description', 'unknown')}")

        self._token = result["access_token"]

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        await self._refresh_token()
        return {
            "platform": "teams",
            "app_id": self._app_id,
            "authenticated": self._token is not None,
        }

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        import httpx

        service_url = kwargs.get("service_url", "https://smba.trafficmanager.net/teams/")
        conversation_id = kwargs.get("conversation_id", channel_id)

        if not self._token:
            await self._refresh_token()

        url = f"{service_url}v3/conversations/{conversation_id}/activities"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        chunks = self._chunk_text(text, _TEAMS_MAX_LEN)
        async with httpx.AsyncClient(timeout=30) as client:
            for chunk in chunks:
                payload = {
                    "type": "message",
                    "text": chunk,
                }
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 401:
                    await self._refresh_token()
                    headers["Authorization"] = f"Bearer {self._token}"
                    await client.post(url, json=payload, headers=headers)

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        activity = platform_message
        sender = activity.get("from", {})

        return RequestEnvelope(
            source=ChannelSource.TEAMS,
            content=activity.get("text", ""),
            raw_payload=activity,
            user=RequestUser(
                id=sender.get("id"),
                name=sender.get("name"),
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "teams",
                "connection_id": self.connection_id,
                "service_url": activity.get("serviceUrl", ""),
                "conversation_id": activity.get("conversation", {}).get("id", ""),
                "reply_to_id": activity.get("id"),
            },
        )

    # ------------------------------------------------------------------
    # Webhook handler
    # ------------------------------------------------------------------

    async def handle_webhook(self, activity: Dict[str, Any]) -> Optional[str]:
        if activity.get("type") != "message":
            return None

        response = await self.handle_message(activity)
        if response:
            await self.send_message(
                channel_id=activity.get("conversation", {}).get("id", ""),
                text=response,
                service_url=activity.get("serviceUrl", ""),
                conversation_id=activity.get("conversation", {}).get("id", ""),
            )
            self._increment_message_count()
        return response

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
