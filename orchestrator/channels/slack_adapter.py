"""
Slack Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges Slack workspace messages into the universal routing pipeline
using slack-bolt (async) with Socket Mode.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional
from uuid import UUID

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)


class SlackAdapter(BaseChannelAdapter):
    """Adapter for Slack workspaces via slack-bolt async + Socket Mode."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._app = None
        self._handler = None  # AsyncSocketModeHandler
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.adapter.socket_mode.async_handler import (
            AsyncSocketModeHandler,
        )

        bot_token = self.config.get("bot_token")
        signing_secret = self.config.get("signing_secret")
        app_token = self.config.get("app_token")  # xapp-... for socket mode

        if not bot_token:
            raise ValueError("Slack adapter requires 'bot_token' in config")
        if not app_token:
            raise ValueError("Slack adapter requires 'app_token' for Socket Mode")

        self._app = AsyncApp(
            token=bot_token,
            signing_secret=signing_secret or "",
        )

        # Register message listener (ignores bot messages automatically
        # when the event doesn't contain a bot_id)
        @self._app.event("message")
        async def _on_message(event: dict, say, client):
            await self._handle_event(event, say, client)

        self._handler = AsyncSocketModeHandler(self._app, app_token)
        self._task = asyncio.create_task(self._handler.start_async())
        self._running = True
        logger.info("Slack adapter %s started (socket mode)", self.connection_id)

    async def stop(self) -> None:
        if self._handler:
            try:
                await self._handler.close_async()
            except Exception as exc:
                logger.warning("Error closing Slack socket handler: %s", exc)
        if self._task and not self._task.done():
            self._task.cancel()
        self._running = False
        logger.info("Slack adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        import httpx

        bot_token = self.config.get("bot_token", "")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {bot_token}"},
            )
            data = resp.json()

        if not data.get("ok"):
            raise ConnectionError(
                f"Slack auth.test failed: {data.get('error', 'unknown error')}"
            )
        return {
            "platform": "slack",
            "team": data.get("team"),
            "team_id": data.get("team_id"),
            "bot_user_id": data.get("user_id"),
            "bot_name": data.get("user"),
        }

    # ------------------------------------------------------------------
    # Incoming messages
    # ------------------------------------------------------------------

    async def _handle_event(
        self, event: dict, say: Any, client: Any
    ) -> None:
        """Process a Slack ``message`` event."""
        # Filter out bot messages
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        text = event.get("text", "")
        if not text:
            return

        # Route through the universal pipeline
        response_text = await self.handle_message(event)

        # Reply in thread only if the original message was in a thread
        thread_ts = event.get("thread_ts")
        channel = event.get("channel", "")

        if response_text:
            if thread_ts:
                await self.send_message(
                    channel, response_text, thread_ts=thread_ts, _client=client
                )
            else:
                await self.send_message(
                    channel, response_text, _client=client
                )

        self._increment_message_count()

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        """Post a message to a Slack channel, optionally in a thread."""
        client = kwargs.get("_client")
        thread_ts = kwargs.get("thread_ts")

        if client:
            await client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts,
            )
        elif self._app:
            await self._app.client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts,
            )
        else:
            logger.error("No Slack client available for sending")

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        event = platform_message  # dict from Slack event

        return RequestEnvelope(
            source=ChannelSource.SLACK,
            content=event.get("text", ""),
            raw_payload={
                "ts": event.get("ts"),
                "channel": event.get("channel"),
                "thread_ts": event.get("thread_ts"),
                "team": event.get("team"),
            },
            user=RequestUser(
                id=event.get("user"),
                name=event.get("user"),  # Display name resolved later
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "slack",
                "connection_id": self.connection_id,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
