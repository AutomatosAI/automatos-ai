"""
Slack Channel Adapter (PRD-55 US-021)
======================================

Uses slack-bolt for async Slack bot integration.
Converts Slack events to RequestEnvelope for routing.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from .base import BaseChannelAdapter

logger = logging.getLogger(__name__)


class SlackAdapter(BaseChannelAdapter):
    """Slack bot adapter using slack-bolt."""

    def __init__(self, connection_id: str, workspace_id: str, config: Dict[str, Any]):
        super().__init__(connection_id, workspace_id, config)
        self._app = None
        self._handler = None
        self._server_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the Slack bot."""
        try:
            from slack_bolt.async_app import AsyncApp

            bot_token = self.config.get("bot_token", "")
            signing_secret = self.config.get("signing_secret", "")

            if not bot_token:
                raise ValueError("bot_token is required")
            if not signing_secret:
                raise ValueError("signing_secret is required")

            self._app = AsyncApp(
                token=bot_token,
                signing_secret=signing_secret,
            )

            # Register message handler (responds to all messages)
            @self._app.message("")
            async def handle_message(message, say):
                await self._on_message(message, say)

            # Start socket mode or web server depending on config
            app_token = self.config.get("app_token")
            if app_token:
                from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
                self._handler = AsyncSocketModeHandler(self._app, app_token)
                self._server_task = asyncio.create_task(self._handler.start_async())
            else:
                logger.warning("[Slack:%s] No app_token — using Events API mode (webhook required)", self.connection_id)

            self.is_running = True
            logger.info("[Slack:%s] Bot started", self.connection_id)

        except ImportError:
            logger.error("[Slack] slack-bolt not installed. Run: pip install slack-bolt")
            raise
        except Exception as e:
            logger.error("[Slack:%s] Failed to start: %s", self.connection_id, e)
            raise

    async def stop(self):
        """Stop the Slack bot."""
        if self._handler:
            try:
                await self._handler.close_async()
            except Exception as e:
                logger.warning("[Slack:%s] Error during stop: %s", self.connection_id, e)
        if self._server_task:
            self._server_task.cancel()
        self.is_running = False
        logger.info("[Slack:%s] Bot stopped", self.connection_id)

    async def send_message(self, channel_id: str, text: str, **kwargs) -> bool:
        """Send a message to a Slack channel or thread."""
        if not self._app:
            return False
        try:
            thread_ts = kwargs.get("thread_ts")
            await self._app.client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts,
            )
            return True
        except Exception as e:
            logger.error("[Slack:%s] Failed to send message: %s", self.connection_id, e)
            return False

    async def test_connection(self) -> Dict[str, Any]:
        """Test Slack bot token validity."""
        try:
            import requests
            token = self.config.get("bot_token", "")
            resp = requests.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            data = resp.json()
            if data.get("ok"):
                return {"status": "connected", "team": data.get("team"), "bot_user": data.get("user")}
            return {"status": "error", "detail": data.get("error", "Unknown")}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    async def _on_message(self, message: dict, say):
        """Handle incoming Slack message."""
        # Ignore bot messages
        if message.get("bot_id") or message.get("subtype") == "bot_message":
            return

        text = message.get("text", "")
        if not text:
            return

        try:
            platform_msg = {
                "channel_id": message.get("channel"),
                "reply_channel_id": message.get("channel"),
                "thread_ts": message.get("thread_ts") or message.get("ts"),
                "user_id": message.get("user"),
                "user_name": message.get("user"),
                "text": text,
                "message_ts": message.get("ts"),
            }

            await self.handle_message(platform_msg)

        except Exception as e:
            logger.error("[Slack:%s] Error handling message: %s", self.connection_id, e)
            try:
                await say("Sorry, I encountered an error processing your message.")
            except Exception:
                pass

    def _to_envelope(self, platform_message: Dict[str, Any]):
        """Convert Slack message to RequestEnvelope."""
        from core.models.routing import RequestEnvelope, RequestUser, ChannelSource

        return RequestEnvelope(
            id=uuid4(),
            source=ChannelSource.SLACK,
            content=platform_message.get("text", ""),
            raw_payload=platform_message,
            user=RequestUser(
                id=platform_message.get("user_id"),
                name=platform_message.get("user_name"),
                auth_type="slack",
            ),
            workspace_id=self.workspace_id,
            metadata={
                "channel_adapter": "slack",
                "connection_id": self.connection_id,
                "channel_id": platform_message.get("channel_id"),
                "thread_ts": platform_message.get("thread_ts"),
            },
        )
