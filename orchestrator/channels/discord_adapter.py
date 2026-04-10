"""
Discord Channel Adapter (PRD-55 US-022)
========================================

Uses discord.py v2.0+ for async Discord bot integration.
Converts Discord messages to RequestEnvelope for routing.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from .base import BaseChannelAdapter

logger = logging.getLogger(__name__)


class DiscordAdapter(BaseChannelAdapter):
    """Discord bot adapter using discord.py."""

    def __init__(self, connection_id: str, workspace_id: str, config: Dict[str, Any]):
        super().__init__(connection_id, workspace_id, config)
        self._client = None
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the Discord bot."""
        try:
            import discord

            token = self.config.get("bot_token", "")
            if not token:
                raise ValueError("bot_token is required")

            intents = discord.Intents.default()
            intents.message_content = True

            self._client = discord.Client(intents=intents)

            @self._client.event
            async def on_ready():
                logger.info("[Discord:%s] Bot connected as %s", self.connection_id, self._client.user)

            @self._client.event
            async def on_message(message):
                # Ignore own messages
                if message.author == self._client.user:
                    return
                # Ignore bot messages
                if message.author.bot:
                    return
                await self._on_message(message)

            # Start in background task
            self._task = asyncio.create_task(self._client.start(token))
            self.is_running = True
            logger.info("[Discord:%s] Bot starting", self.connection_id)

        except ImportError:
            logger.error("[Discord] discord.py not installed. Run: pip install discord.py")
            raise
        except Exception as e:
            logger.error("[Discord:%s] Failed to start: %s", self.connection_id, e)
            raise

    async def stop(self):
        """Stop the Discord bot."""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning("[Discord:%s] Error during stop: %s", self.connection_id, e)
        if self._task:
            self._task.cancel()
        self.is_running = False
        logger.info("[Discord:%s] Bot stopped", self.connection_id)

    async def send_message(self, channel_id: str, text: str, **kwargs) -> bool:
        """Send a message to a Discord channel."""
        if not self._client:
            return False
        try:
            channel = self._client.get_channel(int(channel_id))
            if not channel:
                channel = await self._client.fetch_channel(int(channel_id))

            # Discord limit is 2000 chars
            chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
            for chunk in chunks:
                await channel.send(chunk)
            return True
        except Exception as e:
            logger.error("[Discord:%s] Failed to send message: %s", self.connection_id, e)
            return False

    async def test_connection(self) -> Dict[str, Any]:
        """Test Discord bot token validity."""
        try:
            import requests
            token = self.config.get("bot_token", "")
            resp = requests.get(
                "https://discord.com/api/v10/users/@me",
                headers={"Authorization": f"Bot {token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                user = resp.json()
                return {"status": "connected", "bot_name": user.get("username")}
            return {"status": "error", "detail": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    async def _on_message(self, message):
        """Handle incoming Discord message (text and attachments)."""
        try:
            # Show typing indicator
            async with message.channel.typing():
                # PRD-127: Handle attachments by downloading and uploading to AttachmentStore
                attachment_ids: list[str] = []

                for attachment in message.attachments:
                    try:
                        file_bytes = await attachment.read()
                        attachment_id = await self.upload_attachment(
                            content=file_bytes,
                            filename=attachment.filename,
                            mime_type=attachment.content_type,
                        )
                        if attachment_id:
                            attachment_ids.append(attachment_id)
                    except Exception as e:
                        logger.warning("[Discord:%s] Failed to download attachment %s: %s", self.connection_id, attachment.filename, e)

                text = message.content
                if not text and attachment_ids:
                    text = "[Attachment received]"

                platform_msg = {
                    "channel_id": str(message.channel.id),
                    "reply_channel_id": str(message.channel.id),
                    "user_id": str(message.author.id),
                    "user_name": message.author.display_name,
                    "text": text,
                    "message_id": str(message.id),
                    "guild_id": str(message.guild.id) if message.guild else None,
                    "attachment_ids": attachment_ids,  # PRD-127
                }

                await self.handle_message(platform_msg)

        except Exception as e:
            logger.error("[Discord:%s] Error handling message: %s", self.connection_id, e)
            try:
                await message.channel.send("Sorry, I encountered an error processing your message.")
            except Exception:
                pass

    def _to_envelope(self, platform_message: Dict[str, Any]):
        """Convert Discord message to RequestEnvelope."""
        from core.models.routing import RequestEnvelope, RequestUser, ChannelSource

        return RequestEnvelope(
            id=uuid4(),
            source=ChannelSource.DISCORD,
            content=platform_message.get("text", ""),
            raw_payload=platform_message,
            user=RequestUser(
                id=platform_message.get("user_id"),
                name=platform_message.get("user_name"),
                auth_type="discord",
            ),
            workspace_id=self.workspace_id,
            metadata={
                "channel_adapter": "discord",
                "connection_id": self.connection_id,
                "channel_id": platform_message.get("channel_id"),
                "guild_id": platform_message.get("guild_id"),
            },
        )
