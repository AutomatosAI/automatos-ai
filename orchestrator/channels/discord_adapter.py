"""
Discord Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges Discord guild/DM messages into the universal routing pipeline
using discord.py v2.0+.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)

# Max Discord message length
_DISCORD_MAX_LEN = 2000


class DiscordAdapter(BaseChannelAdapter):
    """Adapter for Discord bots via discord.py v2.0+."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._client = None  # discord.Client
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        import discord

        token = self.config.get("bot_token")
        if not token:
            raise ValueError("Discord adapter requires 'bot_token' in config")

        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True

        self._client = discord.Client(intents=intents)

        adapter_ref = self  # closure reference

        @self._client.event
        async def on_ready():
            logger.info(
                "Discord adapter %s connected as %s",
                adapter_ref.connection_id,
                self._client.user,
            )

        @self._client.event
        async def on_message(message: discord.Message):
            await adapter_ref._on_message(message)

        # Run the client in a background task
        self._task = asyncio.create_task(self._client.start(token))
        self._running = True
        logger.info("Discord adapter %s starting", self.connection_id)

    async def stop(self) -> None:
        if self._client:
            try:
                await self._client.close()
            except Exception as exc:
                logger.warning("Error closing Discord client: %s", exc)
        if self._task and not self._task.done():
            self._task.cancel()
        self._running = False
        logger.info("Discord adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        import httpx

        token = self.config.get("bot_token", "")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://discord.com/api/v10/users/@me",
                headers={"Authorization": f"Bot {token}"},
            )
            if resp.status_code != 200:
                raise ConnectionError(
                    f"Discord users/@me failed ({resp.status_code}): {resp.text}"
                )
            data = resp.json()
        return {
            "platform": "discord",
            "bot_id": data.get("id"),
            "bot_username": data.get("username"),
            "bot_discriminator": data.get("discriminator"),
        }

    # ------------------------------------------------------------------
    # Incoming messages
    # ------------------------------------------------------------------

    async def _on_message(self, message: Any) -> None:
        """discord.py on_message handler."""
        # Ignore own messages
        if self._client and message.author == self._client.user:
            return
        # Ignore other bots
        if message.author.bot:
            return
        if not message.content:
            return

        # Show typing indicator
        response_text = None
        try:
            async with message.channel.typing():
                response_text = await self.handle_message(message)
        except Exception:
            if response_text is None:
                response_text = await self.handle_message(message)

        if response_text:
            await self.send_message(
                str(message.channel.id), response_text, _channel=message.channel
            )

        self._increment_message_count()

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        """Send a message to a Discord channel, auto-chunking if needed."""
        channel = kwargs.get("_channel")

        if channel is None and self._client:
            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                try:
                    channel = await self._client.fetch_channel(int(channel_id))
                except Exception as exc:
                    logger.error("Could not fetch Discord channel %s: %s", channel_id, exc)
                    return

        if channel is None:
            logger.error("No Discord channel available for sending")
            return

        chunks = self._chunk_text(text, _DISCORD_MAX_LEN)
        for chunk in chunks:
            await channel.send(chunk)

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        msg = platform_message  # discord.Message

        return RequestEnvelope(
            source=ChannelSource.DISCORD,
            content=msg.content or "",
            raw_payload={
                "message_id": str(msg.id),
                "channel_id": str(msg.channel.id),
                "guild_id": str(msg.guild.id) if msg.guild else None,
            },
            user=RequestUser(
                id=str(msg.author.id),
                name=str(msg.author),
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "discord",
                "connection_id": self.connection_id,
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
