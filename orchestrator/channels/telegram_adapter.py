"""
Telegram Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges Telegram Bot API messages into the universal routing pipeline
using the python-telegram-bot v22+ async Application.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)

# Max Telegram message length
_TG_MAX_LEN = 4096


class TelegramAdapter(BaseChannelAdapter):
    """Adapter for Telegram bots via python-telegram-bot v22+."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._app = None  # telegram.ext.Application
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        from telegram.ext import Application, MessageHandler, filters

        token = self.config.get("bot_token")
        if not token:
            raise ValueError("Telegram adapter requires 'bot_token' in config")

        self._app = (
            Application.builder()
            .token(token)
            .build()
        )

        # Register handler for text messages
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_update)
        )

        await self._app.initialize()
        await self._app.start()
        # Start polling in a background task
        self._task = asyncio.create_task(self._app.updater.start_polling())
        self._running = True
        logger.info("Telegram adapter %s started", self.connection_id)

    async def stop(self) -> None:
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as exc:
                logger.warning("Error stopping Telegram app: %s", exc)
        if self._task and not self._task.done():
            self._task.cancel()
        self._running = False
        logger.info("Telegram adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        import httpx

        token = self.config.get("bot_token", "")
        url = f"https://api.telegram.org/bot{token}/getMe"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            data = resp.json()
        if not data.get("ok"):
            raise ConnectionError(
                f"Telegram getMe failed: {data.get('description', 'unknown error')}"
            )
        bot = data["result"]
        return {
            "platform": "telegram",
            "bot_id": bot.get("id"),
            "bot_username": bot.get("username"),
            "bot_name": bot.get("first_name"),
        }

    # ------------------------------------------------------------------
    # Incoming messages
    # ------------------------------------------------------------------

    async def _handle_update(self, update: Any, context: Any) -> None:
        """python-telegram-bot handler callback."""
        message = update.effective_message
        if not message or not message.text:
            return

        chat_id = str(message.chat_id)

        # Show typing indicator
        try:
            await context.bot.send_chat_action(
                chat_id=message.chat_id, action="typing"
            )
        except Exception:
            pass

        # Route through the universal pipeline
        response_text = await self.handle_message(message)

        # Send response back
        if response_text:
            await self.send_message(chat_id, response_text, _context=context)

        # Update message count
        self._increment_message_count()

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        """Send a message to a Telegram chat, auto-chunking if needed."""
        ctx = kwargs.get("_context")
        bot = ctx.bot if ctx else (self._app.bot if self._app else None)
        if not bot:
            logger.error("No bot instance available for sending")
            return

        chunks = self._chunk_text(text, _TG_MAX_LEN)
        for chunk in chunks:
            await bot.send_message(chat_id=int(channel_id), text=chunk)

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        msg = platform_message
        user = msg.from_user

        return RequestEnvelope(
            source=ChannelSource.TELEGRAM,
            content=msg.text or "",
            raw_payload={
                "message_id": msg.message_id,
                "chat_id": msg.chat_id,
                "date": msg.date.isoformat() if msg.date else None,
            },
            user=RequestUser(
                id=str(user.id) if user else None,
                name=(
                    f"{user.first_name or ''} {user.last_name or ''}".strip()
                    if user
                    else None
                ),
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "telegram",
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
            from core.models.channels import ChannelConnection
            from sqlalchemy import text as sa_text
            from datetime import datetime, timezone

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
