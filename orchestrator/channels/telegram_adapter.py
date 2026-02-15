"""
Telegram Channel Adapter (PRD-55 US-020)
=========================================

Uses python-telegram-bot v22+ for async Telegram bot integration.
Converts Telegram messages to RequestEnvelope for routing.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from .base import BaseChannelAdapter

logger = logging.getLogger(__name__)


class TelegramAdapter(BaseChannelAdapter):
    """Telegram bot adapter using python-telegram-bot."""

    def __init__(self, connection_id: str, workspace_id: str, config: Dict[str, Any]):
        super().__init__(connection_id, workspace_id, config)
        self._app = None
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the Telegram bot."""
        try:
            from telegram.ext import ApplicationBuilder, MessageHandler, filters

            token = self.config.get("bot_token", "")
            if not token:
                raise ValueError("bot_token is required")

            self._app = ApplicationBuilder().token(token).build()

            # Register message handler
            self._app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
            )

            # Start polling in background
            await self._app.initialize()
            await self._app.start()
            self._task = asyncio.create_task(self._app.updater.start_polling())
            self.is_running = True
            logger.info("[Telegram:%s] Bot started", self.connection_id)

        except ImportError:
            logger.error("[Telegram] python-telegram-bot not installed. Run: pip install python-telegram-bot")
            raise
        except Exception as e:
            logger.error("[Telegram:%s] Failed to start: %s", self.connection_id, e)
            raise

    async def stop(self):
        """Stop the Telegram bot."""
        if self._app:
            try:
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.warning("[Telegram:%s] Error during stop: %s", self.connection_id, e)
        if self._task:
            self._task.cancel()
        self.is_running = False
        logger.info("[Telegram:%s] Bot stopped", self.connection_id)

    async def send_message(self, channel_id: str, text: str, **kwargs) -> bool:
        """Send a message to a Telegram chat."""
        if not self._app or not self._app.bot:
            return False
        try:
            # Auto-chunk messages over 4096 chars
            chunks = [text[i:i + 4096] for i in range(0, len(text), 4096)]
            for chunk in chunks:
                await self._app.bot.send_message(chat_id=int(channel_id), text=chunk)
            return True
        except Exception as e:
            logger.error("[Telegram:%s] Failed to send message: %s", self.connection_id, e)
            return False

    async def test_connection(self) -> Dict[str, Any]:
        """Test Telegram bot token validity."""
        try:
            import requests
            token = self.config.get("bot_token", "")
            resp = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
            if resp.status_code == 200:
                bot = resp.json().get("result", {})
                return {"status": "connected", "bot_name": bot.get("username")}
            return {"status": "error", "detail": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    async def _on_message(self, update, context):
        """Handle incoming Telegram message."""
        if not update.message or not update.message.text:
            return

        try:
            # Send typing indicator
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action="typing"
            )

            platform_msg = {
                "channel_id": str(update.effective_chat.id),
                "reply_channel_id": str(update.effective_chat.id),
                "user_id": str(update.effective_user.id) if update.effective_user else None,
                "user_name": update.effective_user.first_name if update.effective_user else None,
                "text": update.message.text,
                "message_id": str(update.message.message_id),
            }

            await self.handle_message(platform_msg)

        except Exception as e:
            logger.error("[Telegram:%s] Error handling message: %s", self.connection_id, e)
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Sorry, I encountered an error processing your message.",
                )
            except Exception:
                pass

    def _to_envelope(self, platform_message: Dict[str, Any]):
        """Convert Telegram message to RequestEnvelope."""
        from core.models.routing import RequestEnvelope, RequestUser, ChannelSource

        return RequestEnvelope(
            id=uuid4(),
            source=ChannelSource.TELEGRAM,
            content=platform_message.get("text", ""),
            raw_payload=platform_message,
            user=RequestUser(
                id=platform_message.get("user_id"),
                name=platform_message.get("user_name"),
                auth_type="telegram",
            ),
            workspace_id=self.workspace_id,
            metadata={
                "channel_adapter": "telegram",
                "connection_id": self.connection_id,
                "chat_id": platform_message.get("channel_id"),
            },
        )
