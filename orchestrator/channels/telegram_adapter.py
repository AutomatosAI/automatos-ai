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
            from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

            token = self.config.get("bot_token", "")
            if not token:
                raise ValueError("bot_token is required")

            self._app = ApplicationBuilder().token(token).build()

            self._app.add_handler(CommandHandler(["start", "help"], self._on_command))
            self._app.add_handler(
                MessageHandler(
                    (filters.TEXT | filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND,
                    self._on_message,
                )
            )

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

    async def _on_command(self, update, context):
        """Handle /start and /help — captures chat_id and confirms wiring."""
        if not update.effective_chat:
            return
        chat_id = str(update.effective_chat.id)
        self._persist_default_chat_id(chat_id)
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=(
                    "✅ Connected. This chat is now your default for Automatos "
                    "notifications. Send any message to talk to your agents."
                ),
            )
        except Exception as e:
            logger.warning("[Telegram:%s] Failed to ack /start: %s", self.connection_id, e)

    def _persist_default_chat_id(self, chat_id: str) -> None:
        """Seed telegram_default_chat_id in workspace settings.integrations —
        SET ONCE, never retargeted from inbound traffic.

        The stored chat is the delivery target for agent-initiated questions and
        the chat the answer path binds a reply to; overwriting it from arbitrary
        inbound senders let any user who can message the bot repoint the
        operator's questions to their own chat (P225-RVW-10, mirroring the
        webhook path). Only seed it when unset."""
        try:
            from core.database.database import SessionLocal
            from core.models.workspaces import Workspace
            from sqlalchemy.orm.attributes import flag_modified

            db = SessionLocal()
            try:
                ws = db.query(Workspace).get(self.workspace_id)
                if not ws:
                    return
                settings = dict(ws.settings or {})
                integrations = dict(settings.get("integrations", {}))
                if integrations.get("telegram_default_chat_id"):
                    # Already anchored — never silently retarget from inbound.
                    return
                integrations["telegram_default_chat_id"] = chat_id
                settings["integrations"] = integrations
                ws.settings = settings
                flag_modified(ws, "settings")
                db.commit()
                logger.info(
                    "[Telegram:%s] Seeded telegram_default_chat_id for ws=%s (set-once)",
                    self.connection_id, self.workspace_id,
                )
            finally:
                db.close()
        except Exception as e:
            logger.warning("[Telegram:%s] Failed to persist chat_id: %s", self.connection_id, e)

    async def _on_message(self, update, context):
        """Handle incoming Telegram message (text, photos, documents)."""
        if not update.message:
            return

        if update.effective_chat:
            self._persist_default_chat_id(str(update.effective_chat.id))

        has_text = bool(update.message.text or update.message.caption)
        has_photo = bool(update.message.photo)
        has_document = bool(update.message.document)

        if not (has_text or has_photo or has_document):
            return

        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action="typing"
            )

            # PRD-127: Handle photos and documents by uploading to AttachmentStore
            attachment_ids: list[str] = []

            if has_photo:
                # Get highest resolution photo
                photo = update.message.photo[-1]
                try:
                    file = await context.bot.get_file(photo.file_id)
                    file_bytes = await file.download_as_bytearray()
                    filename = f"telegram_photo_{photo.file_unique_id}.jpg"
                    attachment_id = await self.upload_attachment(
                        content=bytes(file_bytes),
                        filename=filename,
                        mime_type="image/jpeg",
                    )
                    if attachment_id:
                        attachment_ids.append(attachment_id)
                except Exception as e:
                    logger.warning("[Telegram:%s] Failed to download photo: %s", self.connection_id, e)

            if has_document:
                doc = update.message.document
                try:
                    file = await context.bot.get_file(doc.file_id)
                    file_bytes = await file.download_as_bytearray()
                    attachment_id = await self.upload_attachment(
                        content=bytes(file_bytes),
                        filename=doc.file_name or f"telegram_doc_{doc.file_unique_id}",
                        mime_type=doc.mime_type,
                    )
                    if attachment_id:
                        attachment_ids.append(attachment_id)
                except Exception as e:
                    logger.warning("[Telegram:%s] Failed to download document: %s", self.connection_id, e)

            # Use text or caption
            text_content = update.message.text or update.message.caption or ""
            if not text_content and attachment_ids:
                text_content = "[Attachment received]"

            platform_msg = {
                "channel_id": str(update.effective_chat.id),
                "reply_channel_id": str(update.effective_chat.id),
                "user_id": str(update.effective_user.id) if update.effective_user else None,
                "user_name": update.effective_user.first_name if update.effective_user else None,
                "text": text_content,
                "message_id": str(update.message.message_id),
                "attachment_ids": attachment_ids,  # PRD-127
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
