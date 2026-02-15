"""
IRC Channel Adapter (PRD-55: Autonomous Assistant Platform).

Bridges IRC messages into the universal routing pipeline using
raw asyncio socket connections (no external dependencies).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from channels.base import BaseChannelAdapter

logger = logging.getLogger(__name__)

_IRC_MAX_LEN = 450  # IRC PRIVMSG max ~512 including protocol overhead


class IRCAdapter(BaseChannelAdapter):
    """Adapter for IRC using raw asyncio sockets."""

    def __init__(
        self,
        connection_id: str,
        workspace_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(connection_id, workspace_id, config)
        self._server = config.get("server", "irc.libera.chat")
        self._port = int(config.get("port", 6697))
        self._use_ssl = config.get("use_ssl", True)
        self._nickname = config.get("nickname", "automatos-bot")
        self._channels_list: List[str] = config.get("channels", ["#automatos"])
        self._password = config.get("password", "")
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._server:
            raise ValueError("IRC adapter requires 'server'")

        if self._use_ssl:
            import ssl as _ssl
            ssl_ctx = _ssl.create_default_context()
            self._reader, self._writer = await asyncio.open_connection(
                self._server, self._port, ssl=ssl_ctx
            )
        else:
            self._reader, self._writer = await asyncio.open_connection(
                self._server, self._port
            )

        if self._password:
            self._send_raw(f"PASS {self._password}")

        self._send_raw(f"NICK {self._nickname}")
        self._send_raw(f"USER {self._nickname} 0 * :Automatos Bot")

        self._running = True
        self._task = asyncio.create_task(self._read_loop())
        logger.info("IRC adapter %s started (%s:%d)", self.connection_id, self._server, self._port)

    async def stop(self) -> None:
        self._running = False
        if self._writer:
            try:
                self._send_raw("QUIT :Automatos shutting down")
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("IRC adapter %s stopped", self.connection_id)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> Dict[str, Any]:
        import ssl as _ssl

        try:
            if self._use_ssl:
                ssl_ctx = _ssl.create_default_context()
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self._server, self._port, ssl=ssl_ctx),
                    timeout=10,
                )
            else:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self._server, self._port),
                    timeout=10,
                )
            writer.close()
            await writer.wait_closed()
            return {
                "platform": "irc",
                "server": self._server,
                "port": self._port,
                "ssl": self._use_ssl,
                "connected": True,
            }
        except Exception as exc:
            raise ConnectionError(f"IRC connection failed: {exc}")

    # ------------------------------------------------------------------
    # Read loop
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        """Read IRC messages and handle them."""
        joined = False
        while self._running and self._reader:
            try:
                line_bytes = await asyncio.wait_for(
                    self._reader.readline(), timeout=300
                )
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # Handle PING
                if line.startswith("PING"):
                    pong_target = line.split(" ", 1)[1] if " " in line else ""
                    self._send_raw(f"PONG {pong_target}")
                    continue

                # Join channels after receiving end-of-MOTD (376) or no MOTD (422)
                if not joined and any(f" {code} " in line for code in ("376", "422")):
                    for ch in self._channels_list:
                        self._send_raw(f"JOIN {ch}")
                    joined = True
                    continue

                # Handle PRIVMSG
                if " PRIVMSG " in line:
                    await self._handle_privmsg(line)

            except asyncio.TimeoutError:
                self._send_raw("PING :keepalive")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("IRC read error: %s", exc)
                await asyncio.sleep(1)

    async def _handle_privmsg(self, line: str) -> None:
        """Parse and handle an IRC PRIVMSG."""
        # Format: :nick!user@host PRIVMSG #channel :message text
        try:
            prefix, rest = line[1:].split(" PRIVMSG ", 1)
            nick = prefix.split("!")[0]
            target, text = rest.split(" :", 1)

            if nick == self._nickname:
                return

            platform_msg = {
                "sender": nick,
                "text": text,
                "target": target.strip(),
                "prefix": prefix,
            }

            response = await self.handle_message(platform_msg)
            if response:
                reply_to = target.strip() if target.strip().startswith("#") else nick
                await self.send_message(reply_to, response)

            self._increment_message_count()
        except Exception as exc:
            logger.debug("IRC PRIVMSG parse error: %s", exc)

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> None:
        chunks = self._chunk_text(text, _IRC_MAX_LEN)
        for chunk in chunks:
            self._send_raw(f"PRIVMSG {channel_id} :{chunk}")
            await asyncio.sleep(0.5)  # Rate limit

    def _send_raw(self, message: str) -> None:
        if self._writer:
            self._writer.write(f"{message}\r\n".encode("utf-8"))

    # ------------------------------------------------------------------
    # Envelope conversion
    # ------------------------------------------------------------------

    def _to_envelope(self, platform_message: Any) -> Any:
        from core.models.routing import RequestEnvelope, ChannelSource, RequestUser

        msg = platform_message

        return RequestEnvelope(
            source=ChannelSource.IRC,
            content=msg.get("text", ""),
            raw_payload=msg,
            user=RequestUser(
                id=msg.get("sender", ""),
                name=msg.get("sender", ""),
            ),
            workspace_id=UUID(self.workspace_id),
            metadata={
                "channel_adapter": "irc",
                "connection_id": self.connection_id,
                "channel": msg.get("target", ""),
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
