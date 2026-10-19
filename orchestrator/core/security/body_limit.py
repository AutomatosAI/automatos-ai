"""Upload caps counted on the wire, as the route reads the body (PRD-261 W-C1, M4).

``main.py``'s ``limit_request_body`` reads only the ``Content-Length`` header, so an upload sent
chunked (no length) passed it, and FastAPI read and spooled the whole multipart form before any
dependency, the auth gates included, ran. This pure ASGI middleware covers the capped upload
routes (``main.PATH_BODY_LIMITS``: brand logo and fonts, theme import and submit), for the
methods that carry a body:

* a ``Content-Length`` over the cap is 413, and one that isn't a whole number is 400;
* a body with neither a ``Content-Length`` nor ``Transfer-Encoding: chunked`` has no bound the
  server can hold it to: 411;
* otherwise the body passes straight to the app, message by message as the app reads it, while
  this middleware counts it. Nothing is buffered here: Starlette's multipart parser spools a
  file part to disk past 1 MB as before, so an upload held open just under the cap pins no more
  memory than it did. The message that passes the cap is never handed on: the app gets
  ``http.disconnect`` in its place (and on every read after), and whatever response the app then
  starts (FastAPI's 400 for a body it couldn't parse) is replaced with 413. So the route never sees
  more than the cap. A response the app started before the cap was passed goes out unchanged.

Every other path, and every method without a body, passes through untouched.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

Scope = Dict[str, Any]
Message = Dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
PAYLOAD_TOO_LARGE, LENGTH_REQUIRED, BAD_REQUEST = 413, 411, 400
TOO_LARGE = "Payload too large"
NO_LENGTH = "Send the upload with a Content-Length header, or chunked."
BAD_LENGTH = "Invalid Content-Length header"


def _headers(scope: Scope) -> Dict[str, str]:
    return {name.decode("latin-1").lower(): value.decode("latin-1") for name, value in scope.get("headers", [])}


def header_refusal(headers: Mapping[str, str], limit: int) -> Optional[Tuple[int, str]]:
    """``(status, message)`` when the headers alone refuse the body, else None (read and count it)."""
    length = headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
        except ValueError:
            return BAD_REQUEST, BAD_LENGTH
        if size < 0:
            return BAD_REQUEST, BAD_LENGTH
        return (PAYLOAD_TOO_LARGE, TOO_LARGE) if size > limit else None
    if "chunked" in headers.get("transfer-encoding", "").lower():
        return None
    return LENGTH_REQUIRED, NO_LENGTH


class CappedExchange:
    """One capped request: counts the body as the app reads it, and answers 413 in the app's place
    once it passes the cap (see the module docstring)."""

    def __init__(self, receive: Receive, send: Send, limit: int) -> None:
        self._receive, self._send, self.limit = receive, send, limit
        self.total = 0
        self.over = False
        self.started = False  # the app's response went out before the cap was passed
        self.answered = False

    async def receive(self) -> Message:
        """The next body message, counted; ``http.disconnect`` from the one that passes the cap on."""
        if self.over:
            return {"type": "http.disconnect"}
        message = await self._receive()
        if message.get("type") == "http.request":
            self.total += len(message.get("body", b""))
            if self.total > self.limit:
                self.over = True
                return {"type": "http.disconnect"}
        return message

    async def send(self, message: Message) -> None:
        """The app's response; once the cap is passed (before it started), 413 in its place."""
        if self.over and not self.started:
            await self.refuse()
            return
        if message.get("type") == "http.response.start":
            self.started = True
        await self._send(message)

    async def refuse(self) -> None:
        """413, once."""
        if not self.answered:
            self.answered = True
            await answer(self._send, PAYLOAD_TOO_LARGE, TOO_LARGE)


async def answer(send: Send, status: int, detail: str) -> None:
    """A JSON ``{"detail": ...}`` refusal that closes the connection (the rest of the body is never read)."""
    body = json.dumps({"detail": detail}).encode("utf-8")
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii")),
               (b"connection", b"close")]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class CappedBodyMiddleware:
    """Counts the body of a capped path's upload as it arrives; see the module docstring."""

    def __init__(self, app: ASGIApp, limits: Mapping[str, int]) -> None:
        self.app = app
        self.limits = dict(limits)

    def limit_for(self, scope: Scope) -> Optional[int]:
        """The cap for this request, or None when it isn't a capped upload."""
        if scope.get("type") != "http" or scope.get("method") not in BODY_METHODS:
            return None
        path = scope.get("path", "")
        return next((limit for prefix, limit in self.limits.items() if path.startswith(prefix)), None)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self.limit_for(scope)
        if limit is None:
            await self.app(scope, receive, send)
            return
        refusal = header_refusal(_headers(scope), limit)
        if refusal is not None:
            logger.info("[BodyLimit] %s %s refused before reading: %d", scope.get("method"), scope.get("path"), refusal[0])
            await answer(send, *refusal)
            return
        await self._counted(scope, CappedExchange(receive, send, limit))

    async def _counted(self, scope: Scope, exchange: CappedExchange) -> None:
        """The app on the counted body; 413 when the body passed the cap before the app answered."""
        try:
            await self.app(scope, exchange.receive, exchange.send)
        except Exception:
            if not exchange.over:
                raise
            logger.info("[BodyLimit] %s %s: the app stopped on the cut-off body",
                        scope.get("method"), scope.get("path"))
        if exchange.over and not exchange.started:
            logger.info("[BodyLimit] %s %s refused: the body passed %d bytes",
                        scope.get("method"), scope.get("path"), exchange.limit)
            await exchange.refuse()
