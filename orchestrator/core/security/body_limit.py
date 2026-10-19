"""Upload caps counted on the wire, before the route parses the body (PRD-261 W-C1, M4).

``main.py``'s ``limit_request_body`` reads only the ``Content-Length`` header, so an upload sent
chunked (no length) passed it, and FastAPI read and spooled the whole multipart form before any
dependency, the auth gates included, ran. This pure ASGI middleware covers the capped upload
routes (``main.PATH_BODY_LIMITS``: brand logo and fonts, theme import and submit), for the
methods that carry a body:

* a ``Content-Length`` over the cap is 413, and one that isn't a whole number is 400;
* a body with neither a ``Content-Length`` nor ``Transfer-Encoding: chunked`` has no bound the
  server can hold it to: 411;
* otherwise the body is read here, counting bytes as each message arrives, and refused with 413
  the moment it passes the cap; a body within it is handed on to the app whole. So the route
  never sees more than the cap, and no more than the cap is ever held in memory.

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


async def read_capped(receive: Receive, limit: int) -> Tuple[Optional[bytes], bool]:
    """``(body, over)``: the whole body when it fits; ``over`` True as soon as it passes ``limit``;
    ``(None, False)`` when the client went away first."""
    parts = []
    total = 0
    while True:
        message = await receive()
        if message.get("type") == "http.disconnect":
            return None, False
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > limit:
            return None, True
        parts.append(chunk)
        if not message.get("more_body", False):
            return b"".join(parts), False


def replay(body: bytes, receive: Receive) -> Receive:
    """A ``receive`` that gives the app the whole body once, then whatever the server sends next."""
    pending = [{"type": "http.request", "body": body, "more_body": False}]

    async def _receive() -> Message:
        if pending:
            return pending.pop()
        return await receive()

    return _receive


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
        body, over = await read_capped(receive, limit)
        if over:
            logger.info("[BodyLimit] %s %s refused: the body passed %d bytes", scope.get("method"), scope.get("path"), limit)
            await answer(send, PAYLOAD_TOO_LARGE, TOO_LARGE)
            return
        if body is None:
            return  # the client went away before the body ended: nobody to answer
        await self.app(scope, replay(body, receive), send)
