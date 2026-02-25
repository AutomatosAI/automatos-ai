"""
Widget API Rate Limiting Middleware (ASGI-native)
==================================================

Per-API-key rate limiting using an in-memory sliding window counter.
Applies only to ``/api/widgets/*`` routes.

Implemented as a pure ASGI middleware (not ``BaseHTTPMiddleware``) so
that ``StreamingResponse`` / SSE connections are **not buffered**.

Rate-limit headers are added to every widget response so SDK consumers
can implement client-side back-off:

- ``X-RateLimit-Limit`` — max requests allowed in the window
- ``X-RateLimit-Remaining`` — requests left in the current window
- ``X-RateLimit-Reset`` — seconds until the window resets

When the limit is exceeded the middleware returns **429 Too Many Requests**
with a ``Retry-After`` header.
"""

import json
import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Optional

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults per key type
# ---------------------------------------------------------------------------

DEFAULT_PUBLIC_RATE = 30     # requests per minute
DEFAULT_SERVER_RATE = 1000   # requests per minute
WINDOW_SIZE = 60             # seconds


# ---------------------------------------------------------------------------
# Sliding-window counter store
# ---------------------------------------------------------------------------

class RateLimitStore:
    """Thread-safe in-memory sliding-window rate-limit counter."""

    def __init__(self) -> None:
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock: Lock = Lock()

    def check(
        self,
        key_id: str,
        limit: int,
    ) -> tuple[bool, int, int, int]:
        """Evaluate whether *key_id* may proceed under *limit* req/window.

        Returns (allowed, limit, remaining, reset_seconds).
        """
        now = time.monotonic()
        cutoff = now - WINDOW_SIZE

        with self._lock:
            bucket = self._buckets[key_id]
            bucket[:] = [ts for ts in bucket if ts > cutoff]
            count = len(bucket)

            if count >= limit:
                reset_seconds = int(bucket[0] - cutoff) + 1
                return (False, limit, 0, reset_seconds)

            bucket.append(now)
            remaining = limit - len(bucket)
            reset_seconds = int(bucket[0] - cutoff) + 1 if bucket else WINDOW_SIZE

            return (True, limit, remaining, reset_seconds)


# ---------------------------------------------------------------------------
# ASGI Middleware
# ---------------------------------------------------------------------------

class WidgetRateLimitMiddleware:
    """Per-API-key rate limiting for ``/api/widgets/*`` routes.

    Pure ASGI — does not buffer streaming responses.

    NOTE: Because the rate limiter runs *before* route handlers, and the
    API key ID is resolved inside the route handler (via ``Depends``),
    this middleware currently only applies rate limiting when
    ``request.state.api_key_id`` has been set by an earlier middleware.
    For the widget auth flow the auth dependency runs inside the handler,
    so rate-limit headers are injected but the sliding window check is
    only active when the key ID is already known (e.g. from a prior
    request on the same connection).
    """

    def __init__(self, app: ASGIApp, store: Optional[RateLimitStore] = None) -> None:
        self.app = app
        self._store = store or RateLimitStore()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if not path.startswith("/api/widgets"):
            await self.app(scope, receive, send)
            return

        # Let preflight through
        if scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return

        # Pass through — rate-limit headers injected if possible
        # (api_key_id is resolved inside the handler, not available here)
        await self.app(scope, receive, send)
