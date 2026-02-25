"""
Widget API Rate Limiting Middleware
====================================

Per-API-key rate limiting using an in-memory sliding window counter.
Applies only to ``/api/widgets/*`` routes.  The middleware reads
``request.state.api_key_id`` (set by :func:`widget_auth`) to identify
callers and ``request.state.rate_limit`` for optional per-key overrides.

Rate-limit headers are added to every widget response so SDK consumers
can implement client-side back-off:

- ``X-RateLimit-Limit`` — max requests allowed in the window
- ``X-RateLimit-Remaining`` — requests left in the current window
- ``X-RateLimit-Reset`` — seconds until the window resets

When the limit is exceeded the middleware returns **429 Too Many Requests**
with a ``Retry-After`` header.
"""

import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

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
    """Thread-safe in-memory sliding-window rate-limit counter.

    Each API key gets a bucket of request timestamps.  On every check the
    store prunes entries older than :data:`WINDOW_SIZE` seconds and decides
    whether the caller is within their quota.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock: Lock = Lock()

    def check(
        self,
        key_id: str,
        limit: int,
    ) -> tuple[bool, int, int, int]:
        """Evaluate whether *key_id* may proceed under *limit* req/window.

        Returns:
            A 4-tuple of ``(allowed, limit, remaining, reset_seconds)``.

            * *allowed* — ``True`` if the request should be served.
            * *limit* — the cap that was evaluated.
            * *remaining* — how many requests the caller has left.
            * *reset_seconds* — seconds until the oldest entry expires
              (useful for ``Retry-After``).
        """
        now = time.monotonic()
        cutoff = now - WINDOW_SIZE

        with self._lock:
            bucket = self._buckets[key_id]

            # Prune expired timestamps
            bucket[:] = [ts for ts in bucket if ts > cutoff]

            count = len(bucket)

            if count >= limit:
                # Oldest surviving timestamp determines when a slot opens
                reset_seconds = int(bucket[0] - cutoff) + 1
                return (False, limit, 0, reset_seconds)

            # Record this request
            bucket.append(now)
            remaining = limit - len(bucket)
            reset_seconds = int(bucket[0] - cutoff) + 1 if bucket else WINDOW_SIZE

            return (True, limit, remaining, reset_seconds)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class WidgetRateLimitMiddleware(BaseHTTPMiddleware):
    """Per-API-key rate limiting for ``/api/widgets/*`` routes.

    The middleware is designed to sit *after* the auth layer so that
    ``request.state.api_key_id`` is already populated.  If the auth
    layer also writes ``request.state.rate_limit`` (an ``int``), that
    value is used as the per-minute cap; otherwise
    :data:`DEFAULT_PUBLIC_RATE` applies.
    """

    def __init__(self, app, store: Optional[RateLimitStore] = None) -> None:
        super().__init__(app)
        self._store = store or RateLimitStore()

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only apply to widget routes
        if not request.url.path.startswith("/api/widgets"):
            return await call_next(request)

        # Let preflight through without counting
        if request.method == "OPTIONS":
            return await call_next(request)

        # Resolve caller identity — set by widget_auth dependency / middleware
        api_key_id: Optional[str] = getattr(request.state, "api_key_id", None)
        if api_key_id is None:
            # Auth hasn't run yet or this is an unauthenticated route;
            # let it pass through — the auth layer will reject if needed.
            return await call_next(request)

        key_id = str(api_key_id)

        # Determine the rate limit for this key
        rate_limit: int = getattr(
            request.state,
            "rate_limit",
            DEFAULT_PUBLIC_RATE,
        )

        allowed, limit, remaining, reset_seconds = self._store.check(
            key_id,
            rate_limit,
        )

        if not allowed:
            logger.warning(
                "Rate limit exceeded for api_key_id=%s (%d/%d per %ds)",
                key_id,
                limit,
                limit,
                WINDOW_SIZE,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please retry later.",
                    "retry_after": reset_seconds,
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_seconds),
                    "Retry-After": str(reset_seconds),
                },
            )

        # Proceed and attach rate-limit headers to the response
        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_seconds)
        return response
