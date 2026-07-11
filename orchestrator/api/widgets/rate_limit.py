"""
Widget API Rate Limiting Middleware (ASGI-native, Redis-backed)
================================================================

Per-key / per-IP rate limiting for ``/api/widgets/*`` routes, backed by a
**Redis sliding window shared across all uvicorn workers** (PRD-194 S5,
P2-13). This REPLACES the previous per-process in-memory window, which
reset on every deploy, was invisible to sibling workers, and never gated
anything real.

Shape:

- The identifier is resolved **pre-handler** in the middleware — a request
  with a Bearer key is keyed on a SHA-256 digest of the presented key (a
  1:1 per-key bucket without writing key material into Redis); a request
  with no key falls back to ``client_ip``. The FIRST request is gated.
- The window lives in a Redis sorted set — the same sliding-window idiom as
  ``core/security/rate_limiter.py`` — via the platform Redis client
  (``core/redis/client.py`` + ``config.REDIS_URL``). No new dependency.
- The two money-spending endpoints (``/api/widgets/chat``,
  ``/api/widgets/callback`` — LLM turns, channel fan-out) additionally get
  a **per-IP ceiling that applies even when a key is presented**: a scraped
  public key replayed from one box hits the IP ceiling regardless of the
  key's own budget.
- **Redis down ⇒ fail OPEN, loudly** (locked decision): the widget must not
  brick on a cache outage — every ungated request increments a counter and
  logs an ERROR, so an outage is visible, not silent.

Implemented as a pure ASGI middleware (not ``BaseHTTPMiddleware``) so that
``StreamingResponse`` / SSE connections are **not buffered**.

Rate-limit headers are added to every widget response so SDK consumers can
implement client-side back-off:

- ``X-RateLimit-Limit`` — max requests allowed in the window
- ``X-RateLimit-Remaining`` — requests left in the current window
- ``X-RateLimit-Reset`` — seconds until the window resets

When the limit is exceeded the middleware returns **429 Too Many Requests**
with a ``Retry-After`` header.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Optional
from uuid import uuid4

from starlette.types import ASGIApp, Receive, Scope, Send

from config import config

logger = logging.getLogger(__name__)

_KEY_PREFIX = "widget:rl"

# Money-spending endpoints that carry a per-IP ceiling ON TOP of the per-key
# limit (exact path → config attribute holding the ceiling).
_IP_CEILING_PATHS: dict[str, str] = {
    "/api/widgets/chat": "WIDGET_CHAT_IP_LIMIT_PER_WINDOW",
    "/api/widgets/callback": "WIDGET_CALLBACK_IP_LIMIT_PER_WINDOW",
}

# Loud fail-open accounting: every request allowed because Redis was
# unreachable bumps this counter (and logs an ERROR). A cache outage must be
# visible in logs/metrics, never silent.
_redis_failures: int = 0


def get_redis_failure_count() -> int:
    """Number of requests allowed ungated because Redis was unreachable."""
    return _redis_failures


def _count_redis_failure(identifier: str) -> None:
    global _redis_failures
    _redis_failures += 1
    logger.error(
        "widget rate limiter FAIL-OPEN (#%d): Redis unreachable — request "
        "for %s allowed ungated (P2-13: availability beats limiting on a "
        "cache outage, but this must not stay silent)",
        _redis_failures,
        identifier,
    )


def _default_redis():
    """Platform Redis connection or ``None`` — the same lazy, fail-soft seam
    as ``core/security/rate_limiter._get_redis``."""
    try:
        from core.redis.client import get_redis_client

        client = get_redis_client()
        if client is None:
            return None
        return client.get_redis()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Redis-backed sliding-window store (shared across workers)
# ---------------------------------------------------------------------------

class RateLimitStore:
    """Sliding-window rate-limit counter in a Redis sorted set.

    One key per identifier; members are request timestamps. The window is
    shared across every process/worker pointing at the same Redis — the
    property the deleted in-memory dict could never have. Same idiom as
    ``core/security/rate_limiter.py`` (zremrangebyscore → zcard → zadd →
    expire in one pipeline; a denied request still marks the window, so
    hammering keeps the window hot rather than resetting it).
    """

    def __init__(self, redis_factory=None) -> None:
        self._redis_factory = redis_factory or _default_redis

    def check(
        self,
        key_id: str,
        limit: int,
        window: Optional[int] = None,
    ) -> tuple[bool, int, int, int]:
        """Evaluate whether *key_id* may proceed under *limit* req/window.

        Returns ``(allowed, limit, remaining, reset_seconds)``. Redis
        unreachable ⇒ **fail OPEN** (allowed, loud counter + ERROR log).
        """
        window = window or config.WIDGET_RATE_LIMIT_WINDOW_SECONDS

        try:
            redis = self._redis_factory()
        except Exception:
            redis = None
        if redis is None:
            _count_redis_failure(key_id)
            return (True, limit, max(0, limit - 1), window)

        key = f"{_KEY_PREFIX}:{key_id}"
        now = time.time()

        try:
            pipe = redis.pipeline()
            pipe.zremrangebyscore(key, 0, now - window)   # drop expired marks
            pipe.zcard(key)                               # count BEFORE this request
            pipe.zrange(key, 0, 0, withscores=True)       # oldest surviving mark
            pipe.zadd(key, {f"{now:.6f}:{uuid4().hex[:6]}": now})
            pipe.expire(key, window + 10)
            results = pipe.execute()
        except Exception:
            _count_redis_failure(key_id)
            return (True, limit, max(0, limit - 1), window)

        count = int(results[1])
        oldest = results[2]
        if oldest:
            oldest_score = float(oldest[0][1])
            reset_seconds = max(1, int(oldest_score + window - now) + 1)
        else:
            reset_seconds = int(window)

        if count >= limit:
            return (False, limit, 0, reset_seconds)
        return (True, limit, max(0, limit - count - 1), reset_seconds)


# ---------------------------------------------------------------------------
# ASGI Middleware
# ---------------------------------------------------------------------------

class WidgetRateLimitMiddleware:
    """Shared, first-request-gating rate limiting for ``/api/widgets/*``.

    Pure ASGI — does not buffer streaming responses. The identifier is
    resolved here, before any route handler runs, so the very first request
    on a connection is limited; the Redis-backed store makes the decision
    hold across all workers.
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

        # Identifier: prefer the presented API key (hashed — no key material
        # in Redis), fall back to client IP. Resolved pre-handler so the
        # FIRST request is gated (P2-13).
        headers = dict(scope.get("headers", []))
        api_key = (headers.get(b"authorization") or b"").decode("latin-1").strip()
        if api_key.lower().startswith("bearer "):
            api_key = api_key[7:].strip()

        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        if api_key:
            digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]
            identifier = f"key:{digest}"
            limit = (
                config.WIDGET_RATE_LIMIT_SERVER_PER_WINDOW
                if api_key.startswith("ak_srv_")
                else config.WIDGET_RATE_LIMIT_PUBLIC_PER_WINDOW
            )
        else:
            identifier = f"ip:{client_ip}"
            limit = config.WIDGET_RATE_LIMIT_PUBLIC_PER_WINDOW

        allowed, rl_limit, remaining, reset_seconds = await asyncio.to_thread(
            self._store.check, identifier, limit
        )

        # Per-IP ceiling on the money-spending endpoints — applies even when
        # a key was presented (PRD-194 S5): a scraped ak_pub_ key replayed
        # from one machine hits the IP ceiling regardless of key limits.
        if allowed:
            ceiling_attr = _IP_CEILING_PATHS.get(path.rstrip("/"))
            if ceiling_attr is not None:
                ip_ceiling = int(getattr(config, ceiling_attr))
                endpoint_tag = path.rstrip("/").rsplit("/", 1)[-1]
                ip_allowed, ip_limit, ip_remaining, ip_reset = await asyncio.to_thread(
                    self._store.check,
                    f"ipceil:{endpoint_tag}:{client_ip}",
                    ip_ceiling,
                )
                if not ip_allowed:
                    allowed = False
                    rl_limit, remaining, reset_seconds = ip_limit, 0, ip_reset
                else:
                    remaining = min(remaining, ip_remaining)

        if not allowed:
            # 429 Too Many Requests
            retry_after = str(reset_seconds).encode()
            response_headers = [
                (b"content-type", b"application/json"),
                (b"retry-after", retry_after),
                (b"x-ratelimit-limit", str(rl_limit).encode()),
                (b"x-ratelimit-remaining", b"0"),
                (b"x-ratelimit-reset", str(reset_seconds).encode()),
            ]
            body = json.dumps({"detail": "Rate limit exceeded", "retry_after": reset_seconds}).encode()
            await send({"type": "http.response.start", "status": 429, "headers": response_headers})
            await send({"type": "http.response.body", "body": body})
            return

        # Inject rate-limit headers into downstream response
        rl_headers = [
            (b"x-ratelimit-limit", str(rl_limit).encode()),
            (b"x-ratelimit-remaining", str(remaining).encode()),
            (b"x-ratelimit-reset", str(reset_seconds).encode()),
        ]

        async def send_with_rate_limit(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(rl_headers)
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_rate_limit)
