"""
Per-workspace rate limiting for security-sensitive operations — PRD-70 FIX-07.

Uses Redis sliding window counters keyed by (workspace_id, operation).
Endpoints call ``check_rate_limit()`` and raise 429 if exceeded.

Usage::

    from core.security.rate_limiter import check_rate_limit

    # Inside an endpoint:
    await check_rate_limit(workspace_id, "git_clone", max_requests=5, window_seconds=3600)
"""

import logging
import time
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Default limits per operation
DEFAULT_LIMITS: dict[str, tuple[int, int]] = {
    # (max_requests, window_seconds)
    "git_clone": (5, 3600),       # 5 per hour
    "nl2sql_query": (30, 60),     # 30 per minute
    "admin_action": (20, 60),     # 20 per minute
    "skill_import": (3, 3600),    # 3 per hour
    "plugin_import": (3, 3600),   # 3 per hour
}


def _get_redis():
    """Lazy import to avoid circular deps and missing redis in test."""
    try:
        from core.redis import get_redis_client
        client = get_redis_client()
        if client is None:
            return None
        return client.get_redis()
    except Exception:
        return None


async def check_rate_limit(
    workspace_id: str,
    operation: str,
    max_requests: Optional[int] = None,
    window_seconds: Optional[int] = None,
) -> None:
    """Check and increment rate limit counter. Raises HTTPException(429) if exceeded.

    Falls back to no-op if Redis is unavailable (fail-open for availability).
    """
    defaults = DEFAULT_LIMITS.get(operation, (60, 60))
    limit = max_requests or defaults[0]
    window = window_seconds or defaults[1]

    redis = _get_redis()
    if redis is None:
        logger.debug("Redis unavailable — rate limit check skipped for %s", operation)
        return

    key = f"ratelimit:{operation}:{workspace_id}"
    now = time.time()
    window_start = now - window

    try:
        pipe = redis.pipeline()
        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current window
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Set TTL so keys auto-expire
        pipe.expire(key, window + 10)
        results = pipe.execute()

        current_count = results[1]

        if current_count >= limit:
            logger.warning(
                "Rate limit exceeded: workspace=%s operation=%s count=%d limit=%d",
                workspace_id, operation, current_count, limit,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for {operation}. "
                       f"Max {limit} requests per {window}s.",
                headers={"Retry-After": str(window)},
            )
    except HTTPException:
        raise
    except Exception as e:
        # Fail open — don't block requests if Redis has issues
        logger.warning("Rate limit check failed (fail-open): %s", e)
