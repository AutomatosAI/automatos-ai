"""
Per-workspace rate limiting for security-sensitive operations — PRD-70 FIX-07.

Uses Redis sliding window counters keyed by (workspace_id[, subject_id], operation).
Endpoints call ``check_rate_limit()`` and raise 429 if exceeded.

Subject scope: pass an optional ``subject_id`` (typically the calling agent
id) to give each subject its own bucket. Without this, every agent in a
workspace shares one bucket — a single chatty Auto session can starve
mission tasks of platform_write headroom.

Usage::

    from core.security.rate_limiter import check_rate_limit

    # Workspace-scoped (default, legacy behaviour):
    await check_rate_limit(workspace_id, "git_clone", max_requests=5, window_seconds=3600)

    # Per-agent-scoped (preferred for in-chat platform tool calls):
    await check_rate_limit(workspace_id, "platform_write", subject_id=str(agent_id))
"""

import logging
import os
import time
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _env_limit(name: str, default_max: int, default_window: int) -> tuple[int, int]:
    """Read MAX/WINDOW pair from env with safe fallbacks."""
    try:
        max_req = int(os.getenv(f"RATE_LIMIT_{name.upper()}_MAX", str(default_max)))
        window = int(os.getenv(f"RATE_LIMIT_{name.upper()}_WINDOW_SECONDS", str(default_window)))
        return max(1, max_req), max(1, window)
    except (TypeError, ValueError):
        return default_max, default_window


# Default limits per operation. Overridable via env vars
# RATE_LIMIT_<OPERATION>_MAX and RATE_LIMIT_<OPERATION>_WINDOW_SECONDS.
DEFAULT_LIMITS: dict[str, tuple[int, int]] = {
    # (max_requests, window_seconds)
    "git_clone":       _env_limit("git_clone", 5, 3600),         # 5 per hour
    "nl2sql_query":    _env_limit("nl2sql_query", 30, 60),       # 30 per minute
    "admin_action":    _env_limit("admin_action", 20, 60),       # 20 per minute
    "skill_import":    _env_limit("skill_import", 3, 3600),      # 3 per hour
    "plugin_import":   _env_limit("plugin_import", 3, 3600),     # 3 per hour
    # Bumped from 10/min → 60/min and now scopes per-agent when caller
    # supplies subject_id. A long working chat that touches many agents
    # (job_title / team / description updates etc.) no longer starves
    # itself or mission tasks running in parallel.
    "platform_write":  _env_limit("platform_write", 60, 60),     # 60 write/destructive actions per minute, per subject
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
    subject_id: Optional[str] = None,
) -> None:
    """Check and increment rate limit counter. Raises HTTPException(429) if exceeded.

    Falls back to no-op if Redis is unavailable (fail-open for availability).

    ``subject_id`` (e.g. agent id) gives each subject its own bucket within
    the workspace. Without it, all callers in the workspace share one bucket.
    """
    defaults = DEFAULT_LIMITS.get(operation, (60, 60))
    limit = max_requests or defaults[0]
    window = window_seconds or defaults[1]

    redis = _get_redis()
    if redis is None:
        logger.debug("Redis unavailable — rate limit check skipped for %s", operation)
        return

    key_suffix = f"{workspace_id}:{subject_id}" if subject_id else workspace_id
    key = f"ratelimit:{operation}:{key_suffix}"
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
            scope = f"workspace={workspace_id}"
            if subject_id:
                scope += f" subject={subject_id}"
            logger.warning(
                "Rate limit exceeded: %s operation=%s count=%d limit=%d window=%ds",
                scope, operation, current_count, limit, window,
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
