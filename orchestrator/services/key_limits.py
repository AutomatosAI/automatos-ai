"""Per-key limits for a product's server key on the routes named for it (PRD-274 Wave 1, §5.2).

"Per-key spend and rate limits are enforced; server keys get a per-key ceiling." A key carries two
limits its owner may set when minting it, both per minute, and until now neither was read:

* **Requests** (``rate_limit_requests``): counted in a Redis sliding window per key, the window the
  widget plane already uses (``api.widgets.rate_limit.RateLimitStore``). The platform's ceiling,
  ``SERVER_KEY_REQUESTS_PER_MINUTE``, applies to every server key; a key's own limit can only
  lower it.
* **Tokens** (``rate_limit_tokens``): the tokens the key used in the last minute, from the usage
  ledger (``llm_usage``): ``board_task:<id>`` for each ticket it filed, and ``chat:<id>`` for each
  chat its service user owns (its headless chat turns).
  At or over the limit, the key starts no new work until the minute rolls on. A run already going
  is not stopped.

Over either limit is a 429 with ``Retry-After``. If Redis can't be reached, the request count
fails open, as the widget limiter does, and is logged loudly; the token count is a database read
and doesn't depend on Redis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy import text

from core.auth.server_key_gate import API_KEY_FILER

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60
LIMITER_LABEL = "server key limiter"
BUCKET = "server-key:{principal}"  # the key's principal, sdk:<key row id>

_TOKENS_LAST_MINUTE = text("""
    SELECT COALESCE(SUM(u.total_tokens), 0)
    FROM llm_usage u
    WHERE u.workspace_id = CAST(:ws AS uuid)
      AND u.created_at > now() - make_interval(secs => :window)
      AND (u.execution_id IN (
              SELECT 'board_task:' || t.id FROM board_tasks t
              WHERE t.workspace_id = CAST(:ws AS uuid)
                AND t.created_by_type = :filer AND t.created_by_id = :by)
           OR u.execution_id IN (
              SELECT 'chat:' || c.id::text FROM chats c
              JOIN sdk_api_keys k ON k.service_user_id = c.user_id
              WHERE k.id = CAST(:key AS uuid)))
""")


@dataclass(frozen=True)
class KeyLimits:
    """A server key's limits, per minute."""

    requests: int
    tokens: Optional[int] = None


def limits_for(db: Any, key_id: Any) -> KeyLimits:
    """The key's own request limit under the platform's ceiling, and its token limit."""
    from config import config
    from core.models.sdk_api_keys import SdkApiKey

    ceiling = int(config.SERVER_KEY_REQUESTS_PER_MINUTE)
    row = (db.query(SdkApiKey.rate_limit_requests, SdkApiKey.rate_limit_tokens)
           .filter(SdkApiKey.id == key_id).first())
    own_requests, own_tokens = (row[0], row[1]) if row else (None, None)
    requests = min(own_requests, ceiling) if own_requests and own_requests > 0 else ceiling
    return KeyLimits(requests=requests, tokens=own_tokens if own_tokens and own_tokens > 0 else None)


def tokens_last_minute(db: Any, workspace_id: Any, key_principal: str, key_id: Any) -> int:
    """Tokens the key used in the last minute: its tickets' runs, and its chat turns (the chats its
    service user owns, ``api.chat_server_key``)."""
    return int(db.execute(_TOKENS_LAST_MINUTE, {"ws": str(workspace_id), "window": WINDOW_SECONDS,
                                                "filer": API_KEY_FILER, "by": key_principal,
                                                "key": str(key_id)}).scalar() or 0)


def enforce_key_limits(db: Any, ctx: Any, store: Any = None) -> None:
    """429 when the key is over its requests or its tokens for the minute; nothing otherwise."""
    from api.widgets.rate_limit import RateLimitStore

    limits = limits_for(db, ctx.api_key_id)
    principal = ctx.user.id  # sdk:<key row id>: names the key without its secret
    allowed, limit, _remaining, reset = (store or RateLimitStore()).check(
        BUCKET.format(principal=principal), limits.requests, WINDOW_SECONDS, label=LIMITER_LABEL)
    if not allowed:
        logger.warning("[key_limits] %s over %d requests a minute", principal, limit)
        raise _over(f"This key's limit of {limit} requests a minute is reached", reset)
    if limits.tokens is None:
        return
    used = tokens_last_minute(db, ctx.workspace_id, principal, ctx.api_key_id)
    if used >= limits.tokens:
        logger.warning("[key_limits] %s used %d of %d tokens this minute", principal, used, limits.tokens)
        raise _over(f"This key's limit of {limits.tokens} tokens a minute is reached", WINDOW_SECONDS)


def _over(detail: str, retry_after: int) -> HTTPException:
    """The 429 for a key over a limit, with when to retry."""
    return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail,
                         headers={"Retry-After": str(max(1, int(retry_after)))})


def enforce_on_own_session(ctx: Any) -> Any:
    """``enforce_key_limits`` on a session of its own, always closed (threadpool); answers ``ctx``."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        enforce_key_limits(db, ctx)
        return ctx
    finally:
        db.close()
