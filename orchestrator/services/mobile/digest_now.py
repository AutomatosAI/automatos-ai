"""The daily digest on request, "Send me today's brief now" (``POST /api/v1/mobile/digest/now``).

The route (``api/mobile_push.py``) answers at once and hands the work to :func:`launch`, which
runs ``services.mobile.digest.post_now`` as a guarded task on the event loop: Auto's Read made
and posted into the caller's Auto thread exactly as the scheduled digest posts it (origin
``daily_digest``, which raises ``auto_message`` and so the phone's push), with no
``mobile_digest_posts`` claim read or written, so the scheduled digest still goes out.

* Who: the caller only, in the workspace they called for, while they belong to it
  (``digest.belongs``, the scheduled digest's own read of who does: an owner or an active
  member of any role, the local operator on a local install). Never a person the request names.
* How often: one ask per person every ``config.DIGEST_NOW_MIN_INTERVAL_SECONDS``
  (:class:`Cooldown`). The mark is a Redis key set only if absent and expiring with the
  interval (``SET NX EX``), shared by every worker; its TTL is the wait. A refused ask leaves
  the mark as it is, so the wait it is told is the true one. Redis unreachable lets the ask
  through and logs an ERROR naming this limiter, as the platform's other limiters do.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Optional

from config import config
from core.utils.background_tasks import launch_guarded
from services.mobile import digest

logger = logging.getLogger(__name__)

COOLDOWN_KEY_PREFIX = "mobile:digest_now"
COOLDOWN_MARK = "1"
LIMITER_LABEL = "digest-now limiter"
# What launch_guarded files an uncaught crash under (none is expected: post_now never raises).
SUBSYSTEM = "mobile"
OPERATION = "digest_now"


def _platform_redis() -> Optional[Any]:
    """The platform Redis connection, or None when there is none."""
    from core.redis.client import get_redis_client

    client = get_redis_client()
    return client.get_redis() if client is not None else None


class Cooldown:
    """One ask per person every ``DIGEST_NOW_MIN_INTERVAL_SECONDS``, held in Redis across workers."""

    def __init__(self, redis_factory: Optional[Callable[[], Any]] = None) -> None:
        self._redis_factory = redis_factory or _platform_redis

    @staticmethod
    def key(user_id: int) -> str:
        """The person's mark: one per person, whichever workspace they ask in."""
        return f"{COOLDOWN_KEY_PREFIX}:{int(user_id)}"

    def _redis(self) -> Optional[Any]:
        try:
            return self._redis_factory()
        except Exception:
            logger.exception("%s: no Redis connection", LIMITER_LABEL)
            return None

    def take(self, user_id: int) -> Optional[int]:
        """Mark the person's ask: None when they may ask now, else the seconds until they may."""
        interval = int(config.DIGEST_NOW_MIN_INTERVAL_SECONDS)
        redis = self._redis()
        if redis is None:
            logger.error("%s FAIL-OPEN: Redis unreachable; person %s's ask let through", LIMITER_LABEL, user_id)
            return None
        key = self.key(user_id)
        try:
            if redis.set(key, COOLDOWN_MARK, nx=True, ex=interval):
                return None
            left = int(redis.ttl(key))
        except Exception:
            logger.exception("%s FAIL-OPEN: Redis failed; person %s's ask let through", LIMITER_LABEL, user_id)
            return None
        return max(1, left)

    def give_back(self, user_id: int) -> None:
        """Drop the person's mark: the ask it counted never started. Never raises."""
        redis = self._redis()
        if redis is None:
            return
        try:
            redis.delete(self.key(user_id))
        except Exception:
            logger.exception("%s: person %s's mark was not dropped", LIMITER_LABEL, user_id)


# Module-level so a test can hand it a fake Redis.
cooldown = Cooldown()


def launch(workspace_id: Any, user_id: int) -> None:
    """Start today's digest for the person, in the background.

    Call it on the event loop: a threadpool route goes through ``anyio.from_thread``.
    """
    workspace = uuid.UUID(str(workspace_id))
    launch_guarded(
        digest.post_now(workspace, int(user_id)),
        subsystem=SUBSYSTEM, operation=OPERATION, workspace_id=workspace,
    )


__all__ = ["COOLDOWN_KEY_PREFIX", "Cooldown", "cooldown", "launch"]
