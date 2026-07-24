"""Webhook replay/dedup guard — PRD-194 S2 (P2-13, security §1.1/§1.3.c).

The three EXTERNAL webhook lanes (Composio ``/webhook``, workspace
``/ws/{key}``, playbook ``/recipe/{id}``) execute real agent/playbook work
from an inbound POST. Providers redeliver on slow responses, and until this
guard existed a redelivered event simply ran again — burning tokens and
re-firing side-effects. This module gives those lanes two checks:

- :func:`seen_before` — event dedup on the provider's event id
  (``webhook-id`` / ``event_id`` / ``update_id``), backed by Redis
  ``SET NX EX`` (atomic first-writer-wins, self-expiring). A redelivered
  event becomes a fast no-op ack at the ingest point.
- :func:`timestamp_is_stale` — replay defence where a provider timestamp
  exists: outside the skew window ⇒ reject.

Deliberate semantics (locked decisions, PRD-194):

- **Redis SETNX + TTL, no table, no migration.** Reuses the platform Redis
  client (``core/redis/client.py`` + ``config.REDIS_URL``); the guard is
  ephemeral by nature, keyed per-lane so a later single-surface
  consolidation (P2-25) is a merge, not a rewrite.
- **Redis down ⇒ fail OPEN for dedup** (process the event, log LOUDLY).
  Availability of the lane beats replay protection when the guard store is
  down — the signature gate (S1) still stands in front of every lane.
- **Mark-on-arrival, not mark-on-success.** The dedup mark is written when
  the event is first accepted, before execution. Marking after success
  would reopen the exact double-execute window this guard closes (a slow
  handler triggers the provider retry *while still running*).
- **The Shopify ``/events`` path is NOT routed through this guard** —
  PRD-189 S3's per-workspace debounce owns that lane (different endpoint,
  complementary guard; it coalesces *distinct* events, this rejects *the
  same* event redelivered).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from config import config

logger = logging.getLogger(__name__)

_KEY_PREFIX = "webhook:dedup"


def _get_redis():
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


def _seen_before_sync(lane: str, event_id: str) -> bool:
    """Atomically mark ``(lane, event_id)`` as seen; ``True`` if it already was.

    ``SET NX EX``: the first delivery writes the mark and returns falsy-negative
    (not seen); every redelivery inside the TTL finds the mark and returns
    ``True``. Redis unavailable/erroring ⇒ **fail OPEN** (return ``False``,
    log loudly) — the lane must keep working through a cache outage; replay
    protection resumes when Redis does.
    """
    redis = _get_redis()
    if redis is None:
        logger.error(
            "[webhook-dedup] Redis unavailable — replay/dedup guard DOWN, "
            "processing event without replay protection (lane=%s)",
            lane,
        )
        return False

    key = f"{_KEY_PREFIX}:{lane}:{event_id}"
    try:
        first_delivery = redis.set(
            key, "1", nx=True, ex=config.WEBHOOK_DEDUP_TTL_SECONDS
        )
        return not bool(first_delivery)
    except Exception:
        logger.error(
            "[webhook-dedup] Redis error — replay/dedup guard DOWN, "
            "processing event without replay protection (lane=%s)",
            lane,
            exc_info=True,
        )
        return False


async def seen_before(lane: str, event_id: Optional[str]) -> bool:
    """``True`` when this event id was already accepted on this lane.

    ``lane`` scopes the key (e.g. ``"composio"``, ``"ws:{workspace_id}"``,
    ``"recipe:{webhook_id}"``) so ids that are only unique per-provider-
    connection (Telegram ``update_id``) cannot collide across tenants.
    No event id ⇒ nothing to dedup on ⇒ process normally.
    """
    if not event_id:
        return False
    return await asyncio.to_thread(_seen_before_sync, lane, str(event_id))


def timestamp_is_stale(ts_raw: Optional[str]) -> bool:
    """Replay defence: ``True`` when a provider timestamp exists and is
    outside the configured skew window.

    No header ⇒ ``False`` (nothing to check — dedup still applies). A
    present-but-garbage timestamp is treated as stale (fail closed): a
    caller that supplies the header must supply a valid one.
    """
    if ts_raw is None or ts_raw == "":
        return False
    try:
        ts = float(ts_raw)
    except (TypeError, ValueError):
        return True
    return abs(time.time() - ts) > config.WEBHOOK_TIMESTAMP_SKEW_SECONDS
