"""
Wizard progress backbone — Redis-backed SSE event stream
==========================================================

Two concerns, one module:

1. ``emit(profile_id, stage, message, ...)``
   - Called from anywhere inside the wizard pipeline to publish an event.
   - Writes to a capped Redis LIST (``wizard:progress:list:{profile_id}``)
     so late subscribers can replay everything that has already happened.
   - Publishes to a Redis pub/sub channel
     (``wizard:progress:{profile_id}``) so live subscribers get the
     event pushed to them instantly.

2. ``stream(profile_id)`` — async generator that:
   - Yields every event currently in the LIST (cold replay),
   - Then subscribes to the channel and yields new events until the
     caller disconnects or a ``stage == "complete"`` / ``"failed"``
     terminal event is observed.

The SSE endpoint in ``api/wizard.py`` wraps ``stream()`` with
``StreamingResponse`` so a browser ``EventSource`` can consume it.

Design notes
------------
- We use ``redis.asyncio`` directly instead of the orchestrator's existing
  ``RedisClient`` helper because that helper is sync-only for publish and
  builds a new async connection per pubsub subscription. This module needs
  async on both sides and shares one async client per process.
- If Redis is not configured we fall back to a no-op emitter. The wizard
  still functions; the UI just won't see live events. This keeps the
  wizard usable in local dev without Redis.
- Events are cheap, coarse, and structured. Stages are drawn from the
  fixed vocabulary below so the frontend can render icons/colors without
  string sniffing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

from config import config

logger = logging.getLogger(__name__)

# Stage taxonomy — used by the frontend for icons / state transitions
STAGE_SCAN = "scan"
STAGE_SCRAPE = "scrape"
STAGE_INGEST = "ingest"
STAGE_GRAPHIFY = "graphify"
STAGE_PROFILE = "profile"
STAGE_PLAN = "plan"
STAGE_COMPLETE = "complete"
STAGE_FAILED = "failed"

# Redis key templates
_LIST_KEY = "wizard:progress:list:{profile_id}"
_CHANNEL = "wizard:progress:{profile_id}"

# Replay buffer cap — enough to hold every event a run can produce
_LIST_MAX = 500
_LIST_TTL_SECONDS = 60 * 60  # 1 hour

# Terminal stages — the stream() generator returns after emitting these
_TERMINAL_STAGES = {STAGE_COMPLETE, STAGE_FAILED}


# ---------------------------------------------------------------------------
# Async Redis client — shared singleton
# ---------------------------------------------------------------------------

_async_redis: Optional[Any] = None
_async_redis_lock = asyncio.Lock()


async def _get_async_redis() -> Optional[Any]:
    """Return a shared ``redis.asyncio.Redis`` client, or ``None``.

    Lazily initialized on first use. Returns ``None`` if Redis is not
    configured so callers can fall back to no-op behavior.
    """
    global _async_redis
    if _async_redis is not None:
        return _async_redis

    async with _async_redis_lock:
        if _async_redis is not None:
            return _async_redis

        url = config.REDIS_URL
        if not url:
            logger.warning(
                "wizard.progress: REDIS_URL not configured — progress events disabled"
            )
            return None

        try:
            import redis.asyncio as aioredis  # local import to keep module cheap
            client = aioredis.from_url(url, decode_responses=True)
            # Sanity check the connection once up front
            await client.ping()
            _async_redis = client
            logger.info("wizard.progress: connected to Redis at %s", _safe_url(url))
            return _async_redis
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "wizard.progress: failed to connect to Redis: %s",
                exc, exc_info=True,
            )
            return None


def _safe_url(url: str) -> str:
    """Strip credentials from a Redis URL for logging."""
    if "@" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        return f"{scheme}://***@{host}"
    return url


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


async def emit(
    profile_id: str,
    stage: str,
    message: str,
    *,
    level: str = "info",
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Publish one progress event for a wizard run.

    Safe to call with or without Redis available. Never raises — progress
    events must not be able to kill the wizard pipeline.

    Args:
        profile_id: ``business_profiles.id`` as a string.
        stage: One of the ``STAGE_*`` constants.
        message: Human-readable log line to render in the terminal feed.
        level: ``info`` | ``warn`` | ``error`` — drives UI colour.
        meta: Optional structured payload (counts, URLs, etc).
    """
    event = {
        "ts": time.time(),
        "stage": stage,
        "level": level,
        "message": message,
        "meta": meta or {},
    }
    payload = json.dumps(event, default=str)

    client = await _get_async_redis()
    if client is None:
        logger.debug("wizard.progress emit (no-redis) profile=%s %s", profile_id, message)
        return

    list_key = _LIST_KEY.format(profile_id=profile_id)
    channel = _CHANNEL.format(profile_id=profile_id)

    try:
        pipe = client.pipeline()
        pipe.rpush(list_key, payload)
        pipe.ltrim(list_key, -_LIST_MAX, -1)
        pipe.expire(list_key, _LIST_TTL_SECONDS)
        pipe.publish(channel, payload)
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "wizard.progress emit failed profile=%s stage=%s: %s",
            profile_id, stage, exc,
        )


async def clear(profile_id: str) -> None:
    """Delete any existing events for a profile.

    Called at the start of a new run so the frontend doesn't replay
    events from a previous attempt.
    """
    client = await _get_async_redis()
    if client is None:
        return
    try:
        await client.delete(_LIST_KEY.format(profile_id=profile_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("wizard.progress clear failed profile=%s: %s", profile_id, exc)


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------


async def stream(profile_id: str) -> AsyncIterator[str]:
    """Yield SSE ``data:`` frames for a profile.

    Behaviour:
      1. Replay everything already in the Redis LIST.
      2. Subscribe to the live channel and yield new events as they arrive.
      3. Return when a terminal stage (``complete``/``failed``) is seen
         OR when the consumer cancels the iterator.

    Emits a ``: keepalive`` comment every 15 seconds so browsers and
    load balancers don't idle-close the connection during long graphify
    phases.
    """
    client = await _get_async_redis()
    if client is None:
        # No Redis → emit a single failure event so the UI gets feedback
        yield _sse_frame({
            "ts": time.time(),
            "stage": STAGE_FAILED,
            "level": "error",
            "message": "Progress stream unavailable (Redis not configured)",
            "meta": {},
        })
        return

    list_key = _LIST_KEY.format(profile_id=profile_id)
    channel = _CHANNEL.format(profile_id=profile_id)

    # --- 1. Replay -----------------------------------------------------
    saw_terminal = False
    try:
        existing = await client.lrange(list_key, 0, -1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("wizard.progress replay failed profile=%s: %s", profile_id, exc)
        existing = []

    for raw in existing:
        yield f"data: {raw}\n\n"
        try:
            parsed = json.loads(raw)
            if parsed.get("stage") in _TERMINAL_STAGES:
                saw_terminal = True
        except (json.JSONDecodeError, TypeError):
            continue

    if saw_terminal:
        # Run already finished — no need to subscribe
        return

    # --- 2. Live subscribe --------------------------------------------
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(channel)
    except Exception as exc:  # noqa: BLE001
        logger.error("wizard.progress subscribe failed profile=%s: %s", profile_id, exc)
        return

    try:
        last_keepalive = time.time()
        while True:
            try:
                msg = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                msg = None

            if msg is not None and msg.get("type") == "message":
                raw = msg.get("data")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                if not isinstance(raw, str):
                    continue
                yield f"data: {raw}\n\n"
                try:
                    parsed = json.loads(raw)
                    if parsed.get("stage") in _TERMINAL_STAGES:
                        return
                except (json.JSONDecodeError, TypeError):
                    continue

            # Keepalive comment — SSE spec: lines starting with ':' are ignored
            now = time.time()
            if now - last_keepalive >= 15:
                yield ": keepalive\n\n"
                last_keepalive = now
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
        except Exception:  # noqa: BLE001
            pass


def _sse_frame(event: Dict[str, Any]) -> str:
    """Serialize an event dict as an SSE ``data:`` frame."""
    return f"data: {json.dumps(event, default=str)}\n\n"
