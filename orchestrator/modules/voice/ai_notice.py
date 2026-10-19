"""Auto Live says it is an AI (PRD-264 EUAI-04, EU AI Act Art. 50(1), Guidelines ¶37).

Two duties on a live call, both kept by ``live_socket.LiveCallSocket``:

* **Auto speaks first, and says it is an AI.** The begin frame (response 0)
  carries :func:`greeting`. Retell speaks the content of response 0 as the
  call's opening line, so it is never empty.
* **Reminders in a longer call.** After an automatic reconnect, and every
  ``config.VOICE_AI_REMINDER_MINUTES`` of call time, the next response opens
  with "Just so you know, I'm an AI assistant." The rule and the texts live in
  :mod:`modules.policy.ai_disclosure`; this module holds the settings and the
  reconnect state.

A reconnect is a new socket for a call that already had one (Retell opens a
new connection to the same URL, up to twice, when pings stop). The API runs
several workers, so the "this call has had a socket" mark lives in Redis,
keyed by a hash of the call id (the call id is the socket's credential), with
a TTL a little past the longest call. When Redis can't be read the socket is
treated as a reconnect: the caller hears one reminder too many, never one too
few.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Optional

from config import config
from modules.policy import ai_disclosure

logger = logging.getLogger(__name__)

SOCKET_MARK_KEY_PREFIX = "ai_notice:voice"
SOCKET_MARK_VALUE = "1"
SECONDS_PER_MINUTE = 60
# The mark outlives the longest call by this much, so a late reconnect still counts.
SOCKET_MARK_GRACE_SECONDS = 300
MIN_CALL_MINUTES = 1
# Between the reminder and the words that follow it in the same response.
REMINDER_SEPARATOR = " "


def greeting() -> str:
    """The line Auto speaks first: ``VOICE_AI_GREETING``, or the default when it lacks the AI term."""
    return ai_disclosure.voice_greeting(config.VOICE_AI_GREETING, config.VOICE_LIVE_LANGUAGE)


def reminder_interval_seconds() -> float:
    """``VOICE_AI_REMINDER_MINUTES`` in seconds (0 or less: every response)."""
    return float(config.VOICE_AI_REMINDER_MINUTES) * SECONDS_PER_MINUTE


def reminder_due(*, reconnected: bool, last_notice_at: float, now: float) -> bool:
    """Whether the next response opens with the reminder, at the configured interval."""
    return ai_disclosure.voice_reminder_due(
        reconnected=reconnected,
        last_notice_at=last_notice_at,
        now=now,
        interval_seconds=reminder_interval_seconds(),
    )


def reminder_frame_text() -> str:
    """The reminder as the first content frame of a response."""
    return f"{ai_disclosure.VOICE_REMINDER}{REMINDER_SEPARATOR}"


def socket_mark_key(call_id: str) -> str:
    """The Redis key that says ``call_id`` has had a socket."""
    digest = hashlib.sha256(str(call_id).encode("utf-8")).hexdigest()
    return f"{SOCKET_MARK_KEY_PREFIX}:{digest}"


def _mark_ttl_seconds() -> int:
    """A little longer than the longest call (``VOICE_LIVE_MAX_CALL_MINUTES``)."""
    minutes = max(MIN_CALL_MINUTES, int(config.VOICE_LIVE_MAX_CALL_MINUTES))
    return minutes * SECONDS_PER_MINUTE + SOCKET_MARK_GRACE_SECONDS


def _get_redis() -> Optional[Any]:
    """The platform Redis connection, or ``None`` when it is not available."""
    try:
        from core.redis.client import get_redis_client

        client = get_redis_client()
        return client.get_redis() if client is not None else None
    except Exception:
        logger.exception("voice ai notice: Redis client unavailable; treating the socket as a reconnect")
        return None


def _mark_socket_open_sync(call_id: str) -> bool:
    """Mark ``call_id`` as having a socket; True when it already had one (a reconnect)."""
    redis = _get_redis()
    if redis is None:
        logger.warning("voice ai notice: Redis not configured; treating the socket as a reconnect")
        return True
    try:
        first = redis.set(socket_mark_key(call_id), SOCKET_MARK_VALUE, nx=True, ex=_mark_ttl_seconds())
    except Exception:
        logger.exception("voice ai notice: Redis write failed; treating the socket as a reconnect")
        return True
    return not first


async def socket_reconnected(call_id: str) -> bool:
    """Record this socket for ``call_id``; True when it is an automatic reconnect.

    The Redis round trip runs off the event loop, which must keep echoing
    Retell's pings.
    """
    return await asyncio.to_thread(_mark_socket_open_sync, call_id)
