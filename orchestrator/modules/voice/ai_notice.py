"""Auto Live says it is an AI (PRD-264 EUAI-04, EU AI Act Art. 50(1), Guidelines ¶37).

Two duties on a live call, both kept by ``live_socket.LiveCallSocket``:

* **Auto speaks first, and says it is an AI.** The begin frame (response 0)
  carries the greeting. Retell speaks the content of response 0 as the call's
  opening line, so it is never empty.
* **Reminders, only when a deployment asks for them.** The notice is once per
  call (the product owner's policy, 23 Oct 2026): ``VOICE_AI_REMINDER_MINUTES``
  is 0 by default, and 0 or less never reminds. Above 0, after an automatic
  reconnect and every that many minutes of call time, the next response opens
  with "Just so you know, I'm an AI assistant." This module holds that rule,
  its setting and the reconnect state.

The texts are built in ``modules/policy/ai_disclosure.py``, the one place a
notice is built. The route (``api/voice_retell.py``) lends them to the socket
through ``live_socket.SocketPorts``, so this feature module never imports
another one (the PRD-211 independence contract in ``orchestrator/.importlinter``).

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

logger = logging.getLogger(__name__)

SOCKET_MARK_KEY_PREFIX = "ai_notice:voice"
SOCKET_MARK_VALUE = "1"
SECONDS_PER_MINUTE = 60
# The mark outlives the longest call by this much, so a late reconnect still counts.
SOCKET_MARK_GRACE_SECONDS = 300
MIN_CALL_MINUTES = 1
# Between the reminder and the words that follow it in the same response.
REMINDER_SEPARATOR = " "


def reminder_interval_seconds() -> float:
    """``VOICE_AI_REMINDER_MINUTES`` in seconds (0 or less: reminders are off)."""
    return float(config.VOICE_AI_REMINDER_MINUTES) * SECONDS_PER_MINUTE


def reminders_on() -> bool:
    """Whether this deployment reminds at all: ``VOICE_AI_REMINDER_MINUTES`` above 0."""
    return reminder_interval_seconds() > 0


def reminder_due(*, reconnected: bool, last_notice_at: float, now: float) -> bool:
    """Whether Auto's next spoken response opens with the reminder.

    Never while reminders are off (the default: the greeting is the call's one
    notice). When they are on: after an automatic reconnect, and once
    ``VOICE_AI_REMINDER_MINUTES`` of call time have passed since the caller last
    heard that Auto is an AI.
    """
    if not reminders_on():
        return False
    if reconnected:
        return True
    return now - last_notice_at >= reminder_interval_seconds()


def reminder_frame_text(reminder: str) -> str:
    """``reminder`` as the first content frame of a response."""
    return f"{reminder}{REMINDER_SEPARATOR}"


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
