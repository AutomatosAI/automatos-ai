"""
Session-Scoped Execution Queue
==============================

Ensures that concurrent requests for the SAME chat session are serialized.
Prevents race conditions where two parallel requests corrupt shared state
(message history, memory writes, tool execution).

Different sessions run in full parallel — only same-session requests queue.

Inspired by OpenClaw's "lane queues" pattern.

Usage:
    session_queue = get_session_queue()
    async with session_queue.acquire("ws123:chat456"):
        # Only one request per session executes here at a time
        response = await process_chat(...)
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict

logger = logging.getLogger(__name__)

# Stale lock cleanup threshold (seconds). Locks unused for this long get pruned.
_STALE_THRESHOLD = 300  # 5 minutes
_CLEANUP_INTERVAL = 60  # Run cleanup every 60 seconds
_ACQUIRE_TIMEOUT = 120  # Max wait time for a lock (seconds)


class _SessionSlot:
    """A single session's lock + last-used timestamp."""
    __slots__ = ("lock", "last_used")

    def __init__(self):
        self.lock = asyncio.Lock()
        self.last_used = time.monotonic()

    def touch(self):
        self.last_used = time.monotonic()


class SessionQueue:
    """
    Per-session serial execution queue.

    Each unique session key gets its own asyncio.Lock.
    Requests with the same key wait in FIFO order.
    Different keys execute concurrently.
    """

    def __init__(self):
        self._slots: Dict[str, _SessionSlot] = {}
        self._slots_lock = asyncio.Lock()  # Protects the dict itself
        self._last_cleanup = time.monotonic()

    @asynccontextmanager
    async def acquire(self, session_key: str):
        """
        Acquire exclusive access for a session.

        Args:
            session_key: Unique session identifier (e.g. "{workspace_id}:{chat_id}")

        Yields:
            None — the block inside `async with` has exclusive session access.

        Raises:
            asyncio.TimeoutError: If lock not acquired within timeout.
        """
        slot = await self._get_or_create_slot(session_key)

        try:
            await asyncio.wait_for(slot.lock.acquire(), timeout=_ACQUIRE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(
                "[SessionQueue] Timeout acquiring lock for session %s after %ds",
                session_key, _ACQUIRE_TIMEOUT,
            )
            raise

        slot.touch()
        logger.debug("[SessionQueue] Acquired lock for session %s", session_key)

        try:
            yield
        finally:
            slot.lock.release()
            slot.touch()
            logger.debug("[SessionQueue] Released lock for session %s", session_key)

        # Periodic cleanup (non-blocking)
        await self._maybe_cleanup()

    async def _get_or_create_slot(self, session_key: str) -> _SessionSlot:
        """Get existing slot or create a new one (thread-safe)."""
        async with self._slots_lock:
            if session_key not in self._slots:
                self._slots[session_key] = _SessionSlot()
                logger.debug("[SessionQueue] Created slot for session %s (total=%d)", session_key, len(self._slots))
            return self._slots[session_key]

    async def _maybe_cleanup(self):
        """Prune stale slots that haven't been used recently."""
        now = time.monotonic()
        if now - self._last_cleanup < _CLEANUP_INTERVAL:
            return

        async with self._slots_lock:
            self._last_cleanup = now
            stale_keys = [
                k for k, slot in self._slots.items()
                if not slot.lock.locked() and (now - slot.last_used) > _STALE_THRESHOLD
            ]
            for key in stale_keys:
                del self._slots[key]

            if stale_keys:
                logger.info(
                    "[SessionQueue] Cleaned up %d stale sessions (remaining=%d)",
                    len(stale_keys), len(self._slots),
                )

    @property
    def active_sessions(self) -> int:
        """Number of currently tracked sessions."""
        return len(self._slots)

    @property
    def locked_sessions(self) -> int:
        """Number of sessions with an active lock."""
        return sum(1 for slot in self._slots.values() if slot.lock.locked())


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: SessionQueue | None = None


def get_session_queue() -> SessionQueue:
    """Get the global SessionQueue singleton."""
    global _instance
    if _instance is None:
        _instance = SessionQueue()
        logger.info("[SessionQueue] Initialized global session queue")
    return _instance
