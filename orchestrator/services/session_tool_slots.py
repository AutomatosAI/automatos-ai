"""F330 (night 9c): how many session tool calls run at once in this process.

Night 9c, build 18 (4 Oct, ~20:20Z): 26 cards were filed at once and the CLI
host had no ``--max-sessions`` cap, so up to 26 Claude Code sessions called
``query_database`` and the other session tools together, beside Auto's chats.
Twelve minutes in, the pool (10 + 20) was gone: 29 connections "idle in
transaction" for 12+ minutes, the loop watchdog showed the event loop stuck in
``pool._do_get``, and /health timed out.

Each call held its request's connection while the tool ran (an NL2SQL model
call lasts seconds) and the tool then opened sessions of its own, on the event
loop. With as many calls as the pool has connections, every connection belonged
to a call waiting for one more, and the wait itself held the loop, so no call
could ever give one back.

A tool call now first takes one of ``SESSION_TOOL_CONCURRENCY`` slots, so a
burst queues HERE, holding no connection, and at most that many calls use the
pool at once. Auth and the call counter are short and run before the slot, off
the loop (``api/session_tools.py``).
"""
from __future__ import annotations

import asyncio
import contextlib
import weakref
from typing import AsyncIterator

from config import config

# A semaphore belongs to the loop it first waits on; each loop (a test's
# ``asyncio.run``) gets its own, sized when that loop first asks.
_SLOTS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()
MIN_SLOTS = 1


def slot_count() -> int:
    """The configured number of slots, never fewer than one."""
    return max(MIN_SLOTS, int(getattr(config, "SESSION_TOOL_CONCURRENCY", MIN_SLOTS) or MIN_SLOTS))


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    slots = _SLOTS.get(loop)
    if slots is None:
        slots = asyncio.Semaphore(slot_count())
        _SLOTS[loop] = slots
    return slots


@contextlib.asynccontextmanager
async def session_tool_slot() -> AsyncIterator[None]:
    """Hold one slot for the length of a session tool call."""
    async with _semaphore():
        yield
