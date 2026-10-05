"""#837: an async client made for one call is closed on that call's own event loop.

Mission dispatch matches a step's agent from synchronous code, so its embedding call
runs in an event loop of its own (``asyncio.run`` on a worker thread) that ends with
the call. The OpenAI SDK's async client, left open, closes itself when it is
garbage-collected, with a task on whatever loop is running at that moment: usually
the main loop, which cannot touch a socket of the loop that has ended. Each dispatch
logged "Task exception was never retrieved ... AsyncClient.aclose() ... unable to
perform operation on <TCPTransport closed=True>" (30 in one evening).

``closes_its_clients`` wraps a coroutine function. Every object ``adopt``-ed while
it runs (an ``EmbeddingManager`` adopts itself when it is made) has its
``aclose()`` awaited when the call ends, on the call's own loop, whether the call
returned, raised or was cancelled by a timeout. Outside such a call ``adopt`` does
nothing.
"""
from __future__ import annotations

import functools
import logging
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, List, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# The objects made during the current ``closes_its_clients`` call; None outside one.
# A list, not a tuple: a task the call starts runs in a copy of its context, and the
# objects made there must reach the same list.
_adopted: ContextVar[Optional[List[Any]]] = ContextVar("loop_scoped_clients", default=None)


def adopt(owner: Any) -> None:
    """Have ``owner`` (it has ``async aclose()``) closed when the current
    ``closes_its_clients`` call ends. Outside such a call, nothing happens."""
    scope = _adopted.get()
    if scope is not None:
        scope.append(owner)


def closes_its_clients(fn: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
    """Wrap the coroutine function ``fn``: what it adopts is closed on its loop when it ends."""
    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> _T:
        scope: List[Any] = []
        token = _adopted.set(scope)
        try:
            return await fn(*args, **kwargs)
        finally:
            _adopted.reset(token)
            await _close_all(scope)
    return wrapped


async def _close_all(owners: Sequence[Any]) -> None:
    """Close each owner; a failed close is logged and never fails the call it ends."""
    for owner in owners:
        try:
            await owner.aclose()
        except Exception:  # noqa: BLE001 — the call's own result or error stands
            logger.warning("[loop-scoped] closing %s failed", type(owner).__name__, exc_info=True)


__all__ = ["adopt", "closes_its_clients"]
