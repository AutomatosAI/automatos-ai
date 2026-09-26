"""F105 (night 3): a best-effort write never freezes the event loop.

Usage rows, tool telemetry, tool-gap rows, heartbeat findings and the NL2SQL
audit were written with sync SQLAlchemy on the event loop. When the connection
pool ran dry (night 3: 10 + 20 connections, 30 s wait), each such write froze
the whole process for up to 30 s — every request's timers expired together,
and unrelated turns' embedding, decision and memory calls "timed out" in the
same instant (8 within 47 ms at 23:53:57Z).

These writers now run on a few threads of their own and carry the caller's
context (workspace, usage scope) with them. A sync writer called on an event
loop hands the write over and returns at once (``off_loop``); an async writer
awaits its write without holding the loop (``awaitable_off_loop``). A stalled
write costs its own thread, never the loop — and not for long: a handed-over
write waits at most ``BEST_EFFORT_POOL_WAIT_S`` for room in the connection pool,
then is dropped with a log line instead of queueing behind requests for the
pool's own 30 s. (The room check and the checkout are not atomic: a request
that takes the last connection in between still makes the write wait the
pool's bound — on its own thread.) With no loop running in the thread
(scripts, sync tests, worker threads) a sync writer runs inline as before. The
writers log their own failures and never raise into the request.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional, Set

from config import config

logger = logging.getLogger(__name__)

_POOL_POLL_S = 0.05

_lock = threading.Lock()
# A write's future is told done before its callback (_finished) runs; drain()
# waits on this for the callback, so a failed write's log line is in by then.
_settled = threading.Condition(_lock)
_executor: Optional[ThreadPoolExecutor] = None
_pending: Set[Future] = set()


def _pool_has_room() -> bool:
    """A connection is idle, or one more may be opened. Unreadable → True."""
    try:
        from core.database import database

        pool = database.engine.pool
        return pool.checkedin() > 0 or pool.overflow() < pool._max_overflow
    except Exception:  # noqa: BLE001 — a faked or foreign pool: just write
        return True


def _within_pool_bound(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    deadline = time.monotonic() + config.BEST_EFFORT_POOL_WAIT_S
    while not _pool_has_room():
        if time.monotonic() >= deadline:
            logger.warning("best-effort write dropped: no connection free within %gs (%s)",
                           config.BEST_EFFORT_POOL_WAIT_S, getattr(fn, "__qualname__", fn))
            return None
        time.sleep(_POOL_POLL_S)
    return fn(*args, **kwargs)


def _submit(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    global _executor
    context = contextvars.copy_context()
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=config.BEST_EFFORT_WRITE_THREADS,
                                           thread_name_prefix="best-effort-write")
        future = _executor.submit(context.run, functools.partial(_within_pool_bound, fn, *args, **kwargs))
        _pending.add(future)
    future.add_done_callback(_finished)
    return future


def _finished(future: Future) -> None:
    if not future.cancelled() and future.exception() is not None:
        logger.warning("best-effort write failed: %r", future.exception())
    with _settled:
        _pending.discard(future)
        _settled.notify_all()


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def off_loop(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A sync writer that, called on an event loop, runs on the best-effort
    threads and returns None at once; called anywhere else, runs inline."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _loop_running():
            return fn(*args, **kwargs)
        _submit(fn, *args, **kwargs)
        return None
    return wrapper


def awaitable_off_loop(fn: Callable[..., Any]) -> Callable[..., Any]:
    """An async writer whose sync body runs on the best-effort threads."""
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.wrap_future(_submit(fn, *args, **kwargs))
    return wrapper


def drain(timeout: float = 5.0) -> bool:
    """Wait for the writes handed over so far, their failure log lines included
    (tests, shutdown); True if all finished."""
    deadline = time.monotonic() + timeout
    with _settled:
        handed_over = set(_pending)
        while handed_over & _pending:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            _settled.wait(left)
    return True
