"""Guarded fire-and-forget task launcher (PRD-142 Wave 1 · WS-C · W1-S5).

``asyncio.create_task`` is a footgun for fire-and-forget work. The event loop
keeps only a *weak* reference to the task, so if the caller doesn't hold a
strong reference the task can be garbage-collected — and silently cancelled —
mid-flight (see the asyncio docs' "Important" note on ``create_task``). And if
the coroutine raises, the exception surfaces only as a "Task exception was never
retrieved" warning at GC time, invisible to our telemetry.

``launch_guarded`` replaces bare ``create_task`` at every fire-and-forget site:

  - it keeps a strong reference in a module-level set until the task finishes,
    so the task can never be GC-cancelled;
  - it attaches a done-callback that retrieves the result and, on an *uncaught*
    exception, fires ``record_error`` so the crash lands on the
    ERRORS-by-subsystem dashboard tile instead of vanishing. Cancellation
    (graceful shutdown) is not reported as a failure.

This is the lightweight half of WS-C. It deliberately does NOT try to make
request-scoped closures (board task runs, wizard scrape pipelines) serializable
onto the Redis ``queued`` backend — that backend runs structured ``AgentTask``
descriptors, not arbitrary coroutines, and inventing a queue for them is the
risk the PRD warns against. Restart-survival for those surfaces is delivered by
the W1-S6 boot reaper, which marks any row stranded by a crash/restart terminal.
Where a true durable executor already exists (the workflow ``queued`` backend),
callers still prefer it; this launcher only replaces the in-process fallback.

F105: a plain ``def`` route runs in FastAPI's threadpool, where no event loop runs, so
``create_task`` there raised "no running event loop" after the route's work was
committed. ``on_the_event_loop`` creates the task on the loop's own thread instead;
``launch_guarded`` and the sync helpers that start loop work go through it.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Coroutine, Dict, Optional, Set, TypeVar
from uuid import UUID

from anyio import NoEventLoopError, from_thread

from core.utils.exception_telemetry import record_error

logger = logging.getLogger(__name__)

# Strong references to in-flight fire-and-forget tasks. Without this, the event
# loop holds only a weak reference and the GC can cancel a task mid-flight.
_BACKGROUND_TASKS: Set[asyncio.Task] = set()

T = TypeVar("T")


def on_the_event_loop(start: Callable[[], T]) -> T:
    """Call ``start`` where it can schedule work on the event loop, and return its answer:
    here when a loop runs in this thread, else on the loop's own thread when this is a
    route's threadpool thread (anyio's worker thread). With neither, a script or a plain
    test, it runs here, as it always did.

    F296 (night 8): the board's re-brief is a plain ``def`` route, so FastAPI runs it in its
    threadpool, where no loop runs. Re-briefing a playbook's card saved the new brief, then
    ``asyncio.create_task`` raised "no running event loop": the owner got a 500 for a change
    that had applied (#0249), and the run stayed pending until the stall watchdog failed it.
    F105 made 479 routes plain ``def``; every helper they reach that starts loop work comes
    through here. The worker waits only while ``start`` runs on the loop, so ``start`` does
    no I/O: it schedules, and the work it schedules runs on the loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return from_thread.run_sync(start)
        except NoEventLoopError:
            pass  # not a route's thread either: start it here
    return start()


def launch_guarded(
    coro: Coroutine[Any, Any, Any],
    *,
    subsystem: str,
    operation: str,
    workspace_id: Optional[UUID] = None,
    agent_id: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> asyncio.Task:
    """Launch ``coro`` as a guarded background task and return the Task.

    The task is strongly referenced until it completes (no GC-cancellation), and
    any *uncaught* exception is reported via ``record_error`` with the caller's
    ``subsystem``/``operation``. Cancellation is not reported as a failure.

    Called on the event loop or from a route's threadpool thread, the task runs on
    the loop (``on_the_event_loop``, F105). With no loop at all (a script, a plain
    test) it raises like ``create_task``.
    """
    return on_the_event_loop(functools.partial(
        _launch_here, coro, subsystem=subsystem, operation=operation,
        workspace_id=workspace_id, agent_id=agent_id, extra=extra,
    ))


def _launch_here(
    coro: Coroutine[Any, Any, Any],
    *,
    subsystem: str,
    operation: str,
    workspace_id: Optional[UUID],
    agent_id: Optional[int],
    extra: Optional[Dict[str, Any]],
) -> asyncio.Task:
    """``launch_guarded`` on the loop's thread: the task, held and watched."""
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)

    def _on_done(t: asyncio.Task) -> None:
        _BACKGROUND_TASKS.discard(t)
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return  # graceful cancellation is not a recordable failure
        if exc is None:
            return
        logger.error(
            "Guarded task %s.%s crashed: %s", subsystem, operation, exc, exc_info=exc
        )
        record_error(
            subsystem=subsystem,
            operation=operation,
            error=exc,
            workspace_id=workspace_id,
            agent_id=agent_id,
            extra=extra,
        )

    task.add_done_callback(_on_done)
    return task
