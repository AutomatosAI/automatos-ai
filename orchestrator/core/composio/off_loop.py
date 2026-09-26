"""F105 (26 Sep): a Composio lookup never freezes the event loop.

Before its model call, a chat turn, a playbook step and an agent run each look
up the Composio actions to offer: ``ComposioToolService.get_tools_for_step``
(an SDK search, 0.4-0.75 s) and, when that finds nothing,
``ComposioHintService.build_hints`` (one SDK ``tools.get`` per matched app,
1.4-5.6 s). Both are sync and ran on the event loop, so the whole backend stood
still for each: 16 of the 21 scheduler slips between 01:41 and 01:57Z on 26 Sep,
and the /health probe of the refresh-4 retest waited 3.87 s behind one.

The lookups run on threads of their own instead and carry the caller's context
(workspace, request, widget surface). Each gets a session of its own: the
caller's session belongs to the caller's thread, which goes on serving other
work while the lookup runs. A pool of their own, so slow lookups never hold up
other work handed to the default executor, and at most COMPOSIO_LOOKUP_THREADS
of them hold a pool connection at once.

A turn waits COMPOSIO_LOOKUP_TIMEOUT_SECONDS at most: then the lookup raises
ComposioLookupTimeout, with a WARNING naming the step and the app(s) it was
asking about (the services name them with ``note_apps`` before each SDK call),
and the turn goes on without Composio tools. The SDK's lookup calls give up
after as long themselves (core.composio.client), so the thread comes free too.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional, TypeVar

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

T = TypeVar("T")

_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_asked_apps: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "composio_lookup_asked_apps", default=None,
)


class ComposioLookupTimeout(TimeoutError):
    """A Composio lookup outlasted COMPOSIO_LOOKUP_TIMEOUT_SECONDS (already
    logged as a WARNING); the turn goes on without Composio tools."""


def note_apps(apps: Iterable[str]) -> None:
    """The app(s) the running lookup is about to ask the SDK about, for the
    timeout WARNING. A no-op outside a lookup."""
    asked = _asked_apps.get()
    if asked is not None:
        asked[:] = [str(app) for app in apps]


def _lookup_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            from config import config

            _executor = ThreadPoolExecutor(
                max_workers=config.COMPOSIO_LOOKUP_THREADS,
                thread_name_prefix="composio-lookup",
            )
        return _executor


def _in_own_session(lookup: Callable[[Session], T]) -> T:
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        return lookup(db)
    finally:
        db.close()


async def composio_lookup(lookup: Callable[[Session], T], *, step: str) -> T:
    """``lookup(db)`` on the Composio lookup threads, with a session of its own
    and the caller's context carried along. Raises what ``lookup`` raises, or
    ComposioLookupTimeout after COMPOSIO_LOOKUP_TIMEOUT_SECONDS; ``step`` names
    the lookup in that WARNING (e.g. "chat turn (agent 1): tool search")."""
    from config import config

    asked: List[str] = []
    context = contextvars.copy_context()
    context.run(_asked_apps.set, asked)
    call = functools.partial(context.run, _in_own_session, lookup)
    waited = config.COMPOSIO_LOOKUP_TIMEOUT_SECONDS
    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(_lookup_executor(), call), timeout=waited,
        )
    except asyncio.TimeoutError:
        apps = ", ".join(asked) or "before naming an app"
        logger.warning(
            "[composio-lookup] %s gave up after %gs waiting on the Composio SDK (%s); "
            "this turn goes on without Composio tools",
            step, waited, apps,
        )
        raise ComposioLookupTimeout(f"{step}: no answer from Composio within {waited:g}s ({apps})") from None
