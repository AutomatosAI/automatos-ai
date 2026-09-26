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
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, TypeVar

from sqlalchemy.orm import Session

T = TypeVar("T")

_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


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


async def composio_lookup(lookup: Callable[[Session], T]) -> T:
    """``lookup(db)`` on the Composio lookup threads, with a session of its own
    and the caller's context carried along. Raises what ``lookup`` raises."""
    call = functools.partial(contextvars.copy_context().run, _in_own_session, lookup)
    return await asyncio.get_running_loop().run_in_executor(_lookup_executor(), call)
