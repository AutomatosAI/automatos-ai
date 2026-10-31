"""PRD-266 US-012a (K10) — run an after-commit notice without holding up the caller.

``send_soon`` keeps the ``modules/socials/notify._send_soon`` contract: on the running loop as a
task held in ``_PENDING`` until it is done (the loop keeps only a weak reference to a task), or
``asyncio.run`` inline when no loop runs (a threadpool route, the leader's tick, a script). It
never raises, and closes the coroutine when it cannot run it. The socials module keeps its own
copy: its tests read ``notify._PENDING``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, Set

logger = logging.getLogger(__name__)

# Scheduled notices, held until done so the loop does not drop them.
_PENDING: Set["asyncio.Task[Any]"] = set()


def send_soon(coro: Coroutine[Any, Any, Any], *, what: str) -> None:
    """Run ``coro`` on the running loop as a tracked task, or inline when none runs. Never raises."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return
        task = loop.create_task(coro)
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)
    except Exception:
        coro.close()
        logger.exception("[notification_events] the %s notice could not be sent", what)
