"""P256-FIX-N17-9 (night 17, F448) — a slow document read answers before the session's client gives up.

WRITER's two ``search_documents`` calls on #1756 (19:56:58Z, 19:58:58Z) ended in "The
operation timed out.", which is not in this repository: it is the session's MCP client
giving up. Every document search shares one small executor with Auto's own chat retrieval
(``modules/rag/service.py``, ``DOCUMENT_SEARCH_THREADS``), behind the session tool slots,
with no deadline of ours; the calls that did succeed took 38-50 s. The session read a
failure it could not act on, and never learned it could read the file by name.

A call to one of ``DEADLINED_TOOLS`` now has ``SESSION_TOOL_READ_DEADLINE_S`` from the
moment it ARRIVED (``api/session_tools.require_session`` stamps it, so the token lookup,
the allowance and the wait for a slot all count), and answers in words when it runs out:
what happened, and what to do instead. This makes the failure clear; it does not make
search faster. The executor's thread is not cancellable and finishes in the background, and
the sizing of that shared executor is its own question (Gerard's). Every other tool, a write
above all, runs as before and is never cancelled.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from config import config

logger = logging.getLogger(__name__)

# The reads that may take long under load, and only reads: a write is never cut off mid-way.
DEADLINED_TOOLS = ("search_documents", "search_knowledge", "read_document")
LATE_WORDS = {
    "search_documents": ("Document search did not answer in time (the server is busy). To read a file your brief "
                         "names, call read_document with its filename; otherwise try once more, then say in your "
                         "result that search was unavailable."),
    "search_knowledge": ("Memory search did not answer in time (the server is busy). Try once more, then say in "
                         "your result that memory search was unavailable."),
    "read_document": "The document read did not answer in time; try once more, then say so in your result.",
}
ARRIVED_AT = "session_call_at"   # request.state.<this>: the loop's clock when the call arrived


def stamp_arrival(request: Any) -> None:
    """Note on the request when the call arrived, before anything else runs for it."""
    state = getattr(request, "state", None)
    if state is not None:
        setattr(state, ARRIVED_AT, asyncio.get_running_loop().time())


def arrived_at(request: Any) -> Optional[float]:
    """When the call arrived (``stamp_arrival``), or None when it was not stamped."""
    return getattr(getattr(request, "state", None), ARRIVED_AT, None)


def _late(tool_name: str) -> Dict[str, Any]:
    return {"success": False, "error": LATE_WORDS[tool_name]}


async def within_read_deadline(tool_name: str, arrived: Optional[float],
                               run: Callable[[], Awaitable[Dict[str, Any]]]) -> Dict[str, Any]:
    """``run()``, cut off for a deadlined read once the budget from ``arrived`` is spent."""
    if tool_name not in DEADLINED_TOOLS:
        return await run()
    loop = asyncio.get_running_loop()
    budget = float(getattr(config, "SESSION_TOOL_READ_DEADLINE_S", 0) or 0)
    remaining = budget - (loop.time() - (arrived if arrived is not None else loop.time()))
    if remaining <= 0:
        logger.warning("[F448] %s: no time left of its %.0f s when it reached its slot", tool_name, budget)
        return _late(tool_name)
    try:
        async with asyncio.timeout(remaining):
            return await run()
    except TimeoutError:
        logger.warning("[F448] %s did not answer within %.0f s of arriving; answered in words", tool_name, budget)
        return _late(tool_name)
