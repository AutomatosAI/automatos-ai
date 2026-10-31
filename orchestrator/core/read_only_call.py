"""PRD-268 P268W1-RVW-1 (§6.2.3, FR-1, K17): a check-in's call writes nothing, not even a side record.

A check-in runs only the actions labelled ``read`` (``check_in_lane``), so the label must
be true there. A few read actions keep a record of their own as they read, through the
caller's session, a session they open themselves, or the workspace's files:

- a search's log row (the code graph's ``codegraph_query_logs``, the documents'
  ``last_accessed``, the substrate tile's search events);
- the onboarding funnel's ``package_offered`` stamp (a package search);
- a spreadsheet's copy in the workspace (``platform_read_document``);
- the NL2SQL audit row, its training example, its primitive heartbeat and the use count
  of each few-shot example it reads (``platform_query_data``);
- a preview's picture in the session's folder (``platform_render_preview``).

In a check-in each is skipped, and everywhere else each runs as before. The check-in lane
marks its calls with ``reading_only()`` (``check_in_lane.carries_the_check_in``), and each
record asks ``only_reading()`` before it writes.

A ContextVar: the work a call starts inherits it (a task, ``asyncio.to_thread``, the
best-effort threads of ``core.best_effort``, which copy the caller's context), so a record
written after the call returns is skipped too. ``loop.run_in_executor`` does not copy it: a
record that hops there asks before the hop. Work a check-in call starts is the call's for
as long as it runs, so a long-lived worker is never started lazily inside one.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_READING_ONLY: ContextVar[bool] = ContextVar("prd268_reading_only", default=False)


def only_reading() -> bool:
    """Whether this call may only read: a check-in's call, which writes nothing."""
    return _READING_ONLY.get()


@contextmanager
def reading_only() -> Iterator[None]:
    """Mark the call made inside this block as one that may only read."""
    token = _READING_ONLY.set(True)
    try:
        yield
    finally:
        _READING_ONLY.reset(token)


__all__ = ["only_reading", "reading_only"]
