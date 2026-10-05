"""F339: a session agent's prepared mission step carries where the mission was started.

F155: ``CoordinatorService._task_io`` refuses a widget-born mission's step for a session
(CLI) agent, because the widget key's restrictions cannot reach a Claude Code session.
It reads the prepared task's ``origin``. ``_prepare_task``'s session-agent branch never
set it, so the check saw ``None`` and the step ran on a session. This wraps
``_prepare_task`` (called as ``(self, db, run, task, agent_id)``) so the session branch's
prepared task carries the run's config as ``origin``, as the API branch's does.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Optional

Prepare = Callable[..., Awaitable[Optional[Dict[str, Any]]]]


def a_session_step_carries_its_origin(prepare: Prepare) -> Prepare:
    """Add ``origin`` (the run's config) to a session step's prepared task that lacks one."""
    @functools.wraps(prepare)
    async def wrapped(self: Any, db: Any, run: Any, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        prepared = await prepare(self, db, run, *args, **kwargs)
        if not prepared or not prepared.get("cli_agent") or "origin" in prepared:
            return prepared
        return {**prepared, "origin": dict(getattr(run, "config", None) or {})}
    return wrapped


__all__ = ["a_session_step_carries_its_origin"]
