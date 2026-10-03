"""Auto's platform_update_playbook keeps the schedule rules the playbook routes keep
(F271, night 7b).

The owner saved a timer's schedule as ``{type: 'none'}`` to turn it off: nothing
changed, and the timer fired again. The playbook routes now refuse a type the
scheduler doesn't know, saying the two ways that turn a timer off
(``core/playbook_schedule``). Auto's update tool stored any schedule it was given, so
the same 'none' from Auto would have been saved as it is. It reads the same rules now,
and refuses with the same words before anything changes.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]


def checks_the_schedule(handler: Handler) -> Handler:
    """Refuse a ``schedule_config`` the playbook routes would refuse; anything else is the handler's."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.playbook_schedule import schedule_problem, unknown_type_refusal

        schedule = (params or {}).get("schedule_config")
        refusal = (unknown_type_refusal(schedule) or schedule_problem(schedule)) if schedule is not None else None
        if refusal:
            return {"success": False, "error": refusal}
        return await handler(db, workspace_id, params)
    return wrapped


__all__ = ["checks_the_schedule"]
