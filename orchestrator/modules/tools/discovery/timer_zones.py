"""The zone a timer Auto sets keeps (F266, night 7b).

The owner, in Bristol, asked for a playbook timer "every day at 20:25". Auto passed
timezone "UTC" (the tool's description already said never to assume UTC), so the timer
was set an hour off; the owner's other timers are on UK time. Asked twice to change it,
Auto sent parameters the tool doesn't take.

An explicit UTC the owner never said is Auto's assumption. The timer takes the
workspace's own zone instead: its heartbeat zone, else the zone its other timers use
(``services.playbook_scheduler.default_schedule_zone``). The owner's words are this
chat's latest messages, found through the turn's usage scope (``chat:<id>``); a call
from no chat has no owner's words, so a UTC there is an assumption too. The answer says
which zone was kept.
"""
from __future__ import annotations

import functools
import re
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

UTC_NAMES = frozenset({"utc", "etc/utc", "gmt", "etc/gmt", "z", "zulu", "universal", "etc/universal",
                       "utc+0", "gmt+0", "utc+00:00", "+00:00"})
_SAID_UTC = re.compile(r"\b(?:utc|gmt|zulu)\b", re.IGNORECASE)
CHAT_SCOPE = "chat:"
ZONE_KEPT = (" The call asked for UTC, which the owner never said, so the timer is set in {zone}, the "
             "workspace's own zone. Tell the owner the time in {zone}.")


def keeps_the_owners_zone(handler: Handler) -> Handler:
    """A UTC the owner never said becomes the workspace's own zone (see the module)."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        said = str((params or {}).get("timezone") or "").strip().lower()
        zone = _workspace_zone(db, workspace_id) if said in UTC_NAMES else None
        if zone is None or owner_said_utc(db, workspace_id):
            return await handler(db, workspace_id, params)
        out = await handler(db, workspace_id, {**params, "timezone": zone})
        if not (isinstance(out, dict) and out.get("success")):
            return out
        return {**out, "timezone_kept": zone, "message": f"{out.get('message', '')}{ZONE_KEPT.format(zone=zone)}"}
    return wrapped


def _workspace_zone(db: Session, workspace_id: Any) -> Optional[str]:
    """The workspace's own zone, when it is not UTC."""
    from services.playbook_scheduler import default_schedule_zone

    zone = default_schedule_zone(db, workspace_id)
    return None if str(zone).strip().lower() in UTC_NAMES else zone


def owner_said_utc(db: Session, workspace_id: Any) -> bool:
    """Whether the owner's latest words in this turn's chat name UTC (or GMT)."""
    from core.llm.usage_context import LANE_CHAT, current_usage_scope
    from modules.tools.discovery.handlers_board_task_review import owner_words

    scope = current_usage_scope()
    where = str(scope.get("execution_id") or "")
    if scope.get("request_type") != LANE_CHAT or not where.startswith(CHAT_SCOPE):
        return False
    try:
        chat_id = UUID(where[len(CHAT_SCOPE):])
    except ValueError:
        return False
    return any(_SAID_UTC.search(words or "") for words in owner_words(db, workspace_id, chat_id))


__all__ = ["UTC_NAMES", "ZONE_KEPT", "keeps_the_owners_zone", "owner_said_utc"]
