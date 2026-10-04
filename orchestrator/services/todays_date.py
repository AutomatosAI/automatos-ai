"""Today's date in the workspace's own zone, in words (F323, night 9b).

Night 9b ran on 4 October 2026. The Business Analyst read "this summer" as 2024
(#1984 run 1: "no data available for June–August 2024"), and Auto searched the web
for "current date" before answering a stock question (chat 32bb645f).

Every agent run's system prompt already carried the time (the context sections'
``datetime_context``: card runs, mission steps and playbook steps alike), but as
"Current UTC time: 2026-10-04T14:00Z", the lowest-priority section, the first the
budget drops; Auto's short chat path (``consumers/chatbot/atom_prompt``) carried none.

``today_line`` says it in words, in the workspace's zone: its heartbeat timezone, the
zone its playbook timers default to (``services.playbook_scheduler.default_schedule_zone``),
else UTC. It is to the hour, so the prompt prefix the provider caches stays the same
for an hour (F025).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

UTC_ZONE = "UTC"
TODAY = "Today is {weekday} {day} {month} {year}, {clock} in {zone}."


def workspace_zone(db: Any, workspace_id: Any) -> str:
    """The workspace's zone name, else UTC; UTC without a database session to read it."""
    from sqlalchemy.orm import Session
    from zoneinfo import ZoneInfo

    if not isinstance(db, Session) or not workspace_id:
        return UTC_ZONE
    from services.playbook_scheduler import default_schedule_zone

    try:
        zone = default_schedule_zone(db, workspace_id)
    except Exception:
        logger.exception("[today] could not read workspace %s's zone; the date is given in UTC", workspace_id)
        return UTC_ZONE
    if not isinstance(zone, str) or not zone.strip():
        return UTC_ZONE
    try:
        ZoneInfo(zone)
    except (ValueError, KeyError):      # zoneinfo raises KeyError subclasses for an unknown zone
        logger.warning("[today] workspace %s names an unknown zone %r; the date is given in UTC", workspace_id, zone)
        return UTC_ZONE
    return zone


def today_line(db: Any, workspace_id: Any, now: Optional[datetime] = None) -> str:
    """"Today is Sunday 4 October 2026, 15:00 in Europe/London.", to the hour."""
    from zoneinfo import ZoneInfo

    zone = workspace_zone(db, workspace_id)
    moment = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    local = moment.astimezone(ZoneInfo(zone))
    return TODAY.format(weekday=local.strftime("%A"), day=local.day, month=local.strftime("%B"),
                        year=local.year, clock=local.strftime("%H:%M"), zone=zone)


__all__ = ["TODAY", "UTC_ZONE", "today_line", "workspace_zone"]
