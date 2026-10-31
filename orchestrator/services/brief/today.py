"""PRD-268 §6.11.2 — the brief's "Today": the activity schedule's items on the person's local date.

The schedule is the Command Center calendar's own read (``ActivityService.get_schedule``,
a one-day range from now), kept to the items whose next run falls on the local date in the
person's zone. Each row says what kind of thing runs, in the person's language, at its time.
A Mission's deadline opens the Mission and a ticket's deadline the ticket; a routine, a
scheduled Playbook, a task or a social post has nowhere to go, so it has no ``opens``. No
calendar row (§12-Q20).
"""
from __future__ import annotations

from datetime import date, datetime, timezone, tzinfo
from types import MappingProxyType
from typing import Any, List, Mapping, Optional, Tuple

from sqlalchemy.orm import Session

from services.activity_service import ActivityService
from services.brief.model import BriefOpens, BriefRow
from services.brief.rows import iso, opens, title_of
from services.brief.strings import say

__all__ = ["SCHEDULE_DAYS", "today_rows"]

SCHEDULE_DAYS = 1
# The schedule's item types (services/activity_service.py and its item modules) to the
# strings key of their reason; any other type reads ``today_other``.
_WHY: Mapping[str, str] = MappingProxyType({
    "routine": "today_routine",
    "recipe": "today_playbook",
    "task": "today_task",
    "mission": "today_mission",
    "task_due": "today_task_due",
    "social": "today_social",
})
_OTHER = "today_other"


def today_rows(
    db: Session, workspace_id: Any, local_date: date, zone: tzinfo, language: str,
) -> Tuple[int, List[BriefRow]]:
    """How many scheduled items fall on ``local_date`` in ``zone``, and each as a brief row."""
    scheduled = ActivityService(db, workspace_id).get_schedule(range_days=SCHEDULE_DAYS).get("scheduled") or []
    on_the_day = [item for item in scheduled if _local_date(item.get("next_run_at"), zone) == local_date]
    return len(on_the_day), [_row(item, language) for item in on_the_day]


def _local_date(moment: Any, zone: tzinfo) -> Optional[date]:
    """The date an ISO 8601 time falls on in ``zone`` (a time without one is UTC); None when unreadable."""
    try:
        parsed = datetime.fromisoformat(str(moment))
    except (TypeError, ValueError):
        return None
    aware = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return aware.astimezone(zone).date()


def _opens(item: Mapping[str, Any]) -> Optional[BriefOpens]:
    if item.get("type") == "mission":
        return opens("mission", item.get("mission_id"))
    if item.get("type") == "task_due":
        return opens("ticket", item.get("board_task_id"))
    return None


def _row(item: Mapping[str, Any], language: str) -> BriefRow:
    return BriefRow(
        id=f"schedule:{item.get('id')}",
        title=title_of(item.get("name"), language),
        why=say(language, _WHY.get(str(item.get("type")), _OTHER)),
        at=iso(item.get("next_run_at")),
        opens=_opens(item),
    )
