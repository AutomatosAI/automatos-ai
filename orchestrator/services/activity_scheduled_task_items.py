"""PRD-266 D27 (US-018a): agent-scheduled tasks in the Command Center calendar.

The third source of ``ActivityService.get_schedule`` (``services/activity_service.py``,
over 800 lines, delegates to it, as it does to ``services/activity_social_items.py``).
Each ``agent_scheduled_tasks`` row of the caller's workspace that is active OR paused,
with its next slot inside the window, is one item, ``task-<id>``. A paused task stays
in the feed with ``paused: true`` instead of vanishing, so the calendar can grey it and
offer Resume (``POST /api/v1/scheduled-tasks/{id}/resume``); a cancelled task never shows.
A recurring task shows at its next cron slot (where it would run), a one-off at its
``next_run_at``. One query per call, whatever the row count.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from services.schedule_util import is_valid_cron, next_run
from services.scheduled_task_steps import DELIVER_FOLLOW_UP

ITEM_TYPE = "task"
ACTIVE = "active"
PAUSED = "paused"
NAME_MAX_CHARS = 80
DEFAULT_NAME = "Scheduled task"
DEFAULT_DELIVERY = "chat"
ONE_OFF = "One-off"
TASK_TIMEZONE = "UTC"

# Active and paused rows only: a cancelled task is gone from the calendar. PRD-268 K8: a
# follow-up is its person's own, never on the workspace's calendar.
_TASKS_SQL = text(
    """
    SELECT t.id, t.task_type, t.description, t.schedule, t.next_run_at, t.status,
           t.deliver_as, t.payload->>'title' AS payload_title,
           ta.name AS target_name
    FROM agent_scheduled_tasks t
    LEFT JOIN agents ta ON ta.id = t.target_agent_id
    WHERE t.workspace_id = :ws AND t.status IN (:active, :paused)
      AND t.deliver_as <> :follow_up
    """
)


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _name(row: Any) -> str:
    # A board-delivery row is the ticket it will file: name it by the title.
    label = getattr(row, "payload_title", None) or row.description or row.task_type or DEFAULT_NAME
    return label.strip()[:NAME_MAX_CHARS]


def _item(row: Any, cron: Optional[str], slot: datetime) -> Dict[str, Any]:
    return {
        "id": f"task-{row.id}",
        "scheduled_task_id": row.id,
        "task_type": row.task_type,
        "deliver_as": getattr(row, "deliver_as", None) or DEFAULT_DELIVERY,
        "name": _name(row),
        "type": ITEM_TYPE,
        "next_run_at": slot.isoformat(),
        "frequency": cron or ONE_OFF,
        "agent_name": row.target_name,
        "agent_id": None,
        "paused": getattr(row, "status", None) == PAUSED,
        "recurrence": {
            "cron_expression": cron,
            "interval_minutes": None,
            "timezone": TASK_TIMEZONE,
            "active_hours": None,
        },
    }


def scheduled_task_items(db: Session, workspace_id: UUID, now: datetime, horizon: datetime) -> List[Dict[str, Any]]:
    """The workspace's active and paused scheduled tasks whose next slot is no later
    than ``horizon``, as a new list of calendar items (one ``execute()``)."""
    rows = db.execute(
        _TASKS_SQL, {"ws": str(workspace_id), "active": ACTIVE, "paused": PAUSED, "follow_up": DELIVER_FOLLOW_UP},
    ).fetchall()
    items: List[Dict[str, Any]] = []
    for row in rows:
        cron = row.schedule if is_valid_cron(row.schedule) else None
        slot = next_run(cron, now=now) if cron else _as_utc(row.next_run_at)
        if slot is not None and slot <= horizon:
            items.append(_item(row, cron, slot))
    return items
