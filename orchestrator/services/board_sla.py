"""Priority → SLA deadline: the ONE table every board-ticket writer uses.

It was duplicated in ``api.board_tasks`` and ``services.orchestration_board_bridge``;
the scheduled-task lane (a ticket filed when a calendar entry fires) made a third
copy inevitable, so the table lives here and the writers import it.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

PRIORITY_SLA_HOURS: dict[str, int] = {
    "urgent": 4,
    "high": 12,
    "medium": 24,
    "low": 72,
}
DEFAULT_SLA_HOURS = 24


def sla_deadline_for(priority: Optional[str], *, now: Optional[datetime] = None) -> datetime:
    """The deadline a ticket of ``priority`` gets when filed at ``now`` (UTC)."""
    base = now or datetime.now(timezone.utc)
    return base + timedelta(hours=PRIORITY_SLA_HOURS.get(priority or "", DEFAULT_SLA_HOURS))
