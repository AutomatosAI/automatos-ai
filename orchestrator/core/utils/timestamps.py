"""Timestamps that SAY they are UTC when they cross the API boundary.

Most naive ``DateTime`` columns in this schema (``timestamp without time zone``)
are written from Postgres ``now()`` or ``datetime.utcnow()`` on a UTC server, so
the stored value IS UTC — but it carries no offset, and plain ``.isoformat()``
emits none either. ECMAScript parses a date-time string with NO offset as LOCAL
time, so ``new Date(...)`` in a UK browser reads such a value an hour early and
any elapsed-time UI ("2h ago") is wrong by the viewer's own UTC offset.
Date-only renderings hide the skew, which is why it went unnoticed for so long.

Stamp the offset once, here, at the boundary — for every reader.

Extracted 2026-09-11 from ``api/admin_workspaces._utc_iso``; six other sites
still hand-roll ``replace(tzinfo=timezone.utc)`` and can move here over time.

``month_window_utc`` is the calendar month a monthly allowance counts in: the
Auto Live voice minutes (``modules/voice/voice_meter.py``) and the Socials
render minutes (``core/media_render_quota.py``). Moved here from the voice
meter (PRD-251 S1.1c) so both meters read the same month.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple


def utc_iso(value: Optional[datetime]) -> Optional[str]:
    """A naive-UTC (or already-aware) timestamp as an explicit-offset ISO string."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def month_window_utc(now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """Pure: the [start, next-start) UTC window of the calendar month."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        nxt = start.replace(year=start.year + 1, month=1)
    else:
        nxt = start.replace(month=start.month + 1)
    return start, nxt
