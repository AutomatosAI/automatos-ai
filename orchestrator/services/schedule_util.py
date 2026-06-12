"""
Schedule recurrence math — single source of truth (PRD-162 WS-7)
================================================================

ONE place that turns a recurrence (a heartbeat interval or a cron expression)
into a concrete `next_run`. Before PRD-162 this logic was smeared across three
ad-hoc parsers (heartbeat interval→cron in `heartbeat_service`, the manual cron
next-run + `_CRON_RE` validator in `scheduled_task_service`, and the inline
frequency formatting in `activity_service.get_schedule`). They are collapsed
here so the calendar, the scheduler, and the `platform_get_schedule` tool all
compute the same answer — statelessly, from the DB, identical on every worker.

`interval_to_cron` is deliberately croniter-free (pure string math) so it stays
importable and unit-testable without the optional `croniter` dependency; only
`next_run`/`is_valid_cron` reach for croniter, lazily.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Sentinels matching the legacy heartbeat thresholds (kept identical so the
# scheduler keeps firing at the same wall-clock times after the consolidation).
_WEEKLY_MINUTES = 10080
_DAILY_MINUTES = 1440
_DEFAULT_INTERVAL_MINUTES = 60
_MAX_ACTIVE_HOURS_SKIPS = 2048  # bound the active-hours search (≈ a week of 5-min ticks)


def interval_to_cron(minutes: int) -> str:
    """Convert a heartbeat interval (minutes) into a 5-field cron string.

    Mirrors the legacy ``HeartbeatService._interval_to_cron_trigger`` exactly so
    behaviour is preserved:

        15    → ``0,15,30,45 * * * *``
        30    → ``0,30 * * * *``
        60    → ``0 * * * *``
        120   → ``0 */2 * * *``
        480   → ``0 */8 * * *``
        1440  → ``0 9 * * *``
        10080 → ``0 9 * * 1``
    """
    if minutes <= 0:
        minutes = _DEFAULT_INTERVAL_MINUTES

    if minutes < 60:
        offsets = ",".join(str(o) for o in range(0, 60, minutes))
        return f"{offsets} * * * *"
    if minutes >= _WEEKLY_MINUTES:
        return "0 9 * * 1"  # weekly: Monday 09:00
    if minutes >= _DAILY_MINUTES:
        return "0 9 * * *"  # daily: 09:00
    hours = minutes // 60
    if hours == 1:
        return "0 * * * *"  # top of every hour
    return f"0 */{hours} * * *"


def is_valid_cron(cron_expression: str) -> bool:
    """True if ``cron_expression`` is a cron string croniter can evaluate."""
    if not cron_expression or not isinstance(cron_expression, str):
        return False
    try:
        from croniter import croniter
        return bool(croniter.is_valid(cron_expression))
    except Exception:
        return False


def next_run(
    cron_expression: str,
    *,
    now: Optional[datetime] = None,
    tz: str = "UTC",
    active_hours: Optional[Dict[str, Any]] = None,
) -> Optional[datetime]:
    """Next firing of ``cron_expression`` at/after ``now`` as a UTC-aware datetime.

    When ``active_hours`` is given, occurrences whose local time falls outside
    the window are skipped (so a paused/quiet window doesn't surface a fake
    "next run"). Returns ``None`` if the expression can't be evaluated.
    """
    base = now or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)

    try:
        from croniter import croniter
    except Exception:
        logger.warning("croniter unavailable — cannot compute next_run for %r", cron_expression)
        return None

    try:
        itr = croniter(cron_expression, base)
        candidate = itr.get_next(datetime)
        if active_hours:
            for _ in range(_MAX_ACTIVE_HOURS_SKIPS):
                if _within_active_hours(candidate, active_hours, tz):
                    break
                candidate = itr.get_next(datetime)
            else:
                return None  # window never matched within the bound
        if candidate.tzinfo is None:
            candidate = candidate.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc)
    except Exception as e:
        logger.warning("next_run failed for %r: %s", cron_expression, e)
        return None


def _within_active_hours(dt: datetime, active_hours: Dict[str, Any], tz: str) -> bool:
    """Whether ``dt`` (UTC-aware) falls inside the active-hours window.

    Lenient by design: a missing/malformed field means "no constraint on that
    axis", and a fully unparseable window means "always active" (never hide a
    configured schedule because its window metadata is junk — Q49).
    """
    try:
        local = _to_local(dt, tz)
    except Exception:
        return True

    days = active_hours.get("days")
    if isinstance(days, (list, tuple)) and days:
        if local.weekday() not in days:
            return False

    start = _parse_hhmm(active_hours.get("start"))
    end = _parse_hhmm(active_hours.get("end"))
    if start is not None and end is not None:
        minutes_of_day = local.hour * 60 + local.minute
        if start <= end:
            return start <= minutes_of_day < end
        return minutes_of_day >= start or minutes_of_day < end  # window crosses midnight
    return True


def _to_local(dt: datetime, tz: str) -> datetime:
    if not tz or tz == "UTC":
        return dt.astimezone(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return dt.astimezone(ZoneInfo(tz))
    except Exception:
        return dt.astimezone(timezone.utc)


def _parse_hhmm(value: Any) -> Optional[int]:
    """'09:30' → 570 (minutes since midnight); None if unparseable."""
    if not isinstance(value, str) or ":" not in value:
        return None
    try:
        h, m = value.split(":", 1)
        minutes = int(h) * 60 + int(m)
        return minutes if 0 <= minutes <= 1440 else None
    except Exception:
        return None
