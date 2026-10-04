"""Cron, in the owner's words (F290, night 8).

Auto's own messages named a cron string and a timezone ("0 9 * * 1-5",
"Europe/London") where the owner would say "weekdays at 09:00 (Europe/London)".
A timer tool should talk the way the owner does, not in cron fields.

Minute and hour are read only when they are an exact number (no "*/5" or lists);
the weekday field is read the way the scheduler reads it
(``services.schedule_util``, which already turns "1-5" into "mon,tue,wed,thu,fri"
the same way croniter does). A day-of-month or month field that is not "*", or a
weekday pattern that is too irregular to name, has no plain form here: the cron
itself is shown instead of guessing.
"""
from __future__ import annotations

from typing import Optional

_WEEKDAYS = frozenset({"mon", "tue", "wed", "thu", "fri"})
_WEEKEND = frozenset({"sat", "sun"})
_ALL_DAYS = _WEEKDAYS | _WEEKEND
_DAY_NAME = {"sun": "Sundays", "mon": "Mondays", "tue": "Tuesdays", "wed": "Wednesdays",
             "thu": "Thursdays", "fri": "Fridays", "sat": "Saturdays"}
_DAY_ORDER = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")


def plain_cron(cron_expression: Optional[str], timezone: Optional[str]) -> str:
    """"weekdays at 09:00 (Europe/London)", or the cron itself in parentheses
    with its zone when the fields are too irregular to say in a sentence."""
    zone = timezone or "UTC"
    said = _plain_days_and_time(cron_expression) if cron_expression else None
    return f"{said} ({zone})" if said else f"cron '{cron_expression}' ({zone})"


def _plain_days_and_time(cron_expression: str) -> Optional[str]:
    """"<days> at HH:MM", or None when a field is too irregular to say plainly."""
    fields = cron_expression.strip().split()
    if len(fields) != 5:
        return None
    minute, hour, day, month, weekday = fields
    if day != "*" or month != "*" or not minute.isdigit() or not hour.isdigit():
        return None
    days = _plain_days(weekday)
    return f"{days} at {int(hour):02d}:{int(minute):02d}" if days else None


def _plain_days(weekday: str) -> Optional[str]:
    """"weekdays", "weekends", "daily" or "Mondays", read the way the scheduler
    reads the field; None for a pattern (nth/last weekday) it cannot expand."""
    from services.schedule_util import _named_weekdays

    try:
        named = _named_weekdays(weekday)
    except ValueError:
        return None
    if named == "*":
        return "daily"
    names = frozenset(named.split(","))
    if names == _ALL_DAYS:
        return "daily"
    if names == _WEEKDAYS:
        return "weekdays"
    if names == _WEEKEND:
        return "weekends"
    return " and ".join(_DAY_NAME[d] for d in sorted(names, key=_DAY_ORDER.index))


CRON_FIELDS = 5
NEEDS_A_TIME = ("A new timer needs a time: send cron_expression, written from the owner's words (every weekday "
                "at 9 is \"0 9 * * 1-5\"). To switch a timer off or back on, send enabled alone.")


def cron_refusal(cron_expression: Optional[str]) -> Optional[dict]:
    """Why a new timer's cron can't be set (none given, or not five fields), or None.
    F290 (night 8): switching a timer off or on needs no cron, so the schema doesn't
    require one; only a new timer does, and this says so."""
    if not cron_expression:
        return {"success": False, "error": NEEDS_A_TIME}
    parts = cron_expression.strip().split()
    if len(parts) != CRON_FIELDS:
        return {"success": False, "error": f"Invalid cron expression: expected 5 fields, got {len(parts)}. "
                                           "Format: minute hour day_of_month month day_of_week"}
    return None


__all__ = ["NEEDS_A_TIME", "cron_refusal", "plain_cron"]
