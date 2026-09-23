"""A CLI's usage limit is a pause, not a failure (F083).

When Claude Code (or another CLI) runs out of its plan's usage window it exits
mid-turn, and the host reported that as an error: every in-flight ticket failed,
and every later claim failed on spawn until the window reopened. The patterns are
the persona runner's (``scripts/ralph/overnight-customer.sh``
``is_usage_limit_error``); the reset time is read when the CLI prints one, and
otherwise the host checks again after ``RECHECK_SECONDS``.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

# The persona runner's patterns — plus a result record whose subtype names a limit.
_LIMIT = re.compile(
    r"hit your limit|usage[ _-]?limit|rate[ _-]?limit|Error:\s*(?:429|529)"
    r"|\"subtype\"\s*:\s*\"[^\"]*(?:error[^\"]*limit|rate_limit)",
    re.IGNORECASE,
)
_TRY_AGAIN = re.compile(r"try again in\s+(\d+)\s*(minute|min|hour|hr)", re.IGNORECASE)
_RESETS = re.compile(
    r"resets?(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s*\(([^)]+)\))?", re.IGNORECASE
)

RECHECK_SECONDS = 15 * 60


def is_usage_limit(text: str) -> bool:
    return bool(text) and bool(_LIMIT.search(text))


def resets_at(text: str, now: datetime) -> Optional[datetime]:
    """When the CLI says its window reopens, or ``None`` when it does not say.

    Reads "try again in N minutes/hours" and "resets 3pm (Europe/Dublin)" /
    "reset at 15:00". ``now`` must be timezone-aware; a clock time with no zone
    is read in ``now``'s zone, and a time already past today means tomorrow.
    """
    text = text or ""
    m = _TRY_AGAIN.search(text)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        return now + (timedelta(hours=amount) if unit.startswith("h") else timedelta(minutes=amount))
    m = _RESETS.search(text)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    meridiem = (m.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    local_now = now.astimezone(_zone(m.group(4)) or now.tzinfo)
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return candidate if candidate > local_now else candidate + timedelta(days=1)


def pause(text: str, now: datetime) -> Tuple[datetime, bool]:
    """``(until, known)``: the CLI's own reset time, or a re-check in RECHECK_SECONDS."""
    until = resets_at(text, now)
    return (until, True) if until else (now + timedelta(seconds=RECHECK_SECONDS), False)


def describe(cli: str, until: datetime, known: bool) -> str:
    """The one honest line for the ticket and the host — never "check your key"."""
    at = until.astimezone().strftime("%H:%M")
    if known:
        return f"paused: {cli} usage limit, resumes ~{at}"
    return f"paused: {cli} usage limit — checking again at ~{at}"


def _zone(name: Optional[str]):
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name.strip())
    except Exception:  # noqa: BLE001 — an unknown zone name falls back to local time
        return None
