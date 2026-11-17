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

# The persona runner's patterns — plus a result record whose subtype names a limit,
# and GitHub Copilot's own words for a plan that ran out (PRD-253 S1.6): its AI
# credits ("You've run out of your AI credits", "…your included AI credits for the
# month") and its rate limits (overall, per model, per session, weekly — all say
# "rate limit"). Copilot's "No model available. Check policy enablement…" is NOT
# a limit: it is the organisation's policy, an error the operator has to fix.
_LIMIT = re.compile(
    r"hit\s+your\s+(?:[\w-]+\s+)?limit\b|usage[ _-]?limit|rate[ _-]?limit|Error:\s*(?:429|529)"
    r"|run out of (?:your )?(?:included )?AI credits|included AI credits for the month"
    r"|\"subtype\"\s*:\s*\"[^\"]*(?:error[^\"]*limit|rate_limit)",
    re.IGNORECASE,
)
_TRY_AGAIN = re.compile(r"try again in\s+(\d+)\s*(minute|min|hour|hr)", re.IGNORECASE)
# Claude Code's weekly limit names the day as well ("resets Nov 20, 9am (Europe/Berlin)").
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_RESETS = re.compile(
    r"resets?(?:\s+at)?\s+"
    r"(?:(?P<month>" + "|".join(_MONTHS) + r")[a-z]*\.?\s+(?P<day>\d{1,2}),?\s+(?:at\s+)?)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm)?(?:\s*\((?P<zone>[^)]+)\))?",
    re.IGNORECASE,
)
# A weekly limit resets within 7 days, so a stated date further ahead than 8 days is
# a misreading, not a plan window: the host re-checks instead of pausing for it.
MAX_STATED_RESET = timedelta(days=8)

# Claude Code does not always exit at its limit. It can park the turn on a usage
# limit line that says it will carry on by itself at the reset hour, and wait for
# the window to reopen, holding the ticket and a session slot for hours. The session
# watches its output for that line and ends at once (5 Nov 2026: two sessions sat
# parked for 4 h).
# It can also open its usage limit menu and wait there for a choice nobody makes:
# three tickets held both slots for about 20 h, each seen only at its 4 h timeout.
# The menu is read by its shape, never by its first option's sentence alone, which
# this file, the README and the tests quote: option one, numbered, directly before
# that sentence, then within ``_MENU_SPAN_CHARS`` the words of option two or three.
# Never the notice above the menu either: a notice can name the menu's command with
# no menu open.
_MENU_SPAN_CHARS = 200
_PARKED = re.compile(
    r"\blimit\b.{0,160}?continuing\s+automatically\s+at\s+\d"
    r"|\b1\.\s+stop\s+and\s+wait\s+for\s+limit\s+to\s+reset\b"
    rf".{{0,{_MENU_SPAN_CHARS}}}?(?:\b2\.\s+wait\s+here\b|\b3\.\s+switch\s+to\s+usage\s+credits\b)",
    re.IGNORECASE | re.DOTALL,
)
# A terminal's escape sequences (colour, cursor moves): read as a space.
_ESCAPES = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")

RECHECK_SECONDS = 15 * 60


def is_usage_limit(text: str) -> bool:
    return bool(text) and bool(_LIMIT.search(text) or is_parked(text))


def is_parked(text: str) -> bool:
    """The CLI is holding its turn until its usage window reopens."""
    return bool(text) and bool(_PARKED.search(_ESCAPES.sub(" ", text)))


def resets_at(text: str, now: datetime) -> Optional[datetime]:
    """When the CLI says its window reopens, or ``None`` when it does not say.

    Reads "try again in N minutes/hours", "resets 3pm (Europe/Dublin)" /
    "reset at 15:00" and a weekly limit's "resets Nov 20, 9am (Europe/Berlin)".
    ``now`` must be timezone-aware; a clock time with no zone is read in ``now``'s
    zone, and a time already past today means tomorrow. The terminal's escape
    sequences are read as spaces. The first phrase that states a clock time
    decides: when it cannot be a real moment the answer is ``None``, so the host
    re-checks instead of trusting a later, weaker match.
    """
    flat = _ESCAPES.sub(" ", text or "")
    m = _TRY_AGAIN.search(flat)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        return now + (timedelta(hours=amount) if unit.startswith("h") else timedelta(minutes=amount))
    for m in _RESETS.finditer(flat):
        # A bare number is no clock time: the line after the menu option that waits
        # for the reset starts with the next option's number, which is not an hour.
        if m.group("minute") or m.group("meridiem"):
            return _stated_reset(m, now)
    return None


def _stated_reset(m: "re.Match[str]", now: datetime) -> Optional[datetime]:
    """The moment one "resets ..." phrase states, or ``None`` when it cannot be one."""
    clock = _clock(int(m.group("hour")), int(m.group("minute") or 0), m.group("meridiem"))
    if clock is None:
        return None
    hour, minute = clock
    local_now = now.astimezone(_zone(m.group("zone")) or now.tzinfo)
    if m.group("month"):
        month = _MONTHS.index(m.group("month").lower()) + 1
        return _dated(local_now, month, int(m.group("day")), hour, minute)
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return candidate if candidate > local_now else candidate + timedelta(days=1)


def _clock(hour: int, minute: int, meridiem: Optional[str]) -> Optional[Tuple[int, int]]:
    """A 12- or 24-hour clock reading as ``(hour, minute)`` on 24 hours, or ``None``."""
    half = (meridiem or "").lower()
    if half == "pm" and hour < 12:
        hour += 12
    elif half == "am" and hour == 12:
        hour = 0
    return (hour, minute) if hour <= 23 and minute <= 59 else None


def _dated(local_now: datetime, month: int, day: int, hour: int, minute: int) -> Optional[datetime]:
    """A stated date and time: this year, or next year when it has passed (a
    December reading of "Jan 2"). ``None`` for a date that does not exist or that
    lies further ahead than ``MAX_STATED_RESET``: a date that has just passed would
    otherwise pause the host for a year."""
    for year in (local_now.year, local_now.year + 1):
        try:
            candidate = local_now.replace(year=year, month=month, day=day, hour=hour, minute=minute,
                                          second=0, microsecond=0)
        except ValueError:
            return None
        if candidate > local_now:
            return candidate if candidate - local_now <= MAX_STATED_RESET else None
    return None


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
