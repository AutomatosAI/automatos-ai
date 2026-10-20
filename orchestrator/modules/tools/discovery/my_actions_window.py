"""P256-FIX-N14-3 (night 14, F417): the window ``platform_my_actions`` reads, in the owner's words.

"What have you done tonight?" at 04:43 means since yesterday evening; "today" since midnight;
"yesterday" the whole of yesterday. Times are UTC: the platform keeps no workspace time zone.
A ``since`` time (ISO 8601) overrides the named window, no further back than ``MAX_LOOKBACK_DAYS``.
A ``ticket`` asked about with no window looks back ``TICKET_LOOKBACK_DAYS``: a decision on a card
is often older than a night. Each window's ``said`` reads after "Nothing" ("Nothing tonight",
"Nothing since 03:04 on 13 Oct", "Nothing in the last 24 hours").
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, NamedTuple, Optional, Tuple

TONIGHT, TODAY, YESTERDAY, LAST_DAY, THIS_WEEK = "tonight", "today", "yesterday", "last_24_hours", "this_week"
WINDOWS = (TONIGHT, TODAY, YESTERDAY, LAST_DAY, THIS_WEEK)
DEFAULT_WINDOW = LAST_DAY
NIGHT_STARTS_HOUR = 18  # "tonight" runs from the evening before the small hours it is asked in
WEEK_DAYS = 7
TICKET_LOOKBACK_DAYS = 30
MAX_LOOKBACK_DAYS = 90
SINCE_SAID = "since {when}"
LOOKBACK_SAID = "in the last {days} days"
SAID = {LAST_DAY: "in the last 24 hours", THIS_WEEK: "this week"}
WHEN = "{hour:%H:%M} on {day} {hour:%b}"  # "03:04 on 13 Oct", read as history ("at 03:04")
BAD_WINDOW = "window must be one of: {windows}"
# P256-FIX-N17-10 (F449): what Auto sent for a window (night 17: '1d' 11 times, '7d' 4), read as the window it means.
WINDOWS_SAID = {"1d": LAST_DAY, "24h": LAST_DAY, "7d": THIS_WEEK, "week": THIS_WEEK}
BAD_SINCE = "since must be a time like 2026-10-12T18:00:00Z; got {said!r}"


class Window(NamedTuple):
    """From ``start`` up to ``end`` (UTC), and how the owner said it."""

    start: datetime
    end: datetime
    said: str


def window_of(params: Dict[str, Any], now: Optional[datetime] = None) -> Tuple[Optional[Window], Optional[str]]:
    """The window ``params`` asks for (``since``, else ``window``, else the ticket's lookback or
    the default), or None and why it cannot be read."""
    now = _utc(now or datetime.now(timezone.utc))
    since = params.get("since")
    if since not in (None, ""):
        start = _parsed(since)
        if start is None:
            return None, BAD_SINCE.format(said=since)
        start = max(start, now - timedelta(days=MAX_LOOKBACK_DAYS))
        return Window(start, now, SINCE_SAID.format(when=when_said(start))), None
    named = params.get("window")
    if named in (None, "") and params.get("ticket") not in (None, ""):
        return Window(now - timedelta(days=TICKET_LOOKBACK_DAYS), now, LOOKBACK_SAID.format(days=TICKET_LOOKBACK_DAYS)), None
    named = str(named or DEFAULT_WINDOW).strip().lower()
    named = WINDOWS_SAID.get(named, named)
    if named not in WINDOWS:
        return None, BAD_WINDOW.format(windows=", ".join(WINDOWS))
    return _named(named, now), None


def when_said(moment: datetime) -> str:
    """'03:04 on 13 Oct' (UTC)."""
    moment = _utc(moment)
    return WHEN.format(hour=moment, day=moment.day)


def _named(named: str, now: datetime) -> Window:
    midnight = datetime.combine(now.date(), time(0), tzinfo=timezone.utc)
    if named == TONIGHT:
        evening = midnight + timedelta(hours=NIGHT_STARTS_HOUR)
        start = evening if now >= evening else evening - timedelta(days=1)
        return Window(start, now, TONIGHT)
    if named == TODAY:
        return Window(midnight, now, TODAY)
    if named == YESTERDAY:
        return Window(midnight - timedelta(days=1), midnight, YESTERDAY)
    if named == THIS_WEEK:
        return Window(now - timedelta(days=WEEK_DAYS), now, SAID[THIS_WEEK])
    return Window(now - timedelta(days=1), now, SAID[LAST_DAY])


def _parsed(said: Any) -> Optional[datetime]:
    try:
        return _utc(datetime.fromisoformat(str(said).strip()))
    except (ValueError, OverflowError):  # not a time, or one out of range ("0001-01-01T00:00+05:00")
        return None


def _utc(moment: datetime) -> datetime:
    """A naive time is UTC (the database's own); an aware one is put in UTC."""
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


__all__ = ["DEFAULT_WINDOW", "MAX_LOOKBACK_DAYS", "TICKET_LOOKBACK_DAYS", "WINDOWS", "WINDOWS_SAID", "Window", "when_said",
           "window_of"]
