"""PRD-268 §6.9, FR-11 — a person's quiet hours, with the workspace's window on top.

A person's quiet hours are ``proactive_person_settings`` ``quiet_start``, ``quiet_end`` and
``timezone``, written through ``PUT /api/proactive/me`` (``services.proactive.settings``).
The window is read on the person's wall clock: their own time zone (the one their phone
sends), else the zone of the workspace's quiet hours, else UTC. A window whose start is later
than its end runs across midnight; a change of clocks moves it with the wall clock.

The workspace's window (``core.services.auto_reporting.is_quiet_hours``, over
``workspace.settings.auto_reporting``) still applies on top: it is quiet for the person when
it is quiet in either. The person's window is read by the same function, so the two agree on
what a window is.

W1 records and exposes the hours; W2's ``suggestion`` push reads ``is_person_quiet`` and holds
in quiet hours (FR-11). Check-ins still run and record in quiet hours.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.services.auto_reporting import is_quiet_hours

logger = logging.getLogger(__name__)

__all__ = ["clock", "is_person_quiet", "parse_clock", "person_timezone"]

CLOCK_FORMAT = "%H:%M"
FALLBACK_TIMEZONE = "UTC"
# workspace.settings.auto_reporting's window, as core.services.auto_reporting shapes it.
QUIET_HOURS_KEY = "quiet_hours"
WINDOW_ENABLED, WINDOW_START, WINDOW_END, WINDOW_TIMEZONE = "enabled", "start", "end", "timezone"
# The person's: proactive_person_settings' columns.
QUIET_START, QUIET_END, TIMEZONE = "quiet_start", "quiet_end", "timezone"


def clock(value: Optional[time]) -> Optional[str]:
    """A time of day as ``HH:MM``; None stays None."""
    return value.strftime(CLOCK_FORMAT) if value is not None else None


def parse_clock(value: Optional[str]) -> Optional[time]:
    """``HH:MM`` as a time of day; None stays None. Anything else raises ``ValueError``."""
    return datetime.strptime(value, CLOCK_FORMAT).time() if value is not None else None


def _known(name: Optional[str]) -> bool:
    if not name:
        return False
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        logger.warning("[proactive] quiet hours: time zone %r is not known; using the next one", name)
        return False
    return True


def _workspace_window(workspace_settings: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    window = (workspace_settings or {}).get(QUIET_HOURS_KEY)
    return window if isinstance(window, Mapping) else {}


def person_timezone(settings: Any, workspace_settings: Optional[Mapping[str, Any]]) -> str:
    """The zone a person's window is read in: their own, else the workspace's quiet-hours
    zone, else UTC. A zone zoneinfo does not know is passed over."""
    for name in (getattr(settings, TIMEZONE, None), _workspace_window(workspace_settings).get(WINDOW_TIMEZONE)):
        if _known(name):
            return str(name)
    return FALLBACK_TIMEZONE


def _aware(now: Optional[datetime]) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)


def _own_window_quiet(settings: Any, workspace_settings: Optional[Mapping[str, Any]], moment: datetime) -> bool:
    start, end = getattr(settings, QUIET_START, None), getattr(settings, QUIET_END, None)
    if start is None or end is None:
        return False
    window = {WINDOW_ENABLED: True, WINDOW_START: clock(start), WINDOW_END: clock(end),
              WINDOW_TIMEZONE: person_timezone(settings, workspace_settings)}
    return is_quiet_hours({QUIET_HOURS_KEY: window}, moment)


def is_person_quiet(settings: Any, workspace_settings: Optional[Mapping[str, Any]],
                    now: Optional[datetime] = None) -> bool:
    """Whether ``now`` is quiet for a person: inside their own window or the workspace's.

    ``settings`` is the person's (anything with ``quiet_start``, ``quiet_end`` and
    ``timezone``: a ``proactive_person_settings`` row or ``settings.PersonSettings``), or None
    when they set none. ``workspace_settings`` is the workspace's ``auto_reporting`` settings
    as ``core.services.auto_reporting.load_auto_reporting_settings`` returns them. ``now``
    defaults to the current time; a naive one is read as UTC.
    """
    moment = _aware(now)
    if _own_window_quiet(settings, workspace_settings, moment):
        return True
    return is_quiet_hours(dict(workspace_settings or {}), moment)
