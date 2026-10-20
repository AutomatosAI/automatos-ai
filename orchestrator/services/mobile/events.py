"""PRD-266 §4.2 — a device's audit trail: the last 50 things that happened to it.

``mobile_devices.events`` is a JSONB list of ``{"event": <name>, "at": <ISO-8601>}``,
oldest first. Writers build a NEW list with ``with_event`` and assign it, so the row's
previous list is never mutated in place (and SQLAlchemy sees the change).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence

MAX_DEVICE_EVENTS = 50
# A device was paired (or paired again) with a one-time code.
ENROLLED = "enrolled"


def with_event(events: Optional[Sequence[Dict[str, str]]], event: str, at: datetime) -> List[Dict[str, str]]:
    """``events`` plus ``{"event": event, "at": at}`` as a new list, keeping the newest 50."""
    entry = {"event": event, "at": at.isoformat()}
    return [*(events or []), entry][-MAX_DEVICE_EVENTS:]
