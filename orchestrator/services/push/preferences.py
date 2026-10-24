"""PRD-266 US-014a (§4.4, K7, K12) — the phone's push switches, as notification preference rows.

A push switch is a ``notification_preferences`` row with destination ``push`` (K7): the
fan-out reads it at send time (``fanout.eligible_devices``: the person's row, else the
workspace's), and the dispatcher never delivers it, so a push row never costs the bell row.
The app has three switches, one per push type, and the daily digest's; ``PUSH_TYPES`` maps
each to the events its rows are for, and ``GET /api/notification-preferences`` returns that
map as ``push_types``:

* ``needs_answer``: question_pending, approval_pending, mission_plan_ready
* ``review``: review_pending
* ``auto_message``: auto_message
* ``digest``: daily_digest, the per-person digest switch (K12). The digest itself is posted
  as Auto's message and pushed as ``auto_message``.

No row means on, so GET shows an enabled ``push`` row for each of those six events that has
no push row yet (``with_push_defaults``).

The contract for the app: ``PUT /api/notification-preferences`` replaces ALL of the caller's
rows for each event_type it names (``bulk_update_preferences``' rule). To change a switch,
the app sends every row GET returned for that event, the push row changed among them, never
the push row alone: the push row alone deletes the caller's other rows for that event.

A PUT may hold a ``push`` row for those six events only, and a ``daily_digest`` row only as a
push row (``push_row_allowed``; the request model answers 422 otherwise).

Pure: no DB, no config, no logging.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

from services.push.payload import PUSH_EVENT_TYPES, PUSH_TYPE_EVENTS

# K7: the destination that is a push switch, never a delivery.
PUSH_DESTINATION = "push"
# K12: the digest's switch, beside the three push types.
DIGEST = "digest"
DAILY_DIGEST_EVENT = "daily_digest"

PUSH_TYPES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    **PUSH_TYPE_EVENTS,
    DIGEST: (DAILY_DIGEST_EVENT,),
})
# Every event a push row may name, in PUSH_TYPES' order: the five push events and the digest.
PUSH_SWITCH_EVENTS: tuple[str, ...] = tuple(event for events in PUSH_TYPES.values() for event in events)

PUSH_ROW_REFUSED = (
    "A 'push' row is only for {events}, and a '" + DAILY_DIGEST_EVENT + "' row only a 'push' row: {refused}"
)


def push_row_allowed(event_type: str, destination: str) -> bool:
    """May a PUT hold this row? A push row names a push event or the digest (K7, K12).

    A ``daily_digest`` row is only ever a push row: the digest is never a bell row of its own.
    """
    if destination == PUSH_DESTINATION:
        return event_type in PUSH_EVENT_TYPES or event_type == DAILY_DIGEST_EVENT
    return event_type != DAILY_DIGEST_EVENT


def refused_message(refused: Sequence[str]) -> str:
    """The 422's words for the event types of the rows ``push_row_allowed`` refused."""
    return PUSH_ROW_REFUSED.format(events=", ".join(PUSH_SWITCH_EVENTS), refused=", ".join(refused))


def push_types() -> dict[str, list[str]]:
    """GET's ``push_types``: each of the app's switches, and the events its rows are for."""
    return {switch: list(events) for switch, events in PUSH_TYPES.items()}


def _default_row(event_type: str) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "destination": PUSH_DESTINATION,
        "enabled": True,
        "channel_connection_id": None,
    }


def with_push_defaults(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``rows`` (copied), then an enabled push row for each switch event without one; a new list.

    ``rows`` are GET's merged preference rows (``event_type``, ``destination``, ``enabled``,
    ``channel_connection_id``). A push row the caller (or the workspace) already has is kept
    as it is, off included; no row means on (K7).
    """
    switched = {row.get("event_type") for row in rows if row.get("destination") == PUSH_DESTINATION}
    defaults = [_default_row(event) for event in PUSH_SWITCH_EVENTS if event not in switched]
    return [dict(row) for row in rows] + defaults


__all__ = [
    "DAILY_DIGEST_EVENT",
    "DIGEST",
    "PUSH_DESTINATION",
    "PUSH_SWITCH_EVENTS",
    "PUSH_TYPES",
    "push_row_allowed",
    "push_types",
    "refused_message",
    "with_push_defaults",
]
