"""
Destination dispatch — common types (PRD-008-A.1).

Callback destinations route through the same per-workspace notification
plumbing heartbeats use (``send_workspace_notification`` →
``channel_connections`` adapter). We don't maintain a parallel
destination zoo.

Each destination names a ``platform`` and (where the platform can't
auto-resolve a target from workspace integrations) any extra field the
adapter needs:

    {"platform": "telegram"}                              # auto-resolves
    {"platform": "slack",    "channel_id":  "C01ABC..."}
    {"platform": "webhook",  "webhook_url": "https://…"}

The legacy ``{"type": "channel_connection", "connection_id": …, "target": …}``
shape from the first iteration is still accepted on read for backward
compatibility — the validator and dispatcher coerce it to the new
platform-keyed shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Platforms a merchant team can route callbacks to. Mirrors
# ``CALLBACK_PLATFORMS`` in the dashboard's lib/channels/types.ts.
CALLBACK_PLATFORMS: tuple[str, ...] = (
    "telegram",
    "slack",
    "whatsapp",
    "webhook",
)

# Legacy alias retained for any external imports.
DESTINATION_TYPES: tuple[str, ...] = ("channel_connection",) + CALLBACK_PLATFORMS


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of attempting to deliver a single callback to a single
    destination on a single attempt."""

    success: bool
    destination_type: str
    latency_ms: int
    # Populated only on failure
    error: Optional[str] = None
    # If True, the orchestrator MAY retry. If False, the failure is
    # permanent (e.g. unknown adapter, cross-workspace connection_id) —
    # don't waste retries.
    retryable: bool = True
    # Free-form context surfaced to widget_event_log.event_data
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CallbackPayload:
    """The data each dispatcher receives. Plaintext phone number lives
    here and gets forwarded to the merchant's destination — never
    persisted anywhere in Automatos beyond the conversation transcript.
    """

    request_id: str
    name: str
    phone: str
    product_context: Optional[str]
    urgency: Optional[str]
    preferred_time: Optional[str]
    site_display_name: str
    site_external_id: Optional[str]
