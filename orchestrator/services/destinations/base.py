"""
Destination dispatch — common types (PRD-008-A.1).

Callback destinations route through the workspace's existing
ChannelConnection records (PRD-55) — same Slack/Telegram/WhatsApp
connections used by heartbeats, agents, and channel inbound. We don't
maintain a parallel destination zoo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Only one destination type. Each entry stores a reference to a
# ChannelConnection the workspace already configured under
# Settings → Channels.
DESTINATION_TYPES: tuple[str, ...] = ("channel_connection",)


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
