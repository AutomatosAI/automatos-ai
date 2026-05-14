"""
Destination dispatch — common types (PRD-008-A Phase 6).

Each destination dispatcher returns a ``DispatchResult`` so the
orchestrator can decide whether to retry, log, or surface the failure
to the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Destination types the merchant can configure. Adding a new type
# requires a matching dispatcher in this package + an entry in the
# DISPATCHERS map (services/destinations/dispatcher.py).
DESTINATION_TYPES: tuple[str, ...] = (
    "email",
    "slack_webhook",
    "crm_webhook",
    "shopify_customer_note",
)


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
    # permanent (e.g. 4xx auth error) — don't waste retries.
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
