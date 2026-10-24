"""PRD-266 US-010a — the push sender's contract: what the fan-out and the receipts job hand over.

The fan-out builds ``PushMessage`` values and calls ``PushSender.send``; the receipts job keeps
the ``ticket_id`` of every accepted ticket and later calls ``PushSender.receipts``. Neither speaks
HTTP: ``services/push/expo.py`` is the one implementation that does (§4.10, AGENTS.md exception
2), and tests hand in a fake.

Both methods take their argument positionally (``sender.send(batch)``): PRD-264 EUAI-03's
guardrail (``tests/test_prd264_euai03_single_outbound_point.py``) reads a keyword-only
``<x>.send(...)`` as a channel driver's send.

A message holds a push token, a title, a body and data: none of them is ever logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, runtime_checkable

TICKET_OK = "ok"
TICKET_ERROR = "error"
TicketStatus = Literal["ok", "error"]

# Expo's own values: a question wakes the phone at once, with the system sound.
PRIORITY_HIGH = "high"
SOUND_DEFAULT = "default"


@dataclass(frozen=True)
class PushMessage:
    """One push to one phone: its Expo push token (``to``), what the phone shows, and ``data``.

    ``category_id`` is the iOS notification category (its action buttons) and ``channel_id``
    the Android channel; ``None`` leaves either out. ``data`` is copied into a read-only
    mapping, so a message never changes after it is built.
    """

    to: str
    title: str
    body: str
    data: Mapping[str, Any] = field(default_factory=dict)
    category_id: Optional[str] = None
    channel_id: Optional[str] = None
    priority: str = PRIORITY_HIGH
    sound: Optional[str] = SOUND_DEFAULT

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))


@dataclass(frozen=True)
class PushTicket:
    """Expo's answer to one message, in the order sent: accepted (``ticket_id``) or ``error``.

    ``error`` is Expo's error code (``DeviceNotRegistered``, ``MessageTooBig`` …) or the
    sender's own (``expo_request_failed``): a code, never Expo's message, which can echo the
    token.
    """

    to: str
    status: TicketStatus
    ticket_id: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class PushReceipt:
    """Expo's later word on one accepted ticket: delivered to Apple or Google (``ok``) or not."""

    ticket_id: str
    status: TicketStatus
    error: Optional[str] = None


@runtime_checkable
class PushSender(Protocol):
    """Hands pushes to a push service and reads their receipts. Never raises for a failed send."""

    def send(self, messages: Sequence[PushMessage], /) -> list[PushTicket]:
        """One ticket per message, in order; a message the service never accepted is an error ticket."""
        ...

    def receipts(self, ticket_ids: Sequence[str], /) -> dict[str, PushReceipt]:
        """The receipts the service has for these ids, by id; an id it has none for is absent."""
        ...
