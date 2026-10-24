"""PRD-266 US-010a — ``ExpoPushSender``: pushes go to Expo's push service, receipts come back.

**AGENTS.md exception 2** (PRD-266 §4.10, approved by Gerard on 20 Oct, tracking issue #1254):
this is the one third-party client of the wave, against "integrations go through Composio; no
new third-party API clients". Push is platform infrastructure, not a workspace integration: it
is how the platform reaches its own app on a member's phone, with no workspace connection to go
through, as Retell is for voice (``modules/voice/retell_api.py``) and Stripe is for PRD-265.
``httpx`` is already a dependency. No other module of the wave imports an HTTP client.

* **Where to.** ``EXPO_SEND_URL`` in batches of at most ``EXPO_SEND_BATCH`` (100) messages and
  ``EXPO_RECEIPTS_URL`` in batches of at most ``EXPO_RECEIPTS_BATCH`` (1,000) ids, over a
  synchronous ``httpx.Client`` with TLS verification on (the fan-out and the receipts job run
  off the event loop: K6), each request timing out after ``config.EXPO_PUSH_TIMEOUT_SECONDS``.
* **The access token.** ``config.EXPO_PUSH_ACCESS_TOKEN`` is an environment setting read through
  ``config.py``, never a key in Settings; sent as ``Authorization: Bearer`` only when it is set.
* **Expo's answers are validated here** (AGENTS.md: third-party responses). A batch whose answer
  is not JSON, carries a request-level ``errors`` array, is not Expo's shape or holds a ticket
  count other than the batch's makes every message of that batch an ``expo_request_failed``
  ticket, logged once; so does a batch whose messages cannot be encoded as JSON. One ticket not
  in Expo's shape is an ``expo_malformed_answer`` ticket for its message only, and one such
  receipt is left out. A 429 or a 5xx is retried once after ``EXPO_RETRY_DELAY_SECONDS`` (Expo
  may already have queued a batch it answered 5xx to, so a retried 5xx can deliver twice); any
  other 4xx is not. A transport error never raises out of ``send`` or ``receipts``.
* **Never logged:** a push token, the access token, a title, a body, data, or Expo's own error
  message (it can echo the token). Logs hold counts and Expo's error codes only.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from config import config
from services.push.sender import TICKET_ERROR, TICKET_OK, PushMessage, PushReceipt, PushTicket

logger = logging.getLogger(__name__)

EXPO_SEND_URL = "https://exp.host/--/api/v2/push/send"
EXPO_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts"
EXPO_SEND_BATCH = 100
EXPO_RECEIPTS_BATCH = 1000
EXPO_RETRY_DELAY_SECONDS = 1.0
TOO_MANY_REQUESTS = 429
FIRST_SERVER_ERROR = 500

REQUEST_FAILED = "expo_request_failed"
MALFORMED_ANSWER = "expo_malformed_answer"
UNKNOWN_ERROR = "expo_unknown_error"
# Expo's error codes are short identifiers (fullmatch: no trailing newline); anything else is
# kept and logged as UNKNOWN_ERROR. Ticket ids are Expo's UUIDs, held to a plain id's characters.
_ERROR_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_TICKET_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"

JSON_TYPE = "application/json"
BASE_HEADERS = {"Accept": JSON_TYPE, "Accept-Encoding": "gzip", "Content-Type": JSON_TYPE}

ClientFactory = Callable[[], httpx.Client]
_Item = TypeVar("_Item")


class _ExpoDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")
    error: Optional[str] = None


class _ExpoTicket(BaseModel):
    """One ticket in a send answer; an accepted one carries its id."""

    model_config = ConfigDict(extra="ignore")
    status: Literal["ok", "error"]
    id: Optional[str] = Field(default=None, pattern=_TICKET_ID_PATTERN)
    details: Optional[_ExpoDetails] = None

    # Not underscored: pydantic takes an underscored class attribute for a private attribute.
    @model_validator(mode="after")
    def accepted_has_id(self) -> "_ExpoTicket":
        if self.status == TICKET_OK and not self.id:
            raise ValueError("an accepted ticket carries its id")
        return self


class _ExpoReceipt(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Literal["ok", "error"]
    details: Optional[_ExpoDetails] = None


class _SendAnswer(BaseModel):
    """A send answer's envelope; each ticket is validated on its own (``_ticket``)."""

    model_config = ConfigDict(extra="ignore")
    data: List[Any] = Field(default_factory=list)
    errors: Optional[List[Any]] = None


class _ReceiptsAnswer(BaseModel):
    """A getReceipts answer's envelope; each receipt is validated on its own (``_receipt``)."""

    model_config = ConfigDict(extra="ignore")
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: Optional[List[Any]] = None


class _BatchFailed(Exception):
    """A request that gave no usable answer; ``reason`` is the sender's own words, never Expo's."""


def _default_client() -> httpx.Client:
    """The production client: Expo's timeout from config, TLS verified, no redirects followed."""
    return httpx.Client(timeout=config.EXPO_PUSH_TIMEOUT_SECONDS, verify=True, follow_redirects=False)


def _headers() -> Dict[str, str]:
    """Expo's JSON headers, with the access token only when one is configured."""
    token = str(config.EXPO_PUSH_ACCESS_TOKEN or "").strip()
    return {**BASE_HEADERS, "Authorization": f"Bearer {token}"} if token else dict(BASE_HEADERS)


def _chunks(items: Sequence[_Item], size: int) -> List[Sequence[_Item]]:
    return [items[start:start + size] for start in range(0, len(items), size)]


def _retryable(status_code: int) -> bool:
    return status_code == TOO_MANY_REQUESTS or status_code >= FIRST_SERVER_ERROR


def _encode(body: Any) -> bytes:
    """The request body as JSON bytes; data that is not JSON fails the batch, never raises out."""
    try:
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as not_json:
        raise _BatchFailed("the messages are not JSON") from not_json


def _error_code(details: Optional[_ExpoDetails]) -> str:
    """Expo's error code when it is a plain identifier; otherwise ``UNKNOWN_ERROR``."""
    code = details.error if details else None
    return code if code and _ERROR_CODE.fullmatch(code) else UNKNOWN_ERROR


def _expo_message(message: PushMessage) -> Dict[str, Any]:
    """One message in Expo's shape; an unset category, channel or sound is left out."""
    optional = {"categoryId": message.category_id, "channelId": message.channel_id, "sound": message.sound}
    return {
        "to": message.to,
        "title": message.title,
        "body": message.body,
        "data": dict(message.data),
        "priority": message.priority,
        **{key: value for key, value in optional.items() if value is not None},
    }


def _ticket(to: str, raw: Any) -> PushTicket:
    """Expo's ticket for the message sent to ``to``; one not in Expo's shape is ``MALFORMED_ANSWER``."""
    try:
        answer = _ExpoTicket.model_validate(raw)
    except ValidationError:
        return PushTicket(to=to, status=TICKET_ERROR, error=MALFORMED_ANSWER)
    if answer.status == TICKET_OK:
        return PushTicket(to=to, status=TICKET_OK, ticket_id=answer.id)
    return PushTicket(to=to, status=TICKET_ERROR, error=_error_code(answer.details))


def _receipt(ticket_id: str, raw: Any) -> Optional[PushReceipt]:
    """Expo's receipt for ``ticket_id``; ``None`` when it is not in Expo's shape."""
    try:
        answer = _ExpoReceipt.model_validate(raw)
    except ValidationError:
        return None
    error = _error_code(answer.details) if answer.status == TICKET_ERROR else None
    return PushReceipt(ticket_id=ticket_id, status=answer.status, error=error)


def _failed_batch(batch: Sequence[PushMessage]) -> List[PushTicket]:
    return [PushTicket(to=message.to, status=TICKET_ERROR, error=REQUEST_FAILED) for message in batch]


def _reason(failed: Exception) -> str:
    """What to log for a failed batch: our reason, never a validation error's echo of the input."""
    return str(failed) if isinstance(failed, _BatchFailed) else "the answer is not in Expo's shape"


def _log_refusals(tickets: Sequence[PushTicket]) -> None:
    """One line with the refused count and the error codes, when any message was refused."""
    codes = Counter(ticket.error for ticket in tickets if ticket.status == TICKET_ERROR)
    if codes:
        summary = ", ".join(f"{code}={count}" for code, count in sorted(codes.items()))
        logger.info("Expo push: %d of %d messages refused (%s)", sum(codes.values()), len(tickets), summary)


class ExpoPushSender:
    """``PushSender`` over Expo's push service. Synchronous: call it off the event loop.

    ``client_factory`` makes the ``httpx.Client`` for one ``send`` or ``receipts`` call (tests
    pass one over ``httpx.MockTransport``); ``sleep`` waits before the one retry.
    """

    def __init__(
        self,
        client_factory: Optional[ClientFactory] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client_factory = client_factory or _default_client
        self._sleep = sleep

    def send(self, messages: Sequence[PushMessage], /) -> list[PushTicket]:
        """One ticket per message, in order, in requests of at most ``EXPO_SEND_BATCH``."""
        batches = _chunks(list(messages), EXPO_SEND_BATCH)
        if not batches:
            return []
        with self._client_factory() as client:
            tickets = [ticket for batch in batches for ticket in self._send_batch(client, batch)]
        _log_refusals(tickets)
        return tickets

    def receipts(self, ticket_ids: Sequence[str], /) -> dict[str, PushReceipt]:
        """Expo's receipts for these ids, by id, in requests of at most ``EXPO_RECEIPTS_BATCH``.

        An id Expo has no receipt for yet, or whose batch failed, is absent: the receipts job
        asks again on its next run.
        """
        batches = _chunks(list(dict.fromkeys(ticket_ids)), EXPO_RECEIPTS_BATCH)
        if not batches:
            return {}
        with self._client_factory() as client:
            return {
                ticket_id: receipt
                for batch in batches
                for ticket_id, receipt in self._receipts_batch(client, batch).items()
            }

    def _send_batch(self, client: httpx.Client, batch: Sequence[PushMessage]) -> List[PushTicket]:
        try:
            body = self._post_json(client, EXPO_SEND_URL, [_expo_message(message) for message in batch])
            answer = _SendAnswer.model_validate(body)
            if answer.errors:
                raise _BatchFailed("Expo answered request errors")
            if len(answer.data) != len(batch):
                raise _BatchFailed(f"{len(answer.data)} tickets for {len(batch)} messages")
        except (_BatchFailed, ValidationError) as failed:
            logger.warning("Expo push send: a batch of %d messages failed (%s)", len(batch), _reason(failed))
            return _failed_batch(batch)
        return [_ticket(message.to, raw) for message, raw in zip(batch, answer.data)]

    def _receipts_batch(self, client: httpx.Client, batch: Sequence[str]) -> Dict[str, PushReceipt]:
        try:
            body = self._post_json(client, EXPO_RECEIPTS_URL, {"ids": list(batch)})
            answer = _ReceiptsAnswer.model_validate(body)
            if answer.errors:
                raise _BatchFailed("Expo answered request errors")
        except (_BatchFailed, ValidationError) as failed:
            logger.warning("Expo push receipts: a batch of %d ids failed (%s)", len(batch), _reason(failed))
            return {}
        asked = set(batch)
        parsed = {
            ticket_id: _receipt(ticket_id, raw) for ticket_id, raw in answer.data.items() if ticket_id in asked
        }
        found = {ticket_id: receipt for ticket_id, receipt in parsed.items() if receipt is not None}
        if len(found) < len(parsed):
            logger.warning("Expo push receipts: %d not in Expo's shape were left out", len(parsed) - len(found))
        return found

    def _post_json(self, client: httpx.Client, url: str, body: Any) -> Any:
        """POST ``body``, retrying a 429 or a 5xx once; the decoded JSON of a 2xx answer."""
        content = _encode(body)
        response = self._post_once(client, url, content)
        if _retryable(response.status_code):
            self._sleep(EXPO_RETRY_DELAY_SECONDS)
            response = self._post_once(client, url, content)
        if not response.is_success:
            raise _BatchFailed(f"status {response.status_code}")
        try:
            return response.json()
        except (ValueError, RecursionError) as not_json:
            raise _BatchFailed("the answer is not JSON") from not_json

    @staticmethod
    def _post_once(client: httpx.Client, url: str, content: bytes) -> httpx.Response:
        try:
            return client.post(url, content=content, headers=_headers())
        except httpx.HTTPError as failed:
            raise _BatchFailed(f"transport error {type(failed).__name__}") from failed
