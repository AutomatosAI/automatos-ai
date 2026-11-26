"""How the last provider call through the LLM manager ended (PRD-282 US-202).

GET /health's model provider entry is never found out by calling the provider. The
manager's two call methods, ``LLMManager.generate_response`` and
``generate_response_sync``, make their provider call inside ``watching``, which notes
here how it ended and when, the moment the provider answers or fails, before any
bookkeeping: a usage row or cost note that fails afterwards leaves the answer noted. Embeddings, and the clients some services call directly, do not pass
through the manager and are not noted.

What is noted, built on the error class a failed check-in records
(services/proactive/failures.failure_class):

* **up:** the provider answered with text or a tool call.
* **down:** the provider could not be reached or would not accept the install: a
  connection error, a time-out, a 5xx, or a 401/403 on the install's own key, on
  FAILURES_BEFORE_DOWN calls in a row. Fewer leave the note as it was, so one failed
  call does not read ``down`` for as long as the reading lasts; an answer starts the
  count again.
* **Nothing changes** for a refusal of one request: any other 4xx (a bad request, a
  content refusal, a rate or output limit), an account out of credit, an empty answer,
  and a 401/403 on a workspace's own key (``is_byok`` on the manager: that workspace's
  key, not the install's provider).

One frozen value, replaced whole under a lock with the failure count: safe to read from
any thread. No database read.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, TypeVar

from core.llm.usage_status import empty_completion

logger = logging.getLogger(__name__)

T = TypeVar("T")
# The statuses that refuse the key rather than the request.
_KEY_REFUSED = frozenset({"401", "403"})
_SERVER_ERROR_FROM = 500
# The class names the SDKs and httpx give a connection that failed: APIConnectionError,
# ConnectionError, ConnectionRefusedError, ConnectError.
_UNREACHED_NAMES = ("Connection", "ConnectError")
# Unreachable calls in a row before the provider is noted down.
FAILURES_BEFORE_DOWN = 3


@dataclass(frozen=True)
class LastCall:
    """Whether the call succeeded, and when it ended (UTC)."""

    succeeded: bool
    ended_at: datetime


_last: Optional[LastCall] = None
_failures_in_a_row = 0
_lock = threading.Lock()


def _note(succeeded: bool) -> None:
    global _last, _failures_in_a_row
    with _lock:
        _failures_in_a_row = 0 if succeeded else _failures_in_a_row + 1
        if succeeded or _failures_in_a_row >= FAILURES_BEFORE_DOWN:
            _last = LastCall(succeeded=succeeded, ended_at=datetime.now(timezone.utc))


def last() -> Optional[LastCall]:
    """The last noted call's outcome in this process, or None before the first."""
    return _last


def _own_key(manager: Any) -> bool:
    return bool((getattr(manager, "_tracking_ctx", None) or {}).get("is_byok"))


def note_answer(response: Any) -> None:
    """A provider answer: ``up``, unless it was empty (that says nothing of the provider).
    Never raises: the answer it follows is the caller's, whatever happens here."""
    try:
        if not empty_completion(response):
            _note(True)
    except Exception:
        logger.exception("[health] a model answer could not be noted; the answer stands")


def note_failure(exc: BaseException, *, own_key: bool) -> None:
    """A provider call that raised: counts toward ``down`` only when the provider could not
    be reached or would not accept the install's key; any other failure leaves the note
    and the count as they were.
    Never raises: the caller's own error is the one that goes on."""
    try:
        if provider_unreachable(exc, own_key=own_key):
            _note(False)
    except Exception:
        logger.exception("[health] a model call's failure could not be noted; its own error goes on")


def provider_unreachable(exc: BaseException, *, own_key: bool) -> bool:
    """Did ``exc`` come from a provider that could not be reached or would not accept the
    install? Read from failure_class's error class, then the exception's class names."""
    from services.proactive.failures import OUT_OF_CREDIT, TIMEOUT, _chain, failure_class, http_status

    error_class = failure_class(exc)
    if error_class == TIMEOUT:
        return True
    if error_class == OUT_OF_CREDIT:
        return False
    status = http_status(error_class)
    if status is not None:
        return int(status) >= _SERVER_ERROR_FROM or (status in _KEY_REFUSED and not own_key)
    return any(name in type(link).__name__ for link in _chain(exc) for name in _UNREACHED_NAMES)


class Watching:
    """Around the provider call: an error raised inside is noted (``note_failure``) and
    goes on unchanged; ``answered`` notes the answer and returns it."""

    def __init__(self, manager: Any) -> None:
        self._own_key = _own_key(manager)

    def __enter__(self) -> "Watching":
        return self

    def __exit__(self, kind: Any, error: Optional[BaseException], traceback: Any) -> bool:
        if isinstance(error, Exception):
            note_failure(error, own_key=self._own_key)
        return False  # the error, if any, is the caller's

    def answered(self, response: T) -> T:
        note_answer(response)
        return response


def watching(manager: Any) -> Watching:
    """The note for one call through ``manager`` (its ``is_byok`` says whose key it used)."""
    return Watching(manager)
