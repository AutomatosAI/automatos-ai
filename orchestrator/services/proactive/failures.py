"""PRD-268 §6.2.2 — the error class a failed check-in records, which keys its failing streak.

A check-in that fails is an ``error`` run, and its ``error_class`` names the failure the way
the person would: one streak, and one card (``services.proactive.failing_streaks``), per
agent and class. So the class says what went wrong, not only which Python type carried it:

* ``out_of_credit``: the AI provider's account has no credit (``core.llm.credit``);
* ``http_<status>``: the provider refused the request with that HTTP status, read from the
  SDK error's ``status_code`` (or the one it was raised from) or its message
  ("Error code: 401 - {…}"); VECTOR's dead key on 26 Oct is ``http_401``;
* ``timeout``: the call or a client under it timed out;
* anything else: the exception's class name, as the runner recorded it before.

The class is built from the exception's type and status only; no text the provider or the
model wrote is stored.
"""
from __future__ import annotations

from typing import Iterator, Optional

from core.llm.credit import is_out_of_credit, provider_status

__all__ = [
    "HTTP_PREFIX",
    "OUT_OF_CREDIT",
    "TIMEOUT",
    "failure_class",
    "http_status",
]

OUT_OF_CREDIT = "out_of_credit"
TIMEOUT = "timeout"
HTTP_PREFIX = "http_"
# The class names the SDKs and httpx give a timeout: APITimeoutError, ReadTimeout, ConnectTimeout.
_TIMEOUT_NAME = "Timeout"
_LOWEST_STATUS, _HIGHEST_STATUS = 100, 599


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """The exception and the one it was raised from, if any (LLMManager re-raises a model
    that is gone as a ValueError ``from`` the provider's error)."""
    yield exc
    if exc.__cause__ is not None:
        yield exc.__cause__


def _status_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int) and not isinstance(code, bool) and _LOWEST_STATUS <= code <= _HIGHEST_STATUS:
        return code
    stated = provider_status(exc)
    return int(stated) if stated else None


def _timed_out(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or _TIMEOUT_NAME in type(exc).__name__


def failure_class(exc: BaseException) -> str:
    """The ``error_class`` of a check-in that failed on ``exc``."""
    chain = tuple(_chain(exc))
    if any(is_out_of_credit(link) for link in chain):
        return OUT_OF_CREDIT
    for link in chain:
        status = _status_code(link)
        if status is not None:
            return f"{HTTP_PREFIX}{status}"
    if any(_timed_out(link) for link in chain):
        return TIMEOUT
    return type(exc).__name__


def http_status(error_class: str) -> Optional[str]:
    """The status an ``http_<status>`` class names, else None."""
    if not error_class.startswith(HTTP_PREFIX):
        return None
    status = error_class[len(HTTP_PREFIX):]
    return status if status.isdecimal() else None
