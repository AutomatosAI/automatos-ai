"""PRD-270 K13 — the one way a local install reads Automatos Cloud: a guarded, anonymous GET.

The Sync's catalogue and a Cloud theme's package (US-270-04a) are both read here, with the guards
the revocation pull uses (``services/themes/revocation_pull.py``, copied, never imported: its tests
pin its own messages and headers):

* **Where.** ``checked_origin`` holds ``AUTOMATOS_CLOUD_URL`` to ``urls.origin_problem``'s rule: https
  (http only to this machine), no user name or password, no path, query or fragment.
* **What is sent.** One ``GET`` with exactly ``Accept``, ``Accept-Encoding: identity`` and
  ``User-Agent: Automatos/<THEME_HOST_VERSION> (local)`` (the host version decides which catalogue
  entries an install can run), from a client of its own: no ``Authorization``, ``Cookie``,
  ``X-Workspace-ID`` or install id.
* **What is read.** Redirects are never followed (a 3xx is a failure), anything but 200 is a failure,
  an encoded body is refused, and the declared and the streamed size are capped. Each read times out
  after ``AUTOMATOS_CLOUD_TIMEOUT_SECONDS`` and the whole read after ``DEADLINE_FACTOR`` times that.

Failures raise ``CloudFetchFailed`` (``CloudAnswerTooLarge`` past a cap) with a plain sentence of this
module's own, echoing nothing it received. ``client_factory`` is the tests' seam (an ``httpx.MockTransport``).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import httpx

from config import config
from services.marketplace_cloud.urls import (
    ORIGIN_CREDENTIALS, ORIGIN_NOT_HTTPS, ORIGIN_NOT_ORIGIN, ORIGIN_OFF, origin_of, origin_problem,
)
from services.themes.contract import KB, THEME_HOST_VERSION

OK_STATUS = 200
REDIRECT_STATUSES = range(300, 400)
IDENTITY_ENCODINGS = frozenset({"", "identity"})
# The whole read may take this many times one read's timeout.
DEADLINE_FACTOR = 4
ACCEPT_JSON = "application/json"
ACCEPT_PACKAGE = "application/zip"
USER_AGENT = f"Automatos/{THEME_HOST_VERSION} (local)"

CLOUD_OFF = "Sync with Automatos Cloud is off: AUTOMATOS_CLOUD_URL is empty."
URL_NOT_HTTPS = "AUTOMATOS_CLOUD_URL must be an https:// address (http:// only to this machine)."
URL_CREDENTIALS = "AUTOMATOS_CLOUD_URL must not carry a user name or password."
URL_NOT_ORIGIN = "AUTOMATOS_CLOUD_URL must be an origin only, with no path, query or fragment."
URL_INVALID = "AUTOMATOS_CLOUD_URL isn't an address this install can read."
UNREACHABLE = "Automatos Cloud couldn't be reached ({error})."
TIMED_OUT = "Automatos Cloud didn't answer in time."
REDIRECTED = "Automatos Cloud answered a redirect ({status}); redirects aren't followed."
BAD_STATUS = "Automatos Cloud answered {status}."
ENCODED = "Automatos Cloud sent an encoded answer; only an unencoded one is read."
TOO_LARGE = "Automatos Cloud's answer is larger than {kb} KB."

ORIGIN_MESSAGES: Dict[str, str] = {
    ORIGIN_OFF: CLOUD_OFF, ORIGIN_NOT_HTTPS: URL_NOT_HTTPS,
    ORIGIN_CREDENTIALS: URL_CREDENTIALS, ORIGIN_NOT_ORIGIN: URL_NOT_ORIGIN,
}

ClientFactory = Callable[[], httpx.Client]


class CloudFetchFailed(ValueError):
    """A read from Automatos Cloud that failed; the message is plain and echoes nothing it received."""


class CloudAnswerTooLarge(CloudFetchFailed):
    """The answer was over its cap, as declared or as streamed (a package install answers it as refused bytes)."""


def timeout_seconds() -> float:
    """``AUTOMATOS_CLOUD_TIMEOUT_SECONDS``: each read of the catalogue or a package (at least one second)."""
    return float(max(1, int(config.AUTOMATOS_CLOUD_TIMEOUT_SECONDS)))


def request_headers(accept: str) -> Dict[str, str]:
    """Exactly the three headers a read sends."""
    return {"Accept": accept, "Accept-Encoding": "identity", "User-Agent": USER_AGENT}


def checked_origin(url: str) -> str:
    """The origin of ``url`` when a Sync may read it, else ``CloudFetchFailed`` saying why."""
    problem = origin_problem(url)
    if problem is not None:
        raise CloudFetchFailed(ORIGIN_MESSAGES[problem])
    return origin_of(url)


def _client() -> httpx.Client:
    """The HTTP client (a seam: tests give it a mock transport). Its own: no auth, no cookies, no redirects."""
    return httpx.Client(follow_redirects=False, timeout=timeout_seconds())


def _too_large(max_bytes: int) -> CloudFetchFailed:
    return CloudAnswerTooLarge(TOO_LARGE.format(kb=max_bytes // KB))


def _read_capped(response: httpx.Response, max_bytes: int, deadline: float) -> bytes:
    """The body as sent, refused past ``max_bytes`` or the deadline."""
    if response.is_stream_consumed:  # a transport that loaded the body already: its bytes, capped
        if len(response.content) > max_bytes:
            raise _too_large(max_bytes)
        return response.content
    chunks: List[bytes] = []
    size = 0
    for chunk in response.iter_raw():  # each chunk as it arrives, so the cap bites early
        size += len(chunk)
        if size > max_bytes:
            raise _too_large(max_bytes)
        if time.monotonic() > deadline:
            raise CloudFetchFailed(TIMED_OUT)
        chunks.append(chunk)
    return b"".join(chunks)


def _read_answer(response: httpx.Response, max_bytes: int, deadline: float) -> bytes:
    status = response.status_code
    if status in REDIRECT_STATUSES:
        raise CloudFetchFailed(REDIRECTED.format(status=status))
    if status != OK_STATUS:
        raise CloudFetchFailed(BAD_STATUS.format(status=status))
    if (response.headers.get("content-encoding") or "").strip().lower() not in IDENTITY_ENCODINGS:
        raise CloudFetchFailed(ENCODED)
    declared = response.headers.get("content-length") or ""
    if declared.isascii() and declared.isdigit() and int(declared) > max_bytes:
        raise _too_large(max_bytes)
    return _read_capped(response, max_bytes, deadline)


def fetch_bytes(
    url: str, max_bytes: int, accept: str = ACCEPT_JSON, client_factory: Optional[ClientFactory] = None,
) -> bytes:
    """GET ``url`` (built from a checked origin) and its body, capped at ``max_bytes``; never follows a redirect."""
    deadline = time.monotonic() + DEADLINE_FACTOR * timeout_seconds()
    try:
        with (client_factory or _client)() as client:
            with client.stream("GET", url, headers=request_headers(accept), follow_redirects=False) as response:
                return _read_answer(response, max_bytes, deadline)
    except httpx.InvalidURL as bad:  # not an HTTPError
        raise CloudFetchFailed(URL_INVALID) from bad
    except httpx.TimeoutException as slow:
        raise CloudFetchFailed(TIMED_OUT) from slow
    except httpx.HTTPError as failed:
        raise CloudFetchFailed(UNREACHABLE.format(error=type(failed).__name__)) from failed


__all__ = [
    "ACCEPT_JSON", "ACCEPT_PACKAGE", "CLOUD_OFF", "ClientFactory", "CloudAnswerTooLarge", "CloudFetchFailed",
    "DEADLINE_FACTOR",
    "USER_AGENT", "checked_origin", "fetch_bytes", "request_headers", "timeout_seconds",
]
