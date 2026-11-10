"""Partner tools: the GET itself (PRD-275).

One request, through the platform's outbound seam (``core.security.web_access``), as an
operator's own destination: the ``WEB_ACCESS`` switch does not govern it, the always-refused
ranges (private, loopback, link-local, metadata) and the operator's denylist do.

* **GET only.** There is no other method here to ask for.
* **https only**, TLS verified, sent to the address that was checked (the hostname rides as
  ``Host`` and SNI), so a DNS answer that changes after the check changes nothing. The
  environment's proxy settings are not read: a proxy would undo the pin.
* **No redirects.** A 3xx is refused and its ``Location`` is never fetched: a redirect would
  carry the key to an address nobody checked.
* **Bytes capped as sent.** Only an unencoded body is read, and reading stops at
  ``MAX_RESPONSE_BYTES``.
* **One deadline for the whole call** (``TOTAL_SECONDS``), so a partner that answers a byte
  at a time cannot hold a call open.

Nothing here logs a header, a URL's query or an exception's text: a client error can quote
the header it choked on. A refused address is logged, and the model is told only that it
was refused.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import httpx

from core.security.web_access import OutboundTarget, build_pinned_request, resolve_outbound_async
from modules.tools.partner.request import HTTPS_PORT, REQUIRED_SCHEME, PartnerRefusal

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 15.0  # each wait: to connect, to send, for the next bytes
TOTAL_SECONDS = 30.0    # the whole call
MAX_RESPONSE_BYTES = 1_000_000
USER_AGENT = "AutomatosPartnerTools/1.0"
FIRST_REDIRECT, AFTER_REDIRECTS = 300, 400
_IDENTITY_ENCODINGS = ("", "identity")

HOST_REFUSED = "partner_host_refused"
REDIRECT_REFUSED = "partner_redirect_refused"
RESPONSE_TOO_LARGE = "partner_response_too_large"
UNREADABLE = "partner_unreadable_response"
TIMED_OUT = "partner_timed_out"
UNREACHABLE = "partner_unreachable"


@dataclass(frozen=True)
class Answer:
    """The partner's answer, within the byte cap."""

    status_code: int
    body: bytes


def _async_client(**kwargs: Any) -> httpx.AsyncClient:
    """The HTTP client factory (tests route it into an ``httpx.MockTransport``)."""
    return httpx.AsyncClient(**kwargs)


async def _read_capped(response: httpx.Response, max_bytes: int) -> Tuple[bytes, bool]:
    """The body up to ``max_bytes``, and whether there was more."""
    chunks: List[bytes] = []
    received = 0
    async for chunk in response.aiter_bytes():
        received += len(chunk)
        if received > max_bytes:
            return b"".join(chunks), True
        chunks.append(chunk)
    return b"".join(chunks), False


async def _answer(response: httpx.Response) -> Answer:
    status = response.status_code
    if FIRST_REDIRECT <= status < AFTER_REDIRECTS:
        raise PartnerRefusal(REDIRECT_REFUSED, "The partner answered with a redirect, which partner tools "
                                               "never follow.", status)
    if (response.headers.get("content-encoding") or "").strip().lower() not in _IDENTITY_ENCODINGS:
        raise PartnerRefusal(UNREADABLE, "The partner sent a compressed answer after being asked for none.", status)
    body, more = await _read_capped(response, MAX_RESPONSE_BYTES)
    if more:
        raise PartnerRefusal(RESPONSE_TOO_LARGE, f"The partner's answer is larger than {MAX_RESPONSE_BYTES} bytes, "
                                                 "so it was not read.", status)
    return Answer(status, body)


def _sent_headers(target: OutboundTarget, headers: Mapping[str, str]) -> Dict[str, str]:
    """What the request carries: the client's own headers, then the key's. On a port other than
    443 the ``Host`` names the port too (the outbound seam's own ``Host`` is the name alone)."""
    own = {"User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "identity"}
    host = {} if target.port == HTTPS_PORT else {"Host": f"{target.host}:{target.port}"}
    return {**own, **host, **headers}


async def _send(url: str, target: OutboundTarget, headers: Mapping[str, str]) -> Answer:
    async with _async_client(follow_redirects=False, verify=True, trust_env=False,
                             timeout=httpx.Timeout(TIMEOUT_SECONDS)) as client:
        response = await client.send(build_pinned_request(client, "GET", url, target, headers=dict(headers)),
                                     stream=True)
        try:
            return await _answer(response)
        finally:
            await response.aclose()


async def get(url: str, headers: Mapping[str, str]) -> Answer:
    """GET ``url`` with ``headers`` (the key among them), or raise :class:`PartnerRefusal`."""
    target = await resolve_outbound_async(url, enforce_switch=False)  # an operator's destination, not web access
    if not target.ok or target.scheme != REQUIRED_SCHEME:
        logger.warning("[partner-tools] address refused: %s", target.reason if not target.ok else "not https")
        raise PartnerRefusal(HOST_REFUSED, "The partner's address was refused: it must be a public https host.")
    try:
        async with asyncio.timeout(TOTAL_SECONDS):
            return await _send(url, target, _sent_headers(target, headers))
    except (TimeoutError, httpx.TimeoutException):
        raise PartnerRefusal(TIMED_OUT, "The partner did not answer in time.") from None
    except httpx.HTTPError as failed:
        logger.warning("[partner-tools] request to %s failed: %s", target.host, type(failed).__name__)
        raise PartnerRefusal(UNREACHABLE, "The partner could not be reached.") from None


__all__ = ["Answer", "HOST_REFUSED", "MAX_RESPONSE_BYTES", "REDIRECT_REFUSED", "RESPONSE_TOO_LARGE", "TIMED_OUT",
           "TOTAL_SECONDS", "UNREACHABLE", "UNREADABLE", "get"]
