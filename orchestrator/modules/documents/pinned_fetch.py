"""F375: a document fetches an http(s) resource only from the address it checked.

The WeasyPrint URL fetcher and the Word renderer's image fetch both checked a URL's
host (every address public), then handed the URL to a client that resolved it again
and (WeasyPrint) followed redirects. A DNS answer that changes between the check and
the connection (rebinding), or a public URL that redirects to 127.0.0.1 or the cloud
metadata address, reached what the check had refused.

Now the fetch is resolve-and-pin, through the platform's one outbound decision
(``core.security.web_access``): ``resolve_outbound`` checks every address the host
resolves to and returns the one it checked; the request goes to exactly that address
with the real hostname as ``Host`` and SNI (``build_pinned_request``), so TLS is still
verified against the name. Redirects are never followed automatically: each hop is
resolved and checked again, at most :data:`MAX_REDIRECTS` times. A document's
resources are not agent web access, so the ``WEB_ACCESS`` switch does not govern them
(as for the heartbeat webhook); the always-refused ranges and the operator's denylist
do, and the address connected to must also be globally routable (``is_global``), the
rule these fetches had before.
"""
from __future__ import annotations

import ipaddress
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from urllib.parse import urljoin

import httpx

from core.security.web_access import build_pinned_request, resolve_outbound

MAX_REDIRECTS = 3
FETCH_TIMEOUT_SECONDS = 10.0
# The whole fetch, every hop and every byte, ends by then: a server that trickles a
# byte at a time cannot hold a render past it (httpx's timeouts are per read).
FETCH_DEADLINE_SECONDS = 20.0
RAW_CHUNK_BYTES = 64 * 1024
# Bytes are read exactly as sent and counted before anything is kept: a compressed body
# could decode to far more than the cap in one chunk, so only an unencoded one is read.
IDENTITY_ENCODINGS = frozenset({"", "identity"})
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
USER_AGENT = "Automatos-DocGen"
DEFAULT_MIME = "application/octet-stream"


class FetchRefused(ValueError):
    """A URL a document may not fetch, or a fetch that failed; the message says which."""


@dataclass(frozen=True)
class Fetched:
    data: bytes
    mime_type: str
    url: str          # the URL the bytes came from, after any checked redirects


def _client() -> httpx.Client:
    """The HTTP client (a seam: tests give it a mock transport). Never follows redirects."""
    return httpx.Client(follow_redirects=False, timeout=FETCH_TIMEOUT_SECONDS)


ClientFactory = Callable[[], httpx.Client]


def _checked_target(url: str):
    target = resolve_outbound(url, enforce_switch=False)
    if not target.ok:
        raise FetchRefused(f"Blocked document URL: {target.reason}")
    if not ipaddress.ip_address(target.ip).is_global:
        raise FetchRefused(f"Blocked non-public address in document URL: {target.host!r}")
    return target


def _read_capped(response: httpx.Response, max_bytes: int, deadline: float) -> bytes:
    """The body as sent (never decompressed), refused past ``max_bytes`` or the deadline."""
    encoding = (response.headers.get("content-encoding") or "").strip().lower()
    if encoding not in IDENTITY_ENCODINGS:
        raise FetchRefused(f"Document resource sent {encoding!r}-encoded; only an unencoded body is read")
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise FetchRefused(f"Document resource is larger than {max_bytes} bytes")
    if response.is_stream_consumed:  # a transport that loaded the body already: its bytes, capped
        data = response.content
        if len(data) > max_bytes:
            raise FetchRefused(f"Document resource is larger than {max_bytes} bytes")
        return data
    chunks, size = [], 0
    for chunk in response.iter_raw(RAW_CHUNK_BYTES):
        size += len(chunk)
        if size > max_bytes:
            raise FetchRefused(f"Document resource is larger than {max_bytes} bytes")
        if time.monotonic() > deadline:
            raise FetchRefused(f"Document resource took longer than {FETCH_DEADLINE_SECONDS:g} s")
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_public(url: str, *, max_bytes: int, client_factory: Optional[ClientFactory] = None) -> Fetched:
    """GET ``url`` from the address it was checked at, each redirect hop checked again."""
    try:
        return _fetch(url, max_bytes, client_factory or _client)
    except httpx.HTTPError as exc:
        raise FetchRefused(f"Document URL could not be fetched: {type(exc).__name__}") from None


def _fetch(url: str, max_bytes: int, client_factory: ClientFactory) -> Fetched:
    deadline = time.monotonic() + FETCH_DEADLINE_SECONDS
    with client_factory() as client:
        for _hop in range(MAX_REDIRECTS + 1):
            if time.monotonic() > deadline:
                raise FetchRefused(f"Document URL took longer than {FETCH_DEADLINE_SECONDS:g} s")
            fetched, url = _one_hop(client, url, max_bytes, deadline)
            if fetched is not None:
                return fetched
    raise FetchRefused(f"More than {MAX_REDIRECTS} redirects")


def _one_hop(client: httpx.Client, url: str, max_bytes: int, deadline: float) -> Tuple[Optional[Fetched], str]:
    """``(the bytes, url)`` for an answer, or ``(None, the next url)`` for a redirect; each
    hop is resolved, checked and sent to the address it was checked at."""
    target = _checked_target(url)
    request = build_pinned_request(client, "GET", url, target,
                                   headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    response = client.send(request, stream=True)
    try:
        if response.status_code in REDIRECT_CODES:
            location = response.headers.get("location")
            if not location:
                raise FetchRefused(f"Redirect with no Location from {target.host!r}")
            return None, urljoin(url, location)
        if response.status_code >= 400:
            raise FetchRefused(f"Document URL answered {response.status_code}: {target.host!r}")
        data = _read_capped(response, max_bytes, deadline)
        mime = (response.headers.get("content-type") or DEFAULT_MIME).split(";")[0].strip()
        return Fetched(data, mime or DEFAULT_MIME, url), url
    finally:
        response.close()


__all__ = ["FETCH_DEADLINE_SECONDS", "FETCH_TIMEOUT_SECONDS", "FetchRefused", "Fetched", "MAX_REDIRECTS", "fetch_public"]
