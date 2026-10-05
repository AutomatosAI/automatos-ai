"""An httpx client whose every request goes to the address it was checked at (#873).

``web_fetch`` and the heartbeat webhook pin one request at a time with
``web_access.build_pinned_request``. An SDK such as OpenAI's builds its own
requests, so for those the pin sits in the transport: each request's host is
resolved and checked by ``web_access.resolve_outbound`` (private, loopback,
link-local and metadata ranges, and ``WEB_ACCESS_DENY``), then sent to exactly
that address with the hostname kept as ``Host`` and SNI, so TLS is still verified
against the name. A DNS answer that changes after the check changes nothing, and
redirects are off, so a 3xx can't hand the request on to an unchecked address.
"""

from __future__ import annotations

import httpx

from core.security.web_access import resolve_outbound


class PinnedTransport(httpx.HTTPTransport):
    """Sends each request to the address ``resolve_outbound`` checked, never re-resolving."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        target = resolve_outbound(str(request.url), enforce_switch=False)
        if not target.ok:
            raise httpx.ConnectError(target.reason, request=request)
        pinned = httpx.Request(
            request.method,
            request.url.copy_with(host=target.ip),
            headers=request.headers,  # Host already carries the hostname
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": target.host},
        )
        return super().handle_request(pinned)


def pinned_client(timeout: float) -> httpx.Client:
    """A client for a user-supplied endpoint: pinned, and never following a redirect."""
    return httpx.Client(transport=PinnedTransport(), follow_redirects=False, timeout=timeout)


__all__ = ["PinnedTransport", "pinned_client"]
