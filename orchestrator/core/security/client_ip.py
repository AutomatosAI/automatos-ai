"""The caller's address, read one way everywhere: every per-IP limit, the audit IPs, the buckets.

The first ``X-Forwarded-For`` hop is whatever the client wrote, unless a proxy in front replaces the
header, so a per-IP limit keyed on it can be dodged by rotating values, or aimed at someone else's
address. A deployment behind a proxy that *sets* a client-IP header names it in
``CLIENT_IP_HEADER``: hosted Automatos on Railway uses ``X-Real-IP``, which Railway's edge sets
"for identifying client's remote IP". Then that header is the address, the socket peer when it is
missing or not an IP address, and ``X-Forwarded-For`` is never read.

``TrustedClientAddressMiddleware`` puts the same address in the ASGI scope's ``client``, so code that
reads the socket peer (the widget limiter) reads the caller too.

With ``CLIENT_IP_HEADER`` unset, each caller reads the address as it always has: a route takes the
first ``X-Forwarded-For`` hop, else the socket peer; the widget limiter takes the socket peer.
Nothing changes until it is set.
"""
from __future__ import annotations

import ipaddress
from typing import Any, Mapping, Optional

UNKNOWN = "unknown"
FORWARDED_FOR = "x-forwarded-for"


def _configured_header() -> Optional[str]:
    """The header the operator's proxy sets, lower-case; None when unset."""
    from config import config

    name = (getattr(config, "CLIENT_IP_HEADER", None) or "").strip().lower()
    return name or None


def _first_value(raw: Optional[str]) -> Optional[str]:
    """A header's first comma-separated value, trimmed; None when empty."""
    value = (raw or "").split(",")[0].strip()
    return value or None


def _configured_address(raw: Optional[str]) -> Optional[str]:
    """The configured header's first value when it is an IP address; None when missing or not one,
    so a junk value (``garbage``, ``1.2.3.4:80``) falls back to the peer instead of naming a bucket."""
    value = _first_value(raw)
    if value is None:
        return None
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return None


def client_ip_of(headers: Mapping[str, str], peer: Optional[str], *, forwarded_hop: bool) -> str:
    """The address from lower-cased ``headers`` and the socket ``peer``: the configured header, else
    (unset) the first ``X-Forwarded-For`` hop when ``forwarded_hop``, else the peer."""
    configured = _configured_header()
    if configured:
        return _configured_address(headers.get(configured)) or peer or UNKNOWN
    if forwarded_hop:
        return _first_value(headers.get(FORWARDED_FOR)) or peer or UNKNOWN
    return peer or UNKNOWN


def client_ip(request: Any) -> str:
    """A route's caller: what ``api.credentials.get_client_ip`` and slowapi read."""
    headers = {k.lower(): v for k, v in request.headers.items()}
    peer = request.client.host if getattr(request, "client", None) else None
    return client_ip_of(headers, peer, forwarded_hop=True)


class TrustedClientAddressMiddleware:
    """Pure ASGI: when ``CLIENT_IP_HEADER`` names a header, the scope's ``client`` becomes the
    address in it, so everything that reads the socket peer (the widget limiter, ``request.client``)
    reads the caller, not the proxy. Unset, or the header missing or not an IP address, the scope is
    left as it is.
    Registered outermost in ``main.py``, as uvicorn's own proxy-headers middleware would be."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        configured = _configured_header() if scope.get("type") in ("http", "websocket") else None
        if configured:
            raw = dict(scope.get("headers", [])).get(configured.encode("latin-1"))
            address = _configured_address(raw.decode("latin-1") if raw else None)
            if address:
                client = scope.get("client")
                scope = {**scope, "client": (address, client[1] if client else 0)}
        await self.app(scope, receive, send)
