"""The caller's address, read one way everywhere: every per-IP limit, the audit IPs, the buckets.

Behind a proxy, name the header it writes the caller's address to in ``CLIENT_IP_HEADER``. Hosted
Automatos on Railway uses ``X-Real-IP``, which Railway's edge sets "for identifying client's remote
IP". If your proxies append to the header instead of setting it (an AWS application load balancer
appends to ``X-Forwarded-For``), say how many of your own proxies append to it in
``CLIENT_IP_PROXY_COUNT``: 1 for one load balancer, 2 for a content network in front of one. The
address is then the value your outermost proxy wrote, that many from the header's end, and the
values before it are never read. Both settings' rules are ``core.client_ip_settings``.

With ``CLIENT_IP_HEADER`` named:

* no count: the first value of the header's last line;
* a count: the header's lines are one list, in order (RFC 9110 §5.3), empty items are skipped, and
  the value that many from the end is the address. It may carry a port, and an IPv6 address may be
  in brackets, as an AWS load balancer writes them; a value with a zone is not an address. At most
  ``ITEMS_EXAMINED_MAX`` items from the end are looked at;
* either way, the socket peer when the header is missing, holds fewer values than the count, or the
  value read is not an IP address. No other value is ever tried in its place;
* a count that is not a whole number from 1 to ``PROXY_COUNT_MAX``: the socket peer, with an error
  in the log at start. The ``enterprise`` edition refuses to start instead (``core.edition_start_up``).

``TrustedClientAddressMiddleware`` puts the same address in the ASGI scope's ``client``, so code that
reads the socket peer (the widget limiter) reads the caller too.

With ``CLIENT_IP_HEADER`` unset, each caller reads the address as it always has: a route takes the
first ``X-Forwarded-For`` hop of its last line, else the socket peer; the widget limiter takes the
socket peer. A count without a header changes nothing, and is logged at start.
"""
from __future__ import annotations

import ipaddress
import logging
import re
from itertools import islice
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

from core import client_ip_settings
from core.client_ip_settings import FORWARDED_FOR

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
# RFC 9110 §5.6.1.2: a recipient ignores "a reasonable number" of empty list elements. This many
# items, counted from the end, are the most one request's header is searched.
ITEMS_EXAMINED_MAX = 64
PORT_MAX = 65535
# The two forms with a port an AWS load balancer writes: "[2001:db8::1]:8080" and "192.0.2.1:8080".
_BRACKETED = re.compile(r"\[([0-9A-Fa-f:.]+)\](?::([0-9]{1,5}))?")
_IPV4_WITH_PORT = re.compile(r"([0-9.]+):([0-9]{1,5})")


def _settings() -> Tuple[Optional[str], Any]:
    """The configured header (lower-case; None when unset) and the count as configured."""
    from config import config

    return (client_ip_settings.header_name(getattr(config, "CLIENT_IP_HEADER", None)),
            getattr(config, "CLIENT_IP_PROXY_COUNT", None))


def _first_value(line: Optional[str]) -> Optional[str]:
    """A header line's first comma-separated value, trimmed; None when empty."""
    value = (line or "").partition(",")[0].strip()
    return value or None


def _plain_address(value: Optional[str]) -> Optional[str]:
    """``value`` when it is an IP address, in its canonical form; None when missing or not one, so a
    junk value (``garbage``, ``1.2.3.4:80``) falls back to the peer instead of naming a bucket."""
    if value is None:
        return None
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return None


def _port_is_valid(port: Optional[str]) -> bool:
    return port is None or int(port) <= PORT_MAX


def _appended_address(value: str) -> Optional[str]:
    """A value one of the deployment's proxies appended, as an address: a plain IP address, an IPv4
    address with a port, or an IPv6 address in brackets with or without one. A zone is not an
    address: it names an interface on the host that wrote it."""
    if "%" in value:
        return None
    bracketed = _BRACKETED.fullmatch(value)
    if bracketed:
        address = _plain_address(bracketed.group(1)) if _port_is_valid(bracketed.group(2)) else None
        return address if address and ":" in address else None
    with_port = _IPV4_WITH_PORT.fullmatch(value)
    if with_port:
        return _plain_address(with_port.group(1)) if _port_is_valid(with_port.group(2)) else None
    return _plain_address(value)


def _values_from_the_end(lines: Sequence[str]) -> Iterator[str]:
    """The non-empty values of the list ``lines`` make, last first, from at most
    ``ITEMS_EXAMINED_MAX`` items: the work is bounded whatever the header's length."""
    examined = 0
    for line in reversed(lines):
        # At most one piece more than the items still allowed; that piece is never examined.
        for piece in reversed(line.rsplit(",", ITEMS_EXAMINED_MAX - examined)):
            examined += 1
            if examined > ITEMS_EXAMINED_MAX:
                return
            if piece.strip():
                yield piece.strip()


def _counted_address(lines: Sequence[str], count: int) -> Optional[str]:
    """The value ``count`` from the end, as an address; None when there are fewer values or that
    one is not an address. No value before it is ever read."""
    value = next(islice(_values_from_the_end(lines), count - 1, None), None)
    return _appended_address(value) if value is not None else None


def _header_address(lines: Sequence[str], count_setting: Any) -> Optional[str]:
    """The address the configured header's ``lines`` hold, or None when the peer is the answer
    (a malformed count among those: it is never read as some other count)."""
    try:
        count = client_ip_settings.proxy_count(count_setting)
    except client_ip_settings.MalformedProxyCount:
        return None
    if count is None:
        return _plain_address(_first_value(lines[-1])) if lines else None
    return _counted_address(lines, count)


def _lines(headers: Iterable[Tuple[str, str]], name: str) -> List[str]:
    """Each line of header ``name`` (lower-case), in the order received."""
    return [value for key, value in headers if key.lower() == name]


def client_ip_of(headers: Iterable[Tuple[str, str]], peer: Optional[str], *, forwarded_hop: bool) -> str:
    """The address from ``headers`` (every ``(name, value)`` line, in the order received) and the
    socket ``peer``: the configured header, else (unset) the first ``X-Forwarded-For`` hop of its
    last line when ``forwarded_hop``, else the peer."""
    header, count = _settings()
    if header:
        return _header_address(_lines(headers, header), count) or peer or UNKNOWN
    if forwarded_hop:
        lines = _lines(headers, FORWARDED_FOR)
        return (_first_value(lines[-1]) if lines else None) or peer or UNKNOWN
    return peer or UNKNOWN


def client_ip(request: Any) -> str:
    """A route's caller: what ``api.credentials.get_client_ip`` and slowapi read. Starlette's
    ``headers.items()`` gives every line, a repeated one included, in the order received."""
    peer = request.client.host if getattr(request, "client", None) else None
    return client_ip_of(request.headers.items(), peer, forwarded_hop=True)


def _log_an_unusable_count() -> None:
    """At start: an error naming the count's problem, never its value. ``enterprise`` then refuses
    to start; every other edition starts, reading no count."""
    from config import config

    problem = client_ip_settings.count_problem(getattr(config, "CLIENT_IP_HEADER", None),
                                               getattr(config, "CLIENT_IP_PROXY_COUNT", None))
    if problem:
        logger.error("%s %s. The count is not read until this is fixed, and while %s is named every "
                     "caller's address is the socket peer.", client_ip_settings.COUNT_SETTING, problem,
                     client_ip_settings.HEADER_SETTING)


class TrustedClientAddressMiddleware:
    """Pure ASGI: when ``CLIENT_IP_HEADER`` names a header, the scope's ``client`` becomes the
    address read from it as ``client_ip`` reads it, so everything that reads the socket peer (the
    widget limiter, ``request.client``) reads the caller, not the proxy. Unset, or no address in
    the header, the scope is left as it is.
    Registered outermost in ``main.py``, as uvicorn's own proxy-headers middleware would be."""

    def __init__(self, app: Any) -> None:
        self.app = app
        _log_an_unusable_count()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        header, count = _settings() if scope.get("type") in ("http", "websocket") else (None, None)
        if header:
            key = header.encode("latin-1")
            lines = [value.decode("latin-1") for name, value in scope.get("headers", []) if name.lower() == key]
            address = _header_address(lines, count)
            if address:
                client = scope.get("client")
                scope = {**scope, "client": (address, client[1] if client else 0)}
        await self.app(scope, receive, send)
