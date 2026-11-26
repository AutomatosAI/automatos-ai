"""The caller's address, read one way everywhere: every per-IP limit, the audit IPs, the buckets.

Behind a proxy, name the header it writes the caller's address to in ``CLIENT_IP_HEADER``. Hosted
Automatos on Railway uses ``X-Real-IP``, which Railway's edge sets "for identifying client's remote
IP". If your proxies append to the header instead of setting it (an AWS application load balancer
appends to ``X-Forwarded-For``), say how many of your own proxies append to it in
``CLIENT_IP_PROXY_COUNT``: 1 for one load balancer, 2 for a content network in front of one. The
address is then the value your outermost proxy wrote, that many from the header's end, and the
values before it are never read. Both settings' rules are ``core.client_ip_settings``.

The count is for proxies that append (an AWS load balancer's default mode). One that replaces the
header with the one address it saw is the only value there: count it as 1, or name a header it
sets. One that passes the header on unchanged adds nothing: do not count it.

With ``CLIENT_IP_HEADER`` named:

* no count: the first value of the header's last line;
* a count: the header's lines are one list, in order (RFC 9110 §5.3), empty items are skipped, and
  the value that many from the end is the address. It may carry a port, and an IPv6 address may be
  in brackets, as an AWS load balancer writes them; a value with a zone is not an address. At most
  ``ITEMS_EXAMINED_MAX`` items from the end are looked at;
* either way, the socket peer when the header is missing, holds fewer values than the count, or the
  value read is not an IP address. No other value is ever tried in its place. The first request that
  carries the header with fewer values than the count logs one warning for the process;
* an IPv4 address written as IPv4-mapped IPv6 (``::ffff:192.0.2.1``) is read as the IPv4 address;
* a count that is not a plain whole number from 1 to ``PROXY_COUNT_MAX``: ignored, the address read
  as if it were unset, with one error in the log at start. The ``enterprise`` edition refuses to
  start instead (``core.edition_start_up``).

``TrustedClientAddressMiddleware`` puts the same address in the ASGI scope's ``client``, so code that
reads the socket peer (the widget limiter) reads the caller too.

With ``CLIENT_IP_HEADER`` unset, each caller reads the address as it always has: a route takes the
first ``X-Forwarded-For`` hop of its last line, else the socket peer; the widget limiter takes the
socket peer. A count without a header changes nothing, and is logged at start. Each log line here is
written at most once per process, and names a setting, never a value or an address.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import threading
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

# The once-per-process log lines already written, by key. Process state on purpose: a line that
# names a misconfiguration is worth one entry, not one per request.
_LOGGED_ONCE_LOCK = threading.Lock()
_logged_once: set = set()
UNUSABLE_COUNT_LOG = "unusable-count"
FEWER_VALUES_LOG = "fewer-values"
FEWER_VALUES_WARNING = (
    "%s carried fewer values than %s on a request, so its caller's address was read from the "
    "connection. Set the count to how many of your own proxies append to the header. Logged once "
    "per process; later requests fall back the same way without a log line."
)
UNUSABLE_COUNT_ERROR = "%s %s. It is ignored until this is fixed: the address is read as if it were unset."


def _log_once(key: str, level: int, message: str, *args: Any) -> None:
    """Write ``message`` the first time ``key`` is logged in this process, and never again."""
    with _LOGGED_ONCE_LOCK:
        if key in _logged_once:
            return
        _logged_once.add(key)
    logger.log(level, message, *args)


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
    junk value (``garbage``, ``1.2.3.4:80``) falls back to the peer instead of naming a bucket.

    The one place an address is made canonical: an IPv4-mapped IPv6 address is its IPv4 address
    (``IPv6Address.ipv4_mapped``), so one client is one text for every consumer."""
    if value is None:
        return None
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return None
    return str(getattr(address, "ipv4_mapped", None) or address)


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
        inner = bracketed.group(1)  # brackets hold IPv6 only, so an IPv4 address inside is refused
        return _plain_address(inner) if ":" in inner and _port_is_valid(bracketed.group(2)) else None
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
    one is not an address. No value before it is ever read. A header that is there with fewer
    values than the count logs one warning for the process."""
    value = next(islice(_values_from_the_end(lines), count - 1, None), None)
    if value is None:
        if lines:
            _log_once(FEWER_VALUES_LOG, logging.WARNING, FEWER_VALUES_WARNING,
                      client_ip_settings.HEADER_SETTING, client_ip_settings.COUNT_SETTING)
        return None
    return _appended_address(value)


def _usable_count(count_setting: Any) -> Optional[int]:
    """The count, or None when unset or not a count: a malformed count is ignored, never read as
    another count (``enterprise`` refuses one at start; the log line is ``_log_an_unusable_count``)."""
    try:
        return client_ip_settings.proxy_count(count_setting)
    except client_ip_settings.MalformedProxyCount:
        return None


def _header_address(lines: Sequence[str], count_setting: Any) -> Optional[str]:
    """The address the configured header's ``lines`` hold, or None when the peer is the answer."""
    count = _usable_count(count_setting)
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
    """At start, once per process: an error naming the count's problem, never its value.
    ``enterprise`` then refuses to start; every other edition starts, reading the address as if the
    count were unset."""
    from config import config

    problem = client_ip_settings.count_problem(getattr(config, "CLIENT_IP_HEADER", None),
                                               getattr(config, "CLIENT_IP_PROXY_COUNT", None))
    if problem:
        _log_once(UNUSABLE_COUNT_LOG, logging.ERROR, UNUSABLE_COUNT_ERROR, client_ip_settings.COUNT_SETTING, problem)


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
