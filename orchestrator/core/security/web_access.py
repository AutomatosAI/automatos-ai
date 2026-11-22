"""PRD-240 — the ONE place that decides whether an agent may reach a URL, and
WHICH address it may connect to.

Two layers, deliberately separate:

* **Always refused, not configurable** — private, loopback, link-local and
  metadata ranges (``url_validator.BLOCKED_NETWORKS``), checked against EVERY
  address the hostname resolves to. This keeps an agent out of
  ``postgres:5432``, ``minio:9000``, ``host.docker.internal`` and
  ``169.254.169.254`` on the compose network.
* **The operator's denylist** — ``WEB_ACCESS_DENY``, suffix-matched hosts. Adds
  to the first layer; can never widen it.

The check is resolve-and-pin, not validate-then-reconnect: ``resolve_outbound``
returns the address it checked, and ``web_fetch`` connects to exactly that
address (with the hostname as ``Host`` and SNI), so a DNS answer that changes
between the check and the connection cannot redirect the request into a
blocked range.

``WEB_ACCESS`` is the switch: on by default in the local edition, off in saas
unless set. Both the platform actions (``handlers_web``) and the provider-side
search tool (``openai_compatible_client``) ask this module.

A destination a workspace user saves (a channel webhook, a credential's database,
Redis, SSH host or API URL) is not agent web access, so the switch never governs it.
Where several people share the server (``editions.current().multi_person``), one
workspace's admin must not reach the server's own network, so it goes through the
same ranges, denylist and pin (``operator_destination_async``,
``operator_host_async``). On a single-operator install the operator is reaching
their own machine or network on purpose, and the destination is used as given.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from config import config
from core import editions
from core.security.url_validator import blocked_network_for

WEB_ACCESS_OFF_REASON = (
    "Web access is off on this server (WEB_ACCESS=off). Set WEB_ACCESS=on in "
    ".env and restart the backend to let agents read and search the web."
)
_ALLOWED_SCHEMES = ("http", "https")
MAX_PORT = 65535

# The resolver seam — tests replace it; production is the system resolver.
_getaddrinfo = socket.getaddrinfo


@dataclass(frozen=True)
class OutboundTarget:
    ok: bool
    reason: str
    host: str = ""
    ip: str = ""       # the address that was checked — connect to THIS
    port: int = 0
    scheme: str = ""

    @property
    def pinned(self) -> bool:
        """True when the request must go to ``ip``; False for a destination used as given."""
        return bool(self.ip)


def web_access_enabled() -> bool:
    return bool(config.WEB_ACCESS)


def denied_hosts() -> Tuple[str, ...]:
    return tuple(config.WEB_ACCESS_DENY or ())


def host_denied(host: Optional[str]) -> bool:
    """True when ``host`` is, or is under, a denied host (``example.com`` also
    covers ``www.example.com``; ``notexample.com`` is untouched)."""
    if not host:
        return False
    h = host.lower().rstrip(".")
    for denied in denied_hosts():
        if h == denied or h.endswith("." + denied):
            return True
    return False


def _resolve_addresses(host: str, port: int) -> Tuple[List[str], Optional[str]]:
    """Every address ``host`` resolves to, or a refusal reason."""
    try:
        infos = _getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError, ValueError):
        return [], f"DNS resolution failed for {host}"
    addresses: List[str] = []
    for family, _type, _proto, _canon, sockaddr in infos:
        ip = str(sockaddr[0])
        if ip not in addresses:
            addresses.append(ip)
    if not addresses:
        return [], f"DNS resolution failed for {host}"
    return addresses, None


def resolve_outbound(url: str, *, enforce_switch: bool = True) -> OutboundTarget:
    """Decide whether ``url`` may be fetched and pin the address to use.

    Order matters: the switch, then the scheme, then the denylist, then the
    port, then DNS + the private-range check on every answer — so a denied
    host is named as denied rather than surfacing a DNS failure, and a
    switched-off server never resolves anything.

    ``enforce_switch=False`` skips ONLY the ``WEB_ACCESS`` switch: an
    operator-configured destination (a heartbeat webhook) is not agent web
    access, so the switch does not govern it — the always-refused ranges and
    the operator denylist still do.
    """
    if enforce_switch and not web_access_enabled():
        return OutboundTarget(False, WEB_ACCESS_OFF_REASON)
    try:
        parsed = urlparse((url or "").strip())
    except Exception:  # noqa: BLE001 — a malformed URL is a refusal, not a crash
        return OutboundTarget(False, "Malformed URL")
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower().rstrip(".")
    if scheme not in _ALLOWED_SCHEMES:
        return OutboundTarget(False, f"Only http and https URLs can be fetched (got '{scheme or 'none'}')", host)
    if not host:
        return OutboundTarget(False, "No hostname in URL", host)
    if host_denied(host):
        return OutboundTarget(False, f"'{host}' is on this server's WEB_ACCESS_DENY list", host)
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError:
        return OutboundTarget(False, "Malformed port in URL", host)
    return _checked_address(host, port, scheme)


def _checked_address(host: str, port: int, scheme: str) -> OutboundTarget:
    """DNS, then the private-range check on EVERY answer, then the pin."""
    addresses, failure = _resolve_addresses(host, port)
    if failure:
        return OutboundTarget(False, f"'{host}' is not reachable from agents: {failure}", host)
    for ip in addresses:
        blocked = blocked_network_for(ip)
        if blocked == "invalid":
            return OutboundTarget(False, f"'{host}' is not reachable from agents: invalid address {ip}", host)
        if blocked:
            return OutboundTarget(
                False, f"'{host}' is not reachable from agents: resolves into a blocked range ({blocked})", host
            )
    # Prefer an IPv4 answer — the compose network and most home routes are v4.
    pinned = next((ip for ip in addresses if ":" not in ip), addresses[0])
    return OutboundTarget(True, "OK", host, pinned, port, scheme)


def resolve_host(host: str, port: int) -> OutboundTarget:
    """:func:`resolve_outbound` for a bare ``host:port`` (a database, Redis or SSH
    server): no scheme and no ``WEB_ACCESS`` switch, the same denylist, the same
    check on every resolved address, and the address to connect to."""
    name = (host or "").strip().strip("[]").lower().rstrip(".")
    if not name:
        return OutboundTarget(False, "No hostname given")
    if host_denied(name):
        return OutboundTarget(False, f"'{name}' is on this server's WEB_ACCESS_DENY list", name)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port <= MAX_PORT:
        return OutboundTarget(False, "Malformed port", name)
    return _checked_address(name, port, "")


# System DNS has no timeout of its own; a resolver that never answers must
# not hold a request (or a thread-pool slot) for longer than this.
RESOLVE_TIMEOUT_SECONDS = 5.0


def _host_of(url: str) -> str:
    """The URL's host, lower-cased, or "" when it has none or will not parse."""
    try:
        return (urlparse((url or "").strip()).hostname or "").lower()
    except ValueError:
        return ""


async def _off_the_loop(check, *args, host: str, **kwargs) -> OutboundTarget:
    """``check(*args, **kwargs)`` off the event loop and bounded: a resolver that
    never answers is a refusal, not a stalled request. (The worker thread finishes
    on its own; only the caller stops waiting.)"""
    import asyncio

    try:
        return await asyncio.wait_for(asyncio.to_thread(check, *args, **kwargs), timeout=RESOLVE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return OutboundTarget(False, f"DNS resolution timed out for {host or 'the URL'}", host)


async def resolve_outbound_async(url: str, *, enforce_switch: bool = True) -> OutboundTarget:
    """:func:`resolve_outbound` off the event loop and bounded (``RESOLVE_TIMEOUT_SECONDS``)."""
    return await _off_the_loop(resolve_outbound, url, host=_host_of(url), enforce_switch=enforce_switch)


def destinations_guarded() -> bool:
    """Whether a destination a workspace user saves goes through the guard: True where
    several people share the server (saas, enterprise), so one workspace's admin can't
    reach the server's own network; False on a single-operator install, whose operator
    may reach their own machine and network on purpose. The one place this is decided."""
    return editions.current().multi_person


async def operator_destination_async(url: str) -> OutboundTarget:
    """Whether a user-saved URL (a webhook, a credential's API) may be called, and where.

    Guarded (:func:`destinations_guarded`): :func:`resolve_outbound` without the
    ``WEB_ACCESS`` switch, so send to the pinned address and follow no redirect.
    Otherwise the URL is used as given (``pinned`` is False)."""
    if destinations_guarded():
        return await resolve_outbound_async(url, enforce_switch=False)
    return OutboundTarget(True, "OK", _host_of(url))


async def operator_host_async(host: str, port: int) -> OutboundTarget:
    """:func:`operator_destination_async` for a bare ``host:port`` (a database, Redis
    or SSH server): guarded, :func:`resolve_host` (connect to ``ip``); otherwise the
    host is used as given."""
    if destinations_guarded():
        return await _off_the_loop(resolve_host, host, port, host=host)
    return OutboundTarget(True, "OK", host, "", port)


def validate_outbound_url(url: str, *, enforce_switch: bool = True) -> Tuple[bool, str, str]:
    """(ok, reason, host) — the yes/no view of :func:`resolve_outbound`."""
    target = resolve_outbound(url, enforce_switch=enforce_switch)
    return target.ok, target.reason, target.host


def pinned_request_parts(url: str, target: OutboundTarget, headers=None) -> Dict[str, Any]:
    """The ``url``, ``headers`` and ``extensions`` of the request for ``url`` sent to
    the address ``target`` was checked at.

    The URL's host becomes the pinned IP; the real hostname rides as ``Host`` and as
    SNI (``sni_hostname``), so TLS is still verified against the name and a DNS
    answer that changes after the check changes nothing. A target that is not
    pinned (:func:`operator_destination_async` on a single-operator install) leaves
    the request as it was.
    """
    if not target.pinned:
        return {"url": url, "headers": dict(headers or {}), "extensions": {}}
    import httpx

    pinned_url = httpx.URL(url).copy_with(host=target.ip)  # httpx brackets IPv6 itself
    return {
        "url": str(pinned_url),
        "headers": {"Host": target.host, **(headers or {})},
        "extensions": {"sni_hostname": target.host},
    }


def build_pinned_request(client, method: str, url: str, target: OutboundTarget, **kwargs):
    """The httpx request for ``url`` sent to the address ``target`` was checked at
    (:func:`pinned_request_parts`). ``kwargs`` (``json=``, ``content=``, extra
    ``headers=`` …) go to ``client.build_request``. Shared by ``web_fetch``, the
    heartbeat webhook and the partner client — the one place the resolve-and-pin
    request is built.
    """
    parts = pinned_request_parts(url, target, kwargs.pop("headers", None))
    return client.build_request(method, parts["url"], headers=parts["headers"],
                                extensions=parts["extensions"], **kwargs)
