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
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from config import config
from core.security.url_validator import blocked_network_for

WEB_ACCESS_OFF_REASON = (
    "Web access is off on this server (WEB_ACCESS=off). Set WEB_ACCESS=on in "
    ".env and restart the backend to let agents read and search the web."
)
_ALLOWED_SCHEMES = ("http", "https")

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


# System DNS has no timeout of its own; a resolver that never answers must
# not hold a request (or a thread-pool slot) for longer than this.
RESOLVE_TIMEOUT_SECONDS = 5.0


async def resolve_outbound_async(url: str, *, enforce_switch: bool = True) -> OutboundTarget:
    """:func:`resolve_outbound` off the event loop and bounded: a resolver that
    never answers is a refusal, not a stalled request. (The worker thread
    finishes on its own; only the caller stops waiting.)"""
    import asyncio

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(resolve_outbound, url, enforce_switch=enforce_switch),
            timeout=RESOLVE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        try:
            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:  # noqa: BLE001 — the URL is already suspect
            host = ""
        return OutboundTarget(False, f"DNS resolution timed out for {host or 'the URL'}", host)


def validate_outbound_url(url: str, *, enforce_switch: bool = True) -> Tuple[bool, str, str]:
    """(ok, reason, host) — the yes/no view of :func:`resolve_outbound`."""
    target = resolve_outbound(url, enforce_switch=enforce_switch)
    return target.ok, target.reason, target.host


def build_pinned_request(client, method: str, url: str, target: OutboundTarget, **kwargs):
    """The request for ``url`` sent to the address ``target`` was checked at.

    The URL's host becomes the pinned IP; the real hostname rides as ``Host``
    and as SNI (``sni_hostname``), so TLS is still verified against the name
    and a DNS answer that changes after the check changes nothing. ``kwargs``
    (``json=``, ``content=``, extra ``headers=`` …) go to
    ``client.build_request``. Shared by ``web_fetch`` and the heartbeat
    webhook — the one place the resolve-and-pin request is built.
    """
    import httpx

    pinned_url = httpx.URL(url).copy_with(host=target.ip)  # httpx brackets IPv6 itself
    headers = {"Host": target.host, **(kwargs.pop("headers", None) or {})}
    return client.build_request(
        method,
        pinned_url,
        headers=headers,
        extensions={"sni_hostname": target.host},
        **kwargs,
    )
