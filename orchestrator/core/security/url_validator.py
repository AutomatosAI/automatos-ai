"""
URL Validator — SSRF prevention for outbound webhook/HTTP calls.

Validates URLs against:
- Allowed schemes (http/https only)
- Private, loopback, link-local, and reserved IP ranges
- DNS resolution to catch hostname-based bypasses
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Blocked networks (IPv4 + IPv6)
_BLOCKED_NETWORKS = [
    # IPv4
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    # CGNAT — cloud private networking (Railway's internal v4 side) and some
    # providers' metadata endpoints live here (2026-09-16 webhook review).
    ipaddress.ip_network("100.64.0.0/10"),
    # IPv6
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("::/128"),
    # IPv4-compatible (``::127.0.0.1``), deprecated by RFC 4291 — nothing
    # legitimate resolves to one, so the whole block is refused (F076).
    ipaddress.ip_network("::/96"),
]

# NAT64's well-known prefix (RFC 6052): ``64:ff9b::a.b.c.d`` is a.b.c.d.
_NAT64_WELL_KNOWN = ipaddress.ip_network("64:ff9b::/96")

_ALLOWED_SCHEMES = {"http", "https"}

# PRD-240: the same list, importable — web access pins its connection to an
# address it has checked against exactly these ranges.
BLOCKED_NETWORKS = tuple(_BLOCKED_NETWORKS)


def _embedded_ipv4(addr: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The v4 address an IPv6 literal carries: IPv4-mapped (``::ffff:10.0.0.1``),
    6to4 (``2002:0a00:0001::``) or NAT64 (``64:ff9b::10.0.0.1``)."""
    if addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    if addr.sixtofour is not None:
        return addr.sixtofour
    if addr in _NAT64_WELL_KNOWN:
        return ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
    return None


def blocked_network_for(ip_str: str) -> str | None:
    """The blocked range ``ip_str`` falls in, ``"invalid"`` for a non-address,
    or ``None`` when the address is routable."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return "invalid"
    # An IPv6 literal that carries a v4 address names that address, and stdlib
    # containment never crosses families — check the address it carries.
    if addr.version == 6:
        carried = _embedded_ipv4(addr)
        if carried is not None:
            addr = carried
    for network in _BLOCKED_NETWORKS:
        if addr.version == network.version and addr in network:
            return str(network)
    return None


def validate_webhook_url(url: str) -> tuple[bool, str]:
    """
    Validate a webhook URL is safe for outbound HTTP POST.

    Returns:
        (is_valid, reason) — True if safe, False with reason if blocked.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Malformed URL"

    # Scheme check
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"Disallowed scheme: {parsed.scheme}"

    hostname = parsed.hostname
    if not hostname:
        return False, "No hostname in URL"

    # Resolve DNS to get actual IP(s). ``parsed.port`` raises ValueError on a
    # malformed port ("example.com:abc") — a refusal, never an exception.
    try:
        port = parsed.port or 443
    except ValueError:
        return False, "Malformed port in URL"
    try:
        addrinfos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError, ValueError):
        return False, f"DNS resolution failed for {hostname}"

    for family, _, _, _, sockaddr in addrinfos:
        ip_str = sockaddr[0]
        blocked = blocked_network_for(ip_str)
        if blocked == "invalid":
            return False, f"Invalid IP from DNS: {ip_str}"
        if blocked:
            return False, f"Resolved to blocked range ({blocked})"

    return True, "OK"
