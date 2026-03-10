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
    # IPv6
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("::/128"),
]

_ALLOWED_SCHEMES = {"http", "https"}


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

    # Resolve DNS to get actual IP(s)
    try:
        addrinfos = socket.getaddrinfo(hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False, f"DNS resolution failed for {hostname}"

    for family, _, _, _, sockaddr in addrinfos:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"Invalid IP from DNS: {ip_str}"

        for network in _BLOCKED_NETWORKS:
            if addr in network:
                return False, f"Resolved to blocked range ({network})"

    return True, "OK"
