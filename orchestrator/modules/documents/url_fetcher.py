"""The URL fetcher every WeasyPrint render uses (PRD-156 S4: SSRF).

Moved out of ``generation_service`` (night 10c) so that file stops growing; the
service imports it back, so ``generation_service._safe_url_fetcher`` still names it.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_NETWORK_SCHEMES = ("http", "https")
INLINE_SCHEME = "data"


def _safe_url_fetcher(url, *args, **kwargs):
    """WeasyPrint URL fetcher that blocks file:// and internal/non-public network
    targets from user-controlled templates (PRD-156 S4 — SSRF).

    Inline ``data:`` URIs (embedded chart images, the bundled and uploaded fonts) are
    allowed; ``http(s)`` is allowed only to PUBLIC addresses; everything else —
    file://, and private/loopback/link-local hosts such as 10.x / 127.x / 169.254.x
    (the cloud metadata endpoint) — is refused.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme == INLINE_SCHEME:
        from weasyprint import default_url_fetcher
        return default_url_fetcher(url, *args, **kwargs)
    if scheme not in ALLOWED_NETWORK_SCHEMES:
        raise ValueError(f"Blocked non-http(s) URL scheme in template: {scheme!r}")
    host = parsed.hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ValueError(f"Cannot resolve template URL host: {host!r}") from None
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError(f"Blocked non-public address in template URL: {host!r}")
    from weasyprint import default_url_fetcher
    return default_url_fetcher(url, *args, **kwargs)


__all__ = ["_safe_url_fetcher"]
