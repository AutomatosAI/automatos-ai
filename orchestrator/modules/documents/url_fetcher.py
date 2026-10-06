"""The URL fetcher every WeasyPrint render uses (PRD-156 S4: SSRF).

Moved out of ``generation_service`` (night 10c) so that file stops growing; the
service imports it back, so ``generation_service._safe_url_fetcher`` still names it.
"""
from __future__ import annotations

from urllib.parse import urlparse

from modules.documents.pinned_fetch import fetch_public

ALLOWED_NETWORK_SCHEMES = ("http", "https")
INLINE_SCHEME = "data"


# What a template may pull into one PDF over http(s) (an image, a stylesheet, a font).
MAX_FETCH_BYTES = 20 * 1024 * 1024


def _safe_url_fetcher(url, *args, **kwargs):
    """WeasyPrint URL fetcher that blocks file:// and internal/non-public network
    targets from user-controlled templates (PRD-156 S4 — SSRF).

    Inline ``data:`` URIs (embedded chart images, the bundled and uploaded fonts) are
    allowed; ``http(s)`` is fetched only from a public address, and (F375) from the
    very address that was checked, with every redirect hop checked again
    (``pinned_fetch``): WeasyPrint's own fetcher resolved the host again and followed
    redirects, so a rebinding answer or a 302 to 127.0.0.1 got past the check.
    Everything else (file://, private/loopback/link-local hosts such as 10.x / 127.x /
    169.254.x, the cloud metadata endpoint) is refused.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme == INLINE_SCHEME:
        from weasyprint import default_url_fetcher
        return default_url_fetcher(url, *args, **kwargs)
    if scheme not in ALLOWED_NETWORK_SCHEMES:
        raise ValueError(f"Blocked non-http(s) URL scheme in template: {scheme!r}")
    fetched = fetch_public(url, max_bytes=MAX_FETCH_BYTES)
    return {"string": fetched.data, "mime_type": fetched.mime_type, "redirected_url": fetched.url}


__all__ = ["_safe_url_fetcher"]
