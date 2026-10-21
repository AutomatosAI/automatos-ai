"""PRD-266 K15 — the two addresses a phone is handed: the API's and the web app's.

* ``web_url`` is ``config.FRONTEND_URL``.
* ``api_url`` is ``https://<PUBLIC_API_HOST>`` in saas, and on a local install only once
  the operator has named their own host. While ``PUBLIC_API_HOST`` is still the hosted
  default (``HOSTED_PUBLIC_API_HOST`` in ``config.py``) a local install answers with the request's
  own base URL: a self-hosted server never advertises the hosted API.
* Both must be http(s) URLs with a host, so the pair link built on them
  (``services.mobile.pairing``) is never a custom scheme; anything else is a
  misconfiguration and raises ``MobileUrlError`` naming its source.

Discovery (US-001) reads the same two helpers, so a QR code and the discovery document
always name the same server.
"""
from __future__ import annotations

from urllib.parse import urlsplit

from config import HOSTED_PUBLIC_API_HOST, config

HTTPS_SCHEME = "https"
WEB_SCHEMES = ("http", HTTPS_SCHEME)


class MobileUrlError(RuntimeError):
    """An address a phone would be handed is not an http(s) URL with a host."""


def _web_address(url: str, source: str) -> str:
    """``url`` without a trailing slash; refused unless it is http(s) with a host."""
    trimmed = (url or "").strip().rstrip("/")
    parts = urlsplit(trimmed)
    if parts.scheme.lower() not in WEB_SCHEMES or not parts.netloc:
        raise MobileUrlError(f"{source} must be an http(s) URL with a host")
    return trimmed


def web_url() -> str:
    """The web app's address (``FRONTEND_URL``): the host of every pair link."""
    return _web_address(config.FRONTEND_URL, "FRONTEND_URL")


def api_url(request_base_url: str) -> str:
    """The API's address as a phone should call it (K15).

    ``request_base_url`` is the base URL the caller reached this API on; it is used only
    on a local install that has not set ``PUBLIC_API_HOST``.
    """
    host = config.PUBLIC_API_HOST
    if config.IS_LOCAL_EDITION and host == HOSTED_PUBLIC_API_HOST:
        return _web_address(request_base_url, "the request's base URL")
    return _web_address(f"{HTTPS_SCHEME}://{host}", "PUBLIC_API_HOST")

