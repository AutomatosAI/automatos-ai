"""PRD-270 K8, K13, K15 — the catalogue's paths, and the Automatos Cloud origin a local install reads them from.

Every URL in the catalogue is a PATH on the origin it was fetched from: the hosted edition has no
setting that names its own public origin, so the reader resolves each path against the origin it
fetched the catalogue from. A file of a listed version is
``/api/marketplace/catalog/files/<digest>/<path>``, and its package is ``package.autotheme`` there.

A local install reads that origin from ``AUTOMATOS_CLOUD_URL`` only (a deploy-time setting): an
ORIGIN, ``https://`` to any host or ``http://`` to a loopback one, with no user name or password and
no path, query or fragment (a lone trailing ``/`` is allowed and dropped). ``origin_problem`` is that
one rule; ``fetch.checked_origin`` says it in plain sentences, and ``cloud_file_url`` answers None
while the setting isn't such an origin.

A leaf: ``config`` and the standard library only (``listings`` imports it, and ``catalog`` imports
``listings``).
"""

from __future__ import annotations

import ipaddress
from typing import Optional
from urllib.parse import urlsplit

from config import config

URL_SETTING = "AUTOMATOS_CLOUD_URL"
CATALOG_ROUTE = "/api/marketplace/catalog"
CATALOG_FILES_ROUTE = f"{CATALOG_ROUTE}/files"
PACKAGE_FILE = "package.autotheme"

HTTPS_SCHEME, HTTP_SCHEME = "https", "http"
LOOPBACK_NAMES = frozenset({"localhost"})
ORIGIN_PATHS = frozenset({"", "/"})
# Why a setting is not an origin a Sync may use (``fetch`` words each one).
ORIGIN_OFF, ORIGIN_NOT_HTTPS, ORIGIN_CREDENTIALS, ORIGIN_NOT_ORIGIN = "off", "not_https", "credentials", "not_origin"


def catalog_file_path(digest: str, path: str) -> str:
    """Where the catalogue's files route serves ``path`` of the version ``digest``."""
    return f"{CATALOG_FILES_ROUTE}/{digest}/{path}"


def package_path(digest: str) -> str:
    """Where the catalogue's files route serves the version ``digest`` as one ``.autotheme``."""
    return catalog_file_path(digest, PACKAGE_FILE)


def configured_url() -> str:
    """``AUTOMATOS_CLOUD_URL``, trimmed; empty turns the Sync off."""
    return str(getattr(config, URL_SETTING, "") or "").strip()


def _is_loopback(host: str) -> bool:
    if host.lower() in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _allowed_target(scheme: str, host: Optional[str]) -> bool:
    """https to any host, http only to a loopback one."""
    if not host:
        return False
    return scheme == HTTPS_SCHEME or (scheme == HTTP_SCHEME and _is_loopback(host))


def origin_problem(url: str) -> Optional[str]:
    """Why ``url`` is not an origin a Sync may use (one of the ``ORIGIN_*`` reasons), or None when it is."""
    if not url:
        return ORIGIN_OFF
    try:
        parts = urlsplit(url)
        host, _port = parts.hostname, parts.port  # a malformed IPv6 host or port raises here
    except ValueError:
        return ORIGIN_NOT_HTTPS
    if parts.username is not None or parts.password is not None:
        return ORIGIN_CREDENTIALS
    if not _allowed_target(parts.scheme, host):
        return ORIGIN_NOT_HTTPS
    if parts.path not in ORIGIN_PATHS or parts.query or parts.fragment or url.endswith(("?", "#")):
        return ORIGIN_NOT_ORIGIN
    return None


def origin_of(url: str) -> str:
    """``scheme://host[:port]`` of a URL ``origin_problem`` passed."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def configured_origin() -> Optional[str]:
    """The Automatos Cloud origin this install syncs from, or None while the setting is empty or not an origin."""
    url = configured_url()
    return None if origin_problem(url) is not None else origin_of(url)


def catalog_url(origin: str) -> str:
    """The catalogue on ``origin``."""
    return f"{origin}{CATALOG_ROUTE}"


def package_url(origin: str, digest: str) -> str:
    """The version ``digest``'s package on ``origin``, derived from the setting and the digest (never a catalogue's)."""
    return f"{origin}{package_path(digest)}"


def cloud_file_url(digest: str, path: str) -> Optional[str]:
    """Where the browser loads ``path`` of the Cloud version ``digest``, or None while the setting is off."""
    origin = configured_origin()
    return None if origin is None else f"{origin}{catalog_file_path(digest, path)}"


__all__ = [
    "CATALOG_FILES_ROUTE", "CATALOG_ROUTE", "ORIGIN_CREDENTIALS", "ORIGIN_NOT_HTTPS", "ORIGIN_NOT_ORIGIN", "ORIGIN_OFF",
    "PACKAGE_FILE", "URL_SETTING", "catalog_file_path", "catalog_url", "cloud_file_url", "configured_origin",
    "configured_url", "origin_of", "origin_problem", "package_path", "package_url",
]
