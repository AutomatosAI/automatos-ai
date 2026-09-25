"""Media inputs come only from OUR storage (PRD-251 D3, D9).

A render bundle names its footage, stills, voice files and SFX as presigned GET
URLs. The service fetches a URL only when it falls under one of the configured
storage prefixes (``MEDIA_RENDER_MEDIA_URL_PREFIXES``): the same scheme, host
and port, and a path under the prefix's path. Anything else is refused before
any fetch, so a bundle can never point the renderer at an arbitrary host.

Presigned URLs carry their signature in the query string, so ``redact`` is the
only form of a URL that reaches a log line or an error message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Tuple
from urllib.parse import SplitResult, urlsplit

DEFAULT_PORTS = {"http": 80, "https": 443}

# Dot segments and encoded separators: an HTTP client normalises them away, so
# a path that starts under the prefix could land outside it once fetched.
_UNSAFE_PATH = re.compile(r"(?:^|/)\.{1,2}(?:/|$)|%2e|%2f|%5c", re.IGNORECASE)
_UNSAFE_CHARS = re.compile(r"[\\\s\x00-\x1f\x7f]")


@dataclass(frozen=True)
class UrlPrefix:
    scheme: str
    host: str
    port: int
    path: str

    def __str__(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}{self.path}"


@dataclass(frozen=True)
class _Parsed:
    parts: SplitResult
    host: str
    port: int


def _parse(url: str) -> _Parsed:
    """Split a URL, refusing every shape that could reach a host other than the one it names."""
    if not isinstance(url, str) or not url or _UNSAFE_CHARS.search(url):
        raise ValueError("is not a clean URL")
    parts = urlsplit(url)
    if parts.scheme not in DEFAULT_PORTS or not parts.hostname:
        raise ValueError("needs an http or https scheme and a host")
    if parts.username is not None or parts.password is not None:
        raise ValueError("carries credentials")
    if parts.fragment:
        raise ValueError("carries a fragment")
    if _UNSAFE_PATH.search(parts.path):
        raise ValueError("has dot segments or encoded separators in its path")
    if parts.netloc.endswith(":"):
        raise ValueError("has an empty port")
    port = parts.port  # raises ValueError on a malformed or out-of-range port
    return _Parsed(parts, parts.hostname.lower(), port or DEFAULT_PORTS[parts.scheme])


def parse_prefix(raw: str) -> UrlPrefix:
    try:
        parsed = _parse(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{raw!r} is not a storage prefix: it {exc}") from None
    if parsed.parts.query:
        raise ValueError(f"{raw!r} is not a storage prefix: it carries a query string")
    path = parsed.parts.path or "/"
    if not path.endswith("/"):
        path += "/"
    return UrlPrefix(parsed.parts.scheme, parsed.host, parsed.port, path)


def parse_prefixes(raw: str) -> Tuple[UrlPrefix, ...]:
    """Comma- or whitespace-separated prefixes; empty means no media URL is accepted."""
    return tuple(parse_prefix(item) for item in re.split(r"[,\s]+", raw.strip()) if item)


def url_allowed(url: str, prefixes: Iterable[UrlPrefix]) -> bool:
    try:
        parsed = _parse(url)
    except ValueError:
        return False
    path = parsed.parts.path or "/"
    return any(
        prefix.scheme == parsed.parts.scheme
        and prefix.host == parsed.host
        and prefix.port == parsed.port
        and path.startswith(prefix.path)
        for prefix in prefixes
    )


def redact(url: str) -> str:
    """The URL without credentials or query string (where a presigned signature lives)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable URL>"
    return f"{parts.scheme}://{parts.netloc.rpartition('@')[2]}{parts.path}"
