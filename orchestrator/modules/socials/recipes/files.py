"""PRD-251 D12: the file a workspace's Composio media tool returned, and its fetch.

A speech or generation action returns its output as a file on a short-lived
link: Composio's file output ``{"name", "mimetype", "s3url"}`` somewhere in the
response (the platform's Composio client never lets the SDK download it), or,
from some toolkits, a plain link or the bytes inline (base64, or a data: URI).
``returned_file`` finds it and ``fetch`` reads a link. Provider links expire, so
a recipe fetches the file the moment its action returns and stores it at once.

``fetch`` reaches only a public address, pinned: every hop goes through the
platform's one outbound check (``core.security.web_access``, PRD-240), so a
link into the compose network or a metadata address is refused, never fetched.
A file over the recipe's limit is refused as it streams in.
"""
from __future__ import annotations

import base64
import binascii
import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional

import httpx

from core.security.web_access import build_pinned_request, resolve_outbound_async

logger = logging.getLogger(__name__)

# How deep a response is searched for its file.
MAX_DEPTH = 6
# Redirect hops a file link may take; each one is checked again.
MAX_REDIRECTS = 3
# Composio's file output carries its link here.
FILE_OUTPUT_KEY = "s3url"
# Keys that carry a plain file link, in the order they are trusted.
URL_KEYS = ("audio_url", "file_url", "download_url", "output_url", "url", "data")
# Keys that may carry the bytes inline, base64.
INLINE_KEYS = ("audio", "audio_base64", "audio_content", "content", "file_content", "file", "data")
MIN_INLINE_CHARS = 64
_DATA_URI = re.compile(r"^data:[a-z0-9.+/-]*(?:;[^,;]*)*;base64,(.*)$", re.IGNORECASE | re.DOTALL)
_BASE64 = re.compile(r"^[A-Za-z0-9+/=\s]+$")
REFUSED_LINK = "the tool's file link is not an address this server may fetch"


class FileOutputError(Exception):
    """The tool returned no file a recipe can use, or its file could not be fetched."""


@dataclass(frozen=True)
class ReturnedFile:
    """Where a tool's output is: a link to fetch, or the bytes themselves."""

    url: Optional[str] = None
    data: Optional[bytes] = None
    mimetype: Optional[str] = None
    name: Optional[str] = None


def _objects(response: Any) -> List[dict]:
    """Every object in a response, shallowest first."""
    found: List[dict] = []
    frontier = [response]
    for _ in range(MAX_DEPTH + 1):
        following: list = []
        for item in frontier:
            if isinstance(item, dict):
                found.append(item)
                following.extend(item.values())
            elif isinstance(item, (list, tuple)):
                following.extend(item)
        if not following:
            break
        frontier = following
    return found


def _link(value: Any) -> Optional[str]:
    text = value.strip() if isinstance(value, str) else ""
    if text.lower().startswith(("https://", "http://")) and not any(c.isspace() for c in text):
        return text
    return None


def _text(value: Any) -> Optional[str]:
    return value.strip() or None if isinstance(value, str) else None


def _inline(value: Any) -> Optional[bytes]:
    """Bytes carried inline: a base64 data: URI, or a long base64 string."""
    text = value.strip() if isinstance(value, str) else ""
    match = _DATA_URI.match(text)
    if match:
        payload = match.group(1)
    elif len(text) >= MIN_INLINE_CHARS and _BASE64.match(text):
        payload = text
    else:
        return None
    try:
        return base64.b64decode(re.sub(r"\s+", "", payload), validate=True) or None
    except (binascii.Error, ValueError):
        return None


def returned_file(response: Any) -> Optional[ReturnedFile]:
    """The file in an action's response: Composio's file output first, then a
    plain link under a known key, then bytes inline; ``None`` when there is none."""
    objects = _objects(response)
    for obj in objects:
        url = _link(obj.get(FILE_OUTPUT_KEY))
        if url:
            return ReturnedFile(url=url, mimetype=_text(obj.get("mimetype")), name=_text(obj.get("name")))
    for key in URL_KEYS:
        for obj in objects:
            url = _link(obj.get(key))
            if url:
                return ReturnedFile(url=url)
    for obj in objects:
        for key in INLINE_KEYS:
            data = _inline(obj.get(key))
            if data:
                return ReturnedFile(data=data)
    return None


async def fetch(url: str, *, max_bytes: int, timeout_seconds: float) -> bytes:
    """The file at ``url``: a public address only, pinned, each redirect checked
    again, and at most ``max_bytes``. :class:`FileOutputError` otherwise."""
    try:
        for _ in range(MAX_REDIRECTS + 1):
            target = await resolve_outbound_async(url, enforce_switch=False)
            if not target.ok:
                # The reason names what the host resolved to: the server log only.
                logger.warning("[SocialsMedia] refused a tool's file link %s: %s", url[:120], target.reason)
                raise FileOutputError(REFUSED_LINK)
            async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=False, trust_env=False) as client:
                response = await client.send(build_pinned_request(client, "GET", url, target), stream=True)
                try:
                    location = response.headers.get("location")
                    if response.is_redirect and location:
                        url = str(httpx.URL(url).join(location))
                        continue
                    return await _read(response, max_bytes)
                finally:
                    await response.aclose()
    except httpx.HTTPError as exc:
        raise FileOutputError(f"the tool's file link could not be read ({type(exc).__name__})") from None
    raise FileOutputError(f"the tool's file link redirected more than {MAX_REDIRECTS} times")


async def _read(response: httpx.Response, max_bytes: int) -> bytes:
    if response.status_code != 200:
        raise FileOutputError(f"the tool's file link answered {response.status_code}")
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise FileOutputError(f"the tool's file is larger than the {max_bytes}-byte limit")
    chunks, size = [], 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > max_bytes:
            raise FileOutputError(f"the tool's file is larger than the {max_bytes}-byte limit")
        chunks.append(chunk)
    if not size:
        raise FileOutputError("the tool's file link returned an empty file")
    return b"".join(chunks)
