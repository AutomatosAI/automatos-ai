"""The file and headers of a Gmail attachment download (``api/widget_email.py``).

Composio's Gmail get-attachment action answers with the attachment either as
Gmail's own body (``{"attachmentId" | "size", "data": <base64url>}``) or as
Composio's file output (``{"name", "mimetype", "s3url"}``). The bytes are read
from whichever the action returned: inline data is decoded, and a file link from
the action's own answer is fetched through the platform's one outbound check
(``modules/socials/recipes/files.fetch``: public addresses only, pinned, every
redirect checked again, refused past the size cap as it streams in). No link
from the request is ever fetched.

The response is always a download: ``Content-Disposition: attachment`` with the
sanitised name, ``nosniff``, a sandboxing CSP, and a Content-Type from a small
allowlist of types that are safe inline (``application/octet-stream`` otherwise).
"""
from __future__ import annotations

import base64
import binascii
import logging
import mimetypes
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import quote

from core.composio.gmail_attachments import GET_ATTACHMENT_ACTION, MAX_FILENAME_CHARS
from modules.socials.recipes import files

logger = logging.getLogger(__name__)

# Gmail's own limit on a message's attachments; nothing bigger is served.
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 60.0
DEFAULT_FILENAME = "attachment"
OCTET_STREAM = "application/octet-stream"
INLINE_SAFE_TYPES = frozenset({
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "text/plain",
    "text/csv",
})
# How deep the action's answer is searched for Gmail's attachment body.
MAX_DEPTH = 6
# Keys that mark an object as Gmail's attachment body (MessagePartBody).
GMAIL_BODY_KEYS = ("attachmentId", "attachment_id", "size")
_BASE64URL = re.compile(r"[A-Za-z0-9_\-+/]*={0,2}")
_PATH_SEPARATORS = re.compile(r"[\\/]")

INVALID_LINK = "That attachment link is not valid."
NOT_CONNECTED = "Gmail isn't connected to this workspace. Connect it in Integrations, then try again."
NOT_FOUND = "That attachment wasn't found in Gmail. It may have been deleted."
TOO_LARGE = f"This attachment is larger than {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB. Open it in Gmail instead."
UNREADABLE = "Gmail did not return the attachment. Try again, or open it in Gmail."
_NOT_FOUND_MARKERS = ("not found", "notfound", "404", "invalid id", "invalid attachment", "invalid message")


class AttachmentError(Exception):
    """The download is refused: the HTTP status and the message the person sees."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def safe_filename(name: Any) -> str:
    """The name to save under: its last path segment, without control or
    direction characters or leading dots, at most ``MAX_FILENAME_CHARS``."""
    text = _PATH_SEPARATORS.split(name)[-1] if isinstance(name, str) else ""
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("C"))
    text = text.strip().lstrip(".").strip()
    return text[:MAX_FILENAME_CHARS] or DEFAULT_FILENAME


def safe_content_type(mimetype: Any, filename: str) -> str:
    """The attachment's type when it is on the inline-safe allowlist, else
    ``application/octet-stream``. With no type given, the name's is used."""
    raw = mimetype if isinstance(mimetype, str) and mimetype.strip() else mimetypes.guess_type(filename)[0]
    kind = (raw or "").split(";", 1)[0].strip().lower()
    return kind if kind in INLINE_SAFE_TYPES else OCTET_STREAM


def download_headers(filename: str) -> Dict[str, str]:
    """Headers that make the response a download of ``filename`` and nothing else."""
    return {
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename, safe='')}",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "sandbox",
        "Cache-Control": "private, no-store",
    }


def check_result(result: Mapping[str, Any]) -> None:
    """Refuse a failed action, or one the executor ran under another name."""
    ran = str(result.get("action") or GET_ATTACHMENT_ACTION).upper()
    if ran != GET_ATTACHMENT_ACTION:
        logger.warning("[EmailAttachment] the executor ran %s instead of %s", ran, GET_ATTACHMENT_ACTION)
        raise AttachmentError(502, UNREADABLE)
    if result.get("success"):
        return
    error = str(result.get("error") or "")
    logger.warning("[EmailAttachment] %s failed: %s", GET_ATTACHMENT_ACTION, error[:300])
    if result.get("error_type") == "composio_not_connected":
        raise AttachmentError(404, NOT_CONNECTED)
    if any(marker in error.lower() for marker in _NOT_FOUND_MARKERS):
        raise AttachmentError(404, NOT_FOUND)
    raise AttachmentError(502, UNREADABLE)


def _dicts(value: Any) -> List[dict]:
    """Every object in an answer, shallowest first, down to ``MAX_DEPTH``."""
    found: List[dict] = []
    frontier = [value]
    for _ in range(MAX_DEPTH + 1):
        following: list = []
        for item in frontier:
            if isinstance(item, dict):
                found.append(item)
                following.extend(item.values())
            elif isinstance(item, list):
                following.extend(item)
        frontier = following
    return found


def _decode_base64url(text: str) -> bytes:
    if len(text) * 3 // 4 > MAX_ATTACHMENT_BYTES:
        raise AttachmentError(413, TOO_LARGE)
    normalised = text.replace("-", "+").replace("_", "/").rstrip("=")
    try:
        return base64.b64decode(normalised + "=" * (-len(normalised) % 4), validate=True)
    except (binascii.Error, ValueError):
        raise AttachmentError(502, UNREADABLE) from None


def gmail_inline_bytes(output: Any) -> Optional[bytes]:
    """The bytes of Gmail's own attachment body in the answer, or ``None``."""
    for obj in _dicts(output):
        text = obj.get("data")
        if not isinstance(text, str) or not any(key in obj for key in GMAIL_BODY_KEYS):
            continue
        if _BASE64URL.fullmatch(text.strip()):
            return _decode_base64url(text.strip())
    return None


async def _fetch(url: str) -> bytes:
    try:
        return await files.fetch(url, max_bytes=MAX_ATTACHMENT_BYTES, timeout_seconds=FETCH_TIMEOUT_SECONDS)
    except files.FileTooLarge:
        raise AttachmentError(413, TOO_LARGE) from None
    except files.FileOutputError as exc:
        logger.warning("[EmailAttachment] the attachment's file link could not be read: %s", exc)
        raise AttachmentError(502, UNREADABLE) from None


async def attachment_file(result: Mapping[str, Any]) -> Tuple[bytes, Optional[str]]:
    """The attachment's bytes and the type the action gave, from the executor's
    result. :class:`AttachmentError` when there are none, or too many."""
    check_result(result)
    output = result.get("data")
    inline = gmail_inline_bytes(output)
    if inline is not None:
        return inline, None
    found = files.returned_file(output)
    if found is None:
        logger.warning("[EmailAttachment] %s answered with no file", GET_ATTACHMENT_ACTION)
        raise AttachmentError(502, UNREADABLE)
    data = found.data if found.data is not None else await _fetch(found.url or "")
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise AttachmentError(413, TOO_LARGE)
    return data, found.mimetype
