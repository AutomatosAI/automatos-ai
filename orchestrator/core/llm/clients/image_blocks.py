"""
Image content blocks, read one way for every provider that sends an image's bytes.

A message's content can carry an image as an Anthropic-style base64 block
(``{"type": "image", "source": {"type": "base64", "media_type", "data"}}``) or as an
OpenAI-style data URL (``{"type": "image_url", "image_url": {"url": "data:…;base64,…"}}``,
as ``modules/attachments/resolver.py`` builds for a small attachment). Bedrock's Converse
(#1172) and Gemini (#1180) both take the bytes and a media type; each says which media
types it accepts, and decodes only those (:func:`image_source`, then :func:`decode_image`).
"""

import base64
import binascii
import re
from typing import Any, Dict, Optional, Tuple

# Media types written two ways, as one.
_ALIASES = {"image/jpg": "image/jpeg"}
# A base64 data URL (RFC 2397): its media type, any parameters (``;charset=…``), the
# ``base64`` marker in any case, and the data.
_DATA_URL = re.compile(r"^data:([^;,]*)(?:;[^;,]*?)*?;base64,(.*)$", re.IGNORECASE | re.DOTALL)


def _mapping(value: Any) -> Dict[str, Any]:
    """``value`` when it is a dict; ``{}`` for anything else (a bare URL string, say), so a
    malformed block is left out like any other non-image block, never an AttributeError."""
    return value if isinstance(value, dict) else {}


def _source(media_type: Any, data: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(media_type, str) or not isinstance(data, str):
        return None   # a field that isn't text: no image, as for any other malformed block
    kind = media_type.strip().lower()
    return _ALIASES.get(kind, kind), data


def image_source(block: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """``(media type, base64 data)``, not yet decoded, for a base64 image block or a data
    URL; None for any other block: a plain ``https://`` image URL (its bytes aren't in the
    message), or one whose fields aren't text. A caller checks the media type first and
    decodes only an image it can send, with :func:`decode_image`."""
    source = _mapping(block.get("source"))
    if block.get("type") == "image" and source.get("type") == "base64":
        return _source(source.get("media_type", ""), source.get("data", ""))
    url = _mapping(block.get("image_url")).get("url", "")
    match = _DATA_URL.match(url) if block.get("type") == "image_url" and isinstance(url, str) else None
    return _source(match.group(1), match.group(2)) if match else None


def decode_image(data: str) -> bytes:
    """An image's base64 data as bytes, strictly; whitespace is dropped first, as
    MIME-wrapped base64 breaks its lines. Raises ``ValueError`` for data that isn't base64."""
    try:
        return base64.b64decode("".join(data.split()), validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("An image in the conversation isn't valid base64, so it can't be sent.") from error
