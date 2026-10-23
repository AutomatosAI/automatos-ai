"""
Image content blocks, read one way for every provider that sends an image's bytes.

A message's content can carry an image as an Anthropic-style base64 block
(``{"type": "image", "source": {"type": "base64", "media_type", "data"}}``) or as an
OpenAI-style data URL (``{"type": "image_url", "image_url": {"url": "data:…;base64,…"}}``,
as ``modules/attachments/resolver.py`` builds for an image attachment). Bedrock's Converse
(#1172), Gemini (#1180) and the Anthropic client all take the bytes and a media type; each
says which media types it accepts.
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


def _decoded(media_type: str, data: str) -> Tuple[str, bytes]:
    kind = (media_type or "").strip().lower()
    try:   # whitespace is dropped first: MIME-wrapped base64 breaks its lines
        return _ALIASES.get(kind, kind), base64.b64decode("".join(data.split()), validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("An image in the conversation isn't valid base64, so it can't be sent.") from error


def _mapping(value: Any) -> Dict[str, Any]:
    """``value`` when it is a dict; ``{}`` for anything else (a bare URL string, say), so a
    malformed block is left out like any other non-image block, never an AttributeError."""
    return value if isinstance(value, dict) else {}


def decoded_image(block: Dict[str, Any]) -> Optional[Tuple[str, bytes]]:
    """``(media type, bytes)`` for a base64 image block or a data URL; None for any other
    block, a plain ``https://`` image URL included (its bytes aren't in the message).
    Raises ``ValueError`` for data that isn't valid base64."""
    source = _mapping(block.get("source"))
    if block.get("type") == "image" and source.get("type") == "base64":
        return _decoded(source.get("media_type", ""), source.get("data", ""))
    url = str(_mapping(block.get("image_url")).get("url", ""))
    match = _DATA_URL.match(url) if block.get("type") == "image_url" else None
    return _decoded(match.group(1), match.group(2)) if match else None
