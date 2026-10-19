"""
Image content blocks, read one way for every provider that sends an image's bytes.

A message's content can carry an image as an Anthropic-style base64 block
(``{"type": "image", "source": {"type": "base64", "media_type", "data"}}``) or as an
OpenAI-style data URL (``{"type": "image_url", "image_url": {"url": "data:…;base64,…"}}``,
as ``modules/attachments/resolver.py`` builds for a small attachment). Bedrock's Converse
(#1172) and Gemini (#1180) both take the bytes and a media type; each says which media
types it accepts.
"""

import base64
import binascii
from typing import Any, Dict, Optional, Tuple

# Media types written two ways, as one.
_ALIASES = {"image/jpg": "image/jpeg"}


def _decoded(media_type: str, data: str) -> Tuple[str, bytes]:
    kind = (media_type or "").strip().lower()
    try:
        return _ALIASES.get(kind, kind), base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("An image in the conversation isn't valid base64, so it can't be sent.") from error


def decoded_image(block: Dict[str, Any]) -> Optional[Tuple[str, bytes]]:
    """``(media type, bytes)`` for a base64 image block or a data URL; None for any other
    block, a plain ``https://`` image URL included (its bytes aren't in the message).
    Raises ``ValueError`` for data that isn't valid base64."""
    source = block.get("source") or {}
    if block.get("type") == "image" and source.get("type") == "base64":
        return _decoded(source.get("media_type", ""), source.get("data", ""))
    url = (block.get("image_url") or {}).get("url", "")
    if block.get("type") == "image_url" and url.startswith("data:") and ";base64," in url:
        media_type, data = url[len("data:"):].split(";base64,", 1)
        return _decoded(media_type, data)
    return None
