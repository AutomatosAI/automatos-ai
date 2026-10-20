"""An image made small for a model to read (the brand kit's style read, the generation guard).

Pillow, when it is there, turns the image into a JPEG at most ``edge_px`` on its longer
side. Without Pillow an image goes as it is, and only when it is at most
``small_enough_bytes``. An image that can't be read is ``None``: the caller leaves it out,
or refuses, as its rules say; it never crashes the read.
"""
from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

JPEG_QUALITY = 80


def shrunk_for_read(data: bytes, content_type: str, *, edge_px: int, small_enough_bytes: int) -> Optional[Tuple[bytes, str]]:
    """``(bytes, content type)`` of the image made small for a model's read; ``None`` when it can't be."""
    try:
        from PIL import Image  # optional: without it the read takes small images as they are
    except ImportError:
        return (data, content_type) if len(data) <= small_enough_bytes else None
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail((edge_px, edge_px))
        out = io.BytesIO()
        image.save(out, "JPEG", quality=JPEG_QUALITY)
        return out.getvalue(), "image/jpeg"
    except (OSError, ValueError):
        logger.warning("[ImageRead] an image could not be made small for a model's read", exc_info=True)
        return None
