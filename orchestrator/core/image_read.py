"""An image made small for a model to read (the brand kit's style read, the generation guard).

Pillow, when it is there, turns the image into a JPEG at most ``edge_px`` on its longer
side, upright: a phone photo's EXIF orientation is applied first, as the JPEG written
here carries no EXIF. Without Pillow an image goes as it is, and only when it is at most
``small_enough_bytes``. An image that can't be read is ``None``: the caller leaves it out,
or refuses, as its rules say; it never crashes the read.
"""
from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

JPEG_QUALITY = 80
# The picture a file names can be far bigger than the file (a 50 MB upload can claim
# 20000 × 20000), and Pillow only refuses past ~179 M pixels. Past this the read refuses
# from the header alone, before decoding; a 48 MP camera photo still fits.
MAX_READ_PIXELS = 50_000_000
# What the read decodes; anything else (PSD, ICO, EPS…) is not opened at all. An iPhone's
# multi-picture JPEG opens through JPEG (Pillow names it MPO once open, but registers no
# MPO opener, so "MPO" here would be a KeyError).
READ_FORMATS = ("JPEG", "PNG", "GIF", "WEBP", "BMP", "TIFF")


def _opened(image_module, data: bytes, edge_px: int):
    """The image, header read only: refused past ``MAX_READ_PIXELS``; a JPEG set to decode
    at the smallest scale still at least ``edge_px``, so its peak memory stays small."""
    image = image_module.open(io.BytesIO(data), formats=READ_FORMATS)
    width, height = image.size
    if width * height > MAX_READ_PIXELS:
        raise ValueError(f"a {width} x {height} image is over the read's {MAX_READ_PIXELS} pixel cap")
    if image.format in ("JPEG", "MPO"):
        image.draft("RGB", (edge_px, edge_px))
    return image


def shrunk_for_read(data: bytes, content_type: str, *, edge_px: int, small_enough_bytes: int) -> Optional[Tuple[bytes, str]]:
    """``(bytes, content type)`` of the image made small for a model's read; ``None`` when it can't be."""
    try:
        from PIL import Image, ImageOps  # optional: without it the read takes small images as they are
    except ImportError:
        return (data, content_type) if len(data) <= small_enough_bytes else None
    try:
        image = ImageOps.exif_transpose(_opened(Image, data, edge_px)).convert("RGB")
        image.thumbnail((edge_px, edge_px))
        out = io.BytesIO()
        image.save(out, "JPEG", quality=JPEG_QUALITY)
        return out.getvalue(), "image/jpeg"
    except (OSError, ValueError, Image.DecompressionBombError):  # Pillow's own bomb check, at open
        logger.warning("[ImageRead] an image could not be made small for a model's read", exc_info=True)
        return None
