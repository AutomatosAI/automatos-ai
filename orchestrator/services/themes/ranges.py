"""PRD-261 A-08a — one HTTP ``Range`` over a theme file held in memory.

Safari plays an HEVC/MP4 clip only over byte ranges, so the files route answers one
``Range: bytes=a-b`` (or ``a-``, or the suffix ``-n``) with a 206 and a 416 when nothing of
the file is in range (RFC 9110 §14). The header is read by the platform's own parser
(``core.services.image_store.parse_byte_range``, the generated-images route's): a
malformed, reversed or multi-range header is ignored and the whole file is served (200),
which RFC 9110 §14.2 allows.

Theme files are small (a clip is at most ``THEME_MAX_CLIP_KB``) and every one is re-hashed
on read, so the range is cut from bytes already in memory, never streamed.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from core.services.image_store import parse_byte_range

BYTES_UNIT = "bytes"
RANGE_PREFIX = f"{BYTES_UNIT}="


class ByteRange(NamedTuple):
    """An inclusive range of a file's bytes, both ends inside the file."""

    start: int
    end: int

    @property
    def length(self) -> int:
        """How many bytes the range holds."""
        return self.end - self.start + 1


class RangeNotSatisfiable(ValueError):
    """No byte of the file is in the requested range (the route's 416)."""

    def __init__(self, size: int) -> None:
        super().__init__(f"The requested range is outside the file's {size} bytes.")
        self.size = size


def _suffix_range(length: int, size: int) -> ByteRange:
    """The last ``length`` bytes (``bytes=-n``)."""
    if length == 0 or size == 0:
        raise RangeNotSatisfiable(size)
    return ByteRange(max(0, size - length), size - 1)


def requested_range(header: Optional[str], size: int) -> Optional[ByteRange]:
    """The one range ``header`` asks for in a file of ``size`` bytes, or None for the whole file.

    Raises :class:`RangeNotSatisfiable` when the range starts past the end, or is an empty suffix.
    """
    normalised = parse_byte_range(header)
    if normalised is None:
        return None
    first, _sep, last = normalised[len(RANGE_PREFIX):].partition("-")
    if not first:
        return _suffix_range(int(last), size)
    start = int(first)
    if start >= size:
        raise RangeNotSatisfiable(size)
    end = size - 1 if not last else min(int(last), size - 1)
    return ByteRange(start, end)


def content_range(byte_range: ByteRange, size: int) -> str:
    """The ``Content-Range`` value of a 206."""
    return f"{BYTES_UNIT} {byte_range.start}-{byte_range.end}/{size}"


def unsatisfied_content_range(size: int) -> str:
    """The ``Content-Range`` value of a 416."""
    return f"{BYTES_UNIT} */{size}"


__all__ = ["ByteRange", "RangeNotSatisfiable", "content_range", "requested_range", "unsatisfied_content_range"]
