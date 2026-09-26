"""PRD-251 US-108 (S1.3): a woff2 font drawn here, with the standard library only.

The media-render CI job proves a template renders with the brand kit's heading
font, an uploaded woff2 named in ``heading_font``. That needs a real woff2
whose use shows plainly in the pixels, and the runner has no font tools, so
this file draws one: "CI Block", a TrueType face whose every printable ASCII
character is a solid block. Set in it, a headline turns into bars of ink.

A woff2 wraps the font's tables in one Brotli stream (W3C WOFF2 §5). This one
is made of Brotli's uncompressed meta-blocks (RFC 7932 §9.2), and glyf and loca
carry WOFF2's null transform (transform version 3), so no compressor is needed
and the file is still a valid woff2. Every byte is written by this file: there
is nothing to license.

    from ci_block_font import block_font_woff2
    data = block_font_woff2()          # about 4 KB
"""

from __future__ import annotations

import struct
from typing import Dict, List, Tuple

FAMILY = "CI Block"
UNITS_PER_EM = 1000
ASCENT, DESCENT = 800, -200
ADVANCE = 600
# The block: the whole advance but a 20-unit gap, from below the baseline to above the caps.
BLOCK = (20, -120, 580, 740)  # xMin, yMin, xMax, yMax
FIRST_CHAR, LAST_CHAR = 0x20, 0x7E  # space, then the blocks
NUM_GLYPHS = 2 + (LAST_CHAR - FIRST_CHAR)  # .notdef, space, one block per printable character

WOFF2_SIGNATURE = b"wOF2"
TRUETYPE_FLAVOR = 0x00010000
# WOFF2 §4.1: the known table tags, by index; transform version 3 is glyf/loca's null transform.
KNOWN_TAGS = {b"cmap": 0, b"head": 1, b"hhea": 2, b"hmtx": 3, b"maxp": 4, b"name": 5, b"OS/2": 6, b"post": 7, b"glyf": 10, b"loca": 11}
NULL_TRANSFORM_GLYF = 3
# head's dates count seconds from 1904-01-01: this is 2026-01-01.
FONT_DATE = 2082844800 + 1767225600
BROTLI_MAX_UNCOMPRESSED_BLOCK = 1 << 16


# ── the TrueType tables ─────────────────────────────────────────────────────
def _head() -> bytes:
    x_min, y_min, x_max, y_max = BLOCK
    return struct.pack(
        ">IIIIHHqqhhhhHHhhh",
        0x00010000,  # version
        0x00010000,  # fontRevision 1.0
        0,  # checkSumAdjustment: the woff2 decoder computes it
        0x5F0F3CF5,  # magicNumber
        0x000B,  # flags: baseline at y=0, left sidebearing at x=0, integer ppem
        UNITS_PER_EM,
        FONT_DATE,  # created
        FONT_DATE,  # modified
        x_min, y_min, x_max, y_max,
        0,  # macStyle
        8,  # lowestRecPPEM
        2,  # fontDirectionHint
        1,  # indexToLocFormat: long offsets
        0,  # glyphDataFormat
    )


def _hhea() -> bytes:
    x_min, _, x_max, _ = BLOCK
    return struct.pack(
        ">IhhhHhhhhhhhhhhhH",
        0x00010000,
        ASCENT, DESCENT, 0,  # ascender, descender, lineGap
        ADVANCE,  # advanceWidthMax
        x_min, ADVANCE - x_max, x_max,  # minLeftSideBearing, minRightSideBearing, xMaxExtent
        1, 0, 0,  # caretSlopeRise, caretSlopeRun, caretOffset
        0, 0, 0, 0,  # reserved
        0,  # metricDataFormat
        NUM_GLYPHS,  # numberOfHMetrics
    )


def _maxp() -> bytes:
    # version 1.0: one contour of four points per glyph, no instructions, no composites.
    return struct.pack(">IHHHHHHHHHHHHHH", 0x00010000, NUM_GLYPHS, 4, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0)


def _hmtx() -> bytes:
    lsb = BLOCK[0]
    metrics = [(ADVANCE, 0), (ADVANCE, 0)] + [(ADVANCE, lsb)] * (NUM_GLYPHS - 2)
    return b"".join(struct.pack(">Hh", advance, side) for advance, side in metrics)


def _cmap() -> bytes:
    # One (3, 1) format 4 subtable: space..~ map to glyphs 1..95 by one delta, then the 0xFFFF terminator.
    segments = [(FIRST_CHAR, LAST_CHAR, 1 - FIRST_CHAR), (0xFFFF, 0xFFFF, 1)]
    seg_count = len(segments)
    log2 = seg_count.bit_length() - 1
    search_range = 2 * (1 << log2)
    subtable_body = (
        b"".join(struct.pack(">H", end) for _, end, _ in segments)
        + struct.pack(">H", 0)  # reservedPad
        + b"".join(struct.pack(">H", start) for start, _, _ in segments)
        + b"".join(struct.pack(">h", delta) for _, _, delta in segments)
        + b"".join(struct.pack(">H", 0) for _ in segments)  # idRangeOffset
    )
    length = 14 + len(subtable_body)
    subtable = struct.pack(">HHHHHHH", 4, length, 0, seg_count * 2, search_range, log2, seg_count * 2 - search_range) + subtable_body
    return struct.pack(">HH", 0, 1) + struct.pack(">HHI", 3, 1, 12) + subtable


def _block_glyph() -> bytes:
    x_min, y_min, x_max, y_max = BLOCK
    # Clockwise from the bottom left: up, across the top, down. Each point on the curve,
    # each coordinate a 16-bit delta (flag 0x01).
    points = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
    xs, ys, last = [], [], (0, 0)
    for x, y in points:
        xs.append(x - last[0])
        ys.append(y - last[1])
        last = (x, y)
    glyph = (
        struct.pack(">hhhhh", 1, x_min, y_min, x_max, y_max)
        + struct.pack(">HH", len(points) - 1, 0)  # endPtsOfContours[0], instructionLength
        + bytes([0x01] * len(points))
        + struct.pack(f">{len(xs)}h", *xs)
        + struct.pack(f">{len(ys)}h", *ys)
    )
    return glyph + b"\x00" * (-len(glyph) % 4)


def _glyf_loca() -> Tuple[bytes, bytes]:
    block = _block_glyph()
    glyphs = [b"", b""] + [block] * (NUM_GLYPHS - 2)  # .notdef and space are empty
    offsets, position = [], 0
    for glyph in glyphs:
        offsets.append(position)
        position += len(glyph)
    offsets.append(position)
    return b"".join(glyphs), b"".join(struct.pack(">I", offset) for offset in offsets)


def _name(family: str) -> bytes:
    postscript = family.replace(" ", "") + "-Regular"
    names = {1: family, 2: "Regular", 4: f"{family} Regular", 5: "Version 1.000", 6: postscript}
    records, strings = [], b""
    for name_id, text in sorted(names.items()):
        encoded = text.encode("utf-16-be")
        records.append(struct.pack(">HHHHHH", 3, 1, 0x0409, name_id, len(encoded), len(strings)))
        strings += encoded
    header = struct.pack(">HHH", 0, len(records), 6 + 12 * len(records))
    return header + b"".join(records) + strings


def _os2() -> bytes:
    return (
        struct.pack(
            ">HhHHHhhhhhhhhhhh",
            4,  # version
            ADVANCE,  # xAvgCharWidth
            400, 5, 0,  # usWeightClass, usWidthClass, fsType (installable)
            650, 600, 0, 75,  # subscript size and offset
            650, 600, 0, 350,  # superscript size and offset
            50, 300,  # strikeout size and position
            0,  # sFamilyClass
        )
        + bytes(10)  # panose
        + struct.pack(">IIII", 1, 0, 0, 0)  # ulUnicodeRange: Basic Latin
        + b"NONE"  # achVendID
        + struct.pack(">HHH", 0x0040, FIRST_CHAR, LAST_CHAR)  # fsSelection REGULAR, first and last char
        + struct.pack(">hhh", ASCENT, DESCENT, 0)  # sTypoAscender, sTypoDescender, sTypoLineGap
        + struct.pack(">HH", ASCENT, -DESCENT)  # usWinAscent, usWinDescent
        + struct.pack(">II", 1, 0)  # ulCodePageRange: Latin 1
        + struct.pack(">hhHHH", 500, 700, 0, FIRST_CHAR, 1)  # sxHeight, sCapHeight, default, break, maxContext
    )


def _post() -> bytes:
    # version 3.0: no glyph names.
    return struct.pack(">IIhhIIIII", 0x00030000, 0, -100, 50, 1, 0, 0, 0, 0)


def font_tables(family: str = FAMILY) -> Dict[bytes, bytes]:
    """The TrueType tables of the face, by tag."""
    glyf, loca = _glyf_loca()
    return {
        b"OS/2": _os2(), b"cmap": _cmap(), b"glyf": glyf, b"head": _head(), b"hhea": _hhea(),
        b"hmtx": _hmtx(), b"loca": loca, b"maxp": _maxp(), b"name": _name(family), b"post": _post(),
    }


def directory_order(tags) -> List[bytes]:
    """Tags sorted, with loca straight after glyf (the order the woff2 reference encoder writes)."""
    ordered = sorted(tag for tag in tags if tag != b"loca")
    if b"loca" in tags:
        ordered.insert(ordered.index(b"glyf") + 1, b"loca")
    return ordered


# ── the woff2 container ─────────────────────────────────────────────────────
class _Bits:
    """Brotli's bit order: least significant bit first."""

    def __init__(self) -> None:
        self.out, self.byte, self.count = bytearray(), 0, 0

    def write(self, value: int, width: int) -> None:
        for i in range(width):
            self.byte |= ((value >> i) & 1) << self.count
            self.count += 1
            if self.count == 8:
                self.flush()

    def flush(self) -> None:
        if self.count:
            self.out.append(self.byte)
            self.byte, self.count = 0, 0


def brotli_stored(data: bytes) -> bytes:
    """``data`` as a Brotli stream of uncompressed meta-blocks (RFC 7932 §9.1-9.2)."""
    bits = _Bits()
    bits.write(0, 1)  # WBITS = 16
    for start in range(0, len(data), BROTLI_MAX_UNCOMPRESSED_BLOCK):
        chunk = data[start : start + BROTLI_MAX_UNCOMPRESSED_BLOCK]
        bits.write(0, 1)  # ISLAST = 0
        bits.write(0, 2)  # MNIBBLES = 4
        bits.write(len(chunk) - 1, 16)  # MLEN - 1
        bits.write(1, 1)  # ISUNCOMPRESSED
        bits.flush()  # the data starts on a byte boundary
        bits.out += chunk
    bits.write(1, 1)  # ISLAST
    bits.write(1, 1)  # ISLASTEMPTY
    bits.flush()
    return bytes(bits.out)


def _base128(value: int) -> bytes:
    """WOFF2's UIntBase128: big-endian 7-bit groups, the high bit set on all but the last."""
    groups = [value & 0x7F]
    value >>= 7
    while value:
        groups.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(groups))


def _pad4(length: int) -> int:
    return length + (-length % 4)


def woff2_from_tables(tables: Dict[bytes, bytes]) -> bytes:
    """A woff2 file holding ``tables``, every one untransformed."""
    order = directory_order(tables)
    directory = b""
    for tag in order:
        version = NULL_TRANSFORM_GLYF if tag in (b"glyf", b"loca") else 0
        directory += bytes([(version << 6) | KNOWN_TAGS[tag]]) + _base128(len(tables[tag]))
    stream = brotli_stored(b"".join(tables[tag] for tag in order))
    sfnt_size = 12 + 16 * len(order) + sum(_pad4(len(tables[tag])) for tag in order)
    header_size = 48
    length = _pad4(header_size + len(directory) + len(stream))
    header = struct.pack(
        ">4sIIHHIIHHIIIII",
        WOFF2_SIGNATURE, TRUETYPE_FLAVOR, length, len(order), 0, sfnt_size, len(stream),
        1, 0,  # majorVersion, minorVersion
        0, 0, 0,  # no metadata
        0, 0,  # no private data
    )
    body = header + directory + stream
    return body + b"\x00" * (length - len(body))


def block_font_woff2(family: str = FAMILY) -> bytes:
    """The "CI Block" face as a woff2 file."""
    return woff2_from_tables(font_tables(family))


__all__ = ["ADVANCE", "BLOCK", "FAMILY", "block_font_woff2", "brotli_stored", "directory_order", "font_tables", "woff2_from_tables"]
