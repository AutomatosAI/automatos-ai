"""Small table PDFs, made here: the fixtures for PRD-262 US-023's reference grids.

Each fixture is a PDF of one or more pages, one table to a page, drawn the way a
report without ruling lines draws one: every cell is its own run of text, set at its
column's left edge (or right edge, for a right-aligned column) on its row's
baseline. The text is Helvetica, one of the 14 standard PDF fonts, so nothing is
embedded and every PDF reader measures it with the same built-in metrics: the bytes,
and the layout a reader finds in them, are the same on every machine.

Two readers use them:

- ``scripts/ci/reference/pdf_tables.py`` records the cell grids that the extractor
  the API ran before US-023 finds in them, into ``pdf_table_grids.json`` here;
- ``tests/test_prd262_pdf_tables.py`` checks that pdfplumber's text strategy, which
  replaced it, finds the same grids.

The PDFs are written to a directory when a test or the reference script runs; none is
committed. They hold nobody else's content, so they carry no licence. Standard library
only: the reference script loads this file in a venv that has none of the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PAGE_SIZE = (612, 792)  # US Letter, in points
FONT = "Helvetica"
CAPTION_SIZE = 12
CELL_SIZE = 10
CAPTION_BASELINE = 730.0
FIRST_ROW_BASELINE = 700.0
# Baseline to baseline: 8 pt of leading under 10 pt text, the spacing a report uses
# (wider than pdfplumber's 3 pt snap tolerance, so its text strategy sees a gap
# between rows).
ROW_PITCH = 18.0
LEFT, RIGHT = "left", "right"

# Helvetica's advance widths in 1/1000 em, printable ASCII from the space (32) to the
# tilde (126): the font's AFM, which PDF readers carry built in. A right-aligned cell
# starts at its column's right edge less its text's width.
_ASCII_WIDTHS = (
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
)
HELVETICA_WIDTHS: Dict[str, int] = {chr(32 + i): width for i, width in enumerate(_ASCII_WIDTHS)}


@dataclass(frozen=True)
class Column:
    """Where a column's text sits: ``x`` is its left edge, or its right edge when ``align`` is right."""

    x: float
    align: str = LEFT


@dataclass(frozen=True)
class TablePage:
    """One page: an optional caption above a table whose first row is its header ("" is an empty cell)."""

    columns: Tuple[Column, ...]
    rows: Tuple[Tuple[str, ...], ...]
    caption: str = ""


_REGIONS = (Column(72), Column(200), Column(400, RIGHT), Column(500, RIGHT))
_TEAMS = (Column(72), Column(170), Column(290), Column(390))
_PARTS = (Column(72), Column(150), Column(320))

FIXTURES: Dict[str, Tuple[TablePage, ...]] = {
    # A caption above the table, and two right-aligned numeric columns.
    "revenue-by-region": (
        TablePage(
            caption="Revenue by region",
            columns=_REGIONS,
            rows=(
                ("Region", "Quarter", "Revenue", "Change"),
                ("North", "2026-Q1", "1,250,400.00", "+4.2%"),
                ("South", "2026-Q1", "980,150.50", "-1.8%"),
                ("East", "2026-Q1", "1,102,000.00", "+0.6%"),
                ("West", "2026-Q1", "760,325.25", "+12.9%"),
                ("Central", "2026-Q1", "455,000.00", "-3.1%"),
            ),
        ),
    ),
    # Two-word cells, an empty cell, one-digit numbers and dates.
    "team-headcount": (
        TablePage(
            columns=_TEAMS,
            rows=(
                ("Team", "Lead", "Headcount", "Opened"),
                ("Platform", "Ana Silva", "12", "2024-03-01"),
                ("Data", "Rui Costa", "7", "2025-01-15"),
                ("Design", "", "4", "2025-06-30"),
                ("Support", "Joana Reis", "9", "2023-11-20"),
                ("Sales", "Marta Lopes", "15", "2024-09-09"),
            ),
        ),
    ),
    # A table carried over two pages, its header repeated.
    "parts-two-pages": (
        TablePage(
            columns=_PARTS,
            rows=(
                ("SKU", "Description", "Qty"),
                ("A-100", "Steel bracket", "40"),
                ("B-220", "Hex bolt M8", "250"),
                ("C-310", "Washer", "1200"),
            ),
        ),
        TablePage(
            columns=_PARTS,
            rows=(
                ("SKU", "Description", "Qty"),
                ("D-415", "Hinge pin", "75"),
                ("E-520", "Cable ties", "500"),
                ("F-630", "Spring clip", "60"),
            ),
        ),
    ),
}


def text_width(text: str, size: float) -> float:
    """The advance width of ``text`` set in Helvetica at ``size`` points."""
    return sum(HELVETICA_WIDTHS[char] for char in text) * size / 1000


def _literal(text: str) -> bytes:
    """``text`` as a PDF string literal; printable ASCII only, so every reader decodes it alike."""
    unknown = sorted({char for char in text if char not in HELVETICA_WIDTHS})
    if unknown:
        raise ValueError(f"not printable ASCII: {unknown!r} in {text!r}")
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return b"(" + escaped.encode("ascii") + b")"


def _show(text: str, x: float, baseline: float, size: float) -> bytes:
    return b"/F1 %d Tf 1 0 0 1 %.2f %.2f Tm %s Tj\n" % (size, x, baseline, _literal(text))


def _cell(text: str, column: Column, baseline: float) -> bytes:
    left = column.x - text_width(text, CELL_SIZE) if column.align == RIGHT else column.x
    return _show(text, left, baseline, CELL_SIZE)


def content_stream(page: TablePage) -> bytes:
    """The page's text, caption first, then the table row by row and left to right."""
    shows: List[bytes] = []
    if page.caption:
        shows.append(_show(page.caption, page.columns[0].x, CAPTION_BASELINE, CAPTION_SIZE))
    for index, row in enumerate(page.rows):
        baseline = FIRST_ROW_BASELINE - index * ROW_PITCH
        shows += [_cell(text, column, baseline) for text, column in zip(row, page.columns) if text]
    return b"BT\n" + b"".join(shows) + b"ET"


def _objects(pages: Sequence[TablePage]) -> Dict[int, bytes]:
    """Numbered objects: catalog, page tree, font, then a page and its content per page."""
    page_ids = [4 + 2 * index for index in range(len(pages))]
    kids = b" ".join(b"%d 0 R" % page_id for page_id in page_ids)
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(pages)),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /%s /Encoding /WinAnsiEncoding >>" % FONT.encode(),
    }
    for page_id, page in zip(page_ids, pages):
        stream = content_stream(page)
        objects[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (*PAGE_SIZE, page_id + 1)
        )
        objects[page_id + 1] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
    return objects


def build_pdf(pages: Sequence[TablePage]) -> bytes:
    """A complete PDF 1.4 file for ``pages``: the objects, a cross-reference table and the trailer."""
    objects = _objects(pages)
    parts, offsets, offset = [b"%PDF-1.4\n"], [], len(b"%PDF-1.4\n")
    for number in sorted(objects):
        body = b"%d 0 obj\n%s\nendobj\n" % (number, objects[number])
        offsets.append(offset)
        parts.append(body)
        offset += len(body)
    size = len(objects) + 1
    entries = b"".join(b"%010d 00000 n \n" % position for position in offsets)
    xref = b"xref\n0 %d\n0000000000 65535 f \n%s" % (size, entries)
    trailer = b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, offset)
    return b"".join(parts) + xref + trailer


def write_fixtures(directory: Path) -> Dict[str, Path]:
    """Write every fixture to ``directory`` as ``<name>.pdf``; name -> path, in FIXTURES order."""
    paths = {name: Path(directory) / f"{name}.pdf" for name in FIXTURES}
    for name, path in paths.items():
        path.write_bytes(build_pdf(FIXTURES[name]))
    return paths
