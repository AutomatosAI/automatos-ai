"""A report's data, read for a chart (PRD-251 S1.7, D7).

An agent report (``agent_reports``) keeps its body as a markdown file and its
numbers in the ``metrics`` JSONB; there is no structured table column. A chart
bound to a report reads one of two things:

* **the report's data table**: the first markdown table in its file
  (:func:`first_table`), its first column the labels and one numeric column the
  figures (:func:`table_series`);
* **its metrics**: every figure in ``metrics``, in the stored order
  (:func:`metrics_series`).

Either way the chart takes the top rows as the source has them: never
re-ordered, never rounded. A figure keeps the text the source wrote ("$1,284",
"4.1%"): that text is what the chart shows, and the number parsed from it only
places the shapes. :data:`FIGURE_PATTERN` is the one rule for reading a figure,
shared with the infographic template's own script (the same pattern, written as
a JavaScript regular expression literal), so the shapes the template draws and
the numbers the platform checks can never disagree.

Pure: the standard library only, so the media-render CI job reads the fixture
report with this very code.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List, Mapping, Optional, Tuple

# A figure: a sign (the minus sign "−" too), a currency, a number with comma
# thousands and a decimal point, and a unit ("$1,284.50", "-3.2%", "USD 1,200",
# "12 ms", "3x"). Written in the subset Python and JavaScript read alike
# ([0-9], a literal space, numbered groups only), because the infographic's
# script carries it verbatim as a regular expression literal. Groups: 1 sign,
# 2 currency, 3 sign, 4 whole part, 5 decimals, 6 unit.
FIGURE_PATTERN = (
    r"^([+\-−])?(?:((?:[A-Z]{1,2})?\$|[€£¥₹₩₽¢]|[A-Z]{3}) ?)?"
    r"([+\-−])?([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(\.[0-9]+)?(?: ?([%×]|[A-Za-z]{1,12}))?$"
)
_FIGURE = re.compile(FIGURE_PATTERN)
NEGATIVE_SIGNS = ("-", "−")
MAX_FIGURE_CHARS = 40

TABLE, METRICS = "table", "metrics"
PARTS = (TABLE, METRICS)
# The rows a chart shows by default: the top five.
DEFAULT_ROWS = 5
# How much of a report file is searched for its table, and the widest table read.
MAX_MARKDOWN_CHARS = 200_000
MAX_COLUMNS = 32

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_DELIMITER_CELL = re.compile(r"^:?-+:?$")
_ESCAPED_PIPE = "\\|"
_PIPE_PLACEHOLDER = "\x00"
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MARKED = re.compile(r"(\*\*|\*|~~|`)(.+?)\1")
# An underscore marks emphasis only at a word's edge: input_tokens stays a name.
_UNDERSCORED = re.compile(r"(?<![0-9A-Za-z])(__|_)(.+?)\1(?![0-9A-Za-z])")
_HTML_TAG = re.compile(r"<[^<>]+>")
_BACKSLASH_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|~<>])")
_SPACES = re.compile(r"\s+")


class SeriesError(ValueError):
    """A report whose data cannot be charted, in words for the post's author."""


@dataclass(frozen=True)
class Figure:
    """One figure: the text as the source wrote it, and the number it reads as."""

    text: str
    value: float
    prefix: str = ""
    suffix: str = ""

    @property
    def unit(self) -> Tuple[str, str]:
        return self.prefix, self.suffix


@dataclass(frozen=True)
class Table:
    headers: Tuple[str, ...]
    rows: Tuple[Tuple[str, ...], ...]


@dataclass(frozen=True)
class Series:
    """What a chart shows: its top rows, each ``(label, figure)``, and where they came from."""

    part: str
    label_header: str
    value_header: str
    rows: Tuple[Tuple[str, Figure], ...]
    # How many rows the source has; the chart shows the first ``len(rows)``.
    total: int

    def same_unit(self) -> bool:
        return len({figure.unit for _, figure in self.rows}) <= 1

    def has_negative(self) -> bool:
        return any(figure.value < 0 for _, figure in self.rows)


# ── figures ─────────────────────────────────────────────────────────────────
def parse_figure(text: Any) -> Optional[Figure]:
    """The figure ``text`` reads as, or ``None`` when it is not one number."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned or len(cleaned) > MAX_FIGURE_CHARS:
        return None
    match = _FIGURE.match(cleaned)
    if match is None:
        return None
    sign_before, prefix, sign_after, whole, decimals, suffix = match.groups()
    if sign_before and sign_after:
        return None
    value = float(whole.replace(",", "") + (decimals or ""))
    if (sign_before or sign_after or "") in NEGATIVE_SIGNS:
        value = -value
    return Figure(text=cleaned, value=value, prefix=prefix or "", suffix=suffix or "")


def number_text(value: Any) -> str:
    """A JSON number as exact positional text with comma thousands: ``1234.5`` → ``"1,234.5"``.

    A float is written as its shortest round-trip form, so the text reads back
    as the very same number (never rounded, never ``1e-05``).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{value!r} is not a number")
    if isinstance(value, int):
        return f"{value:,}"
    if not math.isfinite(value):
        raise ValueError(f"{value!r} is not a finite number")
    return format(Decimal(repr(value)), ",f")


def _json_figure(value: Any) -> Optional[Figure]:
    """A metric's figure: a finite number, or text that reads as one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return Figure(text=number_text(value), value=float(value)) if math.isfinite(value) else None
    return parse_figure(value)


# ── the markdown table ──────────────────────────────────────────────────────
def _split_cells(line: str) -> List[str]:
    """A table line's cells: split on the pipes that are not escaped, the outer ones dropped."""
    body = line.strip().replace(_ESCAPED_PIPE, _PIPE_PLACEHOLDER)
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.replace(_PIPE_PLACEHOLDER, "|") for cell in body.split("|")]


def cell_text(cell: str) -> str:
    """A cell as plain text: links and images become their text; emphasis, code marks and tags go."""
    text = _HTML_TAG.sub(" ", _LINK.sub(r"\1", cell))
    previous = None
    while previous != text:
        previous = text
        text = _UNDERSCORED.sub(r"\2", _MARKED.sub(r"\2", text))
    text = _BACKSLASH_ESCAPE.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()


def _is_delimiter(line: str, width: int) -> bool:
    cells = [cell.strip() for cell in _split_cells(line)]
    return len(cells) == width and all(_DELIMITER_CELL.match(cell) for cell in cells)


def _table_line(line: str) -> bool:
    """A line that can be a table row: it has a pipe and is not an indented code line."""
    return "|" in line and not line.startswith(("    ", "\t"))


def _rows_after(lines: List[str], width: int) -> Tuple[Tuple[str, ...], ...]:
    rows = []
    for line in lines:
        if not line.strip() or not _table_line(line):
            break
        cells = [cell_text(cell) for cell in _split_cells(line)][:width]
        rows.append(tuple(cells + [""] * (width - len(cells))))
    return tuple(rows)


def first_table(markdown: Any) -> Optional[Table]:
    """The first pipe table in ``markdown`` (GitHub's syntax), or ``None``.

    A table is a header line, a delimiter line (``| --- | :---: |``) with as
    many cells, and the lines after it while each has a pipe. A table inside a
    fenced code block does not count, nor does an indented one (code). A row
    with fewer cells than the header is padded with empty ones and a longer one
    cut, as GitHub renders them. Only the first :data:`MAX_MARKDOWN_CHARS`
    characters are read.
    """
    if not isinstance(markdown, str):
        return None
    lines = markdown[:MAX_MARKDOWN_CHARS].splitlines()
    fence: Optional[str] = None
    for i, line in enumerate(lines):
        opened = _FENCE.match(line)
        if fence is not None:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= len(fence):
                fence = None
            continue
        if opened:
            fence = opened.group(1)
            continue
        if not _table_line(line) or i + 1 >= len(lines):
            continue
        headers = tuple(cell_text(cell) for cell in _split_cells(line))
        if len(headers) <= MAX_COLUMNS and _is_delimiter(lines[i + 1], len(headers)):
            return Table(headers=headers, rows=_rows_after(lines[i + 2 :], len(headers)))
    return None


# ── series ──────────────────────────────────────────────────────────────────
def _normalised(text: str) -> str:
    return _SPACES.sub(" ", text).strip().casefold()


def _figures(table: Table, column: int, limit: int) -> List[Optional[Figure]]:
    return [parse_figure(row[column]) for row in table.rows[:limit]]


def numeric_columns(table: Table, limit: int = DEFAULT_ROWS) -> List[int]:
    """The columns after the first whose top ``limit`` cells are all figures."""
    if not table.rows:
        return []
    return [c for c in range(1, len(table.headers)) if None not in _figures(table, c, limit)]


def _series_at(table: Table, column: int, limit: int) -> Series:
    """Column ``column``'s top ``limit`` rows as figures, labelled by the first column."""
    rows = []
    for number, (row, figure) in enumerate(zip(table.rows, _figures(table, column, limit)), start=1):
        if figure is None:
            raise SeriesError(f"row {number} of {table.headers[column]!r} is not a number: {row[column]!r}")
        rows.append((row[0], figure))
    return Series(
        part=TABLE, label_header=table.headers[0], value_header=table.headers[column], rows=tuple(rows), total=len(table.rows)
    )


def _column_index(table: Table, column: str) -> int:
    wanted = _normalised(column)
    for index in range(1, len(table.headers)):
        if _normalised(table.headers[index]) == wanted:
            return index
    choices = ", ".join(repr(h) for h in table.headers[1:]) or "none"
    raise SeriesError(f"the report's table has no column {column!r} after its labels (its columns: {choices})")


def _check_chartable(table: Table) -> None:
    if len(table.headers) < 2:
        raise SeriesError("the report's table needs a column of labels and a column of figures")
    if not table.rows:
        raise SeriesError("the report's table has no rows")


def table_series(table: Table, *, column: Optional[str] = None, limit: int = DEFAULT_ROWS) -> Series:
    """The chart a table gives: its first column the labels, one column the figures, the top ``limit`` rows.

    ``column`` names the figures' column by its header (case and spacing
    ignored; the first of two alike); without it, the first column whose top
    rows are all figures.
    """
    _check_chartable(table)
    if column is not None:
        return _series_at(table, _column_index(table, column), limit)
    columns = numeric_columns(table, limit)
    if not columns:
        shown = min(limit, len(table.rows))
        raise SeriesError(f"no column of the report's table is all numbers in its first {shown} rows")
    return _series_at(table, columns[0], limit)


def metric_label(name: str) -> str:
    """A metric's name as a chart label: its words, the underscores read as spaces."""
    return _SPACES.sub(" ", name.replace("_", " ")).strip()


def _metric_figures(metrics: Any) -> List[Tuple[str, Figure]]:
    found = []
    for name, value in metrics.items() if isinstance(metrics, Mapping) else ():
        label, figure = metric_label(str(name)), _json_figure(value)
        if label and figure is not None:
            found.append((label, figure))
    return found


def metrics_series(metrics: Any, *, limit: int = DEFAULT_ROWS) -> Series:
    """The chart a report's metrics give: each figure, named by its key, in the stored order.

    Only a figure counts: a finite number, or text that reads as one number.
    """
    figures = _metric_figures(metrics)
    if not figures:
        raise SeriesError("the report's metrics carry no figures to chart")
    return Series(part=METRICS, label_header="", value_header="", rows=tuple(figures[:limit]), total=len(figures))


def candidate_series(table: Optional[Table], metrics: Any, *, limit: int = DEFAULT_ROWS) -> List[Series]:
    """Every chart a report can give: one per numeric column of its table, then its metrics."""
    found: List[Series] = []
    if table is not None and len(table.headers) >= 2:
        found += [_series_at(table, column, limit) for column in numeric_columns(table, limit)]
    figures = _metric_figures(metrics)
    if figures:
        found.append(Series(part=METRICS, label_header="", value_header="", rows=tuple(figures[:limit]), total=len(figures)))
    return found


__all__ = [
    "DEFAULT_ROWS",
    "FIGURE_PATTERN",
    "Figure",
    "METRICS",
    "PARTS",
    "Series",
    "SeriesError",
    "TABLE",
    "Table",
    "candidate_series",
    "cell_text",
    "first_table",
    "metric_label",
    "metrics_series",
    "number_text",
    "numeric_columns",
    "parse_figure",
    "table_series",
]
