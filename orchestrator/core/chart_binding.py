"""A social template's chart, bound to a report (PRD-251 S1.7, D7).

A template that charts a report's data says which of its variables hold the
chart in a ``data`` block, beside its html (the template contract,
``core/social_templates.py``, checks it):

    "data": {
      "rows": 5,                  the rows the chart shows: the source's top rows
      "label": "row_{n}_label",   each row's label variable ({n} is 1 to rows)
      "value": "row_{n}_value",   each row's figure variable, a claim (D7)
      "source": "source_label",   the source chip: the report the rows come from
      "header": "subtitle",       optional: takes the figures' column header
      "chart": "chart"            optional: the variable that picks bar, line or grid
    }

:func:`chart_values` fills those variables from a report's series
(``core/report_tables.py``): the top rows in the source's order, each figure as
the source wrote it, and the chip naming the report. :func:`expected_rows` and
:func:`shown_rows` are the two sides of the check a render runs on a chart bound
to a report: what the report gives now, and what the post shows.

The chart kinds: a bar compares magnitudes, so its figures share one unit and
none is negative (a bar grows from zero); a line needs one unit too; the number
grid shows any figures as they are. :func:`kind_problem` says why a kind does
not fit a series, and the template's own script falls back to the grid on the
same rules, so a hand-filled chart never draws a comparison its figures cannot
make.

Pure: the standard library only (the media-render CI job binds the fixture
report with this very code).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.report_tables import METRICS, Series

BAR, LINE, GRID = "bar", "line", "grid"
CHART_KINDS = (BAR, LINE, GRID)
DATA_KEYS = ("rows", "label", "value", "source", "header", "chart")
REQUIRED_DATA_KEYS = ("rows", "label", "value", "source")
MAX_DATA_ROWS = 10
ROW_NUMBER = "{n}"
ELLIPSIS = "…"
CHIP_SEPARATOR = " · "
ISO_DATE_CHARS = 10
# Mirrors core/social_templates.VARIABLE_NAME (this module may not import the contract: it imports this one).
_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class ChartBindingError(ValueError):
    """A series the template cannot show, in words for the post's author."""


def _error(field: str, message: str) -> Dict[str, str]:
    return {"field": field, "message": message}


@dataclass(frozen=True)
class DataSpec:
    """A checked ``data`` block."""

    rows: int
    label: str
    value: str
    source: str
    header: Optional[str] = None
    chart: Optional[str] = None

    def label_name(self, n: int) -> str:
        return self.label.replace(ROW_NUMBER, str(n))

    def value_name(self, n: int) -> str:
        return self.value.replace(ROW_NUMBER, str(n))

    def value_names(self) -> List[str]:
        return [self.value_name(n) for n in range(1, self.rows + 1)]


def spec_of(blocks: Mapping[str, Any]) -> Optional[DataSpec]:
    """The template's ``data`` block, or ``None`` when it charts nothing (a checked template's blocks)."""
    data = blocks.get("data") if isinstance(blocks, Mapping) else None
    if not isinstance(data, Mapping):
        return None
    return DataSpec(**{key: data[key] for key in DATA_KEYS if key in data})


# ── the contract ────────────────────────────────────────────────────────────
def _text_variable(schema: Mapping[str, Any], name: Any) -> Optional[Mapping[str, Any]]:
    spec = schema.get(name) if isinstance(name, str) else None
    return spec if isinstance(spec, Mapping) and spec.get("type") == "text" else None


def _pattern_errors(key: str, pattern: Any, rows: int, schema: Mapping[str, Any]) -> List[Dict[str, str]]:
    where = f"data.{key}"
    if not isinstance(pattern, str) or pattern.count(ROW_NUMBER) != 1:
        return [_error(where, f"must be a variable name with {ROW_NUMBER} once, e.g. row_{ROW_NUMBER}_{key}")]
    errors = []
    for n in range(1, rows + 1):
        name = pattern.replace(ROW_NUMBER, str(n))
        spec = _text_variable(schema, name)
        if not _VARIABLE_NAME.match(name) or spec is None:
            errors.append(_error(where, f"row {n} needs a text variable {name} in variables_schema"))
        elif key == "value" and spec.get("claim") is not True:
            errors.append(_error(where, f"{name} holds a figure: mark it claim: true (D7)"))
        elif key == "label" and spec.get("claim") is True:
            errors.append(_error(where, f"{name} is a row's label, not a claim: leave claim out"))
    return errors


def _named_errors(data: Mapping[str, Any], schema: Mapping[str, Any]) -> List[Dict[str, str]]:
    errors = []
    for key in ("source", "header", "chart"):
        if key not in data:
            continue
        spec = _text_variable(schema, data[key])
        if spec is None:
            errors.append(_error(f"data.{key}", "must name a text variable of variables_schema"))
        elif spec.get("claim") is True:
            errors.append(_error(f"data.{key}", f"{data[key]} is filled by the binding, not a claim: leave claim out"))
        elif key == "chart" and "default" in spec and spec["default"] not in CHART_KINDS:
            errors.append(_error("data.chart", f"{data[key]}'s default must be one of {list(CHART_KINDS)}"))
    return errors


def data_errors(data: Any, schema: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Why a template's ``data`` block cannot bind its chart ([] when it can)."""
    if not isinstance(data, Mapping):
        return [_error("data", 'must be an object such as {"rows": 5, "label": "row_{n}_label", …}')]
    errors = [_error(f"data.{key}", f"is not a data setting ({', '.join(DATA_KEYS)})") for key in data if key not in DATA_KEYS]
    errors += [_error(f"data.{key}", "is required") for key in REQUIRED_DATA_KEYS if key not in data]
    rows = data.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or not 1 <= rows <= MAX_DATA_ROWS:
        return errors + [_error("data.rows", f"must be a whole number from 1 to {MAX_DATA_ROWS}")]
    for key in ("label", "value"):
        if key in data:
            errors += _pattern_errors(key, data[key], rows, schema)
    if "label" in data and data.get("label") == data.get("value"):
        errors.append(_error("data.value", "must name other variables than data.label"))
    return errors + _named_errors(data, schema)


# ── filling the chart ───────────────────────────────────────────────────────
def fit_text(text: str, limit: Optional[int]) -> str:
    """``text`` cut to ``limit`` characters, an ellipsis marking the cut."""
    if limit is None or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + ELLIPSIS


def max_chars_of(schema: Mapping[str, Any], name: str) -> Optional[int]:
    """The most characters variable ``name`` holds (``None``: the contract's own limit)."""
    spec = schema.get(name)
    limit = spec.get("max_chars") if isinstance(spec, Mapping) else None
    return limit if isinstance(limit, int) and not isinstance(limit, bool) else None


def chip_text(title: str, as_of: Optional[str], limit: Optional[int] = None) -> str:
    """The source chip: the report's title and the day it was made, ``Weekly sales · 2026-09-21``.

    A long title is cut, never the date.
    """
    day = (as_of or "")[:ISO_DATE_CHARS]
    tail = f"{CHIP_SEPARATOR}{day}" if day else ""
    room = None if limit is None else max(1, limit - len(tail))
    return fit_text((title or "").strip(), room) + tail if (title or "").strip() else day


def kind_problem(kind: Any, series: Series) -> Optional[str]:
    """Why ``kind`` cannot show ``series`` truthfully, or ``None``."""
    if kind not in CHART_KINDS:
        return f"chart must be one of {', '.join(CHART_KINDS)}"
    if kind == GRID:
        return None
    if series.part == METRICS:
        return "a report's metrics are different measures: they show as a number grid"
    if len(series.rows) < 2:
        return "one figure is a number, not a chart: choose the number grid"
    if not series.same_unit():
        units = sorted({"".join(figure.unit) or "none" for _, figure in series.rows})
        return f"the figures are in different units ({', '.join(units)}): a {kind} chart cannot compare them; choose the number grid"
    if kind == BAR and series.has_negative():
        return "a bar grows from zero, and these figures include a negative one: choose a line or the number grid"
    return None


def default_kind(series: Series, template_default: Any) -> str:
    """The template's own kind when it fits the series, else the number grid (which always does)."""
    return template_default if kind_problem(template_default, series) is None else GRID


def expected_rows(spec: DataSpec, schema: Mapping[str, Any], series: Series) -> List[Tuple[str, str]]:
    """Each row as the chart shows it: ``(label, figure)``, cut as the template's variables hold them; empty past the series."""
    rows: List[Tuple[str, str]] = []
    for n in range(1, spec.rows + 1):
        if n > len(series.rows):
            rows.append(("", ""))
            continue
        label, figure = series.rows[n - 1]
        limit = max_chars_of(schema, spec.value_name(n))
        if limit is not None and len(figure.text) > limit:
            raise ChartBindingError(
                f"row {n}'s figure {figure.text!r} is longer than the {limit} characters the template shows"
            )
        rows.append((fit_text(label, max_chars_of(schema, spec.label_name(n))), figure.text))
    return rows


def chart_values(
    spec: DataSpec, schema: Mapping[str, Any], series: Series, *, kind: str, title: str, as_of: Optional[str]
) -> Dict[str, str]:
    """The template's chart variables for ``series`` shown as ``kind``: rows, chip, header and kind.

    Every row variable is set, those past the series to empty, so a chart bound
    again to a shorter table keeps no stale row.
    """
    problem = kind_problem(kind, series)
    if problem:
        raise ChartBindingError(problem)
    values: Dict[str, str] = {}
    for n, (label, figure) in enumerate(expected_rows(spec, schema, series), start=1):
        values[spec.label_name(n)] = label
        values[spec.value_name(n)] = figure
    values[spec.source] = chip_text(title, as_of, max_chars_of(schema, spec.source))
    if spec.header and series.value_header:
        values[spec.header] = fit_text(series.value_header, max_chars_of(schema, spec.header))
    if spec.chart:
        values[spec.chart] = kind
    return values


def shown_rows(spec: DataSpec, values: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """Each row as a post shows it, ``(label, figure)``, from its resolved values."""
    def text(name: str) -> str:
        value = values.get(name)
        return value.strip() if isinstance(value, str) else ""

    return [(text(spec.label_name(n)), text(spec.value_name(n))) for n in range(1, spec.rows + 1)]


__all__ = [
    "BAR",
    "CHART_KINDS",
    "ChartBindingError",
    "DataSpec",
    "GRID",
    "LINE",
    "chart_values",
    "chip_text",
    "data_errors",
    "default_kind",
    "expected_rows",
    "fit_text",
    "kind_problem",
    "max_chars_of",
    "shown_rows",
    "spec_of",
]
