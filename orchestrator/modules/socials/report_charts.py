"""PRD-251 S1.7 (D7): a chart bound to a report.

A social template with a ``data`` block (``core/chart_binding.py``; the seeded
Infographic) charts a report's data. This module reads that data in the
caller's workspace, two ways:

* :func:`bind_report` gives what a post stores to chart a report: the report's
  top rows as the template's row variables, each figure a claim bound to the
  report (D7), the source chip naming the report as it resolves
  (``sources.resolve``), the figures' column header as the subtitle, and the
  chart kind. ``GET /api/socials/sources/reports/{id}/chart`` answers it for the
  composer, and an agent's draft tool can call it the same way.
* :func:`check_bound_chart` runs before a render: a chart whose first figure is
  bound to a report must show that report's rows exactly as the report has them
  now, and its chip must name the report. Otherwise the render is refused and
  says which row differs. So every number on a rendered chart comes from the
  source it is bound to, whoever typed the post.

The report is an ``agent_reports`` row of the workspace that is not deleted,
read with raw SQL as ``services/report_service.py`` reads it (the table has no
ORM model). Its data is the first markdown table in its file, read through the
workspace worker, or the figures in its ``metrics`` JSONB
(``core/report_tables.py``). The worker's own error text stays in the log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import UUID

import sqlalchemy as sa

from core.chart_binding import (
    ChartBindingError,
    DataSpec,
    chart_values,
    chip_text,
    default_kind,
    expected_rows,
    max_chars_of,
    shown_rows,
    spec_of,
)
from core.report_tables import (
    METRICS,
    PARTS,
    TABLE,
    Series,
    SeriesError,
    candidate_series,
    first_table,
    metrics_series,
    table_series,
)
from core.workspace_client import WorkspaceClient
from modules.socials import render, service
from modules.socials import sources as post_sources

logger = logging.getLogger(__name__)

REPORT_KIND = "report"
# A figures' column is named by its header: at most this long.
COLUMN_MAX_CHARS = 200

_REPORT_DATA = sa.text(
    """
    SELECT r.file_path, r.metrics
      FROM agent_reports r
     WHERE r.id = :id AND r.workspace_id = :workspace_id AND r.deleted_at IS NULL
    """
)


class ReportNotFound(service.SocialsError):
    """No report with this id in the caller's workspace (or it was deleted)."""


class ChartNotBindable(service.SocialsError):
    """The report's data cannot fill this template's chart, and why."""


class ReportUnreadable(service.SocialsError):
    """The report's file could not be read from the workspace right now."""


class ChartNotFromItsReport(render.NotRenderable):
    """A chart bound to a report that no longer shows the report's rows as the report has them."""


@dataclass(frozen=True)
class ReportData:
    source: post_sources.ResolvedSource
    file_path: Optional[str]
    metrics: Dict[str, Any]


@dataclass(frozen=True)
class ChartBinding:
    """What a post stores to chart a report: ``variables`` and ``sources`` in the post's own shapes."""

    variables: Dict[str, Dict[str, Any]]
    sources: Dict[str, Dict[str, Any]]
    source: post_sources.ResolvedSource
    series: Series
    chart: str
    rows: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variables": self.variables,
            "sources": self.sources,
            "source": self.source.to_dict(),
            "part": self.series.part,
            "column": self.series.value_header,
            "chart": self.chart,
            "rows": [{"label": label, "value": value} for label, value in self.rows if label or value],
            "shown": len(self.series.rows),
            "total": self.series.total,
        }


# ── the report ──────────────────────────────────────────────────────────────
def report_data(db: Any, workspace_id: Any, report_id: Any) -> ReportData:
    """The report as it resolves in the caller's workspace, with where its data lives; :class:`ReportNotFound`."""
    try:
        resolved = post_sources.resolve(db, workspace_id, {"kind": REPORT_KIND, "ref": report_id})
    except post_sources.SourceNotResolved as exc:
        raise ReportNotFound(str(exc)) from exc
    row = db.execute(_REPORT_DATA, {"id": str(UUID(resolved.ref)), "workspace_id": str(workspace_id)}).first()
    if row is None:
        raise ReportNotFound("no report with this id in this workspace")
    return ReportData(source=resolved, file_path=row.file_path, metrics=post_sources.report_metrics(row.metrics))


async def report_markdown(workspace_id: Any, report: ReportData) -> str:
    """The report's file, read through the workspace worker; :class:`ReportUnreadable` when it cannot be."""
    if not report.file_path:
        return ""
    result = await WorkspaceClient(str(workspace_id)).read_file(report.file_path)
    if not result.get("success"):
        logger.warning(
            "[Socials] reading report %s (%s) failed: %s", report.source.ref, report.file_path, result.get("error")
        )
        raise ReportUnreadable("The report's file could not be read right now. Try again in a few minutes.")
    content = result.get("content")
    return content if isinstance(content, str) else ""


def _refusing(build, *args, **kwargs) -> Series:
    try:
        return build(*args, **kwargs)
    except SeriesError as exc:
        raise ChartNotBindable(str(exc)) from exc


async def _series(
    workspace_id: Any, report: ReportData, *, part: Optional[str], column: Optional[str], limit: int
) -> Series:
    """The series the caller asked for: the report's table (its ``column``), else its metrics."""
    if part not in (None,) + PARTS:
        raise ChartNotBindable(f"part must be one of {', '.join(PARTS)}")
    if part == METRICS:
        if column is not None:
            raise ChartNotBindable("a column names a figures' column of the report's table; its metrics have none")
        return _refusing(metrics_series, report.metrics, limit=limit)
    table = first_table(await report_markdown(workspace_id, report))
    if table is not None:
        return _refusing(table_series, table, column=column, limit=limit)
    if part == TABLE or column is not None:
        raise ChartNotBindable("the report's file has no data table (a markdown table)")
    try:
        return metrics_series(report.metrics, limit=limit)
    except SeriesError:
        raise ChartNotBindable("the report has no data table, and its metrics carry no figures to chart") from None


# ── binding a chart to a report ─────────────────────────────────────────────
def _chart_default(spec: DataSpec, schema: Mapping[str, Any]) -> Optional[str]:
    variable = schema.get(spec.chart) if spec.chart else None
    return variable.get("default") if isinstance(variable, Mapping) else None


async def bind_report(
    db: Any,
    workspace_id: Any,
    report_id: Any,
    blocks: Mapping[str, Any],
    *,
    chart: Optional[str] = None,
    column: Optional[str] = None,
    part: Optional[str] = None,
) -> ChartBinding:
    """The chart of ``blocks`` (a checked social template's) filled from a report of the caller's workspace.

    ``part`` is ``table`` or ``metrics`` (by default the report's table when its
    file has one, else its metrics), ``column`` the table's figures' column by
    its header (by default its first numeric column), ``chart`` the kind (by
    default the template's own when it fits the figures, else the number grid).
    Every row variable is set, the ones past the report's rows to empty; the
    figures that are shown are claims, each bound to the report.
    """
    spec = spec_of(blocks)
    if spec is None:
        raise ChartNotBindable("this template has no chart to fill: choose the Infographic or another chart template")
    schema = blocks["variables_schema"]
    report = report_data(db, workspace_id, report_id)
    series = await _series(workspace_id, report, part=part, column=column, limit=spec.rows)
    kind = chart.strip().lower() if isinstance(chart, str) else default_kind(series, _chart_default(spec, schema))
    try:
        values = chart_values(spec, schema, series, kind=kind, title=report.source.title, as_of=report.source.as_of)
        rows = expected_rows(spec, schema, series)
    except ChartBindingError as exc:
        raise ChartNotBindable(str(exc)) from exc
    claims = set(spec.value_names())
    source = {"kind": REPORT_KIND, "ref": report.source.ref, "as_of": report.source.as_of}
    return ChartBinding(
        variables={name: {"value": value, "claim": name in claims and bool(value)} for name, value in values.items()},
        sources={name: dict(source) for name in spec.value_names() if values.get(name)},
        source=report.source,
        series=series,
        chart=kind,
        rows=rows,
    )


# ── the check before a render ───────────────────────────────────────────────
def _expected(spec: DataSpec, schema: Mapping[str, Any], series: Series) -> Optional[List[Tuple[str, str]]]:
    try:
        return expected_rows(spec, schema, series)
    except ChartBindingError:
        return None


def _difference(shown: List[Tuple[str, str]], expected: List[Tuple[str, str]]) -> str:
    for n, (was, now) in enumerate(zip(shown, expected), start=1):
        if was != now:
            return f"row {n} shows {' '.join(filter(None, was)) or 'nothing'!r}; the report has {' '.join(filter(None, now)) or 'nothing'!r}"
    return "its rows differ from the report's"


def _mismatch(title: str, shown: List[Tuple[str, str]], candidates: List[List[Tuple[str, str]]]) -> ChartNotFromItsReport:
    if not candidates:
        return ChartNotFromItsReport(
            f"the chart is bound to the report {title!r}, which has no data to chart now: bind it to its report again"
        )
    closest = max(candidates, key=lambda rows: sum(a == b for a, b in zip(shown, rows)))
    return ChartNotFromItsReport(
        f"the chart does not show the rows of its report {title!r} as the report has them now "
        f"({_difference(shown, closest)}): bind the chart to its report again"
    )


async def check_bound_chart(
    db: Any,
    workspace_id: Any,
    blocks: Optional[Mapping[str, Any]],
    post_source_map: Optional[Mapping[str, Any]],
    values: Mapping[str, Any],
) -> None:
    """Refuse the render of a chart bound to a report unless it shows that report's rows as they are now.

    ``blocks`` is the post's template (checked), ``values`` the variables the
    render fills (the bundle's). A chart is bound to a report when its first
    figure's source is a report: then every row must be the report's own (its
    table's rows in one numeric column, or its metrics), the ones past the
    report's rows empty, and the chip must name the report. A chart bound to
    anything else is checked as any claim is, at approval (D7).
    """
    spec = spec_of(blocks) if blocks else None
    if spec is None:
        return
    source = (post_source_map or {}).get(spec.value_name(1))
    if not isinstance(source, Mapping) or source.get("kind") != REPORT_KIND:
        return
    try:
        report = report_data(db, workspace_id, source.get("ref"))
    except ReportNotFound as exc:
        raise ChartNotFromItsReport(f"the chart is bound to a report that cannot be found ({exc}): bind it to a report again") from exc
    schema = blocks["variables_schema"]
    shown = shown_rows(spec, values)
    candidates = [rows for series in candidate_series(None, report.metrics, limit=spec.rows) if (rows := _expected(spec, schema, series))]
    if shown not in candidates:
        table = first_table(await report_markdown(workspace_id, report))
        candidates += [rows for series in candidate_series(table, None, limit=spec.rows) if (rows := _expected(spec, schema, series))]
    if shown not in candidates:
        raise _mismatch(report.source.title, shown, candidates)
    chip = chip_text(report.source.title, report.source.as_of, max_chars_of(schema, spec.source))
    if str(values.get(spec.source) or "").strip() != chip:
        raise ChartNotFromItsReport(f"the chart's source chip must name its report as it is now: {chip!r}")



__all__ = [
    "COLUMN_MAX_CHARS",
    "ChartBinding",
    "ChartNotBindable",
    "ChartNotFromItsReport",
    "ReportNotFound",
    "ReportUnreadable",
    "bind_report",
    "check_bound_chart",
    "report_data",
    "report_markdown",
]
