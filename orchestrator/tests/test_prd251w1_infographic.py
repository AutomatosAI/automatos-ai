"""PRD-251 Wave 1, US-113 (S1.7) — the infographic: a chart bound to a report (D7).

Pins:

* **The parser** (``core/report_tables.py``, a small stdlib one): the first
  markdown table of a report file as GitHub renders it (a fenced or indented
  table is code, inline marks are dropped, an escaped pipe stays in its cell,
  a ragged row is padded or cut); a figure keeps the text its source wrote and
  reads as its number; a JSON number is written exactly; a series is the top
  rows in the source's order, of one numeric column or of the metrics.
* **The contract.** A template's ``data`` block names the variables that hold
  its chart; the contract refuses one that cannot bind.
* **The template.** The Infographic is the eighth image starter: a bar chart,
  a line or a number grid of five rows, each figure a claim. Its script reads a
  figure by the very pattern the platform uses and writes no word or number of
  its own; its sample chart IS a report's binding.
* **AC 1.** Given a report with a table, the chart shows its top 5 rows, each
  figure a claim bound to the report, and the chip names the report as it
  resolves; the post saves and renders.
* **AC 3.** Every number on the rendered chart comes from the bound source: the
  render bundle's figures are the report's cells, value for value, and a render
  of a chart that no longer matches its report (an edited figure, a changed
  report, an edited chip, a deleted report) is refused with the row that
  differs; a report whose file cannot be read refuses it with 503. A chart
  bound to anything else renders as any claim does.
* **The route.** ``GET /api/socials/sources/reports/{id}/chart``: the caller's
  workspace only, each refusal saying why, and in the committed manifest.
* **The CI driver** (AC 1 and AC 2 in CI) binds fixture reports with this very
  code and checks every figure it hands media-render is the report's own cell.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
_ROOT = _ORCH.parent
_MEDIA_RENDER = _ROOT / "services" / "media-render"
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
# media-render's bundle parser (standard library only), found LAST so no orchestrator module is shadowed.
if str(_MEDIA_RENDER) not in sys.path:
    sys.path.append(str(_MEDIA_RENDER))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
import api.socials as socials_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.socials.render as render  # noqa: E402
import modules.socials.report_charts as report_charts  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.chart_binding import (  # noqa: E402
    ChartBindingError,
    chart_values,
    chip_text,
    data_errors,
    default_kind,
    kind_problem,
    spec_of,
)
from core.database.database import get_db  # noqa: E402
from core.models.core import DocumentTemplate, LLMUsage  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from core.report_tables import (  # noqa: E402
    FIGURE_PATTERN,
    Figure,
    Series,
    SeriesError,
    Table,
    candidate_series,
    cell_text,
    first_table,
    metrics_series,
    number_text,
    parse_figure,
    table_series,
)
from core.social_templates import (  # noqa: E402
    SocialTemplateError,
    claim_names,
    resolve_variables,
    validate_social_blocks,
)
from media_render.bundle import parse_bundle  # noqa: E402  (media-render's own parser: stdlib only)
from media_render.config import load_settings  # noqa: E402
from modules.documents.social_starters import SOCIAL_IMAGE_STARTER_SLUGS, social_starters  # noqa: E402

# Hex with letters, so SQLite keeps every UUID column as text.
WS = uuid.UUID("00000000-0000-0000-0000-0000000001b3")
WS_OTHER = uuid.UUID("00000000-0000-0000-0000-0000000001b4")
CREATED = datetime(2026, 9, 1, 9, 0)
MADE = datetime(2026, 9, 21, 9, 30, tzinfo=timezone.utc)
MANIFEST = _ORCH / "reports" / "route-manifest.json"
CHART_ROUTE = "/api/socials/sources/reports/{report_id}/chart"
TITLE = "September social report"
CHANNELS = (
    "# September social report\n\n"
    "Every channel we posted on in September, busiest first.\n\n"
    "| Channel | Posts | Reach | Engagement rate |\n"
    "| --- | ---: | ---: | ---: |\n"
    "| Instagram | 14 | 48,210 | 5.2% |\n"
    "| LinkedIn | 9 | 21,480 | 3.9% |\n"
    "| TikTok | 6 | 19,305 | 6.8% |\n"
    "| Threads | 8 | 7,940 | 2.4% |\n"
    "| X | 11 | 6,115 | 1.7% |\n"
    "| YouTube Shorts | 2 | 3,020 | 4.4% |\n\n"
    "Next month: two more Shorts a week.\n"
)
REACH = [("Instagram", "48,210"), ("LinkedIn", "21,480"), ("TikTok", "19,305"), ("Threads", "7,940"), ("X", "6,115")]


def _infographic():
    (starter,) = [s for s in social_starters("social_image") if s["slug"] == "infographic"]
    return starter


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def test_the_first_table_is_read_as_github_renders_it():
    markdown = (
        "# Report\n\n"
        "```\n| Code | 1 |\n|---|---|\n| in a fence | 2 |\n```\n\n"
        "    | Indented | 1 |\n    |---|---|\n\n"
        "A | pipe in prose with no delimiter after it\n\n"
        "Channel | **Reach** | [Rate](https://example.com/rate)\n"
        ":-- | --: | :-:\n"
        "| _Instagram_ | 48,210 | `5.2%` |\n"
        "| X \\| Twitter | ~~6,000~~ 6,115 | 1.7% | an extra cell |\n"
        "| input_tokens | 3,020 |\n"
        "| <b>Threads</b> | 7,940 | 2.4% |\n"
        "Not a row, and the table ended at it\n"
        "| After | 1 | 2 |\n"
    )
    table = first_table(markdown)
    assert table == Table(
        headers=("Channel", "Reach", "Rate"),
        rows=(
            ("Instagram", "48,210", "5.2%"),
            ("X | Twitter", "6,000 6,115", "1.7%"),
            ("input_tokens", "3,020", ""),
            ("Threads", "7,940", "2.4%"),
        ),
    )


@pytest.mark.parametrize(
    "markdown",
    [
        pytest.param("No table here, only prose.", id="prose"),
        pytest.param("| a | b |\n| c | d |\n", id="no-delimiter-line"),
        pytest.param("| a | b |\n|---|\n| c | d |\n", id="delimiter-of-another-width"),
        pytest.param("~~~\n| a | b |\n|---|---|\n~~~\n", id="only-in-a-fence"),
        pytest.param(None, id="not-text"),
    ],
)
def test_text_without_a_table_has_none(markdown):
    assert first_table(markdown) is None


def test_a_cell_reads_as_its_plain_text():
    assert cell_text("  **Reach**  <br> *this* month  ") == "Reach this month"
    assert cell_text("![logo](a.png) [Site](https://x.test)") == "logo Site"
    assert cell_text("snake_case_name and _emphasis_") == "snake_case_name and emphasis"
    assert cell_text("5\\*3 \\[x\\]") == "5*3 [x]"


@pytest.mark.parametrize(
    "text, value, unit",
    [
        ("1,284", 1284.0, ("", "")),
        ("48,210", 48210.0, ("", "")),
        ("$1,284.50", 1284.5, ("$", "")),
        ("US$5", 5.0, ("US$", "")),
        ("€ 12", 12.0, ("€", "")),
        ("USD 1,200", 1200.0, ("USD", "")),
        ("−3.2%", -3.2, ("", "%")),
        ("-$5", -5.0, ("$", "")),
        ("$-5", -5.0, ("$", "")),
        ("+12%", 12.0, ("", "%")),
        ("3x", 3.0, ("", "x")),
        ("2×", 2.0, ("", "×")),
        ("1.2M", 1.2, ("", "M")),
        ("12 ms", 12.0, ("", "ms")),
        ("4.2 pts", 4.2, ("", "pts")),
        ("2026", 2026.0, ("", "")),
        ("  7,940 ", 7940.0, ("", "")),
    ],
)
def test_a_figure_keeps_its_text_and_reads_as_its_number(text, value, unit):
    figure = parse_figure(text)
    assert figure is not None and figure.text == text.strip()
    assert figure.value == pytest.approx(value) and figure.unit == unit


@pytest.mark.parametrize("text", ["Q1", "n/a", "1,23", "1.2.3", "--5", "12k users", "", "   ", "5 %%", "1 234", None, 12])
def test_what_is_not_one_number_is_no_figure(text):
    assert parse_figure(text) is None


@pytest.mark.parametrize(
    "value, text",
    [
        (1234, "1,234"),
        (-7, "-7"),
        (1234.5, "1,234.5"),
        (0.0421, "0.0421"),
        (1e-05, "0.00001"),
        (3.0, "3.0"),
        (10**20, "100,000,000,000,000,000,000"),
    ],
)
def test_a_json_number_is_written_exactly_and_reads_back_as_itself(value, text):
    assert number_text(value) == text
    assert parse_figure(text).value == pytest.approx(float(value), rel=0, abs=0)


def test_number_text_refuses_what_is_not_a_finite_number():
    for bad in (True, "1", None):
        with pytest.raises(TypeError):
            number_text(bad)
    with pytest.raises(ValueError):
        number_text(float("inf"))


def test_the_series_is_the_top_rows_of_one_numeric_column_in_the_sources_order():
    table = first_table(CHANNELS)
    first = table_series(table)
    assert (first.part, first.label_header, first.value_header, first.total) == ("table", "Channel", "Posts", 6)
    assert [(label, figure.text) for label, figure in first.rows] == [
        ("Instagram", "14"), ("LinkedIn", "9"), ("TikTok", "6"), ("Threads", "8"), ("X", "11"),
    ]
    reach = table_series(table, column="  reach ")
    assert reach.value_header == "Reach" and [(label, figure.text) for label, figure in reach.rows] == REACH
    assert len(table_series(table, column="Reach", limit=3).rows) == 3
    # A text column is skipped: the first column whose top rows are all figures.
    mixed = first_table("| Name | Note | Score |\n|---|---|---|\n| A | fine | 3 |\n| B | 4 | 5 |\n")
    assert table_series(mixed).value_header == "Score"


@pytest.mark.parametrize(
    "markdown, column, message",
    [
        pytest.param(CHANNELS, "Clicks", "no column 'Clicks'", id="unknown-column"),
        pytest.param(CHANNELS, "Channel", "no column 'Channel'", id="the-label-column"),
        pytest.param("| A | B |\n|---|---|\n| x | 1 |\n| y | n/a |\n", "B", "row 2 of 'B' is not a number: 'n/a'", id="a-text-cell"),
        pytest.param("| A | B |\n|---|---|\n| x | one |\n", None, "no column of the report's table is all numbers", id="no-numbers"),
        pytest.param("| A |\n|---|\n| 1 |\n", None, "a column of labels and a column of figures", id="one-column"),
        pytest.param("| A | B |\n|---|---|\n", None, "has no rows", id="no-rows"),
    ],
)
def test_a_table_that_cannot_chart_says_why(markdown, column, message):
    with pytest.raises(SeriesError, match=re.escape(message)):
        table_series(first_table(markdown), column=column)


def test_metrics_give_their_figures_in_stored_order_named_by_their_keys():
    metrics = {
        "model": "gpt-4o", "llm_calls": 3, "cost_usd": 0.0421, "ok": True, "nested": {"a": 1},
        "engagement_rate": "4.2%", "started_at": "2026-09-21T09:30:00", "input_tokens": 12045, "inf": float("inf"),
    }
    series = metrics_series(metrics)
    assert (series.part, series.value_header, series.total) == ("metrics", "", 4)
    assert [(label, figure.text) for label, figure in series.rows] == [
        ("llm calls", "3"), ("cost usd", "0.0421"), ("engagement rate", "4.2%"), ("input tokens", "12,045"),
    ]
    with pytest.raises(SeriesError, match="no figures"):
        metrics_series({"model": "gpt-4o", "ok": False})
    with pytest.raises(SeriesError):
        metrics_series(None)


def test_every_chart_a_report_can_give_one_per_numeric_column_then_its_metrics():
    table = first_table(CHANNELS)
    headers = [series.value_header for series in candidate_series(table, {"followers": 16340})]
    assert headers == ["Posts", "Reach", "Engagement rate", ""]
    assert candidate_series(None, {}) == []


# ---------------------------------------------------------------------------
# The contract: the data block
# ---------------------------------------------------------------------------

CHART_SCHEMA = {
    **{f"row_{n}_label": {"type": "text", "default": "", "max_chars": 12} for n in range(1, 4)},
    **{f"row_{n}_value": {"type": "text", "default": "", "claim": True, "max_chars": 8} for n in range(1, 4)},
    "chip": {"type": "text", "max_chars": 30},
    "kind": {"type": "text", "default": "bar"},
    "sub": {"type": "text", "default": "", "max_chars": 10},
    "number": {"type": "number", "default": 1},
}
DATA = {"rows": 3, "label": "row_{n}_label", "value": "row_{n}_value", "source": "chip", "header": "sub", "chart": "kind"}


def _blocks(data=DATA, schema=CHART_SCHEMA):
    names = " ".join(f"{{{{ {name} }}}}" for name in schema)
    html = (
        '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-start="0" '
        f'data-duration="1" data-width="{{{{ size.width }}}}" data-height="{{{{ size.height }}}}"><p>{names}</p></div></body></html>'
    )
    return {"html": html, "variables_schema": schema, "sizes": ["1080x1350"], "data": data}


@pytest.mark.parametrize(
    "data, message",
    [
        pytest.param(["rows"], "must be an object", id="not-an-object"),
        pytest.param({**DATA, "column": "Reach"}, "is not a data setting", id="unknown-key"),
        pytest.param({key: v for key, v in DATA.items() if key != "source"}, "data.source: is required", id="no-source"),
        pytest.param({**DATA, "rows": 0}, "whole number from 1 to 10", id="no-rows"),
        pytest.param({**DATA, "rows": 11}, "whole number from 1 to 10", id="too-many-rows"),
        pytest.param({**DATA, "rows": True}, "whole number from 1 to 10", id="rows-a-boolean"),
        pytest.param({**DATA, "rows": 4}, "row 4 needs a text variable row_4_label", id="a-row-without-variables"),
        pytest.param({**DATA, "label": "row_label"}, "with {n} once", id="no-row-number"),
        pytest.param({**DATA, "value": "row_{n}_label"}, "holds a figure: mark it claim", id="figures-not-claims"),
        pytest.param({**DATA, "label": "row_{n}_value", "value": "row_{n}_value"}, "is a row's label, not a claim", id="labels-as-claims"),
        pytest.param({**DATA, "source": "row_1_value"}, "not a claim", id="the-chip-a-claim"),
        pytest.param({**DATA, "source": "nope"}, "data.source: must name a text variable", id="an-undeclared-chip"),
        pytest.param({**DATA, "header": "number"}, "data.header: must name a text variable", id="a-number-header"),
    ],
)
def test_the_contract_refuses_a_data_block_that_cannot_bind(data, message):
    with pytest.raises(SocialTemplateError, match=re.escape(message)):
        validate_social_blocks(_blocks(data), "social_image")


def test_the_contract_refuses_a_chart_default_that_is_no_kind():
    schema = {**CHART_SCHEMA, "kind": {"type": "text", "default": "pie"}}
    assert data_errors(DATA, schema) == [{"field": "data.chart", "message": "kind's default must be one of ['bar', 'line', 'grid']"}]


def test_a_good_data_block_passes_and_is_kept():
    checked = validate_social_blocks(_blocks(), "social_image")
    assert checked["data"] == DATA
    spec = spec_of(checked)
    assert (spec.rows, spec.value_names(), spec.label_name(2)) == (3, ["row_1_value", "row_2_value", "row_3_value"], "row_2_label")
    assert spec_of({"html": "x"}) is None


# ---------------------------------------------------------------------------
# Filling a chart
# ---------------------------------------------------------------------------


def _series(rows, part="table", header="Reach", total=None):
    figures = tuple((label, parse_figure(text)) for label, text in rows)
    return Series(part=part, label_header="Channel", value_header=header, rows=figures, total=total or len(rows))


def test_chart_values_fill_every_row_cut_long_labels_and_name_the_report():
    spec = spec_of(_blocks())
    series = _series([("Instagram and friends", "48,210"), ("X", "6,115")])
    values = chart_values(spec, CHART_SCHEMA, series, kind="bar", title="A very long report title indeed, far too long", as_of="2026-09-21T09:30:00+00:00")
    assert values == {
        "row_1_label": "Instagram a…", "row_1_value": "48,210",
        "row_2_label": "X", "row_2_value": "6,115",
        # A row past the report's is set to empty: binding again to a shorter table leaves no stale row.
        "row_3_label": "", "row_3_value": "",
        "chip": "A very long repo… · 2026-09-21",
        "sub": "Reach", "kind": "bar",
    }
    assert len(values["chip"]) == 30
    with pytest.raises(ChartBindingError, match="longer than the 8 characters"):
        chart_values(spec, CHART_SCHEMA, _series([("A", "$1,204,560.75"), ("B", "1")]), kind="grid", title="T", as_of=None)


def test_the_chip_never_loses_its_date():
    assert chip_text("Weekly sales", "2026-09-21T09:30:00+00:00") == "Weekly sales · 2026-09-21"
    assert chip_text("Weekly sales", None) == "Weekly sales"
    assert chip_text("", "2026-09-21T09:30:00+00:00") == "2026-09-21"
    assert chip_text("x" * 200, "2026-09-21", limit=20).endswith("… · 2026-09-21")


@pytest.mark.parametrize(
    "kind, rows, part, problem",
    [
        pytest.param("pie", REACH, "table", "chart must be one of bar, line, grid", id="no-kind"),
        pytest.param("bar", REACH, "metrics", "different measures", id="metrics-as-bars"),
        pytest.param("line", [("A", "1")], "table", "one figure is a number", id="one-row"),
        pytest.param("bar", [("A", "$5"), ("B", "5%")], "table", "different units ($, %)", id="mixed-units"),
        pytest.param("bar", [("A", "5"), ("B", "-3")], "table", "a bar grows from zero", id="a-negative-bar"),
    ],
)
def test_a_kind_its_figures_cannot_make_is_refused(kind, rows, part, problem):
    assert problem in kind_problem(kind, _series(rows, part=part))


def test_a_line_takes_negatives_and_the_grid_takes_anything():
    assert kind_problem("line", _series([("A", "5"), ("B", "-3")])) is None
    assert kind_problem("grid", _series([("A", "$5"), ("B", "5%")], part="metrics")) is None
    assert default_kind(_series(REACH), "bar") == "bar"
    assert default_kind(_series([("A", "$5"), ("B", "5%")]), "bar") == "grid"
    assert default_kind(_series(REACH, part="metrics"), "line") == "grid"


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------


def test_the_infographic_is_the_eighth_image_starter_charting_five_claimed_rows():
    starter = _infographic()
    blocks = starter["blocks"]
    assert SOCIAL_IMAGE_STARTER_SLUGS[-1] == "infographic" and len(SOCIAL_IMAGE_STARTER_SLUGS) == 8
    assert (starter["name"], starter["format"], starter["category"]) == ("Infographic", "social_image", "social")
    assert blocks["sizes"] == ["1080x1350", "1080x1920", "1200x628", "1600x900"]
    assert blocks["data"] == {
        "rows": 5, "label": "row_{n}_label", "value": "row_{n}_value", "source": "source_label", "header": "subtitle", "chart": "chart",
    }
    assert claim_names(blocks["variables_schema"]) == [f"row_{n}_value" for n in range(1, 6)]
    schema = blocks["variables_schema"]
    # Every chart shows its source: the chip has no default; the first row neither.
    assert all("default" not in schema[name] for name in ("source_label", "row_1_label", "row_1_value", "headline"))
    assert schema["chart"]["default"] == "bar" and schema["source_caption"]["default"] == "Source"


def _chart_markup(html):
    """Inside the chart: after its opening tag (whose data-chart names the kind), up to the source chip."""
    start = html.index(">", html.index('<div class="chart fit" id="chart"')) + 1
    return html[start : html.index('<div class="chip fit"', start)]


def _script(html):
    return html[html.rindex("<script>") :]


def test_the_chart_shows_only_its_rows_and_the_script_writes_no_word_of_its_own():
    html = _infographic()["blocks"]["html"]
    rows = {f"row_{n}_{part}" for n in range(1, 6) for part in ("label", "value")}
    # Inside the chart: nothing but the rows' own variables, each once per kind (bar, line, grid).
    shown = re.findall(r"\{\{\s*([\w.]+)\s*\}\}", _chart_markup(html))
    assert set(shown) == rows and len(shown) == 3 * len(rows)
    script = _script(html)
    # The only text the script writes is the headline's own words, set on lines (the family's dress).
    writes = re.findall(r"(\w+)\.textContent\s*\+?=\s*([^;]+);", script)
    assert writes == [("el", '""'), ("row", "line"), ("hot", "accent")]
    dress = script[script.index("function dress(") : script.index('document.querySelectorAll("[data-dress]")')]
    assert all(f"{target}.textContent" in dress for target, _ in writes)
    for writer in ("innerHTML", "outerHTML", "innerText", "insertAdjacent", "createTextNode", "document.write", "textContent +="):
        assert writer not in script, writer
    # The figures are read by the platform's own rule, verbatim.
    assert f"const FIGURE = /{FIGURE_PATTERN}/;" in script


def test_the_sample_chart_is_a_reports_binding():
    driver = _driver()
    starter = _infographic()
    driver.check_sample_binding(starter)
    report, column, kind = driver.SAMPLE_BINDING
    series = table_series(first_table(report["markdown"]), column=column)
    sample = starter["sample_data"]
    assert [(sample[f"row_{n}_label"], sample[f"row_{n}_value"]) for n in range(1, 6)] == [(l, f.text) for l, f in series.rows]
    assert sample["source_label"] == chip_text(report["title"], report["as_of"], 90) and sample["chart"] == kind == "bar"
    with pytest.raises(driver.PreviewFailure, match="not the channel report's binding"):
        driver.check_sample_binding({**starter, "sample_data": {**sample, "row_3_value": "19,350"}})


# ---------------------------------------------------------------------------
# The harness: the socials API over SQLite, reports and their files
# ---------------------------------------------------------------------------


def _portable(col_type):
    if isinstance(col_type, (JSONB, ARRAY)):
        return sa.JSON()
    if isinstance(col_type, sa.Uuid) or type(col_type).__name__.upper() == "UUID":
        return sa.Uuid()
    return col_type


def _sqlite_copy(table, metadata):
    columns = [sa.Column(c.name, _portable(c.type), primary_key=c.primary_key) for c in table.columns]
    return sa.Table(table.name, metadata, *columns)


_TABLES = sa.MetaData()
_sqlite_copy(Workspace.__table__, _TABLES)
_sqlite_copy(DocumentTemplate.__table__, _TABLES)
# agent_reports has no ORM model (alembic prd76_agent_reports + prd133b's deleted_at): the columns read here.
AGENT_REPORTS = sa.Table(
    "agent_reports",
    _TABLES,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("workspace_id", sa.String(36), nullable=False),
    sa.Column("report_type", sa.String(30)),
    sa.Column("title", sa.String(255), nullable=False),
    sa.Column("summary", sa.String(500)),
    sa.Column("file_path", sa.String(1024), nullable=False),
    sa.Column("metrics", sa.JSON),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("deleted_at", sa.DateTime(timezone=True)),
)


class FakeWorkspaceClient:
    """The workspace worker's file read (core.workspace_client.WorkspaceClient), from the test's files."""

    files: dict = {}
    reads: list = []
    down = False

    def __init__(self, workspace_id):
        self.workspace_id = workspace_id

    async def read_file(self, path):
        type(self).reads.append((self.workspace_id, path))
        if type(self).down:
            return {"success": False, "error": "worker at 10.0.0.7:8081 refused the connection", "status_code": 503}
        content = type(self).files.get((self.workspace_id, path))
        if content is None:
            return {"success": False, "error": "File not found", "status_code": 404}
        return {"success": True, "content": content}


def _ctx(workspace_id, user_id="member-1"):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id=user_id, clerk_user_id=f"clerk-{user_id}", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def env(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    _TABLES.create_all(engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    # llm_usage (the render quota reads it) as raw DDL: CI's SQLAlchemy cannot compile its UUID type for SQLite.
    columns = ", ".join(f"{c.name} INTEGER PRIMARY KEY" if c.primary_key else c.name for c in LLMUsage.__table__.columns)
    with engine.begin() as conn:
        conn.exec_driver_sql(f"CREATE TABLE {LLMUsage.__tablename__} ({columns})")
    session = sessionmaker(bind=engine)()
    for ws_id in (WS, WS_OTHER):
        session.add(
            Workspace(
                id=ws_id, name=f"ws-{ws_id.hex[-2:]}", plan="basic", plan_limits={},
                settings={"socials": {"enabled": True}}, onboarding={}, created_at=CREATED, updated_at=CREATED,
            )
        )
    session.commit()

    state = SimpleNamespace(session=session, ctx=_ctx(WS), role="owner", launched=[])
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)
    monkeypatch.setattr(socials_api, "_launch_render", lambda job: state.launched.append(job))

    async def renderer_up(client=None, store=None):
        return None

    monkeypatch.setattr(render, "ensure_renderer", renderer_up)
    FakeWorkspaceClient.files, FakeWorkspaceClient.reads, FakeWorkspaceClient.down = {}, [], False
    monkeypatch.setattr(report_charts, "WorkspaceClient", FakeWorkspaceClient)

    app = FastAPI()
    app.include_router(socials_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _report(env, ws=WS, *, title=TITLE, markdown=CHANNELS, metrics=None, deleted=False) -> str:
    ident = str(uuid.uuid4())
    path = f"reports/scout/2026-09-21_{ident[:6]}_september.md"
    env.session.execute(
        AGENT_REPORTS.insert().values(
            id=ident, workspace_id=str(ws), report_type="summary", title=title, summary=None, file_path=path,
            metrics=metrics or {}, created_at=MADE, deleted_at=MADE if deleted else None,
        )
    )
    env.session.commit()
    if markdown is not None:
        FakeWorkspaceClient.files[(str(ws), path)] = markdown
    return ident


def _template(env, blocks, *, ws=WS, fmt="social_image") -> uuid.UUID:
    template_id = uuid.uuid4()
    env.session.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks) "
            "VALUES (:id, :ws, :name, :fmt, '{}', :blocks)"
        ),
        {"id": template_id.hex, "ws": ws.hex, "name": f"tpl-{template_id.hex[:6]}", "fmt": fmt, "blocks": json.dumps(blocks)},
    )
    env.session.commit()
    return template_id


def _chart(env, report_id, template_id, **params):
    return env.client.get(
        CHART_ROUTE.format(report_id=report_id), params={"template_id": str(template_id), **params}
    )


def _bound_post(env, report_id, template_id, **params):
    """A post whose chart is the report's binding, saved as the composer saves it."""
    resp = _chart(env, report_id, template_id, **params)
    assert resp.status_code == 200, resp.text
    binding = resp.json()
    sample = _infographic()["sample_data"]
    copy_variables = {name: {"value": sample[name], "claim": False} for name in ("headline", "eyebrow", "cta")}
    created = env.client.post(
        "/api/socials/posts",
        json={
            "title": "September reach", "copy": {"base": "Where our reach came from."}, "format": "infographic",
            "template_id": str(template_id), "variables": {**copy_variables, **binding["variables"]},
            "sources": binding["sources"],
        },
    )
    assert created.status_code == 201, created.text
    return binding, created.json()


def _render(env, post):
    return env.client.post(f"/api/socials/posts/{post['id']}/render")


def _patch_variables(env, post, **values):
    variables = {name: dict(spec) for name, spec in post["variables"].items()}
    for name, value in values.items():
        variables[name] = {"value": value, "claim": variables.get(name, {}).get("claim", False)}
    resp = env.client.patch(f"/api/socials/posts/{post['id']}", json={"variables": variables})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _status(env, post):
    env.session.expire_all()
    return env.session.get(SocialPost, uuid.UUID(post["id"])).status


# ---------------------------------------------------------------------------
# AC 1: a report with a table → its top 5 rows, the chip resolved
# ---------------------------------------------------------------------------


def test_given_a_report_with_a_table_the_chart_shows_its_top_5_rows_with_the_chip_resolved(env):
    report_id = _report(env)
    template_id = _template(env, _infographic()["blocks"])

    resp = _chart(env, report_id, template_id, column="Reach")

    assert resp.status_code == 200, resp.text
    binding = resp.json()
    assert (binding["part"], binding["column"], binding["chart"], binding["shown"], binding["total"]) == ("table", "Reach", "bar", 5, 6)
    assert [(row["label"], row["value"]) for row in binding["rows"]] == REACH
    variables = binding["variables"]
    for n, (label, value) in enumerate(REACH, start=1):
        assert variables[f"row_{n}_label"] == {"value": label, "claim": False}
        assert variables[f"row_{n}_value"] == {"value": value, "claim": True}
    # The chip is the report as it resolves in the workspace (sources.resolve): its title and the day it was made.
    assert binding["source"]["kind"] == "report" and binding["source"]["title"] == TITLE
    assert variables["source_label"] == {"value": f"{TITLE} · 2026-09-21", "claim": False}
    assert variables["subtitle"] == {"value": "Reach", "claim": False} and variables["chart"] == {"value": "bar", "claim": False}
    # Every figure shown is a claim bound to the report (D7).
    source = {"kind": "report", "ref": report_id, "as_of": "2026-09-21T09:30:00+00:00"}
    assert binding["sources"] == {f"row_{n}_value": source for n in range(1, 6)}

    _, post = _bound_post(env, report_id, template_id, column="Reach")
    assert post["sources"] == binding["sources"]
    rendered = _render(env, post)
    assert rendered.status_code == 202, rendered.text
    (job,) = env.launched
    assert job.bundle["still"] == {"at": [0.0]}
    assert [(job.bundle["variables"][f"row_{n}_label"], job.bundle["variables"][f"row_{n}_value"]) for n in range(1, 6)] == REACH
    assert job.bundle["variables"]["source_label"] == f"{TITLE} · 2026-09-21"


def test_a_shorter_table_leaves_no_row_and_no_claim_past_its_own(env):
    report_id = _report(env, markdown="| Month | Followers |\n|---|---|\n| July | 11,905 |\n| August | 13,260 |\n")
    template_id = _template(env, _infographic()["blocks"])

    binding = _chart(env, report_id, template_id, chart="line").json()

    assert [(row["label"], row["value"]) for row in binding["rows"]] == [("July", "11,905"), ("August", "13,260")]
    assert binding["variables"]["row_3_value"] == {"value": "", "claim": False}
    assert binding["variables"]["row_5_label"] == {"value": "", "claim": False}
    assert set(binding["sources"]) == {"row_1_value", "row_2_value"}
    assert (binding["chart"], binding["shown"], binding["total"]) == ("line", 2, 2)


def test_the_template_picks_its_own_kind_unless_the_figures_cannot_make_it(env):
    template_id = _template(env, _infographic()["blocks"])
    mixed = _report(env, markdown="| Channel | Result |\n|---|---|\n| A | $5 |\n| B | 5% |\n")
    assert _chart(env, mixed, template_id).json()["chart"] == "grid"
    refused = _chart(env, mixed, template_id, chart="bar")
    assert refused.status_code == 422 and "different units ($, %)" in refused.json()["detail"]


# ---------------------------------------------------------------------------
# AC 3: every number on the rendered chart comes from the bound source
# ---------------------------------------------------------------------------


def test_every_number_on_the_rendered_chart_comes_from_the_bound_source(env):
    report_id = _report(env)
    template_id = _template(env, _infographic()["blocks"])
    _, post = _bound_post(env, report_id, template_id, column="Reach")
    assert _render(env, post).status_code == 202
    (job,) = env.launched
    table = first_table(CHANNELS)
    reach = table.headers.index("Reach")

    shown = [(job.bundle["variables"][f"row_{n}_label"], job.bundle["variables"][f"row_{n}_value"]) for n in range(1, 6)]

    for (label, value), cells in zip(shown, table.rows):
        assert label == cells[0]
        assert value == cells[reach]
        assert parse_figure(value).value == parse_figure(cells[reach]).value
    # The chart area shows nothing but those rows, and the script writes no number of its own
    # (test_the_chart_shows_only_its_rows_and_the_script_writes_no_word_of_its_own): so these are every number on it.
    rows_on_chart = set(re.findall(r"\{\{\s*([\w.]+)\s*\}\}", _chart_markup(job.bundle["composition"]["html"])))
    assert rows_on_chart == {f"row_{n}_{part}" for n in range(1, 6) for part in ("label", "value")}


@pytest.mark.parametrize(
    "change, detail",
    [
        pytest.param({"row_3_value": "19,350"}, "row 3 shows 'TikTok 19,350'; the report has 'TikTok 19,305'", id="an-edited-figure"),
        pytest.param({"row_2_label": "Facebook"}, "row 2 shows 'Facebook 21,480'; the report has 'LinkedIn 21,480'", id="an-edited-label"),
        pytest.param({"row_5_label": "", "row_5_value": ""}, "row 5 shows 'nothing'; the report has 'X 6,115'", id="a-dropped-row"),
        pytest.param({"source_label": "Our own numbers · 2026-09-21"}, "the chart's source chip must name its report", id="an-edited-chip"),
    ],
)
def test_a_bound_chart_edited_away_from_its_report_is_refused_before_anything_renders(env, change, detail):
    report_id = _report(env)
    template_id = _template(env, _infographic()["blocks"])
    _, post = _bound_post(env, report_id, template_id, column="Reach")
    post = _patch_variables(env, post, **change)

    refused = _render(env, post)

    assert refused.status_code == 422, refused.text
    assert detail in refused.json()["detail"]
    assert env.launched == [] and _status(env, post) == "draft"


def test_a_report_that_changed_or_went_refuses_the_render_until_the_chart_is_bound_again(env):
    report_id = _report(env)
    template_id = _template(env, _infographic()["blocks"])
    _, post = _bound_post(env, report_id, template_id, column="Reach")
    (path,) = [p for (_, p) in FakeWorkspaceClient.files]

    FakeWorkspaceClient.files[(str(WS), path)] = CHANNELS.replace("19,305", "19,999")
    changed = _render(env, post)
    assert changed.status_code == 422 and "row 3 shows 'TikTok 19,305'; the report has 'TikTok 19,999'" in changed.json()["detail"]

    # Bound again, it renders.
    binding = _chart(env, report_id, template_id, column="Reach").json()
    post = _patch_variables(env, post, **{name: spec["value"] for name, spec in binding["variables"].items()})
    assert _render(env, post).status_code == 202

    gone = _report(env)
    _, other = _bound_post(env, gone, template_id, column="Reach")
    env.session.execute(AGENT_REPORTS.update().where(AGENT_REPORTS.c.id == gone).values(deleted_at=MADE))
    env.session.commit()
    refused = _render(env, other)
    assert refused.status_code == 422 and "bound to a report that cannot be found (the report was deleted)" in refused.json()["detail"]


def test_a_report_file_that_cannot_be_read_refuses_the_render_with_503_and_keeps_the_workers_words(env):
    report_id = _report(env)
    template_id = _template(env, _infographic()["blocks"])
    _, post = _bound_post(env, report_id, template_id, column="Reach")
    FakeWorkspaceClient.down = True

    refused = _render(env, post)

    assert refused.status_code == 503, refused.text
    assert "could not be read right now" in refused.json()["detail"] and "10.0.0.7" not in refused.json()["detail"]
    assert env.launched == [] and _status(env, post) == "draft"


def test_a_chart_bound_to_a_reports_metrics_is_checked_against_them_without_reading_its_file(env):
    metrics = {"followers": 16340, "engagement_rate": "4.2%", "model": "gpt-4o", "ok": True, "posts": 50}
    report_id = _report(env, markdown="# No table here\n\nJust prose.\n", metrics=metrics)
    template_id = _template(env, _infographic()["blocks"])

    binding, post = _bound_post(env, report_id, template_id)
    assert (binding["part"], binding["chart"]) == ("metrics", "grid")
    assert [(row["label"], row["value"]) for row in binding["rows"]] == [("followers", "16,340"), ("engagement rate", "4.2%"), ("posts", "50")]
    FakeWorkspaceClient.reads = []
    assert _render(env, post).status_code == 202
    assert FakeWorkspaceClient.reads == []
    bars = _chart(env, report_id, template_id, chart="bar")
    assert bars.status_code == 422 and "a report's metrics are different measures" in bars.json()["detail"]


def test_a_chart_bound_to_anything_else_renders_as_any_claim_does(env):
    template_id = _template(env, _infographic()["blocks"])
    url = {"kind": "url", "ref": "https://example.com/reach", "as_of": "2026-09-21T09:30:00+00:00"}
    rows = {f"row_{n}_{part}": {"value": value, "claim": part == "value"} for n, row in enumerate(REACH, start=1) for part, value in zip(("label", "value"), row)}
    created = env.client.post(
        "/api/socials/posts",
        json={
            "title": "Reach", "template_id": str(template_id),
            "variables": {**rows, "headline": {"value": "Reach", "claim": False}, "source_label": {"value": "example.com", "claim": False}},
            "sources": {f"row_{n}_value": url for n in range(1, 6)},
        },
    )
    assert created.status_code == 201, created.text

    assert _render(env, created.json()).status_code == 202
    assert FakeWorkspaceClient.reads == []


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_the_route_reads_only_the_callers_workspace_and_says_why_it_refuses(env):
    blocks = _infographic()["blocks"]
    template_id = _template(env, blocks)
    theirs = _report(env, WS_OTHER)
    deleted = _report(env, deleted=True)
    negative = _report(env, markdown="| Month | Change |\n|---|---|\n| July | +4% |\n| August | -2% |\n")
    no_table = _report(env, markdown="Prose only.\n")

    assert _chart(env, theirs, template_id).status_code == 404
    assert _chart(env, deleted, template_id).json() == {"detail": "Report not found: the report was deleted"}
    assert _chart(env, "not-a-uuid", template_id).status_code == 404
    other_template = _template(env, blocks, ws=WS_OTHER)
    assert _chart(env, _report(env), other_template).status_code == 422
    stats = next(s for s in social_starters("social_image") if s["slug"] == "stats-card")
    no_chart = _chart(env, _report(env), _template(env, stats["blocks"]))
    assert no_chart.status_code == 422 and "has no chart to fill" in no_chart.json()["detail"]

    bars = _chart(env, negative, template_id, chart="bar")
    assert bars.status_code == 422 and "a bar grows from zero" in bars.json()["detail"]
    assert _chart(env, negative, template_id, chart="line").status_code == 200
    unknown = _chart(env, _report(env), template_id, column="Clicks")
    assert unknown.status_code == 422 and "its columns: 'Posts', 'Reach', 'Engagement rate'" in unknown.json()["detail"]
    assert "part must be one of table, metrics" in _chart(env, _report(env), template_id, part="rows").json()["detail"]
    assert "has no data table" in _chart(env, no_table, template_id, part="table").json()["detail"]
    assert "no data table, and its metrics carry no figures" in _chart(env, no_table, template_id).json()["detail"]
    FakeWorkspaceClient.down = True
    assert _chart(env, _report(env), template_id).status_code == 503


def test_a_bound_chart_is_the_binding_function_agents_call_too(env):
    report_id = _report(env)
    blocks = validate_social_blocks(_infographic()["blocks"], "social_image")
    binding = asyncio.run(report_charts.bind_report(env.session, WS, report_id, blocks, column="Reach", chart="grid"))
    assert binding.chart == "grid" and binding.rows == REACH
    assert binding.to_dict()["variables"]["chart"] == {"value": "grid", "claim": False}
    with pytest.raises(report_charts.ReportNotFound):
        asyncio.run(report_charts.bind_report(env.session, WS_OTHER, report_id, blocks))


def test_the_chart_route_is_in_the_committed_route_manifest():
    routes = {(r["method"], r["path"]) for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["routes"]}
    assert ("GET", "/api/socials/sources/reports/{report_id}/chart") in routes


# ---------------------------------------------------------------------------
# The CI driver (AC 1 and AC 2 in the media-render job)
# ---------------------------------------------------------------------------


def _driver():
    spec = importlib.util.spec_from_file_location("social_template_previews", _ROOT / "scripts" / "ci" / "social_template_previews.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ci_driver_binds_fixture_reports_and_hands_media_render_only_their_own_figures():
    driver = _driver()
    starter = _infographic()
    kit = {**driver.KIT, "logo_url": driver.logo_png(driver.KIT["primary_color"])}
    settings = load_settings({})
    kinds = set()
    for name, report, column, kind in driver.INFOGRAPHIC_RENDERS:
        values, series = driver.bind_fixture(starter, report, column, kind)
        for size in starter["blocks"]["sizes"]:
            bundle = driver.chart_bundle_for(starter, kit, size, values)
            shown = driver.check_chart_from_report(bundle, starter, report, series)
            parsed = parse_bundle(bundle, settings, {})
            assert (parsed.composition.width, parsed.composition.height) == tuple(int(x) for x in size.split("x"))
            assert parsed.still.at == (0.0,) and bundle["variables"]["chart"] == kind
            assert len(shown) == 5
        kinds.add(kind)
    assert kinds == {"bar", "line", "grid"}
    # The stress: labels longer than the template holds are cut, and the long title's chip keeps its date.
    values, _ = driver.bind_fixture(starter, driver.PROGRAMME_REPORT, None, "bar")
    assert values["row_1_label"].endswith("…") and len(values["row_1_label"]) == 40
    assert values["source_label"].endswith("fiscal… · 2026-09-24") and len(values["source_label"]) <= 90
    # A figure the report does not have fails the driver.
    values, series = driver.bind_fixture(starter, driver.TREND_REPORT, None, "line")
    tampered = driver.chart_bundle_for(starter, kit, "1080x1350", {**values, "row_4_value": "13,620"})
    with pytest.raises(driver.PreviewFailure, match="row 4's figure '13,620' is not the report's '13,260'"):
        driver.check_chart_from_report(tampered, starter, driver.TREND_REPORT, series)


def test_the_sample_data_fills_the_infographic():
    starter = _infographic()
    resolved = resolve_variables(starter["blocks"]["variables_schema"], starter["sample_data"])
    assert resolved.missing == [] and resolved.invalid == []
    assert isinstance(parse_figure(starter["sample_data"]["row_1_value"]), Figure)
