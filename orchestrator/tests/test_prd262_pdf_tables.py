"""PRD-262 US-023: PDF tables come from pdfplumber, cell for cell what the old extractor found.

The image ran a stream-mode table extractor that pulled in opencv only to import;
its lattice mode failed on every PDF in the image, so stream output was all
production stored. TableProcessor now uses pdfplumber's text strategy. This suite
holds it to that output without the old extractor installed:

- for every fixture tests/fixtures/pdf_tables.py draws, the tables TableProcessor
  returns, page by page, are the committed reference grids
  (tests/fixtures/pdf_table_grids.json), exactly: no cell-level tolerance has been
  set, so none is allowed. test.yml's reference-goldens job re-records the grids with
  the old extractor, pinned and hashed, whenever they or the fixtures change;
- the grids name every fixture, and the reference script draws the same fixtures;
- pdfplumber's rows are cleaned (None to "", a cell's lines joined, all-empty rows
  and columns dropped), the header rule is the old DataFrame rule on lists, and an
  extraction carries its page and position;
- a file pdfplumber cannot read (the ingest also passes DOCX files) yields no tables;
- the fixture PDFs are the same bytes every time, every drawn cell reads back, and
  their Helvetica widths are the ones PDF readers carry.

processors.py is loaded from its file, so sibling tests' stubs of modules.rag.* in
sys.modules cannot get in the way.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from pathlib import Path

import pdfplumber
import pytest
import yaml

from tests.fixtures.pdf_tables import FIXTURES, HELVETICA_WIDTHS, build_pdf, write_fixtures
from tests.fixtures.reference_locks import pins as _pins, unhashed as _unhashed

ROOT = Path(__file__).resolve().parents[2]
PROCESSORS = ROOT / "orchestrator" / "modules" / "rag" / "ingestion" / "multimodal" / "processors.py"
GRIDS = json.loads((ROOT / "orchestrator" / "tests" / "fixtures" / "pdf_table_grids.json")
                   .read_text(encoding="utf-8"))["fixtures"]
REFERENCE = ROOT / "scripts" / "ci" / "reference"
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


processors = _load("prd262_pdf_table_processors", PROCESSORS)


@pytest.fixture(scope="module")
def fixture_pdfs(tmp_path_factory):
    return write_fixtures(tmp_path_factory.mktemp("pdf_tables"))


def _grid(extraction) -> list:
    """The cell grid an extraction came from: its header row, when it found one, then its rows."""
    header = [list(extraction.headers)] if extraction.has_header else []
    return header + [list(row) for row in extraction.rows]


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_pdfplumber_finds_the_reference_grids_exactly(fixture_pdfs, name):
    tables = processors.TableProcessor().extract_tables_from_pdf(str(fixture_pdfs[name]))
    found = [{"page": table.page_number, "rows": _grid(table)} for table in tables]
    assert found == GRIDS[name], (
        f"{name}: pdfplumber's text strategy no longer reproduces the reference grid; "
        "no cell-level tolerance is set (PRD-262 US-023), so a difference is Gerard's call"
    )


def test_the_reference_grids_cover_every_fixture_and_nothing_else():
    assert set(GRIDS) == set(FIXTURES)
    assert all(GRIDS[name] for name in FIXTURES), "every fixture has at least one table"


def test_every_extraction_carries_its_page_and_position(fixture_pdfs):
    tables = processors.TableProcessor().extract_tables_from_pdf(str(fixture_pdfs["parts-two-pages"]))
    assert [table.page_number for table in tables] == [1, 2]
    for table in tables:
        assert table.confidence == processors.TEXT_TABLE_CONFIDENCE
        assert set(table.position) == {"x", "y", "width", "height"}
        assert table.position["width"] > 0 and table.position["height"] > 0
        assert table.markdown.startswith("| SKU | Description | Qty |\n| --- | --- | --- |\n")


# --------------------------------------------------------------------------- rows and headers


def test_rows_are_cleaned_of_empty_rows_and_columns():
    rows = [[None, "", None], ["Region", " North\nAmerica ", ""], ["", "", ""], ["East", None, ""]]
    assert processors.clean_table_rows(rows) == [["Region", "North America"], ["East", ""]]
    assert processors.clean_table_rows([["a"], ["b", "c"]]) == [["a", ""], ["b", "c"]]
    assert processors.clean_table_rows([[None, ""], ["", None]]) == []
    assert processors.clean_table_rows([]) == []


def test_the_header_rule_is_the_old_one_on_lists():
    detect = processors.TableProcessor()._detect_header_row
    assert detect([["Region", "Revenue"], ["North", "1250.50"]]) is True
    assert detect([["2024", "12"], ["2025", "15"]]) is False
    assert detect([["Region", "Revenue"]]) is False
    assert detect([]) is False


def test_a_table_with_a_header_becomes_an_extraction():
    rows = [["Region", "Revenue"], ["North", "1250.50"], ["South", "980.00"]]
    table = processors.TableProcessor()._build_extraction(rows, 2, (72.0, 100.0, 300.0, 160.0))
    assert table.has_header and table.headers == ["Region", "Revenue"]
    assert table.rows == [["North", "1250.50"], ["South", "980.00"]]
    assert (table.row_count, table.column_count, table.page_number) == (2, 2, 2)
    assert table.position == {"x": 72.0, "y": 100.0, "width": 228.0, "height": 60.0}
    assert table.data_types == {"Region": "text", "Revenue": "float"}
    assert table.json == [{"Region": "North", "Revenue": "1250.50"}, {"Region": "South", "Revenue": "980.00"}]


def test_a_table_without_a_header_gets_numbered_columns():
    rows = [["2024", "12"], ["2025", "15"]]
    table = processors.TableProcessor()._build_extraction(rows, 1, (0.0, 0.0, 10.0, 10.0))
    assert not table.has_header
    assert table.headers == ["Column_1", "Column_2"]
    assert table.rows == rows


def test_a_file_pdfplumber_cannot_read_yields_no_tables(tmp_path, caplog):
    docx = tmp_path / "report.docx"
    docx.write_bytes(b"PK\x03\x04 a zip, not a PDF")
    with caplog.at_level(logging.WARNING):
        assert processors.TableProcessor().extract_tables_from_pdf(str(docx)) == []
    assert any("Table extraction skipped" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- the fixtures


def test_the_widths_are_helveticas_own():
    from pdfminer.fontmetrics import FONT_METRICS

    metrics = FONT_METRICS["Helvetica"][1]
    assert len(HELVETICA_WIDTHS) == 95
    assert {char: metrics[char] for char in HELVETICA_WIDTHS} == HELVETICA_WIDTHS


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_a_fixture_is_the_same_pdf_every_time_and_every_cell_reads_back(fixture_pdfs, name):
    data = build_pdf(FIXTURES[name])
    assert data == build_pdf(FIXTURES[name]) == fixture_pdfs[name].read_bytes()
    assert data.startswith(b"%PDF-1.4\n") and data.endswith(b"%%EOF\n")
    with pdfplumber.open(str(fixture_pdfs[name])) as pdf:
        assert len(pdf.pages) == len(FIXTURES[name])
        for page, drawn in zip(pdf.pages, FIXTURES[name]):
            text = page.extract_text()
            cells = [cell for row in drawn.rows for cell in row if cell] + [drawn.caption] * bool(drawn.caption)
            assert [cell for cell in cells if cell not in text] == []


# --------------------------------------------------------------------------- the reference and CI


def test_the_reference_lock_is_its_input_pinned_and_hashed():
    wanted, locked = _pins(REFERENCE / "requirements-pdf-tables.in"), _pins(REFERENCE / "requirements-pdf-tables.lock")
    assert wanted and locked == wanted, "every package the reference runs is pinned in the .in"
    text = (REFERENCE / "requirements-pdf-tables.lock").read_text(encoding="utf-8")
    assert text.splitlines()[1] == (
        "#    uv pip compile scripts/ci/reference/requirements-pdf-tables.in --universal "
        "--python-version 3.11 --only-binary :all: --generate-hashes "
        "-o scripts/ci/reference/requirements-pdf-tables.lock"
    )
    assert _unhashed(text) == []
    assert _unhashed("a==1 \\\n    --hash=sha256:00\nb==2\n") == ["b==2"], "the check finds a bare pin"


def test_the_reference_script_runs_only_the_pinned_extractor(capsys):
    script = _load("prd262_pdf_table_reference", REFERENCE / "pdf_tables.py")
    wanted = _pins(REFERENCE / "requirements-pdf-tables.in")
    assert script.PINNED and all(wanted[name] == version for name, version in script.PINNED.items())
    assert set(script.load_fixtures().FIXTURES) == set(FIXTURES)
    assert script.version_errors(), "this suite's environment has no reference extractor"
    assert script.main([]) == 2, "it refuses to record with anything but the pinned versions"
    assert "::error::the reference needs the pinned extractor" in capsys.readouterr().out


def test_ci_re_records_the_grids_whenever_they_or_their_inputs_change():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    changes = workflow["jobs"]["changes"]
    match = next(step for step in changes["steps"] if step.get("id") == "match")
    paths = re.compile(match["env"]["REFERENCE_PATHS"])
    for path in ("scripts/ci/reference/pdf_tables.py", "scripts/ci/reference/requirements-pdf-tables.lock",
                 "orchestrator/tests/fixtures/pdf_tables.py", "orchestrator/tests/fixtures/pdf_table_grids.json"):
        assert paths.search(path), path
    assert "references" in changes["outputs"]
    assert 'lane references "$REFERENCE_PATHS"' in match["run"]
    assert re.search(r"printf '%s=true\\n' [a-z_ ]*\breferences\b", match["run"]), "every lane runs off a PR"
    job = workflow["jobs"]["reference-goldens"]
    assert job["needs"] == "changes" and job["if"] == "needs.changes.outputs.references == 'true'"
    step = next(step for step in job["steps"] if "pdf_tables.py" in step.get("run", ""))
    assert step["env"]["LOCK"] == "scripts/ci/reference/requirements-pdf-tables.lock"
    assert step["env"]["GRIDS"] == "orchestrator/tests/fixtures/pdf_table_grids.json"
    run = step["run"]
    assert 'python -m venv "$VENV"' in run
    assert '"$VENV/bin/pip" install --no-deps --require-hashes --only-binary :all: -r "$LOCK"' in run
    assert '"$VENV/bin/python" scripts/ci/reference/pdf_tables.py --out' in run
    assert 'diff -u <(jq -S . "$GRIDS")' in run and "exit 1" in run
