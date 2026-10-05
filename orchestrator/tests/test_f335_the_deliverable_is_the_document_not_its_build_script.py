"""F335 (night 10): a session's Deliverable is the document it made, not its build script.

The card's Deliverable was the program the session wrote to build the document:
#2050 ``mkpdf.py`` ("Mkpdf"), #2068 ``make_invoice.py``, #2079 ``build_letter.py``,
#2082 ``make_xlsx.py``, #2096 ``mk.py``, with the PDF, sheet or page beside it.
#2077 registered two identical PDFs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import services.deliverable_service as deliverable_service  # noqa: E402
from services import cli_host_service  # noqa: E402
from services.session_outputs import SessionOutput, brief_asks_for_code, pick_deliverables  # noqa: E402

WS = "00000000-0000-0000-0000-0000000033f5"
TASK_ID = 2050
INVOICE_BRIEF = "Invoice HL-2026-0142 for Lantern Kitchen, as a PDF on my letterhead."


class _RecordingService:
    """Stands in for DeliverableService: records what register() was asked."""

    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def register(self, **kwargs):
        type(self).calls.append(kwargs)
        return {"success": True, "deliverable_id": f"d-{len(type(self).calls)}", "created": True}


@pytest.fixture
def session(monkeypatch, tmp_path):
    """A session folder in the workspace volume; ``write(name, bytes)`` writes a file there."""
    _RecordingService.calls = []
    monkeypatch.setattr(deliverable_service, "DeliverableService", _RecordingService)
    monkeypatch.setattr(cli_host_service.config, "WORKSPACE_VOLUME_PATH", str(tmp_path), raising=False)
    folder = tmp_path / WS / "sessions" / str(TASK_ID)
    folder.mkdir(parents=True)

    def write(name: str, content: bytes) -> str:
        (folder / name).write_bytes(content)
        return f"/host/workspaces/{WS}/sessions/{TASK_ID}/{name}"

    return SimpleNamespace(write=write)


def _register(files, brief=INVOICE_BRIEF):
    task = SimpleNamespace(workspace_id=UUID(WS), id=TASK_ID, title="Invoice HL-2026-0142", description=brief)
    registered = cli_host_service._register_session_deliverables(
        MagicMock(), task, files, agent_id=341, agent_name="Ops", session_id="s-1")
    return [entry["title"] for entry in registered]


def test_the_invoice_pdf_is_the_deliverable_not_the_script_that_built_it(session):
    files = [session.write("mkpdf.py", b"from fpdf import FPDF\n"),
             session.write("HL-2026-0142-Lantern-Kitchen.pdf", b"%PDF-1.7 invoice")]

    assert _register(files) == ["HL-2026-0142-Lantern-Kitchen.pdf"]
    assert [c["file_path"] for c in _RecordingService.calls] == [
        f"sessions/{TASK_ID}/HL-2026-0142-Lantern-Kitchen.pdf"]


@pytest.mark.parametrize("document", ["price-list.xlsx", "letter.html", "notes.md", "prices.csv", "card.png"])
def test_every_kind_of_document_wins_over_the_script(session, document):
    files = [session.write("make_it.py", b"print(1)\n"), session.write("build.sh", b"echo\n"),
             session.write(document, b"the work")]

    assert _register(files) == [document]


def test_a_script_that_is_all_the_session_made_is_still_its_deliverable(session):
    assert _register([session.write("hello.py", b"print('hello')\n")]) == ["hello.py"]


def test_a_brief_that_asks_for_code_keeps_the_script_beside_the_document(session):
    files = [session.write("make_xlsx.py", b"import openpyxl\n"), session.write("prices.xlsx", b"PK sheet")]

    assert _register(files, brief="Write a Python script that builds the price list.") == [
        "make_xlsx.py", "prices.xlsx"]


def test_two_identical_pdfs_are_one_deliverable_under_the_first_name(session):
    files = [session.write("HL-2026-0144-Kiln-Bakehouse.pdf", b"%PDF-1.7 same bytes"),
             session.write("invoice-final.pdf", b"%PDF-1.7 same bytes")]

    assert _register(files) == ["HL-2026-0144-Kiln-Bakehouse.pdf"]


def test_two_pdfs_of_one_size_but_different_content_are_both_deliverables(session):
    files = [session.write("draft.pdf", b"%PDF-1.7 aaaa"), session.write("final.pdf", b"%PDF-1.7 bbbb")]

    assert _register(files) == ["draft.pdf", "final.pdf"]


def test_one_file_reported_twice_is_registered_once(session):
    path = session.write("letter.pdf", b"%PDF-1.7 letter")

    assert _register([path, path]) == ["letter.pdf"]


@pytest.mark.parametrize("brief, asks", [
    (INVOICE_BRIEF, False),
    ("You run on Claude Code. Put the QR code and the discount code CAFE10 on the flyer.", False),
    ("Write a Python script that totals the orders.", True),
    ("Fix the bug in utils.py.", True),
    ("Refactor the repository's build.", True),
])
def test_what_counts_as_a_brief_asking_for_code(brief, asks):
    assert brief_asks_for_code(brief) is asks


def test_a_projects_folder_script_beside_a_document_is_left_out_too():
    outputs = [SessionOutput("/p/mk.py", "projects/deliverables/sessions/2096/mk.py", "code"),
               SessionOutput("/p/inv.pdf", "projects/deliverables/sessions/2096/Harbour-Lights-Cafe-invoice.pdf",
                             "document")]

    assert [o.rel for o in pick_deliverables(outputs, INVOICE_BRIEF)] == [
        "projects/deliverables/sessions/2096/Harbour-Lights-Cafe-invoice.pdf"]
