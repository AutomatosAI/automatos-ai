"""F353 (issue #947): a Deliverable's picture is drawn after it is made, and a failure keeps the icon.

* A drawn page is stored beside the generated documents and recorded on the row
  (``extra.thumbnail.file``), and the card is given its URL.
* A failed render records the reason (no picture, no URL: the card keeps its
  icon); a missing file records nothing; a heartbeat report is never drawn.
* Registering a document queues it and returns at once, exactly as before; a
  full queue, or the switch off, never touches the registration.
* The background thread logs a job that blows up and carries on.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
from types import SimpleNamespace

import pytest

from config import config
from modules.documents.thumbnails import job, schedule, store
from modules.documents.thumbnails.eligibility import thumbnail_url_for
from modules.documents.thumbnails.render import ThumbnailError

WS = "00000000-0000-0000-0000-0000000000c1"
DOC = "5f0c2a8e-2b7d-4c55-9a51-0d6f1e2b2980"
PNG = b"\x89PNG\r\n\x1a\nfirst-page"


def _row(**over):
    base = dict(id=DOC, workspace_id=WS, artifact_type="document", source_type="chat",
                storage_type="generated", file_path="generated/20261005_Invoice.pdf", file_size_bytes=2048)
    return SimpleNamespace(**{**base, **over})


class FakeDb:
    """Answers the job's one SELECT with ``row`` and keeps every UPDATE it is sent."""

    def __init__(self, row):
        self.row, self.updates = row, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params):
        if str(stmt).strip().startswith("UPDATE"):
            self.updates.append(params)
            return SimpleNamespace()
        return SimpleNamespace(fetchone=lambda: self.row)

    def commit(self):
        pass


@pytest.fixture
def disk(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(store, "is_storage_configured", lambda: False)
    monkeypatch.setattr(job, "read_source", lambda ws, storage, path: b"%PDF-1.7 ...")
    return tmp_path


def _recorded(db):
    assert len(db.updates) == 1
    return json.loads(db.updates[0]["patch"])["thumbnail"]


def test_a_drawn_page_is_stored_recorded_and_offered_to_the_card(disk):
    db = FakeDb(_row())
    assert job.render_for_output(lambda: db, WS, DOC, render=lambda data, ext: PNG) == "rendered"
    assert (disk / WS / "generated" / f"thumb_{DOC}.png").read_bytes() == PNG
    recorded = _recorded(db)
    assert recorded["file"] == f"thumb_{DOC}.png"
    assert thumbnail_url_for(_row(extra={"thumbnail": recorded})) == f"/api/deliverables/{DOC}/thumbnail"


def test_a_failed_render_records_why_and_the_card_keeps_its_icon(disk, caplog):
    def broken(data, ext):
        raise ThumbnailError("not a readable PDF: Data format error")

    db = FakeDb(_row())
    with caplog.at_level(logging.WARNING):
        assert job.render_for_output(lambda: db, WS, DOC, render=broken) == "failed"
    recorded = _recorded(db)
    assert "not a readable PDF" in recorded["failed"]
    assert "not a readable PDF" in caplog.text
    assert not (disk / WS / "generated" / f"thumb_{DOC}.png").exists()
    assert thumbnail_url_for(_row(extra={"thumbnail": recorded})) is None


def test_a_missing_file_records_nothing(disk, monkeypatch):
    monkeypatch.setattr(job, "read_source", lambda ws, storage, path: None)
    db = FakeDb(_row())
    assert job.render_for_output(lambda: db, WS, DOC, render=lambda d, e: PNG) == "no-file"
    assert db.updates == []


def test_heartbeat_reports_images_and_unknown_formats_are_never_drawn(disk):
    drawn = []
    for row in (
        _row(artifact_type="report", source_type="heartbeat", file_path="reports/a/x.md"),
        _row(artifact_type="image", file_path="generated/post.png"),
        _row(file_path="generated/deck.pptx"),
    ):
        outcome = job.render_for_output(lambda r=row: FakeDb(r), WS, DOC, render=lambda d, e: drawn.append(e))
        assert outcome == "skipped"
    assert drawn == []


def test_a_report_is_drawn_but_has_no_row_to_record_on(disk):
    db = FakeDb(_row(artifact_type="report", storage_type="workspace", file_path="reports/a/weekly.md"))
    assert job.render_for_output(lambda: db, WS, DOC, render=lambda d, e: PNG) == "rendered"
    assert db.updates == []
    assert thumbnail_url_for(_row(artifact_type="report", file_path="reports/a/weekly.md", extra={})) \
        == f"/api/deliverables/{DOC}/thumbnail"
    assert thumbnail_url_for(_row(artifact_type="report", source_type="heartbeat",
                                  file_path="reports/a/weekly.md", extra={})) is None


@pytest.fixture
def queue_on(monkeypatch):
    monkeypatch.setattr(config, "DOCUMENT_THUMBNAILS_ENABLED", True, raising=False)
    monkeypatch.setattr(schedule, "_ensure_worker", lambda: None)
    fresh = queue.Queue(maxsize=1)
    monkeypatch.setattr(schedule, "_queue", fresh)
    return fresh


def _register(result):
    @schedule.thumbnail_after_register
    def register(self, *, file_path, **kwargs):
        return result

    return lambda **kw: register(SimpleNamespace(workspace_id=WS), **kw)


def test_registering_a_pdf_queues_it_and_returns_what_register_returned(queue_on):
    result = {"success": True, "deliverable_id": DOC, "artifact_type": "document"}
    assert _register(result)(file_path="generated/a.pdf") is result
    assert queue_on.get_nowait() == (WS, DOC)


def test_an_image_or_a_failed_register_queues_nothing(queue_on):
    _register({"success": True, "deliverable_id": DOC, "artifact_type": "image"})(file_path="a.png")
    _register({"success": False, "error": "x"})(file_path="generated/a.pdf")
    assert queue_on.empty()


def test_a_full_queue_or_the_switch_off_never_touches_the_registration(queue_on, monkeypatch):
    result = {"success": True, "deliverable_id": DOC, "artifact_type": "spreadsheet"}
    queue_on.put_nowait(("ws", "other"))
    assert _register(result)(file_path="sheets/stock.xlsx") is result
    monkeypatch.setattr(config, "DOCUMENT_THUMBNAILS_ENABLED", False, raising=False)
    queue_on.get_nowait()
    assert _register(result)(file_path="sheets/stock.xlsx") is result
    assert queue_on.empty()


def test_a_new_report_is_queued_after_it_is_made(queue_on):
    @schedule.thumbnail_after_report
    async def create_report(self, **kwargs):
        return {"success": True, "report_id": DOC}

    out = asyncio.run(create_report(SimpleNamespace(workspace_id=WS), title="Weekly"))
    assert out == {"success": True, "report_id": DOC}
    assert queue_on.get_nowait() == (WS, DOC)


def test_the_thread_logs_a_job_that_blows_up_and_carries_on(queue_on, monkeypatch, caplog):
    def explode(*args, **kwargs):
        raise RuntimeError("database went away")

    monkeypatch.setattr(job, "render_for_output", explode)
    queue_on.put_nowait((WS, DOC))
    with caplog.at_level(logging.ERROR):
        schedule._draw_next()
    assert "thumbnail job failed" in caplog.text
    assert queue_on.unfinished_tasks == 0
