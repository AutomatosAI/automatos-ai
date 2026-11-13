"""13 Nov 2026, Automatos HQ: a report's linked tickets' text files ride with the report.

The News & Research agent's digest was a Deliverable of its ticket (#0308) and its report only
pointed at the session folder. The Social Media Director read the report with
get_latest_report, could not open that folder and parked its ticket on a question. These tests
hold ``services.report_linked_files``: the digest's text comes back with the report, framed and
within one budget, in the caller's workspace only, on a session of its own (never the
caller's), and a failed lookup leaves the report readable.

PURE tests: a fake session, the listing and the worker stubbed; no database, no worker.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

WS = "11111111-1111-1111-1111-111111111111"
REPORT_ID = "76425593-b75a-40bc-b260-9410dd5179fd"
DIGEST = "# Morning AI digest — 13 November 2026\n\n- OpenAI Businesses in ChatGPT (link)\n"


class _Db:
    """The session the lookups open for themselves: one SELECT, then closed."""

    def __init__(self, linked):
        self.linked = linked
        self.params = []
        self.closed = False

    def execute(self, _statement, params):
        self.params.append(params)
        return SimpleNamespace(fetchone=lambda: (self.linked,) if self.linked is not None else None)

    def close(self):
        self.closed = True


def _row(rid, name, file_type, ticket="308", storage="workspace"):
    return {"id": rid, "title": name, "file_path": f"sessions/{ticket}/{name}", "file_type": file_type,
            "storage_type": storage, "source_type": "task", "source_id": ticket}


def _install(monkeypatch, linked, rows, texts):
    """The session factory, the listing and the worker; returns what each was asked."""
    import core.database.database as database
    import core.workspace_client as workspace_client
    import services.step_files as sf

    db, listed, reads = _Db(linked), [], []
    monkeypatch.setattr(database, "SessionLocal", lambda: db)

    def _list(session, workspace_id, **filters):
        assert session is db
        listed.append((str(workspace_id), filters))
        return {"success": True, "deliverables": [r for r in rows if r["source_id"] == filters["source_id"]]}

    class _Worker:
        def __init__(self, workspace_id):
            assert workspace_id == WS

        async def read_file(self, path):
            reads.append(path)
            return {"success": True, "content": texts[path]} if path in texts else {"success": False, "error": "404"}

    monkeypatch.setattr(sf, "listed_or_raise", _list)
    monkeypatch.setattr(workspace_client, "WorkspaceClient", _Worker)
    return SimpleNamespace(db=db, listed=listed, reads=reads)


def _answer(report_id=REPORT_ID):
    return {"success": True, "report": {"id": report_id, "title": "Morning AI digest",
                                        "content": "Full digest saved at /Users/x/deliverables/sessions/308/digest.md"}}


def test_the_digest_a_report_points_at_comes_back_with_it_through_the_tool(monkeypatch):
    from modules.tools.discovery import handlers_reports
    from services import report_linked_files as rlf
    from services import report_service

    async def _latest(self, **_kwargs):
        return _answer()

    monkeypatch.setattr(report_service.ReportService, "get_latest_report", _latest)
    seen = _install(monkeypatch, [308], [_row("d1", "digest-2026-11-13.md", "md"), _row("d2", "deck.pdf", "pdf")],
                    {"sessions/308/digest-2026-11-13.md": DIGEST})
    caller_db = object()          # the tool loop's session: never used by the lookups

    out = asyncio.run(handlers_reports.get_latest_report(caller_db, WS, {"agent_name": "News & Research",
                                                                         "report_type": "research"}))

    (only,) = out["report"]["files"]
    assert only["content"] == DIGEST and only["deliverable_id"] == "d1" and only["ticket_id"] == "308"
    assert out["report"]["files_note"] == rlf.FILES_NOTE                         # framed: not instructions
    assert out["report"]["content"].startswith("Full digest saved")              # the report itself is kept
    assert seen.db.params == [{"id": REPORT_ID, "workspace_id": WS}] and seen.db.closed
    assert seen.listed == [(WS, {"source_type": "task", "source_id": "308", "limit": 5})]
    assert seen.reads == ["sessions/308/digest-2026-11-13.md"]                   # the PDF is not read


def test_the_files_share_one_budget_and_the_rest_is_listed_not_read(monkeypatch):
    from services import report_linked_files as rlf

    big = "x" * (rlf.LINKED_TEXT_MAX_CHARS + 10)
    seen = _install(monkeypatch, [308], [_row("d1", "a.md", "md"), _row("d2", "b.txt", "txt")],
                    {"sessions/308/a.md": big, "sessions/308/b.txt": "never read"})

    files = asyncio.run(rlf.with_linked_files(WS, _answer()))["report"]["files"]

    assert len(files[0]["content"]) == rlf.LINKED_TEXT_MAX_CHARS and files[0]["content_truncated"] is True
    assert "content" not in files[1] and files[1]["note"] == rlf.NOT_READ_NOTE
    assert seen.reads == ["sessions/308/a.md"]


def test_a_file_the_worker_cannot_read_says_so_and_the_others_still_come(monkeypatch):
    from services import report_linked_files as rlf

    _install(monkeypatch, [308], [_row("d1", "gone.md", "md"), _row("d2", "b.txt", "txt")],
             {"sessions/308/b.txt": "kept"})

    files = asyncio.run(rlf.with_linked_files(WS, _answer()))["report"]["files"]

    assert files[0]["content_error"] == rlf.UNREADABLE_FILE and files[1]["content"] == "kept"


def test_a_report_with_no_linked_ticket_or_no_text_file_is_unchanged(monkeypatch):
    from services import report_linked_files as rlf

    for linked in ([], None, [308]):
        _install(monkeypatch, linked, [_row("d2", "deck.pdf", "pdf"), _row("d3", "x.md", "md", storage="s3")], {})
        assert asyncio.run(rlf.with_linked_files(WS, _answer())) == _answer()
    failed = {"success": False, "error": "No reports found for this agent"}
    assert asyncio.run(rlf.with_linked_files(WS, failed)) == failed


def test_only_the_first_linked_tickets_are_read(monkeypatch):
    from services import report_linked_files as rlf

    seen = _install(monkeypatch, [1, 2, 3, 4, 5], [], {})
    asyncio.run(rlf.with_linked_files(WS, _answer()))
    assert [f["source_id"] for _, f in seen.listed] == ["1", "2", "3"][: rlf.LINKED_TICKETS_READ_MAX]


def test_a_listing_that_fails_leaves_the_report_and_says_so(monkeypatch, caplog):
    import services.step_files as sf
    from services import report_linked_files as rlf

    seen = _install(monkeypatch, [308], [], {})

    def _broken(*_args, **_kwargs):
        raise sf.NamedFilesUnreadable("the Deliverables could not be listed")

    monkeypatch.setattr(sf, "listed_or_raise", _broken)
    out = asyncio.run(rlf.with_linked_files(WS, _answer()))

    assert out["report"]["content"] == _answer()["report"]["content"]
    assert out["report"][rlf.FILES_ERROR_KEY] == rlf.FILES_UNREADABLE and "files" not in out["report"]
    assert seen.db.closed and "could not be read" in caplog.text


@pytest.mark.parametrize("row, is_text", [
    ({"file_type": "md", "file_path": "a.md"}, True), ({"file_type": "CSV", "file_path": "a"}, True),
    ({"file_path": "a/b/notes.txt"}, True), ({"file_type": "pdf", "file_path": "a.pdf"}, False),
    ({"file_path": "a/b/post.png"}, False), ({"file_type": "md"}, False), ({}, False),
    ({"file_type": "md", "file_path": "a.md", "storage_type": "blog"}, False),
])
def test_only_text_files_in_the_workspace_are_read(row, is_text):
    from services.report_linked_files import _is_text

    assert _is_text(row) is is_text
