"""F349 (night 10b): a document a session makes is copied into its ticket's folder.

A Claude Code session working a ticket called generate_document; the PDF lived only
behind /api/documents/generated/<file> and an object storage link, which the
session's sandbox cannot reach, so the agent never opened what it made. The file is
now also written into ``sessions/<ticket>/`` through the workspace worker, and the
answer names the copy. A board run's card and a chat call are left as they were;
nothing is ever written outside the ticket's folder.

Boundaries faked: the workspace worker, the document render and the session.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace as NS

import pytest

TICKET = 2099
WS = "6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e62"
FILENAME = "20261005_101500_Invoice_HL-2026-0145.pdf"
PDF = b"%PDF-1.7 the document " * 40
PARAMS = {"title": "Invoice HL-2026-0145", "format": "pdf",
          "data": {"sections": [{"title": "Invoice", "content": "Two bags of Guji, 2 x 12.50."}]}}


class _Worker:
    """WorkspaceClient, faked: records what is written where, or refuses."""

    written: dict = {}
    refuse = False

    def __init__(self, workspace_id):
        self.workspace_id = workspace_id

    async def write_binary(self, path, pieces):
        data = b"".join([piece async for piece in pieces])
        if self.refuse:
            return {"success": False, "error": "permission denied"}
        self.written[(self.workspace_id, path)] = data
        return {"success": True, "path": path}


@pytest.fixture
def worker(monkeypatch, tmp_path):
    """The worker faked, and the generated document on disk where the service writes it."""
    import modules.documents.generation_service as gs
    from modules.tools.execution import session_document_folder

    _Worker.written, _Worker.refuse = {}, False
    monkeypatch.setattr(session_document_folder, "WorkspaceClient", _Worker)
    monkeypatch.setattr(gs, "GENERATED_DIR", str(tmp_path))
    folder = tmp_path / WS / "generated"
    folder.mkdir(parents=True)
    (folder / FILENAME).write_bytes(PDF)
    return _Worker


def _call(caller_context, filename=FILENAME, workspace_id=WS):
    """The executor's generate_document hand-off, with the render faked."""
    from modules.tools.execution import exec_document

    async def execute_tool(**kwargs):
        return {"success": True, "status": "success",
                "results": [{"status": "success", "filename": filename, "format": "pdf"}]}

    db = NS(get=lambda model, key: NS(settings={}), query=lambda *a, **k: None)
    executor = NS(db=db, platform_tools=NS(execute_tool=execute_tool))
    return asyncio.run(exec_document.execute_generate_document(
        executor, "generate_document", PARAMS, 341, workspace_id=workspace_id, caller_context=caller_context))


def test_a_sessions_document_is_written_into_its_ticket_folder_and_named(worker):
    answer = _call({"session_task_id": TICKET})

    path = f"sessions/{TICKET}/{FILENAME}"
    assert worker.written == {(WS, path): PDF}
    row = answer["results"][0]
    assert row["session_copy"] == path
    assert f"A copy is in your folder: {path}" in row["session_copy_note"]
    assert row["filename"] == FILENAME and answer["success"] is True


def test_a_mission_sessions_field_beside_the_ticket_still_gets_the_copy(worker):
    _call({"field_context": {"field_id": "f-224"}, "session_task_id": TICKET})

    assert list(worker.written) == [(WS, f"sessions/{TICKET}/{FILENAME}")]


@pytest.mark.parametrize("context", [
    None,                                    # chat
    {"conversation_id": "c-1"},              # chat with a context
    {"board_task_id": 612},                  # an API agent's board card
    {"field_context": {"field_id": "f-1"}},  # a mission step with no session
])
def test_a_call_from_no_session_writes_nothing_and_answers_as_before(worker, context):
    answer = _call(context)

    assert worker.written == {}
    assert "session_copy" not in answer["results"][0]


@pytest.mark.parametrize("filename", [
    "../../../etc/passwd", "../2100/x.pdf", "sub/x.pdf", "..", ".", "", "a\\..\\b.pdf", "/abs/x.pdf", None,
])
def test_a_file_name_that_is_not_a_bare_name_is_never_written(worker, filename):
    answer = _call({"session_task_id": TICKET}, filename=filename)

    assert worker.written == {}
    assert "session_copy" not in answer["results"][0]


@pytest.mark.parametrize("ticket", ["../1", "2099/../../x", -4, 0, True, "not-a-ticket"])
def test_a_ticket_that_is_not_a_ticket_id_is_never_written(worker, ticket):
    _call({"session_task_id": ticket})

    assert worker.written == {}


def test_the_copy_path_stays_inside_the_tickets_folder():
    from modules.tools.execution.session_document_folder import session_copy_path

    assert session_copy_path(TICKET, FILENAME) == f"sessions/{TICKET}/{FILENAME}"
    for name in ("../x.pdf", "a/../../x.pdf", "..", "x/..", "\\x.pdf"):
        assert session_copy_path(TICKET, name) is None


def test_a_document_of_another_workspace_is_never_copied(worker):
    _call({"session_task_id": TICKET}, workspace_id="0b6f1d2e-0000-4000-8000-00000000f349")

    assert worker.written == {}


def test_a_folder_the_worker_cannot_write_is_skipped_and_logged(worker, caplog):
    worker.refuse = True
    with caplog.at_level(logging.WARNING):
        answer = _call({"session_task_id": TICKET})

    assert worker.written == {}
    assert "session_copy" not in answer["results"][0] and answer["success"] is True
    assert "permission denied" in caplog.text


def test_a_failed_generation_is_left_as_it_was(worker):
    from modules.tools.execution.session_document_folder import with_session_copy

    failed = {"success": False, "status": "error", "error": "The document was not made"}
    assert asyncio.run(with_session_copy(failed, {"session_task_id": TICKET}, WS)) is failed
    assert worker.written == {}
