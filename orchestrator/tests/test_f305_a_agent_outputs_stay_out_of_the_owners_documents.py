"""F305 (night 9, Gerard's decision): an agent's or a mission's output is not filed as
the owner's document, nor read by the Knowledge Graph, unless the workspace opted in.

Night 9: 28 task reports and 2 mission outputs became documents #1527–#1555 and Auto
cited #1547 (a ticket's own answer) for café payment terms. The report file with its
Execution Metrics, the card's answer, Deliverables and the Reports page stay.
"""
from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from sqlalchemy import text

from services import agent_output_scope as scope
from services import knowledge_flywheel as kf

METRICS = {"model": "openai/gpt-5.2", "llm_calls": 3, "tokens_used": 1234, "cost_usd": 0.0123, "duration_ms": 900}
REPORT = ("# Writer — Task Report\n**Task:** Payment terms for cafés\n**Status:** done\n\n## Result\n"
          "Cafés pay in 30 days.\n\n## Execution Metrics\n- Model: openai/gpt-5.2\n- LLM calls: 3\n")


def _settings(db, ws, settings):
    db.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:ws AS uuid)"),
               {"s": __import__("json").dumps(settings), "ws": str(ws)})
    db.flush()


def _never_uploads(workspace_id):
    raise AssertionError("an agent's output was filed as a document")


def test_a_new_workspace_files_no_agent_output(db_session, seed_workspace, monkeypatch):
    ws = seed_workspace()
    monkeypatch.setattr("api.documents.get_document_manager", _never_uploads)

    assert kf.flywheel_enabled(db_session, ws) is False
    for source in kf.AGENT_OUTPUT_SOURCES:
        got = asyncio.run(kf.ingest_agent_output(db_session, ws, content=REPORT, filename="r.md", source=source,
                                                 source_id="x"))
        assert got is None
    count = db_session.execute(text("SELECT count(*) FROM documents WHERE workspace_id = CAST(:ws AS uuid)"),
                               {"ws": ws}).scalar()
    assert count == 0


def test_an_explicit_opt_in_still_files_it(db_session, seed_workspace, monkeypatch):
    ws = seed_workspace()
    _settings(db_session, ws, {kf.FLYWHEEL_SETTINGS_KEY: True})
    manager = NS(upload_document=AsyncMock(return_value=4242))
    monkeypatch.setattr("api.documents.get_document_manager", lambda workspace_id: manager)
    monkeypatch.setattr(kf, "_kg_extraction_allowed", lambda *args: False)

    got = asyncio.run(kf.ingest_agent_output(db_session, ws, content=REPORT, filename="m.md",
                                             source=kf.SOURCE_MISSION_SYNTHESIS, source_id="run-1"))
    assert got == 4242
    assert manager.upload_document.call_args.kwargs["source_type"] == kf.AGENT_OUTPUT_SOURCE_TYPE


def test_the_report_file_and_its_metrics_stay_and_nothing_else_is_touched(monkeypatch):
    """The default-off path writes the report file and its row with the metrics, and
    runs no statement against llm_usage or the cards the F249 lessons are read from."""
    from services.report_service import ReportService

    written = {}

    async def _write(self, path, content):
        written[path] = content
        return {"success": True}

    monkeypatch.setattr("services.report_service.WorkspaceClient.write_file", _write)
    monkeypatch.setattr("core.services.notification_dispatcher.NotificationDispatcher.dispatch", AsyncMock())
    monkeypatch.setattr("api.documents.get_document_manager", _never_uploads)
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = ("8b0c1e9e-0000-4000-8000-000000000001",)

    got = asyncio.run(ReportService(db, UUID(int=5)).create_report(
        agent_id=7, agent_name="Writer", title="Task: Payment terms for cafés", content=REPORT,
        report_type="task", metrics=METRICS, linked_task_ids=[1851]))

    assert got["success"] is True
    assert "## Execution Metrics" in written[got["file_path"]]
    insert = db.execute.call_args_list[0]
    assert "INSERT INTO agent_reports" in str(insert.args[0])
    assert __import__("json").loads(insert.args[1]["metrics"]) == METRICS
    statements = " ".join(str(c.args[0]) for c in db.execute.call_args_list if c.args)
    assert "llm_usage" not in statements and "board_tasks" not in statements


def test_a_done_ticket_files_nothing_and_leaves_earlier_filings(db_session, seed_workspace, monkeypatch):
    from services import report_knowledge as rk

    removed = []
    monkeypatch.setattr(rk, "remove_documents", lambda db, ws, ids: removed.extend(ids) or list(ids))
    monkeypatch.setattr(rk, "file_report", AsyncMock(side_effect=AssertionError("filed")))
    ws = seed_workspace()

    assert asyncio.run(rk.file_ticket_report(db_session, ws, NS(id=1851, started_at=None))) is None
    assert removed == []


def test_a_finished_mission_is_delivered_but_not_filed(monkeypatch):
    run = NS(id="run-35", workspace_id=UUID(int=9), goal="Christmas boxes", config={})
    steps = [NS(sequence_number=1, title="Plan", output="Three boxes.")]
    coordinator = NS(_emit_mission_document=AsyncMock(return_value=None),
                     _register_final_output_deliverable=AsyncMock(return_value="d-1"))
    save = AsyncMock(side_effect=AssertionError("filed as a document"))
    monkeypatch.setattr(scope, "_opted_in", lambda db, ws: False)
    monkeypatch.setattr(scope, "_verified_tasks", lambda db, r: steps)

    got = asyncio.run(scope.ingests_only_when_opted_in(save)(coordinator, MagicMock(), run))

    assert got is None
    coordinator._register_final_output_deliverable.assert_awaited_once()
    assert run.config["output_ingest"] == scope.SKIPPED_MARKER
    assert "# Mission: Christmas boxes" in coordinator._emit_mission_document.await_args.args[2]


def test_an_opted_in_mission_is_filed_as_before(monkeypatch):
    run = NS(id="run-36", workspace_id=UUID(int=9), goal="g", config={})
    save = AsyncMock(return_value=99)
    monkeypatch.setattr(scope, "_opted_in", lambda db, ws: True)

    assert asyncio.run(scope.ingests_only_when_opted_in(save)(NS(), MagicMock(), run)) == 99


@pytest.fixture
def papers(db_session, seed_workspace, monkeypatch):
    from core.models.core import Document

    ws = seed_workspace()

    @contextlib.contextmanager
    def _session():
        yield db_session

    monkeypatch.setattr("core.database.database.get_db_session", _session)

    def document(filename, source_type=None):
        doc = Document(filename=filename, workspace_id=UUID(ws), source_type=source_type, status="completed")
        db_session.add(doc)
        db_session.flush()
        return {"type": "document", "id": doc.id, "path": filename, "text": "Brazil Cerrado is late."}

    return NS(ws=ws, owner=document("price-list.md"), report=document("playbook-report.md", "agent_output"))


def test_the_graph_reads_only_the_owners_documents(db_session, papers):
    roster = {"type": "agents", "rows": []}

    kept = scope.owners_sources(papers.ws, [papers.owner, papers.report, roster])

    assert kept == [papers.owner, roster]


def test_an_opted_in_graph_reads_both(db_session, papers):
    _settings(db_session, papers.ws, {kf.FLYWHEEL_SETTINGS_KEY: True})

    assert scope.owners_sources(papers.ws, [papers.owner, papers.report]) == [papers.owner, papers.report]


def test_the_graph_drops_a_reports_pending(db_session, papers):
    pending = [{"type": "report", "id": "r1", "text": "Brazil Cerrado"}, {"type": "document", "id": 5},
               {"type": "mission_synthesis", "id": "run", "document_id": 6}]

    assert scope.owners_pending(papers.ws, pending) == [{"type": "document", "id": 5}]
