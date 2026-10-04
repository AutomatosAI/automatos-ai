"""F235 (night 6, the persona's #4; Gerard, 2 Oct: "approved work only"): a job
report becomes a Document, the knowledge Auto quotes, only once its work is approved.

Night 6 ended with 122 of 129 Documents being the product's own job reports, filed
the moment each run finished: rejected rounds, failures and empty answers included,
each "Status: review" whether approved or not, and Auto quoted them back as fact.
"I only want what I upload or approve to become knowledge, and an approved answer
should replace its drafts."
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from services import report_knowledge as rk

STARTED = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
PLAYBOOK_REPORT = "# Weekly social posts — Playbook Report\n**Execution:** exec-4c311516f861\n**Status:** completed"


@pytest.fixture
def reports(db_session):
    """agent_reports is raw DDL (alembic prd76/wave1c), which the test schema never
    builds: a temp table with the columns the wait reads."""
    db_session.execute(text("DROP TABLE IF EXISTS pg_temp.agent_reports"))
    db_session.execute(text(
        "CREATE TEMP TABLE agent_reports (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), workspace_id uuid NOT NULL, "
        "agent_name varchar(255), report_type varchar(50) DEFAULT 'task', title text DEFAULT 'A report', "
        "file_path text DEFAULT 'reports/writer/r.md', linked_task_ids jsonb NOT NULL DEFAULT '[]'::jsonb, "
        "grade integer, acknowledged_at timestamptz, created_at timestamptz NOT NULL DEFAULT now())"))

    def _report(ws, *, tasks=(), grade=None, acknowledged=False, created=None):
        return str(db_session.execute(text(
            "INSERT INTO agent_reports (workspace_id, linked_task_ids, grade, acknowledged_at, created_at) "
            "VALUES (CAST(:ws AS uuid), CAST(:tasks AS jsonb), :grade, CASE WHEN :ack THEN now() END, "
            "COALESCE(CAST(:created AS timestamptz), now())) RETURNING id"),
            {"ws": str(ws), "tasks": f"[{', '.join(str(t) for t in tasks)}]", "grade": grade, "ack": acknowledged,
             "created": created}).scalar())

    return _report


def _task(db, ws, status, **over):
    from core.models.core import BoardTask

    task = BoardTask(workspace_id=UUID(str(ws)), title=f"A {status} ticket", status=status, priority="low", **over)
    db.add(task)
    db.flush()
    return task


def _document(db, ws, report_id):
    """A Document filed from a report, the way the flywheel tags it."""
    from core.models.core import Document

    doc = Document(filename=f"{report_id}.md", workspace_id=UUID(str(ws)), source_type="agent_output",
                   tags=["agent_output", "report", f"report:{report_id}"], status="processed")
    db.add(doc)
    db.flush()
    return doc.id


# ── whether a report waits ───────────────────────────────────────────────────

def test_a_tickets_report_waits_for_the_ticket_to_be_done(db_session, seed_workspace, reports):
    ws = seed_workspace()
    for status, held in (("review", True), ("failed", True), ("assigned", True), ("done", False)):
        report_id = reports(ws, tasks=[_task(db_session, ws, status).id])
        assert rk.held_for_approval(db_session, ws, report_id, "the round's answer") is held, status


def test_a_playbook_runs_report_follows_its_card(db_session, seed_workspace, reports):
    ws = seed_workspace()
    card = _task(db_session, ws, "done", source_type="recipe", source_id="exec-4c311516f861")
    report_id = reports(ws)
    assert rk.held_for_approval(db_session, ws, report_id, PLAYBOOK_REPORT) is False
    card.status = "failed"
    db_session.flush()
    assert rk.held_for_approval(db_session, ws, report_id, PLAYBOOK_REPORT) is True


def test_a_report_for_no_ticket_waits_for_the_owner(db_session, seed_workspace, reports):
    ws = seed_workspace()
    assert rk.held_for_approval(db_session, ws, reports(ws), "Morning numbers") is True       # a heartbeat's
    assert rk.held_for_approval(db_session, ws, reports(ws, grade=3), "x") is True
    assert rk.held_for_approval(db_session, ws, reports(ws, grade=5), "x") is False
    assert rk.held_for_approval(db_session, ws, reports(ws, acknowledged=True), "x") is False
    assert rk.held_for_approval(db_session, ws, "rep-1", "x") is False         # no report row: filed as before


def test_the_flywheel_never_files_a_report_that_waits(db_session, seed_workspace, reports, monkeypatch):
    from services import knowledge_flywheel as kf

    monkeypatch.setattr(kf, "flywheel_enabled", lambda *args: True)   # F305: filing is the workspace's opt-in
    ws = UUID(seed_workspace())
    task = _task(db_session, ws, "review")
    report_id = reports(ws, tasks=[task.id])

    def _no_upload(workspace_id):
        raise AssertionError("a report waiting for its approval is never uploaded")

    monkeypatch.setattr("api.documents.get_document_manager", _no_upload)
    ingest = dict(content="Draft: refund £11.50", filename="r.md", source="report", source_id=report_id)
    assert asyncio.run(kf.ingest_agent_output(db_session, ws, **ingest)) is None       # night 6: filed at once
    task.status = "done"
    db_session.flush()
    manager = NS(upload_document=AsyncMock(return_value=901))
    monkeypatch.setattr("api.documents.get_document_manager", lambda workspace_id: manager)
    monkeypatch.setattr(kf, "_kg_extraction_allowed", lambda *args: False)
    assert asyncio.run(kf.ingest_agent_output(db_session, ws, **ingest)) == 901


# ── filing it once approved ─────────────────────────────────────────────────

@pytest.fixture
def filing(monkeypatch):
    filed, removed = [], []

    async def _file(db, workspace_id, report_id):
        filed.append(report_id)
        return 77

    monkeypatch.setattr(rk, "file_report", _file)
    monkeypatch.setattr("services.knowledge_flywheel.flywheel_enabled", lambda *args: True)   # F305: opted in
    monkeypatch.setattr(rk, "remove_documents", lambda db, workspace_id, ids: removed.extend(ids) or list(ids))
    return filed, removed


def test_a_done_ticket_files_its_approved_round_and_drops_the_rest(db_session, seed_workspace, reports, filing):
    ws = seed_workspace()
    task = _task(db_session, ws, "done", started_at=STARTED)
    rejected = reports(ws, tasks=[task.id], created=STARTED - timedelta(hours=1))
    approved = reports(ws, tasks=[task.id], created=STARTED + timedelta(minutes=5))
    old_copy = _document(db_session, ws, rejected)               # filed at once, before F235
    filed, removed = filing

    assert asyncio.run(rk.file_ticket_report(db_session, ws, task)) == 77
    assert filed == [approved] and removed == [old_copy]


def test_a_round_filed_when_it_was_written_is_not_filed_twice(db_session, seed_workspace, reports, filing):
    ws = seed_workspace()
    task = _task(db_session, ws, "done", started_at=STARTED)       # review off: filed at its creation
    _document(db_session, ws, reports(ws, tasks=[task.id], created=STARTED + timedelta(minutes=1)))
    filed, removed = filing
    assert asyncio.run(rk.file_ticket_report(db_session, ws, task)) is None
    assert filed == [] and removed == []


def test_before_this_runs_report_is_written_nothing_older_is_filed(db_session, seed_workspace, reports, filing):
    """Finalize marks the ticket done before it writes the round's report: the
    previous round must not be filed as the approved one meanwhile."""
    ws = seed_workspace()
    task = _task(db_session, ws, "done", started_at=STARTED)
    reports(ws, tasks=[task.id], created=STARTED - timedelta(hours=2))
    filed, _removed = filing
    assert asyncio.run(rk.file_ticket_report(db_session, ws, task)) is None and filed == []


def test_the_filed_report_says_approved(db_session, seed_workspace, reports, monkeypatch):
    ws = seed_workspace()
    report_id = reports(ws, tasks=[_task(db_session, ws, "done").id])
    content = "# Draft the club email\n**Status:** review\n\nHi all, the October coffee is the Guji."

    async def _get(self, rid, include_content=True):
        return {"success": True, "report": {"id": rid, "title": "Draft the club email", "report_type": "task",
                                            "agent_name": "WRITER", "file_path": "reports/writer/r.md",
                                            "content": content}}

    rewritten, ingested = [], []

    async def _rewrite(workspace_id, path, new_content):
        rewritten.append(new_content)

    async def _ingest(db, workspace_id, **kwargs):
        ingested.append(kwargs)
        return 5

    monkeypatch.setattr("services.report_service.ReportService.get_report", _get)
    monkeypatch.setattr(rk, "_rewrite", _rewrite)
    monkeypatch.setattr("services.knowledge_flywheel.ingest_agent_output", _ingest)
    assert asyncio.run(rk.file_report(db_session, ws, report_id)) == 5
    filed = ingested[0]["content"]
    assert "**Status:** approved" in filed and "**Status:** review" not in filed      # night 6: 'review', approved
    assert rewritten == [filed] and ingested[0]["extra_tags"] == [f"report:{report_id}"]


def test_a_filing_that_fails_never_breaks_the_tickets_completion(db_session, seed_workspace, monkeypatch):
    async def _down(*args):
        raise RuntimeError("the workspace worker is down")

    monkeypatch.setattr(rk, "file_ticket_report", _down)
    ws = seed_workspace()
    asyncio.run(rk.file_done_ticket(db_session, ws, _task(db_session, ws, "done")))
    assert db_session.execute(text("SELECT 1")).scalar() == 1                 # the caller's transaction is usable


def test_the_board_the_grade_and_the_acknowledgement_file_it():
    from api import board_tasks
    from api import reports as reports_api
    from modules.tools.discovery import handlers_reports

    assert "await file_done_ticket(db, workspace_id, task)" in inspect.getsource(board_tasks._dispatch_task_complete)
    assert "await file_owner_approved(db, ctx.workspace_id, report_id)" in inspect.getsource(reports_api.grade_report)
    assert "await file_owner_approved(db, workspace_id, str(result[0]))" in inspect.getsource(
        handlers_reports.acknowledge_report)


def test_auto_moving_a_ticket_to_done_files_its_report(db_session, seed_workspace, monkeypatch):
    """"Approve 1215" in chat is the owner's word; the status tool is a plain write
    that never passes the board's completion path, so its result is followed up."""
    from modules.tools.discovery import handlers_board_task_done as done_tool
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ws = UUID(seed_workspace())
    one, two = _task(db_session, ws, "done"), _task(db_session, ws, "done")
    filed = []

    async def _file(db, workspace_id, task):
        filed.append(task.id)

    monkeypatch.setattr("services.report_knowledge.file_done_ticket", _file)
    for answer in ({"success": True, "task_id": one.id, "status": "done"},
                   {"success": True, "status": "done", "updated": [one.id, two.id], "failed": []},
                   {"success": True, "task_id": one.id, "status": "review"},
                   {"success": False, "task_id": two.id, "status": "done", "error": "no agent"}):
        async def _move(db, workspace_id, params, answer=answer):
            return answer

        monkeypatch.setattr(done_tool, "_move_status", _move)
        asyncio.run(done_tool.update_board_task_status(db_session, ws, {"status": answer["status"]}))
    assert sorted(filed) == sorted([one.id, one.id, two.id])
    handlers = PlatformActionExecutor(db=None, workspace_id=None)._handlers
    assert handlers["platform_update_task_status"] is done_tool.update_board_task_status


def test_a_grade_of_4_or_5_files_the_report(monkeypatch):
    from api import reports as reports_api

    approved = []

    async def _grade(self, **kwargs):
        return {"success": True}

    async def _file(db, workspace_id, report_id):
        approved.append(report_id)

    monkeypatch.setattr(reports_api.ReportService, "grade_report", _grade)
    monkeypatch.setattr("services.report_knowledge.file_owner_approved", _file)
    ctx = NS(workspace_id=uuid4(), user=NS(id=None))
    for grade in (3, 5):
        asyncio.run(reports_api.grade_report("r-1", reports_api.GradeRequest(grade=grade), ctx=ctx, db=MagicMock()))
    assert approved == ["r-1"]


def test_removing_a_document_drops_its_file_and_its_vectors(tmp_path, monkeypatch):
    from services.document_removal import remove_document

    stored = tmp_path / "r.md"
    stored.write_text("x")
    deleted = []
    monkeypatch.setattr("api.documents.get_document_manager", lambda workspace_id: NS(delete_document=deleted.append))
    remove_document(NS(id=12, file_path=str(stored)), "ws-1")
    assert deleted == [12] and not stored.exists()


# ── the one-off cleanup ─────────────────────────────────────────────────────

def test_the_cleanup_keeps_only_approved_work(db_session, seed_workspace, reports, monkeypatch):
    from scripts import remove_unapproved_report_documents as cleanup

    async def _get(self, rid, include_content=True):
        return {"success": True, "report": {"content": "Morning numbers: 14 orders."}}

    monkeypatch.setattr("services.report_service.ReportService.get_report", _get)
    ws = UUID(seed_workspace())
    done = _task(db_session, ws, "done", started_at=STARTED)
    approved = reports(ws, tasks=[done.id], created=STARTED + timedelta(minutes=5))
    earlier = reports(ws, tasks=[done.id], created=STARTED - timedelta(hours=1))
    waiting = reports(ws, tasks=[_task(db_session, ws, "review").id])
    graded, heartbeat = reports(ws, grade=5), reports(ws)

    def kind(report_id):
        return asyncio.run(cleanup.verdict(db_session, ws, report_id))[0]

    assert [kind(r) for r in (approved, earlier, waiting, graded, heartbeat)] == [
        "keep", "remove", "remove", "keep", "remove"]
    assert kind(str(uuid4())) == "unknown"
    doc = _document(db_session, ws, earlier)
    _document(db_session, ws, approved)
    plan = asyncio.run(cleanup.review(db_session, ws))
    assert [d["document"] for d in plan["remove"]] == [doc] and len(plan["keep"]) == 1


@pytest.mark.parametrize("filename, suffix", [
    ("2026-10-02_monday-checklist.md", ".md"), ("totals.CSV", ".csv"), ("notes", ".md"),
    ("run.sh", ".md"), ("x.md/../../etc/passwd", ".md"), ("a.md\x00.py", ".md"),
])
def test_a_reports_name_never_chooses_its_temp_path(filename, suffix):
    """CodeQL (py/path-injection) on this PR: the report's file name comes from
    the agent's name, and its extension became the temp file's suffix."""
    from services.knowledge_flywheel import temp_suffix

    assert temp_suffix(filename) == suffix
