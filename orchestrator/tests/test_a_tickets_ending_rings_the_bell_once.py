"""A ticket's ending tells the owner once: its own notice, not its report's as well.

``finalize_board_task_run`` sends the ticket's notice (``task_complete``, ``task_failed``,
or ``review_pending`` when it enters Review), then files the ticket's report through
``services.task_report.auto_create_task_report``, and ``ReportService.create_report``
raised ``report_submitted`` for it: two notices with the same text for every ticket.

The report every ending files (a run's, a session's, the stall sweep's, the boot
reaper's, all through ``auto_create_task_report``) is still written as before, inside
``notification_dispatcher.holding("report_submitted")``. A report an agent submits
itself still raises ``report_submitted``.

This drives the real completion writer, report writer, ReportService and dispatcher, as
the F197 report test does. Only the workspace file, the metrics rollup, the knowledge
ingest, the triage shadow and the notification rows are stubbed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
AGENT = 325
REPORT_INSERT = "INSERT INTO agent_reports"


@pytest.fixture
def bell(monkeypatch):
    import services.knowledge_flywheel as flywheel
    import services.report_service as report_service
    from core.services.notification_dispatcher import NotificationDispatcher

    rung = []

    class _Files:
        def __init__(self, workspace_id):
            pass

        async def write_file(self, path, content):
            return {"success": True}

    async def _no_ingest(*a, **k):
        return None

    monkeypatch.setattr(report_service, "WorkspaceClient", _Files)
    monkeypatch.setattr(report_service, "compute_execution_metrics", lambda *a, **k: {})
    monkeypatch.setattr(report_service, "_shadow_report_triage", lambda **k: None)
    monkeypatch.setattr(flywheel, "ingest_agent_output", _no_ingest)
    monkeypatch.setattr(NotificationDispatcher, "_load_auto_reporting", lambda self: {})
    monkeypatch.setattr(NotificationDispatcher, "_get_preferences", lambda self, *a, **k: [])
    monkeypatch.setattr(NotificationDispatcher, "_insert_in_app",
                        lambda self, user_id, event_type, title, message, *a, **k: rung.append((event_type, title)))
    return rung


class _Session:
    """Enough of a session for the completion and report writers; it keeps the SQL it ran."""

    def __init__(self, task):
        self.task, self.sql = task, []

    def get(self, *a, **k):
        return self.task

    def query(self, *a):
        return NS(filter=lambda *a, **k: NS(first=lambda: NS(name="Bookkeeper")))

    def execute(self, statement, *a, **k):
        self.sql.append(str(statement))
        return NS(fetchone=lambda: ("report-1",), scalar=lambda: None)

    def commit(self):
        pass

    def rollback(self):
        pass


def _ticket(task_id, title, status="in_progress"):
    return NS(id=task_id, title=title, status=status, error_message=None, completed_at=None,
              assigned_agent_id=AGENT, result=None, started_at=None, execution_id=None)


def _reported(session):
    return [sql for sql in session.sql if REPORT_INSERT in sql]


def test_a_failed_ticket_rings_once_and_its_report_is_still_written(bell):
    from api import board_tasks

    task = _ticket(1170, "Reconcile the October invoices")
    session = _Session(task)
    asyncio.run(board_tasks.finalize_board_task_run(
        session, task_id=task.id, workspace_id=WS, agent_id=AGENT,
        exec_result={"status": "error", "error": "The invoices folder was not found."}))
    assert task.status == "failed"
    assert bell == [("task_failed", "Task failed: Reconcile the October invoices")]  # was: and report_submitted
    assert len(_reported(session)) == 1


def test_a_finished_ticket_s_report_is_written_without_a_notice(bell):
    from services.task_report import auto_create_task_report

    task = _ticket(1171, "Weekly numbers", status="done")
    session = _Session(task)
    asyncio.run(auto_create_task_report(session, WS, task, {"result": "Revenue is up 4% on the week."}))
    assert bell == []  # the ticket's own task_complete is the notice
    assert len(_reported(session)) == 1


def test_a_report_an_agent_submits_itself_still_rings(bell):
    from services.report_service import ReportService

    session = _Session(None)
    filed = asyncio.run(ReportService(session, WS).create_report(
        agent_id=AGENT, agent_name="Bookkeeper", title="Weekly accounts",
        content="# Weekly accounts\nTwo orders bounced this week.", report_type="standup"))
    assert filed["success"] is True
    assert bell == [("report_submitted", "Report: Weekly accounts")]


def test_holding_keeps_back_only_the_named_events_and_only_inside_the_block(bell):
    from core.services.notification_dispatcher import HELD_BY_CALLER, NotificationDispatcher, holding

    def _ring(event_type):
        return asyncio.run(NotificationDispatcher(_Session(None), WS).dispatch(event_type=event_type, title=event_type))

    with holding("report_submitted"):
        held = _ring("report_submitted")
        _ring("task_complete")
    _ring("report_submitted")
    assert held == {"dispatched_to": [], "held_back": HELD_BY_CALLER}
    assert bell == [("task_complete", "task_complete"), ("report_submitted", "report_submitted")]
