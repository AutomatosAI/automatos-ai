"""F197, partial after the refresh-6 build: a failed task's report no longer rings
the bell again during a credit outage.

TESTER's probe on 7deca984d with the credit empty failed tickets 1170 and 1171
(agent 325). Both failed with the plain sentence, and the bell said "Out of AI
credit" once and held 1171's task_failed. But each failed task's REPORT still
rang: two report_submitted notices at 09:30:11 ("Report: Task: F197 probe 1/2 …").
A failed task's result is blank, so its report's summary fell back to the
content's first line, "**Task:** …". The notice carries the summary, so the bell
never saw the credit failure. The error sits under "## Error", further down.

This drives the real completion writer, report writer and ReportService. Only
the workspace file, the metrics rollup, the knowledge ingest and the notification
rows are stubbed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
REFUSED = ("Task execution failed after 2 attempts: Error code: 402 - {'error': {'message': 'This request requires "
           "more credits, or fewer max_tokens. You requested up to 8000 tokens, but can only afford 3972', "
           "'code': 402}}")


@pytest.fixture
def night(monkeypatch):
    import services.report_service as report_service
    from core.llm import credit
    from core.services.notification_dispatcher import NotificationDispatcher

    credit.reset()
    rung = []

    class _Files:
        def __init__(self, workspace_id):
            pass

        async def write_file(self, path, content):
            return {"success": True}

    monkeypatch.setattr(report_service, "WorkspaceClient", _Files)
    monkeypatch.setattr(report_service, "compute_execution_metrics", lambda *a, **k: {})
    monkeypatch.setattr(report_service, "_shadow_report_triage", lambda **k: None)
    import services.knowledge_flywheel as flywheel

    async def _no_ingest(*a, **k):
        return None

    monkeypatch.setattr(flywheel, "ingest_agent_output", _no_ingest)            # no RAG ingest of a test report
    monkeypatch.setattr(NotificationDispatcher, "_load_auto_reporting", lambda self: {})
    monkeypatch.setattr(NotificationDispatcher, "_get_preferences", lambda self, *a, **k: [])
    monkeypatch.setattr(NotificationDispatcher, "_insert_in_app",
                        lambda self, user_id, event_type, title, message, *a, **k: rung.append((event_type, title)))

    def fail(task_id, title):
        from api import board_tasks

        task = NS(id=task_id, title=title, status="in_progress", error_message=None, completed_at=None,
                  assigned_agent_id=325, result=None, started_at=None, execution_id=None)

        class _Session:
            def get(self, *a, **k):
                return task

            def query(self, *a):
                return NS(filter=lambda *a, **k: NS(first=lambda: NS(name="Club newsletter helper")))

            def execute(self, *a, **k):
                return NS(fetchone=lambda: (f"report-{task_id}",))

            def commit(self):
                pass

            def rollback(self):
                pass

        asyncio.run(board_tasks.finalize_board_task_run(
            _Session(), task_id=task_id, workspace_id=WS, agent_id=325,
            exec_result={"status": "error", "error": REFUSED}))
        return task
    return NS(fail=fail, rung=rung)


def test_two_tasks_that_ran_out_of_credit_ring_the_bell_once(night):
    night.fail(1170, "F197 probe 1/2: reply with the word hello")
    night.fail(1171, "F197 probe 2/2: reply with the word hello")
    assert night.rung == [("task_failed", "Out of AI credit")]                 # today: 3, one per report too


def test_a_failed_tasks_report_says_why_it_failed(night, monkeypatch):
    import services.report_service as report_service

    summaries = []
    real = report_service.ReportService.create_report

    async def keep(self, **kwargs):
        summaries.append(kwargs.get("summary"))
        return await real(self, **kwargs)

    monkeypatch.setattr(report_service.ReportService, "create_report", keep)
    night.fail(1170, "F197 probe 1/2: reply with the word hello")
    assert summaries == ["The AI provider's account ran out of credit, so this stopped before it finished. "
                         "Top up the provider account, then run it again."]
