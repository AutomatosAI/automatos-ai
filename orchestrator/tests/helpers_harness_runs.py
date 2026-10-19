"""A HARNESS tick on the real database, for the PRD-263 harness_runs suites.

A tick commits as it goes (its run row, each phase), so these suites seed a
committed workspace and sweep it afterwards. The audit report, the one write a
tick makes outside its row (ReportService.create_report, PRD-263 US-048), is faked
and succeeds, so it is "ok"; any platform action is recorded too, and a run calls
none. COLLECT reads the real tables (services.harness_metrics). The dormancy check
is stubbed to "ready", because a seeded workspace holds fewer than its three active
agents. The judged week is pinned so a run can never straddle a week boundary.
Shared by test_prd263_harness_run_record.py, test_prd263_harness_advisory_lock.py and
test_prd263_harness_diagnose_realdb.py. The suites that run the real ReportService
(test_prd263_harness_report_only.py, test_prd263_harness_audit_report.py) pin the
run only (pin_run), record platform actions without running them
(record_platform_actions), build agent_reports where the models did not
(agent_reports_engine) and stub the report's side effects (stub_report_side_effects).
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Iterator, List
from uuid import UUID

from sqlalchemy import text

WINDOW = "2026-W42"
READY = {"active_agents": 3, "min_required_agents": 3, "agents_ok": True}
# What a faked audit report is recorded as: the report is no platform action (US-048).
REPORT = "ReportService.create_report"
# How long a faked audit report takes: a running tick yields while it holds the lock.
ACTION_SECONDS = 0.01
# agent_reports has no SQLAlchemy model, so a database built from the models has no
# such table; the audit report's insert needs these columns (ReportService.create_report).
AGENT_REPORTS = """
    CREATE TABLE agent_reports (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL, agent_id INTEGER, agent_name TEXT, heartbeat_result_id INTEGER,
        report_type TEXT, title TEXT, summary TEXT, status TEXT,
        file_path TEXT, file_type TEXT, file_size_bytes INTEGER,
        metrics JSONB, attachments JSONB, recommendations JSONB, action_items JSONB, linked_task_ids JSONB,
        requires_approval BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ DEFAULT now(), updated_at TIMESTAMPTZ DEFAULT now()
    )
"""


def seed_workspace(new_session) -> UUID:
    """A committed, active workspace with its Auto, for the audit report to name."""
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, is_active) VALUES (CAST(:id AS uuid), 'prd263-harness-runs', true)"),
              {"id": ws})
    s.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, is_system_agent) "
                   "VALUES ('Auto', 'system', CAST(:id AS uuid), 'active', true)"), {"id": ws})
    s.commit()
    s.close()
    return UUID(ws)


def sweep_workspace(new_session, ws: UUID) -> None:
    s = new_session.sweep()
    for sql in ("DELETE FROM harness_runs WHERE workspace_id = CAST(:id AS uuid)",
                "DELETE FROM harness_task_ledger WHERE workspace_id = CAST(:id AS uuid)",
                "DELETE FROM board_tasks WHERE workspace_id = CAST(:id AS uuid)",
                "DELETE FROM llm_usage WHERE workspace_id = CAST(:id AS uuid)",
                "DELETE FROM agents WHERE workspace_id = CAST(:id AS uuid)",
                "DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"):
        s.execute(text(sql), {"id": str(ws)})
    s.commit()
    s.close()


def pin_run(monkeypatch, tmp_path) -> None:
    """Stub dormancy, pin the week and the volume."""
    from config import config
    from services import harness_runs
    from services.harness_service import HarnessService

    monkeypatch.setattr(HarnessService, "_sufficiency_breakdown", lambda self, db, ws: dict(READY))
    monkeypatch.setattr(harness_runs, "last_complete_iso_week", lambda now=None: WINDOW)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))


def fake_platform(monkeypatch, tmp_path) -> List[tuple]:
    """Pin the run (pin_run), fake the audit report and record any platform action.
    Returns the (workspace id, action) calls, in order: REPORT for the audit report,
    and a platform action's name for any action, which a run never calls (US-048)."""
    from services.report_service import ReportService

    calls: List[tuple] = []

    class _Executor:
        def __init__(self, db=None, workspace_id=None, **_kwargs):
            self.workspace_id = workspace_id

        async def execute(self, action: str, params: Dict[str, Any], caller_context=None) -> Dict[str, Any]:
            calls.append((str(self.workspace_id), action))
            return {"success": True, "data": {}}

    async def _create_report(self, **_fields) -> Dict[str, Any]:
        calls.append((str(self.workspace_id), REPORT))
        await asyncio.sleep(ACTION_SECONDS)
        return {"success": True, "report_id": str(uuid.uuid4())}

    pin_run(monkeypatch, tmp_path)
    monkeypatch.setattr("modules.tools.discovery.platform_executor.PlatformActionExecutor", _Executor)
    monkeypatch.setattr(ReportService, "create_report", _create_report)
    return calls


def record_platform_actions(monkeypatch, actions: List[str]) -> None:
    """Record each platform action's name in ``actions`` and answer it without running
    its handler, so an audit report submitted as an action would leave no row."""

    class _Executor:
        def __init__(self, db=None, workspace_id=None, **_kwargs):
            pass

        async def execute(self, action: str, params: Dict[str, Any], caller_context=None) -> Dict[str, Any]:
            actions.append(action)
            return {"success": True, "data": {"id": 1}}

    monkeypatch.setattr("modules.tools.discovery.platform_executor.PlatformActionExecutor", _Executor)


def agent_reports_engine() -> Iterator[Any]:
    """A module-scoped ``engine`` fixture's body (conftest's new_session binds to it)
    that creates agent_reports when the models built the database, and drops it after."""
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    with eng.begin() as conn:
        built = not conn.execute(text("SELECT to_regclass('agent_reports') IS NOT NULL")).scalar()
        if built:
            conn.execute(text(AGENT_REPORTS))
    yield eng
    if built:
        with eng.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS agent_reports"))
    eng.dispose()


def stub_report_side_effects(monkeypatch, seen) -> None:
    """Run the real ReportService with its workspace file and knowledge ingest stubbed
    and thumbnails off; record each notification it dispatches (``seen.notifications``)
    and each channel notice sent (``seen.channel_notices``)."""
    import core.services.notification_service as notification_service
    import services.knowledge_flywheel as flywheel
    import services.report_service as report_service
    from config import config
    from core.services.notification_dispatcher import NotificationDispatcher

    class _Files:
        def __init__(self, workspace_id):
            pass

        async def write_file(self, path, content):
            return {"success": True}

    async def _no_ingest(*_args, **_kwargs):
        return None

    async def _dispatch(self, event_type, title, *_args, **_kwargs):
        seen.notifications.append(event_type)
        return {}

    async def _channel_notice(workspace_id, message, channel=None):
        seen.channel_notices.append(message)
        return True

    monkeypatch.setattr(report_service, "WorkspaceClient", _Files)
    monkeypatch.setattr(flywheel, "ingest_agent_output", _no_ingest)
    monkeypatch.setattr(NotificationDispatcher, "dispatch", _dispatch)
    monkeypatch.setattr(notification_service, "send_workspace_notification", _channel_notice)
    monkeypatch.setattr(config, "DOCUMENT_THUMBNAILS_ENABLED", False)


def runs(new_session, ws: UUID) -> List[Dict[str, Any]]:
    """The workspace's harness_runs rows, oldest first, read on a fresh session."""
    s = new_session()
    rows = s.execute(text("SELECT * FROM harness_runs WHERE workspace_id = CAST(:id AS uuid) ORDER BY id"),
                     {"id": str(ws)}).mappings().all()
    s.close()
    return [dict(row) for row in rows]


def reports(calls: List[tuple], ws: UUID) -> int:
    """How many audit reports the workspace's runs submitted."""
    return sum(1 for workspace, action in calls if workspace == str(ws) and action == REPORT)
