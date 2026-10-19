"""A HARNESS tick on the real database, for the PRD-263 harness_runs suites.

A tick commits as it goes (its run row, each phase), so these suites seed a
committed workspace and sweep it afterwards. The platform actions a tick calls
are faked: every action succeeds, so the audit report is "ok". COLLECT reads the
real tables (services.harness_metrics). The dormancy check is stubbed to "ready",
because a seeded workspace holds fewer than its three active agents. The judged
week is pinned so a run can never straddle a week boundary. Shared by
test_prd263_harness_run_record.py, test_prd263_harness_advisory_lock.py and
test_prd263_harness_diagnose_realdb.py.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import text

WINDOW = "2026-W42"
READY = {"active_agents": 3, "min_required_agents": 3, "agents_ok": True}
REPORT = "platform_submit_report"
# How long a faked platform action takes: a running tick yields while it holds the lock.
ACTION_SECONDS = 0.01


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


def fake_platform(monkeypatch, tmp_path) -> List[tuple]:
    """Fake the platform actions, stub dormancy, pin the week and the volume.
    Returns the (workspace id, action) calls, in order."""
    from config import config
    from services import harness_runs
    from services.harness_service import HarnessService

    calls: List[tuple] = []

    class _Executor:
        def __init__(self, db=None, workspace_id=None, **_kwargs):
            self.workspace_id = workspace_id

        async def execute(self, action: str, params: Dict[str, Any], caller_context=None) -> Dict[str, Any]:
            calls.append((str(self.workspace_id), action))
            await asyncio.sleep(ACTION_SECONDS)
            return {"success": True, "data": [] if action == "platform_list_agents" else {}}

    monkeypatch.setattr("modules.tools.discovery.platform_executor.PlatformActionExecutor", _Executor)
    monkeypatch.setattr(HarnessService, "_sufficiency_breakdown", lambda self, db, ws: dict(READY))
    monkeypatch.setattr(harness_runs, "last_complete_iso_week", lambda now=None: WINDOW)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    return calls


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
