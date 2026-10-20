"""Seeding and running HARNESS DIAGNOSE on the real database, for the PRD-263 suites.

Rows go into the transactional ``db_session`` (rolled back after each test), and
``judge`` runs the real pipeline (``HarnessService._run_recorded``: COLLECT,
DIAGNOSE, PRESCRIBE, REPORT) for one ISO week on its own harness_runs row.
The audit report must be faked first (``helpers_harness_runs.fake_platform``).
Weeks are named by their offset from 2026-W42 (Monday 12 Oct 2026), so nothing
depends on today's date. Shared by test_prd263_harness_diagnose_realdb.py,
test_prd263_harness_verdict_history_realdb.py, test_prd263_harness_tool_failure_window.py
and test_prd263_harness_report_only.py.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import text

WEEK = "2026-W42"
MONDAY = datetime(2026, 10, 12, tzinfo=timezone.utc)  # Monday of W42
PLAN_TIER = "subscription"


def at(week_offset: int) -> datetime:
    """Wednesday 10:00 UTC of the week ``week_offset`` weeks from W42 (-1 is W41)."""
    return MONDAY + timedelta(weeks=week_offset, days=2, hours=10)


def agent(db, ws, name, **over) -> int:
    """An active agent in ``ws``; returns its id."""
    from core.models.core import Agent

    row = Agent(name=name, agent_type=over.pop("agent_type", "custom"), workspace_id=UUID(str(ws)),
                status="active", **over)
    db.add(row)
    db.flush()
    return row.id


def auto(db, ws) -> int:
    """The workspace's Auto: it signs the audit report and is never judged."""
    return agent(db, ws, "Auto", agent_type="system", slug=f"auto-{ws}", is_system_agent=True)


def tickets(db, ws, agent_id, week_offset, done, failed) -> None:
    """``done`` done and ``failed`` failed tickets for the agent in that week."""
    from core.models.core import BoardTask

    moment = at(week_offset)
    for status, count in (("done", done), ("failed", failed)):
        for _ in range(count):
            db.add(BoardTask(workspace_id=UUID(str(ws)), title=f"A {status} ticket", status=status,
                             priority="medium", assigned_agent_id=agent_id, completed_at=moment, updated_at=moment))
    db.flush()


def usage(db, ws, agent_id, moment, *, cost=0.0, tokens=0, tier="direct", request_type="session",
          status="success", execution_id=None) -> None:
    """One llm_usage row."""
    from core.models.core import LLMUsage

    db.add(LLMUsage(workspace_id=UUID(str(ws)), model_id="anthropic/claude-haiku-5.5", provider="anthropic",
                    tier=tier, agent_id=agent_id, execution_id=execution_id, request_type=request_type,
                    input_tokens=tokens, output_tokens=0, total_tokens=tokens, input_cost=cost,
                    output_cost=0.0, total_cost=cost, status=status, created_at=moment))


def runs(db, ws, agent_id, week_offset, count, *, cost=0.0, tokens=1000, tier="direct") -> None:
    """``count`` runs in that week: one llm_usage row each, with its own execution_id."""
    for i in range(count):
        usage(db, ws, agent_id, at(week_offset), cost=cost, tokens=tokens, tier=tier,
              execution_id=f"session:{agent_id}:{week_offset}:{i}")
    db.flush()


def tool_runs(db, ws, agent_id, week_offset, count, *, app="GMAIL", status="success", error_code=None,
              error_message=None, source="production") -> None:
    """``count`` tool runs of ``app`` by the agent in that week (tool_execution_logs is naive UTC)."""
    from core.models.composio_cache import ToolExecutionLog

    moment = at(week_offset).replace(tzinfo=None)
    for _ in range(count):
        db.add(ToolExecutionLog(workspace_id=UUID(str(ws)), agent_id=agent_id, app_name=app,
                                action_name=f"{app}_SEND_EMAIL", status=status, error_code=error_code,
                                error_message=error_message, telemetry_source=source, executed_at=moment))
    db.flush()


def applied_change(db, ws, agent_id, moment, board_task_id=1) -> None:
    """A harness_task_ledger row for a change HARNESS applied to the agent at ``moment``."""
    db.execute(text("INSERT INTO harness_task_ledger (workspace_id, board_task_id, state, entry, created_at) "
                    "VALUES (CAST(:ws AS uuid), :task, 'applied', CAST(:entry AS jsonb), :at)"),
               {"ws": str(ws), "task": board_task_id, "entry": f'{{"target_id": {agent_id}}}', "at": moment})


def judge(db, ws, week=WEEK) -> Dict[str, Any]:
    """Run the pipeline for ``week`` on its own harness_runs row; return the closed row."""
    from services import harness_runs
    from services.harness_service import HarnessService

    run_id = harness_runs.open_run(db, UUID(str(ws)), week)
    asyncio.run(HarnessService()._run_recorded(db, UUID(str(ws)), run_id, week))
    row = db.execute(text("SELECT status, report_status, coverage, counts, verdicts, prescriptions, report_md "
                          "FROM harness_runs WHERE id = :id"), {"id": run_id}).mappings().one()
    assert row["status"] == "completed", row["report_status"]
    return dict(row)


def metric(run, agent_id, name="task_success") -> Dict[str, Any]:
    """One agent's verdict entry for one metric, from a closed run row."""
    return run["verdicts"]["agents"][str(agent_id)]["metrics"][name]
