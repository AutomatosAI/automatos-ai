"""F156 — the HARNESS task ledger is in the database.

Which done [HARNESS] board tasks self-management applied, and which wait for
an owner's or admin's /approve, lived in harness/applied_tasks.json on the
workspace volume: a path the generic workspace_write_file tool writes. A
prompted agent could seed task ids there, suppress the "needs /approve" note,
or have a change counted as applied. The ledger is now the harness_task_ledger
table, which only the HARNESS service writes.
"""
from __future__ import annotations

import asyncio
import json
from uuid import UUID

from sqlalchemy import text

from config import config


def test_the_ledger_round_trips_in_the_database(db_session, seed_workspace):
    from services.harness_service import HarnessService

    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    entry = {"task_id": "7", "change_type": "heartbeat_tune", "current_value_before": {"interval_minutes": 30}}
    HarnessService._write_applied_tasks(db_session, ws, [entry], ["9"])
    assert HarnessService._read_applied_tasks(db_session, ws) == {
        "applied_task_ids": ["7"], "needs_approve_task_ids": ["9"]}
    stored = db_session.execute(text("SELECT entry FROM harness_task_ledger WHERE workspace_id = CAST(:ws AS uuid) "
                                     "AND board_task_id = 7"), {"ws": str(ws)}).scalar()
    assert stored["current_value_before"] == {"interval_minutes": 30}

    # A held task approved later is applied; an applied one never goes back to held.
    HarnessService._write_applied_tasks(db_session, ws, [{"task_id": "9"}], ["7"])
    assert HarnessService._read_applied_tasks(db_session, ws) == {
        "applied_task_ids": ["7", "9"], "needs_approve_task_ids": []}
    assert HarnessService._read_applied_tasks(db_session, other) == {
        "applied_task_ids": [], "needs_approve_task_ids": []}


def _done_harness_task(task_id):
    return {
        "id": task_id,
        "title": "[HARNESS] heartbeat_tune for ScribeAgent",
        "description": (f"**Change Type:** heartbeat_tune\n\n**Current:** {json.dumps({'interval_minutes': 30})}\n\n"
                        f"**Proposed:** {json.dumps({'interval_minutes': 90})}\n\n**Rationale:** idle"),
        "tags": ["harness", "risk-2", "rx:rx-heartbeat-1"],
        "status": "done",
    }


class _Executor:
    """The platform actions the weekly tick calls, on the real database."""

    def __init__(self, db, tasks, agents):
        self.db, self._tasks, self._agents, self.calls = db, tasks, agents, []

    async def execute(self, action, params, caller_context=None):
        self.calls.append((action, params))
        if action == "platform_list_tasks":
            return {"data": self._tasks}
        if action == "platform_list_agents":
            return {"data": self._agents}
        return {"success": True}


def test_a_file_a_workspace_tool_can_write_is_not_the_ledger(db_session, seed_workspace, monkeypatch, tmp_path):
    """A task held in a file at the old path is still applied: the file is not
    the ledger, and the change lands in the database's."""
    from services.harness_service import HarnessService

    ws = UUID(seed_workspace())
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    seeded = tmp_path / str(ws) / "harness"
    seeded.mkdir(parents=True)
    (seeded / "applied_tasks.json").write_text(json.dumps({"applied_task_ids": [], "needs_approve_task_ids": ["7"]}))

    executor = _Executor(db_session, [_done_harness_task(7)], [{"id": 42, "name": "ScribeAgent"}])
    asyncio.run(HarnessService()._apply_approved_board_tasks(executor, ws, {}))

    assert ("platform_configure_agent_heartbeat", {"agent_id": 42, "interval_minutes": 90}) in executor.calls
    assert HarnessService._read_applied_tasks(db_session, ws)["applied_task_ids"] == ["7"]
