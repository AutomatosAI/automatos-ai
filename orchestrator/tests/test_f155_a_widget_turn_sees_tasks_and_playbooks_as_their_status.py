"""F155: a widget turn with tasks:read or playbooks:read sees status, not contents.

Both scopes can be granted to public keys. Like platform_get_mission before
30c6f08a7, their tools returned what the owner sees:
- platform_get_task: a task's description, raw prompt, result (2,000
  characters) and errors. Task ids are sequential, so a visitor could walk
  the whole board.
- platform_list_tasks: every task's description and errors.
- platform_wait_for_task: the session's tools and files.
- platform_board_summary / platform_board_snapshot: who is busiest, what
  failed and why, and the owner's scheduled routines.
- platform_get_playbook: the steps' prompts, agents and outputs.
- platform_get_playbook_execution: step outputs, errors and inputs, and
  anyone's recent runs.
On a widget turn each now returns a visitor's view: what it is and where it
stands.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

from core.security.surface import WIDGET, turn_surface

PRIVATE = "£42k"
VISITOR_TASK = {"id", "title", "status", "created_at", "started_at", "completed_at"}
VISITOR_PLAYBOOK = {"id", "name", "description", "step_count", "created_at"}
VISITOR_RUN = {"execution_id", "playbook_id", "status", "current_step", "started_at", "completed_at"}


def _widget_turn(scope):
    return turn_surface(WIDGET, ("chat", scope), None)


def _task(db, ws, **fields):
    from core.models.core import BoardTask

    task = BoardTask(workspace_id=ws, title="Draft the offer letter", description=f"Offer {PRIVATE}",
                     raw_prompt=f"Write it at {PRIVATE}", priority="high", tags=["hr"], **fields)
    db.add(task)
    db.flush()
    return task


def test_a_widget_turn_sees_a_board_task_as_its_status(db_session, seed_workspace):
    from modules.tools.discovery import handlers_board_tasks as board

    ws = UUID(seed_workspace())
    task = _task(db_session, ws, status="done", result=f"Dear chef, {PRIVATE}", error_message=f"retried at {PRIVATE}")
    with _widget_turn("tasks:read"):
        one = asyncio.run(board.get_board_task(db_session, ws, {"task_id": task.id}))
        listed = asyncio.run(board.list_board_tasks(db_session, ws, {}))
        waited = asyncio.run(board.wait_for_board_task(db_session, ws, {"task_id": task.id}))
    assert set(one["task"]) == VISITOR_TASK and "frontend_data" not in one
    assert [set(t) for t in listed["tasks"]] == [VISITOR_TASK]
    assert set(waited["task"]) == VISITOR_TASK == set(waited["frontend_data"]["task_card"])
    assert PRIVATE not in str((one, listed, waited))

    owner = asyncio.run(board.get_board_task(db_session, ws, {"task_id": task.id}))
    assert owner["task"]["raw_prompt"] == f"Write it at {PRIVATE}"


def test_a_widget_turn_sees_the_board_as_counts_and_titles(db_session, seed_workspace):
    from modules.tools.discovery import handlers_analytics as analytics

    ws = UUID(seed_workspace())
    _task(db_session, ws, status="in_progress")
    _task(db_session, ws, status="failed", error_message=f"stopped at {PRIVATE}")
    with _widget_turn("tasks:read"):
        summary = asyncio.run(analytics.board_summary(db_session, ws, {}))
        snapshot = asyncio.run(analytics.board_snapshot(db_session, ws, {}))
    assert set(summary) == {"success", "total_tasks", "by_status", "by_priority"}
    assert summary["by_status"] == {"in_progress": 1, "failed": 1}
    assert "scheduled" not in snapshot and "busiest_agents" not in snapshot["counts"]
    assert all(set(t) == {"id", "title", "status"} for t in snapshot["open_tasks"])
    assert PRIVATE not in str((summary, snapshot))

    owner = asyncio.run(analytics.board_summary(db_session, ws, {}))
    assert owner["failed_tasks"][0]["error"] == f"stopped at {PRIVATE}"


def test_a_widget_turn_sees_a_playbook_and_its_runs_as_their_status(db_session, seed_workspace):
    from core.models.core import RecipeExecution, WorkflowTemplate
    from modules.tools.discovery import handlers_playbooks as playbooks

    ws = UUID(seed_workspace())
    playbook = WorkflowTemplate(template_id="f155-offer-letters", name="Offer letters", description="Drafts offer letters",
                                workspace_id=ws, template_definition={"steps": []}, tags=["hr"], created_by="f155",
                                steps=[{"order": 1, "agent_id": 7, "prompt_template": f"Offer {PRIVATE}"}])
    db_session.add(playbook)
    db_session.flush()
    db_session.add(RecipeExecution(execution_id="exec-f155-offer", recipe_id=playbook.id, workspace_id=ws, status="failed",
                                   input_data={"salary": PRIVATE}, step_results=[{"status": "done", "output": f"Dear chef, {PRIVATE}"}],
                                   error_message=f"stopped at {PRIVATE}", current_step=1, attempt_count=1, triggered_by="f155",
                                   started_at=datetime.now(timezone.utc)))
    db_session.flush()
    with _widget_turn("playbooks:read"):
        listed = asyncio.run(playbooks.list_playbooks(db_session, ws, {}))
        one = asyncio.run(playbooks.get_playbook(db_session, ws, {"playbook_id": playbook.id}))
        run = asyncio.run(playbooks.get_playbook_execution(db_session, ws, {"execution_id": "exec-f155-offer"}))
        recent = asyncio.run(playbooks.get_playbook_execution(db_session, ws, {"playbook_id": playbook.id}))
    assert [set(p) for p in listed["playbooks"]] == [VISITOR_PLAYBOOK] and set(one["playbook"]) == VISITOR_PLAYBOOK
    assert (one["playbook"]["step_count"], set(run["execution"])) == (1, VISITOR_RUN)
    assert [set(e) for e in recent["executions"]] == [VISITOR_RUN]
    assert PRIVATE not in str((listed, one, run, recent))

    owner = asyncio.run(playbooks.get_playbook(db_session, ws, {"playbook_id": playbook.id}))
    assert owner["playbook"]["steps"][0]["prompt_preview"] == f"Offer {PRIVATE}"


def test_a_widget_turn_cannot_tell_a_name_no_agent_has_from_an_agent_without_tickets(db_session, seed_workspace):
    """Filtering by assigned agent answered "No agent named 'X' found": an
    agent-name oracle for a key that holds tasks:read but not agents:read."""
    from sqlalchemy import text

    from modules.tools.discovery import handlers_board_tasks as board

    ws = UUID(seed_workspace())
    db_session.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
                            "VALUES ('Payroll Clerk', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json))"),
                       {"w": str(ws)})
    with _widget_turn("tasks:read"):
        idle = asyncio.run(board.list_board_tasks(db_session, ws, {"assigned_agent_name": "Payroll Clerk"}))
        nobody = asyncio.run(board.list_board_tasks(db_session, ws, {"assigned_agent_name": "Nobody Here"}))
    assert idle == nobody == {"success": True, "tasks": [], "total": 0, "total_matching": 0, "limit": 20}

    owner = asyncio.run(board.list_board_tasks(db_session, ws, {"assigned_agent_name": "Nobody Here"}))
    assert owner["note"] == "No agent named 'Nobody Here' found"
