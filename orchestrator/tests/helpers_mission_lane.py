"""A mission step run by a Claude Code session, on the real schema (F094, F224).

Shared by the tests of a step card's life: the mission's step, its card claimed
by the session lane, and a paired host that has claimed it — a session working.
"""
from __future__ import annotations

from typing import Any, List

from core.models import Agent
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState, TaskState
from services import cli_host_service as svc
from services import cli_ticket_lane as lane
from services import coordinator_service as cs
from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task


def quiet_board(monkeypatch: Any) -> List[Any]:
    """The board's fan-out (approvals, notices, reports, the Canvas) is its own
    suites' business. Returns the Canvas events published meanwhile."""
    import api.board_tasks as bt

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bt, "_board_task_blocked_pending_approval", lambda *a, **k: False)
    monkeypatch.setattr(bt, "_dispatch_task_complete", _noop)
    monkeypatch.setattr(bt, "_dispatch_task_failed", _noop)
    monkeypatch.setattr(bt, "_auto_create_task_report", _noop)
    published: List[Any] = []
    monkeypatch.setattr(svc, "publish_canvas_events", lambda ws, events: published.extend(events))
    monkeypatch.setattr(lane, "_notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(cs, "_narrate_mission", lambda *args, **kwargs: None)
    return published


def session_working(db: Any, ws: Any):
    """A mission step on a Claude Code agent, its card claimed by the lane and by a
    host. Returns ``(run, task, card, host, ticket)``; ``ticket`` is the claim."""
    agent = Agent(name="NEWSROOM", agent_type="chatbot", description="", status="active",
                  configuration={"runtime": "cli", "provider": "claude", "model": "sonnet"}, model_config=None,
                  workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    run = OrchestrationRun(workspace_id=ws, goal="A Christmas box offer for the cafés", state=RunState.RUNNING.value,
                           created_by="user_test", config={})
    db.add(run)
    db.flush()
    task = OrchestrationTask(run_id=run.id, title="Synthesize Christmas Box Offer Details", description="Do it.",
                             sequence_number=2, state=TaskState.RUNNING.value, state_type="active",
                             assigned_agent_id=agent.id, max_retries=3)
    db.add(task)
    db.flush()
    create_mission_board_task(db, run)
    create_task_board_task(db, run, task)
    card = lane.file_cli_ticket(db, workspace_id=ws, agent_id=agent.id, title=task.title, prompt="Work on it.",
                                source_type="mission", source_id=f"mission:{run.id}:{task.id}", tags=["mission"],
                                orchestration_run_id=run.id, orchestration_task_id=task.id)
    host, code, _ = svc.create_pairing_code(db, ws, "laptop")
    host, _token = svc.pair_host(db, code)
    (ticket,) = svc.claim_for_host(db, host, 1)["tasks"]
    assert ticket["task_id"] == card.id and card.status == "in_progress"
    return run, task, card, host, ticket
