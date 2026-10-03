"""F247 (night 7): a mission whose AI credit runs out pauses, and Resume continues it.

Mission #0176: the credit ran out mid-mission. Each retry failed the same way, the
ledger read it as a stall, re-planned twice by itself (filing unrun steps as Done),
then halted ("Joiner halt"). Resume and Replan were both refused, and four cards
could run nowhere. A step stopped by the credit is now queued again with its attempt
unspent, and the mission pauses with the reason on its card. Resume carries on.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState, TaskState

OPENROUTER_402 = ("Error code: 402 - {'error': {'message': 'This request requires more credits, or fewer max_tokens. "
                  "You requested up to 8000 tokens, but can only afford 1771.', 'code': 402}}")


@pytest.fixture
def mission(db_session, seed_workspace):
    """#0176's shape: a running mission, its card, and two steps out with agents."""
    from core.models import Agent
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task

    ws = UUID(seed_workspace())
    agent = Agent(name="Shopify Business Analyst", agent_type="chatbot", description="", status="active",
                  configuration={}, model_config=None, workspace_id=ws, created_by="test", owner_type="workspace",
                  owner_id=str(ws))
    db_session.add(agent)
    db_session.flush()
    run = OrchestrationRun(workspace_id=ws, goal="Weekly wholesale invoices", state=RunState.RUNNING.value,
                           created_by="user_test", config={})
    db_session.add(run)
    db_session.flush()
    steps = [OrchestrationTask(run_id=run.id, title=title, description="Do it.", sequence_number=n,
                               state=TaskState.RUNNING.value, state_type="active", assigned_agent_id=agent.id,
                               max_retries=3, attempt_number=0)
             for n, title in ((1, "Total the unpaid invoices"), (2, "Draft the reminders"))]
    db_session.add_all(steps)
    db_session.flush()
    card = create_mission_board_task(db_session, run)
    return NS(db=db_session, run=run, steps=steps, card=card)


def _record(mission, step, error):
    from modules.coordination.dispatcher import MissionDispatcher

    MissionDispatcher.record_task_completion(mission.db, step, {"status": "error", "error": error})
    mission.db.flush()


def test_a_step_stopped_by_the_credit_pauses_its_mission_with_the_attempt_unspent(mission):
    from core.llm.credit import OUT_OF_CREDIT_TEXT
    from modules.coordination.credit_pause import PAUSED_TEXT

    first, second = mission.steps
    _record(mission, first, OPENROUTER_402)

    assert (first.state, first.attempt_number, first.assigned_agent_id) == (TaskState.QUEUED.value, 0, None)
    assert first.failure_detail == OUT_OF_CREDIT_TEXT
    assert (mission.run.state, mission.run.stop_reason, mission.run.stop_detail) == (
        RunState.PAUSED.value, "out_of_credit", PAUSED_TEXT)
    assert (mission.card.status, mission.card.blocked_reason) == ("blocked", PAUSED_TEXT)

    _record(mission, second, OUT_OF_CREDIT_TEXT)        # the same outage stops the next step: still one pause
    assert second.state == TaskState.QUEUED.value and mission.run.state == RunState.PAUSED.value


def test_resume_carries_on_once_the_credit_is_back(mission):
    from services.coordinator_service import CoordinatorService

    _record(mission, mission.steps[0], OPENROUTER_402)
    CoordinatorService().resume_mission(mission.db, mission.run.id, "user_test")

    assert (mission.run.state, mission.run.stop_reason, mission.run.stop_detail) == (RunState.RUNNING.value, None, None)
    assert mission.steps[0].state == TaskState.QUEUED.value   # the dispatcher picks it up on the next tick


def test_any_other_failure_is_a_spent_attempt_as_before(mission):
    _record(mission, mission.steps[0], "the agent's tool timed out")

    assert (mission.steps[0].state, mission.steps[0].attempt_number) == (TaskState.QUEUED.value, 1)
    assert mission.run.state == RunState.RUNNING.value and mission.run.stop_reason is None


# ── a mission that failed is retried, from its page or by Auto ──────────────

HALT = "Joiner halt: no forward progress across 3 ledger checks"


def _failed(mission):
    """#0176 at 07:45: a step failed for credit, one never ran, both re-plans spent, halted."""
    from datetime import datetime, timezone

    from core.models.orchestration_enums import ActorType
    from services.orchestration_board_bridge import create_task_board_task
    from services.orchestration_state import transition_run

    failed, waiting = mission.steps
    failed.state, failed.state_type, failed.attempt_number = TaskState.FAILED.value, "terminal", 3
    failed.failure_detail, failed.completed_at = "Your AI credit ran out", datetime.now(timezone.utc)
    waiting.state, waiting.state_type, waiting.assigned_agent_id = TaskState.PENDING.value, "initial", None
    card = create_task_board_task(mission.db, mission.run, failed)
    card.status, card.error_message = "blocked", "Your AI credit ran out"
    mission.run.replan_count, mission.run.config = 2, {"progress_ledger": {"stall_streak": 3}}
    transition_run(db=mission.db, run=mission.run, new_state=RunState.FAILED, actor_type=ActorType.COORDINATOR,
                   actor_id="joiner", reason=HALT, stop_reason="stalled", stop_detail=HALT)
    mission.db.flush()
    return card


def _ran_again(mission, card):
    from services.orchestration_deps import DependencyResolver

    failed, waiting = mission.steps
    run = mission.run
    assert (run.state, run.stop_reason, run.completed_at, run.replan_count) == (RunState.RUNNING.value, None, None, 2)
    assert "progress_ledger" not in run.config                     # old churn is not a new stall
    assert (failed.state, failed.attempt_number, failed.failure_detail, failed.assigned_agent_id) == (
        TaskState.PENDING.value, 0, None, None)
    assert waiting.state == TaskState.PENDING.value
    assert (card.status, card.error_message) == ("inbox", None)
    assert failed in DependencyResolver.get_ready_tasks(mission.db, run.id)   # the next tick dispatches it


def test_resume_retries_a_failed_mission_with_its_re_plans_spent(mission):
    from services.coordinator_service import CoordinatorService

    card = _failed(mission)
    CoordinatorService().resume_mission(mission.db, mission.run.id, "user_test")
    _ran_again(mission, card)


def test_the_mission_pages_resume_retries_it(mission):
    import asyncio

    from api.missions import resume_mission

    class _Request:
        async def json(self):
            return {}

    card = _failed(mission)
    ctx = NS(workspace_id=mission.run.workspace_id, user=NS(id="user_test"))
    reply = asyncio.run(resume_mission(mission_id=mission.run.id, request=_Request(), ctx=ctx, db=mission.db))

    assert reply["state"] == RunState.RUNNING.value
    _ran_again(mission, card)


def test_auto_retries_it_with_platform_resume_mission(mission):
    import asyncio

    from modules.tools.discovery.handlers_missions import resume_mission

    card = _failed(mission)
    reply = asyncio.run(resume_mission(mission.db, mission.run.workspace_id, {"mission_id": str(mission.run.id)}))

    assert reply["success"] is True
    _ran_again(mission, card)


def test_auto_is_told_resume_retries_a_failed_mission():
    from modules.tools.discovery.action_registry import get_action_registry

    resume = get_action_registry().get("platform_resume_mission")
    assert "or retry a failed one: its failed steps run again" in resume.description
    assert "retry the failed mission" in resume.examples
