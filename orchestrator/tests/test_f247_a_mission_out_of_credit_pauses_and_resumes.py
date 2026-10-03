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
