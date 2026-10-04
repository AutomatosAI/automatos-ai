"""F287 (night 8) — a mission step runs on the agent its card was given to.

Night 8: #0237.1, given to the Inventory Watchdog on the board, ran on the Analyst
once the plan was approved; #0237.2 and #0237.3, given to the Ops Manager, ran on the
Content Creator; #0352.2 showed the Ops Manager and the Content Creator did it; and
#0333.2, given to the Support Agent, was redone by the Content Creator. The board's
Assign wrote only the card's agent, and the dispatcher chose by the step's role.

On the real schema, through MissionDispatcher._dispatch_single (only the embedding
call that blends capability cards into the ranking, and the blueprint check, are
stubbed).
"""
from __future__ import annotations

from uuid import UUID

import pytest

from core.models import Agent
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState, TaskState
from modules.coordination.agent_matcher import AgentMatcher
from modules.coordination.dispatcher import MissionDispatcher
from modules.coordination.step_card_agent import pin_the_cards_agent
from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

ROSTER = {"Analyst": "Works out margins, prices and the numbers.",
          "Inventory Watchdog": "Counts the green coffee and the bags in stock.",
          "Content Creator": "Writes the shop's words.",
          "Support Agent": "Answers the cafés' emails."}


@pytest.fixture
def roastery(db_session, seed_workspace, monkeypatch):
    """Harbourline's agents and a running mission (no step checks), its card made."""
    monkeypatch.setattr(AgentMatcher, "compute_semantic_signals_sync", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr("services.blueprint_validator.check_authority", lambda db, ws, agent_id: (True, []))
    ws = UUID(seed_workspace())
    agents = {}
    for name, description in ROSTER.items():
        agent = Agent(name=name, agent_type="chatbot", description=description, status="active", configuration={},
                      model_config=None, workspace_id=ws, created_by="test", owner_type="workspace",
                      owner_id=str(ws))
        db_session.add(agent)
        agents[name] = agent
    run = OrchestrationRun(workspace_id=ws, goal="Burundi Kayanza for the shop", state=RunState.RUNNING.value,
                           created_by="user_test", config={})
    db_session.add(run)
    db_session.flush()
    create_mission_board_task(db_session, run)
    return db_session, run, agents


def _step(db, run, *, state, role, context, ran_on=None):
    """A step of ``run`` and its card."""
    step = OrchestrationTask(run_id=run.id, title="Calculate margin for Burundi Kayanza 250g bag",
                             description="Do it.", sequence_number=1, agent_role=role, state=state.value,
                             assigned_agent_id=ran_on, input_context=context, max_retries=3)
    db.add(step)
    db.flush()
    return step, create_task_board_task(db, run, step)


def _chosen(agent, at):
    """The dispatcher's choice as it stores it on the step (a preview while the plan waits)."""
    return {"agent_match": {"agent_id": agent.id, "agent_name": agent.name, "decided_at": at}}


def test_a_step_given_to_another_agent_on_the_board_runs_on_it(roastery):
    db, run, agents = roastery
    analyst, watchdog = agents["Analyst"], agents["Inventory Watchdog"]
    step, card = _step(db, run, state=TaskState.PENDING, role="analyst", context=_chosen(analyst, "plan"))
    card.assigned_agent_id, card.status = watchdog.id, "assigned"               # the owner's Assign
    db.flush()

    result = MissionDispatcher._dispatch_single(db, run, step, list(agents.values()))

    assert (result.dispatched, result.agent_id) == (True, watchdog.id)          # night 8: the Analyst
    assert (step.assigned_agent_id, step.agent_role) == (watchdog.id, "Inventory Watchdog")
    assert step.input_context["pinned_agent_id"] == watchdog.id
    assert step.input_context["agent_match"]["reason"].startswith(
        "Explicitly assigned: a person chose agent 'Inventory Watchdog'")


def test_a_step_sent_back_after_its_card_was_given_to_another_agent_is_redone_by_it(roastery):
    """#0333.2: run by the Content Creator, given to the Support Agent, sent back."""
    db, run, agents = roastery
    writer, support = agents["Content Creator"], agents["Support Agent"]
    step, card = _step(db, run, state=TaskState.RETRYING, role="writer", ran_on=writer.id,
                       context={**_chosen(writer, "dispatch"), "previous_output": "A truly unique cup."})
    card.assigned_agent_id, card.status = support.id, "in_progress"            # Assign, then Reject
    db.flush()

    result = MissionDispatcher._dispatch_single(db, run, step, list(agents.values()))

    assert (result.dispatched, result.agent_id) == (True, support.id)           # night 8: the Content Creator
    assert step.input_context["previous_output"] == "A truly unique cup."       # the redo keeps what it corrects


def test_a_card_that_shows_the_agent_its_step_ran_on_pins_nothing(roastery):
    """A step queued again after a stall or a credit pause has no agent; its card
    still shows the one it ran on, and the matcher is free to choose again."""
    db, run, agents = roastery
    writer = agents["Content Creator"]
    step, card = _step(db, run, state=TaskState.QUEUED, role="writer", context=_chosen(writer, "dispatch"))
    card.assigned_agent_id = writer.id
    db.flush()

    assert pin_the_cards_agent(db, run, step, list(agents.values())) is None
    assert step.agent_role == "writer" and "pinned_agent_id" not in step.input_context


def test_a_card_given_to_an_agent_that_is_switched_off_pins_nothing(roastery):
    db, run, agents = roastery
    asleep = Agent(name="Old Watchdog", agent_type="chatbot", description="", status="inactive", configuration={},
                   model_config=None, workspace_id=run.workspace_id, created_by="test", owner_type="workspace",
                   owner_id=str(run.workspace_id))
    db.add(asleep)
    db.flush()
    step, card = _step(db, run, state=TaskState.PENDING, role="analyst", context={})
    card.assigned_agent_id = asleep.id
    db.flush()

    assert pin_the_cards_agent(db, run, step, list(agents.values())) is None    # the roster: active agents
    assert "pinned_agent_id" not in step.input_context
