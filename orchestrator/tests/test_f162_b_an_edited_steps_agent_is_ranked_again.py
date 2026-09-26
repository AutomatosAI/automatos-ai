"""F162 (b) (night 5, JEV) — an edited step's "who would run it" is ranked again.

After the owner edited a plan, the approval card kept each step's plan-time
agent pick. Mission 5da0c52b's step 3 "Draft Roastery Page for Priya", edited to
OPS, still said NEWSROOM, and step 4, edited to COUNTINGHOUSE, still said
RESEARCHER. fb8e0ef4's decaf steps said RESEARCHER under GREEN BUYER and
81e8f37a's said TRACKER under CLUB SECRETARY. update_mission_plan changed the
rows and the plan's titles and roles, and never ranked the edited steps again.
On the real schema, through CoordinatorService.update_mission_plan.
"""
from __future__ import annotations

from uuid import UUID

import pytest

from core.models import Agent
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState, TaskState
from services import coordinator_service as cs

ROSTER = ("NEWSROOM", "OPS", "COUNTINGHOUSE", "RESEARCHER", "CLUB SECRETARY", "TRACKER")


def _roster(db, ws):
    agents = {}
    for name in ROSTER:
        agent = Agent(name=name, agent_type="chatbot", description=f"{name.title()}.", status="active",
                      configuration={}, model_config=None, workspace_id=ws, created_by="test",
                      owner_type="workspace", owner_id=str(ws))
        db.add(agent)
        agents[name] = agent
    db.flush()
    return agents


def _planned(db, ws, agents, steps):
    """A plan awaiting approval whose plan-time picks are the agents named in ``steps``."""
    run = OrchestrationRun(workspace_id=ws, goal="Priya's roastery page", state=RunState.AWAITING_APPROVAL.value,
                           created_by="user_test", config={},
                           plan={"tasks": [{"temp_id": temp, "sequence_number": seq, "title": title,
                                            "description": title, "agent_role": "writer"}
                                           for temp, seq, title, _picked in steps]})
    db.add(run)
    db.flush()
    rows = []
    for temp, seq, title, picked in steps:
        row = OrchestrationTask(run_id=run.id, title=title, description=title, sequence_number=seq,
                                agent_role="writer", state=TaskState.PENDING.value, state_type="initial",
                                input_context={"plan_temp_id": temp, "pinned_agent_id": agents[picked].id})
        db.add(row)
        rows.append(row)
    db.flush()
    cs.CoordinatorService()._annotate_match_previews(db, run, list(agents.values()), rows, None)
    for row in rows:                                          # the plan-time picks stand; the pins go
        row.input_context = {k: v for k, v in row.input_context.items() if k != "pinned_agent_id"}
    db.flush()
    return run


def _shown(run):
    return {pt["temp_id"]: pt.get("match_agent") for pt in run.plan["tasks"]}


def test_an_edited_step_shows_the_agent_it_was_edited_to(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    agents = _roster(db_session, ws)
    run = _planned(db_session, ws, agents, [
        ("task_1", 1, "Gather Green Coffee Order Arrival Dates", "RESEARCHER"),
        ("task_3", 3, "Draft Roastery Page for Priya", "NEWSROOM"),
        ("task_4", 4, "Verify Roastery Page Content", "RESEARCHER"),
    ])
    assert _shown(run) == {"task_1": "RESEARCHER", "task_3": "NEWSROOM", "task_4": "RESEARCHER"}

    cs.CoordinatorService().update_mission_plan(db_session, run.id, "user_test", [
        {"temp_id": "task_3", "agent_role": "OPS"}, {"temp_id": "task_4", "agent_role": "COUNTINGHOUSE"}])

    assert _shown(run) == {"task_1": "RESEARCHER", "task_3": "OPS", "task_4": "COUNTINGHOUSE"}   # old: NEWSROOM/RESEARCHER
    rows = db_session.query(OrchestrationTask).filter(OrchestrationTask.run_id == run.id).all()
    assert {r.input_context["plan_temp_id"]: r.input_context["agent_match"]["agent_name"] for r in rows} == {
        "task_1": "RESEARCHER", "task_3": "OPS", "task_4": "COUNTINGHOUSE"}


def test_an_edited_side_by_side_step_changes_only_its_own_pick(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    agents = _roster(db_session, ws)
    run = _planned(db_session, ws, agents, [
        ("task_1", 1, "Split the box for the club", "TRACKER"),
        ("task_2", 1, "Draft Club Member Email for Christmas Box", "NEWSROOM"),
        ("task_3", 1, "Draft the web shop page", "RESEARCHER"),
    ])

    cs.CoordinatorService().update_mission_plan(db_session, run.id, "user_test",
                                                [{"temp_id": "task_2", "agent_role": "CLUB SECRETARY"}])

    assert _shown(run) == {"task_1": "TRACKER", "task_2": "CLUB SECRETARY", "task_3": "RESEARCHER"}
