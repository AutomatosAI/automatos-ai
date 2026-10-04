"""F308 (night 9): "check each step with me", asked twice through Auto, made #0033 with no check.

The owner: "Please set up a mission, and check each step with me before moving on: …".
Auto: "… Each step will pause for your approval before proceeding. …", and asked. The
owner: "Yes — that's what I asked for. Go ahead." Auto's call: platform_create_mission
{goal, tags, config: {"auto_approve_steps": false}}. Then: "Yes, approve and run it — and
remember each step stops for me before the next one starts." #0033 was made with
check_each_step false and ran through in a minute.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

from core.models.orchestration_enums import RunState, TaskState
from tests.test_f242_wait_for_me_holds import _held
from tests.test_f245_cancel_stops_a_mission import _mission

ASKED = ("Please set up a mission, and check each step with me before moving on: first, get the Analyst to work "
         "out September's retail takings by where they were sold, from the shop system; then have the Content "
         "Creator write a two-line note for the team about it. Tag it sim-night-2026-10-04.")
AUTOS_PROPOSAL = ("This is a complex task that could benefit from a **Multi-Agent Mission**. Would you like me to "
                  "launch a mission for this?\n\nIf so, I'll set up a mission with the Analyst to calculate "
                  "September's retail takings by sales channel, and then the Content Creator will draft a two-line "
                  "note for the team based on those findings. Each step will pause for your approval before "
                  "proceeding. The mission will be tagged `sim-night-2026-10-04`.")
GO_AHEAD = "Yes — that's what I asked for. Go ahead."
APPROVE_AND_REMEMBER = "Yes, approve and run it — and remember each step stops for me before the next one starts."
NIGHT_9_CALL = {"tags": ["sim-night-2026-10-04"], "config": {"auto_approve_steps": False},
                "goal": ("Calculate September's retail takings by sales channel using the shop system, then have "
                         "the Content Creator write a two-line note for the team about the findings.")}
CHAT = uuid4()


@pytest.fixture
def chat(monkeypatch):
    """The owner's words and Auto's last reply in the chat the turn runs in (its usage scope)."""
    import modules.tools.discovery.handlers_board_task_review as review
    import modules.tools.discovery.owner_turn as turn

    said = NS(owner=[], auto="")
    monkeypatch.setattr(review, "owner_words", lambda db, ws, chat_id: list(said.owner) if chat_id == CHAT else [])
    monkeypatch.setattr(turn, "autos_last_reply", lambda db, ws, t: said.auto if t.chat_id == CHAT else "")
    return said


def _in_the_chat(call):
    from core.llm.usage_context import LANE_CHAT, usage_scope

    with usage_scope(request_type=LANE_CHAT, execution_id=f"chat:{CHAT}"):
        return asyncio.run(call)


def _created(db, ws, monkeypatch, params):
    import modules.tools.discovery.handlers_missions as missions
    import modules.tools.discovery.handlers_watches as watches
    from services import coordinator_service

    made = {}

    async def create_mission(self, db, workspace_id, goal, created_by, config, staffing=None):
        made.update(goal=goal, config=config)
        return NS(id=UUID(int=33), state="awaiting_approval", plan={"tasks": []}, goal=goal)

    monkeypatch.setattr(coordinator_service.CoordinatorService, "create_mission", create_mission)
    monkeypatch.setattr(watches, "auto_create_watch", lambda *a, **k: None)
    monkeypatch.setattr(missions, "_recent_chat_context", lambda *a, **k: [])
    out = _in_the_chat(missions.create_mission(db, ws, dict(params)))
    return out, made


def test_auto_approve_steps_false_is_the_check_of_each_step(db_session, seed_workspace):
    from modules.coordination.owner_checks import checks_each_step, with_step_checks

    assert checks_each_step({"auto_approve_steps": False}) and checks_each_step({"auto_advance": "false"})
    assert not checks_each_step({"auto_approve_steps": True})
    assert with_step_checks({"auto_approve_steps": False, "output_format": "markdown"}) == {
        "output_format": "markdown", "check_each_step": True}
    run, _card, task, step_card = _held(db_session, UUID(seed_workspace()), config={"auto_approve_steps": False})
    assert (task.state, step_card.status, run.state) == (TaskState.VERIFYING.value, "review", RunState.PAUSED.value)


@pytest.mark.parametrize("said, autos_reply, checks", [
    ([GO_AHEAD, ASKED], "", True),                                  # the ask is the message before
    ([GO_AHEAD, "Start a mission on the Burundi launch."], AUTOS_PROPOSAL, True),   # Auto promised it
    ([GO_AHEAD, "Start a mission on the Burundi launch."], "", False),
    (["Start it. Let it run.", ASKED], AUTOS_PROPOSAL, False),
    ([GO_AHEAD], "It runs straight through: each step won't pause for you.", False),
])
def test_a_go_ahead_takes_the_check_from_what_was_said_before(said, autos_reply, checks):
    from modules.tools.discovery.mission_owner_words import asks_for_checks

    assert asks_for_checks(said, autos_reply) is checks


def test_mission_0033_as_night_9_made_it_checks_each_step(db_session, seed_workspace, monkeypatch, chat):
    chat.owner, chat.auto = [GO_AHEAD, ASKED], AUTOS_PROPOSAL

    out, made = _created(db_session, UUID(seed_workspace()), monkeypatch, NIGHT_9_CALL)

    assert made["config"]["check_each_step"] is True                         # night 9: false
    assert "auto_approve_steps" not in made["config"]
    assert out["checks_each_step"] is True and "waits in Review for the owner's check" in out["message"]


def test_a_go_ahead_to_autos_promise_alone_checks_each_step(db_session, seed_workspace, monkeypatch, chat):
    chat.owner = [GO_AHEAD, ("Work out September's retail takings by sales channel from the shop system, then the "
                             "Content Creator writes a two-line note for the team about the findings.")]
    chat.auto = AUTOS_PROPOSAL
    params = {**NIGHT_9_CALL, "config": {}}

    _out, made = _created(db_session, UUID(seed_workspace()), monkeypatch, params)

    assert made["config"]["check_each_step"] is True


def test_approving_with_each_step_stops_for_me_switches_the_check_on(db_session, seed_workspace, monkeypatch, chat):
    from modules.tools.discovery.handlers_missions import approve_mission
    from services import coordinator_service

    ws = UUID(seed_workspace())
    run, _card, _steps = _mission(db_session, ws, state=RunState.AWAITING_APPROVAL)
    chat.owner = [APPROVE_AND_REMEMBER, GO_AHEAD]
    monkeypatch.setattr(coordinator_service.CoordinatorService, "approve_plan",
                        lambda self, db, run_id, actor: NS(id=run_id, state="running"))
    monkeypatch.setattr("services.mission_wait.wait_note_of", lambda db, run_id: "")

    out = _in_the_chat(approve_mission(db_session, ws, {"mission_id": str(run.id), "_created_by": "owner@local"}))

    db_session.refresh(run)
    assert run.config.get("check_each_step") is True                         # night 9: #0033 ran through
    assert out["success"] is True and out["checks_each_step"] is True
    assert "waits in Review for the owner's check" in out["message"]


def test_an_approval_says_when_the_steps_run_unchecked(db_session, seed_workspace, monkeypatch, chat):
    from modules.tools.discovery.handlers_missions import approve_mission
    from services import coordinator_service

    ws = UUID(seed_workspace())
    run, _card, _steps = _mission(db_session, ws, state=RunState.AWAITING_APPROVAL)
    chat.owner = ["Approve it and start."]
    monkeypatch.setattr(coordinator_service.CoordinatorService, "approve_plan",
                        lambda self, db, run_id, actor: NS(id=run_id, state="running"))
    monkeypatch.setattr("services.mission_wait.wait_note_of", lambda db, run_id: "")

    out = _in_the_chat(approve_mission(db_session, ws, {"mission_id": str(run.id), "_created_by": "owner@local"}))

    assert out["checks_each_step"] is False and "run without waiting for the owner's check" in out["message"]
