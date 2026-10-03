"""F262 (night 7b): Auto creates a mission as the owner asks for it.

"Start a mission … stop after every step for my approval" failed three times on the
tool's own arguments, and the owner made #0188 on the Missions page instead:
1. {name, wait_for_me, tags, staffing: {coordinator, agents}, goal}: refused for name,
   wait_for_me and tags;
2. {"params": "<JSON text>"} with label, objective and steps: refused;
3. no call at all ("Let me try creating the mission one more time").
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS
from uuid import UUID

GOAL = "Get the Christmas gift subscription ready: the coffee for 140 boxes, the shop words, the club email."
FIRST_CALL = {"name": "sim-night-2026-10-03", "wait_for_me": True, "tags": ["Christmas", "gift-subscription"],
              "staffing": {"coordinator": "Auto", "agents": ["Analyst", "Content Creator"]}, "goal": GOAL,
              "config": {"category": "product_launch", "output_format": "markdown"}}
SECOND_CALL = {"params": json.dumps({"label": "sim-night-2026-10-03", "objective": GOAL, "steps": [
    {"name": "Calculate coffee quantities", "objective": "Roasted and green kilos for 140 boxes.",
     "pause_before_start": True},
    {"name": "Draft shop page words", "objective": "Three months, £42, orders open 2 November."},
]})}


def _action():
    from modules.tools.discovery import get_action_registry

    return get_action_registry().get("platform_create_mission")


def test_the_first_call_is_refused_only_for_its_name_and_says_to_send_it_again():
    from modules.tools.execution.unified_executor import REFUSED_CALL_IS_YOURS, undeclared_params_refusal

    refused = undeclared_params_refusal("platform_create_mission", _action(), FIRST_CALL, "t")
    assert "does not take ['name']" in refused                          # wait_for_me and tags are taken now
    assert "titled from its goal" in refused and "tags" in refused
    assert REFUSED_CALL_IS_YOURS in refused                              # night 7b: Auto asked instead of resending


def test_the_second_calls_json_text_is_unwrapped_and_its_objective_is_the_goal():
    from modules.tools.execution.unified_executor import _fill_required_from_aliases, undeclared_params_refusal

    filled = _fill_required_from_aliases(SECOND_CALL, ["goal"])
    assert filled["goal"] == GOAL and len(filled["steps"]) == 2
    refused = undeclared_params_refusal("platform_create_mission", _action(), filled, "t")
    assert "does not take ['label']" in refused                          # steps are taken now


def test_wait_for_me_steps_and_tags_are_the_missions_own_settings():
    from modules.tools.discovery.mission_asks import as_the_mission_settings

    asked, note = as_the_mission_settings({**FIRST_CALL, "steps": [
        "Work out the coffee", {"name": "Shop words", "objective": "Plain and warm"}]})
    del asked["name"]

    assert asked["config"]["check_each_step"] is True and asked["config"]["category"] == "product_launch"
    assert asked["config"]["card_tags"] == ["Christmas", "gift-subscription"]
    assert asked["goal"].startswith(GOAL)
    assert "1. Work out the coffee" in asked["goal"] and "2. Shop words: Plain and warm" in asked["goal"]
    assert "staffing" not in asked and "pinned nobody" in note           # names without their work pin no one
    assert not {"wait_for_me", "steps", "tags"} & set(asked)


def test_a_step_that_says_to_pause_first_makes_the_mission_wait():
    from modules.tools.discovery.mission_asks import as_the_mission_settings

    asked, _ = as_the_mission_settings({"goal": GOAL, "steps": json.loads(SECOND_CALL["params"])["steps"]})
    assert asked["config"]["check_each_step"] is True


def test_staffing_that_names_each_agents_work_is_kept():
    from modules.tools.discovery.mission_asks import as_the_mission_settings

    staffing = [{"agent": "Analyst", "does": "works out the coffee"}]
    asked, note = as_the_mission_settings({"goal": GOAL, "staffing": staffing})
    assert asked["staffing"] == staffing and note is None and "config" not in asked


def test_the_tool_passes_the_owners_settings_to_the_coordinator(db_session, seed_workspace, monkeypatch):
    import modules.tools.discovery.handlers_missions as missions
    import modules.tools.discovery.handlers_watches as watches
    from services import coordinator_service

    made = {}

    async def create_mission(self, db, workspace_id, goal, created_by, config, staffing=None):
        made.update(goal=goal, config=config, staffing=staffing)
        return NS(id=UUID(int=7), state="awaiting_approval", plan={"tasks": []}, goal=goal)

    monkeypatch.setattr(coordinator_service.CoordinatorService, "create_mission", create_mission)
    monkeypatch.setattr(watches, "auto_create_watch", lambda *a, **k: None)
    monkeypatch.setattr(missions, "_recent_chat_context", lambda *a, **k: [])
    ws = UUID(seed_workspace())
    call = {k: v for k, v in FIRST_CALL.items() if k != "name"}

    out = asyncio.run(missions.create_mission(db_session, ws, call))

    assert out["success"] is True and "pinned nobody" in out["staffing_note"]
    assert made["config"]["check_each_step"] is True and made["config"]["card_tags"] == call["tags"]
    assert made["staffing"] is None


def test_the_missions_card_carries_the_owners_tags(db_session, seed_workspace):
    from core.models.orchestration import OrchestrationRun
    from services.orchestration_board_bridge import create_mission_board_task

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal=GOAL, state="awaiting_approval", created_by="user_test",
                           config={"card_tags": ["sim-night-2026-10-03"]})
    db_session.add(run)
    db_session.flush()

    card = create_mission_board_task(db_session, run)
    assert card.tags == ["mission", "orchestration", "sim-night-2026-10-03"]
