"""F287 and F282 (night 8): a mission Auto starts keeps the owner's agents and the owner's check.

- The planner swapped the agents the owner named on #0323, #0324, #0352, #0356, #0394
  and #0400: Auto passed {"Analyst": "Analyst"}, [{"agent_name": …}] or nothing, and
  staffing that names no agent's work pins nobody.
- "Check each step with me before it moves on to the next one" got no setting on #0454
  and #0458, and both ran start to finish unseen.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

PRICE_LIST = ("Start a mission where I check every step myself before it moves on to the next one. It's for the "
              "Christmas wholesale list: the Analyst works out the margin on a 1 kg bag at £21.00 plus VAT (green "
              "£8.10 a kilo), and the Operations Manager drafts a short email to the wholesale cafés offering the "
              "Christmas blend. The Business Analyst checks last December's orders.")


@pytest.fixture
def team(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    for name in ("Analyst", "Shopify Operations Manager", "Shopify Business Analyst", "Content Creator"):
        db_session.execute(text(
            "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
            "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json), 'workspace')"),
            {"n": name, "w": str(ws)})
    db_session.flush()
    return NS(db=db_session, ws=ws)


def test_each_agent_the_owners_words_give_work_to_is_named_with_it(team):
    from modules.tools.discovery.mission_owner_words import staffing_from_words

    staffing = staffing_from_words(team.db, team.ws, [PRICE_LIST])

    assert [entry["agent"] for entry in staffing] == ["Analyst", "Shopify Operations Manager",
                                                      "Shopify Business Analyst"]
    assert staffing[0]["does"] == "works out the margin on a 1 kg bag at £21.00 plus VAT (green £8.10 a kilo)"
    assert staffing[1]["does"].startswith("drafts a short email to the wholesale cafés")
    assert staffing[2]["does"].startswith("checks last December's orders")


def test_an_agent_only_mentioned_is_not_pinned(team):
    from modules.tools.discovery.mission_owner_words import staffing_from_words

    said = ["Start a mission for the Burundi launch. The Content Creator keeps using 'delightful', so watch that."]
    assert staffing_from_words(team.db, team.ws, said) == []


def test_loose_staffing_takes_the_owners_words_for_the_work(team):
    from modules.tools.discovery.mission_owner_words import chosen_staffing

    params = {"goal": "Christmas wholesale list", "staffing": {"Analyst": "Analyst",
                                                               "Shopify Operations Manager": "Operations Manager"}}
    chosen = chosen_staffing(team.db, team.ws, params, ["Yes, go ahead.", PRICE_LIST])

    assert [entry["agent"] for entry in chosen][:2] == ["Analyst", "Shopify Operations Manager"]


def test_steps_that_name_their_agent_pin_it(team):
    from modules.tools.discovery.mission_owner_words import chosen_staffing

    params = {"goal": "Brazil Cerrado launch", "steps": [
        {"agent_name": "Analyst", "objective": "Margin on a 1 kg bag at £34 including VAT."},
        {"agent_id": "Content Creator", "objective": "A 40-word club page line."}]}
    assert chosen_staffing(team.db, team.ws, params, []) == [
        {"agent": "Analyst", "does": "Margin on a 1 kg bag at £34 including VAT."},
        {"agent": "Content Creator", "does": "A 40-word club page line."}]


@pytest.mark.parametrize("said, checks", [
    ("Check each step with me before it moves on to the next one.", True),
    ("Show me each step when it's done and don't move on until I've said it's right.", True),
    ("Start a mission, and wait for my OK on each step before you go on to the next one.", True),
    ("Every step waits for my OK before it counts as done.", True),
    ("Start a mission to get my wholesale price list ready for December. Let it run.", False),
])
def test_the_owners_words_ask_for_a_check_of_each_step(said, checks):
    from modules.tools.discovery.mission_owner_words import asks_for_checks

    assert asks_for_checks([said]) is checks


def test_the_tool_pins_the_owners_agents_and_checks_each_step_when_they_asked(team, monkeypatch):
    """#0352: approval_mode came out right, the agents didn't; #0454: neither did."""
    import modules.tools.discovery.handlers_board_task_review as review
    import modules.tools.discovery.handlers_missions as missions
    import modules.tools.discovery.handlers_watches as watches
    from services import coordinator_service

    made = {}

    async def create_mission(self, db, workspace_id, goal, created_by, config, staffing=None):
        made.update(goal=goal, config=config, staffing=staffing)
        return NS(id=UUID(int=9), state="awaiting_approval", plan={"tasks": []}, goal=goal)

    monkeypatch.setattr(coordinator_service.CoordinatorService, "create_mission", create_mission)
    monkeypatch.setattr(watches, "auto_create_watch", lambda *a, **k: None)
    monkeypatch.setattr(missions, "_recent_chat_context", lambda *a, **k: [])
    monkeypatch.setattr(review, "owner_words", lambda db, ws, chat: [PRICE_LIST])

    out = asyncio.run(missions.create_mission(team.db, team.ws, {
        "goal": "Christmas wholesale list: margin on a 1 kg bag and an email to the cafés",
        "config": {"output_format": "markdown"}, "_origin_chat_id": str(uuid4())}))

    assert out["success"] is True and out["checks_each_step"] is True
    assert "waits in Review for the owner's check" in out["message"]
    assert made["config"]["check_each_step"] is True
    assert [entry["agent"] for entry in made["staffing"]] == ["Analyst", "Shopify Operations Manager",
                                                              "Shopify Business Analyst"]


def test_tags_put_in_the_missions_settings_reach_its_card(team):
    """F262 (part): #0295's and #0333's tags went into config as "tags", and the board's
    own mission call sends them there (#0433): the card was untagged."""
    from core.models.orchestration import OrchestrationRun
    from services.orchestration_board_bridge import create_mission_board_task

    run = OrchestrationRun(workspace_id=team.ws, goal="Kenya Kiambu AA launch", state="awaiting_approval",
                           created_by="user_test", config={"tags": ["sim-night-2026-10-04", "kiambu"]})
    team.db.add(run)
    team.db.flush()

    assert create_mission_board_task(team.db, run).tags == ["mission", "orchestration", "sim-night-2026-10-04", "kiambu"]
