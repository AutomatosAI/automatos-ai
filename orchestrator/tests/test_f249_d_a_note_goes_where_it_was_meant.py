"""F249 (night 8): each of the owner's notes reaches the work it was meant for, once.

- A playbook run's card belongs to its first step's agent, so playbook 102's notes
  ("Signed Gerard, not the crew", "£19.50 a kilo plus VAT") were the Analyst's
  "corrections" on its margin cards. They are the playbook's only now.
- The Analyst's "just the table" broke on six mission steps (#0352.1, #0374.1,
  #0383.1, #0394.1, …): a step's text is the planner's, and the lessons said "unless
  this brief says otherwise". A mission step is told the owner's notes win over it.
- A playbook card's or a mission step's redo carried the agent's lessons in its redo
  words, and its steps carried them again.
- Auto's move of a card to In progress launched the bare brief, with no lessons (#0346).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

JUST_THE_TABLE = "Just the table: nothing before it, no working under it and no bold rows."
CREW = "Signed Gerard, not Gerard & the Harbourline Crew. Every run of this playbook like this."
MARGIN = "Margin per bag: Colombia Huila, 1 kg wholesale."


@pytest.fixture
def shop(db_session, seed_workspace):
    """The Analyst: a margin card sent back, and playbook 102's run card with its note."""
    from core.models import Agent
    from core.models.core import BoardTask

    db, ws = db_session, UUID(seed_workspace())
    analyst = Agent(name="Analyst", agent_type="custom", description="", status="active", configuration={},
                    workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(analyst)
    db.flush()

    def card(**fields):
        made = BoardTask(workspace_id=ws, status="done", assigned_agent_id=analyst.id, **fields)
        db.add(made)
        db.flush()
        return made

    card(title="Margin per bag: Peru", planning_data={"owner_corrections": [
        {"note": JUST_THE_TABLE, "by": "user:o", "at": "2026-10-04T04:57:00+00:00"}]})
    run_card = card(title="Recipe: New Cafe Onboarding", source_type="recipe", source_id="exec-0441",
                    created_by_type="recipe", review_feedback=CREW,
                    planning_data={"recipe_id": 102, "execution_id": "exec-0441", "owner_corrections": [
                        {"note": CREW, "by": "user:o", "at": "2026-10-04T07:16:00+00:00"}]})
    return NS(db=db, ws=ws, analyst=analyst, card=card, run_card=run_card)


def test_a_playbooks_note_is_never_the_agents_lesson(shop):
    from services.ticket_redo import agent_lessons, playbook_lessons

    assert agent_lessons(shop.db, shop.ws, shop.analyst.id) == [JUST_THE_TABLE]       # night 8: CREW too
    assert playbook_lessons(shop.db, shop.ws, 102) == [CREW]


def test_a_playbook_steps_session_ticket_is_still_the_agents_card(shop):
    from services.ticket_redo import agent_lessons

    shop.card(title="Recipe step 2", source_type="recipe", source_id="recipe:exec-0441:2", planning_data={
        "owner_corrections": [{"note": "Put the kilos on the first line.", "by": "user:o",
                               "at": "2026-10-04T07:20:00+00:00"}]})

    assert agent_lessons(shop.db, shop.ws, shop.analyst.id) == ["Put the kilos on the first line.", JUST_THE_TABLE]


def test_a_playbook_cards_redo_words_leave_the_lessons_to_its_steps(shop):
    from services.ticket_redo import STANDING_HEADING, redo_block

    words = redo_block(shop.run_card)

    assert CREW in words and STANDING_HEADING not in words          # night 8: the lessons came twice
    step = shop.card(title="Christmas margin", source_type="orchestration_task", review_feedback=JUST_THE_TABLE)
    assert STANDING_HEADING not in redo_block(step)
    plain = shop.card(title="Margin per bag: Kenya", review_feedback="Pounds and pence, please.")
    assert STANDING_HEADING in redo_block(plain)                     # a plain card's redo still carries them


def test_a_mission_step_is_told_the_owners_notes_win_over_the_planners_text(shop):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from modules.coordination.dispatcher import MissionDispatcher
    from services.ticket_redo import MISSION_STEP_ASK, STANDING_HEADING

    run = OrchestrationRun(workspace_id=shop.ws, goal="Christmas wholesale margin and email", state="running",
                           created_by="user_test", config={})
    shop.db.add(run)
    shop.db.flush()
    step = OrchestrationTask(run_id=run.id, title="Calculate the margin",
                             description="Show the calculation step by step, then the table.", sequence_number=1,
                             state="assigned", state_type="active", assigned_agent_id=shop.analyst.id)
    shop.db.add(step)
    shop.db.flush()

    prompt = MissionDispatcher.build_task_prompt(step)

    block = prompt.split(f"{STANDING_HEADING}\n", 1)[1].splitlines()
    assert block[0].startswith(MISSION_STEP_ASK) and block[1] == f"- {JUST_THE_TABLE}"
    assert "follow the note" in MISSION_STEP_ASK and CREW not in prompt


def test_a_card_auto_starts_carries_its_agents_lessons_before_where_its_answer_goes(shop):
    from services.draft_guides import guides_for_draft
    from services.step_lessons import ON_THE_CARD
    from services.ticket_redo import STANDING_HEADING

    prompt = asyncio.run(guides_for_draft(shop.db, str(shop.ws), shop.analyst.id, f"{MARGIN}\n\n{ON_THE_CARD}"))

    assert prompt.startswith(MARGIN) and prompt.endswith(ON_THE_CARD)               # night 8: the bare brief
    assert prompt.index(STANDING_HEADING) < prompt.index(ON_THE_CARD) and f"- {JUST_THE_TABLE}" in prompt
    assert prompt.count(JUST_THE_TABLE) == 1 and CREW not in prompt


def test_a_card_the_board_claimed_is_not_given_them_twice(shop):
    from services.draft_guides import guides_for_draft
    from services.ticket_redo import STANDING_HEADING, lessons_block

    claimed = f"{MARGIN}\n\n{lessons_block(shop.db, shop.ws, shop.analyst.id)}"

    assert asyncio.run(guides_for_draft(shop.db, str(shop.ws), shop.analyst.id, claimed)) == claimed
    assert claimed.count(STANDING_HEADING) == 1
