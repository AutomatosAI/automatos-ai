"""F322, F323, F327 and F320 (night 9b): what every agent run is told about a wrong
brief, the team's other figures, looking things up, and the answer itself.

- F322: the Watchdog bent its sum to fit the owner's wrong 30 kg of Guji (#1978); the
  Content Creator flipped a right draft on "I said two bags of Guji, is that right?"
  (#1982); the newsletter helper invented an origin and tasting notes (#1986); Auto:
  "You're right, Lantern Kitchen is a great customer!" (e95c1b6b, 28th of 31).
- F323: the Watchdog said "5.6 kg buffer" and "runs out ~13 October" in the same hour
  (#1987/#1990); the Business Analyst read "this summer" as 2024 (#1984).
- F327: agents and Auto asked the owner for an account id, a lot code and parameter
  names; Auto offered a web search for the owner's own stock; drafts were signed
  "[Your name]" or began "the 'Harbourline voice' skill is not available".
- F320: answers opened "Perfect! Now I have all the information I need…".

A plain card, a mission step and a playbook step all get the rules; so do both of
Auto's chat prompts for its half.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

WS = "febae41b-374b-4580-a5ef-f698bdd382e4"
HARBOUR_LOG = "Write the Harbour Log intro: this month's box is two bags of Guji."

CHECK_THE_BRIEF = "Check each fact in the brief"
SAY_IT_FIRST = "say so at the top of your answer: what the brief says, what the source says"
NO_BENT_SUMS = "never bend a sum to fit it"
A_QUESTION_IS_A_QUESTION = "is a question: check it and answer it"
NOTHING_MADE_UP = "Never make up a fact about the business or its products"
THE_TEAMS_FIGURES = "search_knowledge with scope \"past_work\""
NEVER_ASK_FOR_IDS = "never ask the owner for an id, a code, a parameter name"
NOT_THE_WEB = "never search the web for it"
THIS_SUMMER = "against today's date in your instructions"
NO_NARRATION = "No narration anywhere in it"
READY_TO_SEND = "A draft is ready to send: signed and filled in"


def _told_the_night_9b_rules(prompt: str) -> None:
    for rule in (CHECK_THE_BRIEF, SAY_IT_FIRST, NO_BENT_SUMS, A_QUESTION_IS_A_QUESTION, NOTHING_MADE_UP,
                 THE_TEAMS_FIGURES, NEVER_ASK_FOR_IDS, NOT_THE_WEB, THIS_SUMMER, NO_NARRATION, READY_TO_SEND):
        assert rule in prompt, rule


def test_a_plain_cards_run_is_told_the_rules():
    from services.step_lessons import a_cards_answer_goes_on_the_card

    launched = {}
    a_cards_answer_goes_on_the_card(lambda **kwargs: launched.update(kwargs))(
        task_id=1982, agent_id=344, workspace_id=WS, prompt=HARBOUR_LOG, review_mode="human")

    assert launched["prompt"].startswith(HARBOUR_LOG)
    _told_the_night_9b_rules(launched["prompt"])


@pytest.fixture
def roastery(db_session, seed_workspace):
    from core.models import Agent

    ws = UUID(seed_workspace())
    made = Agent(name="Content Creator", agent_type="custom", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db_session.add(made)
    db_session.flush()
    return NS(db=db_session, ws=ws, writer=made)


def test_a_mission_step_is_told_the_rules(roastery):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from modules.coordination.dispatcher import MissionDispatcher

    run = OrchestrationRun(workspace_id=roastery.ws, goal="October club newsletter", state="running",
                           created_by="user_test", config={})
    roastery.db.add(run)
    roastery.db.flush()
    step = OrchestrationTask(run_id=run.id, title="Harbour Log intro", description=HARBOUR_LOG, sequence_number=1,
                             state="assigned", state_type="active", assigned_agent_id=roastery.writer.id)
    roastery.db.add(step)
    roastery.db.flush()

    _told_the_night_9b_rules(MissionDispatcher.build_task_prompt(step))


def test_a_playbook_step_is_told_the_rules(roastery):
    from services.step_lessons import a_playbook_step_carries_its_lessons

    sent = {}

    async def execute(**kwargs):
        sent.update(kwargs)
        return {"status": "success"}

    asyncio.run(a_playbook_step_carries_its_lessons(execute)(
        db=roastery.db, agent=roastery.writer, clean_prompt=HARBOUR_LOG, workspace_id=roastery.ws))

    assert sent["clean_prompt"].startswith(HARBOUR_LOG)
    _told_the_night_9b_rules(sent["clean_prompt"])


AUTO_CHECKS = "I check it in their documents or their system before I agree"
AUTO_LOOKS_IT_UP = "Account ids, codes, column or parameter names, a date range they already gave: I find them"


def test_autos_full_chat_prompt_checks_the_owners_facts():
    from consumers.chatbot.personality import AutomatosPersonality

    avoid = AutomatosPersonality.get_anti_patterns()
    assert AUTO_CHECKS in avoid and "\"Is that right?\" is a question" in avoid and AUTO_LOOKS_IT_UP in avoid


def test_autos_short_chat_prompt_checks_the_owners_facts_but_a_visitors_does_not():
    from consumers.chatbot.atom_prompt import atom_system_prompt

    auto = NS(name="Auto", description="", persona=None)
    owners = atom_system_prompt(auto, identity="", memory_block="", facts="## Automatos itself\nLocal edition.")
    visitors = atom_system_prompt(auto, identity="", memory_block="", facts="")

    assert AUTO_CHECKS in owners and AUTO_LOOKS_IT_UP in owners
    assert AUTO_CHECKS not in visitors
