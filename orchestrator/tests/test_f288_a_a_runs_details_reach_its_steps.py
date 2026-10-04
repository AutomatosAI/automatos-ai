"""F288 (night 8, part): the details a playbook run carries reach its steps.

Auto ran New Cafe Onboarding (playbook 102) for Larder & Loaf with cafe_name,
owner_name, owner_email, first_order and delivery_day. No step's prompt names them
(step 1 asks the owner for "cafe_name, contact_person, contact_email,
usual_harbour_blend_kg…", step 2 fills {{cafe_details.…}} from step 1's answer), so
none reached a step: the run asked the owner again and wrote the template's Thursday
and "the crew" (#0366; #0440 asked again as #1411).
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

ASK_FOR_DETAILS = ("Please provide the following details for the new cafe: cafe_name, contact_person, "
                   "contact_email, usual_harbour_blend_kg (e.g., 6), usual_decaf_kg (e.g., 2).")
WELCOME = ("Draft a welcome email for the new wholesale cafe using the provided details. Subject: Welcome to "
           "Harbourline Coffee - {{cafe_details.cafe_name}}  Hi {{cafe_details.contact_person}}, deliveries on "
           "Thursdays. Cheers, Gerard & the Harbourline Crew")
FROM_AUTO = {"cafe_name": "Larder & Loaf", "owner_name": "Ffion Hughes", "owner_email": "ffion@larderandloaf.example",
             "first_order": "4 kg Harbour Blend", "delivery_day": "Tuesdays", "include_decaf": False}


@pytest.fixture
def onboarding(db_session, seed_workspace):
    """Playbook 102's two steps, both the Analyst's, and the runs that start it."""
    from core.models import Agent
    from core.models.core import RecipeExecution, WorkflowTemplate

    db, ws = db_session, UUID(seed_workspace())

    def agent(name, configuration=None):
        made = Agent(name=name, agent_type="custom", description="", status="active",
                     configuration=configuration or {}, workspace_id=ws, created_by="test", owner_type="workspace",
                     owner_id=str(ws))
        db.add(made)
        db.flush()
        return made

    analyst = agent("Analyst")
    playbook = WorkflowTemplate(template_id=f"f288-{uuid.uuid4().hex[:8]}", name="New Cafe Onboarding",
                                description="d", workspace_id=ws, template_definition={"steps": []}, created_by="f288",
                                steps=[{"order": 1, "agent_id": analyst.id, "prompt_template": ASK_FOR_DETAILS},
                                       {"order": 2, "agent_id": analyst.id, "prompt_template": WELCOME}])
    db.add(playbook)
    db.flush()

    def run(input_data, triggered_by="platform_action"):
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        db.add(RecipeExecution(execution_id=execution_id, recipe_id=playbook.id, workspace_id=ws, status="running",
                               input_data=input_data, triggered_by=triggered_by))
        db.flush()
        return execution_id

    return NS(db=db, ws=ws, analyst=analyst, run=run, agent=agent)


def _step_prompt(place, step, input_data, *, agent=None, triggered_by="platform_action"):
    from services.step_lessons import a_playbook_step_carries_its_lessons

    sent = {}

    async def execute(**kwargs):
        sent.update(kwargs)
        return {"status": "success"}

    asyncio.run(a_playbook_step_carries_its_lessons(execute)(
        db=place.db, agent=agent or place.analyst, clean_prompt=step, workspace_id=place.ws,
        input_data=input_data, recipe_execution_id=place.run(input_data, triggered_by)))
    return sent["clean_prompt"]


def _given(prompt):
    from services.playbook_given import GIVEN_HEADING

    return prompt.split(f"{GIVEN_HEADING}\n", 1)[1].split("\n\n", 1)[0].splitlines()[1:]


def test_every_agent_step_gets_the_details_no_step_names(onboarding):
    for step in (ASK_FOR_DETAILS, WELCOME):
        prompt = _step_prompt(onboarding, step, FROM_AUTO)
        assert prompt.startswith(step)
        assert _given(prompt) == ["cafe_name: Larder & Loaf", "owner_name: Ffion Hughes",
                                  "owner_email: ffion@larderandloaf.example", "first_order: 4 kg Harbour Blend",
                                  "delivery_day: Tuesdays", "include_decaf: false"]


def test_a_session_agents_step_gets_them_too(onboarding):
    from services.playbook_given import GIVEN_HEADING
    from services.step_lessons import ON_THE_CARD

    session = onboarding.agent("Numbers (on my Mac)", {"runtime": "cli"})
    prompt = _step_prompt(onboarding, WELCOME, FROM_AUTO, agent=session)
    assert GIVEN_HEADING in prompt and ON_THE_CARD not in prompt       # its ticket shows only its prompt


def test_a_trigger_or_a_webhook_run_is_left_as_it_was(onboarding):
    from services.playbook_given import GIVEN_HEADING

    assert GIVEN_HEADING not in _step_prompt(onboarding, WELCOME, {"cafe_name": "Tidewater"}, triggered_by="webhook")
    assert GIVEN_HEADING not in _step_prompt(onboarding, WELCOME, {"content": "New order", "sender": "shop"})
    assert GIVEN_HEADING not in _step_prompt(onboarding, WELCOME, {})


def test_a_detail_a_step_names_reaches_it_there_and_not_again():
    from services.playbook_given import unnamed_details

    named = ["Deliveries on {{ delivery_day }}.", "Write to {input.owner_email}."]
    assert unnamed_details(FROM_AUTO, [ASK_FOR_DETAILS, *named]) == {
        "cafe_name": "Larder & Loaf", "owner_name": "Ffion Hughes", "first_order": "4 kg Harbour Blend",
        "include_decaf": False}
    assert unnamed_details(FROM_AUTO, ["Onboard the cafe: {input}"]) == {}      # a bare {input} reads them all
    assert unnamed_details({"cafe_name": "", "notes": None}, [WELCOME]) == {}
