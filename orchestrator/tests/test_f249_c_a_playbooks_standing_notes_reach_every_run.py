"""F249 (night 8): a playbook's standing notes reach every run of it, whoever starts it.

Playbook 102 (New Cafe Onboarding) ignored the owner's standing note on three runs
running: #0440 (run by Auto), #0441 and #0455 (from the board) wrote £21 a kilo for the
£19.50 on the December list, "Hi <full name>", "usually take" and "Gerard & the
Harbourline Crew". A run's card belongs to its first step's agent, the Analyst, so the
playbook's notes were only the Analyst's five newest lessons, and the Analyst's other
cards pushed them out. "Every run of this playbook like this" did not count as a lesson,
and a note's figures came back on other cards: #0271 charged fees on the Rwanda's £10.75
for an £11.25 coffee, #0273 answered "12 kg" from the approval of #0265.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

PRICES = "Harbour Blend is £19.50 a kilo plus VAT on my December list, not £21. Every run of this playbook like this."
RIGHT_AT_LAST = "Right at last. Every run of this playbook like this, whether I start it or Auto does."
FIRST_NAME = "Hi Ola, not Hi Ola Brennan. Signed Gerard, not Gerard & the Harbourline Crew."
WHOEVER = "Right: To: first, Hi Ola, prices plus VAT, Gerard. Every run of this playbook like this, whoever starts it."
OWN_REDO = "Deliveries are on Wednesdays, so orders by Tuesday noon."
OTHER_PLAYBOOK = "Put the roast date on the first line."
STEP = "Draft a welcome email for the new wholesale cafe using the provided details."


@pytest.fixture
def harbourline(db_session, seed_workspace):
    """Playbook 102 with two earlier runs' cards carrying the owner's notes, the
    Analyst's own cards with newer notes, and the run that starts now."""
    from core.models import Agent
    from core.models.core import BoardTask, RecipeExecution, WorkflowTemplate

    db, ws = db_session, UUID(seed_workspace())

    def agent(name, configuration=None):
        made = Agent(name=name, agent_type="custom", description="", status="active",
                     configuration=configuration or {}, workspace_id=ws, created_by="test", owner_type="workspace",
                     owner_id=str(ws))
        db.add(made)
        db.flush()
        return made

    def playbook(name):
        made = WorkflowTemplate(template_id=f"f249c-{uuid.uuid4().hex[:8]}", name=name, description="d",
                                workspace_id=ws, template_definition={"steps": []}, created_by="f249c",
                                steps=[{"order": 1, "agent_id": analyst.id, "prompt_template": STEP}])
        db.add(made)
        db.flush()
        return made

    def run(of):
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        db.add(RecipeExecution(execution_id=execution_id, recipe_id=of.id, workspace_id=ws, status="running",
                               input_data={}, triggered_by="platform_action"))
        db.flush()
        return execution_id

    def card(of, agent_id, *, corrections=(), approved=None):
        execution_id = run(of)
        notes = [{"note": f"Approved: {approved[0]}", "by": "you", "at": approved[1]}] if approved else []
        db.add(BoardTask(workspace_id=ws, title=f"Recipe: {of.name}", status="done", source_type="recipe",
                         source_id=execution_id, assigned_agent_id=agent_id, created_by_type="recipe",
                         planning_data={"recipe_id": of.id, "execution_id": execution_id,
                                        "owner_corrections": [{"note": n, "by": "user:o", "at": at}
                                                              for n, at in corrections]},
                         runtime_ref={"session_notes": notes}))
        db.flush()
        return execution_id

    analyst, creator = agent("Analyst"), agent("Content Creator")
    onboarding, dispatch = playbook("New Cafe Onboarding"), playbook("Monday Dispatch Checklist")
    card(onboarding, analyst.id, corrections=[(PRICES, "2026-10-04T06:20:00+00:00")],
         approved=(RIGHT_AT_LAST, "2026-10-04T06:25:00+00:00"))                                    # #0415
    analysts = [f"Margin card {n}: just the table, no bold." for n in range(6)]
    for n, note in enumerate(analysts):                                                             # newer, elsewhere
        db.add(BoardTask(workspace_id=ws, title=f"Margin {n}", status="done", assigned_agent_id=analyst.id,
                         planning_data={"owner_corrections": [
                             {"note": note, "by": "user:o", "at": f"2026-10-04T06:3{n}:00+00:00"}]}))
    card(onboarding, analyst.id, corrections=[(FIRST_NAME, "2026-10-04T07:16:00+00:00")],
         approved=(WHOEVER, "2026-10-04T07:17:00+00:00"))                                          # #0441
    card(dispatch, creator.id, corrections=[(OTHER_PLAYBOOK, "2026-10-04T07:20:00+00:00")])
    now = card(onboarding, analyst.id, corrections=[(OWN_REDO, "2026-10-04T07:50:00+00:00")])     # #0455, sent back
    db.flush()
    return NS(db=db, ws=ws, analyst=analyst, onboarding=onboarding, now=now, analysts=analysts, agent=agent)


def _step_prompt(place, agent, **extra):
    from services.step_lessons import a_playbook_step_carries_its_lessons

    sent = {}

    async def execute(**kwargs):
        sent.update(kwargs)
        return {"status": "success"}

    asyncio.run(a_playbook_step_carries_its_lessons(execute)(
        db=place.db, agent=agent, clean_prompt=STEP, workspace_id=place.ws, recipe_execution_id=place.now, **extra))
    return sent["clean_prompt"]


def _notes_under(prompt, heading):
    block = prompt.split(f"{heading}\n", 1)[1].split("\n\n", 1)[0]
    return block.splitlines()[1:]


def test_the_playbooks_notes_reach_its_next_run_though_the_agents_other_cards_are_newer(harbourline):
    from services.step_lessons import ON_THE_CARD
    from services.ticket_redo import PLAYBOOK_HEADING, STANDING_HEADING

    prompt = _step_prompt(harbourline, harbourline.analyst)

    assert prompt.startswith(STEP) and prompt.endswith(ON_THE_CARD)
    # night 8: none of these reached #0440, #0441 or #0455; the Analyst's five newest were margin cards
    assert _notes_under(prompt, PLAYBOOK_HEADING) == [f"- {WHOEVER}", f"- {FIRST_NAME}", f"- {RIGHT_AT_LAST}",
                                                       f"- {PRICES}"]
    assert _notes_under(prompt, STANDING_HEADING) == [f"- {note}" for note in reversed(harbourline.analysts[1:])]
    assert all(prompt.count(note) == 1 for note in (WHOEVER, FIRST_NAME, RIGHT_AT_LAST, PRICES))


def test_the_runs_own_card_and_another_playbooks_cards_are_left_out(harbourline):
    prompt = _step_prompt(harbourline, harbourline.analyst)

    assert OWN_REDO not in prompt            # the redo carries the card's own notes already
    assert OTHER_PLAYBOOK not in prompt


def test_a_playbooks_notes_are_read_by_its_id_in_its_workspace(harbourline):
    from services.ticket_redo import playbook_lessons

    assert playbook_lessons(harbourline.db, harbourline.ws, harbourline.onboarding.id)[:2] == [OWN_REDO, WHOEVER]
    assert playbook_lessons(harbourline.db, uuid.uuid4(), harbourline.onboarding.id) == []


def test_a_session_agents_step_gets_the_playbooks_notes_and_leaves_its_lessons_to_its_ticket(harbourline):
    from services.step_lessons import ON_THE_CARD
    from services.ticket_redo import PLAYBOOK_HEADING, STANDING_HEADING

    session = harbourline.agent("Numbers (on my Mac)", {"runtime": "cli"})
    prompt = _step_prompt(harbourline, session)

    assert PLAYBOOK_HEADING in prompt and f"- {WHOEVER}" in prompt          # the playbook's, whoever runs the step
    assert STANDING_HEADING not in prompt and ON_THE_CARD not in prompt     # its ticket's prompt brings those


@pytest.mark.parametrize("note, teaches", [
    ("Approved: Right: Hi Jo, prices plus VAT, Gerard. Every run of this playbook like this, whoever starts it.", True),
    ("Approved: Good. Plain, right facts, starts at To:. That is how every cafe email should look.", True),
    ("Approved: Right: 5.5 kg with the sum under it. That is how I want it every time.", True),
    ("Approved: Do the Guji's card like this from now on.", True),
    ("Approved: I like this one, thanks.", False),
    ("Approved: Spot on. That is how I talk to Mel.", False),
])
def test_an_approve_note_that_says_how_every_run_should_be_is_a_lesson(note, teaches):
    from services.ticket_redo import NEXT_TIME

    assert bool(NEXT_TIME.search(note)) is teaches


def test_every_block_of_notes_says_their_figures_stay_on_their_card(harbourline):
    from services.ticket_redo import FIGURES_STAY, PLAYBOOK_HEADING, STANDING_HEADING, lessons_block, playbook_block

    standing = lessons_block(harbourline.db, harbourline.ws, harbourline.analyst.id).splitlines()
    assert standing[0] == STANDING_HEADING and standing[1].endswith(FIGURES_STAY)
    playbook = playbook_block([PRICES]).splitlines()
    assert playbook[0] == PLAYBOOK_HEADING and playbook[1].endswith(FIGURES_STAY) and playbook[2] == f"- {PRICES}"
    assert playbook_block([]) is None
    assert "belong to the card they were written on" in FIGURES_STAY and "never copy" in FIGURES_STAY
