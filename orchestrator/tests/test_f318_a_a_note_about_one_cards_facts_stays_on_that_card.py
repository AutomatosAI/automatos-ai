"""F318 (night 9b): a send-back note about one card's facts stays on that card.

The owner's send-back notes came back on other cards as things they had said there:
#0088 "I found the September margin sheet which the owner mentioned in their
corrections", #0097 "the 'September margin sheet' you mentioned", #1977 "£2.87 per bag
(not per kg, as you corrected)", #1976 "Addressing your question about roaster loss",
#1991 "roasting loss factor (mentioned in your corrections)". Every send-back note was a
lesson for the agent's other cards, and most notes about a card's facts name nothing
F315's scope test could hold them back by. Only a note the owner said holds in general,
or one about the work's form, is carried now; and the block never calls them corrections.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

# #0072, the BA's margin card, sent back at 16:20.
MARGIN_FACTS = ("The September margin sheet is my real costing. £2.87 is per bag, not per kg. Cerrado and Sumatra "
                "go in the blend.")
# #0099, the Watchdog's Christmas card, sent back at 16:43.
ROASTER = "Does the 30 kg allow for what we lose in the roaster? Where did the figures come from?"
# #0107, the newsletter helper's intro, sent back at 17:01: about the form, so a rule.
VOICE = "Close, but check it against my brand voice paper: it says how long the intro should be and how I sign off."
NEXT = "We make about £11 a bag on Kirinyaga, don't we? Just confirm it."      # #1977


@pytest.fixture
def team(db_session, seed_workspace):
    """One agent with three cards sent back (two about their facts, one about form) and #1977 to run."""
    from core.models import Agent
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    agent = Agent(name="Shopify Business Analyst", agent_type="custom", description="", status="active",
                  configuration={}, workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db_session.add(agent)
    db_session.flush()

    def card(title, status="done", notes=(), **fields):
        made = BoardTask(workspace_id=ws, title=title, description=title, raw_prompt=title, status=status,
                         assigned_agent_id=agent.id, planning_data={"owner_corrections": [
                             {"note": note, "by": "user:o", "at": f"2026-10-04T16:2{n}:00+00:00"}
                             for n, note in enumerate(notes)]}, **fields)
        db_session.add(made)
        db_session.flush()
        return made

    margin = card("Most profitable single origin", notes=[MARGIN_FACTS])
    card("Enough green coffee for the Christmas boxes?", notes=[ROASTER])
    card("Harbour Log intro for October", notes=[VOICE])
    return NS(db=db_session, ws=ws, agent=agent, card=card, margin=margin)


def test_another_card_gets_the_owners_rules_and_none_of_another_cards_facts(team):
    from services.ticket_redo import NOT_SAID_HERE, STANDING_HEADING, redo_block

    told = redo_block(team.card(NEXT, status="assigned"))

    assert STANDING_HEADING in told and f"- {VOICE}" in told              # a rule about form still travels
    assert MARGIN_FACTS not in told and "£2.87" not in told                 # night 9b: "as you corrected" on #1977
    assert ROASTER not in told and "roaster" not in told                    # night 9b: #1976, #1991
    assert NOT_SAID_HERE in told and "as you corrected" in NOT_SAID_HERE    # never credited to the owner here
    assert "correction" not in STANDING_HEADING.lower()


def test_the_card_a_note_was_written_on_still_redoes_with_it(team):
    from services.ticket_redo import redo_block

    team.margin.review_feedback = MARGIN_FACTS
    team.margin.status = "assigned"

    assert f"1. {MARGIN_FACTS}" in redo_block(team.margin)


def test_a_card_auto_starts_reads_its_own_names_from_its_brief_not_the_guides(team):
    """The guide passages a draft gains name the owner's people: Rosa's sign-off note stayed
    with Rosa's card only while the card's words, not its prompt, were read."""
    from services.step_lessons import a_cards_run_carries_its_lessons

    rosa = ("You signed it as Lantern Kitchen. Start 'Hi Rosa,' and sign off 'Gerard, Harbourline Coffee "
            "Roasters'.")
    team.card("Reply to Rosa at Lantern Kitchen about delivery", notes=[rosa])
    brief = "A café wants 10 kg next week. What do we charge for delivery?"

    async def read(db, workspace_id, agent_id, given):
        return f"{given}\n\nFrom your documents: Rosa at Lantern Kitchen takes 12 kg a week."

    prompt = asyncio.run(a_cards_run_carries_its_lessons(read)(team.db, str(team.ws), team.agent.id, brief))

    assert "Hi Rosa" not in prompt and f"- {VOICE}" in prompt


@pytest.mark.parametrize("note, standing", [
    (MARGIN_FACTS, False),
    (ROASTER, False),
    ("My September margin sheet has a different figure for Kirinyaga. Check it and tell me which is right.", False),
    ("I did not mention a sheet, and anyway it is already in my documents: margin-sheet-sep-2026. Please just look.",
     False),
    ("Is that everything? Check what I have told the team about Quay before you answer.", False),
    ("You say No at the top and then 82 kg short at the bottom. Which is it?", False),
    ("I said two bags of Guji, is that right?", False),
    (VOICE, True),
    ("85–95 words, less 'exceptional' and 'unique'.", True),
    ("Leave off 'Perfect!' at the start.", True),
    ("Just the table: nothing before it, no working under it and no bold rows.", True),
    ("Fine, I will add a line myself. Still short of 80 words, so next time count.", True),
    ("Say which of my documents each figure came from.", True),
])
def test_a_standing_lesson_is_a_rule_about_the_work_not_one_cards_facts(note, standing):
    from services.lesson_scope import is_standing

    assert is_standing(note) is standing
