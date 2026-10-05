"""F337(c) (night 10): customer paperwork Auto wrote itself graded 4+ 8% of the time; the team's,
on tickets, 57%. The owner's call (5 Oct): a template named, Auto fills it (F351-c); a letter,
invoice, quote, flyer or price list asked for with no template named goes to the agent whose
job it is. The turn gets a note saying so; a question about paperwork, or "do it yourself",
gets none.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from consumers.chatbot.named_template_note import fills_the_named_template
from consumers.chatbot.paperwork_to_the_team import asks_for_paperwork, team_note
from modules.coordination.dispatch_contract import DISPATCH_CONTRACT_FRAGMENT
from modules.documents.presets import INVOICE

PASSAGES = "passages from wholesale-terms.md"
D2564ECF = ("Can you make me a wholesale price list for cafés, as a spreadsheet? Every coffee we sell, price per kg, "
            "minimum order, carriage and payment terms.")


@pytest.mark.parametrize("said, kind", [
    (D2564ECF, "price list"),                                                            # chat d2564ecf
    ("Write the Quay letter: to Maya Osei, moving to 30-day terms from 1 November.", "letter"),
    ("Please draft a quote for Rosa, 10 kg vs 12 kg of Harbour Blend.", "quote"),
    ("Do me a flyer for the October Harvest Club box.", "flyer"),
    ("Put together an invoice for Salt Kitchen: 8 kg Harbour Blend at £22.", "invoice"),
])
def test_paperwork_asked_for_in_plain_english_is_read(said, kind):
    assert asks_for_paperwork(said) == kind


@pytest.mark.parametrize("said", [
    "Did Rosa pay the invoice from September?",                                          # a question about one
    "Send the invoice to Rosa.",                                                         # sending, not making
    "How many bags of Kayon Mountain did Lantern Kitchen order in September?",
    "Can you write the Quay letter yourself? I want it now.",                            # kept with Auto
    "Make the flyer, but don't bother the team with it.",
    "",
])
def test_anything_else_is_not(said):
    assert asks_for_paperwork(said) is None
    assert team_note(said) is None


def test_the_note_hands_it_to_the_right_agent_with_the_owners_facts():
    note = team_note(D2564ECF)

    assert note.startswith("The owner asked for a price list for their customers and named none of their templates.")
    assert "don't write it yourself in this reply" in note
    assert "platform_recommend_agent" in note and "platform_create_task with assigned_agent_name" in note
    assert "exactly as the owner gave them and invents none" in note
    assert "who has it and its card number" in note and "on one of their templates" in note
    assert DISPATCH_CONTRACT_FRAGMENT in note


@pytest.fixture
def shop(db_session, seed_workspace):
    from core.models.core import DocumentTemplate

    ws = UUID(seed_workspace())
    db_session.add(DocumentTemplate(workspace_id=ws, name="Harbourline Invoice", format="pdf",
                                    category=INVOICE["category"], blocks=INVOICE["blocks"], data_schema={},
                                    sample_data={}, created_by="owner"))
    db_session.flush()
    return NS(db=db_session, ws=ws)


def _turn(shop, said, widget_mode=False):
    chat = NS(db=shop.db, workspace_id=str(shop.ws), widget_mode=widget_mode)
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": said}]

    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        llm_messages.append({"role": "system", "content": PASSAGES})
        yield "searched"

    async def run():
        return [f async for f in fills_the_named_template(retrieval_first)(chat, said, messages, None, "c", [])]

    assert asyncio.run(run()) == ["searched"]
    return messages


def test_a_turn_asking_for_paperwork_with_no_template_gets_the_team_note(shop):
    messages = _turn(shop, D2564ECF)

    assert messages[-2]["content"] == PASSAGES
    assert messages[-1]["role"] == "system" and messages[-1]["content"] == team_note(D2564ECF)


def test_a_named_template_is_still_filled_by_auto(shop):
    messages = _turn(shop, "Make the Lantern Kitchen invoice on my Harbourline Invoice template.")

    assert "Use THIS template" in messages[-1]["content"]
    assert "platform_create_task" not in messages[-1]["content"]


def test_a_widget_visitors_turn_gets_no_team_note(shop):
    messages = _turn(shop, D2564ECF, widget_mode=True)

    assert messages[-1]["content"] == PASSAGES


def test_the_note_reads_as_english():
    assert team_note("Make an invoice for Salt Kitchen.").startswith("The owner asked for an invoice for their customers")
