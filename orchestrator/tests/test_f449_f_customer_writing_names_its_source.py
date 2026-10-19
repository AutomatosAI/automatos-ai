"""P256-FIX-N17-10 (F449, night 17, ask 181): customer writing names where its content comes from.

"draft the newsletter intro" briefed #1755 "Include the three facts below exactly as written": the owner's
three standing rules, the only owner-given terms in Auto's context, as the content (graded 1/5). The
hand-off note told Auto to carry every term "exactly as the owner gave them" and said nothing of a request
with no content. It now says the terms come from this request or the document that holds them, and that a
request with no content is briefed with its source or a question for the owner on the card, never filled
from memory or the Standing Rules. The Standing Rules' own lead and the dispatch contract are unchanged.
"""
from __future__ import annotations

from consumers.chatbot.handoffs import CUSTOMER_WRITING_SOURCE, TEAM_NOTE
from modules.context.sections.memory import STANDING_RULES_LEAD
from modules.coordination.dispatch_contract import DISPATCH_CONTRACT_FRAGMENT
from services.contested_facts import RULES_OUTRANK_DOCUMENTS


def _note():
    return TEAM_NOTE.format(article="a", kind="newsletter")


def test_the_terms_come_from_this_request_or_the_document_that_holds_them():
    note = _note()
    assert ("carries every name, address, figure, date and term as the owner gave them in this request, or names "
            "the document that holds them, and invents none.") in note
    assert "exactly as the owner gave them" not in note


def test_a_request_with_no_content_is_never_filled_from_memory_or_the_standing_rules():
    assert CUSTOMER_WRITING_SOURCE == (
        "When the owner gives no content for customer writing, the brief names the document that holds it or tells "
        "the agent to ask the owner on the card; it never fills the content from memory or the Standing Rules, which "
        "bound what the writing says but are not what it is about unless the owner said so.")
    assert CUSTOMER_WRITING_SOURCE in _note()


def test_the_standing_rules_lead_and_the_dispatch_contract_are_unchanged():
    assert STANDING_RULES_LEAD.startswith("What the person you are talking to asked you to remember, newest first.")
    assert STANDING_RULES_LEAD.endswith(RULES_OUTRANK_DOCUMENTS)
    assert DISPATCH_CONTRACT_FRAGMENT.startswith("A dispatch contract has four parts")
    assert DISPATCH_CONTRACT_FRAGMENT.endswith("Reference artifacts by name or id; never paste their content in.")
    assert "Standing Rules" not in DISPATCH_CONTRACT_FRAGMENT


def test_the_note_still_hands_the_work_to_the_team():
    note = _note()
    assert note.startswith("The owner asked for a newsletter for their customers and named none of their templates.")
    assert "platform_create_task with assigned_agent_name" in note and "don't write it yourself in this reply" in note
