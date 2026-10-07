"""F263 (night 7b): "What's on my board?" is answered from the board.

"Evening Auto. What's on my board right now, and what's waiting for me?" was
classified a general question, so the turn had no tool at all. The automatic
document search put five passages from old reports in front of Auto, and it named
#0124, #0112, #0111 and #0110 as in Review (all done or cancelled, titles wrong)
and missed the five real Review cards. The same night, card actions by number
("Approve #0177 …", "Update #0199 …") went to the card's own agent (DELEGATE) and
"Give #0192 to the Support Agent" down the ASSIGN lane, which filed a copy.

PRD-256 US-010 (Decision D2, 7 Oct): the chat has no DELEGATE lane any more, so no card
action goes to the card's own agent; and a card handed to a named agent is the ASSIGN
lane ON THAT CARD (platform_assign_task, never a new card): the one board message here
that leaves Auto's own hands, by the owner's word.
"""
from __future__ import annotations

import asyncio

import pytest

BOARD = "Evening Auto. What's on my board right now, and what's waiting for me?"
ABOUT_THE_BOARD = [
    BOARD,
    "Approve #0177 with this note: Going with Kestrel's 250-box run, I'll order it Monday.",
    "Let's talk about #0188.3.",
    "Cancel #0193 please, I'll do the Lantern Room price list myself.",
    "Please give #0192 to the Shopify Support Agent.",
    "Yes, that's good. Update the card with it and send it back to the agent.",
    "What does ticket 12 say?",
]
NOT_THE_BOARD = [
    "Reply to Amy about order #1043 please",
    "Write three captions with #HarbourBlend",
    "Draft the shop page words: three months, £42, a gift card in the box.",
    "What's our returns policy for wholesale cafes?",
]


@pytest.mark.parametrize("said", ABOUT_THE_BOARD)
def test_a_card_number_the_board_or_the_card_is_about_the_board(said):
    from consumers.chatbot.board_questions import about_the_board

    assert about_the_board(said) is True


@pytest.mark.parametrize("said", NOT_THE_BOARD)
def test_an_order_number_a_hashtag_or_a_gift_card_is_not(said):
    from consumers.chatbot.board_questions import about_the_board

    assert about_the_board(said) is False


@pytest.mark.parametrize("said", ABOUT_THE_BOARD)
def test_auto_keeps_a_board_message_with_its_platform_tools(said):
    """AutoBrain's fast path: MOLECULE + the "platform" hint, so the turn has the board's
    tools and is never a new ticket. Since PRD-256 US-010 nothing in the chat is delegated;
    the hand-off below is the ASSIGN lane on its own card."""
    from consumers.chatbot.auto import AutoBrain

    assert AutoBrain._match_platform_query(said.lower()) is not None


def test_only_the_card_handed_to_a_named_agent_leaves_autos_hands():
    """PRD-256 US-010: "give #0192 to the … Agent" hands the card on (ASSIGN on #0192);
    every other board message stays Auto's, with its board tools."""
    from consumers.chatbot.handoffs import handed_card

    assert [said for said in ABOUT_THE_BOARD if handed_card(said)] == [
        "Please give #0192 to the Shopify Support Agent."]
    assert handed_card("Please give #0192 to the Shopify Support Agent.") == "#0192"


def test_the_board_question_gets_the_board_tools_from_the_classifier_too():
    from consumers.chatbot.intent_classifier import SmartIntentClassifier

    result = SmartIntentClassifier().classify(BOARD)
    assert result.requires_tools is True and "platform_board_snapshot" in result.suggested_tools


def test_a_board_question_is_not_answered_from_documents():
    from consumers.chatbot.knowledge_prefetch import asks_the_documents, prefetch

    searched = []

    async def search(args):
        searched.append(args)
        return {}

    found = asyncio.run(prefetch(None, "ws", BOARD, search=search, enabled=True, limit=5, min_score=0.3))
    assert found is None and searched == []
    assert asks_the_documents(BOARD) is False
    assert asks_the_documents("What's our returns policy for wholesale cafes?") is True
