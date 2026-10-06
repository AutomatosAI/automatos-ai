"""F363 (night 10c, chat ee769185): Auto's board claims are checked against the ticket the turn made.

17:57:26 — "Go with A … Get the designer to send me the approval card." Auto called only
platform_store_memory and answered "I'll get the Brand Designer to prepare the approval card":
no ticket, nothing on the board, and nothing under the reply said so. 17:58 — "Nothing's on my
board. Please actually put it on a ticket." This time the ticket was filed (#0891), but the
reply repeated "I've noted your decision" and was told "Just to be clear: I didn't save anything
in this reply", beside "You should now see this on your board".

Now handing work to someone ("I'll get the Brand Designer to …") and "you should now see this on
your board" are board claims a ticket made this turn must back, and a ticket made backs "noted".
"""
from __future__ import annotations

import pytest

from consumers.chatbot.claim_check import Verdict
from modules.tools.execution.action_claims import claimed_action_not_done

MEMORY_ONLY = {"platform_store_memory"}
TICKET_FILED = {"platform_list_tasks", "platform_create_task"}   # platform_execute's inner action
AT_17_57 = ("Right, you've reviewed the palettes and you're going with **Option A** for the warmer colours.\n\n"
            "I've noted your decision. I'll get the Brand Designer to prepare the approval card for you to confirm "
            "these changes.\n\n**Action:** Brand Designer to prepare approval card for Gerard's final confirmation.")
AT_17_58 = ("My apologies, Gerard. I've now created a task on your board for the Brand Designer:\n\n"
            "**Task Title:** Implement Brand Kit Update - Option A & Warm Sand Band\n\n"
            "You should now see this on your board, with task number #0891. The Brand Designer will proceed with the "
            "update.\n\nI've noted your decision.")


def test_handing_work_on_with_no_ticket_is_corrected_on_the_board():
    claim = claimed_action_not_done(AT_17_57, MEMORY_ONLY, promises=True)

    assert claim == "put on the board"
    assert Verdict(tools=1, claim=claim).correction == (
        "Just to be clear: I didn't put anything on the board in this reply. Ask me again if you want it done.")


@pytest.mark.parametrize("said", [
    "You should now see this on your board, with task number #0891.",
    "It's now on your board for the Brand Designer.",
    "You'll see the new ticket on your board in a moment.",
])
def test_saying_it_is_on_the_board_needs_a_ticket_this_turn(said):
    assert claimed_action_not_done(said, MEMORY_ONLY, promises=True) == "put on the board"
    assert claimed_action_not_done(said, TICKET_FILED, promises=True) is None


def test_the_reply_that_filed_the_ticket_is_not_told_nothing_was_saved():
    assert claimed_action_not_done(AT_17_58, TICKET_FILED, promises=True) is None
    assert claimed_action_not_done(AT_17_58, MEMORY_ONLY, promises=True) == "put on the board"


def test_a_hand_off_the_turn_filed_stands():
    said = "I'll get the Analyst to check the figures and report back on the card."
    assert claimed_action_not_done(said, TICKET_FILED, promises=True) is None


@pytest.mark.parametrize("said", [
    "Would you like me to get the designer to draft a proposal?",          # an offer
    "Once you approve it, I'll get the designer to save the kit.",         # later, on a condition
    "I'll have to check the board first.",
    "I'll ask you to confirm the colours before anything is saved.",       # asks the owner
])
def test_an_offer_a_condition_or_the_owner_is_no_hand_off(said):
    assert claimed_action_not_done(said, MEMORY_ONLY, promises=True) is None


def test_an_agents_draft_hands_nothing_on():
    assert claimed_action_not_done(AT_17_57, MEMORY_ONLY, promises=False) is None   # its writer's voice
