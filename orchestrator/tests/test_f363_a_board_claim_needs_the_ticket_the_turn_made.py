"""F363 (night 10c, chat ee769185): Auto's board claims are checked against the ticket the turn made.

17:57:26 — "Go with A … Get the designer to send me the approval card." Auto called only
platform_store_memory and answered "I'll get the Brand Designer to prepare the approval card":
no ticket, nothing on the board, and nothing under the reply said so. 17:58 — "Nothing's on my
board. Please actually put it on a ticket." This time the ticket was filed (#0891), but the
reply repeated "I've noted your decision" and was told "Just to be clear: I didn't save anything
in this reply", beside "You should now see this on your board".

Then, handing work to someone ("I'll get the Brand Designer to …") and "you should now see this
on your board" were board claims a ticket made this turn had to back, and a ticket made backed
"noted".

PRD-256 FX-007 (D10): the families are gone. The receipts' rule keeps the false denial fixed (a
card the turn made backs "noted") and reads "I've created a task" against a create. A hand-off
said as a plan ("I'll get … to") and "it's now on your board" are cleared by the rule: the
receipts above the answer show what ran (a memory note, and no card).
"""
from __future__ import annotations

import pytest

from consumers.chatbot.receipts import build_receipts
from tests.helpers_receipts_rule import line, nudged, tracker_of

MEMORY_ONLY = ("platform_store_memory",)
TICKET_FILED = ("platform_list_tasks", "platform_create_task")   # platform_execute's inner action
AT_17_57 = ("Right, you've reviewed the palettes and you're going with **Option A** for the warmer colours.\n\n"
            "I've noted your decision. I'll get the Brand Designer to prepare the approval card for you to confirm "
            "these changes.\n\n**Action:** Brand Designer to prepare approval card for Gerard's final confirmation.")
AT_17_58 = ("My apologies, Gerard. I've now created a task on your board for the Brand Designer:\n\n"
            "**Task Title:** Implement Brand Kit Update - Option A & Warm Sand Band\n\n"
            "You should now see this on your board, with task number #0891. The Brand Designer will proceed with the "
            "update.\n\nI've noted your decision.")


def test_handing_work_on_with_no_ticket_is_shown_by_the_receipts():
    """17:57: the memory note backs "noted"; the hand-off is a plan; the receipts show no card."""
    assert nudged(AT_17_57, *MEMORY_ONLY) is None and line(AT_17_57, *MEMORY_ONLY) is None
    (note,) = build_receipts(tracker_of(MEMORY_ONLY))
    assert note["effect"] == "memory saved" and not note["subject"].startswith("#")
    assert nudged(AT_17_57) == "noted"                                         # nothing at all ran: caught


@pytest.mark.parametrize("said, over_a_read", [
    ("You should now see this on your board, with task number #0891.", "done"),
    ("It's now on your board for the Brand Designer.", "done"),
    ("You'll see the new ticket on your board in a moment.", None),      # a future: cleared
])
def test_saying_it_is_on_the_board_needs_a_write_this_turn(said, over_a_read):
    assert nudged(said, "platform_list_tasks") == over_a_read
    assert nudged(said, *TICKET_FILED) is None


def test_the_reply_that_filed_the_ticket_is_not_told_nothing_was_saved():
    assert nudged(AT_17_58, *TICKET_FILED) is None and line(AT_17_58, *TICKET_FILED) is None   # F363
    assert nudged(AT_17_58, *MEMORY_ONLY) == "created"


def test_an_agents_draft_hands_nothing_on():
    """P256-FIX-RVW-7 (restored against the receipts rule): an agent's run, its writer's voice."""
    assert nudged(AT_17_57, *MEMORY_ONLY, promises=False) is None


def test_a_hand_off_the_turn_filed_stands():
    said = "I'll get the Analyst to check the figures and report back on the card."
    assert nudged(said, *TICKET_FILED) is None


@pytest.mark.parametrize("said", [
    "Would you like me to get the designer to draft a proposal?",          # an offer
    "Once you approve it, I'll get the designer to save the kit.",         # later, on a condition
    "I'll have to check the board first.",
    "I'll ask you to confirm the colours before anything is saved.",       # asks the owner
])
def test_an_offer_a_condition_or_the_owner_is_no_hand_off(said):
    assert nudged(said, *MEMORY_ONLY) is None and line(said, *MEMORY_ONLY) is None
