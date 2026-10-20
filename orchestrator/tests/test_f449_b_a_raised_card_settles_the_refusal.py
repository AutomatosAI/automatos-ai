"""P256-FIX-N17-10 (F449, night 17): a refusal followed by the card it asked for shows only the card.

Asks 14, 140, 265, 360, 375 and 430 printed "I tried … and it didn't go through" over a refusal the next
call made good by raising the owner's card. P256-FIX-N17-5 settled a refusal by a later done or waiting
write of the SAME action; ask 140's refusal was a new card (platform_create_task) refused as a copy of
#1746, and the send-back raised for #1746 is another action. A refusal now carries the card it named
(``acts_on``), and a later done or waiting write on that card settles it too. A refusal nothing made good
is still said, in the owner's words.
"""
from __future__ import annotations

from consumers.chatbot.claim_removed import not_done_in_place, without_unbacked_claims
from consumers.chatbot.receipts import build_receipts, honesty_lines, receipt
from modules.tools.execution.owner_reasons import ACTS_ON, OWNER_REASON, SENT_WRONG
from tests.helpers_receipts_rule import call, tracker_of

ASKED_1746 = {"success": False, "requires_confirmation": True, "grant_id": 1746, "act": "send the card #1746 back"}
ASKED_1757 = {"success": False, "requires_confirmation": True, "grant_id": 1757, "act": "change the card #1757"}
SENT_WRONG_RESULT = {"success": False, "error": "'platform_update_task' does not take ['card']", OWNER_REASON: SENT_WRONG}
A_COPY = {"success": False, "error": "#1746 ('Guji quote', review) is already on the owner's board: act on that card, "
                                     "don't make a new one. Nothing was done.",
          OWNER_REASON: "#1746 is already on your board, so no new card was made", ACTS_ON: "#1746"}
SEND_BACK_1746 = call("platform_update_task_status", {"task_id": "#1746", "status": "assigned",
                                                      "note": "make it 8kg fortnightly instead"}, ASKED_1746)
NEW_QUOTE_CARD = call("platform_create_task", {"title": "Read the existing Guji quote", "description": "Find it."},
                      A_COPY)


def _lines(*calls, answer="Here's where it stands."):
    return honesty_lines(build_receipts(tracker_of(calls)), answer)


def test_a_refusal_then_the_card_raised_for_the_same_call_draws_no_line():
    """Asks 14 and 265: the call came in the wrong shape, then went through as an ask for the click."""
    edit = {"task_id": "#1757", "title": "Tidewater reorder"}
    assert _lines(call("platform_update_task", {"card": "#1757"}, SENT_WRONG_RESULT),
                  call("platform_update_task", edit, ASKED_1757)) == []


def test_the_ask_is_said_and_the_refusal_it_made_good_is_not():
    """Ask 430: 'I asked for your click …', never 'I tried to change the card #1757'."""
    outcomes = tracker_of([call("platform_update_task", {"task_id": "#1757"}, SENT_WRONG_RESULT),
                           call("platform_update_task", {"task_id": "#1757", "title": "Tidewater"}, ASKED_1757)]).outcomes
    text = without_unbacked_claims("I've changed the card #1757.", outcomes, True).text

    assert "I asked for your click to change the card #1757" in text
    assert "I tried to change the card #1757" not in text


def test_a_new_card_refused_as_a_copy_is_settled_by_the_send_back_on_the_card_it_named():
    """Ask 140: the copy of #1746 refused, then #1746 sent back with the owner's words."""
    (refused,) = build_receipts(tracker_of([NEW_QUOTE_CARD]))

    assert refused[ACTS_ON] == "#1746"
    assert _lines(NEW_QUOTE_CARD, SEND_BACK_1746) == []
    in_place = not_done_in_place(["sent"], list(tracker_of([NEW_QUOTE_CARD, SEND_BACK_1746]).outcomes))
    assert "I asked for your click to send the card #1746 back" in in_place and "I tried to create" not in in_place


def test_a_refusal_on_one_card_is_not_settled_by_a_write_to_another():
    other = call("platform_update_task_status", {"task_id": "#1747", "status": "assigned", "note": "redo"}, ASKED_1746)
    assert _lines(NEW_QUOTE_CARD, other) == [
        "I tried to create the card \"Read the existing Guji quote\" and it didn't go through: #1746 is already on your "
        "board, so no new card was made."]


def test_a_lone_refusal_is_still_said_in_the_owners_words():
    move = call("platform_update_task_status", {"task_id": "#0422", "stage": "done"}, SENT_WRONG_RESULT)
    assert _lines(move) == ["I tried to move the card #0422 and it didn't go through: it was sent in the wrong form, "
                            "so nothing ran."]


def test_a_refusal_with_no_card_named_keeps_no_acts_on():
    (refused,) = build_receipts(tracker_of([call("platform_update_task", {"task_id": "#1757"}, SENT_WRONG_RESULT)]))
    assert ACTS_ON not in refused and ACTS_ON not in receipt("platform_update_task", {}, {"success": True})
