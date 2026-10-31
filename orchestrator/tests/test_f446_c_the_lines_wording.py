"""P256-FIX-N17-5 (F446, night 17): the line says what happened to a refused write, in words the owner reads.

Idx 430: "I asked for your click …; I tried to change the card #1757 and it didn't go through" over a refused
update_task then the card raised for it: a refusal was settled only by a DONE write of the same action, so the ask
that followed it read as a failure. Idx 181: "I tried to do that" for GMAIL_SEND_EMAIL: ``receipts.tried`` read the
slug's toolkit ("gmail") as its verb.

Now one helper, ``receipts.refused_writes_left``, settles a refused write by any LATER done or waiting write of the
same action, for the line in place (``claim_removed``) and the lines above (``honesty_lines``); and a Composio slug
is named by its words after the toolkit ("send the email"), "do that" only when its first word is no verb.
"""
from __future__ import annotations

from consumers.chatbot.claim_removed import without_unbacked_claims
from consumers.chatbot.receipts import DONE, REFUSED, WAITING, build_receipts, honesty_lines, receipt, \
    refused_writes_left, tried
from tests.helpers_receipts_rule import call, tracker_of

NOT_YOURS = {"success": False, "error": "Changing this card needs the owner's click."}
ASKED = {"success": False, "requires_confirmation": True, "grant_id": 1757, "act": "change the card #1757"}
REFUSED_CHANGE = call("platform_update_task", {"task_id": "#1757", "title": "Tidewater reorder"}, NOT_YOURS)
CARD_RAISED = call("platform_update_task", {"task_id": "#1757", "title": "Tidewater reorder"}, ASKED)


def _composio(slug, result):
    return ("composio_execute", {"action": slug, "params": {"to": "declan@example.com"}}, result)


def test_a_refusal_then_the_card_raised_for_it_is_an_ask_not_a_failure():
    outcomes = tracker_of([REFUSED_CHANGE, CARD_RAISED]).outcomes
    text = without_unbacked_claims("I've changed the card #1757.", outcomes, True).text
    assert "I asked for your click to change the card #1757" in text
    assert "didn't go through" not in text
    assert honesty_lines(build_receipts(tracker_of([REFUSED_CHANGE, CARD_RAISED])), "Here's where #1757 stands.") == []


def test_a_refusal_after_a_done_write_of_the_same_action_still_stands():
    """Only a LATER write settles a refusal: a card moved, then another card's move refused, is said."""
    moved = call("platform_update_task_status", {"task_id": "#1750", "status": "done"})
    refused = call("platform_update_task_status", {"task_id": "#1751", "status": "done"},
                   {"success": False, "error": "A running ticket can only be cancelled."})
    (line,) = honesty_lines(build_receipts(tracker_of([moved, refused])), "Here's the board.")
    assert line.startswith("I tried to move the card #1751 and it didn't go through")
    assert honesty_lines(build_receipts(tracker_of([refused, moved])), "Here's the board.") == []


def test_the_first_standing_refusal_of_each_action_is_kept():
    writes = [{"action": "platform_update_task", "status": REFUSED, "n": 1},
              {"action": "platform_update_task", "status": WAITING, "n": 2},
              {"action": "platform_update_task", "status": REFUSED, "n": 3},
              {"action": "platform_update_task", "status": REFUSED, "n": 4},
              {"action": "platform_create_task", "status": REFUSED, "n": 5},
              {"action": "platform_create_task", "status": DONE, "n": 6}]
    assert [r["n"] for r in refused_writes_left(writes)] == [3]


def test_a_composio_slug_is_named_by_its_words_after_the_toolkit():
    (refused,) = tracker_of([_composio("GMAIL_SEND_EMAIL", {"successful": False, "error": "no recipient"})]).outcomes
    assert tried(receipt(*refused)) == "send the email"
    (archived,) = tracker_of([_composio("NOTION_PAGE_ARCHIVE", {"successful": False, "error": "no page"})]).outcomes
    assert tried(receipt(*archived)) == "do that"                       # "page" is no verb
    (moved,) = tracker_of([call("platform_update_task_status", {"task_id": "#0422"},
                                {"success": False, "error": "no status"})]).outcomes
    assert tried(receipt(*moved)) == "move the card #0422"              # a platform call reads as before
