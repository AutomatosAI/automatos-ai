"""F308 (night 9): Auto never says a mission's steps pause for the owner when they don't.

#0033 was made with no check of each step, and approved with "Great! Mission #0033 has
been approved and is now running. … I'll let you know as soon as that step is complete
and ready for your review." Earlier, on #0027 and #0035: "Each step will pause for your
approval." A reply that said the steps wait, over a mission call this turn whose answer
said they run unchecked, was a claim no call backed (F261's families).

PRD-256 FX-007 (D10): the families are gone. "Each step will pause" says how the mission
will run, not work done, so the receipts' claim rule clears it; what the owner reads
instead is the mission's own receipt, from its call's answer: "its steps run without your
check", or, once the check is switched on in the same turn, "each step waits for your OK".

P256-FIX-RVW-37: a step the owner thinks waits for them is a wrong statement they act on. A claim
that each step pauses needs a receipt that says each step waits for the owner's OK
(``claims_by_kind``): over an unchecked mission, or none, it is nudged and gets the line.
"""
from __future__ import annotations

import pytest

from consumers.chatbot.claims_by_kind import STEPS_LABEL
from consumers.chatbot.receipts import build_receipts
from tests.helpers_receipts_rule import call, line, nudged, tracker_of

CREATE = {"goal": "September retail takings, then a team note"}
SWITCH_ON = {"mission_id": "0033", "check_each_step": True}
UNCHECKED = call("platform_create_mission", CREATE, {"success": True, "mission_id": "x", "checks_each_step": False})
CHECKED = call("platform_create_mission", CREATE, {"success": True, "mission_id": "x", "checks_each_step": True})
SWITCHED_ON = call("platform_update_mission_plan", SWITCH_ON,
                   {"success": True, "message": "Every step … waits", "checks_each_step": True})


def _effects(*calls):
    return [r["effect"] for r in build_receipts(tracker_of(calls))]


@pytest.mark.parametrize("said", [
    "I've created Mission #0033. Each step will pause for your approval before proceeding.",
    "Mission #0033 is set up, and it will stop after every step for your OK.",
])
def test_steps_said_to_pause_over_an_unchecked_mission_is_caught(said):
    assert nudged(said, UNCHECKED) == STEPS_LABEL                              # the create backs "created" only
    assert line(said, UNCHECKED) == ("Just to be clear: nothing in this reply set the mission's steps to wait for "
                                     "your OK. Ask me again if you want it done.")
    assert _effects(UNCHECKED) == ["its steps run without your check"]
    assert nudged(said, CHECKED) is None and line(said, CHECKED) is None
    assert nudged(said, UNCHECKED, SWITCHED_ON) is None and line(said, UNCHECKED, SWITCHED_ON) is None


def test_the_receipt_says_so_when_the_mission_checks_each_step():
    assert _effects(CHECKED) == ["each step waits for your OK"]


def test_the_receipt_says_so_once_the_check_is_switched_on_in_the_same_turn():
    assert _effects(UNCHECKED, SWITCHED_ON)[-1] == "each step waits for your OK"


def test_a_question_or_a_denial_is_no_claim_and_a_turn_with_no_mission_backs_none():
    assert nudged("Would you like each step to pause for your approval?", UNCHECKED) is None
    assert nudged("Each step will not pause for your approval: they run on their own.", UNCHECKED) is None
    assert nudged("Each step will pause for your approval.") == STEPS_LABEL      # no receipt says so
