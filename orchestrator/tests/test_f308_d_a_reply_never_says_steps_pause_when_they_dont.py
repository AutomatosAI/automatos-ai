"""F308 (night 9): Auto never says a mission's steps pause for the owner when they don't.

#0033 was made with no check of each step, and approved with "Great! Mission #0033 has
been approved and is now running. … I'll let you know as soon as that step is complete
and ready for your review." Earlier, on #0027 and #0035: "Each step will pause for your
approval." A reply that says the steps wait, over a mission call this turn whose answer
said they run unchecked, is a claim no call backs (F261's families), and is nudged and
corrected like one; once the check is switched on in the same turn, it is backed.
"""
from __future__ import annotations

import pytest

from modules.tools.execution.action_claims import STEPS_WAIT_CLAIM, claimed_action_not_done
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

CREATE = {"action": "platform_create_mission", "params": {"goal": "September retail takings, then a team note"}}
SWITCH_ON = {"action": "platform_update_mission_plan", "params": {"mission_id": "0033", "check_each_step": True}}


def _did(*calls):
    tracker = ToolExecutionTracker()
    for args, result in calls:
        tracker.record_outcome("platform_execute", args, result)
    return tracker.succeeded


@pytest.mark.parametrize("said", [
    "I've created Mission #0033. Each step will pause for your approval before proceeding.",
    "Mission #0033 is set up, and it will stop after every step for your OK.",
])
def test_steps_said_to_pause_over_an_unchecked_mission_is_corrected(said):
    done = _did((CREATE, {"success": True, "mission_id": "x", "checks_each_step": False}))

    assert claimed_action_not_done(said, done, promises=False) == STEPS_WAIT_CLAIM


def test_it_stands_when_the_mission_checks_each_step():
    done = _did((CREATE, {"success": True, "mission_id": "x", "checks_each_step": True}))

    assert claimed_action_not_done("Each step will pause for your approval.", done, promises=False) is None


def test_it_stands_once_the_check_is_switched_on_in_the_same_turn():
    done = _did((CREATE, {"success": True, "mission_id": "x", "checks_each_step": False}),
                (SWITCH_ON, {"success": True, "message": "Every step … waits", "checks_each_step": True}))

    assert claimed_action_not_done("Each step will pause for your approval.", done, promises=False) is None


def test_a_question_about_it_or_a_turn_with_no_mission_is_no_claim():
    unchecked = _did((CREATE, {"success": True, "mission_id": "x", "checks_each_step": False}))

    assert claimed_action_not_done("Would you like each step to pause for your approval?", unchecked,
                                   promises=False) is None
    assert claimed_action_not_done("Each step will pause for your approval.", set(), promises=False) is None
