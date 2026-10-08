"""F261 (night 7b): Auto said it would do things and called nothing.

"I will now send this updated brief to the agent." (#0181, nothing sent), "Let me try
creating the mission one more time" (no call), and "Here's what I'll do: 1. Assign the
'Analyst' agent…" (nothing started) all ended replies. Each is work said to be under
way now, which F108's one nudge was for when no action that does it ran this turn.

PRD-256 FX-007 (D10): the promise family is gone. A promise is a plan, not a report of
work done, so the receipts' rule (``claims_backed``) clears it: no nudge, no line; the
receipts show the owner that nothing ran. A report of the same work as done is caught.
"""
from __future__ import annotations

import pytest

from tests.helpers_receipts_rule import line, nudged

PROMISES = [
    "I will now send this updated brief to the agent.",
    "Let me try creating the mission one more time, ensuring the parameters are passed correctly.",
    "I will try again with the correct cron expression.",
    "Here's what I'll do: 1. Assign the 'Analyst' agent to the steps of the playbook.",
]
NOT_PROMISES = [
    "Would you like me to try again?",
    "Let me know if you'd like anything else.",
    "Shall I approve it again with the note?",
]


@pytest.mark.parametrize("said", PROMISES)
def test_a_promise_with_no_call_behind_it_is_a_plan_the_receipts_clear(said):
    assert nudged(said) is None and line(said) is None


@pytest.mark.parametrize("said", NOT_PROMISES)
def test_an_offer_or_a_question_promises_nothing(said):
    assert nudged(said) is None and line(said) is None


def test_the_same_work_said_to_be_done_is_caught_unless_the_turn_did_it():
    assert nudged("I've sent the updated brief to the agent.") == "sent"
    assert nudged("I've created the mission.") == "created"
    assert nudged("I've created the mission.", "platform_create_mission") is None
