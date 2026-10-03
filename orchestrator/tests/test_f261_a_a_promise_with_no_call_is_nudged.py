"""F261 (night 7b): Auto said it would do things and called nothing.

"I will now send this updated brief to the agent." (#0181, nothing sent), "Let me try
creating the mission one more time" (no call), and "Here's what I'll do: 1. Assign the
'Analyst' agent…" (nothing started) all ended replies. Each is work said to be under
way now, which F108's one nudge is for when no action that does it ran this turn.
"""
from __future__ import annotations

import pytest

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
def test_a_promise_with_no_call_behind_it_is_work_said_to_be_under_way(said):
    from modules.tools.execution.action_claims import claimed_action_not_done

    assert claimed_action_not_done(said, set(), promises=True) == "under way"


@pytest.mark.parametrize("said", NOT_PROMISES)
def test_an_offer_or_a_question_promises_nothing(said):
    from modules.tools.execution.action_claims import claimed_action_not_done

    assert claimed_action_not_done(said, set(), promises=True) is None


def test_a_promise_the_turn_kept_is_no_claim():
    from modules.tools.execution.action_claims import claimed_action_not_done

    done = {"platform_create_mission"}
    assert claimed_action_not_done("Let me try creating the mission one more time.", done, promises=True) is None
