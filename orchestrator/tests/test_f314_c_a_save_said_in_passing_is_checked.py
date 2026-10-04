"""F314 (night 9, L100): "Just the draft here, please" got "I've saved this as a draft
social post", then, after the nudge, "This draft has now been saved as a social post";
nothing showed under Socials. The save told in passing was no claim family's, so it
was never checked, and a post made by create_social_post backed no "I've saved".
"""
from __future__ import annotations

from consumers.chatbot.claim_check import Verdict
from modules.tools.execution.action_claims import claimed_action_not_done

SAID_IN_PASSING = "This draft has now been saved as a social post. Let me know if you'd like any changes!"
SAID_OUTRIGHT = "I've saved this as a draft social post. Let me know if you'd like any changes!"


def test_a_save_said_in_passing_with_nothing_saved_is_a_claim():
    assert claimed_action_not_done(SAID_IN_PASSING, set(), promises=True) == "noted"


def test_a_post_that_was_made_backs_the_save():
    made = {"platform_create_social_post"}

    assert claimed_action_not_done(SAID_IN_PASSING, made, promises=True) is None
    assert claimed_action_not_done(SAID_OUTRIGHT, made, promises=True) is None
    assert claimed_action_not_done(SAID_OUTRIGHT, set(), promises=True) == "noted"


def test_the_owner_is_told_plainly_that_nothing_was_saved():
    assert Verdict(tools=1, claim="noted").correction == (
        "Just to be clear: I didn't save anything in this reply. Ask me again if you want it done.")
