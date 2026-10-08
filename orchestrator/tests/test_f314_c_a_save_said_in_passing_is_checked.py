"""F314 (night 9, L100): "Just the draft here, please" got "I've saved this as a draft
social post", then, after the nudge, "This draft has now been saved as a social post";
nothing showed under Socials. The save told in passing was no claim family's, so it
was never checked, and a post made by create_social_post backed no "I've saved".

PRD-256 FX-007: the receipts' rule reads both (a passive is a claim) and a post made
backs the save (``saved``: a memory or a thing saved).
"""
from __future__ import annotations

from tests.helpers_receipts_rule import line, nudged

SAID_IN_PASSING = "This draft has now been saved as a social post. Let me know if you'd like any changes!"
SAID_OUTRIGHT = "I've saved this as a draft social post. Let me know if you'd like any changes!"


def test_a_save_said_in_passing_with_nothing_saved_is_a_claim():
    assert nudged(SAID_IN_PASSING) == "saved"


def test_a_post_that_was_made_backs_the_save():
    made = "platform_create_social_post"

    assert nudged(SAID_IN_PASSING, made) is None
    assert nudged(SAID_OUTRIGHT, made) is None
    assert nudged(SAID_OUTRIGHT) == "saved"


def test_the_owner_is_told_plainly_that_nothing_was_saved():
    assert line(SAID_OUTRIGHT, "platform_list_social_posts") == (
        "Just to be clear: I haven't done that yet, and nothing has changed. Ask me again if you want it done.")
    assert line(SAID_OUTRIGHT, "platform_assign_task") == (
        "Just to be clear: I haven't saved anything in this reply. Ask me again if you want it done.")
