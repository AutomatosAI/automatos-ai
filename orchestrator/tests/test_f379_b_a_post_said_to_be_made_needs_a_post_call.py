"""F379 (night 11, 7 Oct): a social post said to be made, changed or waiting needs a post call this turn.

Night 11: "I've drafted the three social media posts" (chat 508b4e05) with none made (B4); "I've
updated the carousel … and removed any placeholder tasting notes" with the notes still there
(B-i3-3); "it's waiting in the Socials tab" over a PNG (B-i3-2). Each was backed then only by a
create or an update of a social post that succeeded (a submit, for "waiting"); a read of the
posts backed none of them, nor "I've saved it", nor "I've posted it".

PRD-256 FX-007 (D10): the post families are gone; the receipts' rule reads each report of work
done against a done write of its kind (a read backs none), and "removed" is backed by an edit
(F379's fix kept). Where a post is waiting is no report of work done: cleared, the receipts show
whether a post was made or submitted.
"""
from __future__ import annotations

import pytest

from tests.helpers_receipts_rule import line, nudged

MADE = ("platform_create_social_post",)
CHANGED = ("platform_update_social_post",)
READ = ("platform_list_social_posts", "platform_get_social_post")
PUBLISHED = ("composio_execute", {"action": "INSTAGRAM_PUBLISH_MEDIA", "params": {}}, {"successful": True})

NIGHT_11 = [
    ("I've drafted the three social media posts for Monday, Tuesday and Thursday.", "done", MADE),
    ("I've made a carousel for the Harvest Club box.", "made", MADE),
    ("I've updated the carousel with the date and price, and removed any placeholder tasting notes.",
     "updated", CHANGED),
    ("I've removed the tasting notes from the carousel.", "removed", CHANGED),
]
WAITING = [
    ("It's waiting in the Socials tab for your approval.", ("platform_submit_social_post",)),
    ("You'll find them in your Socials tab.", MADE),
    ("The carousel is now waiting for your approval.", CHANGED),
]


@pytest.mark.parametrize("reply, claim, backing", NIGHT_11, ids=[r[:40] for r, _, _ in NIGHT_11])
def test_a_post_claim_needs_a_post_call_and_a_read_never_backs_it(reply, claim, backing):
    assert nudged(reply) == claim
    assert nudged(reply, *READ) == claim
    assert nudged(reply, *backing) is None


@pytest.mark.parametrize("reply, backing", WAITING, ids=[r[:40] for r, _ in WAITING])
def test_where_a_post_waits_is_no_report_of_work_done(reply, backing):
    assert nudged(reply) is None and nudged(reply, *READ) is None and nudged(reply, *backing) is None


def test_a_change_to_a_post_is_not_read_as_a_cancel_or_a_delete():
    """Before F379, "removed" was the "deleted" family's: a post edit never backed it."""
    assert nudged("I've removed the tasting notes from the carousel.", *CHANGED) is None


@pytest.mark.parametrize("reply", [
    "Would you like me to draft three Instagram posts?",
    "You can approve posts in the Socials tab.",
    "The mission is waiting for your approval.",
    "I drafted those posts earlier, before the price changed.",
])
def test_an_offer_a_place_and_a_back_reference_are_no_claim(reply):
    assert nudged(reply) is None and line(reply) is None


@pytest.mark.parametrize("reply", [
    "I've drafted a blog post about the Harvest Club.",
    'I\'ve turned off the timer for your "Weekly Instagram posts" playbook.',
])
def test_a_blog_post_or_a_timer_said_done_with_nothing_run_is_caught(reply):
    """No post claim, as the families read them; a report of work done with nothing run, as the receipts do."""
    assert line(reply) == ("Just to be clear: I haven't done that yet, and nothing has changed. "
                           "Ask me again if you want it done.")


def test_a_read_of_the_posts_never_backs_a_save_or_a_post_out():
    assert nudged("I've saved it as a draft social post.", *READ) == "saved"
    assert nudged("I've saved it as a draft social post.", *MADE) is None                 # F314 kept
    assert nudged("I've posted it to Instagram.", *MADE, *READ) == "posted"
    assert nudged("I've posted it to Instagram.", PUBLISHED) is None


def test_the_rest_of_the_reply_is_still_checked():
    assert nudged("I've drafted the carousel. I've also approved the mission.", *MADE) == "approved"


def test_the_owner_reads_what_did_not_happen_in_plain_words():
    assert line("I've made a carousel.", *READ) == (
        "Just to be clear: I haven't done that yet, and nothing has changed. Ask me again if you want it done.")
    assert line("I've made a carousel.", "platform_store_memory") == (
        "Just to be clear: I haven't made anything in this reply. Ask me again if you want it done.")
