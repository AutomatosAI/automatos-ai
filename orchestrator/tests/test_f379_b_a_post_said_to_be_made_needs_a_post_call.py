"""F379 (night 11, 7 Oct): a social post said to be made, changed or waiting needs a post call this turn.

Night 11: "I've drafted the three social media posts" (chat 508b4e05) with none made (B4); "I've
updated the carousel … and removed any placeholder tasting notes" with the notes still there
(B-i3-3); "it's waiting in the Socials tab" over a PNG (B-i3-2). Each is backed now only by a
create or an update of a social post that succeeded (a submit, for "waiting"); a read of the
posts backs none of them, nor "I've saved it", nor "I've posted it".
"""
from __future__ import annotations

import pytest

from consumers.chatbot.claim_check import Verdict, not_done
from modules.tools.execution import social_post_claims as posts
from modules.tools.execution.action_claims import claimed_action_not_done

MADE = {"platform_create_social_post"}
CHANGED = {"platform_update_social_post"}
READ = {"platform_list_social_posts", "platform_get_social_post"}

NIGHT_11 = [
    ("I've drafted the three social media posts for Monday, Tuesday and Thursday.", posts.POST_MADE, MADE),
    ("I've made a carousel for the Harvest Club box.", posts.POST_MADE, MADE),
    ("I've updated the carousel with the date and price, and removed any placeholder tasting notes.",
     posts.POST_CHANGED, CHANGED),
    ("I've removed the tasting notes from the carousel.", posts.POST_CHANGED, CHANGED),
    ("It's waiting in the Socials tab for your approval.", posts.POST_WAITING, {"platform_submit_social_post"}),
    ("You'll find them in your Socials tab.", posts.POST_WAITING, MADE),
    ("The carousel is now waiting for your approval.", posts.POST_WAITING, CHANGED),
]


@pytest.mark.parametrize("reply, claim, backing", NIGHT_11, ids=[r[:40] for r, _, _ in NIGHT_11])
def test_a_post_claim_needs_a_post_call_and_a_read_never_backs_it(reply, claim, backing):
    assert claimed_action_not_done(reply, set(), promises=True) == claim
    assert claimed_action_not_done(reply, READ, promises=True) == claim
    assert claimed_action_not_done(reply, backing, promises=True) is None


def test_a_change_to_a_post_is_not_read_as_a_cancel_or_a_delete():
    """Before F379, "removed" was the "deleted" family's: a post edit never backed it."""
    assert claimed_action_not_done("I've removed the tasting notes from the carousel.", CHANGED) is None


@pytest.mark.parametrize("reply", [
    "Would you like me to draft three Instagram posts?",
    "I've drafted a blog post about the Harvest Club.",
    'I\'ve turned off the timer for your "Weekly Instagram posts" playbook.',
    "You can approve posts in the Socials tab.",
    "The mission is waiting for your approval.",
    "I drafted those posts earlier, before the price changed.",
])
def test_an_offer_a_blog_post_a_playbook_and_a_back_reference_are_no_post_claim(reply):
    assert posts.judged_here(reply, [])[0] is None


def test_a_read_of_the_posts_no_longer_backs_a_save_or_a_post_out():
    assert claimed_action_not_done("I've saved it as a draft social post.", READ, promises=True) == "noted"
    assert claimed_action_not_done("I've saved it as a draft social post.", MADE, promises=True) is None  # F314 kept
    assert claimed_action_not_done("I've posted it to Instagram.", MADE | READ, promises=True) == "sent"
    assert claimed_action_not_done("I've posted it to Instagram.", {"instagram_publish_media"}, promises=True) is None


def test_the_rest_of_the_reply_is_still_checked_by_the_other_families():
    reply = "I've drafted the carousel. I've also approved the mission."
    assert claimed_action_not_done(reply, MADE, promises=True) == "approved"


def test_the_owner_reads_what_did_not_happen_in_plain_words():
    assert Verdict(tools=1, claim=posts.POST_MADE).correction is None and not_done(posts.POST_MADE) == (
        "Just to be clear: I didn't make that post in this reply, so nothing new is waiting for your approval in "
        "the Socials tab. Ask me again if you want it done.")
