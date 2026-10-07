"""F380 (night 11, 7 Oct): "I cannot directly create the social post" is not done.

#2159 (Social Media Director) answered "I cannot directly create the social post… Please
use the platform_create_social_post action with the following parameters…" and closed
done: F334's opening check wanted the verb straight after "cannot", and "directly" hid
it. An adverb before or after the verb is read through now, and so are render, post,
publish, save and upload. An opening that names the owner's approval as the next step
is still not a verdict, and F334's finished answers still close done.
"""
from __future__ import annotations

import pytest

from core.services.ticket_reasons import SAYS_NOT_DONE_NOTE_PREFIX
from services.said_not_done import said_not_done_note

ANSWER_2159 = (
    "I cannot directly create the social post… the interpreter does not have access to these platform-specific "
    "actions. Please use the `platform_create_social_post` action with the following parameters: "
    '`media: {"1080x1350": {"template_name": "Quote card"}}`.')


@pytest.mark.parametrize("answer", [
    ANSWER_2159,
    "I'm not able to actually post it. The caption is below.",
    "I’m not able to actually post it.",                       # a curly apostrophe
    "I can't currently render the video: the template needs a 40-second length.",
    "Unfortunately, I was unable to directly generate the image for the card.",
    "I am unable to create the post in Socials.",
    "I couldn't save the draft.",
])
def test_an_opening_that_says_it_could_not_make_it_is_not_done(answer):
    note = said_not_done_note(answer)

    assert note is not None and note.startswith(SAYS_NOT_DONE_NOTE_PREFIX)


def test_the_note_quotes_the_agents_own_opening():
    assert "I cannot directly create the social post" in said_not_done_note(ANSWER_2159)


@pytest.mark.parametrize("answer", [
    "I can't post it myself: you approve it in the Socials tab, and the platform publishes it.",
    "I cannot publish the post until you approve it. It is rendered and waiting in Socials.",
    "I couldn't find a 14-day term for Lantern Kitchen, so the invoice uses your standard 30 days.",
    "I can create the post as soon as the template is chosen, and here is the draft caption.",
    "The quote card is in Socials, waiting for your approval.",
])
def test_a_finished_or_approval_bound_answer_is_not_a_verdict(answer):
    assert said_not_done_note(answer) is None
