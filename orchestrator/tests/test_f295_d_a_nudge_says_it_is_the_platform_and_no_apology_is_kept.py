"""F295 follow-up, night 9b (6586c8bf, 8578eeaf) — a nudge says it is the platform's check, and a nudged
reply keeps no apology.

Since F295 a nudge is the user's turn, so the model reads it as the owner speaking. After
the platform's claim check, Auto opened "You are absolutely right to call me out on that,
Gerard. My apologies. It seems I made a mistake in my previous response…", and the next
reply "You are absolutely right, Gerard. My apologies. I clearly missed the mark on that
one." Every nudge now opens with a line saying the platform wrote it and is not to be
answered, and an apology the reply opens with anyway is taken off.
"""
from __future__ import annotations

from core.llm.clients.base import LLMResponse
from modules.tools.execution.nudges import PLATFORM_CHECK, kept_if_blank, nudge, without_the_apology

APOLOGY_8578 = ("You are absolutely right to call me out on that, Gerard. My apologies. It seems I made a mistake "
                "in my previous response by reporting an action as done when the tool call itself didn't actually "
                "execute successfully. I will ensure that doesn't happen again. I've put the card on the board for "
                "the Operations Manager to draft the reorder email to Maya.")
WORK = "I've put the card on the board for the Operations Manager to draft the reorder email to Maya."


def test_every_nudge_says_the_platform_wrote_it():
    sent = nudge("Make the call now, in this response.")

    assert sent["role"] == "user"
    assert sent["content"].startswith(PLATFORM_CHECK)
    assert "not a message from the owner" in sent["content"] and "apologise" in sent["content"]


def test_an_apology_the_nudged_reply_opens_with_is_taken_off():
    kept = kept_if_blank(LLMResponse(content="Done."), LLMResponse(content=APOLOGY_8578))

    assert kept.content == WORK                                     # night 9b: 'You are absolutely right…'


def test_a_reply_that_is_only_an_apology_or_has_none_is_left_alone():
    only_sorry = LLMResponse(content="My apologies.")
    plain = LLMResponse(content="Kiln Bakehouse took 214 kg. You are right to ask about Crane.")

    assert without_the_apology(only_sorry) is only_sorry            # nothing would be left: kept as it is
    assert without_the_apology(plain) is plain                      # an apology mid-answer is not touched


def test_the_response_is_never_changed_in_place():
    original = LLMResponse(content=APOLOGY_8578)

    out = without_the_apology(original)

    assert original.content == APOLOGY_8578 and out is not original
