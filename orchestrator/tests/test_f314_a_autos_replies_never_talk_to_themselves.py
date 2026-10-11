"""F314 (night 9): Auto's replies talked to themselves.

- "You are absolutely right to call me out on that" with nothing said by the owner
  (chats ddd6799f, b8d9121f, 02159865, 046788d5, f5142b57, a471c235): each followed a
  re-prompt the platform sent in the user's turn ("Your previous reply says something
  was … Never report an action as done without a tool result"), which the model read
  as the owner's rebuke. Now a re-prompt in the user's turn says it is the platform's
  check, not the owner, and asks for a reply that stands on its own.
- The owner read "Correction: this reply says something was under way, but no action
  in it did that …" (b8d9121f). The line is now Auto's, in plain words (PRD-256 FX-007: the
  receipts' line, naming the claim's verb when another write went through).
"""
from __future__ import annotations

import asyncio

from consumers.chatbot.claim_check import Verdict
from consumers.chatbot.claims_backed import FAMILIES
from core.llm.turn_order import (
    as_the_platforms_check, as_the_users_turn, reprompts_in_the_users_turn,
)
from modules.tools.execution.nudges import PLATFORM_CHECK

OWNER = {"role": "user", "content": "Quick update for your records: Quay Coffee House moves to 30-day terms."}
ANSWER = {"role": "assistant", "content": "I've noted that Quay Coffee House moves to 30-day terms."}
NUDGE = ("Your previous reply says something was noted, but no tool call in this turn did that, so it has not "
         "happened. Make the call now, in this response, or say plainly that it has not been done.")


class Nudge(dict):
    """Stands in for the tool loop's user-turn nudge (FIXER's modules/tools/execution/nudges.Nudge)."""


def test_a_reprompt_in_the_users_turn_says_it_is_the_platform_not_the_owner():
    sent = as_the_users_turn([OWNER, ANSWER, {"role": "system", "content": NUDGE}])

    turn = sent[-1]["content"]
    assert sent[-1]["role"] == "user"
    assert turn.startswith(PLATFORM_CHECK) and "not a message from the owner" in turn   # F295 (9b): FIXER's line
    assert NUDGE in turn
    assert "Don't apologise" in turn and "don't thank anyone for a correction" in turn
    assert "if they didn't ask for something, say plainly that it wasn't done instead of doing it now" in turn


def test_the_tool_loops_own_user_turn_nudge_is_framed_too():
    chat = [OWNER, ANSWER, Nudge(role="user", content=NUDGE)]
    sent = {}

    async def generate(self, messages, tools=None, on_delta=None):
        sent["messages"] = messages
        return "reply"

    asyncio.run(reprompts_in_the_users_turn(generate)(None, chat, tools=None))

    assert sent["messages"][-1] == {"role": "user", "content": as_the_platforms_check(NUDGE)}
    assert chat[-1]["content"] == NUDGE                                  # the caller's list is not changed


def test_the_owners_own_words_are_never_framed():
    chat = [ANSWER, OWNER]
    sent = {}

    async def generate(self, messages, tools=None, on_delta=None):
        sent["messages"] = messages
        return "reply"

    asyncio.run(reprompts_in_the_users_turn(generate)(None, chat, tools=None))

    assert sent["messages"] is chat


def test_a_framed_reprompt_is_framed_once():
    assert as_the_platforms_check(as_the_platforms_check(NUDGE)) == as_the_platforms_check(NUDGE)


def test_the_owner_reads_plain_words_under_a_claim_no_action_backed():
    from tests.helpers_receipts_rule import line

    assert Verdict(tools=2).correction is None                          # PRD-256: said from receipts
    plain = line("I've started the weekly report.")
    named = line("I've started the weekly report.", "platform_store_memory")

    assert plain == ("Just to be clear: I haven't done that yet, and nothing has changed. "
                     "Ask me again if you want it done.")
    assert named == "Just to be clear: I haven't started anything in this reply. Ask me again if you want it done."
    for said in (plain, named):
        assert "Correction:" not in said and "this reply says something was" not in said


def test_every_kind_of_claim_has_its_own_plain_line():
    from tests.helpers_receipts_rule import line

    for verbs, _backs in FAMILIES:
        verb = sorted(verbs)[0]
        assert line(f"I've {verb} it.", "platform_pin_card") == (
            f"Just to be clear: I haven't {verb} anything in this reply. Ask me again if you want it done."), verb
