"""F295 (night 9b): the reply to a re-prompt opens with the work, never an apology.

After the platform's claim check Auto answered "You are absolutely right to call me out
on that, Gerard. My apologies…" before the real answer (6586c8bf, 8578eeaf), and the
apology streamed live. The chat's re-prompts carry FIXER's platform line now, and the
apology the reply opens with comes off what streams and what is saved.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.llm.turn_order import is_a_reprompt, reprompts_in_the_users_turn
from modules.tools.execution.nudges import PLATFORM_CHECK

OWNER = {"role": "user", "content": "Quick update: Quay Coffee House moves to 30-day terms from November."}
ANSWER = {"role": "assistant", "content": "I've noted that Quay moves to 30-day terms."}
CHECK = {"role": "system", "content": "Your previous reply says something was noted, but no tool call did that."}
APOLOGY = "You are absolutely right to call me out on that, Gerard. My apologies for the confusion. "
WORK = "I haven't saved that anywhere yet. Shall I add it to your terms note?"


@dataclass
class Reply:
    content: str


def _run(chat, chunks):
    """The wrapped call, streaming ``chunks``: what streamed and what came back."""
    streamed = []

    async def on_delta(kind, text):
        streamed.append((kind, text))

    async def generate(self, messages, tools=None, on_delta=None):
        for chunk in chunks:
            await on_delta("text", chunk)
        return Reply("".join(chunks))

    reply = asyncio.run(reprompts_in_the_users_turn(generate)(None, chat, tools=None, on_delta=on_delta))
    return "".join(text for kind, text in streamed if kind == "text"), reply.content


def test_the_chats_reprompt_carries_the_platforms_line():
    from core.llm.turn_order import as_the_users_turn

    sent = as_the_users_turn([OWNER, ANSWER, CHECK])
    assert sent[-1]["content"].startswith(PLATFORM_CHECK) and is_a_reprompt(sent)


def test_the_apology_comes_off_what_streams_and_what_is_saved():
    streamed, saved = _run([OWNER, ANSWER, CHECK], ["You are absolutely right ", "to call me out on that, Gerard. ",
                                                    "My apologies for the confusion. ", "I haven't saved that ",
                                                    "anywhere yet. Shall I add it to your terms note?"])
    assert streamed.strip() == WORK and saved.strip() == WORK


def test_a_reply_with_no_apology_streams_as_it_came():
    streamed, saved = _run([OWNER, ANSWER, CHECK], ["I haven't saved that anywhere yet. ", "Shall I add it?"])
    assert streamed == "I haven't saved that anywhere yet. Shall I add it?" == saved


def test_the_owners_own_turn_is_never_touched():
    streamed, saved = _run([ANSWER, OWNER], [APOLOGY, WORK])
    assert streamed == APOLOGY + WORK == saved
