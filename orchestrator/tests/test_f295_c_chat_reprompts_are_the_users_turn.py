"""F295 (night 8): a re-prompt after the model's own reply is sent as the user's turn.

OpenRouter folds system messages into Anthropic's system prompt, so a re-prompt sent
as a system message after Auto's reply left the conversation ending on that reply,
and Sonnet answered with 3 tokens. F187's and F205's chat re-prompts are built that
way; what the model is sent now ends on a user turn.
"""
from __future__ import annotations

import asyncio

from consumers.chatbot.claim_check import id_nudge
from consumers.chatbot.owner_words import owner_words_nudge
from core.llm.manager import LLMManager
from core.llm.turn_order import as_the_users_turn, reprompts_in_the_users_turn

OWNER = {"role": "user", "content": "What did #0226 do?"}
ANSWER = {"role": "assistant", "content": "Agent 226 wrote the counter card."}


def test_a_reprompt_after_autos_answer_is_the_users_turn():
    """F187's re-prompt, as the chat builds it."""
    chat = [{"role": "system", "content": "prompt"}, OWNER, ANSWER,
            {"role": "system", "content": id_nudge([("agent", "226")])}]

    sent = as_the_users_turn(chat)

    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert id_nudge([("agent", "226")]) in sent[-1]["content"]       # F314: framed as the platform's check
    assert chat[-1]["role"] == "system"                                   # the chat's own list is unchanged


def test_two_system_messages_after_the_answer_are_one_turn():
    chat = [OWNER, ANSWER, {"role": "system", "content": owner_words_nudge(["send_back"])},
            {"role": "system", "content": "Most relevant actions: …"}]

    sent = as_the_users_turn(chat)

    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert owner_words_nudge(["send_back"]) in sent[-1]["content"] and "Most relevant" in sent[-1]["content"]


def test_any_other_conversation_goes_as_it_came():
    after_tools = [OWNER, {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
                   {"role": "tool", "tool_call_id": "1", "content": "{}"},
                   {"role": "system", "content": "Do NOT call it again."}]
    calling = [OWNER, {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
               {"role": "system", "content": "note"}]
    first_turn = [{"role": "system", "content": "prompt"}, OWNER, {"role": "system", "content": "passages"}]

    for chat in (after_tools, calling, first_turn, [{"role": "system", "content": "only"}], []):
        assert as_the_users_turn(chat) is chat


def test_the_manager_sends_it_on_every_route():
    sent = {}

    async def generate(self, messages, tools=None, on_delta=None):
        sent["messages"] = messages
        return "reply"

    chat = [OWNER, ANSWER, {"role": "system", "content": "Say it in the owner's words."}]
    asyncio.run(reprompts_in_the_users_turn(generate)(None, messages=chat, tools=None))

    assert sent["messages"][-1]["role"] == "user"
    assert "Say it in the owner's words." in sent["messages"][-1]["content"]   # F314: framed as the platform's check
    assert LLMManager.generate_response.__code__ is reprompts_in_the_users_turn(generate).__code__
