"""F381 (night 11, 7 Oct): a how-to question from the owner changes nothing; Auto answers it and offers.

Night 11 (chat a576f1de, B-I4-1): "how do I give my agent two photos?" became platform_update_agent
×2, platform_assign_tool_to_agent (DROPBOX onto the Social Media Director) and task 2163 to upload
the photos to Dropbox. Now, in a turn whose owner asks how, where or whether, a call that changes
an agent, its tools, a connection, the board's work, the kit or a setting is refused, before any
gate or handler, with the answer to give instead. A request ("can you make me a post?") is not a
question. The persona says the same: a request is done, a question answered.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS

import pytest

from modules.tools.discovery import owner_turn as owner_turn_module
from modules.tools.discovery import question_turns as questions
from modules.tools.discovery.platform_executor import PlatformActionExecutor

WS = uuid.UUID("00000000-0000-0000-0000-0000000381a1")
CHAT = {"conversation_id": "00000000-0000-0000-0000-0000000381a2"}
NIGHT_11 = "how do I give my agent two photos?"


@pytest.mark.parametrize("said", [
    NIGHT_11,
    "Where do I add photos?",
    "Auto, how can I get two photos to the Social Media Director?",
    "Is there a way to give the director the before and after photos?",
    "Can I add my own photo to the carousel?",
    "Quick question: what's the best way to connect Instagram?",
])
def test_a_how_where_or_whether_question_is_one(said):
    assert questions.asks_how(said) is True


@pytest.mark.parametrize("said", [
    "Can you make me a post?",
    "Could you give my agent Dropbox?",
    "Please add Dropbox to the director.",
    "How do I add photos? Just do it for me.",
    "Where do they go? Give my agent Dropbox.",
    "Can I have a carousel for Friday?",
    "Give the Social Media Director Dropbox.",
    "Yes, go ahead.",
])
def test_a_request_or_an_instruction_is_not(said):
    assert questions.asks_how(said) is False


@pytest.fixture
def owner_said(monkeypatch):
    said = {"latest": NIGHT_11}
    monkeypatch.setattr(owner_turn_module, "owner_turn",
                        lambda db, workspace_id, caller_context: NS(latest=said["latest"]) if caller_context else None)
    return said


@pytest.mark.parametrize("action", ["platform_update_agent", "platform_assign_tool_to_agent", "platform_create_task",
                                    "platform_update_brand_kit", "platform_update_system_setting"])
def test_a_question_turn_refuses_the_changes_with_the_answer_to_give(owner_said, action):
    refusal = questions.refusal_on_a_question(None, WS, action, CHAT)

    assert refusal.startswith(f'The owner asked a question, "{NIGHT_11}", not for a change. Nothing was changed.')
    assert "offer to do it for them" in refusal


def test_a_read_a_request_and_a_call_outside_a_chat_go_on(owner_said):
    assert questions.refusal_on_a_question(None, WS, "platform_list_agents", CHAT) is None
    assert questions.refusal_on_a_question(None, WS, "platform_update_agent", None) is None
    owner_said["latest"] = "Can you give my agent Dropbox?"
    assert questions.refusal_on_a_question(None, WS, "platform_assign_tool_to_agent", CHAT) is None


def test_a_fault_reading_the_owner_never_stops_the_call(monkeypatch):
    def broken(db, workspace_id, caller_context):
        raise RuntimeError("no chat")

    monkeypatch.setattr(owner_turn_module, "owner_turn", broken)
    assert questions.refusal_on_a_question(None, WS, "platform_update_agent", CHAT) is None


def test_the_executor_refuses_the_dropbox_tool_before_any_gate_or_handler(owner_said):
    result = asyncio.run(PlatformActionExecutor(None, WS).execute(
        "platform_assign_tool_to_agent", {"agent_id": 348, "tool_name": "DROPBOX"}, CHAT))

    assert result["success"] is False and result["error"].startswith("The owner asked a question")


def test_the_persona_answers_a_question_and_does_a_request():
    from consumers.chatbot import personality
    from core.seeds import seed_auto_agent as seed
    from modules.context.sections.identity import _PERSONALITY_MAP

    line = "when you ask how, where or whether, I answer that first and offer to do it"
    assert line in personality._FRIENDLY_PERSONALITY and line in _PERSONALITY_MAP["friendly"]
    prompt = personality.AutomatosPersonality.get_base_system_prompt()
    assert "asked how, where or whether, I answer the question, offer to do it, and change nothing" in prompt
    assert line in prompt
    assert "when you ask how or where, I answer that first and offer to do it" in seed._FRIENDLY_FALLBACK
    assert seed._PREVIOUS_FRIENDLY_FALLBACK != seed._FRIENDLY_FALLBACK


@pytest.mark.parametrize("old", ["fallback", "with the doctrine"])
def test_a_row_on_the_earlier_friendly_default_is_lifted_to_the_new_one(old):
    from core.seeds import seed_auto_agent as seed

    persona = seed._PREVIOUS_FRIENDLY_FALLBACK
    if old == "with the doctrine":
        persona = seed.compose_persona_with_doctrine(persona)
    row = NS(custom_persona_prompt=persona, use_custom_persona=True, workspace_id=WS, slug="auto-x", configuration={})

    assert seed._backfill_auto_persona(row) == "updated"
    assert row.custom_persona_prompt == seed._default_persona()
