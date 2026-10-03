"""The identity block leads Auto's cached prompt prefix (PRD-201 S4), so nothing
in it may change from one turn to the next.

After F025 fixed the tool enum and the clock, the first call of most turns still
read 0 cached tokens: the identity block said how many messages the conversation
had, and greeted by the hour.
"""
import inspect

from consumers.chatbot.personality import AutomatosPersonality

SETTINGS = {"personality_mode": "friendly", "custom_soul": "", "communication_style": "balanced"}


def test_the_base_prompt_takes_nothing_that_moves_per_turn():
    assert "msg_count" not in inspect.signature(AutomatosPersonality.get_base_system_prompt).parameters


def test_the_base_prompt_is_the_same_on_every_turn():
    first = AutomatosPersonality.get_base_system_prompt(user_name="Sam", agent_name="Auto", orchestrator_settings=SETTINGS)
    second = AutomatosPersonality.get_base_system_prompt(user_name="Sam", agent_name="Auto", orchestrator_settings=SETTINGS)
    assert first == second
    assert "messages so far" not in first
    assert not any(word in first for word in ("Good morning", "Good afternoon", "Good evening"))
