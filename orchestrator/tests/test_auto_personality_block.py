"""Auto's personality block is the preset written for its prompt.

The PromptRegistry rows seeded for chatbot-friendly/-professional/-technical are
templates ({agent_name}, {agent_role}, {tools_list}). The base prompt inserted
them unformatted, so every workspace on a preset carried the placeholders and a
second "You are ..." line.
"""
import pytest

from consumers.chatbot.personality import AutomatosPersonality


@pytest.mark.parametrize("mode", ["friendly", "professional", "technical"])
def test_no_template_placeholder_reaches_the_prompt(mode):
    settings = {"personality_mode": mode, "custom_soul": "", "communication_style": "balanced"}
    prompt = AutomatosPersonality.get_base_system_prompt(agent_name="Auto", orchestrator_settings=settings)
    assert not any(slot in prompt for slot in ("{agent_name}", "{agent_role}", "{tools_list}"))
    assert "**My personality:**" in prompt
