"""F155: a widget turn is told where it is, not what the owner's platform can do.

The chat identity gave every widget turn Auto's owner-facing platform skill,
even when a custom agent answered. That meant: "I am Auto", create agents,
install skills, 100+ connected integrations, analytics, Mission Zero, and "My
tools are real… I call the tool and do it". It also gave the memory
instruction ("what to store via platform_store_memory"), though a widget turn
stores no memory (F154). The gate refused those calls, so the model
overclaimed and the refusals were noisy, and visitors were told which flows
and integrations the owner's platform has. On a widget turn both are left
out. One line says where the turn is, true for every key: no scope lets a
widget change the owner's agents, playbooks, settings or connected apps.
Base identity, persona, tool guidance, response style and anti-patterns stay.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

from core.security.surface import WIDGET, turn_surface
from modules.context.sections.base import SectionContext

PLATFORM_SKILL = "## Platform Skill"
MEMORY_INSTRUCTION = "## Memory — What to Remember"


def _identity():
    from modules.context.sections.identity import IdentitySection

    ctx = SectionContext(agent=NS(id=7, name="Club Concierge", description="Answers members' questions.",
                                  use_custom_persona=False, custom_persona_prompt=None, persona=None),
                         workspace_id="ws-1", kwargs={"personality": True, "orchestrator_settings": {"tone": "warm"}})
    return asyncio.run(IdentitySection().render(ctx))


def test_a_widget_turn_is_told_where_it_is_not_what_the_owners_platform_can_do():
    from modules.context.sections.identity import WIDGET_CHAT_NOTE

    with turn_surface(WIDGET, ("chat",), None):
        visitor = _identity()
    assert PLATFORM_SKILL not in visitor and MEMORY_INSTRUCTION not in visitor
    assert visitor.rstrip().endswith(WIDGET_CHAT_NOTE.strip())  # the last thing the identity says
    assert "Answers members' questions." in visitor  # the agent's own description stays

    owner = _identity()
    assert PLATFORM_SKILL in owner and MEMORY_INSTRUCTION in owner and WIDGET_CHAT_NOTE.strip() not in owner
