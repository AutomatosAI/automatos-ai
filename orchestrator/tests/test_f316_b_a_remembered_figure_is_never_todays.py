"""F316 (night 9b): Auto gave its own earlier answers back as today's figures: "2 … I retrieved
that information from our previous conversation" (7c726e13), order 5207 "from my memory … stored
on October 4th" (11b9a74a). Each chat exchange is distilled into durable facts ("preserve
specifics … numbers"), and the next chat's prompt carried them under "What You Know About This
User" with nothing to say they were old. Now the memory block opens with the rule that a
remembered figure is not today's, and the distiller leaves counted figures out.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch

from consumers.chatbot.atom_prompt import atom_memory_block
from consumers.chatbot.smart_memory import SmartMemoryManager
from modules.context.remembered_figures import DISTILL_RULE, REMEMBERED_RULE
from modules.context.sections.base import SectionContext
from modules.context.sections.memory import MemorySection

REMEMBERED = "- Between April and September, 2 Harvest Club members cancelled; the most common reason was too much coffee."
HEADING = "## What You Know About This User"


def _ctx():
    return SectionContext(agent=NS(id=322, name="Auto"), workspace_id="ws",
                          messages=[{"role": "user", "content": "How many cancelled April to September?"}], kwargs={})


def test_the_chat_memory_block_says_a_remembered_figure_is_not_todays():
    section = MemorySection()
    with patch.object(MemorySection, "_try_context_router", return_value=None), \
            patch.object(MemorySection, "_build_from_smart_memory", return_value=f"{HEADING}\n\n{REMEMBERED}"):
        rendered = asyncio.run(section.render(_ctx()))

    assert rendered.startswith(f"{HEADING}\n{REMEMBERED_RULE}\n")
    assert rendered.index(REMEMBERED_RULE) < rendered.index(REMEMBERED)
    assert "is not today's figure: count it again" in REMEMBERED_RULE
    assert "What the owner told you (a change, a correction, a decision) stands" in REMEMBERED_RULE


def test_no_memory_is_still_no_block():
    with patch.object(MemorySection, "_try_context_router", return_value=None), \
            patch.object(MemorySection, "_build_from_smart_memory", return_value=""):
        assert asyncio.run(MemorySection().render(_ctx())) == ""


def test_the_short_path_memory_block_says_it_too():
    async def retrieve_memories(**_kwargs):
        return NS(formatted_context=REMEMBERED, memories=[{"memory": REMEMBERED}])

    orchestrator = NS(memory_manager=NS(retrieve_memories=retrieve_memories))
    block = asyncio.run(atom_memory_block(orchestrator, [{"role": "user", "content": "How many cancelled?"}],
                                          workspace_id="ws", agent_id=322, widget_mode=False,
                                          viewer_subject_id=None))

    assert REMEMBERED_RULE in block and block.index(REMEMBERED_RULE) < block.index(REMEMBERED)


def test_the_distiller_leaves_counted_figures_out():
    prompt = SmartMemoryManager._build_distill_prompt(
        "How many cancelled April to September?", "Between April and September, 2 Harvest Club members cancelled.")

    assert DISTILL_RULE in prompt
    assert prompt.index(DISTILL_RULE) < prompt.index("Return ONLY a JSON array")
    assert "is not a durable fact" in DISTILL_RULE and "a change or a correction" in DISTILL_RULE
