"""F297 (night 8, the agents' prompts): a plain board card's answer goes on the card.

Mission and playbook steps were told where their answer goes (F269, night 7b); a plain
card's prompt never was. Night 8's cards got a description of a saved file instead of
the work (#0256, #0412), a tool's error as the answer's first line ("I am unable to
write the file, as there seems to be an issue with the `file_path` parameter.", #0346;
"I seem to have made a mistake in calling the `run_code` tool…", #0360) and a raw
Composio error as the whole answer (#0254).
"""
from __future__ import annotations

import asyncio

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
BRIEF = ("Reply to Ines Duarte, club member, about her new address: 14 Quay Street, Clevedon. "
         "Draft only, do not send.")
NUMBERS = "Tuesday's numbers (pasted in)\n\nTell me how many orders and how much we took."
GUIDE = "Refunds or anything with money: say Gerard will sort it personally."


def test_where_the_answer_goes_says_what_to_do_when_a_tool_fails():
    from services.step_lessons import ON_THE_CARD

    assert ON_THE_CARD.startswith("## Where your answer goes\n")
    assert "If a tool fails, never put its error" in ON_THE_CARD and "say plainly what is missing" in ON_THE_CARD


def test_a_plain_cards_launch_ends_its_prompt_with_where_the_answer_goes():
    from services.step_lessons import ON_THE_CARD, a_cards_answer_goes_on_the_card

    launched = {}
    a_cards_answer_goes_on_the_card(lambda **kwargs: launched.update(kwargs))(
        task_id=346, agent_id=7, workspace_id=WS, prompt=BRIEF, review_mode="human")

    assert launched["prompt"] == f"{BRIEF}\n\n{ON_THE_CARD}"
    assert (launched["task_id"], launched["review_mode"]) == (346, "human")


def test_every_launch_of_a_plain_card_runs_through_it():
    """The dispatcher's claim, Run now on the board and Auto's move to In progress all
    launch a card through ``_launch_task_execution``."""
    from api import board_tasks

    assert board_tasks._launch_task_execution.__wrapped__.__name__ == "_launch_task_execution"


@pytest.fixture
def guides(monkeypatch):
    import consumers.chatbot.knowledge_prefetch as knowledge_prefetch
    import modules.tools.tool_router as tool_router
    from config import config

    searched = []

    class _Router:
        async def execute_and_format(self, tool_name, tool_args, **kwargs):
            searched.append(tool_args["query"])
            return {"raw_result": {"results": [{"filename": "harbourline-brand-voice.md", "similarity": 0.82,
                                                "content": GUIDE}]}, "frontend_data": None}

    monkeypatch.setattr(knowledge_prefetch, "documents_in", lambda db, ws: 3)
    monkeypatch.setattr(tool_router, "get_tool_router", lambda: _Router())
    monkeypatch.setattr(type(config), "CHATBOT_KNOWLEDGE_PREFETCH", True, raising=False)
    monkeypatch.setattr(type(config), "KNOWLEDGE_PREFETCH_PASSAGES", 5, raising=False)
    monkeypatch.setattr(type(config), "KNOWLEDGE_PREFETCH_MIN_SCORE", 0.5, raising=False)
    return searched


def test_a_drafts_guide_search_reads_the_brief_not_the_boards_rules(guides):
    from services.draft_guides import guides_for_draft
    from services.step_lessons import ON_THE_CARD

    prompt = asyncio.run(guides_for_draft(None, WS, 330, f"{BRIEF}\n\n{ON_THE_CARD}"))

    (query,) = guides
    assert "14 Quay Street" in query and "Where your answer goes" not in query and "tool fails" not in query
    assert prompt.startswith(f"{BRIEF}\n\n{ON_THE_CARD}") and GUIDE in prompt


def test_a_card_that_is_not_a_draft_keeps_its_prompt_and_is_not_searched(guides):
    from services.draft_guides import guides_for_draft
    from services.step_lessons import ON_THE_CARD

    prompt = f"{NUMBERS}\n\n{ON_THE_CARD}"
    assert asyncio.run(guides_for_draft(None, WS, 329, prompt)) == prompt and guides == []
