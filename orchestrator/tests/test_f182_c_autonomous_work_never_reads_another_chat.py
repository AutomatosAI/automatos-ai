"""F182 (night 6) — autonomous work never reads another conversation as its facts.

Run 207's step 1 (agent 325, the Analyst) wrote "Hello Maya … Lamplight Café"
for a run meant for Gull & Anchor, and ticket #1119 wrote to Maya again. The
Lamplight text was memory_short_term a61e1034, agent 324's chat from 02:40
("Lamplight Café in Clifton, the owner is Maya Osei…"). It was recalled 13
times that night by the Context Router, whose L2 recall is always on and spans
the whole workspace. A playbook step (RECIPE) and an agent run for a ticket,
a mission task, a trigger or a schedule (TASK_EXECUTION) now recall the owner's
curated workspace memories, their agent's own and their playbook's own
learnings. They get no raw chat transcripts (L2, or promoted into L3) and no
daily logs. What they can look up, the router's awareness block, stays. A
chat turn, and a person's message on a channel, recall as before.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from modules.context.sections.base import SectionContext
from modules.context.sections.memory import MemorySection

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
LAMPLIGHT = ("User: Hi Auto. A new café wants to start buying from us wholesale — Lamplight Café in Clifton, the "
             "owner is Maya Osei (maya@lamplight-cafe.example).")
DAILY = "[03:53] Discussed: That's the Lamplight one — it's to Maya, not Priya."
CURATED = "- Harbourline's wholesale minimum order is 3 kg; orders by Wednesday noon go out Thursday."
AWARE = "## What You Can Look Up\n- 2 documents: the brand voice guide and the wholesale price list."
STEP_1 = "Draft a welcome email for the new café with our wholesale prices."


@pytest.fixture
def recall(monkeypatch):
    """The Context Router (L2 always on) and SmartMemoryManager, as night 6 had them."""
    import consumers.chatbot.smart_memory as smart_memory
    import modules.memory.unified_memory_service as unified
    from modules.memory.context_router import ContextBundle, ContextRouter

    bundle = ContextBundle(
        long_term_memories=({"memory": "Prefers short emails.", "metadata": {}},),
        temporal_results=({"content": LAMPLIGHT, "content_type": "transcript", "agent_id": 324},),
        daily_logs=DAILY, knowledge_awareness=AWARE)
    router = SimpleNamespace(retrieve_context=AsyncMock(return_value=bundle))
    manager = SimpleNamespace(
        retrieve_memories=AsyncMock(return_value=SimpleNamespace(formatted_context=CURATED, user_context=None,
                                                                 memories=[{"memory": CURATED}])),
        get_daily_logs=AsyncMock(return_value=DAILY))
    monkeypatch.setattr(unified, "get_unified_memory_service", lambda: router)
    monkeypatch.setattr(smart_memory, "get_smart_memory_manager", lambda: manager)
    monkeypatch.setattr(ContextRouter, "build_knowledge_awareness", AsyncMock(return_value=AWARE))
    return SimpleNamespace(router=router, manager=manager)


def _render(mode, **kwargs):
    ctx = SectionContext(agent=SimpleNamespace(id=325), workspace_id=WS, context_mode=mode,
                         kwargs={"query": STEP_1, **kwargs})
    return asyncio.run(MemorySection().render(ctx))


@pytest.mark.parametrize("mode", ["recipe", "task_execution"], ids=["run-207-step-1", "ticket-1119"])
def test_autonomous_work_never_reads_agent_324s_chat(recall, mode):
    rendered = _render(mode)

    assert "Lamplight" not in rendered and "Maya" not in rendered
    assert CURATED in rendered and AWARE in rendered
    recall.router.retrieve_context.assert_not_awaited()
    recall.manager.get_daily_logs.assert_not_awaited()
    kwargs = recall.manager.retrieve_memories.await_args.kwargs
    assert (kwargs["widget_mode"], kwargs["chat_transcripts"]) == (False, False)   # workspace + agent tiers


def test_the_playbooks_own_learnings_stay(recall):
    rendered = _render("recipe", recipe_memories={"summary": "Last run: the café wanted Thursday deliveries."})
    assert rendered.endswith("## Learnings from Previous Runs\nLast run: the café wanted Thursday deliveries.")


@pytest.mark.parametrize("mode, kwargs", [("chatbot", {}), ("task_execution", {"conversation": True})],
                         ids=["chat-turn", "channel-message"])
def test_a_conversation_recalls_as_before(recall, mode, kwargs):
    rendered = _render(mode, **kwargs)

    recall.router.retrieve_context.assert_awaited_once()
    assert LAMPLIGHT in rendered and DAILY in rendered


def test_a_channel_message_is_a_conversation():
    import channels.base as channel
    from modules.agents.factory import agent_factory

    assert "conversation=True" in inspect.getsource(channel)
    body = inspect.getsource(agent_factory.AgentFactory._execute_with_prompt_scoped)
    assert "conversation=conversation," in body


# ── a transcript promoted into L3 ───────────────────────────────────────────

PROMOTED = {"memory": LAMPLIGHT, "score": 0.9, "metadata": {"category": "transcript", "promoted_from_l2": "a61e1034"}}
FACT = {"memory": "Wholesale minimum order is 3 kg.", "score": 0.8, "metadata": {"type": "business_fact"}}


def _manager():
    from consumers.chatbot.smart_memory import SmartMemoryManager

    manager = SmartMemoryManager()
    manager._unified_service = SimpleNamespace(
        search_long_term=AsyncMock(side_effect=lambda ws, q, agent_id=None, limit=8: [dict(PROMOTED), dict(FACT)]
                                   if agent_id is None else []))
    return manager


def test_autonomous_work_leaves_a_promoted_transcript_out():
    result = asyncio.run(_manager().retrieve_memories(WS, 325, STEP_1, chat_transcripts=False))
    assert [m["memory"] for m in result.memories] == [FACT["memory"]]


def test_a_chat_turns_cached_recall_is_never_served_to_autonomous_work():
    manager = _manager()
    chat = asyncio.run(manager.retrieve_memories(WS, 325, STEP_1))
    work = asyncio.run(manager.retrieve_memories(WS, 325, STEP_1, chat_transcripts=False))
    assert LAMPLIGHT in [m["memory"] for m in chat.memories]
    assert LAMPLIGHT not in [m["memory"] for m in work.memories]
