"""PRD-256 US-003 — memory takes receipts: what the turn did, told apart from what it said.

Every chat exchange is distilled into durable facts. The distiller read the reply alone, so
"I've created the ticket" after a call that failed, or after no call at all, could come back
next turn as a ticket that exists. The turn's receipts (built by the platform from the tool
tracker; the model never writes them) now ride the memory store, and the distiller reads them
as the record of actions, the reply as what was said.

The fake model here distils as the prompt tells it: from the record of actions when the prompt
has one, from the reply when it has none (what a model given only the reply does). These tests
call no real model, no provider and no database.
"""
from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

import core.llm as core_llm
from consumers.chatbot.receipts import build_receipts
from consumers.chatbot.smart_memory import SmartMemoryManager
from modules.memory.remembered_receipts import (
    NO_ACTIONS, RECORD_HEADING, RECORD_RULE, REPLY_LABEL, record_of_actions, the_record_beside_the_reply,
)
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
ASKED = "Make a ticket for the fridge repair, please."
SAID_MADE = "I've created the ticket for the fridge repair. It's now on your board."
CLAIMED_FACT = "A ticket was created for the fridge repair."
MADE = ("platform_execute", {"action": "platform_create_task", "params": {"title": "Fridge repair"}},
        {"success": True, "raw_result": {"success": True, "task_id": 422, "number": "#0422"}})
REFUSED = ("platform_execute", {"action": "platform_create_task", "params": {"title": "Fridge repair"}},
           {"success": False, "error": "title and description are required"})
CARD = re.compile(r"#\d+")


def _receipts(*calls):
    tracker = ToolExecutionTracker()
    for tool, args, result in calls:
        tracker.record_outcome(tool, args, result)
    return build_receipts(tracker)


def _distils_as_told(prompt: str):
    """The fake model: from the record of actions when the prompt carries one (a done write is a
    card by its number; a claim in the reply is not), else from the reply alone."""
    if RECORD_HEADING not in prompt:
        return [{"fact": CLAIMED_FACT, "type": "task_learning", "importance": 0.7}] if "I've created" in prompt else []
    record = prompt.split(RECORD_HEADING, 1)[1].split("\n\nUser: ", 1)[0]
    return [{"fact": f"Card {number} was created on the board for the fridge repair.", "type": "task_learning",
             "importance": 0.7}
            for line in record.splitlines() if "write done" in line for number in CARD.findall(line)]


class _Model:
    def __init__(self):
        self.prompts = []

    async def generate_response(self, messages, tools=None):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        return NS(content=json.dumps(_distils_as_told(prompt)))


class _Unified:
    """Records the L3 facts and the L2 transcript; no durable store."""

    def __init__(self):
        self.facts, self.transcripts = [], []

    async def store_two_tier(self, **kwargs):
        self.facts.append(" ".join(m["content"] for m in kwargs["messages"]))
        return [("global", {"success": True})]

    async def store_transcript(self, **kwargs):
        self.transcripts.append(kwargs)
        return "row-id"


@pytest.fixture
def model(monkeypatch):
    fake = _Model()
    monkeypatch.setattr(core_llm, "create_llm_manager", lambda **kwargs: fake)
    return fake


async def _remembered(receipts, answer=SAID_MADE):
    mgr = SmartMemoryManager()
    unified = _Unified()
    mgr._unified_service = unified
    ok = await mgr.store_conversation(workspace_id=WS, agent_id=1, user_message=ASKED, assistant_response=answer,
                                      chat_id="c1", receipts=receipts)
    return ok, unified


# ── the distilled facts ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_claimed_ticket_with_no_receipts_is_not_remembered(model):
    ok, unified = await _remembered([])

    assert ok is True
    assert unified.facts == []                       # no fact that a ticket was created
    assert NO_ACTIONS in model.prompts[-1]
    assert unified.transcripts                       # the verbatim turn is still kept in L2


@pytest.mark.asyncio
async def test_a_refused_create_task_is_not_remembered_as_a_ticket(model):
    ok, unified = await _remembered(_receipts(REFUSED))

    assert ok is True
    assert unified.facts == []
    assert "platform_create_task, write refused" in model.prompts[-1]
    assert "title and description are required" in model.prompts[-1]


@pytest.mark.asyncio
async def test_a_done_create_task_is_remembered_by_its_number(model):
    ok, unified = await _remembered(_receipts(MADE))

    assert ok is True
    assert len(unified.facts) == 1 and "#0422" in unified.facts[0]
    assert "platform_create_task, write done: #0422" in model.prompts[-1]


@pytest.mark.asyncio
async def test_without_receipts_the_distiller_reads_the_exchange_as_before(model):
    """A store that is not a chat turn's (no receipts) is distilled as it was."""
    mgr = SmartMemoryManager()
    mgr._unified_service = _Unified()
    await mgr.store_conversation(workspace_id=WS, agent_id=1, user_message=ASKED, assistant_response=SAID_MADE)

    prompt = model.prompts[-1]
    assert RECORD_HEADING not in prompt and REPLY_LABEL not in prompt
    assert prompt == SmartMemoryManager._build_distill_prompt(ASKED, SAID_MADE)


# ── the prompt: the record of actions, told apart from the reply ─────────────

@pytest.mark.asyncio
async def test_the_prompt_tells_the_record_from_the_reply(model):
    await _remembered(_receipts(MADE))
    prompt = model.prompts[-1]

    assert RECORD_RULE in prompt
    assert prompt.index(RECORD_HEADING) < prompt.index(f"User: {ASKED}") < prompt.index(REPLY_LABEL + SAID_MADE)
    assert f"Assistant: {SAID_MADE}" not in prompt   # the reply is labelled as what was said
    assert "Return ONLY a JSON array" in prompt       # the prompt's own rules (F316's too) are kept


def test_the_record_has_a_line_per_receipt_or_says_none_ran():
    assert record_of_actions([]) == f"{RECORD_HEADING}\n{NO_ACTIONS}"
    made, refused = _receipts(MADE) + _receipts(REFUSED)
    lines = record_of_actions([made, refused]).splitlines()[1:]
    assert lines[0].startswith("- platform_create_task, write done: #0422")
    assert lines[1].startswith("- platform_create_task, write refused") and "(title and description" in lines[1]


def test_a_prompt_without_its_exchange_is_kept_as_it_is():
    from modules.memory.remembered_receipts import with_the_record

    assert with_the_record("no exchange here", [], ASKED, SAID_MADE) == "no exchange here"


def test_outside_a_store_the_prompt_builder_is_unchanged():
    built = the_record_beside_the_reply(lambda u, a: f"rules\n\nUser: {u}\nAssistant: {a}\n")
    assert built(ASKED, SAID_MADE) == f"rules\n\nUser: {ASKED}\nAssistant: {SAID_MADE}\n"


# ── the wiring: the turn → SmartChatIntegration.store → store_exchange → store_conversation ──

@pytest.mark.asyncio
async def test_the_integration_and_the_orchestrator_pass_the_receipts_on():
    from consumers.chatbot.integration import SmartChatIntegration
    from consumers.chatbot.smart_orchestrator import SmartChatOrchestrator

    receipts = _receipts(MADE)
    integration = SmartChatIntegration.__new__(SmartChatIntegration)
    integration.orchestrator = NS(store_exchange=AsyncMock(return_value=True))
    assert await integration.store(ASKED, SAID_MADE, "c1", subject_id="user:7", receipts=receipts) is True
    assert integration.orchestrator.store_exchange.await_args.kwargs["receipts"] == receipts

    store_conversation = AsyncMock(return_value=True)
    fake = NS(workspace_id=WS, agent_id=1, widget_mode=False,
              memory_manager=NS(store_conversation=store_conversation, store_daily_summary=AsyncMock()),
              _unified_memory=MagicMock(update_session=AsyncMock()))
    await SmartChatOrchestrator.store_exchange(fake, ASKED, SAID_MADE, chat_id=None, receipts=receipts)
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
    await asyncio.gather(*pending, return_exceptions=True)
    assert store_conversation.call_args.kwargs["receipts"] == receipts


def _turn_service(store, widget=False):
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.widget_mode, svc.streaming_handler = widget, get_streaming_handler()
    svc._smart_chat = NS(store=store, orchestrator=NS(memory_manager=NS(_last_l3_facts_stored=1,
                                                                         _last_tier="global")))
    return svc


async def _drain(gen):
    return [chunk async for chunk in gen]


@pytest.mark.asyncio
async def test_the_turn_passes_its_receipts_and_its_answer_to_memory(monkeypatch):
    import consumers.chatbot.service as service

    receipts = _receipts(MADE)
    monkeypatch.setattr(service, "current_receipts", lambda: receipts)
    store = AsyncMock(return_value=True)

    frames = await _drain(_turn_service(store)._remember_the_turn(ASKED, SAID_MADE, "c1", 7))

    assert store.await_args.args == (ASKED, SAID_MADE, "c1")
    assert store.await_args.kwargs == {"subject_id": "user:7", "receipts": receipts}
    assert len(frames) == 1 and "memory-stored" in frames[0]      # the memory-stored event, as before


@pytest.mark.asyncio
async def test_a_widget_visitors_turn_still_stores_no_memory(monkeypatch):
    import consumers.chatbot.service as service

    monkeypatch.setattr(service, "current_receipts", lambda: [])
    store = AsyncMock(return_value=True)

    assert await _drain(_turn_service(store, widget=True)._remember_the_turn(ASKED, SAID_MADE, "c1", 7)) == []
    store.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_memory_store_does_not_break_the_turn(monkeypatch):
    import consumers.chatbot.service as service

    monkeypatch.setattr(service, "current_receipts", lambda: [])
    store = AsyncMock(side_effect=RuntimeError("memory down"))

    assert await _drain(_turn_service(store)._remember_the_turn(ASKED, SAID_MADE, "c1", None)) == []
    assert store.await_args.kwargs == {"subject_id": None, "receipts": []}
