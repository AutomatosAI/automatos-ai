"""F155 — a widget turn recalls only its agent's own memories.

A widget turn is an anonymous visitor's. The prompt's memory section tried the
Context Router first, which ignores the widget flag: it recalls the
workspace's short-term memory by meaning on every turn (the owner's other
conversations, PRD-187 S3) and, on the default path, the workspace's daily
logs. A visitor's question could pull the owner's conversations into the
answer. On a widget turn the section now skips the router, the daily logs and
a playbook's learnings from the owner's past runs, and recalls through the
SmartMemoryManager's widget lane: the agent's own memories, never the
workspace's.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

OWNER_CONVERSATION = "Owner said: keep the margin target at 40% until March"
OWNER_ACTIVITY = "Owner priced the Q4 plan with the accountant"
AGENT_MEMORY = "Visitors are answered in short paragraphs"


class _Router:
    def __init__(self):
        self.calls = []

    async def retrieve_context(self, **kwargs):
        self.calls.append(kwargs)
        return NS(long_term_memories=[], session_summary=None, daily_logs=OWNER_ACTIVITY,
                  temporal_results=[{"memory": OWNER_CONVERSATION}], knowledge_awareness=None,
                  total_tokens_estimate=40, signals=[])

    async def get_session(self, **kwargs):
        return None


class _SmartMemory:
    def __init__(self):
        self.recalls, self.daily_logs = [], 0

    async def retrieve_memories(self, **kwargs):
        self.recalls.append(kwargs)
        return NS(formatted_context=f"- {AGENT_MEMORY}", memories=[{"memory": AGENT_MEMORY}], user_context=None)

    async def get_daily_logs(self, **kwargs):
        self.daily_logs += 1
        return OWNER_ACTIVITY


def _render(monkeypatch, widget_mode):
    from modules.context.sections.base import SectionContext
    from modules.context.sections.memory import MemorySection

    router, smart = _Router(), _SmartMemory()
    monkeypatch.setattr("modules.memory.unified_memory_service.get_unified_memory_service", lambda: router)
    monkeypatch.setattr("consumers.chatbot.smart_memory.get_smart_memory_manager", lambda: smart)
    ctx = SectionContext(agent=NS(id=7), workspace_id=str(uuid4()), widget_mode=widget_mode,
                         kwargs={"query": "What is the margin target?", "chat_id": "widget-chat-1",
                                 "recipe_memories": {"summary": "Last run: the owner's supplier prices"}})
    return asyncio.run(MemorySection().render(ctx)), router, smart


def test_a_widget_turn_recalls_only_its_agents_memories(monkeypatch):
    text, router, smart = _render(monkeypatch, widget_mode=True)
    assert router.calls == []
    assert [recall["widget_mode"] for recall in smart.recalls] == [True]
    assert smart.daily_logs == 0
    assert AGENT_MEMORY in text
    for owners in (OWNER_CONVERSATION, OWNER_ACTIVITY, "supplier prices"):
        assert owners not in text


def test_a_dashboard_turn_still_recalls_through_the_context_router(monkeypatch):
    text, router, _smart = _render(monkeypatch, widget_mode=False)
    assert len(router.calls) == 1
    assert OWNER_CONVERSATION in text and "supplier prices" in text
