"""PRD-159 S5 — honest memory_stored signal (orchestrator side).

The memory_stored SSE must fire ONLY after durable facts actually persisted to
L3, with the real tier. SmartMemoryManager exposes ``_last_l3_facts_stored``
(0 on zero-fact turns) + ``_last_tier`` which the streaming layer gates on.
(The retired fork's UPDATE/DELETE → SQL-view sync no longer applies — PRD-187 S1.)
"""
import os
import sys
import types
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

import core.llm as core_llm  # noqa: E402
sys.modules.setdefault("camelot", types.ModuleType("camelot"))
from consumers.chatbot.smart_memory import SmartMemoryManager  # noqa: E402


class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    async def generate_response(self, messages, tools=None):
        return _FakeResp(self._content)


class _FakeUnified:
    def __init__(self):
        self.two_tier_calls = []

    async def store_two_tier(self, **kwargs):
        self.two_tier_calls.append(kwargs)
        return [("global", {"success": True})]

    async def store_transcript(self, **kwargs):
        return "row-id"


def _patch(monkeypatch, content):
    monkeypatch.setattr(core_llm, "create_llm_manager", lambda **kw: _FakeLLM(content))


@pytest.mark.asyncio
async def test_no_facts_means_no_sse_signal(monkeypatch):
    _patch(monkeypatch, "[]")          # nothing durable
    mgr = SmartMemoryManager()
    mgr._unified_service = _FakeUnified()
    await mgr.store_conversation(
        workspace_id="ws1", agent_id=3,
        user_message="fire a mission to do the thing",
        assistant_response="OK, firing now.",
    )
    # Zero durable facts persisted → the streaming layer must NOT emit.
    assert mgr._last_l3_facts_stored == 0


@pytest.mark.asyncio
async def test_facts_stored_sets_count_and_tier(monkeypatch):
    _patch(
        monkeypatch,
        '[{"fact": "User prefers concise posts.", "type": "preference", '
        '"importance": 0.6}]',
    )
    mgr = SmartMemoryManager()
    mgr._unified_service = _FakeUnified()
    await mgr.store_conversation(
        workspace_id="ws1", agent_id=3,
        user_message="keep my posts concise please",
        assistant_response="Will do.",
    )
    assert mgr._last_l3_facts_stored == 1
    assert isinstance(mgr._last_tier, str) and mgr._last_tier


@pytest.mark.asyncio
async def test_distill_failure_means_no_sse_signal(monkeypatch):
    # LLM raises → distill returns None → S1 stores nothing → no SSE.
    def _raise(**kw):
        class _Boom:
            async def generate_response(self, *a, **k):
                raise RuntimeError("down")
        return _Boom()
    monkeypatch.setattr(core_llm, "create_llm_manager", _raise)
    mgr = SmartMemoryManager()
    mgr._unified_service = _FakeUnified()
    await mgr.store_conversation(
        workspace_id="ws1", agent_id=3,
        user_message="a substantive message",
        assistant_response="a substantive reply",
    )
    assert mgr._last_l3_facts_stored == 0


# --- PRD-159 S5: no 'both'-tier double-write -------------------------------

def test_tier_classification_never_returns_both():
    mgr = SmartMemoryManager()
    samples = [
        ("My name is Gerard", "Noted."),
        ("I prefer concise blog posts", "Will do."),
        ("Use the #ops slack channel", "OK."),          # '#' must NOT force agent/both
        ("email me at x@y.com", "OK."),                  # '@' must NOT force agent/both
        ("random chit chat", "sure"),
    ]
    for um, ar in samples:
        assert mgr._classify_memory_tier(um, ar) in ("global", "agent")


def test_default_tier_is_single_global():
    mgr = SmartMemoryManager()
    # Ordinary message → single workspace namespace, not a double-write.
    assert mgr._classify_memory_tier("here is a durable fact", "ok") == "global"


def test_explicit_agent_instruction_routes_to_agent():
    mgr = SmartMemoryManager()
    assert mgr._classify_memory_tier("always cc my manager", "ok") == "agent"
    assert mgr._classify_memory_tier("for this agent, use formal tone", "ok") == "agent"
