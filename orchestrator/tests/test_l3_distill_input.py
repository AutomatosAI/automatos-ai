"""L3 input curation — distill durable facts before feeding Mem0.

The chat path used to send the raw user+assistant exchange to L3 (Mem0). Mem0's
default *server-side* extraction then produced thin, episodic facts like
"User requested to fire a mission…" / "User was informed that…" — interaction
logs, not durable knowledge.

The fix curates the L3 input *in the orchestrator*: distil 0..N durable facts
from the exchange first, and:
  - non-empty facts → store those to L3,
  - ``[]`` (nothing durable) → skip L3 entirely,
  - ``None`` (LLM/parse failure) → fall back to the raw exchange so a transient
    outage never drops the memory.

In every case the verbatim transcript is still dual-written to L2 (Postgres),
so the literal conversation is preserved for the Memory Explorer.

These tests use a fake LLM manager (no network) and a recording fake of the
UnifiedMemoryService (no Mem0, no DB).
"""
import os
import sys
import types
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Some imports down-chain build config; never touches a real DB in these tests.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

import core.llm as core_llm  # noqa: E402  (patch target for create_llm_manager)

# consumers/__init__.py eagerly imports the chatbot stack → RAG → camelot, an
# optional PDF dep that isn't installed in the test env. Stub it so the import
# chain resolves without the real package.
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from consumers.chatbot.smart_memory import SmartMemoryManager  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """Returns a canned ``.content`` (or raises) from generate_response."""

    def __init__(self, content_or_exc):
        self._content_or_exc = content_or_exc
        self.calls = []

    async def generate_response(self, messages, tools=None):
        self.calls.append(messages)
        if isinstance(self._content_or_exc, Exception):
            raise self._content_or_exc
        return _FakeResp(self._content_or_exc)


def _patch_llm(monkeypatch, content_or_exc) -> _FakeLLM:
    """Swap create_llm_manager so the distiller talks to a fake LLM."""
    fake = _FakeLLM(content_or_exc)
    monkeypatch.setattr(core_llm, "create_llm_manager", lambda **kwargs: fake)
    return fake


class _FakeUnified:
    """Records store_two_tier (L3) and store_transcript (L2) calls."""

    def __init__(self):
        self.two_tier_calls = []
        self.transcript_calls = []

    async def store_two_tier(self, **kwargs):
        self.two_tier_calls.append(kwargs)
        return [("global", {"success": True})]

    async def store_transcript(self, **kwargs):
        self.transcript_calls.append(kwargs)
        return "row-id"


# ---------------------------------------------------------------------------
# _distill_durable_facts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distill_parses_json_array_of_facts(monkeypatch):
    fake = _patch_llm(
        monkeypatch,
        '["InBuildUK is a UK smoke-ventilation contractor.", '
        '"Blog posts should cite EN 12101-2, EN 12101-6 and EN 12101-10."]',
    )
    mgr = SmartMemoryManager()
    facts = await mgr._distill_durable_facts(
        "We do smoke ventilation for UK contractors",
        "Noted — I'll cite EN 12101-2/-6/-10 in posts.",
        workspace_id="ws1",
        agent_id=3,
    )
    assert facts == [
        "InBuildUK is a UK smoke-ventilation contractor.",
        "Blog posts should cite EN 12101-2, EN 12101-6 and EN 12101-10.",
    ]
    # The distiller actually invoked the LLM once.
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_distill_returns_empty_list_when_nothing_durable(monkeypatch):
    _patch_llm(monkeypatch, "[]")
    mgr = SmartMemoryManager()
    facts = await mgr._distill_durable_facts(
        "fire a mission to write the blog post",
        "OK, firing the mission now.",
        workspace_id="ws1",
        agent_id=None,
    )
    assert facts == []


@pytest.mark.asyncio
async def test_distill_tolerates_prose_around_the_json(monkeypatch):
    # Models sometimes wrap the array in prose / code fences.
    _patch_llm(
        monkeypatch,
        'Here are the durable facts:\n```json\n["User prefers British English."]\n```',
    )
    mgr = SmartMemoryManager()
    facts = await mgr._distill_durable_facts(
        "Please always use British spelling.",
        "Will do.",
        workspace_id="ws1",
        agent_id=None,
    )
    assert facts == ["User prefers British English."]


@pytest.mark.asyncio
async def test_distill_returns_none_on_llm_error(monkeypatch):
    _patch_llm(monkeypatch, RuntimeError("llm provider down"))
    mgr = SmartMemoryManager()
    facts = await mgr._distill_durable_facts(
        "anything", "anything", workspace_id="ws1", agent_id=None
    )
    assert facts is None


@pytest.mark.asyncio
async def test_distill_returns_none_on_unparseable_output(monkeypatch):
    _patch_llm(monkeypatch, "I could not find any facts, sorry.")
    mgr = SmartMemoryManager()
    facts = await mgr._distill_durable_facts(
        "anything", "anything", workspace_id="ws1", agent_id=None
    )
    assert facts is None


# ---------------------------------------------------------------------------
# store_conversation routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_conversation_feeds_distilled_facts_to_l3(monkeypatch):
    _patch_llm(monkeypatch, '["User prefers concise blog posts."]')
    mgr = SmartMemoryManager()
    fake = _FakeUnified()
    mgr._unified_service = fake

    ok = await mgr.store_conversation(
        workspace_id="ws1",
        agent_id=3,
        user_message="I like my blog posts kept concise and skimmable",
        assistant_response="Understood, I'll keep them concise from now on.",
    )

    assert ok is True
    # L3 received exactly one store, and its content is the distilled fact —
    # NOT the raw assistant response.
    assert len(fake.two_tier_calls) == 1
    l3_text = " ".join(m["content"] for m in fake.two_tier_calls[0]["messages"])
    assert "User prefers concise blog posts." in l3_text
    assert "Understood, I'll keep them concise" not in l3_text
    # L2 transcript still preserved verbatim.
    assert len(fake.transcript_calls) == 1


@pytest.mark.asyncio
async def test_store_conversation_skips_l3_when_nothing_durable(monkeypatch):
    _patch_llm(monkeypatch, "[]")
    mgr = SmartMemoryManager()
    fake = _FakeUnified()
    mgr._unified_service = fake

    ok = await mgr.store_conversation(
        workspace_id="ws1",
        agent_id=3,
        user_message="fire a mission to create the EN 12101 blog post",
        assistant_response="OK, firing the mission now.",
    )

    # Nothing durable → L3 is skipped entirely (no episodic noise stored).
    assert len(fake.two_tier_calls) == 0
    # But the raw transcript is still preserved in L2.
    assert len(fake.transcript_calls) == 1
    assert ok is True


@pytest.mark.asyncio
async def test_store_conversation_falls_back_to_raw_on_distill_error(monkeypatch):
    _patch_llm(monkeypatch, RuntimeError("llm provider down"))
    mgr = SmartMemoryManager()
    fake = _FakeUnified()
    mgr._unified_service = fake

    ok = await mgr.store_conversation(
        workspace_id="ws1",
        agent_id=3,
        user_message="Some substantive message about the project",
        assistant_response="A substantive assistant reply worth keeping.",
    )

    # On distill failure we fall back to the raw exchange so memory isn't lost.
    assert ok is True
    assert len(fake.two_tier_calls) == 1
    l3_text = " ".join(m["content"] for m in fake.two_tier_calls[0]["messages"])
    assert "A substantive assistant reply worth keeping." in l3_text
    # L2 transcript still written.
    assert len(fake.transcript_calls) == 1


@pytest.mark.asyncio
async def test_store_conversation_still_skips_trivial_before_distilling(monkeypatch):
    # Trivial greetings must short-circuit BEFORE any LLM call.
    fake_llm = _patch_llm(monkeypatch, '["should not be reached"]')
    mgr = SmartMemoryManager()
    fake = _FakeUnified()
    mgr._unified_service = fake

    ok = await mgr.store_conversation(
        workspace_id="ws1",
        agent_id=3,
        user_message="hi",
        assistant_response="Hello! How can I help?",
    )

    assert ok is False
    assert len(fake_llm.calls) == 0
    assert len(fake.two_tier_calls) == 0
    assert len(fake.transcript_calls) == 0
