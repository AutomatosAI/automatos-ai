"""L3 input curation — distil TYPED durable facts before the L3 write (PRD-159 S1).

The chat path used to send the raw user+assistant exchange to L3. The old fork's
default *server-side* extraction then produced thin, episodic facts like
"User requested to fire a mission…" — interaction logs, not durable knowledge.

PRD-159 S1 rewrites this:
  - the distiller emits TYPED ``{fact, type, importance}`` objects over the
    operational taxonomy (tool_outcome / task_learning / playbook_pattern /
    user_fact / business_fact / preference / procedure),
  - it runs on the cheap model tier (``config.MEMORY_DISTILL_MODEL``),
  - non-empty facts → store each with its typed metadata (category + importance),
  - ``[]`` (nothing durable) → skip L3 entirely,
  - ``None`` (LLM/parse failure) → store NOTHING (no raw-exchange fallback — that
    fallback was the "user said hello" source). The L2 transcript still keeps
    the verbatim turn either way.

These tests use a fake LLM manager (no network) and a recording fake of the
UnifiedMemoryService (no durable store, no DB).
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

from consumers.chatbot.smart_memory import (  # noqa: E402
    SmartMemoryManager,
    MEMORY_FACT_TYPES,
    DEFAULT_FACT_TYPE,
)


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
        self.factory_kwargs = {}

    async def generate_response(self, messages, tools=None):
        self.calls.append(messages)
        if isinstance(self._content_or_exc, Exception):
            raise self._content_or_exc
        return _FakeResp(self._content_or_exc)


def _patch_llm(monkeypatch, content_or_exc) -> _FakeLLM:
    """Swap create_llm_manager so the distiller talks to a fake LLM.

    Captures the factory kwargs (incl. ``model``) on the returned fake so tests
    can assert the cheap-tier routing.
    """
    fake = _FakeLLM(content_or_exc)

    def _factory(**kwargs):
        fake.factory_kwargs = kwargs
        return fake

    monkeypatch.setattr(core_llm, "create_llm_manager", _factory)
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
# Golden suite — typed parse (strict on type + presence, fuzzy on wording)
# ---------------------------------------------------------------------------

# (llm_json_response, expected [(fact_substr, type)]) — one fixture per taxonomy
# kind plus mixed/zero-write cases. Wording is fuzzy (substring); type + presence
# are strict.
GOLDEN = [
    # business_fact
    ('[{"fact": "InBuildUK is a UK smoke-ventilation contractor.", '
     '"type": "business_fact", "importance": 0.9}]',
     [("smoke-ventilation contractor", "business_fact")]),
    # preference
    ('[{"fact": "The user prefers British English spelling.", '
     '"type": "preference", "importance": 0.7}]',
     [("British English", "preference")]),
    # tool_outcome
    ('[{"fact": "SLACK_SEND_MESSAGE failed with not_in_channel for #ops.", '
     '"type": "tool_outcome", "importance": 0.6}]',
     [("not_in_channel", "tool_outcome")]),
    # task_learning
    ('[{"fact": "The deploy mission failed because the alembic migration was '
     'not applied first.", "type": "task_learning", "importance": 0.8}]',
     [("alembic migration", "task_learning")]),
    # playbook_pattern
    ('[{"fact": "Blog posts work best as draft then cite standards then review.", '
     '"type": "playbook_pattern", "importance": 0.5}]',
     [("cite standards", "playbook_pattern")]),
    # procedure
    ('[{"fact": "To publish, push to main then run the deploy playbook.", '
     '"type": "procedure", "importance": 0.8}]',
     [("run the deploy playbook", "procedure")]),
    # user_fact
    ('[{"fact": "The user is named Gerard.", "type": "user_fact", '
     '"importance": 0.9}]',
     [("Gerard", "user_fact")]),
    # mixed multi-fact
    ('[{"fact": "Posts must cite EN 12101-2.", "type": "business_fact", '
     '"importance": 0.8}, {"fact": "The user prefers concise posts.", '
     '"type": "preference", "importance": 0.6}]',
     [("EN 12101-2", "business_fact"), ("concise", "preference")]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_json,expected", GOLDEN)
async def test_golden_typed_facts(monkeypatch, llm_json, expected):
    _patch_llm(monkeypatch, llm_json)
    mgr = SmartMemoryManager()
    facts = await mgr._distill_durable_facts(
        "transcript", "reply", workspace_id="ws1", agent_id=3
    )
    assert facts is not None
    assert len(facts) == len(expected)
    for got, (sub, ftype) in zip(facts, expected):
        assert sub.lower() in got["fact"].lower()      # fuzzy on wording
        assert got["type"] == ftype                    # strict on type
        assert ftype in MEMORY_FACT_TYPES
        assert 0.0 <= got["importance"] <= 1.0


# 'user said hello'-class fixtures must produce ZERO durable facts.
ZERO_WRITE = [
    "[]",                                  # model returns nothing durable
    "  []  ",                              # with whitespace
    '```json\n[]\n```',                    # fenced empty
]


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_json", ZERO_WRITE)
async def test_zero_write_fixtures_yield_no_facts(monkeypatch, llm_json):
    _patch_llm(monkeypatch, llm_json)
    mgr = SmartMemoryManager()
    facts = await mgr._distill_durable_facts(
        "user said hello", "Hi there!", workspace_id="ws1", agent_id=None
    )
    assert facts == []


# ---------------------------------------------------------------------------
# _distill_durable_facts — parsing + routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distill_routes_to_cheap_model_tier(monkeypatch):
    """The distiller must run on config.MEMORY_DISTILL_MODEL (cheap tier)."""
    from config import config
    fake = _patch_llm(monkeypatch, "[]")
    mgr = SmartMemoryManager()
    await mgr._distill_durable_facts(
        "x", "y", workspace_id="ws1", agent_id=None
    )
    assert fake.factory_kwargs.get("model") == config.MEMORY_DISTILL_MODEL
    assert fake.factory_kwargs.get("request_type") == "memory_distill"


@pytest.mark.asyncio
async def test_distill_tolerates_prose_and_fences(monkeypatch):
    _patch_llm(
        monkeypatch,
        'Here are the facts:\n```json\n'
        '[{"fact": "User prefers British English.", "type": "preference", '
        '"importance": 0.7}]\n```',
    )
    mgr = SmartMemoryManager()
    facts = await mgr._distill_durable_facts(
        "Please always use British spelling.", "Will do.",
        workspace_id="ws1", agent_id=None,
    )
    assert facts == [
        {"fact": "User prefers British English.", "type": "preference",
         "importance": 0.7}
    ]


@pytest.mark.asyncio
async def test_distill_unknown_type_falls_back_to_default(monkeypatch):
    _patch_llm(
        monkeypatch,
        '[{"fact": "Something durable.", "type": "made_up_kind", '
        '"importance": 5}]',
    )
    mgr = SmartMemoryManager()
    facts = await mgr._distill_durable_facts(
        "x", "y", workspace_id="ws1", agent_id=None
    )
    assert facts[0]["type"] == DEFAULT_FACT_TYPE        # unknown → default
    assert facts[0]["importance"] == 1.0                # clamped into [0,1]


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


def test_distill_prompt_has_no_exclusion_and_lists_taxonomy():
    """The transient-event exclusion is DELETED; the taxonomy is present."""
    prompt = SmartMemoryManager._build_distill_prompt("u", "a")
    assert "Do NOT record transient interaction" not in prompt
    for t in MEMORY_FACT_TYPES:
        assert t in prompt


# ---------------------------------------------------------------------------
# store_conversation routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_conversation_feeds_typed_fact_to_l3(monkeypatch):
    _patch_llm(
        monkeypatch,
        '[{"fact": "User prefers concise blog posts.", "type": "preference", '
        '"importance": 0.6}]',
    )
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
    # L3 received exactly one store; content is the distilled fact, NOT the raw
    # assistant response; metadata carries the typed category + importance.
    assert len(fake.two_tier_calls) == 1
    call = fake.two_tier_calls[0]
    l3_text = " ".join(m["content"] for m in call["messages"])
    assert "User prefers concise blog posts." in l3_text
    assert "Understood, I'll keep them concise" not in l3_text
    assert call["metadata"]["category"] == "preference"
    assert call["metadata"]["importance"] == 0.6
    # L2 transcript still preserved verbatim.
    assert len(fake.transcript_calls) == 1


@pytest.mark.asyncio
async def test_store_conversation_writes_one_l3_row_per_typed_fact(monkeypatch):
    _patch_llm(
        monkeypatch,
        '[{"fact": "Posts cite EN 12101-2.", "type": "business_fact", '
        '"importance": 0.8}, {"fact": "User prefers concise posts.", '
        '"type": "preference", "importance": 0.6}]',
    )
    mgr = SmartMemoryManager()
    fake = _FakeUnified()
    mgr._unified_service = fake

    ok = await mgr.store_conversation(
        workspace_id="ws1", agent_id=3,
        user_message="Cite EN 12101-2 and keep posts concise please",
        assistant_response="Got it.",
    )

    assert ok is True
    # Two typed facts → two L3 writes, each with its own category.
    assert len(fake.two_tier_calls) == 2
    cats = {c["metadata"]["category"] for c in fake.two_tier_calls}
    assert cats == {"business_fact", "preference"}


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

    # Nothing durable → L3 skipped entirely (no episodic noise stored).
    assert len(fake.two_tier_calls) == 0
    # But the raw transcript is still preserved in L2.
    assert len(fake.transcript_calls) == 1
    assert ok is True


@pytest.mark.asyncio
async def test_store_conversation_stores_nothing_to_l3_on_distill_error(monkeypatch):
    """PRD-159 S1: distill failure stores NOTHING to L3 (no raw fallback)."""
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

    # No raw-exchange fallback — L3 gets nothing on distill failure.
    assert len(fake.two_tier_calls) == 0
    # The verbatim turn is still preserved in L2.
    assert len(fake.transcript_calls) == 1
    assert ok is True


@pytest.mark.asyncio
async def test_store_conversation_still_skips_trivial_before_distilling(monkeypatch):
    # Trivial greetings must short-circuit BEFORE any LLM call.
    fake_llm = _patch_llm(monkeypatch, '[{"fact": "x", "type": "user_fact", "importance": 0.5}]')
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
