"""PRD-142 Wave 2 · WS-I / W2-S9 — chat & reasoning-entry tests.

This is the single biggest coverage hole in the platform: the code that
decides *what the platform does with every message* had no direct test. The
three deciders are all deterministic before any LLM call, so their verdict
tables can be pinned exactly:

* ``AutoBrain.assess`` — complexity (ATOM…ORGANISM) + action
  (RESPOND/DELEGATE/MISSION). Tier 1 is a Redis cache, Tier 2 is free regex
  heuristics, Tier 3 is the LLM. We disable the cache and exercise the Tier-2
  verdict table directly; the Tier-3 fall-through is asserted with the LLM
  call stubbed (no spend, no network).
* ``IntentClassifier`` (core, universal-router Tier 2c) — category + action
  from rule-based patterns.
* ``SmartIntentClassifier`` (chatbot pipeline) — Intent enum + tool/memory
  flags.

The last block proves the two classifiers do **not** cross-contaminate: same
input, independent pipelines, type-distinct results. See
[[intent-classifier-disambiguation]].
"""

from __future__ import annotations

import os
import sys
import types

# auto.py → core.database/config pulls Postgres env at import. Seed harmless
# defaults; nothing in this module touches a real DB or network.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

# consumers/__init__.py eagerly imports the chatbot stack → RAG → camelot, an
# optional PDF table-extraction dep that isn't installed in the test env. Stub
# it so the import chain resolves (same pattern as test_l3_distill_input.py).
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402

from consumers.chatbot.auto import (  # noqa: E402
    Action,
    AutoBrain,
    Complexity,
    ComplexityAssessment,
)
from consumers.chatbot.intent_classifier import (  # noqa: E402
    Intent,
    IntentResult,
    SmartIntentClassifier,
)
from core.services.intent_classifier import (  # noqa: E402
    IntentClassification,
    IntentClassifier,
)


# --------------------------------------------------------------- helpers


def _brain() -> AutoBrain:
    """AutoBrain with the Redis cache forced to a miss/no-op.

    Tier 1 must not shadow the Tier-2 heuristics we're exercising, and no test
    here should touch a real Redis. Overriding the two cache methods on the
    instance is bulletproof regardless of whether a client was constructed.
    """
    brain = AutoBrain(db=MagicMock(), workspace_id="ws-test")
    brain._redis = None
    brain._cache_lookup = lambda *a, **k: None
    brain._cache_store = lambda *a, **k: None
    return brain


# ============================================================ AutoBrain
# Tier 2 verdict table — deterministic, no LLM.


@pytest.mark.asyncio
async def test_assess_empty_message_is_atom_respond():
    out = await _brain().assess("   ")
    assert out.complexity is Complexity.ATOM
    assert out.action is Action.RESPOND
    assert out.confidence == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "hello",
        "Hi Auto",
        "morning auto",
        "thanks!",
        "what can you do?",
        "tell me a joke",
        "who are you",
    ],
)
async def test_assess_chitchat_is_atom_respond(message):
    out = await _brain().assess(message)
    assert out.complexity is Complexity.ATOM, message
    assert out.action is Action.RESPOND, message
    assert out.needs_memory is False
    assert out.tool_hints == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message,expected_tool",
    [
        ("list my agents", "platform_list_agents"),
        ("show my playbooks", "platform_list_recipes"),
        ("token usage", "platform_get_llm_usage"),
        ("what models are available", "platform_list_llms"),
    ],
)
async def test_assess_platform_query_is_molecule_with_tool(message, expected_tool):
    out = await _brain().assess(message)
    assert out.complexity is Complexity.MOLECULE, message
    assert out.action is Action.RESPOND, message
    assert out.tool_hints == ["platform"]
    assert expected_tool in out.matched_tools, (message, out.matched_tools)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "my name is Gerard",
        "what did we say about the launch date",
    ],
)
async def test_assess_memory_recall_is_cell_needs_memory(message):
    """Passive self-disclosure / conversational recall → CELL, needs_memory.

    These reach the CELL tier because they match the memory-recall pattern but
    *not* any platform-tool pattern. Contrast with explicit memory-*search*
    phrasing, which is intercepted earlier — see the precedence test below.
    """
    out = await _brain().assess(message)
    assert out.complexity is Complexity.CELL, message
    assert out.action is Action.RESPOND, message
    assert out.needs_memory is True, message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "do you remember when we discussed the Q4 budget",
        "what do you remember about my preferences",
        "recall when we set the budget",
    ],
)
async def test_assess_memory_search_phrasing_routes_to_platform_tool(message):
    """Precedence guard: explicit memory-*search* phrasing is a tool call.

    "do you remember…", "what do you remember…", "recall…" overlap the
    CELL memory-recall regex, but the MOLECULE platform-query tier is checked
    *first*, so they route to the ``platform_search_memory`` tool (an active
    search) rather than passive CELL recall. This ordering is intentional and
    easy to break by reordering the heuristics — pin it.
    """
    out = await _brain().assess(message)
    assert out.complexity is Complexity.MOLECULE, message
    assert out.action is Action.RESPOND, message
    assert out.needs_memory is False, message
    assert out.tool_hints == ["platform"], message
    assert "platform_search_memory" in out.matched_tools, (message, out.matched_tools)


@pytest.mark.asyncio
async def test_assess_falls_through_to_llm_when_no_heuristic_matches():
    """A message that hits no Tier-2 pattern must reach Tier 3 (the LLM),
    and assess must return whatever the LLM tier produced."""
    brain = _brain()
    sentinel = ComplexityAssessment(
        complexity=Complexity.ORGAN,
        action=Action.MISSION,
        reasoning="stub",
        confidence=0.7,
    )
    brain._llm_classify = AsyncMock(return_value=sentinel)

    out = await brain.assess(
        "research our top 3 competitors, write a positioning report, "
        "and email it to the leadership team"
    )

    assert out is sentinel
    brain._llm_classify.assert_awaited_once()


@pytest.mark.asyncio
async def test_assess_heuristic_path_never_calls_the_llm():
    """Cost guard: a greeting must be resolved for free — the LLM tier is
    never invoked on a Tier-2 hit."""
    brain = _brain()
    brain._llm_classify = AsyncMock(
        side_effect=AssertionError("LLM must not be called for chitchat")
    )

    out = await brain.assess("hey there")

    assert out.complexity is Complexity.ATOM
    brain._llm_classify.assert_not_awaited()


@pytest.mark.asyncio
async def test_assess_tier1_cache_short_circuits_before_heuristics():
    """A cache hit returns immediately without running the heuristics."""
    brain = AutoBrain(db=MagicMock(), workspace_id="ws-test")
    cached = ComplexityAssessment(
        complexity=Complexity.ORGANISM,
        action=Action.MISSION,
        reasoning="cached",
        confidence=0.99,
    )
    brain._cache_lookup = lambda *a, **k: cached
    brain._run_fast_heuristics = MagicMock(
        side_effect=AssertionError("heuristics must be skipped on a cache hit")
    )

    out = await brain.assess("hello")  # would be ATOM via heuristics

    assert out is cached
    brain._run_fast_heuristics.assert_not_called()


# ====================================================== IntentClassifier
# Core / universal-router Tier 2c — category + action.


@pytest.mark.parametrize(
    "query,category",
    [
        ("send an email to John", "EMAIL"),
        ("what's on my calendar tomorrow", "CALENDAR"),
        ("open a pull request on github", "CODE"),
        ("post a message to the slack channel", "COMMUNICATION"),
        ("generate an image of a logo", "IMAGE"),
        ("run a nl2sql query on the database", "DATABASE"),
    ],
)
def test_core_intent_classifier_category(query, category):
    out = IntentClassifier().classify(query)
    assert isinstance(out, IntentClassification)
    assert out.category == category, (query, out.category)


def test_core_intent_classifier_action_create_and_delete():
    assert IntentClassifier().classify("send an email to John").action_type == "CREATE"
    assert (
        IntentClassifier().classify("delete the calendar event").action_type
        == "DELETE"
    )


def test_core_intent_classifier_unmatched_is_general_low_confidence():
    out = IntentClassifier().classify("qwerty zxcv")
    assert out.category == "GENERAL"
    assert out.action_type == "UNKNOWN"
    assert out.confidence < 0.5


# ================================================ SmartIntentClassifier
# Chatbot pipeline — Intent enum + tool/memory flags.


@pytest.mark.parametrize(
    "query,intent,requires_tools,requires_memory",
    [
        ("hi there", Intent.GREETING, False, False),
        ("send an email to the team", Intent.EXTERNAL_ACTION, True, False),
        ("my name is Gerard", Intent.MEMORY_RECALL, False, True),
        ("how many agents do I have", Intent.DATA_QUERY, True, False),
    ],
)
def test_smart_intent_classifier_table(query, intent, requires_tools, requires_memory):
    out = SmartIntentClassifier().classify(query)
    assert isinstance(out, IntentResult)
    assert out.primary_intent is intent, (query, out.primary_intent)
    assert out.requires_tools is requires_tools, query
    assert out.requires_memory is requires_memory, query


def test_smart_intent_classifier_empty_is_simple_chitchat():
    out = SmartIntentClassifier().classify("")
    assert out.primary_intent is Intent.CHITCHAT
    assert out.is_simple is True
    assert out.requires_tools is False


# ================================================ no cross-contamination


def test_two_classifiers_are_distinct_pipelines():
    """The universal-router classifier and the chatbot classifier are
    separate classes with type-distinct results and disjoint vocabularies.
    Same input → independent classification, no shared state."""
    assert IntentClassifier is not SmartIntentClassifier

    query = "send an email to John"
    core = IntentClassifier().classify(query)
    smart = SmartIntentClassifier().classify(query)

    # Type-distinct results.
    assert isinstance(core, IntentClassification)
    assert isinstance(smart, IntentResult)

    # Disjoint vocabularies — one speaks "category", the other "primary_intent".
    assert hasattr(core, "category") and not hasattr(core, "primary_intent")
    assert hasattr(smart, "primary_intent") and not hasattr(smart, "category")

    # Both recognise the email action, in their own terms.
    assert core.category == "EMAIL"
    assert smart.primary_intent is Intent.EXTERNAL_ACTION
