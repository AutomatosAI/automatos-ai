"""Wave 2: ``evals.lean_turn`` holds a lean turn's prompt overhead target in CI.

The offline run builds the sample lean agent's real ``LEAN`` context, describes its
granted tools as a turn does, and measures what the platform adds beyond the agent's
own context (§5.3: within 2,000 tokens). The full turn is measured beside it.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def lean_on(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "LEAN_TURNS_ENABLED", True)


def test_the_lean_turns_overhead_is_within_its_target(lean_on):
    from config import config
    from evals import lean_turn

    result = lean_turn.run(False, None, 0, config.DEFAULT_WORKSPACE_ID)
    assert result["overhead_met"], result
    assert 0 < result["overhead_tokens"] <= lean_turn.OVERHEAD_TARGET_TOKENS
    assert result["lean_tools"] >= 1
    assert result["lean_system_tokens"] < result["full_system_tokens"] / 4


def test_live_needs_a_model():
    from evals import lean_turn

    with pytest.raises(SystemExit):
        lean_turn.main(["--live"])


def test_the_live_run_checks_the_facts_the_question_needs_were_recalled():
    from evals import lean_turn

    block = "\n".join(text for memory_id, text in lean_turn.MEMORIES if memory_id in ("due-date", "plan"))
    assert lean_turn.recalled(block) == ["due-date", "plan"]
    assert not set(lean_turn.RECALL_GOLD) <= set(lean_turn.recalled(block))  # move-date is missing
    assert lean_turn.recalled("") == []


def test_the_live_embedder_passes_the_memory_stores_health_check():
    from core.llm.embedding_status import provider_ready
    from evals.memory_pipeline import OpenRouterEmbedder

    assert provider_ready(OpenRouterEmbedder("any/model", "not-a-key", "http://127.0.0.1:9", 8))
