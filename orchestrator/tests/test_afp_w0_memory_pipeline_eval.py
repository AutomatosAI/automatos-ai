"""#1430 Wave 0: the memory-pipeline eval's self-tests (evals/memory_pipeline.py).

The eval runs ``evals.memory_recall``'s corpus and gold set through the real chat-turn
path: the distillation write into a workspace, then ``ContextRouter.retrieve_context``
for an agent's turn. These tests hold the harness honest: its in-memory store applies
Qdrant's filters, it is deterministic, a namespace regression scores zero (the bug
#1434 fixed), its look-alikes are other facts of the same shape, and it never writes to
a database.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import evals.memory_pipeline as mp
from core.qdrant.models import FieldCondition, Filter, MatchAny, MatchValue
from evals.memory_recall import load_corpus, load_gold_set, run_memory_recall_eval


def _report(service):
    loop = asyncio.new_event_loop()
    try:
        factory = mp.chat_turn_retriever_factory(service, loop.run_until_complete)
        return run_memory_recall_eval(load_corpus(), load_gold_set(), factory)
    finally:
        loop.close()


# ── the in-memory store ──────────────────────────────────────────────────────────


def test_the_in_memory_store_applies_qdrants_filter_clauses():
    payload = {"namespace": "mem:w", "metadata": {"owner": "user:7"}}
    is_ns = FieldCondition(key="namespace", match=MatchValue(value="mem:w"))
    either = FieldCondition(key="namespace", match=MatchAny(any=["mem:x", "mem:w"]))
    owner = FieldCondition(key="metadata.owner", match=MatchValue(value="user:7"))
    stranger = FieldCondition(key="metadata.owner", match=MatchValue(value="user:8"))
    assert mp.matches(payload, Filter(must=[is_ns, owner]))
    assert mp.matches(payload, Filter(must=[either], should=[stranger, owner]))
    assert not mp.matches(payload, Filter(must=[is_ns], should=[stranger]))
    assert not mp.matches(payload, Filter(must=[is_ns], must_not=[owner]))
    assert mp.matches(payload, None)


def test_the_hash_embedder_is_deterministic_and_unit_length():
    embed = mp.HashEmbedder()
    a = asyncio.run(embed.generate_embedding("Acme renews every March"))
    assert a == asyncio.run(embed.generate_embedding("Acme renews every March"))
    assert len(a) == mp.OFFLINE_DIMENSION and abs(sum(v * v for v in a) - 1.0) < 1e-9


# ── the chat-turn path ────────────────────────────────────────────────────────────


def test_every_tenant_recalls_through_the_chat_turn_path_and_the_run_is_repeatable():
    first, second = _report(mp.offline_service()), _report(mp.offline_service())
    assert all(t.recall_at_5 > 0 for t in first.tenants)
    assert [t.recall_at_5 for t in first.tenants] == [t.recall_at_5 for t in second.tenants]


def test_an_agent_only_read_scores_zero_so_the_eval_catches_that_regression():
    """Before #1434 the router read only the agent's namespace; distilled facts sit in the workspace's."""
    service = mp.offline_service()
    search = service.search_long_term

    async def agent_only(*args, include_workspace=False, **kwargs):
        return await search(*args, **kwargs)

    service.search_long_term = agent_only
    assert _report(service).mean_recall_at_5 == 0.0


def test_a_query_ranks_only_its_own_tenants_facts():
    service, loop = mp.offline_service(), asyncio.new_event_loop()
    try:
        factory = mp.chat_turn_retriever_factory(service, loop.run_until_complete)
        corpus = load_corpus()
        acme = factory([d for d in corpus if d.workspace_id == "tenant-acme"])
        factory([d for d in corpus if d.workspace_id == "tenant-bloom"])
        ranked = acme("Who owns the flower shop and how do they take wedding enquiries?")
    finally:
        loop.close()
    assert ranked and all(mid.startswith("acme-") for mid in ranked)


def test_the_eval_writes_no_substrate_metric(monkeypatch):
    import core.observability.substrate_metrics as metrics

    def refuse(**_):
        raise AssertionError("the eval wrote a substrate metric")

    monkeypatch.setattr(metrics, "record_substrate_search_nowait", refuse)
    assert _report(mp.offline_service()).mean_recall_at_5 > 0


# ── look-alikes ───────────────────────────────────────────────────────────────────


def test_a_lookalike_is_another_fact_of_the_same_shape():
    corpus = load_corpus()
    names = mp.proper_names(corpus)
    acme = next(d for d in corpus if d.memory_id == "acme-01")
    twin = mp.lookalike(acme, 1, names)
    assert twin.memory_id == "acme-01~1" and twin.workspace_id == acme.workspace_id
    assert "Acme" not in twin.text and "March 3rd" in twin.text      # renamed, the date moved, an ordinal kept
    assert {"March", "Monday"}.isdisjoint(names)


def test_with_lookalikes_adds_only_facts_that_differ():
    """A fact with no name or number to change gets no look-alike: a copy is not another fact."""
    corpus = load_corpus()
    names = mp.proper_names(corpus)
    changeable = [d for d in corpus if mp.lookalike(d, 1, names).text != d.text]
    scaled = mp.with_lookalikes(corpus, 8)
    assert 0 < len(changeable) < len(corpus)
    assert len(scaled) == len(corpus) + 8 * len(changeable) == 414
    assert len({d.memory_id for d in scaled}) == len({d.text for d in scaled}) == len(scaled)


# ── the command ───────────────────────────────────────────────────────────────────


def test_the_command_publishes_both_numbers_and_exits_zero(capsys):
    assert mp.main(["--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["facts"] == len(load_corpus())
    assert set(result) >= {"store", "proxy", "chat_turn"}
    assert set(result["chat_turn"]["slices"]) == {"core", "continuity"}


@pytest.mark.parametrize("missing", ["QDRANT_URL", "OPENROUTER_API_KEY"])
def test_live_says_what_it_needs(monkeypatch, missing):
    from config import config

    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.test")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(config, missing, "")
    with pytest.raises(SystemExit, match="--live needs"):
        mp.live_service(mp.LIVE_EMBEDDING_MODEL, mp.LIVE_DIMENSION)


def test_live_embeddings_never_go_through_the_usage_tracker():
    """The platform's embedding provider records each call as LLM usage in the configured
    database; the eval's own embedder calls OpenRouter directly and records nothing."""
    import inspect

    source = inspect.getsource(mp.OpenRouterEmbedder) + inspect.getsource(mp.live_service)
    assert "record_embedding_usage" not in source and "OpenRouterEmbeddingProvider" not in source
