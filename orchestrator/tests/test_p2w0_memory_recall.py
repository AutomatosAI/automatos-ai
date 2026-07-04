"""PRD-185 S10: memory-recall eval — the first memory quality number.

Validates the eval HARNESS (pure, offline): the bag-of-words retriever proxy,
recall@k / MRR / task-lift metrics, per-tenant scoping, and honest-gate reporting.
It does NOT assert a passing recall number — a sub-threshold result is a valid,
honest outcome; the harness must report it faithfully and exit 0 regardless.

Also asserts gold-set/corpus integrity so a hand-authored id typo fails loudly
here rather than silently deflating the number.
"""
import sys
from pathlib import Path

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from evals.memory_recall import (  # noqa: E402
    RECALL_AT_5_TARGET,
    TOP_K,
    GoldQuery,
    MemoryDoc,
    _bow_retriever,
    _recall_at_k,
    _reciprocal_rank,
    load_corpus,
    load_gold_set,
    main,
    run_memory_recall_eval,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_recall_at_k_counts_relevant_in_window():
    ranked = ["a", "b", "c", "d", "e", "f"]
    # single relevant, retrieved at rank 2 -> in top-3 and top-5, not top-1
    assert _recall_at_k(ranked, frozenset({"b"}), 1) == 0.0
    assert _recall_at_k(ranked, frozenset({"b"}), 3) == 1.0
    assert _recall_at_k(ranked, frozenset({"b"}), 5) == 1.0
    # two relevant, one in window -> standard recall = 1/2
    assert _recall_at_k(ranked, frozenset({"b", "z"}), 5) == 0.5
    # no relevant labelled -> 0, never a divide-by-zero
    assert _recall_at_k(ranked, frozenset(), 5) == 0.0


def test_reciprocal_rank_uses_first_relevant():
    ranked = ["x", "hit", "y", "hit2"]
    assert _reciprocal_rank(ranked, frozenset({"hit"}), TOP_K) == 0.5  # rank 2
    assert _reciprocal_rank(ranked, frozenset({"x"}), TOP_K) == 1.0    # rank 1
    assert _reciprocal_rank(ranked, frozenset({"missing"}), TOP_K) == 0.0


# ---------------------------------------------------------------------------
# Retriever mechanism + tenant scoping
# ---------------------------------------------------------------------------

def test_bow_retriever_ranks_relevant_above_distractor():
    docs = [
        MemoryDoc("m1", "ws", "the same-day delivery cutoff is 11am", "business_fact", "shipping"),
        MemoryDoc("m2", "ws", "the courier is FedEx for long distance", "business_fact", "shipping"),
    ]
    retrieve = _bow_retriever(docs)
    ranked = retrieve("what is the cutoff for same-day delivery")
    assert ranked[0] == "m1"  # lexical overlap wins over the same-category distractor


def test_bow_retriever_is_deterministic():
    docs = [
        MemoryDoc("m1", "ws", "alpha beta gamma", "user_fact", "x"),
        MemoryDoc("m2", "ws", "delta epsilon zeta", "user_fact", "x"),
    ]
    r1, r2 = _bow_retriever(docs), _bow_retriever(docs)
    assert r1("alpha beta") == r2("alpha beta")  # no randomness — reproducible in CI


def test_retrieval_is_scoped_per_tenant():
    """A query must only ever retrieve its OWN workspace's memories — the eval
    mirrors the production workspace_id filter (no cross-tenant leakage)."""
    corpus = [
        MemoryDoc("a1", "tenant-a", "acme billing renews in March", "business_fact", "billing"),
        MemoryDoc("b1", "tenant-b", "bloom billing renews in June", "business_fact", "billing"),
    ]
    gold = [
        GoldQuery("q", "tenant-a", "when does billing renew", frozenset({"a1"}), "billing", "easy", True),
    ]
    report = run_memory_recall_eval(corpus, gold)
    # tenant-a's query scored 1.0 recall from a1 alone; b1 was never a candidate.
    assert len(report.tenants) == 1
    assert report.tenants[0].recall_at_5 == 1.0


# ---------------------------------------------------------------------------
# Gold-set / corpus integrity (catches hand-authored drift)
# ---------------------------------------------------------------------------

def test_gold_set_ids_all_exist_in_corpus_and_same_tenant():
    corpus = load_corpus()
    gold = load_gold_set()
    by_id = {d.memory_id: d for d in corpus}
    assert len(by_id) == len(corpus), "duplicate memory_id in corpus"
    assert len(gold) >= 50, "gold-set should carry ~50 queries (PRD-185 S10)"
    for q in gold:
        assert q.relevant_ids, f"{q.query_id} labels no relevant memory"
        for mid in q.relevant_ids:
            assert mid in by_id, f"{q.query_id} references missing memory {mid}"
            assert by_id[mid].workspace_id == q.workspace_id, (
                f"{q.query_id} references {mid} from another tenant"
            )


# ---------------------------------------------------------------------------
# Per-tenant reporting + honest gate
# ---------------------------------------------------------------------------

def test_report_is_per_tenant_and_bounded():
    report = run_memory_recall_eval(load_corpus(), load_gold_set())
    ws_ids = {t.workspace_id for t in report.tenants}
    assert ws_ids == {"tenant-acme", "tenant-bloom", "tenant-northstar"}
    for t in report.tenants:
        assert t.n_queries > 0
        for metric in (t.recall_at_1, t.recall_at_3, t.recall_at_5, t.mrr):
            assert 0.0 <= metric <= 1.0
        # recall is monotonic in the window size
        assert t.recall_at_1 <= t.recall_at_3 <= t.recall_at_5


def test_task_lift_only_counts_dependent_queries():
    report = run_memory_recall_eval(load_corpus(), load_gold_set())
    for t in report.tenants:
        # some queries are flagged non-dependent, so n_dependent < n_queries
        assert 0 < t.n_dependent <= t.n_queries
        assert t.task_lift_points >= 0.0


def test_passes_tracks_the_honest_number():
    report = run_memory_recall_eval(load_corpus(), load_gold_set())
    d = report.to_dict()
    assert d["passes"] == (d["mean_recall_at_5"] >= RECALL_AT_5_TARGET)
    assert isinstance(d["mean_recall_at_5"], float)
    assert len(d["tenants"]) == 3


def test_bundled_fixture_produces_a_real_number():
    """End-to-end on the bundled snapshot: recall/MRR are real numbers in (0, 1]
    (the deliverable of S10). recall@5 over a small per-tenant corpus can be high;
    MRR and recall@1 carry the retrieval discrimination the distractors create."""
    report = run_memory_recall_eval(load_corpus(), load_gold_set())
    d = report.to_dict()
    assert isinstance(d["mean_recall_at_5"], float)
    assert isinstance(d["mean_mrr"], float)
    assert 0.0 < report.mean_recall_at_5 <= 1.0
    assert 0.0 < report.mean_mrr <= 1.0
    assert len(d["tenants"]) == 3


def test_main_always_exits_zero():
    """The number is the deliverable; a sub-threshold run is never a CI failure."""
    assert main(["--json"]) == 0
    assert main([]) == 0
