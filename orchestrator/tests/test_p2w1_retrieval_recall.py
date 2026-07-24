"""PRD-188 S5: retrieval-recall eval — the RAG quality number.

Validates the eval HARNESS (pure, offline): the dense bag-of-words proxy, the
BM25 leg, the weighted-RRF hybrid, document-level recall@k / MRR, per-tenant
scoping, the phrasing-sensitivity slice (Internal Audit v1's NL-vs-keyword
failure mode), and honest-gate reporting. It does NOT assert a passing recall
number — a sub-threshold result is a valid, honest outcome; the harness must
report it faithfully and exit 0 regardless.

Also asserts gold-set/corpus integrity so a hand-authored id typo fails loudly
here rather than silently deflating the number.
"""
import json
import sys
from pathlib import Path

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from evals.retrieval_recall import (  # noqa: E402
    HYBRID_KEYWORD_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    OFFLINE_VARIANTS,
    PHRASING_STYLES,
    RETRIEVAL_RECALL_AT_5_TARGET,
    TOP_K,
    ChunkDoc,
    GoldQuery,
    RetrievalRecallReport,
    VariantTenantResult,
    _recall_at_k,
    _reciprocal_rank,
    _weighted_rrf,
    bm25_retriever,
    dense_proxy_retriever,
    documents_from_chunk_ranking,
    load_corpus,
    load_gold_set,
    main,
    run_retrieval_recall_eval,
)


def _chunk(cid, ws, doc, text, category="general"):
    return ChunkDoc(chunk_id=cid, workspace_id=ws, document_id=doc, text=text, category=category)


def _query(qid, ws, query, relevant, phrasing="natural", pair_id=None):
    return GoldQuery(
        query_id=qid,
        workspace_id=ws,
        query=query,
        relevant_doc_ids=frozenset(relevant),
        category="general",
        difficulty="easy",
        phrasing=phrasing,
        pair_id=pair_id,
    )


# ---------------------------------------------------------------------------
# Metrics (documents)
# ---------------------------------------------------------------------------

def test_recall_at_k_counts_relevant_in_window():
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert _recall_at_k(ranked, frozenset({"b"}), 1) == 0.0
    assert _recall_at_k(ranked, frozenset({"b"}), 3) == 1.0
    assert _recall_at_k(ranked, frozenset({"b", "z"}), 5) == 0.5
    assert _recall_at_k(ranked, frozenset(), 5) == 0.0  # never divide-by-zero


def test_reciprocal_rank_uses_first_relevant():
    ranked = ["x", "hit", "y", "hit2"]
    assert _reciprocal_rank(ranked, frozenset({"hit"}), TOP_K) == 0.5
    assert _reciprocal_rank(ranked, frozenset({"x"}), TOP_K) == 1.0
    assert _reciprocal_rank(ranked, frozenset({"missing"}), TOP_K) == 0.0


def test_documents_from_chunk_ranking_first_appearance_wins():
    chunk_to_doc = {"c1": "docA", "c2": "docB", "c3": "docA", "c4": "docC"}
    # docA appears at ranks 1 and 3 — deduped to its first appearance.
    assert documents_from_chunk_ranking(["c1", "c2", "c3", "c4"], chunk_to_doc) == [
        "docA",
        "docB",
        "docC",
    ]
    # Ids absent from the map pass through as document ids — live mode ranks
    # real document ids the fixture map has never seen.
    assert documents_from_chunk_ranking(["nope", "c2"], chunk_to_doc) == ["nope", "docB"]


def test_documents_from_chunk_ranking_live_document_id_passthrough():
    # Live retrievers return document ids directly (build_live_variants); with
    # a fixture-only map every non-empty id must survive, deduped, in order —
    # before the passthrough, live runs collapsed to zero recall.
    live_ranked = ["doc-live-1", "", "doc-live-1", "doc-live-2"]
    assert documents_from_chunk_ranking(live_ranked, {}) == ["doc-live-1", "doc-live-2"]


# ---------------------------------------------------------------------------
# The two offline legs + fusion
# ---------------------------------------------------------------------------

def test_bm25_ranks_exact_term_chunk_first():
    chunks = [
        _chunk("c-err", "ws", "docA", "ERR-SYNC-409 means the warehouse rejected a stale stock update."),
        _chunk("c-other", "ws", "docB", "Shipping zones ship DPD next day for mainland orders."),
        _chunk("c-more", "ws", "docC", "Returns are accepted within thirty days of delivery."),
    ]
    ranked = bm25_retriever(chunks)("ERR-SYNC-409")
    assert ranked[0] == "c-err"


def test_dense_proxy_is_deterministic_with_tiebreak():
    chunks = [
        _chunk("c-b", "ws", "docB", "identical text"),
        _chunk("c-a", "ws", "docA", "identical text"),
    ]
    ranked = dense_proxy_retriever(chunks)("identical text")
    # Equal cosine — deterministic chunk_id tie-break, run-to-run stable.
    assert ranked == ["c-a", "c-b"]


def test_weighted_rrf_hand_computed_order_and_weights_are_real():
    # weights 0.7/0.3, k=60:
    #   x: 0.7/60 + 0.3/61 = 0.016585   y: 0.7/61 + 0.3/60 = 0.016475
    fused = _weighted_rrf([["x", "y", "z"], ["y", "x"]], [0.7, 0.3], k=60)
    assert fused == ["x", "y", "z"]
    # Flipping the weights flips the winner — the knobs are real, not decorative.
    fused_flipped = _weighted_rrf([["x", "y", "z"], ["y", "x"]], [0.3, 0.7], k=60)
    assert fused_flipped[0] == "y"


def test_hybrid_beats_single_leg_on_mixed_queries():
    """A corpus where the lexical leg wins an exact-term query and the dense
    proxy wins a paraphrase — the fused variant must surface the right doc in
    top-1 for BOTH (the reason hybrid exists)."""
    chunks = [
        _chunk("c-sku", "ws", "doc-sku", "SKU MT-4417-BLK Aurora parka black fill weight 700."),
        _chunk("c-ret", "ws", "doc-ret", "Customers may return items within thirty days for a refund to the original payment method."),
        _chunk("c-noise", "ws", "doc-noise", "Quarterly planning notes cover hiring, budget and roadmap."),
    ]
    hybrid = OFFLINE_VARIANTS["hybrid_rrf"](chunks)
    chunk_to_doc = {c.chunk_id: c.document_id for c in chunks}
    assert documents_from_chunk_ranking(hybrid("MT-4417-BLK"), chunk_to_doc)[0] == "doc-sku"
    assert documents_from_chunk_ranking(hybrid("send something back refund money"), chunk_to_doc)[0] == "doc-ret"


# ---------------------------------------------------------------------------
# Driver: tenant scoping + phrasing slice
# ---------------------------------------------------------------------------

def test_retrieval_is_tenant_scoped():
    """A query only ranks its own workspace's chunks — a perfect cross-tenant
    text match must NOT surface (mirrors build_retrieval_filters fail-closed)."""
    corpus = [_chunk("c-a", "ws-a", "doc-a", "the secret onboarding checklist")]
    gold = [_query("q1", "ws-b", "secret onboarding checklist", ["doc-a"])]
    report = run_retrieval_recall_eval(corpus, gold)
    for row in report.results:
        assert row.recall_at_5 == 0.0


def test_phrasing_slice_is_split_and_gap_computed():
    corpus = [
        _chunk("c1", "ws", "doc-play", "Playbooks fire on a cron schedule evaluated every minute."),
        _chunk("c2", "ws", "doc-noise", "Team lunch is on Friday at the usual place."),
    ]
    gold = [
        _query("q-nat", "ws", "how do recurring automations run", ["doc-play"], phrasing="natural", pair_id="p1"),
        _query("q-key", "ws", "playbook cron schedule", ["doc-play"], phrasing="keyword", pair_id="p1"),
    ]
    report = run_retrieval_recall_eval(corpus, gold, variants={"bm25": bm25_retriever})
    row = report.results[0]
    assert row.n_natural == 1 and row.n_keyword == 1
    # Keyword phrasing hits (exact lexical overlap); the natural phrasing shares
    # no content token with the chunk — the audit's failure mode, reproduced.
    assert row.recall_at_5_keyword == 1.0
    assert row.recall_at_5_natural == 0.0
    assert row.phrasing_gap == -1.0


def test_report_covers_every_offline_variant():
    corpus = load_corpus()
    gold = load_gold_set()
    report = run_retrieval_recall_eval(corpus, gold)
    assert set(report.variants()) == set(OFFLINE_VARIANTS)
    tenants = {r.workspace_id for r in report.results}
    assert tenants == {"ws-nova", "ws-mercury"}
    for row in report.results:
        assert 0.0 <= row.recall_at_5 <= 1.0
        assert 0.0 <= row.mrr <= 1.0


# ---------------------------------------------------------------------------
# Honest gate + exit code
# ---------------------------------------------------------------------------

def test_gate_reads_the_shipped_hybrid_variant():
    def _row(variant, r5):
        return VariantTenantResult(
            variant=variant, workspace_id="ws", n_queries=10,
            recall_at_1=r5, recall_at_3=r5, recall_at_5=r5, mrr=r5,
            recall_at_5_natural=r5, recall_at_5_keyword=r5, n_natural=5, n_keyword=5,
        )

    below = RetrievalRecallReport(results=[_row("dense_proxy", 1.0), _row("hybrid_rrf", RETRIEVAL_RECALL_AT_5_TARGET - 0.05)])
    assert below.passes is False  # a perfect non-shipped variant cannot mask the gate
    at_target = RetrievalRecallReport(results=[_row("hybrid_rrf", RETRIEVAL_RECALL_AT_5_TARGET)])
    assert at_target.passes is True


def test_main_exits_zero_regardless_of_number(capsys):
    """The lane never reds CI: sub-threshold or not, main() returns 0 and the
    number is published."""
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "recall@5" in out.lower() or "recall_at_5" in out


def test_main_json_is_machine_readable(capsys):
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gated_variant"] == "hybrid_rrf"
    assert payload["recall_at_5_target"] == RETRIEVAL_RECALL_AT_5_TARGET
    assert set(payload["variants"]) == set(OFFLINE_VARIANTS)
    assert isinstance(payload["passes"], bool)


def test_live_mode_requires_workspace():
    assert main(["--live"]) == 2


# ---------------------------------------------------------------------------
# Fixture integrity — a typo fails loudly here, not silently in the number
# ---------------------------------------------------------------------------

def test_fixture_gold_ids_resolve_and_pairs_are_consistent():
    corpus = load_corpus()
    gold = load_gold_set()

    docs_by_ws = {}
    for c in corpus:
        docs_by_ws.setdefault(c.workspace_id, set()).add(c.document_id)

    assert len(gold) == 50
    pairs = {}
    for q in gold:
        assert q.phrasing in PHRASING_STYLES, f"{q.query_id}: bad phrasing tag {q.phrasing!r}"
        assert q.relevant_doc_ids, f"{q.query_id}: no relevant documents labelled"
        missing = q.relevant_doc_ids - docs_by_ws.get(q.workspace_id, set())
        assert not missing, f"{q.query_id}: relevant ids not in its workspace corpus: {missing}"
        if q.pair_id:
            pairs.setdefault(q.pair_id, []).append(q)

    assert len(pairs) == 12  # the audit's phrasing-sensitivity slice
    for pair_id, members in pairs.items():
        assert len(members) == 2, f"{pair_id}: a phrasing pair needs exactly 2 members"
        a, b = members
        assert a.relevant_doc_ids == b.relevant_doc_ids, f"{pair_id}: pair labels diverge"
        assert {a.phrasing, b.phrasing} == set(PHRASING_STYLES), f"{pair_id}: pair must be natural+keyword"
        assert a.workspace_id == b.workspace_id, f"{pair_id}: pair spans workspaces"


def test_fixture_corpus_is_well_formed():
    corpus = load_corpus()
    assert len(corpus) >= 40
    chunk_ids = [c.chunk_id for c in corpus]
    assert len(chunk_ids) == len(set(chunk_ids)), "duplicate chunk_id in corpus"
    for c in corpus:
        assert c.text.strip() and c.document_id and c.workspace_id


def test_fusion_weights_mirror_shipped_defaults():
    """The offline hybrid must weight legs the way the shipped RAGConfig does
    (hybrid_vector_weight=0.7 / hybrid_keyword_weight=0.3) so the published
    number describes the shipped shape. If the product defaults change, change
    this on purpose, together."""
    assert HYBRID_VECTOR_WEIGHT == pytest.approx(0.7)
    assert HYBRID_KEYWORD_WEIGHT == pytest.approx(0.3)
