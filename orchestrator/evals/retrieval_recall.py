"""Retrieval-recall eval — the RAG quality number (PRD-188 S5, report id P2-07).

The Phase-2 review's verdict on the retrieval plane — *"the best-integrated
scaffold in the platform and none of the quality levers turned on"* — has never
had a number attached. This is that number: recall@k / MRR of document
retrieval over a labelled workspace gold-set, per retrieval variant, so every
lever PRD-188 turns on (rerank S1, contextual annotations S2, the BM25 sparse
leg S3) reports a before/after against the same instrument.

What it measures (offline, pure — no LLM, no network, no live store):

  For each tenant, over a labelled gold-set of queries against a bundled
  document-chunk *snapshot* (a fixture standing in for a real workspace's
  corpus so the eval runs during pilot and in CI), for each retrieval variant:

    * recall@1 / recall@3 / recall@5 — of the documents labelled relevant to a
      query, the fraction whose chunks surface in the retriever's top-k
      (ranked chunks are deduped to documents, first appearance wins).
    * MRR — mean reciprocal rank of the first relevant document.
    * phrasing sensitivity — the Internal Audit v1 (2026-07-09) confirmed
      3-for-3 that natural-language phrasings returned 0 hits where keyword
      phrasings returned 5-6. The gold set carries that exact failure mode:
      paired natural/keyword phrasings of the same intent (same relevant
      docs, same ``pair_id``), and the report publishes recall split by
      phrasing style plus the gap. The levers are graded on the failure mode
      observed live, not just aggregate recall.

  Offline variants (each a stand-in for one leg of the shipped pipeline):

    * ``dense_proxy`` — bag-of-words cosine, the deterministic proxy for the
      dense vector leg (same idiom as ``evals/memory_recall``).
    * ``bm25`` — pure-Python Okapi BM25 over the same tokens: the *actual
      algorithm* of the S3 sparse leg (Postgres ts_rank is its production
      seat), honestly runnable offline.
    * ``hybrid_rrf`` — weighted reciprocal-rank fusion of the two, using the
      shipped fusion constants (k=60, vector 0.7 / keyword 0.3). The fusion
      math here is pinned byte-for-byte to ``modules.rag.fusion`` by an
      equivalence test in the main suite (``tests/test_p2w1_hybrid_fusion``);
      it is re-stated here only because this eval's CI lane installs no
      third-party deps and ``modules.rag`` imports tiktoken at package import.

  A provisioned run (``--live --workspace <id>``) drives the REAL
  ``RAGService.retrieve`` instead, with each lever toggled per variant —
  rerank on/off, hybrid on/off, query-enhancement on/off — against a live
  workspace and a gold-set authored for it (``--gold``/``--corpus``). That run
  is what finally answers "does HyDE's 4-LLM-call enhancement pay for itself"
  with a number. It is never part of CI.

Honesty (mirrors evals/memory_recall): the gold-set labels relevance
independently of any ranker, the corpus carries same-topic distractor
documents, and a sub-threshold recall is a valid, honest outcome — it is
reported, not massaged, and the process exits 0 regardless. The published
gate below is a target for the *shipped hybrid variant*, not a CI gate.

Run:
    cd orchestrator
    python -m evals.retrieval_recall              # bundled snapshot (Markdown)
    python -m evals.retrieval_recall --json       # machine-readable
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# Published target for the SHIPPED variant (hybrid RRF), recall@5 mean across
# tenants. Below it, the stack has not yet earned the dossier's cited figures
# on this corpus — still published, still exit 0; the number is the point.
RETRIEVAL_RECALL_AT_5_TARGET = 0.70

# The retrieval window recall/MRR are computed over (documents, not chunks).
TOP_K = 5

# Fusion constants mirroring the shipped defaults: RAGConfig.rrf_k and the
# hybrid_vector_weight / hybrid_keyword_weight knobs S3 makes real.
RRF_K = 60
HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_KEYWORD_WEIGHT = 0.3

_HERE = Path(__file__).resolve().parent
_DEFAULT_CORPUS = _HERE.parent / "scripts" / "eval" / "retrieval_recall" / "corpus.jsonl"
_DEFAULT_GOLD_SET = _HERE.parent / "scripts" / "eval" / "retrieval_recall" / "gold_set.jsonl"

PHRASING_STYLES = ("natural", "keyword")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkDoc:
    """One document chunk in the corpus snapshot the retrievers rank."""

    chunk_id: str
    workspace_id: str
    document_id: str
    text: str
    category: str


@dataclass(frozen=True)
class GoldQuery:
    """A labelled query: which documents a good retriever should surface."""

    query_id: str
    workspace_id: str
    query: str
    relevant_doc_ids: frozenset
    category: str
    difficulty: str
    # The audit's phrasing-sensitivity slice: every query is tagged with its
    # phrasing style; paired natural/keyword restatements of one intent share
    # a pair_id (and identical relevant_doc_ids).
    phrasing: str
    pair_id: Optional[str]


@dataclass
class VariantTenantResult:
    variant: str
    workspace_id: str
    n_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    # recall@5 split by phrasing style, and the natural-minus-keyword gap.
    recall_at_5_natural: float
    recall_at_5_keyword: float
    n_natural: int
    n_keyword: int

    @property
    def phrasing_gap(self) -> float:
        return self.recall_at_5_natural - self.recall_at_5_keyword


@dataclass
class RetrievalRecallReport:
    results: List[VariantTenantResult] = field(default_factory=list)

    def variants(self) -> List[str]:
        seen: List[str] = []
        for r in self.results:
            if r.variant not in seen:
                seen.append(r.variant)
        return seen

    def _mean(self, variant: str, attr: str) -> float:
        rows = [r for r in self.results if r.variant == variant]
        if not rows:
            return 0.0
        return sum(getattr(r, attr) for r in rows) / len(rows)

    def mean_recall_at_5(self, variant: str) -> float:
        return self._mean(variant, "recall_at_5")

    def mean_mrr(self, variant: str) -> float:
        return self._mean(variant, "mrr")

    def mean_phrasing_gap(self, variant: str) -> float:
        return self._mean(variant, "phrasing_gap")

    @property
    def passes(self) -> bool:
        """The honest gate reads the SHIPPED variant (hybrid RRF)."""
        return self.mean_recall_at_5("hybrid_rrf") >= RETRIEVAL_RECALL_AT_5_TARGET

    def to_dict(self) -> Dict:
        return {
            "recall_at_5_target": RETRIEVAL_RECALL_AT_5_TARGET,
            "gated_variant": "hybrid_rrf",
            "top_k": TOP_K,
            "passes": self.passes,
            "variants": {
                v: {
                    "mean_recall_at_5": round(self.mean_recall_at_5(v), 4),
                    "mean_mrr": round(self.mean_mrr(v), 4),
                    "mean_phrasing_gap": round(self.mean_phrasing_gap(v), 4),
                }
                for v in self.variants()
            },
            "tenants": [
                {
                    "variant": r.variant,
                    "workspace_id": r.workspace_id,
                    "n_queries": r.n_queries,
                    "recall_at_1": round(r.recall_at_1, 4),
                    "recall_at_3": round(r.recall_at_3, 4),
                    "recall_at_5": round(r.recall_at_5, 4),
                    "mrr": round(r.mrr, 4),
                    "recall_at_5_natural": round(r.recall_at_5_natural, 4),
                    "recall_at_5_keyword": round(r.recall_at_5_keyword, 4),
                    "n_natural": r.n_natural,
                    "n_keyword": r.n_keyword,
                    "phrasing_gap": round(r.phrasing_gap, 4),
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Tokenisation + the two offline legs
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _bow_vector(tokens: Sequence[str]) -> Dict[str, float]:
    c = Counter(tokens)
    norm = math.sqrt(sum(v * v for v in c.values())) or 1.0
    return {t: v / norm for t, v in c.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


# A retriever ranks a workspace's chunks for a query, best-first, returning
# chunk_ids. The factory builds one bound to a specific corpus slice.
Retriever = Callable[[str], List[str]]
RetrieverFactory = Callable[[List[ChunkDoc]], Retriever]


def dense_proxy_retriever(chunks: List[ChunkDoc]) -> Retriever:
    """Deterministic bag-of-words cosine — the offline proxy for the dense
    vector leg (same idiom as evals/memory_recall's retriever proxy)."""
    vecs = [(c.chunk_id, _bow_vector(_tokenize(f"{c.text} {c.category}"))) for c in chunks]

    def retrieve(query: str) -> List[str]:
        qv = _bow_vector(_tokenize(query))
        # Zero cosine = no shared vocabulary: NOT retrieved (mirrors the dense
        # leg's min_similarity floor — an unmatched chunk never reaches the
        # candidate list, it doesn't limp in at the bottom of the ranking).
        scored = [(s, cid) for cid, v in vecs if (s := _cosine(qv, v)) > 0.0]
        scored.sort(key=lambda x: (-x[0], x[1]))  # tie-break: chunk_id, deterministic
        return [cid for _, cid in scored]

    return retrieve


# Okapi BM25 constants (standard values).
_BM25_K1 = 1.5
_BM25_B = 0.75


def bm25_retriever(chunks: List[ChunkDoc]) -> Retriever:
    """Pure-Python Okapi BM25 — the S3 sparse leg's algorithm (production seat:
    Postgres ts_rank over document_chunks.search_vector)."""
    docs = [(c.chunk_id, _tokenize(c.text)) for c in chunks]
    n_docs = len(docs) or 1
    avg_len = (sum(len(toks) for _, toks in docs) / n_docs) or 1.0
    doc_freq: Counter = Counter()
    for _, toks in docs:
        doc_freq.update(set(toks))

    def _idf(term: str) -> float:
        df = doc_freq.get(term, 0)
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def retrieve(query: str) -> List[str]:
        q_terms = _tokenize(query)
        scored = []
        for cid, toks in docs:
            tf = Counter(toks)
            length_norm = 1.0 - _BM25_B + _BM25_B * (len(toks) / avg_len)
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                score += _idf(term) * (f * (_BM25_K1 + 1.0)) / (f + _BM25_K1 * length_norm)
            # Zero score = no query term present: NOT retrieved (mirrors the
            # production leg, where the tsquery @@ predicate filters the rows
            # before ts_rank ever orders them).
            if score > 0.0:
                scored.append((score, cid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [cid for _, cid in scored]

    return retrieve


def _weighted_rrf(ranked_lists: Sequence[List[str]], weights: Sequence[float], k: int = RRF_K) -> List[str]:
    """Weighted reciprocal-rank fusion over id lists: score(id) =
    Σ_legs weight_leg / (k + rank_in_leg). Pinned to modules.rag.fusion by the
    main-suite equivalence test — keep the math identical."""
    scores: Dict[str, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, cid in enumerate(ranked):
            scores[cid] += weight / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def hybrid_rrf_retriever(chunks: List[ChunkDoc]) -> Retriever:
    """The shipped shape: dense + sparse fused by weighted RRF (0.7 / 0.3)."""
    dense = dense_proxy_retriever(chunks)
    sparse = bm25_retriever(chunks)

    def retrieve(query: str) -> List[str]:
        return _weighted_rrf(
            [dense(query), sparse(query)],
            [HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT],
        )

    return retrieve


OFFLINE_VARIANTS: Dict[str, RetrieverFactory] = {
    "dense_proxy": dense_proxy_retriever,
    "bm25": bm25_retriever,
    "hybrid_rrf": hybrid_rrf_retriever,
}


# ---------------------------------------------------------------------------
# Metrics (documents, not chunks)
# ---------------------------------------------------------------------------

def documents_from_chunk_ranking(ranked_chunk_ids: List[str], chunk_to_doc: Dict[str, str]) -> List[str]:
    """Collapse a ranked chunk list to a ranked document list — first
    appearance of each document wins (the "did we land on the right doc" view
    the gold-set labels).

    Ids absent from the map pass through unchanged: live mode ranks real
    document ids the fixture chunk→doc map has never seen (the live
    retrievers already return document ids). Empty ids are dropped.
    """
    seen = set()
    docs: List[str] = []
    for cid in ranked_chunk_ids:
        doc = chunk_to_doc.get(cid, cid)
        if not doc or doc in seen:
            continue
        seen.add(doc)
        docs.append(doc)
    return docs


def _recall_at_k(ranked_ids: List[str], relevant: frozenset, k: int) -> float:
    """Fraction of the relevant documents present in the top-k."""
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def _reciprocal_rank(ranked_ids: List[str], relevant: frozenset, k: int) -> float:
    """1 / (1-indexed rank of the first relevant document) within top-k, else 0."""
    for i, did in enumerate(ranked_ids[:k]):
        if did in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

def run_retrieval_recall_eval(
    corpus: List[ChunkDoc],
    gold: List[GoldQuery],
    variants: Optional[Dict[str, RetrieverFactory]] = None,
) -> RetrievalRecallReport:
    """Compute the per-variant, per-tenant recall/MRR/phrasing report.

    Retrieval is scoped per tenant — a query only ever ranks its own
    workspace's chunks, mirroring the production ``build_retrieval_filters``
    choke point (no cross-tenant leakage in the eval, same as in the product).
    """
    variant_factories = variants or OFFLINE_VARIANTS

    corpus_by_tenant: Dict[str, List[ChunkDoc]] = defaultdict(list)
    chunk_to_doc: Dict[str, str] = {}
    for c in corpus:
        corpus_by_tenant[c.workspace_id].append(c)
        chunk_to_doc[c.chunk_id] = c.document_id

    gold_by_tenant: Dict[str, List[GoldQuery]] = defaultdict(list)
    for q in gold:
        gold_by_tenant[q.workspace_id].append(q)

    report = RetrievalRecallReport()
    for variant_name, factory in variant_factories.items():
        for ws in sorted(gold_by_tenant):
            queries = gold_by_tenant[ws]
            retriever = factory(corpus_by_tenant.get(ws, []))

            r1 = r3 = r5 = mrr = 0.0
            by_phrasing: Dict[str, List[float]] = {s: [] for s in PHRASING_STYLES}
            for q in queries:
                ranked_docs = documents_from_chunk_ranking(retriever(q.query), chunk_to_doc)
                r1 += _recall_at_k(ranked_docs, q.relevant_doc_ids, 1)
                r3 += _recall_at_k(ranked_docs, q.relevant_doc_ids, 3)
                q_r5 = _recall_at_k(ranked_docs, q.relevant_doc_ids, TOP_K)
                r5 += q_r5
                mrr += _reciprocal_rank(ranked_docs, q.relevant_doc_ids, TOP_K)
                if q.phrasing in by_phrasing:
                    by_phrasing[q.phrasing].append(q_r5)

            n = len(queries)
            nat = by_phrasing["natural"]
            key = by_phrasing["keyword"]
            report.results.append(
                VariantTenantResult(
                    variant=variant_name,
                    workspace_id=ws,
                    n_queries=n,
                    recall_at_1=(r1 / n) if n else 0.0,
                    recall_at_3=(r3 / n) if n else 0.0,
                    recall_at_5=(r5 / n) if n else 0.0,
                    mrr=(mrr / n) if n else 0.0,
                    recall_at_5_natural=(sum(nat) / len(nat)) if nat else 0.0,
                    recall_at_5_keyword=(sum(key) / len(key)) if key else 0.0,
                    n_natural=len(nat),
                    n_keyword=len(key),
                )
            )
    return report


# ---------------------------------------------------------------------------
# Live mode (provisioned runs only — NEVER in CI)
# ---------------------------------------------------------------------------

def build_live_variants(workspace_id: str) -> Dict[str, RetrieverFactory]:
    """Lever-grid variants driving the REAL ``RAGService.retrieve`` — rerank
    (S1), hybrid/BM25 (S3), and query enhancement each toggled so their uplift
    is read off the same instrument. Imports are local: this path needs the
    full orchestrator environment and a live store; the offline lane must
    never pay for these imports.

    Chunks come back as ``RAGResult.chunks``; ranking is their returned order;
    document ids are read from each chunk's ``document_id`` (the eval reuses
    ``retrieve`` — the S9 score seam inside it fires as in production).
    """
    import asyncio

    from modules.rag.service import RAGConfig, RAGService  # noqa: PLC0415 (live-only)

    def _factory(**config_kwargs) -> RetrieverFactory:
        def factory(_chunks: List[ChunkDoc]) -> Retriever:
            service = RAGService(RAGConfig(**config_kwargs), workspace_id=workspace_id)

            def retrieve(query: str) -> List[str]:
                result = asyncio.run(
                    service.retrieve(query, max_chunks=TOP_K * 4, workspace_id=workspace_id)
                )
                # Document ids ARE the ranking ids here — the driver's
                # chunk→doc collapse passes ids it can't map straight through
                # (live ids never appear in the fixture map) and drops empties.
                return [str(c.get("document_id", "")) for c in result.chunks]

            return retrieve

        return factory

    return {
        "live_baseline": _factory(
            enable_reranking=False, hybrid_search_enabled=False, enable_query_enhancement=False
        ),
        "live_rerank": _factory(
            enable_reranking=True, hybrid_search_enabled=False, enable_query_enhancement=False
        ),
        "live_hybrid": _factory(
            enable_reranking=False, hybrid_search_enabled=True, enable_query_enhancement=False
        ),
        "live_enhancement": _factory(
            enable_reranking=False, hybrid_search_enabled=False, enable_query_enhancement=True
        ),
        "live_full_stack": _factory(
            enable_reranking=True, hybrid_search_enabled=True, enable_query_enhancement=True
        ),
    }


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"fixture not found: {path}")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_corpus(path: Path = _DEFAULT_CORPUS) -> List[ChunkDoc]:
    return [
        ChunkDoc(
            chunk_id=r["chunk_id"],
            workspace_id=r["workspace_id"],
            document_id=r["document_id"],
            text=r["text"],
            category=r.get("category", "uncategorized"),
        )
        for r in _read_jsonl(path)
    ]


def load_gold_set(path: Path = _DEFAULT_GOLD_SET) -> List[GoldQuery]:
    return [
        GoldQuery(
            query_id=r["query_id"],
            workspace_id=r["workspace_id"],
            query=r["query"],
            relevant_doc_ids=frozenset(r["relevant_document_ids"]),
            category=r.get("category", "uncategorized"),
            difficulty=r.get("difficulty", "medium"),
            phrasing=r.get("phrasing", "natural"),
            pair_id=r.get("pair_id"),
        )
        for r in _read_jsonl(path)
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_report(report: RetrievalRecallReport) -> str:
    lines = [
        "# Retrieval-recall eval (PRD-188 S5 / P2-07)",
        "",
        f"Retrieval window: top-{TOP_K} documents. Published recall@5 target for "
        f"the shipped hybrid variant: {RETRIEVAL_RECALL_AT_5_TARGET:.2f} (mean across tenants).",
        "",
        "| variant | tenant | n(q) | recall@1 | recall@3 | recall@5 | MRR | r@5 natural | r@5 keyword | gap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        lines.append(
            f"| {r.variant} | {r.workspace_id} | {r.n_queries} | {r.recall_at_1*100:.1f}% | "
            f"{r.recall_at_3*100:.1f}% | {r.recall_at_5*100:.1f}% | {r.mrr:.3f} | "
            f"{r.recall_at_5_natural*100:.1f}% | {r.recall_at_5_keyword*100:.1f}% | "
            f"{r.phrasing_gap*100:+.1f} |"
        )
    lines.append("")
    for v in report.variants():
        lines.append(
            f"**{v}: mean recall@5 {report.mean_recall_at_5(v)*100:.1f}%** "
            f"(MRR {report.mean_mrr(v):.3f}, phrasing gap {report.mean_phrasing_gap(v)*100:+.1f} pts)"
        )
    lines += [
        "",
        f"Gate ({RETRIEVAL_RECALL_AT_5_TARGET*100:.0f}% on hybrid_rrf): "
        f"{'PASSES' if report.passes else 'BELOW'}.",
        "",
    ]
    if not report.passes:
        lines.append(
            "Recommendation: the shipped hybrid variant is BELOW the published target — "
            "an honest sub-threshold outcome on the bundled snapshot + offline leg proxies. "
            "Re-run `--live` against a provisioned workspace (levers toggled per variant) "
            "before treating this number as the stack's verdict."
        )
    else:
        lines.append(
            "Recommendation: the hybrid variant clears the published target on the bundled "
            "snapshot. The real verdict is the `--live` lever grid against a provisioned "
            "workspace with a gold-set authored for it."
        )
    lines.append(
        "\nPhrasing slice: a NEGATIVE gap (natural below keyword) reproduces the Internal "
        "Audit v1 failure mode (natural phrasings 0-for-3 where keyword phrasings hit). "
        "The levers are judged on closing it, not just on aggregate recall."
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval-recall eval (PRD-188 S5)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS, help="corpus snapshot jsonl")
    parser.add_argument("--gold", type=Path, default=_DEFAULT_GOLD_SET, help="gold-set jsonl")
    parser.add_argument(
        "--live",
        action="store_true",
        help="drive the REAL RAGService.retrieve lever grid (provisioned runs only, never CI)",
    )
    parser.add_argument("--workspace", type=str, default=None, help="workspace id for --live")
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    gold = load_gold_set(args.gold)

    variants: Optional[Dict[str, RetrieverFactory]] = None
    if args.live:
        if not args.workspace:
            print("--live requires --workspace <id>", file=sys.stderr)
            return 2
        variants = build_live_variants(args.workspace)

    report = run_retrieval_recall_eval(corpus, gold, variants=variants)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_report(report))
    # A sub-threshold recall is a valid, honest result to publish — never a CI
    # failure. Exit 0 always; the number is the deliverable. (CI runs this
    # non-required, mirroring evals/memory_recall.)
    return 0


if __name__ == "__main__":
    sys.exit(main())
