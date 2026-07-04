"""Memory-recall eval — the first memory quality number (PRD-185 S10).

The Phase-2 review's seed complaint — *"memory saves low-quality memories"* — has
never been a tracked number. This is that number: recall@k / MRR of the memory
retriever over a labelled workspace gold-set, plus a with-vs-without task-lift A/B.
It is the exit criterion the T1 graph-substrate decision (Wave 3 P2-17) is gated
on, so it must be honest, reproducible, and never massaged to pass.

What it measures (offline, pure — no LLM, no network, no live store):

  For each tenant, over a labelled gold-set of queries against a bundled memory
  *snapshot* (the "store snapshot" the PRD calls for — a fixture standing in for a
  real workspace's memories so the eval runs during pilot and in CI):

    * recall@1 / recall@3 / recall@5 — of the memories labelled relevant to a
      query, the fraction surfaced in the retriever's top-k.
    * MRR — mean reciprocal rank of the first relevant memory (0 when none is in
      the top-k window), the LongMemEval-shaped ranking signal.
    * task-lift (with vs without memory) — for the memory-dependent queries (a
      workspace-specific fact you cannot answer without retrieval), the lift in
      answerability from having the fact in the top-k versus an empty context.
      Reuses the W7 uplift honest-gate shape: treatment minus baseline, a number
      that is published, never a CI gate.

  The retriever is an OFFLINE bag-of-words cosine proxy for the production
  ActionSemanticIndex / Mem0 vector search, so the harness runs in CI without an
  embedding provider or a vector store. A provisioned run injects the real
  retriever via ``retriever_factory`` and points ``--corpus`` at a live snapshot.

Honesty (mirrors PRD-177 S6): the gold-set labels relevance independently of the
ranker, and the corpus carries same-topic distractors, so the number reflects
real retrieval discrimination — not a fixture rigged to 100%. A sub-threshold
recall is a valid, honest outcome: it is reported, not massaged, and the T1
decision must not treat a low number as a pass.

Run:
    cd orchestrator
    python -m evals.memory_recall              # bundled snapshot (Markdown)
    python -m evals.memory_recall --json       # machine-readable
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

# Published gate for the memory retriever (recall@5, mean across tenants). Below
# this, the retriever is not yet good enough to trust as the T1-decision baseline
# — but a sub-threshold run is still published (exit 0); the number is the point.
RECALL_AT_5_TARGET = 0.70

# The retrieval window the recall/MRR/task-lift are computed over.
TOP_K = 5

_HERE = Path(__file__).resolve().parent
_DEFAULT_CORPUS = _HERE.parent / "scripts" / "eval" / "memory_recall" / "corpus.jsonl"
_DEFAULT_GOLD_SET = _HERE.parent / "scripts" / "eval" / "memory_recall" / "gold_set.jsonl"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryDoc:
    """One memory in the store snapshot the retriever searches over."""

    memory_id: str
    workspace_id: str
    text: str
    content_type: str
    category: str


@dataclass(frozen=True)
class GoldQuery:
    """A labelled query: which memories a good retriever should surface."""

    query_id: str
    workspace_id: str
    query: str
    relevant_ids: frozenset
    category: str
    difficulty: str
    # True when answering the query REQUIRES a workspace-specific stored fact —
    # i.e. it cannot be answered from general knowledge without retrieval. Only
    # these count toward the task-lift A/B (a general-knowledge query gets no
    # honest lift from memory).
    memory_dependent: bool


@dataclass
class TenantResult:
    workspace_id: str
    n_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    # with-vs-without on the memory-dependent subset (points, treatment-baseline)
    n_dependent: int
    task_lift_points: float


@dataclass
class MemoryRecallReport:
    tenants: List[TenantResult] = field(default_factory=list)

    def _mean(self, attr: str) -> float:
        if not self.tenants:
            return 0.0
        return sum(getattr(t, attr) for t in self.tenants) / len(self.tenants)

    @property
    def mean_recall_at_5(self) -> float:
        return self._mean("recall_at_5")

    @property
    def mean_mrr(self) -> float:
        return self._mean("mrr")

    @property
    def mean_task_lift_points(self) -> float:
        return self._mean("task_lift_points")

    @property
    def passes(self) -> bool:
        return self.mean_recall_at_5 >= RECALL_AT_5_TARGET

    def to_dict(self) -> Dict:
        return {
            "recall_at_5_target": RECALL_AT_5_TARGET,
            "top_k": TOP_K,
            "mean_recall_at_5": round(self.mean_recall_at_5, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_task_lift_points": round(self.mean_task_lift_points, 2),
            "passes": self.passes,
            "tenants": [
                {
                    "workspace_id": t.workspace_id,
                    "n_queries": t.n_queries,
                    "recall_at_1": round(t.recall_at_1, 4),
                    "recall_at_3": round(t.recall_at_3, 4),
                    "recall_at_5": round(t.recall_at_5, 4),
                    "mrr": round(t.mrr, 4),
                    "n_dependent": t.n_dependent,
                    "task_lift_points": round(t.task_lift_points, 2),
                }
                for t in self.tenants
            ],
        }


# ---------------------------------------------------------------------------
# Tokenisation + bag-of-words cosine (offline retriever proxy)
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


# A retriever ranks a workspace's memories for a query, best-first, returning
# their ids. The factory builds one bound to a specific corpus slice.
Retriever = Callable[[str], List[str]]
RetrieverFactory = Callable[[List[MemoryDoc]], Retriever]


def _bow_retriever(docs: List[MemoryDoc]) -> Retriever:
    """Deterministic bag-of-words cosine over memory text + category — an offline
    proxy for the production vector search. A provisioned run injects the real
    retriever (Mem0 / ActionSemanticIndex) against a live snapshot instead."""
    vecs = [(d.memory_id, _bow_vector(_tokenize(f"{d.text} {d.category}"))) for d in docs]

    def retrieve(query: str) -> List[str]:
        qv = _bow_vector(_tokenize(query))
        scored = [(_cosine(qv, v), mid) for mid, v in vecs]
        # Stable: break ties by memory_id so the eval is deterministic.
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [mid for _, mid in scored]

    return retrieve


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _recall_at_k(ranked_ids: List[str], relevant: frozenset, k: int) -> float:
    """Fraction of the relevant memories that appear in the top-k. For a query
    with a single relevant memory this is 1.0 iff it is retrieved in the top-k
    (hit@k); for multi-relevant queries it is the standard recall@k."""
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def _reciprocal_rank(ranked_ids: List[str], relevant: frozenset, k: int) -> float:
    """1 / (1-indexed rank of the first relevant memory) within the top-k, else 0."""
    for i, mid in enumerate(ranked_ids[:k]):
        if mid in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

def run_memory_recall_eval(
    corpus: List[MemoryDoc],
    gold: List[GoldQuery],
    retriever_factory: Optional[RetrieverFactory] = None,
) -> MemoryRecallReport:
    """Compute the per-tenant recall/MRR/task-lift report.

    Args:
        corpus: the memory store snapshot (per-tenant memories).
        gold: labelled queries tagged with workspace_id + relevant memory ids.
        retriever_factory: optional injection of the REAL retriever for a
            provisioned run; defaults to the offline bag-of-words proxy.

    Retrieval is scoped per tenant — a query only ever ranks its own workspace's
    memories, mirroring the production ``workspace_id`` filter (no cross-tenant
    leakage in the eval, same as in the product).
    """
    factory = retriever_factory or _bow_retriever

    corpus_by_tenant: Dict[str, List[MemoryDoc]] = defaultdict(list)
    for d in corpus:
        corpus_by_tenant[d.workspace_id].append(d)

    gold_by_tenant: Dict[str, List[GoldQuery]] = defaultdict(list)
    for q in gold:
        gold_by_tenant[q.workspace_id].append(q)

    report = MemoryRecallReport()
    for ws in sorted(gold_by_tenant):
        queries = gold_by_tenant[ws]
        retriever = factory(corpus_by_tenant.get(ws, []))

        r1 = r3 = r5 = mrr = 0.0
        dependent_hits = 0
        n_dependent = 0
        for q in queries:
            ranked = retriever(q.query)
            r1 += _recall_at_k(ranked, q.relevant_ids, 1)
            r3 += _recall_at_k(ranked, q.relevant_ids, 3)
            r5 += _recall_at_k(ranked, q.relevant_ids, TOP_K)
            mrr += _reciprocal_rank(ranked, q.relevant_ids, TOP_K)
            if q.memory_dependent:
                n_dependent += 1
                # "with memory" answerability: the needed fact is in the top-k.
                if set(ranked[:TOP_K]) & q.relevant_ids:
                    dependent_hits += 1

        n = len(queries)
        # task-lift: treatment (with memory) minus baseline (empty context). A
        # workspace-specific fact is unanswerable from an empty context, so the
        # honest baseline is 0 — the lift is the fraction of memory-dependent
        # questions memory makes answerable, in points. Same shape as W7 uplift.
        with_memory = (dependent_hits / n_dependent) if n_dependent else 0.0
        task_lift_points = with_memory * 100.0

        report.tenants.append(
            TenantResult(
                workspace_id=ws,
                n_queries=n,
                recall_at_1=(r1 / n) if n else 0.0,
                recall_at_3=(r3 / n) if n else 0.0,
                recall_at_5=(r5 / n) if n else 0.0,
                mrr=(mrr / n) if n else 0.0,
                n_dependent=n_dependent,
                task_lift_points=task_lift_points,
            )
        )
    return report


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


def load_corpus(path: Path = _DEFAULT_CORPUS) -> List[MemoryDoc]:
    return [
        MemoryDoc(
            memory_id=r["memory_id"],
            workspace_id=r["workspace_id"],
            text=r["text"],
            content_type=r.get("content_type", "unknown"),
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
            relevant_ids=frozenset(r["relevant_memory_ids"]),
            category=r.get("category", "uncategorized"),
            difficulty=r.get("difficulty", "medium"),
            memory_dependent=bool(r.get("memory_dependent", True)),
        )
        for r in _read_jsonl(path)
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_report(report: MemoryRecallReport) -> str:
    lines = [
        "# Memory-recall eval (PRD-185 S10)",
        "",
        f"Retrieval window: top-{TOP_K}. Published recall@5 gate: "
        f"{RECALL_AT_5_TARGET:.2f} (mean across tenants).",
        "",
        "| tenant | n(q) | recall@1 | recall@3 | recall@5 | MRR | n(dep) | task-lift (pts) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in report.tenants:
        lines.append(
            f"| {t.workspace_id} | {t.n_queries} | {t.recall_at_1*100:.1f}% | "
            f"{t.recall_at_3*100:.1f}% | {t.recall_at_5*100:.1f}% | {t.mrr:.3f} | "
            f"{t.n_dependent} | {t.task_lift_points:+.1f} |"
        )
    lines += [
        "",
        f"**Mean recall@5: {report.mean_recall_at_5*100:.1f}%** "
        f"(MRR {report.mean_mrr:.3f}, task-lift {report.mean_task_lift_points:+.1f} pts) "
        f"— {'PASSES' if report.passes else 'BELOW'} the "
        f"{RECALL_AT_5_TARGET*100:.0f}% gate.",
        "",
    ]
    if not report.passes:
        lines.append(
            "Recommendation: recall is BELOW the published gate — this is an honest "
            "sub-threshold outcome on the bundled snapshot + offline retriever proxy. "
            "Re-run against a live workspace snapshot with the real retriever injected "
            "before treating this number as the T1-decision baseline (Wave 3 P2-17)."
        )
    else:
        lines.append(
            "Recommendation: recall clears the published gate on the bundled snapshot. "
            "The real gate is this same harness against a live workspace snapshot with "
            "the production retriever injected."
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Memory-recall eval (PRD-185 S10)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS, help="corpus snapshot jsonl")
    parser.add_argument("--gold", type=Path, default=_DEFAULT_GOLD_SET, help="gold-set jsonl")
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    gold = load_gold_set(args.gold)
    report = run_memory_recall_eval(corpus, gold)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_report(report))
    # A sub-threshold recall is a valid, honest result to publish — never a CI
    # failure. Exit 0 always; the number is the deliverable. (CI runs this
    # non-required, mirroring evals/operating_graph_uplift.)
    return 0


if __name__ == "__main__":
    sys.exit(main())
