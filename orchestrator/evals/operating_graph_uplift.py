"""Operating-graph uplift eval — the business gate (PRD-177 S6).

Measures whether the learned per-tenant operating graph actually beats
retrieval baselines at selecting the right tool for a query. If it doesn't, the
moat claim (§8) fails honest review and the ``TOOL_ROUTING_GRAPH`` default must
NOT be flipped on.

What it measures (offline, pure — no LLM, no network):

  For each tenant, top-1 tool-selection accuracy of three rankers over held-out
  queries, and the UPLIFT of the learned-edge ranker over the best baseline:

    * BM25            — lexical ranking over action text (Okapi BM25, in-module).
    * Embedding       — cosine over a deterministic bag-of-words vector. This is
                        an OFFLINE PROXY for the production ActionSemanticIndex so
                        the eval runs in CI without an embedding provider; a
                        provisioned run substitutes the real index via
                        ``embedding_ranker`` injection.
    * Learned edge    — per-tenant succeeds-for-intent signal learned from a
                        TRAIN split of that tenant's own history, falling back to
                        the embedding proxy when a tenant/cluster has no signal.

Honesty (PRD-177 trap #3): the number is computed from a TRAIN/TEST split — the
learned signal is derived from held-out history, never hand-tuned to pass. When
run on the bundled synthetic fixture the number reflects THAT fixture; the real
gate is running this same harness against production per-tenant telemetry. A
sub-threshold number is a valid, honest outcome — it is reported, not massaged,
and the caller must not flip the flag on it.

Run:
    cd orchestrator
    python -m evals.operating_graph_uplift              # bundled fixture
    python -m evals.operating_graph_uplift --json       # machine-readable
    python -m evals.operating_graph_uplift --from-telemetry --json   # the real gate
                                                        # (inside the API environment)
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# The pass band for the moat claim (points of top-1 accuracy uplift, per tenant,
# averaged across tenants). Below this, do NOT flip TOOL_ROUTING_GRAPH on.
UPLIFT_THRESHOLD_POINTS = 5.0

_HERE = Path(__file__).resolve().parent
_DEFAULT_EVAL_SET = (
    _HERE.parent / "scripts" / "eval" / "tool_routing" / "eval_set.jsonl"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalCase:
    query: str
    correct_action: str
    category: str
    workspace_id: str


@dataclass
class TenantResult:
    workspace_id: str
    n_test: int
    bm25_acc: float
    embedding_acc: float
    learned_acc: float

    @property
    def best_baseline(self) -> float:
        return max(self.bm25_acc, self.embedding_acc)

    @property
    def uplift_points(self) -> float:
        return (self.learned_acc - self.best_baseline) * 100.0


@dataclass
class UpliftReport:
    tenants: List[TenantResult] = field(default_factory=list)
    # Where the cases came from ("fixture" | "telemetry") plus the loader's
    # counts. Printed WITH the number so a fixture run can never be mistaken
    # for the production gate (PRD-232 §6.4).
    meta: Dict[str, Any] = field(default_factory=lambda: {"source": "fixture"})

    @property
    def mean_uplift_points(self) -> float:
        if not self.tenants:
            return 0.0
        return sum(t.uplift_points for t in self.tenants) / len(self.tenants)

    @property
    def passes(self) -> bool:
        return self.mean_uplift_points >= UPLIFT_THRESHOLD_POINTS

    def to_dict(self) -> Dict:
        return {
            "source": self.meta.get("source", "fixture"),
            "loader": {k: v for k, v in self.meta.items() if k != "source"},
            "uplift_threshold_points": UPLIFT_THRESHOLD_POINTS,
            "mean_uplift_points": round(self.mean_uplift_points, 2),
            "passes": self.passes,
            "flip_flag_recommended": self.passes,
            "tenants": [
                {
                    "workspace_id": t.workspace_id,
                    "n_test": t.n_test,
                    "bm25_top1": round(t.bm25_acc, 4),
                    "embedding_top1": round(t.embedding_acc, 4),
                    "learned_top1": round(t.learned_acc, 4),
                    "uplift_points": round(t.uplift_points, 2),
                }
                for t in self.tenants
            ],
        }


# ---------------------------------------------------------------------------
# Tokenisation + text corpus for the action space
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _action_text(action: str, category: str) -> str:
    """Searchable text for an action: its underscore-split name + category.

    e.g. ``platform_get_cost_breakdown`` / ``analytics`` ->
    'platform get cost breakdown analytics'. This stands in for the action's
    description/keywords; a provisioned run can pass richer action metadata.
    """
    name_tokens = action.replace("platform_", "").replace("_", " ")
    return f"{name_tokens} {category}"


# ---------------------------------------------------------------------------
# BM25 (Okapi) — small, dependency-free
# ---------------------------------------------------------------------------

class _BM25:
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.n = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.df: Counter = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] += 1
        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            # BM25+ idf floor keeps every term's contribution non-negative.
            self.idf[term] = math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
        self.tf: List[Counter] = [Counter(doc) for doc in corpus_tokens]

    def score(self, query_tokens: Sequence[str], index: int) -> float:
        score = 0.0
        tf = self.tf[index]
        dl = self.doc_len[index]
        for term in query_tokens:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1)
        return score


# ---------------------------------------------------------------------------
# Rankers — each returns the top-1 predicted action for a query
# ---------------------------------------------------------------------------

Ranker = Callable[[str], str]


def _bm25_ranker(actions: List[str], category_by_action: Dict[str, str]) -> Ranker:
    corpus = [_tokenize(_action_text(a, category_by_action[a])) for a in actions]
    bm25 = _BM25(corpus)

    def rank(query: str) -> str:
        q = _tokenize(query)
        scores = [(bm25.score(q, i), actions[i]) for i in range(len(actions))]
        # Stable: break ties by action name so the eval is deterministic.
        scores.sort(key=lambda x: (-x[0], x[1]))
        return scores[0][1]

    return rank


def _bow_vector(tokens: Sequence[str]) -> Dict[str, float]:
    c = Counter(tokens)
    norm = math.sqrt(sum(v * v for v in c.values())) or 1.0
    return {t: v / norm for t, v in c.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


def _embedding_proxy_ranker(
    actions: List[str], category_by_action: Dict[str, str]
) -> Ranker:
    """Deterministic bag-of-words cosine — an offline proxy for the real
    ActionSemanticIndex. A provisioned run injects the real ranker instead."""
    vecs = {a: _bow_vector(_tokenize(_action_text(a, category_by_action[a]))) for a in actions}

    def rank(query: str) -> str:
        qv = _bow_vector(_tokenize(query))
        scores = [(_cosine(qv, vecs[a]), a) for a in actions]
        scores.sort(key=lambda x: (-x[0], x[1]))
        return scores[0][1]

    return rank


def _learned_edge_ranker(
    train: List[EvalCase],
    actions: List[str],
    category_by_action: Dict[str, str],
    fallback: Ranker,
    boost_weight: float = 1.0,
) -> Ranker:
    """Per-tenant succeeds-for-intent signal learned from the TRAIN split,
    combined WITH the embedding baseline exactly as production does.

    Production scores ``cosine * edge_confidence + boost`` — the learned signal
    BOOSTS the embedding ranking, it does not replace it. Here we replicate that:
    the base score is the embedding-proxy cosine to each action's text, plus a
    per-action boost proportional to how strongly a similar historical intent
    succeeded with that action (offline stand-in for succeeds_for_intent, keyed
    on query-embedding similarity rather than category). Cold intents get zero
    boost, so the ranker degrades to the embedding baseline — never worse.

    No hand-tuning: the boost is derived from the held-out history's actual
    intent→action pairings, so a non-positive uplift is an honest result.
    """
    action_text_vecs = {
        a: _bow_vector(_tokenize(_action_text(a, category_by_action[a]))) for a in actions
    }
    train_vecs: List[Tuple[Dict[str, float], str]] = [
        (_bow_vector(_tokenize(c.query)), c.correct_action) for c in train
    ]

    def rank(query: str) -> str:
        qv = _bow_vector(_tokenize(query))
        # Accumulate a per-action learned boost from every historical intent,
        # weighted by how similar that intent is to this query.
        boost: Dict[str, float] = defaultdict(float)
        for vec, action in train_vecs:
            sim = _cosine(qv, vec)
            if sim > 0.0:
                boost[action] += sim
        scored = []
        for a in actions:
            base = _cosine(qv, action_text_vecs[a])  # embedding-proxy base score
            scored.append((base + boost_weight * boost.get(a, 0.0), a))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1]

    return rank


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

def _accuracy(cases: List[EvalCase], ranker: Ranker) -> float:
    if not cases:
        return 0.0
    hits = sum(1 for c in cases if ranker(c.query) == c.correct_action)
    return hits / len(cases)


def _split_train_test(
    cases: List[EvalCase], test_every: int = 2
) -> Tuple[List[EvalCase], List[EvalCase]]:
    """Deterministic per-tenant split: every ``test_every``-th case (ordered) is
    held out for TEST, the rest train the learned signal. No randomness so the
    number is reproducible in CI."""
    train, test = [], []
    for i, c in enumerate(sorted(cases, key=lambda x: (x.category, x.query))):
        (test if i % test_every == 0 else train).append(c)
    return train, test


def run_uplift_eval(
    cases: List[EvalCase],
    embedding_ranker_factory: Optional[
        Callable[[List[str], Dict[str, str]], Ranker]
    ] = None,
) -> UpliftReport:
    """Compute the per-tenant uplift report.

    Args:
        cases: eval cases tagged with workspace_id (per-tenant).
        embedding_ranker_factory: optional injection of the REAL embedding
            ranker (production ActionSemanticIndex) for a provisioned run;
            defaults to the offline bag-of-words proxy.
    """
    emb_factory = embedding_ranker_factory or _embedding_proxy_ranker

    actions = sorted({c.correct_action for c in cases})
    category_by_action: Dict[str, str] = {}
    for c in cases:
        category_by_action.setdefault(c.correct_action, c.category)
    # Some actions may appear under multiple categories in real data; the first
    # wins (deterministic given the sort above is not applied to cases, so we
    # make it stable by preferring the lexicographically-smallest category).
    cat_candidates: Dict[str, set] = defaultdict(set)
    for c in cases:
        cat_candidates[c.correct_action].add(c.category)
    category_by_action = {a: sorted(cs)[0] for a, cs in cat_candidates.items()}

    by_tenant: Dict[str, List[EvalCase]] = defaultdict(list)
    for c in cases:
        by_tenant[c.workspace_id].append(c)

    report = UpliftReport()
    for ws in sorted(by_tenant):
        tenant_cases = by_tenant[ws]
        train, test = _split_train_test(tenant_cases)
        if not test:
            continue

        bm25 = _bm25_ranker(actions, category_by_action)
        embedding = emb_factory(actions, category_by_action)
        learned = _learned_edge_ranker(train, actions, category_by_action, embedding)

        report.tenants.append(
            TenantResult(
                workspace_id=ws,
                n_test=len(test),
                bm25_acc=_accuracy(test, bm25),
                embedding_acc=_accuracy(test, embedding),
                learned_acc=_accuracy(test, learned),
            )
        )
    return report


# ---------------------------------------------------------------------------
# Fixture loading (bundled eval_set.jsonl, sharded into synthetic tenants)
# ---------------------------------------------------------------------------

def load_cases_from_eval_set(
    path: Path = _DEFAULT_EVAL_SET, num_tenants: int = 2
) -> List[EvalCase]:
    """Load the bundled eval set and shard it into ``num_tenants`` synthetic
    tenants (round-robin by category so each tenant sees every category). This
    is a FIXTURE — the real gate reads production per-tenant telemetry."""
    if not path.exists():
        raise FileNotFoundError(f"eval set not found: {path}")

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    cases: List[EvalCase] = []
    cat_counter: Counter = Counter()
    for r in rows:
        correct = (r.get("correct_actions") or [None])[0]
        if not correct:
            continue
        category = r.get("category", "uncategorized")
        # Round-robin assignment within a category -> even per-tenant coverage.
        tenant_idx = cat_counter[category] % max(num_tenants, 1)
        cat_counter[category] += 1
        cases.append(
            EvalCase(
                query=r["query"],
                correct_action=correct,
                category=category,
                workspace_id=f"tenant-{tenant_idx}",
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Telemetry loading — the REAL gate (PRD-232 §6.4)
# ---------------------------------------------------------------------------
#
# One case per (tenant, distinct query) from tool_execution_logs, labelled with
# the action that actually succeeded for it. Honesty rules, in order:
#   * production rows only (telemetry_source 'production' / NULL) — eval and
#     replay rows never grade themselves;
#   * status == 'success' only — the ground truth is what served the intent;
#   * the resolved inner action name must exist in the ActionRegistry — the
#     `platform_execute` wrapper and unknown/retired names are dropped ("assert
#     on effects, not tool names");
#   * repeated identical queries collapse to ONE case labelled with the
#     majority successful action (ties → lexicographically smallest), so a
#     heartbeat loop cannot vote a hundred times;
#   * a tenant with fewer than ``min_cases_per_tenant`` distinct cases is
#     dropped — a three-case tenant makes the train/test split meaningless.
# The pure half (``cases_from_log_rows``) is unit-tested with no database; the
# thin wrapper (``load_cases_from_telemetry``) is the only place that imports
# the ORM, lazily, so the stdlib-only CI lane stays stdlib-only.

DEFAULT_TELEMETRY_WINDOW_DAYS = 30
DEFAULT_MIN_CASES_PER_TENANT = 10
_PRODUCTION_SOURCES = (None, "", "production")
_DISPATCHER_WRAPPER = "platform_execute"


def _normalise_query(query: str) -> str:
    return " ".join(query.lower().split())


def cases_from_log_rows(
    rows: Sequence[Dict[str, Any]],
    category_by_action: Dict[str, str],
    *,
    min_cases_per_tenant: int = DEFAULT_MIN_CASES_PER_TENANT,
) -> Tuple[List[EvalCase], Dict[str, Any]]:
    """Turn raw tool_execution_logs rows (dicts) into per-tenant eval cases.

    ``rows`` carry ``workspace_id``, ``user_query``, ``action_name``, ``status``
    and ``telemetry_source``; ``category_by_action`` is the registry's
    action → category map and doubles as the "known action" filter. Returns
    the cases plus the loader counts that explain everything that was dropped.
    """
    counts: Dict[str, Any] = {
        "rows_loaded": len(rows),
        "rows_non_production": 0,
        "rows_not_success": 0,
        "rows_without_query": 0,
        "rows_unknown_action": 0,
        "rows_without_workspace": 0,
        "distinct_queries": 0,
        "tenants_seen": 0,
        "tenants_dropped_small": 0,
        "tenants_evaluated": 0,
        "cases": 0,
        "min_cases_per_tenant": min_cases_per_tenant,
    }
    votes: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    first_query: Dict[Tuple[str, str], str] = {}
    for row in rows:
        if row.get("telemetry_source") not in _PRODUCTION_SOURCES:
            counts["rows_non_production"] += 1
            continue
        if row.get("status") != "success":
            counts["rows_not_success"] += 1
            continue
        query = (row.get("user_query") or "").strip()
        if not query:
            counts["rows_without_query"] += 1
            continue
        action = row.get("action_name") or ""
        if action == _DISPATCHER_WRAPPER or action not in category_by_action:
            counts["rows_unknown_action"] += 1
            continue
        workspace_id = row.get("workspace_id")
        if not workspace_id:
            counts["rows_without_workspace"] += 1
            continue
        key = (str(workspace_id), _normalise_query(query))
        votes[key][action] += 1
        first_query.setdefault(key, query)

    counts["distinct_queries"] = len(votes)
    by_tenant: Dict[str, List[EvalCase]] = defaultdict(list)
    for key, counter in votes.items():
        workspace_id, _ = key
        # Majority successful action; ties break to the lexicographically
        # smallest name so the label is identical run to run.
        action = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        by_tenant[workspace_id].append(
            EvalCase(
                query=first_query[key],
                correct_action=action,
                category=category_by_action[action],
                workspace_id=workspace_id,
            )
        )

    counts["tenants_seen"] = len(by_tenant)
    cases: List[EvalCase] = []
    for workspace_id in sorted(by_tenant):
        tenant_cases = by_tenant[workspace_id]
        if len(tenant_cases) < min_cases_per_tenant:
            counts["tenants_dropped_small"] += 1
            continue
        cases.extend(sorted(tenant_cases, key=lambda c: (c.category, c.query)))
    counts["tenants_evaluated"] = counts["tenants_seen"] - counts["tenants_dropped_small"]
    counts["cases"] = len(cases)
    return cases, counts


def load_cases_from_telemetry(
    window_days: int = DEFAULT_TELEMETRY_WINDOW_DAYS,
    min_cases_per_tenant: int = DEFAULT_MIN_CASES_PER_TENANT,
) -> Tuple[List[EvalCase], Dict[str, Any]]:
    """Read production tool_execution_logs for the last ``window_days`` and
    build per-tenant cases. Imports the ORM lazily: the harness stays
    importable (and the CI fixture lane runnable) without a database."""
    from datetime import datetime, timedelta

    from core.database.database import get_db_session
    from core.models.composio_cache import ToolExecutionLog
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    category_by_action = {a.name: a.category for a in registry.get_all()}

    cutoff = datetime.utcnow() - timedelta(days=window_days)
    with get_db_session() as db:
        query = (
            db.query(
                ToolExecutionLog.workspace_id,
                ToolExecutionLog.user_query,
                ToolExecutionLog.action_name,
                ToolExecutionLog.status,
                ToolExecutionLog.telemetry_source,
            )
            .filter(ToolExecutionLog.executed_at >= cutoff)
            .order_by(ToolExecutionLog.executed_at.asc())
        )
        rows = [
            {
                "workspace_id": str(ws) if ws else None,
                "user_query": q,
                "action_name": action,
                "status": status,
                "telemetry_source": source,
            }
            for ws, q, action, status, source in query.all()
        ]
    cases, counts = cases_from_log_rows(
        rows, category_by_action, min_cases_per_tenant=min_cases_per_tenant
    )
    counts["window_days"] = window_days
    counts["registry_actions"] = len(category_by_action)
    return cases, counts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _describe_source(meta: Dict[str, Any]) -> str:
    if meta.get("source") == "telemetry":
        return (
            f"Source: PRODUCTION TELEMETRY — {meta.get('cases', 0)} cases across "
            f"{meta.get('tenants_evaluated', 0)} tenant(s), window "
            f"{meta.get('window_days', '?')} days; {meta.get('tenants_dropped_small', 0)} "
            f"tenant(s) below the {meta.get('min_cases_per_tenant', '?')}-case minimum dropped."
        )
    return (
        f"Source: bundled fixture ({meta.get('cases', 0)} cases sharded into "
        f"{meta.get('tenants', '?')} synthetic tenants) — NOT the production gate."
    )


def render_report(report: UpliftReport) -> str:
    lines = [
        "# Operating-graph uplift eval (PRD-177 S6)",
        "",
        f"Uplift threshold: {UPLIFT_THRESHOLD_POINTS:.1f} points of top-1 accuracy "
        "(learned-edge over best baseline, per tenant, mean across tenants).",
        _describe_source(report.meta),
        "",
        "| tenant | n(test) | BM25 top-1 | embedding top-1 | learned top-1 | uplift (pts) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for t in report.tenants:
        lines.append(
            f"| {t.workspace_id} | {t.n_test} | {t.bm25_acc*100:.1f}% | "
            f"{t.embedding_acc*100:.1f}% | {t.learned_acc*100:.1f}% | "
            f"{t.uplift_points:+.1f} |"
        )
    lines += [
        "",
        f"**Mean uplift: {report.mean_uplift_points:+.1f} points** — "
        f"{'PASSES' if report.passes else 'BELOW'} the {UPLIFT_THRESHOLD_POINTS:.1f}-point gate.",
        "",
    ]
    if report.passes:
        lines.append(
            "Recommendation: uplift clears the gate — flipping `TOOL_ROUTING_GRAPH` "
            "on is supported by this run."
        )
    else:
        note = (
            "this IS the production-telemetry gate number"
            if report.meta.get("source") == "telemetry"
            else "this is the bundled fixture, not the gate — run with "
            "--from-telemetry inside the API environment for the real number"
        )
        lines.append(
            "Recommendation: uplift is BELOW the gate — do NOT flip "
            f"`TOOL_ROUTING_GRAPH` on. This is an honest sub-threshold outcome ({note})."
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operating-graph uplift eval (PRD-177 S6)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--tenants", type=int, default=2, help="synthetic tenant shards (fixture mode)"
    )
    parser.add_argument(
        "--from-telemetry",
        action="store_true",
        help="the real gate: build cases from production tool_execution_logs "
        "(needs the app database — run inside the API environment)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_TELEMETRY_WINDOW_DAYS,
        help="telemetry lookback in days (with --from-telemetry)",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=DEFAULT_MIN_CASES_PER_TENANT,
        help="drop tenants with fewer distinct cases (with --from-telemetry)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.from_telemetry:
        cases, counts = load_cases_from_telemetry(
            window_days=args.window_days, min_cases_per_tenant=args.min_cases
        )
        meta: Dict[str, Any] = {"source": "telemetry", **counts}
    else:
        cases = load_cases_from_eval_set(num_tenants=args.tenants)
        meta = {"source": "fixture", "tenants": args.tenants, "cases": len(cases)}
    report = run_uplift_eval(cases)
    report.meta = meta

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_report(report))
    # The eval never "fails" the CI job on a sub-threshold number — a low uplift
    # is a valid, honest result to publish. Exit 0 always; the number is the
    # deliverable. (CI runs this non-required.)
    return 0


if __name__ == "__main__":
    sys.exit(main())
