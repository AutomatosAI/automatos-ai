"""
Score `results/results.jsonl` and emit a markdown report.

Computes per (model, mode):
- top-1 accuracy            chosen_action ∈ correct_actions
- in-set hit rate           any correct_action ∈ surfaced
- prompt tokens (mean)
- completion tokens (mean)
- total tokens (mean)
- cost per call (mean USD)  using models.yaml `cost_in` / `cost_out`
- cost per correct (USD)
- latency p50 / p95 (ms)
- error rate

Plus a per-category breakdown so we can see *where* semantic routing
helps most (the hypothesis is: paraphrase + ambiguous categories).

Run:

    cd orchestrator
    python -m scripts.eval.tool_routing.score
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

HERE = Path(__file__).resolve().parent
RESULTS_JSONL = HERE / "results" / "results.jsonl"
MODELS_YAML = HERE / "models.yaml"
REPORT_MD = HERE / "results" / "report.md"
SUMMARY_CSV = HERE / "results" / "summary.csv"


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile. Returns None for empty input."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return s[lo]
    weight = rank - lo
    return s[lo] * (1 - weight) + s[hi] * weight


def _mean(values: Iterable[float]) -> Optional[float]:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return statistics.fmean(vs)


def _is_correct(row: Dict[str, Any]) -> bool:
    chosen = row.get("chosen_action")
    if not chosen:
        return False
    return chosen in (row.get("correct_actions") or [])


# Top-level tools are always available regardless of the filtered platform-action
# catalog — they're separate function-calling tools in the LLM's tool list. So if
# `correct_actions` references one of these, the agent can always reach it; the
# filter never "drops" them. Mirrors `KNOWN_TOP_LEVEL_TOOLS` in seed_eval_set.py.
_ALWAYS_AVAILABLE: frozenset = frozenset({
    "composio_execute",
    "search_knowledge",
    "semantic_search",
    "generate_document",
    "write_file",
    "smart_query_database",
    "query_database",
    "workspace_read_file",
    "workspace_write_file",
    "workspace_list_dir",
    "workspace_grep",
    "workspace_exec",
    "workspace_git",
})


def _is_in_set(row: Dict[str, Any]) -> bool:
    """Did the prompt at least surface a correct action?

    Top-level tools (composio_execute, workspace_*, etc.) are always surfaced
    because they're independent function-calling tools, not entries in the
    filtered platform-action catalog.
    """
    correct = set(row.get("correct_actions") or [])
    if correct & _ALWAYS_AVAILABLE:
        return True
    surfaced = set(row.get("surfaced") or [])
    return bool(correct & surfaced)


def _cost_for(row: Dict[str, Any], model_costs: Dict[str, Tuple[float, float]]) -> Optional[float]:
    pt = row.get("prompt_tokens")
    ct = row.get("completion_tokens")
    if pt is None or ct is None:
        return None
    cost_in, cost_out = model_costs.get(row["model"], (0.0, 0.0))
    return (pt * cost_in + ct * cost_out) / 1_000_000


# ──────────────────────────────────────────────────────────────────


def _load_results() -> List[Dict[str, Any]]:
    if not RESULTS_JSONL.exists():
        print(f"missing {RESULTS_JSONL}", file=sys.stderr)
        sys.exit(2)
    rows: List[Dict[str, Any]] = []
    for line in RESULTS_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_model_costs() -> Tuple[Dict[str, Tuple[float, float]], Dict[str, str]]:
    cfg = yaml.safe_load(MODELS_YAML.read_text())
    costs: Dict[str, Tuple[float, float]] = {}
    tiers: Dict[str, str] = {}
    for m in cfg.get("models") or []:
        costs[m["id"]] = (float(m.get("cost_in", 0.0)), float(m.get("cost_out", 0.0)))
        tiers[m["id"]] = m.get("tier", "unknown")
    return costs, tiers


# ──────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────


def _aggregate(
    rows: List[Dict[str, Any]],
    model_costs: Dict[str, Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """One summary row per (model, mode)."""
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[(r["model"], r["mode"])].append(r)

    out: List[Dict[str, Any]] = []
    for (model, mode), bucket in buckets.items():
        n = len(bucket)
        errors = [r for r in bucket if r.get("error")]
        correct = [r for r in bucket if _is_correct(r)]
        in_set = [r for r in bucket if _is_in_set(r)]
        latencies = [r["latency_ms"] for r in bucket if r.get("latency_ms") is not None]
        prompt_toks = [r["prompt_tokens"] for r in bucket if r.get("prompt_tokens") is not None]
        comp_toks = [r["completion_tokens"] for r in bucket if r.get("completion_tokens") is not None]
        total_toks = [r["total_tokens"] for r in bucket if r.get("total_tokens") is not None]
        costs = [c for c in (_cost_for(r, model_costs) for r in bucket) if c is not None]

        acc = len(correct) / n if n else 0.0
        in_set_rate = len(in_set) / n if n else 0.0
        cost_total = sum(costs) if costs else 0.0
        cost_per_correct = (cost_total / len(correct)) if correct else math.inf

        out.append(
            {
                "model": model,
                "mode": mode,
                "n": n,
                "accuracy": acc,
                "in_set_rate": in_set_rate,
                "error_rate": len(errors) / n if n else 0.0,
                "prompt_tokens_mean": _mean(prompt_toks),
                "completion_tokens_mean": _mean(comp_toks),
                "total_tokens_mean": _mean(total_toks),
                "cost_total_usd": cost_total,
                "cost_per_call_usd": cost_total / n if n else 0.0,
                "cost_per_correct_usd": cost_per_correct,
                "latency_p50_ms": _percentile(latencies, 50),
                "latency_p95_ms": _percentile(latencies, 95),
            }
        )
    return out


def _aggregate_by_category(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    For each (mode, category), return (n_correct, n_total).
    Used to spot where filtered mode beats full mode.
    """
    by_mode_cat: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cat = r.get("category") or "uncategorized"
        by_mode_cat[r["mode"]][cat].append(_is_correct(r))

    out: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for mode, cat_map in by_mode_cat.items():
        out[mode] = {cat: (sum(vs), len(vs)) for cat, vs in cat_map.items()}
    return out


# ──────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x*100:.1f}%" if x is not None else "—"


def _fmt_int(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{int(round(x))}"


def _fmt_usd(x: Optional[float]) -> str:
    if x is None or x == math.inf:
        return "—"
    if x < 0.01:
        return f"${x*100:.3f}¢"
    return f"${x:.4f}"


def _render_main_table(summary: List[Dict[str, Any]], tiers: Dict[str, str]) -> str:
    headers = [
        "model",
        "tier",
        "mode",
        "n",
        "accuracy",
        "in-set",
        "errors",
        "prompt tok",
        "comp tok",
        "$/call",
        "$/correct",
        "p50 ms",
        "p95 ms",
    ]
    sep = ["---"] * len(headers)

    tier_order = {"frontier": 0, "mid": 1, "small": 2, "unknown": 3}
    summary_sorted = sorted(
        summary,
        key=lambda s: (
            tier_order.get(tiers.get(s["model"], "unknown"), 9),
            s["model"],
            0 if s["mode"] == "full" else 1,  # full first to make the diff easy to read
        ),
    )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for s in summary_sorted:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{s['model']}`",
                    tiers.get(s["model"], "—"),
                    s["mode"],
                    str(s["n"]),
                    _fmt_pct(s["accuracy"]),
                    _fmt_pct(s["in_set_rate"]),
                    _fmt_pct(s["error_rate"]),
                    _fmt_int(s["prompt_tokens_mean"]),
                    _fmt_int(s["completion_tokens_mean"]),
                    _fmt_usd(s["cost_per_call_usd"]),
                    _fmt_usd(s["cost_per_correct_usd"]),
                    _fmt_int(s["latency_p50_ms"]),
                    _fmt_int(s["latency_p95_ms"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _render_category_table(
    by_mode_cat: Dict[str, Dict[str, Tuple[int, int]]],
) -> str:
    """One row per category, columns = modes (full / filtered)."""
    if not by_mode_cat:
        return ""

    modes = sorted(by_mode_cat.keys())
    all_cats = sorted({c for m in by_mode_cat.values() for c in m.keys()})

    headers = ["category"] + [f"{m} (acc)" for m in modes]
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for cat in all_cats:
        row = [cat]
        for m in modes:
            stats = by_mode_cat[m].get(cat)
            if not stats:
                row.append("—")
            else:
                ok, total = stats
                row.append(f"{ok}/{total} ({_fmt_pct(ok/total if total else 0.0)})")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_pair_diff(summary: List[Dict[str, Any]], tiers: Dict[str, str]) -> str:
    """Per-model: show full vs filtered side by side with deltas."""
    by_model: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for s in summary:
        by_model[s["model"]][s["mode"]] = s

    headers = [
        "model",
        "tier",
        "Δ accuracy",
        "Δ prompt tok",
        "Δ $/correct",
        "Δ p50 ms",
    ]
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    tier_order = {"frontier": 0, "mid": 1, "small": 2, "unknown": 3}
    for model in sorted(
        by_model.keys(), key=lambda m: (tier_order.get(tiers.get(m, "unknown"), 9), m)
    ):
        full = by_model[model].get("full")
        filt = by_model[model].get("filtered")
        if not full or not filt:
            continue

        d_acc = (filt["accuracy"] - full["accuracy"]) * 100
        d_prompt = (filt["prompt_tokens_mean"] or 0) - (full["prompt_tokens_mean"] or 0)
        d_ppc = (filt["cost_per_correct_usd"] - full["cost_per_correct_usd"])
        d_lat = (filt["latency_p50_ms"] or 0) - (full["latency_p50_ms"] or 0)

        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{model}`",
                    tiers.get(model, "—"),
                    f"{d_acc:+.1f}pp",
                    f"{d_prompt:+.0f}",
                    f"{d_ppc:+.4f}" if d_ppc != math.inf and not math.isinf(d_ppc) else "—",
                    f"{d_lat:+.0f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_csv(summary: List[Dict[str, Any]], tiers: Dict[str, str]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model",
        "tier",
        "mode",
        "n",
        "accuracy",
        "in_set_rate",
        "error_rate",
        "prompt_tokens_mean",
        "completion_tokens_mean",
        "total_tokens_mean",
        "cost_total_usd",
        "cost_per_call_usd",
        "cost_per_correct_usd",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for s in summary:
            vals = [
                s["model"],
                tiers.get(s["model"], ""),
                s["mode"],
                str(s["n"]),
                f"{s['accuracy']:.4f}",
                f"{s['in_set_rate']:.4f}",
                f"{s['error_rate']:.4f}",
                f"{s['prompt_tokens_mean'] or 0:.2f}",
                f"{s['completion_tokens_mean'] or 0:.2f}",
                f"{s['total_tokens_mean'] or 0:.2f}",
                f"{s['cost_total_usd']:.6f}",
                f"{s['cost_per_call_usd']:.6f}",
                f"{s['cost_per_correct_usd']:.6f}" if s["cost_per_correct_usd"] != math.inf else "inf",
                f"{s['latency_p50_ms'] or 0:.0f}",
                f"{s['latency_p95_ms'] or 0:.0f}",
            ]
            f.write(",".join(vals) + "\n")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────


def main() -> int:
    rows = _load_results()
    if not rows:
        print("No results to score", file=sys.stderr)
        return 1

    model_costs, tiers = _load_model_costs()
    summary = _aggregate(rows, model_costs)
    by_mode_cat = _aggregate_by_category(rows)

    main_table = _render_main_table(summary, tiers)
    pair_table = _render_pair_diff(summary, tiers)
    cat_table = _render_category_table(by_mode_cat)

    report = "\n".join(
        [
            "# PRD-138 — Tool-routing eval results",
            "",
            f"Total cells: {len(rows)} across {len({(r['model'], r['mode']) for r in rows})} (model, mode) pairs.",
            "",
            "## Per (model, mode)",
            "",
            main_table,
            "",
            "## Δ filtered − full (per model)",
            "",
            "Positive Δ accuracy = filtered helps. Negative Δ tokens = filtered cheaper.",
            "",
            pair_table,
            "",
            "## Per category × mode",
            "",
            "Where does semantic routing actually win? Look for paraphrase + ambiguous + cross.",
            "",
            cat_table,
            "",
        ]
    )

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, encoding="utf-8")
    _write_csv(summary, tiers)

    print(report)
    print(f"\n[score] wrote {REPORT_MD}")
    print(f"[score] wrote {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
