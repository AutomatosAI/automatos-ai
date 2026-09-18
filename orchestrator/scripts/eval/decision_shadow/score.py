"""Score the PRD-248 shadow log.

Reads the JSON-lines file the decision engine appends in shadow mode (one row
per classified turn: the tier's verdict, the engine's answers, field-by-field
agreement, latency) and prints what a go/no-go needs: coverage, agreement per
field overall and per tier, agreement by the engine's own confidence band
(the calibration check), latency percentiles, and how many turns the engine
would have decided on its own at the configured floor.

Run from ``orchestrator/``::

    python -m scripts.eval.decision_shadow.score                 # DECISION_SHADOW_LOG_PATH
    python -m scripts.eval.decision_shadow.score path.jsonl      # an explicit file
    python -m scripts.eval.decision_shadow.score --only sim      # PRD-247 simulation turns only
    python -m scripts.eval.decision_shadow.score --only real     # the operator's own turns only
    python -m scripts.eval.decision_shadow.score --purpose tool_rerank

Rows written under a PRD-247 campaign carry ``execution_id`` starting with
``sim:``; everything else is real traffic. Stdlib only, so it also runs inside
the backend container::

    docker exec automatos_backend python -m scripts.eval.decision_shadow.score
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

FIELDS = ("complexity", "action", "needs_memory", "needs_multi_agent", "tool_domain", "target_agent")
BANDS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))
SIM_PREFIX = "sim:"


def is_simulated(row: Dict[str, Any]) -> bool:
    return str(row.get("execution_id") or "").startswith(SIM_PREFIX)


def split_traffic(rows: List[Dict[str, Any]], only: Optional[str]) -> List[Dict[str, Any]]:
    """``only``: 'sim' keeps PRD-247 campaign rows, 'real' keeps the rest, None keeps all."""
    if only == "sim":
        return [r for r in rows if is_simulated(r)]
    if only == "real":
        return [r for r in rows if not is_simulated(r)]
    return rows


def parse_when(value: Optional[str]) -> Optional[float]:
    """An ISO-8601 timestamp (naive = UTC, 'Z' accepted) or epoch seconds → epoch
    seconds; None stays None. The customer-night ledger cuts by the same window."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    from datetime import datetime, timezone

    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def filter_window(
    rows: List[Dict[str, Any]], since: Optional[float] = None, until: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Rows whose ``ts`` (epoch seconds) falls in [since, until]; an unbounded
    side keeps everything on that side."""
    out = []
    for r in rows:
        try:
            ts = float(r.get("ts"))
        except (TypeError, ValueError):
            continue
        if since is not None and ts < since:
            continue
        if until is not None and ts > until:
            continue
        out.append(r)
    return out


def summarize_rerank(rows: List[Dict[str, Any]]) -> str:
    """The tool-rerank rows: how the judged cut compares with the embedding cut."""
    out: List[str] = []
    scored = [r for r in rows if not r.get("error") and isinstance(r.get("compare"), dict)]
    errors = Counter(str(r.get("error")) for r in rows if r.get("error"))
    out.append(f"tool_rerank rows={len(rows)} scored={len(scored)} errors={sum(errors.values())}")
    if errors:
        out.append("  errors: " + ", ".join(f"{k}×{v}" for k, v in errors.most_common(6)))
    if not scored:
        return "\n".join(out)
    latencies = [float(r["latency_ms"]) for r in scored if r.get("latency_ms") is not None]
    if latencies:
        out.append(
            f"  engine latency ms: p50={percentile(latencies, 50):.0f} "
            f"p95={percentile(latencies, 95):.0f} max={max(latencies):.0f}"
        )
    emb = [r["compare"]["embedding_size"] for r in scored]
    kept = [r["compare"]["rerank_size"] for r in scored]
    overlap = [r["compare"]["overlap"] for r in scored]
    out.append(
        f"  surface size: embedding mean={sum(emb) / len(emb):.1f} → judged mean={sum(kept) / len(kept):.1f}; "
        f"mean overlap={sum(overlap) / len(overlap):.1f}"
    )
    same_top = sum(1 for r in scored if r["compare"].get("same_top"))
    nothing = sum(1 for r in scored if r.get("nothing_fits"))
    out.append(f"  same top action: {same_top / len(scored):.0%}; nothing fits: {nothing / len(scored):.0%}")
    dropped = Counter(n for r in scored for n in r["compare"].get("dropped", []))
    added = Counter(n for r in scored for n in r["compare"].get("added", []))
    if dropped:
        out.append("  most dropped: " + ", ".join(f"{k}×{v}" for k, v in dropped.most_common(5)))
    if added:
        out.append("  most added: " + ", ".join(f"{k}×{v}" for k, v in added.most_common(5)))
    return "\n".join(out)


def _default_path() -> Path:
    try:
        from config import config

        return Path(config.DECISION_SHADOW_LOG_PATH)
    except Exception:  # noqa: BLE001 — standalone use without the app env
        return Path("logs/decision_shadow.jsonl")


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    rank = (pct / 100.0) * (len(s) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def _rate_value(flags: Iterable[Optional[bool]]) -> Dict[str, Any]:
    """{'rate': share of True among the known flags or None, 'n': known flags}."""
    known = [f for f in flags if f is not None]
    if not known:
        return {"rate": None, "n": 0}
    return {"rate": round(sum(1 for f in known if f) / len(known), 4), "n": len(known)}


def _rate(flags: Iterable[Optional[bool]]) -> str:
    value = _rate_value(flags)
    if value["rate"] is None:
        return "n/a"
    return f"{value['rate']:.0%} ({value['n']})"


def _latency(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "p50": round(percentile(values, 50) or 0.0, 1),
        "p95": round(percentile(values, 95) or 0.0, 1),
        "max": round(max(values), 1),
        "n": len(values),
    }


def classifier_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The classifier rows as numbers: coverage, routes, latency, tokens,
    agreement per field / per tier / per engine-confidence band, would-decide."""
    scored = [r for r in rows if not r.get("error") and r.get("agree")]
    errors = Counter(str(r.get("error")) for r in rows if r.get("error"))
    out: Dict[str, Any] = {"rows": len(rows), "scored": len(scored), "errors": dict(errors)}
    if not scored:
        return out
    out["routes"] = dict(Counter(f"{r.get('provider')}/{r.get('model')}" for r in scored))
    out["latency_ms"] = _latency([float(r["latency_ms"]) for r in scored if r.get("latency_ms") is not None])
    tokens = [int(r.get("input_tokens") or 0) for r in scored]
    out["input_tokens"] = {"mean": round(sum(tokens) / len(tokens), 1), "max": max(tokens)}
    out["agreement"] = {field: _rate_value(r["agree"].get(field) for r in scored) for field in FIELDS}
    by_tier: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored:
        by_tier[r.get("tier")].append(r)
    out["by_tier"] = {
        str(tier): {
            "n": len(group),
            "complexity": _rate_value(r["agree"].get("complexity") for r in group),
            "action": _rate_value(r["agree"].get("action") for r in group),
        }
        for tier, group in sorted(by_tier.items(), key=lambda kv: float(kv[0]) if kv[0] is not None else 99)
    }
    bands = []
    for lo, hi in BANDS:
        group = [r for r in scored if (c := _confidence_of(r)) is not None and lo <= c < hi]
        bands.append({
            "band": f"[{lo:.1f}, {min(hi, 1.0):.1f}{')' if hi <= 1.0 else ']'}",
            "n": len(group),
            "complexity": _rate_value(r["agree"].get("complexity") for r in group),
        })
    out["by_confidence_band"] = bands
    decided = sum(1 for r in scored if r.get("engine_verdict"))
    out["would_decide"] = {"n": decided, "share": round(decided / len(scored), 4)}
    return out


def rerank_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The tool-rerank rows as numbers: coverage, latency, surface sizes, overlap,
    same-top and nothing-fits shares, the most dropped and added actions."""
    scored = [r for r in rows if not r.get("error") and isinstance(r.get("compare"), dict)]
    errors = Counter(str(r.get("error")) for r in rows if r.get("error"))
    out: Dict[str, Any] = {"rows": len(rows), "scored": len(scored), "errors": dict(errors)}
    if not scored:
        return out
    out["routes"] = dict(Counter(f"{r.get('provider')}/{r.get('model')}" for r in scored))
    out["latency_ms"] = _latency([float(r["latency_ms"]) for r in scored if r.get("latency_ms") is not None])
    emb = [r["compare"]["embedding_size"] for r in scored]
    kept = [r["compare"]["rerank_size"] for r in scored]
    overlap = [r["compare"]["overlap"] for r in scored]
    out["surface"] = {
        "embedding_mean": round(sum(emb) / len(emb), 2),
        "judged_mean": round(sum(kept) / len(kept), 2),
        "overlap_mean": round(sum(overlap) / len(overlap), 2),
    }
    out["same_top_share"] = round(sum(1 for r in scored if r["compare"].get("same_top")) / len(scored), 4)
    out["nothing_fits_share"] = round(sum(1 for r in scored if r.get("nothing_fits")) / len(scored), 4)
    out["most_dropped"] = Counter(n for r in scored for n in r["compare"].get("dropped", [])).most_common(5)
    out["most_added"] = Counter(n for r in scored for n in r["compare"].get("added", [])).most_common(5)
    return out


def summary_dict(
    rows: List[Dict[str, Any]],
    *,
    only: Optional[str] = None,
    since: Optional[float] = None,
    until: Optional[float] = None,
) -> Dict[str, Any]:
    """Everything the night ledger needs from the shadow log, JSON-serialisable."""
    rows = split_traffic(rows, only)
    if since is not None or until is not None:
        rows = filter_window(rows, since, until)
    simulated = sum(1 for r in rows if is_simulated(r))
    return {
        "only": only,
        "window": {"since": since, "until": until},
        "rows": len(rows),
        "simulated": simulated,
        "real": len(rows) - simulated,
        "classifier": classifier_summary([r for r in rows if r.get("purpose", "classifier") == "classifier"]),
        "tool_rerank": rerank_summary([r for r in rows if r.get("purpose") == "tool_rerank"]),
    }


def _confidence_of(row: Dict[str, Any]) -> Optional[float]:
    answers = row.get("answers") or {}
    values = []
    for key in ("complexity", "action"):
        answer = answers.get(key) or {}
        conf = answer.get("confidence")
        if conf is None and isinstance(answer.get("probabilities"), dict) and answer["probabilities"]:
            conf = max(answer["probabilities"].values())
        if conf is not None:
            values.append(float(conf))
    return min(values) if values else None


def summarize(rows: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    scored = [r for r in rows if not r.get("error") and r.get("agree")]
    errors = Counter(str(r.get("error")) for r in rows if r.get("error"))
    out.append(f"rows={len(rows)} scored={len(scored)} errors={sum(errors.values())}")
    if errors:
        out.append("  errors: " + ", ".join(f"{k}×{v}" for k, v in errors.most_common(6)))
    if not scored:
        return "\n".join(out)

    providers = Counter(f"{r.get('provider')}/{r.get('model')}" for r in scored)
    out.append("  routes: " + ", ".join(f"{k}×{v}" for k, v in providers.most_common()))

    latencies = [float(r["latency_ms"]) for r in scored if r.get("latency_ms") is not None]
    if latencies:
        out.append(
            f"  engine latency ms: p50={percentile(latencies, 50):.0f} "
            f"p95={percentile(latencies, 95):.0f} max={max(latencies):.0f}"
        )
    tokens = [int(r.get("input_tokens") or 0) for r in scored]
    if tokens:
        out.append(f"  input tokens/call: mean={sum(tokens) / len(tokens):.0f} max={max(tokens)}")

    out.append("agreement with the tier that answered (share, n):")
    for field in FIELDS:
        out.append(f"  {field:<17} {_rate(r['agree'].get(field) for r in scored)}")

    by_tier: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored:
        by_tier[r.get("tier")].append(r)
    out.append("by tier (complexity / action agreement):")
    for tier in sorted(by_tier, key=lambda t: float(t) if t is not None else 99):
        group = by_tier[tier]
        out.append(
            f"  tier {tier}: n={len(group)} complexity={_rate(r['agree'].get('complexity') for r in group)} "
            f"action={_rate(r['agree'].get('action') for r in group)}"
        )

    out.append("by engine confidence band (complexity agreement — the calibration check):")
    for lo, hi in BANDS:
        group = [r for r in scored if (c := _confidence_of(r)) is not None and lo <= c < hi]
        label = f"[{lo:.1f}, {min(hi, 1.0):.1f}{')' if hi <= 1.0 else ']'}"
        out.append(f"  {label:<11} n={len(group):<4} {_rate(r['agree'].get('complexity') for r in group)}")

    decided = sum(1 for r in scored if r.get("engine_verdict"))
    out.append(f"would decide alone at the configured floor: {decided}/{len(scored)}")
    return "\n".join(out)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the PRD-248 shadow log")
    parser.add_argument("path", nargs="?", default=None, help="the JSON-lines file (default: DECISION_SHADOW_LOG_PATH)")
    parser.add_argument("--only", choices=["sim", "real"], default=None, help="PRD-247 campaign rows only, or the operator's own turns only")
    parser.add_argument("--purpose", choices=["classifier", "tool_rerank", "all"], default="all")
    parser.add_argument("--since", default=None, help="ISO-8601 or epoch seconds; rows at or after this (a customer night's start)")
    parser.add_argument("--until", default=None, help="ISO-8601 or epoch seconds; rows at or before this (the night's end)")
    parser.add_argument("--json", action="store_true", help="print summary_dict() as JSON and nothing else (for the night ledger)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    path = Path(args.path) if args.path else _default_path()
    since, until = parse_when(args.since), parse_when(args.until)
    if args.json:
        print(json.dumps(summary_dict(load_rows(path), only=args.only, since=since, until=until), indent=2))
        return 0
    rows = split_traffic(load_rows(path), args.only)
    if since is not None or until is not None:
        rows = filter_window(rows, since, until)
    simulated = sum(1 for r in rows if is_simulated(r))
    window = f" window=[{args.since or '…'}, {args.until or '…'}]" if (since is not None or until is not None) else ""
    print(f"shadow log: {path}")
    print(f"rows={len(rows)} simulated={simulated} real={len(rows) - simulated}" + (f" (only={args.only})" if args.only else "") + window)
    if args.purpose in ("classifier", "all"):
        print("== classifier ==")
        print(summarize([r for r in rows if r.get("purpose", "classifier") == "classifier"]))
    if args.purpose in ("tool_rerank", "all"):
        print("== tool_rerank ==")
        print(summarize_rerank([r for r in rows if r.get("purpose") == "tool_rerank"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
