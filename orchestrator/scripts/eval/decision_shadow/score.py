"""Score the PRD-248 shadow log.

Reads the JSON-lines file the decision engine appends in shadow mode (one row
per classified turn: the tier's verdict, the engine's answers, field-by-field
agreement, latency) and prints what a go/no-go needs: coverage, agreement per
field overall and per tier, agreement by the engine's own confidence band
(the calibration check), latency percentiles, and how many turns the engine
would have decided on its own at the configured floor.

Run from ``orchestrator/``::

    python -m scripts.eval.decision_shadow.score            # DECISION_SHADOW_LOG_PATH
    python -m scripts.eval.decision_shadow.score path.jsonl # an explicit file

Stdlib only, so it also runs inside the backend container::

    docker exec automatos_backend python -m scripts.eval.decision_shadow.score
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

FIELDS = ("complexity", "action", "needs_memory", "needs_multi_agent", "tool_domain", "target_agent")
BANDS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))


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


def _rate(flags: Iterable[Optional[bool]]) -> str:
    known = [f for f in flags if f is not None]
    if not known:
        return "n/a"
    return f"{sum(1 for f in known if f) / len(known):.0%} ({len(known)})"


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


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0]) if args else _default_path()
    rows = load_rows(path)
    print(f"shadow log: {path}")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
