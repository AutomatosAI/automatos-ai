"""The scorecard (PRD-247 S0.4): four rows a customer would recognise.

    usability  — did it just work? deductions for every error, every extra
                 question, every nudge the runner had to give, every empty reply
    cost       — dollars from llm_usage, attributed per scenario and in total
    quality    — the judge's grade (1–5 → 0–1), else the pack's cheap checks
    usefulness — the judge's grade, else "a deliverable exists and the ticket closed"

Every number is explainable: ``deductions`` lists what usability lost and
why; findings quote the evidence they rest on and are all trace-backed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .results import ScenarioResult

OUTCOME_SCORE = {"done": 1.0, "ok": 1.0, "review": 0.5, "timeout": 0.0, "failed": 0.0, "blocked": 0.0,
                 "cancelled": 0.0, "error": 0.0}
ERROR_PENALTY, ASK_PENALTY, NUDGE_PENALTY, EMPTY_REPLY_PENALTY, CHECK_PENALTY = 0.25, 0.10, 0.25, 0.50, 0.10
FREE_ASKS = 1


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def usability(res: ScenarioResult) -> tuple[float, tuple[str, ...]]:
    deductions: list[tuple[float, str]] = []
    for error in res.errors:
        deductions.append((ERROR_PENALTY, f"error: {error[:120]}"))
    extra_asks = max(0, len(res.asks) - FREE_ASKS)
    if extra_asks:
        deductions.append((ASK_PENALTY * extra_asks, f"{len(res.asks)} questions/approvals needed (first is free)"))
    for note in res.notes:
        if "run-now sent" in note:
            deductions.append((NUDGE_PENALTY, note))
    if res.kind == "chat" and not any(t.get("text") for t in res.chat):
        deductions.append((EMPTY_REPLY_PENALTY, "no reply text on any turn"))
    for check in res.checks:
        if not check.ok and not check.name.startswith("outcome") and not check.name.startswith("effect:status"):
            deductions.append((CHECK_PENALTY, f"check failed: {check.name} ({check.detail})"))
    score = _clamp(1.0 - sum(amount for amount, _ in deductions))
    return score, tuple(f"-{amount:.2f} {why}" for amount, why in deductions)


def quality(res: ScenarioResult, verdict: Mapping[str, Any] | None) -> tuple[float | None, str]:
    if verdict and "quality" in verdict:
        return round((int(verdict["quality"]) - 1) / 4, 3), "judge"
    contains = [c for c in res.checks if c.name.startswith("contains:")]
    if contains:
        return round(sum(1 for c in contains if c.ok) / len(contains), 3), "must_contain"
    return None, "no judge and no must_contain checks"


def usefulness(res: ScenarioResult, verdict: Mapping[str, Any] | None) -> tuple[float | None, str]:
    if verdict and "usefulness" in verdict:
        return round((int(verdict["usefulness"]) - 1) / 4, 3), "judge"
    if res.kind == "task":
        delivered = bool(res.artifacts.get("deliverables") or res.artifacts.get("reports"))
        return (1.0 if delivered and res.outcome == "done" else 0.5 if delivered else 0.0), "artifact+outcome"
    if res.kind == "chat":
        return (1.0 if res.ok else 0.0), "checks"
    return None, "not applicable"


def score_scenario(res: ScenarioResult, cost: Mapping[str, Any], verdict: Mapping[str, Any] | None) -> dict[str, Any]:
    use, deductions = usability(res)
    q, q_basis = quality(res, verdict)
    u, u_basis = usefulness(res, verdict)
    return {
        "outcome": OUTCOME_SCORE.get(res.outcome, 0.0), "usability": use, "deductions": list(deductions),
        "quality": q, "quality_basis": q_basis, "usefulness": u, "usefulness_basis": u_basis,
        "cost_usd": float(cost.get("cost_usd") or 0.0), "calls": int(cost.get("calls") or 0),
        "duration_s": res.duration_s,
    }


def findings(res: ScenarioResult, cost: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Trace-backed statements about one scenario, each pointing at its evidence."""
    out: list[dict[str, Any]] = []

    def add(kind: str, text: str, evidence: Any) -> None:
        out.append({"scenario": res.id, "kind": kind, "text": text, "evidence": evidence, "basis": "trace"})

    if res.outcome == "timeout":
        add("stall", f"{res.id}: ticket never reached a terminal state", {"timeline": list(res.timeline)})
    if res.outcome == "review":
        add("review", f"{res.id}: ticket ended in review", {"review_feedback": res.artifacts.get("review_feedback")})
    if res.outcome in ("failed", "blocked", "cancelled"):
        add("failure", f"{res.id}: ticket ended '{res.outcome}'", {"errors": list(res.errors)})
    for error in res.errors:
        add("error", f"{res.id}: {error[:200]}", None)
    for note in res.notes:
        if "run-now sent" in note or "runtime_ref" in note:
            add("nudge" if "run-now" in note else "denial", f"{res.id}: {note}", None)
    if res.kind == "task" and res.outcome == "done" and not (res.artifacts.get("deliverables") or res.artifacts.get("reports")):
        add("missing_artifact", f"{res.id}: closed as done with no deliverable or report to show for it",
            {"result_chars": len(str((res.task or {}).get("result") or ""))})
    if res.asks:
        add("asks", f"{res.id}: {len(res.asks)} questions/approvals were raised", {"asks": list(res.asks)[:5]})
    if int(cost.get("errors") or 0):
        add("model_errors", f"{res.id}: {cost['errors']} model calls did not succeed", {"by_model": cost.get("by_model")})
    for check in res.checks:
        if not check.ok:
            add("check", f"{res.id}: {check.name} — {check.detail}", None)
    return tuple(out)


def _mean(values: Sequence[float | None]) -> float | None:
    real = [v for v in values if isinstance(v, (int, float))]
    return round(sum(real) / len(real), 3) if real else None


def score_pack(scenarios: Sequence[Mapping[str, Any]], workspace_total_usd: float) -> dict[str, Any]:
    scores = [s.get("scores") or {} for s in scenarios]
    return {
        "scenarios": len(scenarios), "ok": sum(1 for s in scenarios if s.get("ok")),
        "outcome_rate": _mean([s.get("outcome") for s in scores]),
        "usability": _mean([s.get("usability") for s in scores]),
        "quality": _mean([s.get("quality") for s in scores]),
        "usefulness": _mean([s.get("usefulness") for s in scores]),
        "cost_usd": round(workspace_total_usd, 4),
        "attributed_usd": round(sum(float(s.get("cost_usd") or 0.0) for s in scores), 4),
        "calls": sum(int(s.get("calls") or 0) for s in scores),
        "duration_s": round(sum(float(s.get("duration_s") or 0.0) for s in scores), 1),
    }


def _fmt(value: float | None, scale: float = 1.0, suffix: str = "") -> str:
    return "—" if value is None else f"{value * scale:.{0 if scale == 100 else 3}f}{suffix}"


def render_markdown(run: Mapping[str, Any]) -> str:
    card = run.get("scorecard") or {}
    cost = run.get("cost") or {}
    lines = [
        f"# Simulation night — pack `{run.get('pack')}` · {run.get('started_at', '')[:16]}",
        "",
        f"Run `{run.get('run_id')}` · status **{run.get('status')}** · model `{(run.get('model') or {}).get('model_id')}` "
        f"· workspace `{(run.get('workspace') or {}).get('slug')}`"
        + (" (kept)" if not (run.get('workspace') or {}).get('purged') else " (purged)"),
        "",
        "| Row | Score | Basis |", "|---|---|---|",
        f"| Usability | {_fmt(card.get('usability'), 100, '%')} | errors, extra questions, nudges, empty replies |",
        f"| Cost | ${card.get('cost_usd', 0):.4f} (attributed ${card.get('attributed_usd', 0):.4f}, {card.get('calls', 0)} calls) | llm_usage, {cost.get('source', '?')} |",
        f"| Quality | {_fmt(card.get('quality'), 100, '%')} | judge or must_contain |",
        f"| Usefulness | {_fmt(card.get('usefulness'), 100, '%')} | judge or artifact+outcome |",
        f"| Outcomes | {card.get('ok', 0)}/{card.get('scenarios', 0)} scenarios passed all checks; outcome rate {_fmt(card.get('outcome_rate'), 100, '%')} | board status / checks |",
        "",
        "## Scenarios", "",
        "| id | kind | outcome | secs | usability | quality | usefulness | cost | calls |", "|---|---|---|---|---|---|---|---|---|",
    ]
    for sc in run.get("scenarios") or []:
        s = sc.get("scores") or {}
        lines.append(f"| {sc.get('id')} | {sc.get('kind')} | {sc.get('outcome')} | {s.get('duration_s', 0):.0f} | "
                     f"{_fmt(s.get('usability'), 100, '%')} | {_fmt(s.get('quality'), 100, '%')} | {_fmt(s.get('usefulness'), 100, '%')} | "
                     f"${s.get('cost_usd', 0):.4f} | {s.get('calls', 0)} |")
    lines += ["", "## Findings (all trace-backed)", ""]
    found = run.get("findings") or []
    lines += [f"- **{f.get('kind')}** — {f.get('text')}" for f in found] or ["- none"]
    notes = run.get("notes") or []
    if notes:
        lines += ["", "## Runner notes", ""] + [f"- {n}" for n in notes]
    lines += ["", f"Models seen in llm_usage: {', '.join(cost.get('models_seen') or []) or '—'}", ""]
    return "\n".join(lines)
