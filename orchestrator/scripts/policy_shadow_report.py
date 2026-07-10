#!/usr/bin/env python3
"""PRD-192 S2 (P2-11): read-only policy-plane shadow-stage evidence report.

Stage advancement on the ``AUTOMATOS_POLICY_PLANE`` mode dial is an EVIDENCE
call, not a vibe. This script aggregates the policy-verdict ``audit_logs`` rows
the plane writes (PRD-181's handler + the PRD-192 ``mode``/``risk`` keys) over
a window and prints:

  1. the verdict rollup — decision × risk × mode, top tools, actor split;
  2. would-block counts — deny/ask verdicts that shadow (or the destructive
     stage's open classes) let proceed: exactly what enforcement WOULD stop;
  3. the fail-open / plane-fault rate (dossier G.5) — ``[policy-fail-open]``
     marker rows (open-class faults that proceeded) + ``policy_plane_error``
     denials (closed-class faults that blocked);
  4. enforcement-coverage by lane (G.1) — which tool families produce verdicts;
  5. priced-call % (G.2) — the fraction of gate evaluations that carried a
     non-zero token estimate (PRD-192 S3 threads them; structural 0 before);
  6. stage-gate verdicts against the rollout criteria (Gerard's box #3).

Rollout runbook (stage flips are Gerard's ops actions on Railway — code ships
default ``off``; the kill switch is always the previous stage value):

  1. ``AUTOMATOS_POLICY_PLANE=shadow`` for ≥ 7 days.
  2. Run this report: fail-open rate ~0, would-block verdicts reviewed with a
     false-block rate < 5% (human review over §2's list), row volume sane.
  3. ``AUTOMATOS_POLICY_PLANE=destructive`` for ≥ 7 days — with PRD-193's
     approval card live so `ask` verdicts are answerable, not dead-ends.
  4. ``AUTOMATOS_POLICY_PLANE=on``; then default-on in ``envs/api.defaults``.

Run against a prod-configured environment (DB reachable; read-only)::

    python -m scripts.policy_shadow_report                # last 7 days
    python -m scripts.policy_shadow_report --days 14
    python -m scripts.policy_shadow_report --workspace <uuid>
    python -m scripts.policy_shadow_report --json          # machine-readable

Import-safe (no work at import time): the pure aggregation / stage-gate logic
— :func:`aggregate_rows` / :func:`stage_gate_verdicts` — is unit-tested with no
DB or network.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Stage-gate thresholds (Gerard's box #3 defaults — CLI-overridable)
# ---------------------------------------------------------------------------

# G.5: plane faults per evaluated call. "~0" — anything above this is ATTENTION.
FAIL_OPEN_RATE_MAX = 0.005
# Volume sanity: fewer rows than this over the window means the plane is not
# actually carrying load (dark lane, broken audit, or no traffic) — advancing
# a stage on no evidence is a vibe, not a call.
MIN_ROWS_SANE = 50
# The false-block criterion (< 5%) is a HUMAN review over the would-block list
# — the report surfaces the list and states the criterion; it cannot label
# blocks true/false by itself.
FALSE_BLOCK_RATE_MAX = 0.05

# The greppable marker the executor logs+audits on an open-class plane fault.
FAIL_OPEN_MARKER = "[policy-fail-open]"
# The errors-as-data code a closed-class plane fault denies with.
PLANE_ERROR_CODE = "policy_plane_error"

_BLOCKING = ("deny", "ask")


# ---------------------------------------------------------------------------
# Pure aggregation (unit-tested, no I/O)
# ---------------------------------------------------------------------------


def _lane_for_tool(tool_name: str) -> str:
    """Bucket a tool name into its execution lane (G.1 coverage view)."""
    name = (tool_name or "").strip()
    lowered = name.lower()
    if lowered.startswith("workspace_exec") or lowered.startswith("workspace_git"):
        return "workspace-exec"
    if lowered.startswith("workspace_"):
        return "workspace"
    if lowered.startswith("platform_"):
        return "platform"
    if lowered.startswith("composio_") or lowered == "composio_execute":
        return "composio"
    # Per-action Composio names are UPPER_SNAKE (GMAIL_SEND_EMAIL, ...).
    if name.isupper() and "_" in name:
        return "composio"
    return "builtin"


def aggregate_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate policy-verdict audit rows into the report summary. Pure.

    Each row is a dict with (at least) the keys the fetch produces:
    ``action`` ("policy:allow" | "policy:ask" | "policy:deny"),
    ``resource_name`` (the tool), ``actor_type``, and ``details`` (the
    handler's dict: verdict / risk / mode / reason / error_code /
    est_tokens...). Missing keys degrade gracefully — the report must render
    over rows written before every key existed.
    """
    total = 0
    by_decision: Counter = Counter()
    by_decision_risk: Counter = Counter()
    by_mode: Counter = Counter()
    by_lane: Counter = Counter()
    by_actor: Counter = Counter()
    tool_blocks: Counter = Counter()
    would_block: List[Dict[str, Any]] = []
    fail_open_count = 0
    plane_error_count = 0
    budget_evals = 0
    priced_calls = 0

    for row in rows:
        details = row.get("details") or {}
        if not isinstance(details, dict):
            details = {}
        decision = (
            details.get("verdict")
            or str(row.get("action") or "").replace("policy:", "", 1)
            or "unknown"
        )
        risk = details.get("risk") or "unknown"
        mode = details.get("mode") or "unknown"
        tool = row.get("resource_name") or "unknown"
        actor = details.get("actor_type") or row.get("actor_type") or "unknown"
        reason = str(details.get("reason") or "")

        total += 1
        by_decision[decision] += 1
        by_decision_risk[(decision, risk)] += 1
        by_mode[mode] += 1
        by_lane[_lane_for_tool(tool)] += 1
        by_actor[actor] += 1

        if decision in _BLOCKING:
            tool_blocks[tool] += 1
            would_block.append(
                {
                    "tool": tool,
                    "decision": decision,
                    "risk": risk,
                    "mode": mode,
                    "actor_type": actor,
                    "error_code": details.get("error_code"),
                }
            )

        if FAIL_OPEN_MARKER in reason:
            fail_open_count += 1
        if details.get("error_code") == PLANE_ERROR_CODE:
            plane_error_count += 1

        # G.2 priced-call %: rows that carry the estimate keys are budget
        # evaluations; non-zero estimates mean the gate priced the call.
        if "est_tokens" in details:
            budget_evals += 1
            try:
                if int(details.get("est_tokens") or 0) > 0:
                    priced_calls += 1
            except (TypeError, ValueError):
                pass

    fault_count = fail_open_count + plane_error_count
    return {
        "total": total,
        "by_decision": dict(by_decision),
        "by_decision_risk": {
            f"{d}/{r}": c for (d, r), c in sorted(by_decision_risk.items())
        },
        "by_mode": dict(by_mode),
        "by_lane": dict(by_lane),
        "by_actor": dict(by_actor),
        "would_block_count": len(would_block),
        "would_block_top_tools": tool_blocks.most_common(15),
        "would_block_sample": would_block[:50],
        "fail_open_count": fail_open_count,
        "plane_error_count": plane_error_count,
        "fault_rate": (fault_count / total) if total else 0.0,
        "budget_evals": budget_evals,
        "priced_calls": priced_calls,
        "priced_call_pct": (priced_calls / budget_evals) if budget_evals else 0.0,
    }


def stage_gate_verdicts(
    summary: Dict[str, Any],
    *,
    fail_open_rate_max: float = FAIL_OPEN_RATE_MAX,
    min_rows: int = MIN_ROWS_SANE,
) -> List[Dict[str, str]]:
    """Judge the summary against the stage-advance criteria (box #3). Pure.

    Returns ``[{"gate", "verdict", "detail"}, ...]`` where verdict is
    ``PASS`` | ``ATTENTION`` | ``REVIEW`` (REVIEW = needs the human call the
    report cannot make — the false-block criterion).
    """
    verdicts: List[Dict[str, str]] = []

    total = summary.get("total", 0)
    verdicts.append(
        {
            "gate": f"row volume sane (≥ {min_rows} verdict rows in window)",
            "verdict": "PASS" if total >= min_rows else "ATTENTION",
            "detail": f"{total} policy-verdict rows",
        }
    )

    rate = summary.get("fault_rate", 0.0)
    verdicts.append(
        {
            "gate": f"fail-open / plane-fault rate ~0 (≤ {fail_open_rate_max:.2%})",
            "verdict": "PASS" if rate <= fail_open_rate_max else "ATTENTION",
            "detail": (
                f"{rate:.4%} — {summary.get('fail_open_count', 0)} fail-open "
                f"(marker) + {summary.get('plane_error_count', 0)} fail-closed "
                f"({PLANE_ERROR_CODE})"
            ),
        }
    )

    wb = summary.get("would_block_count", 0)
    verdicts.append(
        {
            "gate": (
                f"would-block verdicts reviewed (false-block rate < "
                f"{FALSE_BLOCK_RATE_MAX:.0%} is a human review over the list)"
            ),
            "verdict": "REVIEW" if wb else "PASS",
            "detail": f"{wb} would-block verdicts (see the sample/table)",
        }
    )

    priced = summary.get("priced_call_pct", 0.0)
    evals = summary.get("budget_evals", 0)
    verdicts.append(
        {
            "gate": "priced-call % off its structural 0 (G.2 — informational)",
            "verdict": "PASS" if (evals and priced > 0) else "ATTENTION",
            "detail": (
                f"{priced:.1%} of {evals} estimate-carrying evaluations priced"
                if evals
                else "no estimate-carrying rows in window (S3 threads them)"
            ),
        }
    )

    return verdicts


# ---------------------------------------------------------------------------
# I/O — read-only fetch + rendering (no unit-test coverage required)
# ---------------------------------------------------------------------------


def fetch_rows(
    db: Any, *, days: int, workspace_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Read policy-verdict audit rows for the window. Read-only."""
    from datetime import datetime, timedelta, timezone

    from core.workspaces.audit import AuditLog

    since = datetime.now(timezone.utc) - timedelta(days=days)
    q = (
        db.query(AuditLog)
        .filter(AuditLog.action.like("policy:%"))
        .filter(AuditLog.created_at >= since)
    )
    if workspace_id:
        q = q.filter(AuditLog.workspace_id == workspace_id)
    return [
        {
            "action": r.action,
            "resource_name": r.resource_name,
            "actor_type": r.actor_type,
            "details": r.details or {},
            "created_at": r.created_at,
        }
        for r in q.all()
    ]


def render_text(summary: Dict[str, Any], verdicts: List[Dict[str, str]], days: int) -> str:
    lines: List[str] = []
    add = lines.append
    add(f"POLICY-PLANE SHADOW REPORT — last {days} day(s)")
    add("=" * 60)
    add(f"verdict rows: {summary['total']}")
    add("")
    add("decision × risk:")
    for key, count in summary["by_decision_risk"].items():
        add(f"  {key:<40} {count}")
    add("")
    add(f"by mode:  {summary['by_mode']}")
    add(f"by lane:  {summary['by_lane']}")
    add(f"by actor: {summary['by_actor']}")
    add("")
    add(f"would-block verdicts: {summary['would_block_count']}")
    for tool, count in summary["would_block_top_tools"]:
        add(f"  {tool:<40} {count}")
    add("")
    add(
        f"plane faults: {summary['fail_open_count']} fail-open (marker) + "
        f"{summary['plane_error_count']} fail-closed ({PLANE_ERROR_CODE}) "
        f"= rate {summary['fault_rate']:.4%}"
    )
    add(
        f"priced-call %: {summary['priced_call_pct']:.1%} "
        f"({summary['priced_calls']}/{summary['budget_evals']} estimate-carrying evaluations)"
    )
    add("")
    add("STAGE-GATE VERDICTS (box #3 criteria — flips are ops actions):")
    for v in verdicts:
        add(f"  [{v['verdict']:<9}] {v['gate']}")
        add(f"              {v['detail']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=7, help="window in days (default 7)")
    parser.add_argument("--workspace", type=str, default=None, help="limit to one workspace uuid")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--min-rows", type=int, default=MIN_ROWS_SANE,
        help=f"volume-sanity threshold (default {MIN_ROWS_SANE})",
    )
    args = parser.parse_args(argv)

    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        rows = fetch_rows(db, days=args.days, workspace_id=args.workspace)
    finally:
        db.close()

    summary = aggregate_rows(rows)
    verdicts = stage_gate_verdicts(summary, min_rows=args.min_rows)

    if args.json:
        print(json.dumps({"summary": summary, "stage_gates": verdicts}, default=str, indent=2))
    else:
        print(render_text(summary, verdicts, args.days))

    return 0


if __name__ == "__main__":
    sys.exit(main())
