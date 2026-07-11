"""PRD-192 S2 (P2-11) — shadow-stage evidence report.

Pins the PURE aggregation + stage-gate logic of
``scripts/policy_shadow_report.py`` over seeded audit-row dicts — no DB, no
network (the script is import-safe by design, mirroring
``probe_document_vectors.py`` from PRD-185 S8).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from scripts.policy_shadow_report import (  # noqa: E402
    FAIL_OPEN_MARKER,
    PLANE_ERROR_CODE,
    aggregate_rows,
    stage_gate_verdicts,
)


def _row(
    decision="allow",
    risk="read",
    mode="shadow",
    tool="platform_get_agent",
    actor="agent",
    reason="",
    error_code=None,
    est_tokens=None,
):
    details = {
        "verdict": decision,
        "risk": risk,
        "mode": mode,
        "reason": reason,
        "actor_type": actor,
        "error_code": error_code,
    }
    if est_tokens is not None:
        details["est_tokens"] = est_tokens
    return {
        "action": f"policy:{decision}",
        "resource_name": tool,
        "actor_type": actor,
        "details": details,
    }


def test_shadow_report_aggregates():
    """The verdict × risk rollup, would-block list, and fail-open-rate math."""
    rows = [
        _row("allow", "read", "shadow"),
        _row("allow", "internal_write", "shadow", tool="write_file"),
        _row("deny", "external_side_effect", "shadow", tool="GMAIL_SEND_EMAIL"),
        _row("ask", "destructive", "shadow", tool="platform_delete_agent"),
        _row(
            "allow", "read", "shadow",
            reason=f"{FAIL_OPEN_MARKER} plane fault — proceeded (risk=read, mode=shadow)",
        ),
        _row(
            "deny", "external_side_effect", "on",
            tool="GMAIL_SEND_EMAIL", error_code=PLANE_ERROR_CODE,
        ),
    ]
    summary = aggregate_rows(rows)

    assert summary["total"] == 6
    assert summary["by_decision"] == {"allow": 3, "deny": 2, "ask": 1}
    assert summary["by_decision_risk"]["deny/external_side_effect"] == 2
    assert summary["by_decision_risk"]["ask/destructive"] == 1
    assert summary["by_mode"] == {"shadow": 5, "on": 1}

    # would-block = every deny/ask verdict.
    assert summary["would_block_count"] == 3
    assert dict(summary["would_block_top_tools"])["GMAIL_SEND_EMAIL"] == 2

    # G.5 fault-rate math: 1 fail-open marker + 1 policy_plane_error over 6.
    assert summary["fail_open_count"] == 1
    assert summary["plane_error_count"] == 1
    assert summary["fault_rate"] == (2 / 6)


def test_shadow_report_lane_buckets():
    rows = [
        _row(tool="GMAIL_SEND_EMAIL"),
        _row(tool="composio_execute"),
        _row(tool="platform_list_agents"),
        _row(tool="workspace_exec"),
        _row(tool="write_file"),
    ]
    summary = aggregate_rows(rows)
    assert summary["by_lane"] == {
        "composio": 2,
        "platform": 1,
        "workspace-exec": 1,
        "builtin": 1,
    }


def test_shadow_report_priced_call_pct():
    """G.2: fraction of estimate-carrying evaluations with non-zero tokens."""
    rows = [
        _row(est_tokens=0),
        _row(est_tokens=1200),
        _row(est_tokens=800),
        _row(),  # legacy row without the key — not a budget evaluation
    ]
    summary = aggregate_rows(rows)
    assert summary["budget_evals"] == 3
    assert summary["priced_calls"] == 2
    assert summary["priced_call_pct"] == (2 / 3)


def test_report_flags_stage_gate():
    """Over/under the advance thresholds yields the right verdict lines."""
    # Healthy: enough rows, zero faults, would-blocks present, priced calls.
    healthy_rows = [_row(est_tokens=1000) for _ in range(60)] + [
        _row("deny", "external_side_effect", tool="GMAIL_SEND_EMAIL")
    ]
    healthy = stage_gate_verdicts(aggregate_rows(healthy_rows), min_rows=50)
    by_gate = {v["gate"]: v["verdict"] for v in healthy}
    assert by_gate[[g for g in by_gate if "row volume" in g][0]] == "PASS"
    assert by_gate[[g for g in by_gate if "fail-open" in g][0]] == "PASS"
    assert by_gate[[g for g in by_gate if "would-block" in g][0]] == "REVIEW"
    assert by_gate[[g for g in by_gate if "priced-call" in g][0]] == "PASS"

    # Unhealthy: thin volume + a fault rate way over the ceiling, nothing priced.
    faulty_rows = [
        _row(reason=f"{FAIL_OPEN_MARKER} plane fault"),
        _row("deny", error_code=PLANE_ERROR_CODE),
        _row(),
    ]
    unhealthy = stage_gate_verdicts(aggregate_rows(faulty_rows), min_rows=50)
    by_gate = {v["gate"]: v["verdict"] for v in unhealthy}
    assert by_gate[[g for g in by_gate if "row volume" in g][0]] == "ATTENTION"
    assert by_gate[[g for g in by_gate if "fail-open" in g][0]] == "ATTENTION"
    assert by_gate[[g for g in by_gate if "priced-call" in g][0]] == "ATTENTION"


def test_report_tolerates_malformed_rows():
    """Rows written before every key existed (or with corrupt details) must
    not break the report — they degrade to 'unknown' buckets."""
    rows = [
        {"action": "policy:allow", "resource_name": None, "actor_type": None, "details": None},
        {"action": "policy:deny", "resource_name": "t", "actor_type": "agent", "details": "corrupt"},
        {"action": None, "resource_name": "t2", "actor_type": "user", "details": {}},
    ]
    summary = aggregate_rows(rows)
    assert summary["total"] == 3
    assert summary["by_decision"]["allow"] == 1
    assert summary["by_decision"]["deny"] == 1
    assert summary["would_block_count"] == 1
