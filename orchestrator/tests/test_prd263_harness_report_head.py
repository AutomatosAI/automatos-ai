"""PRD-263: the HARNESS audit report puts what asks for action first.

Auto reads the audit report through the chat, which keeps only the first TOOL_RESULT_TOKEN_BUDGET
tokens of a tool's answer (consumers/chatbot/tool_cut.py). The report listed every agent in name
order and the suggested changes after them, so on c1 (23 Oct 2026: 49 agents, two in REGRESSION)
the cut kept Brand Designer's regression and dropped STOPGAP's, and Auto told the owner the second
one was "in the part of the report I couldn't read". The weekly review must name the agent behind
each WATCH and REGRESSION (core/services/auto_cadence.py), and the suggested changes are the
report's actionable lines. Now the report runs: the counts, every agent in WATCH or REGRESSION,
the post-change checks, the issues, the suggested changes, and last the other agents.

Pure: a hand-built diagnosis rendered by the real report builder, cut by the real budget.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from consumers.chatbot.tool_cut import TOOL_RESULT_TOKEN_BUDGET
from modules.rag.budget import truncate_to_token_budget
from services import harness_diagnose as hd
from services.harness_service import SUGGESTIONS_HEADING, HarnessService

ACTIVE_AGENTS = 99   # well over the chat budget; c1 had 49 and was already over it
LATE_NAMES = ("Zephyr Desk", "Yardarm")     # regressions whose names sort after every other agent
WATCHED = "Xylo Watch"
MIN_SAMPLES = 20


def _rate(verdict: str, *, n_cur: int, n_ref: int, x_cur: int, x_ref: int, p: float) -> Dict[str, Any]:
    return {"unit": hd.UNIT_RATE, hd.VERDICT_KEY: verdict, "n_cur": n_cur, "n_ref": n_ref, "x_cur": x_cur,
            "x_ref": x_ref, "cur_value": x_cur / n_cur if n_cur else None,
            "ref_value": x_ref / n_ref if n_ref else None, "p_value": p, "reason": hd.REASON_STRONG_P}


def _entry(name: str, verdict: str) -> Dict[str, Any]:
    if verdict == "INSUFFICIENT_DATA":
        metric = _rate(verdict, n_cur=3, n_ref=11, x_cur=0, x_ref=1, p=1.0)
    elif verdict in ("REGRESSION", "WATCH"):
        metric = _rate(verdict, n_cur=52, n_ref=402, x_cur=13, x_ref=26, p=3.6e-06)
    else:
        metric = _rate(verdict, n_cur=40, n_ref=300, x_cur=2, x_ref=15, p=0.7)
    return {"agent_name": name, "model_id": "claude-haiku-5-5", hd.VERDICT_KEY: verdict,
            hd.METRICS_KEY: {"tool_failure": metric}}


def _diagnosis() -> Dict[str, Any]:
    agents = {str(i): _entry(f"Agent {i:02d}", "INSUFFICIENT_DATA" if i % 4 else "STABLE")
              for i in range(ACTIVE_AGENTS - 3)}
    agents.update({"r1": _entry(LATE_NAMES[0], "REGRESSION"), "r2": _entry(LATE_NAMES[1], "REGRESSION"),
                   "w1": _entry(WATCHED, "WATCH")})
    verdicts = [a[hd.VERDICT_KEY] for a in agents.values()]
    counts = {"active_agents": ACTIVE_AGENTS, "judged_agents": sum(v != "INSUFFICIENT_DATA" for v in verdicts),
              "insufficient_data": verdicts.count("INSUFFICIENT_DATA"), "stable": verdicts.count("STABLE"),
              "watch": verdicts.count("WATCH"), "regression": verdicts.count("REGRESSION")}
    return {"window": "2026-W42", "ref_weeks": ["2026-W38", "2026-W41"], "counts": counts,
            "coverage": counts["judged_agents"] / ACTIVE_AGENTS, hd.CONVERGENCE_KEY: hd.CONVERGENCE_INSUFFICIENT,
            "min_samples": MIN_SAMPLES, hd.AGENTS_KEY: agents, "issues": []}


def _suggestions() -> List[Dict[str, Any]]:
    return [{"target_name": LATE_NAMES[0], "change_type": "tool_assignment_remove", "prescription_id": "rx-late-1",
             "risk_score": 2, "rationale": "Its GMAIL calls failed 13 of 52 times this week."}]


def _report() -> str:
    return HarnessService()._build_changelog_markdown(1, _diagnosis(), _suggestions())


def test_flagged_agents_and_suggestions_come_before_the_other_agents():
    report = _report()
    other = report.index(hd.OTHER_AGENTS_HEADING)
    attention = report.index(hd.NEEDS_ATTENTION_HEADING)
    assert attention < report.index(SUGGESTIONS_HEADING) < other
    for name in (*LATE_NAMES, WATCHED):
        assert attention < report.index(f"**{name}**") < other
    assert report.index(f"**{LATE_NAMES[1]}**") < report.index(f"**{WATCHED}**")   # REGRESSION before WATCH
    assert "## Agents" not in report


def test_the_chats_cut_keeps_every_flagged_agent_and_the_suggestion():
    answer = json.dumps({"success": True, "data": {"title": "HARNESS Weekly Org Review — Run #1",
                                                   "content": _report()}})
    kept = truncate_to_token_budget(answer, TOOL_RESULT_TOKEN_BUDGET)
    assert len(kept) < len(answer), "the fixture must be over the chat's budget to prove anything"
    for name in (*LATE_NAMES, WATCHED):
        assert name in kept
    assert "rx-late-1" in kept


def test_no_flagged_agent_means_no_attention_section():
    diagnosis = _diagnosis()
    diagnosis[hd.AGENTS_KEY] = {k: v for k, v in diagnosis[hd.AGENTS_KEY].items()
                                if v[hd.VERDICT_KEY] not in ("REGRESSION", "WATCH")}
    head, tail = hd.report_lines(diagnosis), hd.report_tail_lines(diagnosis)
    assert hd.NEEDS_ATTENTION_HEADING not in head
    assert tail[0] == hd.OTHER_AGENTS_HEADING
    steady = [i for i, line in enumerate(tail) if ": Steady" in line]
    unjudged = [i for i, line in enumerate(tail) if ": Not enough data" in line]
    assert steady and unjudged and max(steady) < min(unjudged)   # steady before not enough data
