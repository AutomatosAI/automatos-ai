"""PRD-234 S2 — the "Claude Code session" section of a task report.

Kept apart from ``api.board_tasks`` so it stays a pure function: what ran, where,
what it produced, what was refused and how to take it over. Empty for an API run,
so the report shape stays uniform across runtimes.
"""
from __future__ import annotations

from typing import Any, Dict, List

from core.cli_runtime import RUNTIME_CLI
from services.session_denials import denial_kind_label, group_denials_by_kind


def refused_calls_lines(denials: List[Any]) -> List[str]:
    """PRD-245 S0.3: the refusals grouped by what they MEAN (D6) — holds first,
    since they are why a ticket sits in review; a refused read outside the
    directory or a tool a session never has is listed, not blamed."""
    lines: List[str] = ["### Refused tool calls"]
    for kind, rows in group_denials_by_kind(denials).items():
        lines.append(f"**{denial_kind_label(kind)}**")
        for d in rows:
            lines.append(f"- {d.get('tool') or '?'}: {d.get('reason') or d.get('subject') or 'refused'}")
    return lines


# F167: a hold's outcome, in the report's words. An approval the host reported
# that the backend has no record of (cli_host_service.approval_on_record) is
# never shown as the operator's.
APPROVAL_NOT_ON_RECORD = "approval not on record"
HOLD_OUTCOMES = {
    "approved": "the operator approved it",
    "denied": "the operator denied it",
    "no answer": "no answer in time, so it did not run",
    APPROVAL_NOT_ON_RECORD: "the host reported an approval that is not on record",
}


def tool_call_verdict(entry: Dict[str, Any]) -> str:
    """F167: what the host decided for one call — "" when the host did not say
    (older hosts). A call that ran with nobody asked never reads as approved."""
    decision, reason = entry.get("decision"), entry.get("reason")
    because = f" ({reason})" if reason else ""
    if decision == "allow":
        return f"ran, nobody was asked{because}"
    if decision == "ask":
        return f"held for the operator: {HOLD_OUTCOMES.get(entry.get('answer'), 'no answer recorded')}"
    if decision == "deny":
        return f"refused by the gate{because}"
    return ""


def tool_decisions_line(tally: Dict[str, Any]) -> str:
    """F167: every call of the session, counted by what the host decided."""
    held, unrecorded = int(tally.get("ask") or 0), int(tally.get("unrecorded") or 0)
    not_on_record = f", {unrecorded} approval{'s' if unrecorded != 1 else ''} not on record" if unrecorded else ""
    held_part = f"{held} held for the operator ({int(tally.get('approved') or 0)} approved{not_on_record})" if held \
        else "none held for the operator"
    return (f"- Tool calls: {int(tally.get('allow') or 0)} ran with nobody asked · {held_part} · "
            f"{int(tally.get('deny') or 0)} refused by the gate")


def session_report_lines(exec_result: Dict[str, Any]) -> List[str]:
    """PRD-234 S2: the "Claude Code session" part of a task report — what ran,
    where, what it produced, what was refused and how to take it over. Empty for
    an API run, so the report shape stays uniform."""
    if exec_result.get("runtime") != RUNTIME_CLI:
        return []
    session = exec_result.get("session") or {}
    usage = exec_result.get("usage") or {}
    lines: List[str] = ["## Claude Code session"]
    sid = session.get("session_id") or exec_result.get("session_id") or "unknown"
    lines.append(f"- Session: {sid}")
    model = usage.get("model") or session.get("model") or "the CLI's default model"
    lines.append(f"- Model: {model} — tokens in/out {usage.get('input_tokens', 0)} / {usage.get('output_tokens', 0)} (plan usage, no cost)")
    if session.get("cwd"):
        lines.append(f"- Working directory: {session['cwd']}")
    if session.get("exit_reason"):
        lines.append(f"- Ended: {session['exit_reason']}")
    if isinstance(session.get("tool_decisions"), dict) and session["tool_decisions"]:
        lines.append(tool_decisions_line(session["tool_decisions"]))
    deliverables = exec_result.get("deliverables") or []
    if deliverables:
        lines.append("")
        lines.append("### Deliverables")
        for d in deliverables:
            lines.append(f"- {d.get('title') or d.get('file_path')} — `{d.get('file_path')}`")
    files = exec_result.get("files_touched") or []
    if files:
        lines.append("")
        lines.append("### Files touched")
        lines.extend(f"- `{f}`" for f in files[:50])
    denials = session.get("permission_denials") or exec_result.get("permission_denials") or []
    if denials:
        lines.append("")
        lines.extend(refused_calls_lines(denials[:20]))
    recent = session.get("recent_tools") or []
    if recent:
        lines.append("")
        lines.append(f"### Tool calls (last {len(recent)})")
        for r in recent:
            if isinstance(r, dict):
                verdict = tool_call_verdict(r)
                lines.append(f"- {r.get('at', '')} {r.get('tool', '?')}" + (f" — `{r['subject']}`" if r.get("subject") else "")
                             + (f" · {verdict}" if verdict else ""))
    if session.get("transcript_path"):
        lines.append("")
        lines.append(f"- Transcript: `{session['transcript_path']}`")
    if session.get("session_id"):
        cd = f"cd {session['cwd']} && " if session.get("cwd") else ""
        lines.append(f"- Take over in your terminal: `{cd}claude --resume {session['session_id']}`")
    lines.append("")
    return lines
