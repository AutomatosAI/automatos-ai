"""PRD-234 S2 — the "Claude Code session" section of a task report.

Kept apart from ``api.board_tasks`` so it stays a pure function: what ran, where,
what it produced, what was refused and how to take it over. Empty for an API run,
so the report shape stays uniform across runtimes.
"""
from __future__ import annotations

from typing import Any, Dict, List

from core.cli_runtime import RUNTIME_CLI


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
        lines.append("### Refused tool calls")
        for d in denials[:20]:
            if isinstance(d, dict):
                lines.append(f"- {d.get('tool') or '?'}: {d.get('reason') or d.get('subject') or 'refused'}")
    recent = session.get("recent_tools") or []
    if recent:
        lines.append("")
        lines.append(f"### Tool calls (last {len(recent)})")
        for r in recent:
            if isinstance(r, dict):
                lines.append(f"- {r.get('at', '')} {r.get('tool', '?')}" + (f" — `{r['subject']}`" if r.get("subject") else ""))
    if session.get("transcript_path"):
        lines.append("")
        lines.append(f"- Transcript: `{session['transcript_path']}`")
    if session.get("session_id"):
        cd = f"cd {session['cwd']} && " if session.get("cwd") else ""
        lines.append(f"- Take over in your terminal: `{cd}claude --resume {session['session_id']}`")
    lines.append("")
    return lines
