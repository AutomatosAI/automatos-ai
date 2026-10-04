"""The report an agent's heartbeat files when it runs (moved out of
``HeartbeatService._auto_create_report``).

F321 (night 9b, build 15): the heartbeat, task and Playbook reports each wrote
their own "## Execution Metrics" block. A heartbeat's now comes from
``services.report_metrics.metrics_lines`` like the others: the agent's
``llm_usage`` rows over the tick, and a number that is not known said in words.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SUMMARY_MAX_CHARS = 497
ERROR_CHECK = "error"
SUCCESS = "success"


def _finding_lines(findings: List[dict], actions: List[Any]) -> List[str]:
    """The tick's findings and the actions it took."""
    lines: List[str] = []
    if findings:
        lines += ["## Findings"] + [f"- **{f.get('check', 'unknown')}:** {f.get('detail', '')}" for f in findings] + [""]
    if actions:
        lines.append("## Actions Taken")
        lines += [f"- {a.get('action', '')} → {a.get('result', '')}" if isinstance(a, dict) else f"- {a}"
                  for a in actions]
        lines.append("")
    return lines


def heartbeat_report_content(agent_name: str, result: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """The report's markdown: status, findings, actions, then the shared metrics block."""
    from services.report_metrics import metrics_lines

    findings, actions = result.get("findings", []), result.get("actions_taken", [])
    lines = [f"# {agent_name} — Heartbeat Report", f"**Status:** {result.get('status', SUCCESS)}", ""]
    lines += _finding_lines(findings, actions)
    lines += metrics_lines(metrics)
    lines += [f"- Findings: {len(findings)}", f"- Actions: {len(actions)}"]
    return "\n".join(lines)


def _report_status(result: Dict[str, Any]) -> str:
    if any(f.get("check") == ERROR_CHECK for f in result.get("findings", [])):
        return "critical"
    return "ok" if result.get("status", SUCCESS) == SUCCESS else "warning"


def _summary(findings: List[dict]) -> Optional[str]:
    """The first finding's detail that is not an error."""
    for finding in findings:
        detail = finding.get("detail", "")
        if detail and finding.get("check") != ERROR_CHECK:
            return detail[:SUMMARY_MAX_CHARS] + "..." if len(detail) > SUMMARY_MAX_CHARS else detail
    return None


async def _file(db: Any, agent_id: int, workspace_id: str, result: Dict[str, Any]) -> None:
    from core.models import Agent
    from services.report_metrics import settled_execution_metrics
    from services.report_service import ReportService

    agent = db.query(Agent).get(agent_id)
    agent_name = agent.name if agent else f"agent-{agent_id}"
    findings = result.get("findings", [])
    metrics = await settled_execution_metrics(
        db, workspace_id, agent_id=agent_id,
        started_at=result.get("_run_started_at"), completed_at=result.get("_run_completed_at"),
        extra={"findings_count": len(findings), "actions_count": len(result.get("actions_taken", [])),
               "trigger": "heartbeat", "reported_tokens": int(result.get("tokens_used") or 0)},
    )
    filed = await ReportService(db, workspace_id).create_report(
        agent_id=agent_id, agent_name=agent_name, title=f"{agent_name} Heartbeat",
        content=heartbeat_report_content(agent_name, result, metrics), report_type="standup",
        status=_report_status(result), summary=_summary(findings), metrics=metrics,
        heartbeat_result_id=result.get("_heartbeat_result_id"),
    )
    if filed.get("success"):
        logger.info("[Heartbeat] Auto-created report %s for agent=%s", filed.get("report_id"), agent_id)
    else:
        logger.warning("[Heartbeat] Auto-report creation failed for agent=%s: %s", agent_id, filed.get("error"))


async def auto_create_heartbeat_report(agent_id: int, workspace_id: str, result: Dict[str, Any]) -> None:
    """File a report for every heartbeat tick, even when the agent filed none.
    Runs on its own session. Never raises: a failure is logged."""
    from core.database.database import SessionLocal

    try:
        db = SessionLocal()
        try:
            await _file(db, agent_id, workspace_id, result)
        finally:
            db.close()
    except Exception:
        logger.exception("[Heartbeat] the heartbeat report for agent=%s was not filed", agent_id)


__all__ = ["auto_create_heartbeat_report", "heartbeat_report_content"]
