"""The report a board ticket files when its run ends (moved out of api/board_tasks).

F321 (night 9b, build 15): the task, heartbeat and Playbook reports each wrote
their own "## Execution Metrics" block, and a ticket whose calls were not found
printed "LLM calls: 0" beside the tokens its run reported. The block now comes
from ``services.report_metrics.metrics_lines``: the agent's ``llm_usage`` rows
over the ticket's run, a subscription session's cost as plan usage, and a
number that is not known said in words.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SUMMARY_MAX_CHARS = 497
TASK_REPORT_TYPE = "task"
HEARTBEAT_REPORT_TYPE = "heartbeat"  # v_workspace_outputs classes it as a heartbeat (outputs_heartbeat_reports)
HEARTBEAT_SOURCE = "heartbeat"
RESULT_KEYS = ("result", "response", "output", "content")


def _agent_name(db: Session, agent_id: Optional[int]) -> str:
    """The ticket's agent by name, or "Unknown Agent"."""
    from core.models import Agent

    if not agent_id:
        return "Unknown Agent"
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    return agent.name if agent else "Unknown Agent"


def _answer(task: Any, exec_result: Dict[str, Any]) -> str:
    """The agent's own response, else whatever the ticket's result holds."""
    return next((exec_result[k] for k in RESULT_KEYS if exec_result.get(k)), None) or task.result or ""


def _reported_tokens(exec_result: Dict[str, Any]) -> int:
    """The tokens the run itself reported (a session's usage), kept beside the
    rollup and said only when the rollup found no calls."""
    usage = exec_result.get("usage") or {}
    return int(usage.get("total_tokens") or exec_result.get("tokens_used") or 0)


def _cut(line: str) -> str:
    return (line[:SUMMARY_MAX_CHARS] + "...") if len(line) > SUMMARY_MAX_CHARS else line


def _summary(answer: str, error_message: Optional[str]) -> Optional[str]:
    """The answer's first non-empty line. F197: a failed task's result is blank,
    so its report is summarised by why it failed."""
    for line in str(answer).split("\n"):
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return _cut(stripped)
    if error_message:
        return _cut(str(error_message).strip().splitlines()[0])
    return None


def task_report_content(task: Any, agent_name: str, answer: str, exec_result: Dict[str, Any],
                        metrics: Dict[str, Any]) -> str:
    """The report's markdown, the shape heartbeat reports use."""
    from core.cli_runtime import RUNTIME_CLI
    from services.report_metrics import metrics_lines
    from services.session_report import session_report_lines

    lines: List[str] = [f"# {agent_name} — Task Report", f"**Task:** {task.title}", f"**Status:** {task.status}", ""]
    if task.error_message:
        lines += ["## Error", str(task.error_message), ""]
    if answer:
        lines += ["## Result", str(answer), ""]
    lines += session_report_lines(exec_result)  # PRD-234 S2 (empty for API runs)
    lines += metrics_lines(metrics, subscription=exec_result.get("runtime") == RUNTIME_CLI)
    return "\n".join(lines)


def _report_status(task: Any) -> str:
    if task.error_message:
        return "critical"
    return "ok" if task.status in ("done", "review") else "warning"


def report_type_for(task: Any) -> str:
    """A heartbeat ticket's report is a heartbeat report, so the feed hides it like the rest."""
    return HEARTBEAT_REPORT_TYPE if getattr(task, "source_type", None) == HEARTBEAT_SOURCE else TASK_REPORT_TYPE


async def auto_create_task_report(db: Session, workspace_id: str, task: Any, exec_result: Dict[str, Any]) -> None:
    """File an agent_reports row for a finished ticket, so it shows in Reports,
    Deliverables and the Activity Feed. Never raises: a failure is logged."""
    from services.report_metrics import settled_execution_metrics
    from services.report_service import ReportService

    try:
        agent_name, answer = _agent_name(db, task.assigned_agent_id), _answer(task, exec_result)
        metrics = await settled_execution_metrics(
            db, workspace_id, agent_id=task.assigned_agent_id, execution_id=getattr(task, "execution_id", None),
            started_at=getattr(task, "started_at", None), completed_at=getattr(task, "completed_at", None),
            extra={"task_id": task.id, "task_status": task.status, "trigger": "task",
                   "reported_tokens": _reported_tokens(exec_result)},
        )
        filed = await ReportService(db, workspace_id).create_report(
            agent_id=task.assigned_agent_id, agent_name=agent_name, title=f"Task: {task.title}",
            content=task_report_content(task, agent_name, answer, exec_result, metrics), report_type=report_type_for(task),
            status=_report_status(task), summary=_summary(answer, task.error_message), metrics=metrics,
            linked_task_ids=[task.id],
        )
        if not filed.get("success"):
            logger.warning("[BoardTasks] Auto-report creation failed for task=%s: %s", task.id, filed.get("error"))
    except Exception:
        logger.exception("[BoardTasks] the task report for task=%s was not filed", getattr(task, "id", "?"))


__all__ = ["auto_create_task_report", "report_type_for", "task_report_content"]
