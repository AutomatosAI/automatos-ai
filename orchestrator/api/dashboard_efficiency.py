"""The efficiency score behind GET /api/analytics/dashboard/efficiency-score.

Split out of ``api/analytics_real.py`` (#1100), where the handler computed it inline.
The score is a weighted mix of CPU, memory, agent utilisation and the last day's
completion rate (workflows and missions together), graded A to D.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Tuple

from sqlalchemy import and_
from sqlalchemy.orm import Session

from core.models import Agent, WorkflowExecution
from core.models.orchestration import OrchestrationRun
from core.models.orchestration_enums import RunState

CPU_WEIGHT, MEMORY_WEIGHT, AGENT_WEIGHT, WORKFLOW_WEIGHT = 0.3, 0.25, 0.25, 0.2
# Normalisers that favour moderate use.
CPU_FACTOR, MEMORY_FACTOR = 1.2, 1.1
GRADES = ((90, "A", "green"), (80, "B", "blue"), (70, "C", "yellow"))
LOWEST_GRADE = ("D", "red")


def agent_utilisation(db: Session, workspace_id: Any) -> float:
    """Active agents as a percentage of the workspace's agents (0 with none)."""
    total = db.query(Agent).filter(Agent.workspace_id == workspace_id).count()
    active = db.query(Agent).filter(Agent.workspace_id == workspace_id, Agent.status == "active").count()
    return (active / total * 100) if total > 0 else 0


def completion_rate(db: Session, workspace_id: Any, since: datetime) -> float:
    """Completed workflows and missions started since ``since``, as a percentage (0 with none)."""
    wf_recent = db.query(WorkflowExecution).filter(
        WorkflowExecution.workspace_id == workspace_id, WorkflowExecution.started_at >= since
    ).count()
    wf_completed = db.query(WorkflowExecution).filter(and_(
        WorkflowExecution.workspace_id == workspace_id, WorkflowExecution.status == "completed",
        WorkflowExecution.started_at >= since,
    )).count()
    m_recent = db.query(OrchestrationRun).filter(
        OrchestrationRun.workspace_id == workspace_id, OrchestrationRun.created_at >= since
    ).count()
    m_completed = db.query(OrchestrationRun).filter(and_(
        OrchestrationRun.workspace_id == workspace_id,
        OrchestrationRun.state == RunState.COMPLETED.value,
        OrchestrationRun.created_at >= since,
    )).count()
    recent = wf_recent + m_recent
    return ((wf_completed + m_completed) / recent * 100) if recent > 0 else 0


def grade_for(score: float) -> Tuple[str, str]:
    """(grade, colour) for a 0-100 score."""
    return next(((grade, colour) for floor, grade, colour in GRADES if score >= floor), LOWEST_GRADE)


def efficiency_report(cpu_usage: float, memory_percent: float, agent_efficiency: float,
                      workflow_efficiency: float) -> Dict[str, Any]:
    """The score, its grade and its breakdown."""
    cpu_efficiency = min(100, cpu_usage * CPU_FACTOR)
    memory_efficiency = min(100, memory_percent * MEMORY_FACTOR)
    score = round(cpu_efficiency * CPU_WEIGHT + memory_efficiency * MEMORY_WEIGHT
                  + agent_efficiency * AGENT_WEIGHT + workflow_efficiency * WORKFLOW_WEIGHT, 0)
    grade, colour = grade_for(score)
    return {
        "score": int(score),
        "grade": grade,
        "color": colour,
        "breakdown": {
            "cpu_efficiency": round(cpu_efficiency, 1),
            "memory_efficiency": round(memory_efficiency, 1),
            "agent_efficiency": round(agent_efficiency, 1),
            "workflow_efficiency": round(workflow_efficiency, 1),
        },
    }


__all__ = ["agent_utilisation", "completion_rate", "efficiency_report", "grade_for"]
