"""
Analytics API endpoints
=======================

Provides analytics and monitoring data for missions, agents, and system performance.

PRD-125 Phase 3: Removed 4 dead endpoints that depended on WorkflowAnalyticsService
(trends, agent-performance, skill-demand, execution-report). None had frontend callers.
Kept: dashboard/summary (UNION queries) and agent-selection/analysis.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.auth.super_admin import require_super_admin
from sqlalchemy.orm import Session
from core.models import Workflow, WorkflowExecution as WorkflowExecutionModel, Agent as AgentModel
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState

# PRD-143 S6: observability tier — router-wide super-admin lock (fail-closed).
router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    dependencies=[Depends(require_super_admin)],
)
logger = logging.getLogger(__name__)


@router.get("/dashboard/summary")
async def get_dashboard_summary(
    days: int = Query(7, ge=1, le=90, description="Number of days for summary"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """Get summary statistics for dashboard"""

    try:
        from core.models import WorkflowExecution, Agent, Workflow, ExecutionStatus
        from sqlalchemy import and_, func

        ws = ctx.workspace_id
        since_date = datetime.now() - timedelta(days=days)

        # --- Legacy workflow executions ---
        wf_total = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == ws,
            WorkflowExecution.started_at >= since_date
        ).scalar() or 0
        wf_completed = db.query(func.count(WorkflowExecution.id)).filter(and_(
            WorkflowExecution.workspace_id == ws,
            WorkflowExecution.started_at >= since_date,
            WorkflowExecution.status == ExecutionStatus.COMPLETED.value
        )).scalar() or 0
        wf_failed = db.query(func.count(WorkflowExecution.id)).filter(and_(
            WorkflowExecution.workspace_id == ws,
            WorkflowExecution.started_at >= since_date,
            WorkflowExecution.status == ExecutionStatus.FAILED.value
        )).scalar() or 0

        # --- PRD-125 Phase 2: Mission runs ---
        m_total = db.query(func.count(OrchestrationRun.id)).filter(
            OrchestrationRun.workspace_id == ws,
            OrchestrationRun.created_at >= since_date
        ).scalar() or 0
        m_completed = db.query(func.count(OrchestrationRun.id)).filter(and_(
            OrchestrationRun.workspace_id == ws,
            OrchestrationRun.created_at >= since_date,
            OrchestrationRun.state == RunState.COMPLETED.value
        )).scalar() or 0
        m_failed = db.query(func.count(OrchestrationRun.id)).filter(and_(
            OrchestrationRun.workspace_id == ws,
            OrchestrationRun.created_at >= since_date,
            OrchestrationRun.state == RunState.FAILED.value
        )).scalar() or 0

        # Combined
        total_executions = wf_total + m_total
        completed_executions = wf_completed + m_completed
        failed_executions = wf_failed + m_failed

        # Active agents
        active_agents = db.query(func.count(Agent.id)).filter(
            Agent.workspace_id == ws, Agent.status == "active"
        ).scalar() or 0

        # Active workflows
        total_workflows = db.query(func.count(Workflow.id)).filter(
            Workflow.workspace_id == ws, Workflow.status == "active"
        ).scalar() or 0

        # Success rate
        success_rate = (completed_executions / total_executions * 100) if total_executions > 0 else 0

        # Cost from legacy workflow metadata
        recent_wf_executions = db.query(WorkflowExecution).filter(and_(
            WorkflowExecution.workspace_id == ws,
            WorkflowExecution.started_at >= since_date,
            WorkflowExecution.status == ExecutionStatus.COMPLETED.value
        )).all()

        total_cost = 0
        total_tokens = 0
        total_duration = 0

        for exec_row in recent_wf_executions:
            analytics_data = exec_row.metadata.get("analytics", {}) if exec_row.metadata else {}
            total_cost += analytics_data.get("total_cost", 0)
            total_tokens += analytics_data.get("total_tokens_used", 0)
            total_duration += analytics_data.get("duration_seconds", 0)

        # Add mission token usage
        m_tokens = db.query(func.sum(OrchestrationRun.tokens_used)).filter(and_(
            OrchestrationRun.workspace_id == ws,
            OrchestrationRun.created_at >= since_date,
            OrchestrationRun.state == RunState.COMPLETED.value,
        )).scalar() or 0
        total_tokens += m_tokens

        all_completed = completed_executions or 1
        avg_cost = total_cost / all_completed
        avg_tokens = total_tokens / all_completed
        avg_duration = total_duration / all_completed

        return {
            "success": True,
            "data": {
                "period_days": days,
                "timestamp": datetime.now().isoformat(),
                "executions": {
                    "total": total_executions,
                    "completed": completed_executions,
                    "failed": failed_executions,
                    "success_rate": success_rate,
                    "sources": {"workflows": wf_total, "missions": m_total},
                },
                "agents": {
                    "active": active_agents
                },
                "workflows": {
                    "active": total_workflows
                },
                "missions": {
                    "total": m_total,
                    "completed": m_completed,
                    "failed": m_failed,
                },
                "performance": {
                    "avg_duration_seconds": avg_duration,
                    "avg_tokens": avg_tokens,
                    "avg_cost": avg_cost,
                    "total_cost": total_cost
                }
            }
        }

    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agent-selection/analysis")
async def analyze_agent_selection(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """Analyze agent selection patterns and effectiveness"""

    try:
        from core.models import WorkflowExecution, ExecutionStatus
        from sqlalchemy import and_

        since_date = datetime.now() - timedelta(days=days)

        # Get completed executions
        executions = db.query(WorkflowExecution).filter(
            and_(
                WorkflowExecution.workspace_id == ctx.workspace_id,
                WorkflowExecution.started_at >= since_date,
                WorkflowExecution.status == ExecutionStatus.COMPLETED.value
            )
        ).all()

        # Analyze agent selection patterns
        selection_patterns = {}
        skill_match_scores = []
        coverage_rates = []

        for exec in executions:
            input_data = exec.input_data or {}
            agent_selection = input_data.get("agent_selection", {})

            if agent_selection.get("is_real"):
                summary = agent_selection.get("summary", {})
                assignments = agent_selection.get("assignments", {})

                # Track coverage and scores
                coverage_rates.append(summary.get("coverage", 0))
                skill_match_scores.append(summary.get("avg_match_score", 0))

                # Analyze selection patterns
                for subtask_id, matches in assignments.items():
                    if matches:
                        best_match = matches[0]
                        agent_type = best_match.get("agent_type", "unknown")

                        if agent_type not in selection_patterns:
                            selection_patterns[agent_type] = {
                                "agent_type": agent_type,
                                "selection_count": 0,
                                "total_match_score": 0,
                                "total_skill_coverage": 0
                            }

                        selection_patterns[agent_type]["selection_count"] += 1
                        selection_patterns[agent_type]["total_match_score"] += best_match.get("match_score", 0)
                        selection_patterns[agent_type]["total_skill_coverage"] += best_match.get("skill_coverage", 0)

        # Calculate averages
        for pattern in selection_patterns.values():
            count = pattern["selection_count"]
            if count > 0:
                pattern["avg_match_score"] = pattern["total_match_score"] / count
                pattern["avg_skill_coverage"] = pattern["total_skill_coverage"] / count
            del pattern["total_match_score"]
            del pattern["total_skill_coverage"]

        # Sort by selection count
        sorted_patterns = sorted(
            selection_patterns.values(),
            key=lambda x: x["selection_count"],
            reverse=True
        )

        return {
            "success": True,
            "data": {
                "period_days": days,
                "total_executions": len(executions),
                "avg_coverage_rate": sum(coverage_rates) / len(coverage_rates) if coverage_rates else 0,
                "avg_skill_match_score": sum(skill_match_scores) / len(skill_match_scores) if skill_match_scores else 0,
                "agent_type_patterns": sorted_patterns,
                "recommendations": []
            }
        }

    except Exception as e:
        logger.error(f"Error analyzing agent selection: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
