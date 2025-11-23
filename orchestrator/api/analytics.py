"""
Analytics API endpoints
=======================

Provides analytics and monitoring data for workflows, agents, and system performance.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from database.database import get_db
from sqlalchemy.orm import Session
from services.workflow_analytics_service import WorkflowAnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])
logger = logging.getLogger(__name__)


@router.get("/workflows/{workflow_id}/trends")
async def get_workflow_trends(
    workflow_id: int,
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get performance trends for a specific workflow"""
    
    try:
        analytics_service = WorkflowAnalyticsService(db)
        trends = analytics_service.get_workflow_performance_trends(workflow_id, days)
        
        return {
            "success": True,
            "data": trends
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/performance")
async def get_agent_performance(
    agent_id: Optional[int] = Query(None, description="Specific agent ID (optional)"),
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get agent performance statistics"""
    
    try:
        analytics_service = WorkflowAnalyticsService(db)
        stats = analytics_service.get_agent_performance_stats(agent_id, days)
        
        return {
            "success": True,
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting agent performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills/demand")
async def get_skill_demand(
    days: int = Query(30, ge=1, le=180, description="Number of days to analyze"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Analyze skill demand and coverage"""
    
    try:
        analytics_service = WorkflowAnalyticsService(db)
        analysis = analytics_service.get_skill_demand_analysis(days)
        
        return {
            "success": True,
            "data": analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing skill demand: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/{execution_id}/report")
async def get_execution_report(
    execution_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get comprehensive report for a specific execution"""
    
    try:
        analytics_service = WorkflowAnalyticsService(db)
        report = analytics_service.generate_execution_report(execution_id)
        
        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])
        
        return {
            "success": True,
            "data": report
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating execution report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/summary")
async def get_dashboard_summary(
    days: int = Query(7, ge=1, le=90, description="Number of days for summary"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get summary statistics for dashboard"""
    
    try:
        from models import WorkflowExecution, Agent, Workflow, ExecutionStatus
        from sqlalchemy import and_, func
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Get execution statistics
        total_executions = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.started_at >= since_date
        ).scalar() or 0
        
        completed_executions = db.query(func.count(WorkflowExecution.id)).filter(
            and_(
                WorkflowExecution.started_at >= since_date,
                WorkflowExecution.status == ExecutionStatus.COMPLETED.value
            )
        ).scalar() or 0
        
        failed_executions = db.query(func.count(WorkflowExecution.id)).filter(
            and_(
                WorkflowExecution.started_at >= since_date,
                WorkflowExecution.status == ExecutionStatus.FAILED.value
            )
        ).scalar() or 0
        
        # Get active agents
        active_agents = db.query(func.count(Agent.id)).filter(
            Agent.status == "active"
        ).scalar() or 0
        
        # Get total workflows
        total_workflows = db.query(func.count(Workflow.id)).filter(
            Workflow.status == "active"
        ).scalar() or 0
        
        # Calculate success rate
        success_rate = (completed_executions / total_executions * 100) if total_executions > 0 else 0
        
        # Get recent executions for cost calculation
        recent_executions = db.query(WorkflowExecution).filter(
            and_(
                WorkflowExecution.started_at >= since_date,
                WorkflowExecution.status == ExecutionStatus.COMPLETED.value
            )
        ).all()
        
        total_cost = 0
        total_tokens = 0
        total_duration = 0
        
        for exec in recent_executions:
            analytics = exec.metadata.get("analytics", {}) if exec.metadata else {}
            total_cost += analytics.get("total_cost", 0)
            total_tokens += analytics.get("total_tokens_used", 0)
            total_duration += analytics.get("duration_seconds", 0)
        
        avg_cost = total_cost / completed_executions if completed_executions > 0 else 0
        avg_tokens = total_tokens / completed_executions if completed_executions > 0 else 0
        avg_duration = total_duration / completed_executions if completed_executions > 0 else 0
        
        return {
            "success": True,
            "data": {
                "period_days": days,
                "timestamp": datetime.now().isoformat(),
                "executions": {
                    "total": total_executions,
                    "completed": completed_executions,
                    "failed": failed_executions,
                    "success_rate": success_rate
                },
                "agents": {
                    "active": active_agents
                },
                "workflows": {
                    "active": total_workflows
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent-selection/analysis")
async def analyze_agent_selection(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Analyze agent selection patterns and effectiveness"""
    
    try:
        from models import WorkflowExecution, ExecutionStatus
        from sqlalchemy import and_
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Get completed executions
        executions = db.query(WorkflowExecution).filter(
            and_(
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
        raise HTTPException(status_code=500, detail=str(e))
