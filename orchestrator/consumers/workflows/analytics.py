"""
Workflow Analytics Service
==========================

Records and analyzes workflow execution data for monitoring and improvement.
Tracks agent selection, task completion, costs, and performance metrics.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from dataclasses import dataclass, asdict
import json

from core.models import (
    WorkflowExecution, 
    Agent, 
    Workflow,
    ExecutionStatus
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowAnalytics:
    """Workflow execution analytics"""
    execution_id: int
    workflow_id: int
    workflow_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: float
    status: str
    
    # Agent metrics
    agents_used: List[Dict[str, Any]]
    agent_selection_score: float
    skill_coverage: float
    
    # Task metrics
    total_subtasks: int
    completed_subtasks: int
    failed_subtasks: int
    success_rate: float
    
    # Performance metrics
    total_tokens_used: int
    total_cost: float
    avg_task_duration: float
    
    # Quality metrics
    overall_quality_score: float
    completeness_score: float
    accuracy_score: float
    
    # Learning metrics
    memories_retrieved: int
    memories_stored: int
    patterns_extracted: int
    learning_improvements: int


class WorkflowAnalyticsService:
    """Service for recording and analyzing workflow analytics"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
    
    def record_workflow_analytics(
        self,
        execution: WorkflowExecution,
        detailed_results: Optional[Dict[str, Any]] = None
    ) -> WorkflowAnalytics:
        """Record analytics for a workflow execution"""
        
        try:
            # Extract data from execution
            input_data = execution.input_data or {}
            output_data = execution.output_data or {}
            
            # Get workflow details
            workflow = self.db.query(Workflow).filter(
                Workflow.id == execution.workflow_id
            ).first()
            
            # Calculate duration
            duration = 0
            if execution.completed_at and execution.started_at:
                duration = (execution.completed_at - execution.started_at).total_seconds()
            
            # Extract agent selection data
            agent_selection = input_data.get("agent_selection", {})
            agent_assignments = agent_selection.get("assignments", {})
            
            # Track all agent assignments per subtask
            all_assignments = []
            total_match_score = 0
            total_skill_coverage = 0
            
            for subtask_id, matches in agent_assignments.items():
                if matches and len(matches) > 0:
                    best_match = matches[0]
                    all_assignments.append({
                        "agent_id": best_match.get("agent_id"),
                        "agent_name": best_match.get("agent_name"),
                        "subtask": subtask_id,
                        "match_score": best_match.get("match_score", 0),
                        "skill_coverage": best_match.get("skill_coverage", 0),
                        "matched_skills": best_match.get("matched_skills", []),
                        "missing_skills": best_match.get("missing_skills", [])
                    })
                    total_match_score += best_match.get("match_score", 0)
                    total_skill_coverage += best_match.get("skill_coverage", 0)
            
            # Deduplicate agents_used by agent_id (show unique agents, not per-subtask)
            unique_agents = {}
            for assignment in all_assignments:
                agent_id = assignment["agent_id"]
                if agent_id not in unique_agents:
                    unique_agents[agent_id] = {
                        "agent_id": agent_id,
                        "agent_name": assignment["agent_name"],
                        "subtasks": [],
                        "total_match_score": 0,
                        "total_skill_coverage": 0,
                        "matched_skills": set(),
                        "missing_skills": set()
                    }
                unique_agents[agent_id]["subtasks"].append(assignment["subtask"])
                unique_agents[agent_id]["total_match_score"] += assignment["match_score"]
                unique_agents[agent_id]["total_skill_coverage"] += assignment["skill_coverage"]
                unique_agents[agent_id]["matched_skills"].update(assignment.get("matched_skills", []))
                unique_agents[agent_id]["missing_skills"].update(assignment.get("missing_skills", []))
            
            # Convert to list with aggregated metrics per agent
            agents_used = [
                {
                    "agent_id": data["agent_id"],
                    "agent_name": data["agent_name"],
                    "subtask_count": len(data["subtasks"]),
                    "subtasks": data["subtasks"],
                    "avg_match_score": data["total_match_score"] / len(data["subtasks"]),
                    "avg_skill_coverage": data["total_skill_coverage"] / len(data["subtasks"]),
                    "matched_skills": list(data["matched_skills"]),
                    "missing_skills": list(data["missing_skills"])
                }
                for data in unique_agents.values()
            ]
            
            logger.info(f"🔍 ANALYTICS DEBUG: all_assignments={len(all_assignments)}, unique_agents={len(unique_agents)}, agents_used={len(agents_used)}")
            for agent in agents_used:
                logger.info(f"  - Agent {agent['agent_id']} ({agent['agent_name']}): {agent['subtask_count']} subtasks")
            
            avg_match_score = total_match_score / len(all_assignments) if all_assignments else 0
            avg_skill_coverage = total_skill_coverage / len(all_assignments) if all_assignments else 0
            
            # Extract execution metrics
            agent_execution = input_data.get("agent_execution", {})
            exec_summary = agent_execution.get("summary", {})
            
            total_subtasks = exec_summary.get("total", 0)
            completed_subtasks = exec_summary.get("completed", 0)
            failed_subtasks = exec_summary.get("failed", 0)
            success_rate = exec_summary.get("success_rate", 0)
            total_tokens = exec_summary.get("total_tokens_used", 0)
            
            # Extract quality scores
            result_aggregation = input_data.get("result_aggregation", {})
            quality_scores = result_aggregation.get("quality_scores", {})
            
            # Extract learning metrics
            memory_summary = input_data.get("memory_integration_summary", {})
            learning_updates = input_data.get("learning_system", {}).get("updates", {})
            
            # Calculate costs (estimate based on tokens)
            # GPT-4 pricing: ~$0.03 per 1K input tokens, $0.06 per 1K output tokens
            # Using average of $0.045 per 1K tokens
            total_cost = (total_tokens / 1000) * 0.045
            
            # Get model usage if available
            if hasattr(execution, 'models_used') and execution.models_used:
                actual_cost = sum(model.get("cost", 0) for model in execution.models_used)
                if actual_cost > 0:
                    total_cost = actual_cost
            
            # Create analytics record
            analytics = WorkflowAnalytics(
                execution_id=execution.id,
                workflow_id=execution.workflow_id,
                workflow_name=workflow.name if workflow else "Unknown",
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                duration_seconds=duration,
                status=execution.status,
                
                # Agent metrics
                agents_used=agents_used,
                agent_selection_score=avg_match_score,
                skill_coverage=avg_skill_coverage,
                
                # Task metrics
                total_subtasks=total_subtasks,
                completed_subtasks=completed_subtasks,
                failed_subtasks=failed_subtasks,
                success_rate=success_rate,
                
                # Performance metrics
                total_tokens_used=total_tokens,
                total_cost=total_cost,
                avg_task_duration=duration / total_subtasks if total_subtasks > 0 else 0,
                
                # Quality metrics
                overall_quality_score=quality_scores.get("overall", 0),
                completeness_score=quality_scores.get("completeness", 0),
                accuracy_score=quality_scores.get("accuracy", 0),
                
                # Learning metrics
                memories_retrieved=memory_summary.get("memories_retrieved", 0),
                memories_stored=memory_summary.get("experiences_stored", 0),
                patterns_extracted=memory_summary.get("patterns_extracted", 0),
                learning_improvements=learning_updates.get("total_updates", 0)
            )
            
            # Store analytics in execution metadata
            if not execution.execution_metadata:
                execution.execution_metadata = {}  # Changed from metadata (reserved word in SQLAlchemy)
            
            # Convert analytics to dict and handle datetime serialization
            analytics_dict = asdict(analytics)
            # Convert datetime objects to ISO format strings
            if analytics_dict.get("started_at"):
                analytics_dict["started_at"] = analytics_dict["started_at"].isoformat() if isinstance(analytics_dict["started_at"], datetime) else analytics_dict["started_at"]
            if analytics_dict.get("completed_at"):
                analytics_dict["completed_at"] = analytics_dict["completed_at"].isoformat() if isinstance(analytics_dict["completed_at"], datetime) else analytics_dict["completed_at"]
            
            execution.execution_metadata["analytics"] = analytics_dict
            self.db.commit()
            
            self.logger.info(
                f"📊 Recorded analytics for execution {execution.id}: "
                f"{completed_subtasks}/{total_subtasks} tasks, "
                f"{len(agents_used)} agents, "
                f"${total_cost:.4f} cost"
            )
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to record analytics: {e}")
            raise
    
    def get_workflow_performance_trends(
        self,
        workflow_id: int,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get performance trends for a workflow over time"""
        
        since_date = datetime.now() - timedelta(days=days)
        
        executions = self.db.query(WorkflowExecution).filter(
            and_(
                WorkflowExecution.workflow_id == workflow_id,
                WorkflowExecution.started_at >= since_date,
                WorkflowExecution.status == ExecutionStatus.COMPLETED.value
            )
        ).order_by(WorkflowExecution.started_at).all()
        
        if not executions:
            return {
                "workflow_id": workflow_id,
                "period_days": days,
                "total_executions": 0,
                "trends": {}
            }
        
        # Extract trends
        duration_trend = []
        cost_trend = []
        quality_trend = []
        token_trend = []
        
        for exec in executions:
            analytics = exec.metadata.get("analytics", {}) if exec.metadata else {}
            
            if analytics:
                duration_trend.append({
                    "timestamp": exec.started_at.isoformat(),
                    "value": analytics.get("duration_seconds", 0)
                })
                
                cost_trend.append({
                    "timestamp": exec.started_at.isoformat(),
                    "value": analytics.get("total_cost", 0)
                })
                
                quality_trend.append({
                    "timestamp": exec.started_at.isoformat(),
                    "value": analytics.get("overall_quality_score", 0)
                })
                
                token_trend.append({
                    "timestamp": exec.started_at.isoformat(),
                    "value": analytics.get("total_tokens_used", 0)
                })
        
        # Calculate improvements
        improvements = {}
        
        if len(duration_trend) >= 2:
            first_duration = duration_trend[0]["value"]
            last_duration = duration_trend[-1]["value"]
            if first_duration > 0:
                improvements["duration"] = ((first_duration - last_duration) / first_duration) * 100
        
        if len(cost_trend) >= 2:
            first_cost = cost_trend[0]["value"]
            last_cost = cost_trend[-1]["value"]
            if first_cost > 0:
                improvements["cost"] = ((first_cost - last_cost) / first_cost) * 100
        
        if len(quality_trend) >= 2:
            first_quality = quality_trend[0]["value"]
            last_quality = quality_trend[-1]["value"]
            if first_quality > 0:
                improvements["quality"] = ((last_quality - first_quality) / first_quality) * 100
        
        return {
            "workflow_id": workflow_id,
            "period_days": days,
            "total_executions": len(executions),
            "trends": {
                "duration": duration_trend,
                "cost": cost_trend,
                "quality": quality_trend,
                "tokens": token_trend
            },
            "improvements": improvements,
            "latest_analytics": executions[-1].metadata.get("analytics", {}) if executions else {}
        }
    
    def get_agent_performance_stats(
        self,
        agent_id: Optional[int] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get agent performance statistics"""
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Query all completed executions in period
        query = self.db.query(WorkflowExecution).filter(
            and_(
                WorkflowExecution.started_at >= since_date,
                WorkflowExecution.status == ExecutionStatus.COMPLETED.value
            )
        )
        
        executions = query.all()
        
        # Aggregate agent stats
        agent_stats = {}
        
        for exec in executions:
            analytics = exec.metadata.get("analytics", {}) if exec.metadata else {}
            agents_used = analytics.get("agents_used", [])
            
            for agent_data in agents_used:
                aid = agent_data.get("agent_id")
                
                # Filter by agent if specified
                if agent_id and aid != agent_id:
                    continue
                
                if aid not in agent_stats:
                    agent_stats[aid] = {
                        "agent_id": aid,
                        "agent_name": agent_data.get("agent_name"),
                        "total_tasks": 0,
                        "total_match_score": 0,
                        "total_skill_coverage": 0,
                        "matched_skills": set(),
                        "missing_skills": set()
                    }
                
                agent_stats[aid]["total_tasks"] += 1
                agent_stats[aid]["total_match_score"] += agent_data.get("match_score", 0)
                agent_stats[aid]["total_skill_coverage"] += agent_data.get("skill_coverage", 0)
                
                # Collect skills
                for skill in agent_data.get("matched_skills", []):
                    agent_stats[aid]["matched_skills"].add(skill)
                
                for skill in agent_data.get("missing_skills", []):
                    agent_stats[aid]["missing_skills"].add(skill)
        
        # Calculate averages and format
        formatted_stats = []
        
        for aid, stats in agent_stats.items():
            tasks = stats["total_tasks"]
            if tasks > 0:
                formatted_stats.append({
                    "agent_id": aid,
                    "agent_name": stats["agent_name"],
                    "total_tasks": tasks,
                    "avg_match_score": stats["total_match_score"] / tasks,
                    "avg_skill_coverage": stats["total_skill_coverage"] / tasks,
                    "unique_matched_skills": list(stats["matched_skills"]),
                    "frequently_missing_skills": list(stats["missing_skills"])
                })
        
        # Sort by total tasks
        formatted_stats.sort(key=lambda x: x["total_tasks"], reverse=True)
        
        return {
            "period_days": days,
            "total_agents": len(formatted_stats),
            "agent_stats": formatted_stats
        }
    
    def get_skill_demand_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Analyze which skills are most in demand"""
        
        since_date = datetime.now() - timedelta(days=days)
        
        executions = self.db.query(WorkflowExecution).filter(
            and_(
                WorkflowExecution.started_at >= since_date,
                WorkflowExecution.status == ExecutionStatus.COMPLETED.value
            )
        ).all()
        
        # Track skill requests and coverage
        skill_demand = {}
        
        for exec in executions:
            decomposition = exec.input_data.get("decomposition", {}) if exec.input_data else {}
            subtasks = decomposition.get("subtasks", [])
            
            for subtask in subtasks:
                required_skills = subtask.get("required_skills", [])
                
                for skill in required_skills:
                    if skill not in skill_demand:
                        skill_demand[skill] = {
                            "skill": skill,
                            "requested_count": 0,
                            "matched_count": 0,
                            "coverage_rate": 0
                        }
                    
                    skill_demand[skill]["requested_count"] += 1
            
            # Check which skills were matched
            analytics = exec.metadata.get("analytics", {}) if exec.metadata else {}
            agents_used = analytics.get("agents_used", [])
            
            for agent_data in agents_used:
                for skill in agent_data.get("matched_skills", []):
                    if skill in skill_demand:
                        skill_demand[skill]["matched_count"] += 1
        
        # Calculate coverage rates
        for skill_data in skill_demand.values():
            if skill_data["requested_count"] > 0:
                skill_data["coverage_rate"] = (
                    skill_data["matched_count"] / skill_data["requested_count"]
                )
        
        # Sort by demand
        sorted_skills = sorted(
            skill_demand.values(),
            key=lambda x: x["requested_count"],
            reverse=True
        )
        
        return {
            "period_days": days,
            "total_unique_skills": len(skill_demand),
            "skill_demand": sorted_skills[:20],  # Top 20 skills
            "skills_needing_agents": [
                s for s in sorted_skills 
                if s["coverage_rate"] < 0.5
            ][:10]  # Skills with < 50% coverage
        }
    
    def generate_execution_report(
        self,
        execution_id: int
    ) -> Dict[str, Any]:
        """Generate comprehensive report for a single execution"""
        
        execution = self.db.query(WorkflowExecution).filter(
            WorkflowExecution.id == execution_id
        ).first()
        
        if not execution:
            return {"error": "Execution not found"}
        
        analytics = execution.execution_metadata.get("analytics", {}) if execution.execution_metadata else {}
        
        # Get workflow details
        workflow = self.db.query(Workflow).filter(
            Workflow.id == execution.workflow_id
        ).first()
        
        return {
            "execution_id": execution_id,
            "workflow": {
                "id": workflow.id if workflow else None,
                "name": workflow.name if workflow else "Unknown",
                "category": workflow.category if workflow else None
            },
            "timing": {
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_seconds": analytics.get("duration_seconds", 0)
            },
            "agents": {
                "total_used": len(analytics.get("agents_used", [])),
                "avg_match_score": analytics.get("agent_selection_score", 0),
                "skill_coverage": analytics.get("skill_coverage", 0),
                "details": analytics.get("agents_used", [])
            },
            "tasks": {
                "total": analytics.get("total_subtasks", 0),
                "completed": analytics.get("completed_subtasks", 0),
                "failed": analytics.get("failed_subtasks", 0),
                "success_rate": analytics.get("success_rate", 0)
            },
            "performance": {
                "tokens_used": analytics.get("total_tokens_used", 0),
                "cost": analytics.get("total_cost", 0),
                "avg_task_duration": analytics.get("avg_task_duration", 0)
            },
            "quality": {
                "overall": analytics.get("overall_quality_score", 0),
                "completeness": analytics.get("completeness_score", 0),
                "accuracy": analytics.get("accuracy_score", 0)
            },
            "learning": {
                "memories_retrieved": analytics.get("memories_retrieved", 0),
                "memories_stored": analytics.get("memories_stored", 0),
                "patterns_extracted": analytics.get("patterns_extracted", 0),
                "improvements": analytics.get("learning_improvements", 0)
            },
            "status": execution.status,
            "error": execution.error_message if execution.error_message else None
        }


# Utility function for quick analytics
def record_execution_analytics(
    db: Session,
    execution: WorkflowExecution
) -> WorkflowAnalytics:
    """Quick function to record analytics for an execution"""
    
    service = WorkflowAnalyticsService(db)
    return service.record_workflow_analytics(execution)
