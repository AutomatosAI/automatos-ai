"""
Benchmarking and Performance Analytics API
==========================================
Provides comprehensive metrics proving the system gets smarter,
faster, and more cost-effective over time.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, cast, String
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import statistics
import logging

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.models import WorkflowExecution, Workflow, Agent
from modules.orchestrator import orchestration_tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/benchmarking", tags=["benchmarking"])

@router.get("/performance-summary")
async def get_performance_summary(
    days: int = 30,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """
    Get comprehensive performance summary showing system improvements.
    Proves the system is getting smarter and more efficient.
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get all executions in time window - scoped to workspace
        executions = db.query(WorkflowExecution)\
            .filter(WorkflowExecution.workspace_id == ctx.workspace_id)\
            .filter(WorkflowExecution.started_at >= cutoff_date)\
            .order_by(WorkflowExecution.started_at)\
            .all()
        
        if not executions:
            return {
                "status": "no_data",
                "message": f"No executions found in the last {days} days"
            }
        
        # Split into first half and second half for comparison
        mid_point = len(executions) // 2
        first_half = executions[:mid_point]
        second_half = executions[mid_point:]
        
        # Calculate metrics for each half
        first_metrics = _calculate_execution_metrics(first_half)
        second_metrics = _calculate_execution_metrics(second_half)
        
        # Calculate improvements
        improvements = {}
        
        # Time improvement
        if first_metrics["avg_execution_time"] > 0:
            time_improvement = (first_metrics["avg_execution_time"] - second_metrics["avg_execution_time"]) / first_metrics["avg_execution_time"] * 100
            improvements["execution_time"] = {
                "improvement_percentage": round(time_improvement, 1),
                "first_half_avg": round(first_metrics["avg_execution_time"], 2),
                "second_half_avg": round(second_metrics["avg_execution_time"], 2),
                "status": "improving" if time_improvement > 0 else "degrading"
            }
        
        # Cost improvement
        if first_metrics["avg_cost"] > 0:
            cost_improvement = (first_metrics["avg_cost"] - second_metrics["avg_cost"]) / first_metrics["avg_cost"] * 100
            improvements["cost_efficiency"] = {
                "improvement_percentage": round(cost_improvement, 1),
                "first_half_avg": round(first_metrics["avg_cost"], 4),
                "second_half_avg": round(second_metrics["avg_cost"], 4),
                "total_savings": round((first_metrics["avg_cost"] - second_metrics["avg_cost"]) * len(second_half), 2),
                "status": "improving" if cost_improvement > 0 else "degrading"
            }
        
        # Success rate improvement
        success_improvement = second_metrics["success_rate"] - first_metrics["success_rate"]
        improvements["success_rate"] = {
            "improvement_percentage": round(success_improvement * 100, 1),
            "first_half_rate": round(first_metrics["success_rate"] * 100, 1),
            "second_half_rate": round(second_metrics["success_rate"] * 100, 1),
            "status": "improving" if success_improvement > 0 else "degrading"
        }
        
        # Token efficiency
        if first_metrics["avg_tokens"] > 0:
            token_improvement = (first_metrics["avg_tokens"] - second_metrics["avg_tokens"]) / first_metrics["avg_tokens"] * 100
            improvements["token_efficiency"] = {
                "improvement_percentage": round(token_improvement, 1),
                "first_half_avg": first_metrics["avg_tokens"],
                "second_half_avg": second_metrics["avg_tokens"],
                "status": "improving" if token_improvement > 0 else "degrading"
            }
        
        # Learning metrics
        learning_metrics = {
            "patterns_learned": _count_patterns_learned(second_half),
            "memory_utilization": _calculate_memory_utilization(second_half),
            "agent_specialization": _calculate_agent_specialization(db, second_half),
            "workflow_optimization": _calculate_workflow_optimization(second_half)
        }
        
        # Overall score (weighted average of improvements)
        overall_score = statistics.mean([
            improvements.get("execution_time", {}).get("improvement_percentage", 0) * 0.3,
            improvements.get("cost_efficiency", {}).get("improvement_percentage", 0) * 0.3,
            improvements.get("success_rate", {}).get("improvement_percentage", 0) * 0.25,
            improvements.get("token_efficiency", {}).get("improvement_percentage", 0) * 0.15
        ])
        
        # Generate trend data for charts (sample every N executions for performance)
        max_points = 20
        step = max(1, len(executions) // max_points)
        sampled_executions = executions[::step][:max_points]
        
        trend_data = []
        for i, execution in enumerate(sampled_executions):
            metrics = _extract_execution_metrics(execution)
            # Format timestamp for display
            time_label = execution.started_at.strftime("%m/%d %H:%M") if execution.started_at else f"Ex {i+1}"
            trend_data.append({
                "time": time_label,
                "execution_time": round(metrics.get("execution_time", 0), 2),
                "cost": round(metrics.get("cost", 0), 4),
                "tokens_used": metrics.get("tokens", 0),
                "success": 1 if metrics.get("success") else 0
            })
        
        return {
            "period": {
                "days": days,
                "total_executions": len(executions),
                "first_half_count": len(first_half),
                "second_half_count": len(second_half)
            },
            "improvements": improvements,
            "learning_metrics": learning_metrics,
            "overall_improvement": round(overall_score, 1),
            "status": _get_overall_status(overall_score),
            "key_insights": _generate_insights(improvements, learning_metrics),
            "trend_data": trend_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating performance summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/workflow/{workflow_id}/performance-trend")
async def get_workflow_performance_trend(
    workflow_id: int,
    limit: int = 20,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """
    Get performance trend for a specific workflow over time.
    Shows how the system learns and improves for repeated workflows.
    """
    try:
        executions = db.query(WorkflowExecution)\
            .filter(WorkflowExecution.workspace_id == ctx.workspace_id)\
            .filter(WorkflowExecution.workflow_id == workflow_id)\
            .filter(WorkflowExecution.status == "completed")\
            .order_by(WorkflowExecution.started_at)\
            .limit(limit)\
            .all()
        
        if len(executions) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 executions to show trends"
            }
        
        trend_data = []
        for i, execution in enumerate(executions, 1):
            metrics = _extract_execution_metrics(execution)
            trend_data.append({
                "execution_number": i,
                "execution_id": execution.id,
                "timestamp": execution.started_at.isoformat() if execution.started_at else None,
                "execution_time": metrics["execution_time"],
                "tokens_used": metrics["tokens"],
                "cost": metrics["cost"],
                "success": metrics["success"],
                "quality_score": metrics["quality"],
                "memory_hits": metrics.get("memory_hits", 0),
                "patterns_applied": metrics.get("patterns_applied", 0)
            })
        
        # Calculate trend lines
        times = [d["execution_time"] for d in trend_data if d["execution_time"]]
        costs = [d["cost"] for d in trend_data if d["cost"]]
        tokens = [d["tokens_used"] for d in trend_data if d["tokens_used"]]
        
        trends = {
            "execution_time_trend": _calculate_trend_coefficient(times),
            "cost_trend": _calculate_trend_coefficient(costs),
            "token_trend": _calculate_trend_coefficient(tokens),
            "quality_trend": _calculate_trend_coefficient([d["quality_score"] for d in trend_data if d["quality_score"]])
        }
        
        # Calculate learning acceleration
        first_third = trend_data[:len(trend_data)//3]
        last_third = trend_data[-len(trend_data)//3:]
        
        learning_acceleration = {
            "time_reduction": _calculate_percentage_change(
                statistics.mean([d["execution_time"] for d in first_third if d["execution_time"]]) if first_third else 0,
                statistics.mean([d["execution_time"] for d in last_third if d["execution_time"]]) if last_third else 0
            ),
            "cost_reduction": _calculate_percentage_change(
                statistics.mean([d["cost"] for d in first_third if d["cost"]]) if first_third else 0,
                statistics.mean([d["cost"] for d in last_third if d["cost"]]) if last_third else 0
            ),
            "quality_improvement": _calculate_percentage_change(
                statistics.mean([d["quality_score"] for d in first_third if d["quality_score"]]) if first_third else 0,
                statistics.mean([d["quality_score"] for d in last_third if d["quality_score"]]) if last_third else 0
            )
        }
        
        return {
            "workflow_id": workflow_id,
            "execution_count": len(executions),
            "trend_data": trend_data,
            "trends": trends,
            "learning_acceleration": learning_acceleration,
            "summary": {
                "is_improving": trends["execution_time_trend"] < 0 and trends["cost_trend"] < 0,
                "average_improvement_per_execution": abs(trends["execution_time_trend"]) if trends["execution_time_trend"] < 0 else 0,
                "projected_next_execution": {
                    "execution_time": max(0, trend_data[-1]["execution_time"] + trends["execution_time_trend"]),
                    "cost": max(0, trend_data[-1]["cost"] + trends["cost_trend"]),
                    "tokens": max(0, trend_data[-1]["tokens_used"] + trends["token_trend"])
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow performance trend: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/agent-performance")
async def get_agent_performance_metrics(
    days: int = 30,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """
    Get agent performance metrics showing specialization and improvement.
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get all agents - scoped to workspace
        agents = db.query(Agent).filter(Agent.workspace_id == ctx.workspace_id, Agent.status == 'active').all()
        
        agent_metrics = []
        for agent in agents:
            # Get workflow executions where this agent was used
            # Cast JSONB to text to use LIKE operator
            executions = db.query(WorkflowExecution)\
                .filter(WorkflowExecution.started_at >= cutoff_date)\
                .filter(cast(WorkflowExecution.input_data, String).like(f'%"agent_id":{agent.id}%'))\
                .all()
            
            if not executions:
                continue
            
            # Calculate metrics from workflow executions
            success_count = sum(1 for e in executions if e.status == "completed")
            total_time = sum((e.completed_at - e.started_at).total_seconds() if e.completed_at and e.started_at else 0 for e in executions)
            total_tokens = sum(e.input_data.get("analytics", {}).get("total_tokens", 0) if e.input_data else 0 for e in executions)
            total_cost = sum(e.input_data.get("analytics", {}).get("total_cost", 0) if e.input_data else 0 for e in executions)
            
            # Calculate improvement over time
            if len(executions) > 1:
                first_half = executions[:len(executions)//2]
                second_half = executions[len(executions)//2:]
                
                first_avg_time = statistics.mean([(e.completed_at - e.started_at).total_seconds() if e.completed_at and e.started_at else 0 for e in first_half])
                second_avg_time = statistics.mean([(e.completed_at - e.started_at).total_seconds() if e.completed_at and e.started_at else 0 for e in second_half])
                
                improvement = ((first_avg_time - second_avg_time) / first_avg_time * 100) if first_avg_time > 0 else 0
            else:
                improvement = 0
            
            agent_metrics.append({
                "agent_id": agent.id,
                "agent_name": agent.name,
                "agent_type": agent.type,
                "total_tasks": len(executions),
                "success_rate": (success_count / len(executions) * 100) if executions else 0,
                "avg_execution_time": (total_time / len(executions)) if executions else 0,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "efficiency_score": (success_count / total_cost * 100) if total_cost > 0 else 0,
                "improvement_percentage": round(improvement, 1),
                "specialization_score": _calculate_specialization_score(agent, executions)
            })
        
        # Sort by efficiency
        agent_metrics.sort(key=lambda x: x["efficiency_score"], reverse=True)
        
        return {
            "period_days": days,
            "total_agents": len(agent_metrics),
            "agent_metrics": agent_metrics,
            "top_performers": agent_metrics[:5] if len(agent_metrics) > 5 else agent_metrics,
            "most_improved": sorted(agent_metrics, key=lambda x: x["improvement_percentage"], reverse=True)[:5],
            "summary": {
                "avg_success_rate": statistics.mean([a["success_rate"] for a in agent_metrics]) if agent_metrics else 0,
                "avg_improvement": statistics.mean([a["improvement_percentage"] for a in agent_metrics]) if agent_metrics else 0,
                "total_cost_saved": sum(a["total_cost"] * (a["improvement_percentage"] / 100) for a in agent_metrics)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting agent performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/learning-effectiveness")
async def get_learning_effectiveness(
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """
    Measure the effectiveness of the learning system.
    Shows how patterns and memory improve performance.
    """
    try:
        # Get recent executions with learning data - scoped to workspace
        recent_executions = db.query(WorkflowExecution)\
            .filter(WorkflowExecution.workspace_id == ctx.workspace_id)\
            .filter(WorkflowExecution.input_data.isnot(None))\
            .order_by(desc(WorkflowExecution.started_at))\
            .limit(100)\
            .all()
        
        learning_metrics = {
            "total_patterns_learned": 0,
            "memory_hit_rate": [],
            "pattern_application_rate": [],
            "performance_improvements": [],
            "cost_reductions": []
        }
        
        for execution in recent_executions:
            if not execution.input_data:
                continue
            
            # Extract learning data
            learning_data = execution.input_data.get("learning_system", {})
            memory_data = execution.input_data.get("memory_retrieval", {})
            
            # Count patterns
            if learning_data.get("updates", {}).get("workflow_pattern_updates", {}).get("pattern_recorded"):
                learning_metrics["total_patterns_learned"] += 1
            
            # Memory hit rate
            if memory_data:
                total_memories = sum(
                    len(memories.get(mem_type, []))
                    for memories in memory_data.get("agent_memories", {}).values()
                    for mem_type in ["working_memory", "short_term", "long_term"]
                )
                if total_memories > 0:
                    learning_metrics["memory_hit_rate"].append(min(1.0, total_memories / 10))
            
            # Pattern application
            if "patterns_applied" in execution.input_data:
                learning_metrics["pattern_application_rate"].append(execution.input_data["patterns_applied"])
        
        # Calculate aggregates
        summary = {
            "total_patterns_learned": learning_metrics["total_patterns_learned"],
            "avg_memory_hit_rate": statistics.mean(learning_metrics["memory_hit_rate"]) * 100 if learning_metrics["memory_hit_rate"] else 0,
            "memory_utilization_trend": _calculate_trend_coefficient(learning_metrics["memory_hit_rate"]),
            "pattern_growth_rate": learning_metrics["total_patterns_learned"] / len(recent_executions) if recent_executions else 0,
            "learning_velocity": _calculate_learning_velocity(recent_executions)
        }
        
        # Get performance trends from tracker
        tracker_trends = await orchestration_tracker.get_performance_trends()
        
        # Generate trend data for learning charts
        # Reverse so oldest is first for trend visualization
        ordered_executions = list(reversed(recent_executions[-50:]))  # Last 50 executions
        trend_data = []
        for i, execution in enumerate(ordered_executions, 1):
            learning_data = execution.input_data.get("learning_system", {}) if execution.input_data else {}
            memory_data = execution.input_data.get("memory_retrieval", {}) if execution.input_data else {}
            
            # Count patterns applied
            patterns_applied = execution.input_data.get("patterns_applied", 0) if execution.input_data else 0
            
            # Calculate memory hits
            memory_hits = 0
            if memory_data:
                for memories in memory_data.get("agent_memories", {}).values():
                    memory_hits += len(memories.get("working_memory", []))
                    memory_hits += len(memories.get("short_term", []))
                    memory_hits += len(memories.get("long_term", []))
            
            # Quality score from output
            quality_score = 85  # Default
            if execution.output_data and "quality_scores" in execution.output_data:
                quality_score = execution.output_data["quality_scores"].get("overall_score", 85)
            
            trend_data.append({
                "execution_number": i,
                "patterns_applied": patterns_applied,
                "memory_hits": min(100, memory_hits * 10),  # Normalize to percentage
                "quality_score": quality_score
            })
        
        return {
            "learning_metrics": learning_metrics,
            "summary": summary,
            "tracker_trends": tracker_trends,
            "trend_data": trend_data,
            "effectiveness_score": _calculate_effectiveness_score(summary, tracker_trends),
            "insights": _generate_learning_insights(summary, tracker_trends)
        }
        
    except Exception as e:
        logger.error(f"Error getting learning effectiveness: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Helper functions

def _calculate_execution_metrics(executions: List[WorkflowExecution]) -> Dict[str, Any]:
    """Calculate aggregate metrics for a set of executions"""
    if not executions:
        return {
            "avg_execution_time": 0,
            "avg_cost": 0,
            "avg_tokens": 0,
            "success_rate": 0
        }
    
    total_time = 0
    total_cost = 0
    total_tokens = 0
    success_count = 0
    
    for execution in executions:
        metrics = _extract_execution_metrics(execution)
        total_time += metrics["execution_time"]
        total_cost += metrics["cost"]
        total_tokens += metrics["tokens"]
        if metrics["success"]:
            success_count += 1
    
    return {
        "avg_execution_time": total_time / len(executions),
        "avg_cost": total_cost / len(executions),
        "avg_tokens": total_tokens / len(executions),
        "success_rate": success_count / len(executions)
    }

def _extract_execution_metrics(execution: WorkflowExecution) -> Dict[str, Any]:
    """Extract metrics from a single execution"""
    metrics = {
        "execution_time": 0,
        "tokens": 0,
        "cost": 0,
        "success": execution.status == "completed",
        "quality": 0
    }
    
    if execution.started_at and execution.completed_at:
        metrics["execution_time"] = (execution.completed_at - execution.started_at).total_seconds()
    
    # Check output_data first (new location), then input_data (legacy)
    if execution.output_data:
        metrics["tokens"] = execution.output_data.get("total_tokens_used", 0)
        metrics["cost"] = execution.output_data.get("total_cost", 0)
        metrics["quality"] = execution.output_data.get("quality_scores", {}).get("overall_score", 0)
    elif execution.input_data:
        # Fallback to input_data for older executions
        analytics = execution.input_data.get("analytics", {})
        metrics["tokens"] = analytics.get("total_tokens", 0)
        metrics["cost"] = analytics.get("total_cost", 0)
        metrics["quality"] = analytics.get("overall_quality", 0)
    
    # Memory and pattern data from input_data
    if execution.input_data:
        metrics["memory_hits"] = execution.input_data.get("memory_operation_count", 0)
        metrics["patterns_applied"] = execution.input_data.get("patterns_learned", 0)
    
    return metrics

def _calculate_trend_coefficient(values: List[float]) -> float:
    """Calculate linear trend coefficient"""
    if len(values) < 2:
        return 0
    
    n = len(values)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    
    numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

def _calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100

def _get_overall_status(score: float) -> str:
    """Determine overall status based on improvement score"""
    if score > 20:
        return "excellent_improvement"
    elif score > 10:
        return "significant_improvement"
    elif score > 5:
        return "moderate_improvement"
    elif score > 0:
        return "slight_improvement"
    elif score > -5:
        return "stable"
    else:
        return "needs_attention"

def _generate_insights(improvements: Dict, learning_metrics: Dict) -> List[str]:
    """Generate actionable insights from metrics"""
    insights = []
    
    # Time improvement insight
    if improvements.get("execution_time", {}).get("improvement_percentage", 0) > 10:
        insights.append(f"⚡ Execution time improved by {improvements['execution_time']['improvement_percentage']}% - system is learning optimal execution paths")
    
    # Cost saving insight
    if improvements.get("cost_efficiency", {}).get("total_savings", 0) > 0:
        insights.append(f"💰 Saved ${improvements['cost_efficiency']['total_savings']:.2f} through improved efficiency")
    
    # Success rate insight
    if improvements.get("success_rate", {}).get("improvement_percentage", 0) > 5:
        insights.append(f"✅ Success rate increased by {improvements['success_rate']['improvement_percentage']}% - better task understanding")
    
    # Learning insight
    if learning_metrics.get("patterns_learned", 0) > 0:
        insights.append(f"🧠 {learning_metrics['patterns_learned']} patterns learned and being applied")
    
    # Memory insight
    if learning_metrics.get("memory_utilization", 0) > 0.7:
        insights.append(f"💾 High memory utilization ({learning_metrics['memory_utilization']*100:.0f}%) reducing redundant processing")
    
    return insights

def _count_patterns_learned(executions: List[WorkflowExecution]) -> int:
    """Count total patterns learned from executions"""
    count = 0
    for execution in executions:
        if execution.input_data and "learning_system" in execution.input_data:
            if execution.input_data["learning_system"].get("updates", {}).get("workflow_pattern_updates", {}).get("pattern_recorded"):
                count += 1
    return count

def _calculate_memory_utilization(executions: List[WorkflowExecution]) -> float:
    """Calculate average memory utilization"""
    utilizations = []
    for execution in executions:
        if execution.input_data and "memory_operation_count" in execution.input_data:
            count = execution.input_data["memory_operation_count"]
            utilizations.append(min(1.0, count / 20))  # Normalize to 0-1
    
    return statistics.mean(utilizations) if utilizations else 0

def _calculate_agent_specialization(db: Session, executions: List[WorkflowExecution]) -> float:
    """Calculate how specialized agents are becoming"""
    # This would analyze agent task assignments to see if agents are specializing
    return 0.75  # Placeholder

def _calculate_workflow_optimization(executions: List[WorkflowExecution]) -> float:
    """Calculate workflow optimization level"""
    if len(executions) < 2:
        return 0
    
    # Compare first and last execution
    first = _extract_execution_metrics(executions[0])
    last = _extract_execution_metrics(executions[-1])
    
    improvements = []
    if first["execution_time"] > 0:
        improvements.append((first["execution_time"] - last["execution_time"]) / first["execution_time"])
    if first["cost"] > 0:
        improvements.append((first["cost"] - last["cost"]) / first["cost"])
    
    return statistics.mean(improvements) if improvements else 0

def _calculate_specialization_score(agent: Agent, executions: List[WorkflowExecution]) -> float:
    """Calculate how specialized an agent is becoming"""
    if not executions:
        return 0
    
    # Analyze task types from workflow definitions
    task_types = {}
    for execution in executions:
        # Get task type from workflow definition
        task_type = "general"
        if execution.input_data:
            workflow_def = execution.input_data.get("workflow_definition", {})
            task_type = workflow_def.get("type", "general")
        
        task_types[task_type] = task_types.get(task_type, 0) + 1
    
    # Calculate specialization (1 - entropy)
    total = sum(task_types.values())
    if total == 0:
        return 0
    
    entropy = 0
    for count in task_types.values():
        if count > 0:
            p = count / total
            entropy -= p * (p if p > 0 else 0)
    
    return 1 - entropy

def _calculate_learning_velocity(executions: List[WorkflowExecution]) -> float:
    """Calculate how fast the system is learning"""
    if len(executions) < 2:
        return 0
    
    improvements = []
    for i in range(1, len(executions)):
        prev = _extract_execution_metrics(executions[i-1])
        curr = _extract_execution_metrics(executions[i])
        
        if prev["execution_time"] > 0:
            improvement = (prev["execution_time"] - curr["execution_time"]) / prev["execution_time"]
            improvements.append(improvement)
    
    return statistics.mean(improvements) if improvements else 0

def _calculate_effectiveness_score(summary: Dict, trends: Dict) -> float:
    """Calculate overall learning effectiveness score"""
    scores = []
    
    if summary.get("avg_memory_hit_rate", 0) > 0:
        scores.append(summary["avg_memory_hit_rate"])
    
    if summary.get("pattern_growth_rate", 0) > 0:
        scores.append(min(100, summary["pattern_growth_rate"] * 100))
    
    if trends.get("overall_trend") == "improving":
        scores.append(80)
    elif trends.get("overall_trend") == "stable":
        scores.append(50)
    else:
        scores.append(20)
    
    return statistics.mean(scores) if scores else 0

def _generate_learning_insights(summary: Dict, trends: Dict) -> List[str]:
    """Generate insights about learning effectiveness"""
    insights = []
    
    if summary.get("avg_memory_hit_rate", 0) > 70:
        insights.append("Memory system is highly effective with >70% hit rate")
    
    if summary.get("pattern_growth_rate", 0) > 0.1:
        insights.append("Actively learning new patterns from executions")
    
    if trends.get("learning_adoption", 0) > 0.5:
        insights.append("Majority of executions utilize learned patterns")
    
    if summary.get("learning_velocity", 0) > 0.05:
        insights.append("Rapid learning velocity - improving >5% per execution")
    
    return insights
