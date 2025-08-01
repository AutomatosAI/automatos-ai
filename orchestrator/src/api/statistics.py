
"""
Statistics and Metrics API
==========================

API endpoints for system statistics, agent metrics, and dashboard data.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import Dict, Any
from datetime import datetime, timedelta
from src.database.database import get_db
from src.database.models import (
    Agent, Skill, Pattern, Workflow, WorkflowExecution, 
    AgentStatistics, SystemMetrics
)
import logging
import psutil
import time

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["statistics"])

# Store system start time for uptime calculation
SYSTEM_START_TIME = time.time()

@router.get("/agents/statistics", response_model=AgentStatistics)
async def get_agent_statistics(db: Session = Depends(get_db)):
    """Get comprehensive agent statistics for dashboard"""
    try:
        # Basic agent counts
        total_agents = db.query(Agent).count()
        active_agents = db.query(Agent).filter(Agent.status == 'active').count()
        inactive_agents = db.query(Agent).filter(Agent.status == 'inactive').count()
        
        # Agents by type
        agent_types = db.query(
            Agent.agent_type, 
            func.count(Agent.id).label('count')
        ).group_by(Agent.agent_type).all()
        
        agents_by_type = {agent_type: count for agent_type, count in agent_types}
        
        # Workflow execution statistics
        total_executions = db.query(WorkflowExecution).count()
        successful_executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.status == 'completed'
        ).count()
        failed_executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.status == 'failed'
        ).count()
        
        # Calculate average performance (mock calculation based on success rate)
        if total_executions > 0:
            success_rate = successful_executions / total_executions
            average_performance = success_rate * 100  # Convert to percentage
        else:
            average_performance = 0.0
        
        return AgentStatistics(
            total_agents=total_agents,
            active_agents=active_agents,
            inactive_agents=inactive_agents,
            agents_by_type=agents_by_type,
            average_performance=round(average_performance, 2),
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions
        )
        
    except Exception as e:
        logger.error(f"Error getting agent statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        # Calculate uptime
        uptime_seconds = time.time() - SYSTEM_START_TIME
        uptime_hours = int(uptime_seconds // 3600)
        uptime_minutes = int((uptime_seconds % 3600) // 60)
        uptime = f"{uptime_hours}h {uptime_minutes}m"
        
        # Get system metrics using psutil
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Mock values for other metrics (in a real system, these would come from actual monitoring)
        active_connections = 10  # Would come from connection pool or monitoring
        total_requests = 1000  # Would come from request counter
        error_rate = 2.5  # Would be calculated from error logs
        
        return SystemMetrics(
            uptime=uptime,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_connections=active_connections,
            total_requests=total_requests,
            error_rate=error_rate
        )
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/skills/statistics")
async def get_skill_statistics(db: Session = Depends(get_db)):
    """Get skill-related statistics"""
    try:
        # Skills by category
        skill_categories = db.query(
            Skill.category,
            func.count(Skill.id).label('count')
        ).filter(Skill.is_active == True).group_by(Skill.category).all()
        
        skills_by_category = {category: count for category, count in skill_categories}
        
        # Skills by type
        skill_types = db.query(
            Skill.skill_type,
            func.count(Skill.id).label('count')
        ).filter(Skill.is_active == True).group_by(Skill.skill_type).all()
        
        skills_by_type = {skill_type: count for skill_type, count in skill_types}
        
        # Total skills
        total_skills = db.query(Skill).filter(Skill.is_active == True).count()
        
        # Most used skills (based on agent associations)
        from sqlalchemy import text
        most_used_skills = db.execute(text("""
            SELECT s.name, s.category, COUNT(ags.agent_id) as usage_count
            FROM skills s
            LEFT JOIN agent_skills ags ON s.id = ags.skill_id
            WHERE s.is_active = 1
            GROUP BY s.id, s.name, s.category
            ORDER BY usage_count DESC
            LIMIT 10
        """)).fetchall()
        
        most_used = [
            {
                "name": row[0],
                "category": row[1],
                "usage_count": row[2]
            } for row in most_used_skills
        ]
        
        return {
            "total_skills": total_skills,
            "skills_by_category": skills_by_category,
            "skills_by_type": skills_by_type,
            "most_used_skills": most_used
        }
        
    except Exception as e:
        logger.error(f"Error getting skill statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patterns/statistics")
async def get_pattern_statistics(db: Session = Depends(get_db)):
    """Get pattern-related statistics"""
    try:
        # Patterns by type
        pattern_types = db.query(
            Pattern.pattern_type,
            func.count(Pattern.id).label('count')
        ).filter(Pattern.is_active == True).group_by(Pattern.pattern_type).all()
        
        patterns_by_type = {pattern_type: count for pattern_type, count in pattern_types}
        
        # Total patterns
        total_patterns = db.query(Pattern).filter(Pattern.is_active == True).count()
        
        # Most used patterns
        most_used_patterns = db.query(Pattern).filter(
            Pattern.is_active == True
        ).order_by(Pattern.usage_count.desc()).limit(10).all()
        
        most_used = [
            {
                "id": pattern.id,
                "name": pattern.name,
                "pattern_type": pattern.pattern_type,
                "usage_count": pattern.usage_count,
                "effectiveness_score": pattern.effectiveness_score
            } for pattern in most_used_patterns
        ]
        
        # Average effectiveness score
        avg_effectiveness = db.query(
            func.avg(Pattern.effectiveness_score)
        ).filter(
            Pattern.is_active == True,
            Pattern.usage_count > 0
        ).scalar() or 0.0
        
        return {
            "total_patterns": total_patterns,
            "patterns_by_type": patterns_by_type,
            "most_used_patterns": most_used,
            "average_effectiveness": round(avg_effectiveness, 3)
        }
        
    except Exception as e:
        logger.error(f"Error getting pattern statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/statistics")
async def get_performance_statistics(db: Session = Depends(get_db)):
    """Get performance-related statistics"""
    try:
        # Recent executions (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        
        recent_executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.started_at >= yesterday
        ).count()
        
        recent_successful = db.query(WorkflowExecution).filter(
            and_(
                WorkflowExecution.started_at >= yesterday,
                WorkflowExecution.status == 'completed'
            )
        ).count()
        
        recent_failed = db.query(WorkflowExecution).filter(
            and_(
                WorkflowExecution.started_at >= yesterday,
                WorkflowExecution.status == 'failed'
            )
        ).count()
        
        # Success rate
        success_rate = (recent_successful / recent_executions * 100) if recent_executions > 0 else 0
        
        # Average execution time (mock calculation)
        # In a real system, this would calculate actual execution times
        avg_execution_time = 45.2  # seconds
        
        # Performance by agent type
        agent_performance = db.execute(text("""
            SELECT a.agent_type, 
                   COUNT(we.id) as total_executions,
                   SUM(CASE WHEN we.status = 'completed' THEN 1 ELSE 0 END) as successful_executions
            FROM agents a
            LEFT JOIN workflow_executions we ON a.id = we.agent_id
            WHERE we.started_at >= :yesterday
            GROUP BY a.agent_type
        """), {"yesterday": yesterday}).fetchall()
        
        performance_by_type = {}
        for row in agent_performance:
            agent_type, total, successful = row
            success_rate_type = (successful / total * 100) if total > 0 else 0
            performance_by_type[agent_type] = {
                "total_executions": total,
                "successful_executions": successful,
                "success_rate": round(success_rate_type, 2)
            }
        
        return {
            "recent_executions_24h": recent_executions,
            "recent_successful_24h": recent_successful,
            "recent_failed_24h": recent_failed,
            "success_rate_24h": round(success_rate, 2),
            "average_execution_time_seconds": avg_execution_time,
            "performance_by_agent_type": performance_by_type
        }
        
    except Exception as e:
        logger.error(f"Error getting performance statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
