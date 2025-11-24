"""
Agent Performance Service
=========================

Provides real-time agent performance metrics from the database for intelligent agent selection.
Calculates success rates, quality scores, workload, and cost efficiency from historical executions.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from models import Agent, WorkflowExecution
from database.database import get_db

logger = logging.getLogger(__name__)


class AgentPerformanceService:
    """
    Service for querying and calculating agent performance metrics.
    Used by IntelligentAgentSelector for data-driven agent selection.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)
        
    async def get_agent_performance_data(
        self,
        agent_ids: Optional[List[int]] = None,
        lookback_days: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for multiple agents.
        
        Args:
            agent_ids: List of agent IDs to query (None = all active agents)
            lookback_days: Number of days to look back for execution history
            
        Returns:
            Dictionary mapping agent name -> performance metrics:
            {
                "agent_name": {
                    "success_rate": 0.95,
                    "avg_quality": 0.87,
                    "total_executions": 42,
                    "avg_tokens": 1500,
                    "avg_execution_time": 3.2
                }
            }
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # Query agents with their aggregated execution stats
        query = self.db.query(
            Agent.name,
            Agent.quality_score,
            Agent.performance,
            Agent.reliability,
            Agent.model_usage_stats,
            func.count(WorkflowExecution.id).label('total_executions'),
            func.sum(
                func.cast(WorkflowExecution.status == 'completed', Integer)
            ).label('successful_executions')
        ).outerjoin(
            WorkflowExecution,
            and_(
                WorkflowExecution.agent_id == Agent.id,
                WorkflowExecution.started_at >= cutoff_date
            )
        ).filter(Agent.status == 'active')
        
        if agent_ids:
            query = query.filter(Agent.id.in_(agent_ids))
            
        query = query.group_by(Agent.id, Agent.name, Agent.quality_score, 
                               Agent.performance, Agent.reliability, 
                               Agent.model_usage_stats)
        
        results = query.all()
        
        # Calculate performance metrics for each agent
        performance_data = {}
        for result in results:
            name = result.name
            total_execs = result.total_executions or 0
            successful_execs = result.successful_executions or 0
            
            # Calculate success rate
            success_rate = (successful_execs / total_execs) if total_execs > 0 else 0.8
            
            # Use quality_score from agent if available, otherwise fallback to performance
            avg_quality = result.quality_score or result.performance or 0.7
            
            # Extract token usage from model_usage_stats
            usage_stats = result.model_usage_stats or {}
            avg_tokens = usage_stats.get('avg_tokens_per_request', 0)
            
            # Calculate avg execution time (placeholder - would need to query execution logs)
            # For now, estimate based on reliability: higher reliability = faster
            reliability = result.reliability or 0.5
            avg_execution_time = 5.0 - (reliability * 3.0)  # 2-5 seconds range
            
            performance_data[name] = {
                "success_rate": min(1.0, success_rate),
                "avg_quality": min(1.0, avg_quality),
                "total_executions": total_execs,
                "avg_tokens": int(avg_tokens),
                "avg_execution_time": round(avg_execution_time, 2),
                "reliability": result.reliability or 0.5
            }
            
        self.logger.info(f"Retrieved performance data for {len(performance_data)} agents")
        return performance_data
        
    async def get_agent_workload(self, agent_ids: Optional[List[int]] = None) -> Dict[int, int]:
        """
        Get current workload (running tasks) for agents.
        
        Args:
            agent_ids: List of agent IDs to query (None = all agents)
            
        Returns:
            Dictionary mapping agent_id -> count of running tasks
        """
        query = self.db.query(
            WorkflowExecution.agent_id,
            func.count(WorkflowExecution.id).label('running_count')
        ).filter(
            WorkflowExecution.status.in_(['pending', 'running'])
        )
        
        if agent_ids:
            query = query.filter(WorkflowExecution.agent_id.in_(agent_ids))
            
        query = query.group_by(WorkflowExecution.agent_id)
        
        results = query.all()
        
        workload = {row.agent_id: row.running_count for row in results}
        
        self.logger.debug(f"Current workload: {workload}")
        return workload
        
    async def update_agent_metrics(
        self,
        agent_id: int,
        execution_success: bool,
        quality: float,
        tokens_used: int,
        execution_time: float
    ):
        """
        Update agent performance metrics after an execution.
        
        Args:
            agent_id: Agent ID
            execution_success: Whether execution succeeded
            quality: Quality score (0-1)
            tokens_used: Tokens consumed
            execution_time: Execution duration in seconds
        """
        agent = self.db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            self.logger.warning(f"Agent {agent_id} not found for metrics update")
            return
            
        # Update usage stats
        usage_stats = agent.model_usage_stats or {}
        total_requests = usage_stats.get('total_requests', 0) + 1
        total_tokens = usage_stats.get('total_tokens', 0) + tokens_used
        
        usage_stats['total_requests'] = total_requests
        usage_stats['total_tokens'] = total_tokens
        usage_stats['avg_tokens_per_request'] = total_tokens / total_requests
        usage_stats['last_used_at'] = datetime.now().isoformat()
        
        agent.model_usage_stats = usage_stats
        
        # Update quality score (exponential moving average)
        alpha = 0.2  # Smoothing factor
        current_quality = agent.quality_score or 0.7
        agent.quality_score = (alpha * quality) + ((1 - alpha) * current_quality)
        
        # Update reliability score (based on success)
        current_reliability = agent.reliability or 0.5
        reliability_update = 1.0 if execution_success else 0.0
        agent.reliability = (alpha * reliability_update) + ((1 - alpha) * current_reliability)
        
        self.db.commit()
        self.logger.debug(f"Updated metrics for agent {agent_id}")


def get_performance_service(db: Optional[Session] = None) -> AgentPerformanceService:
    """
    Factory function to get an AgentPerformanceService instance.
    
    Args:
        db: Database session (if None, gets a new session)
        
    Returns:
        AgentPerformanceService instance
    """
    if not db:
        db = next(get_db())
    return AgentPerformanceService(db)
