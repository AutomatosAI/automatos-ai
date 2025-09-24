"""
PRD-06: Analytics Engine
Real-time analytics and metrics calculation for the dashboard
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, desc
import psutil
import redis

from database.database import get_db
from database.models import (
    Agent, Workflow, WorkflowExecution, CollaborationSession,
    SystemMetrics
)

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """
    Calculates and aggregates analytics data for the dashboard
    """
    
    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client
        self.db = next(get_db())
        
        # Initialize Redis connection if not provided
        if not self.redis_client:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host="127.0.0.1",
                    port=6379,
                    password="redis_password_123",
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis connection established successfully")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None
    
    async def get_dashboard_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard overview data
        """
        try:
            # Get system health metrics
            system_health = await self._get_system_health()
            
            # Get agent metrics
            agent_metrics = await self._get_agent_metrics()
            
            # Get workflow metrics
            workflow_metrics = await self._get_workflow_metrics()
            
            # Get context optimization metrics
            context_metrics = await self._get_context_metrics()
            
            # Get learning metrics
            learning_metrics = await self._get_learning_metrics()
            
            return {
                "systemHealth": system_health,
                "agentMetrics": agent_metrics,
                "workflowMetrics": workflow_metrics,
                "contextMetrics": context_metrics,
                "learningMetrics": learning_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            return {"error": str(e)}
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get real-time system health metrics"""
        try:
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Database connection health
            db_health = await self._check_database_health()
            
            # Redis health
            redis_health = await self._check_redis_health()
            
            return {
                "cpuUsage": cpu_percent,
                "memoryUsage": memory.percent,
                "diskUsage": disk.percent,
                "databaseStatus": "healthy" if db_health else "unhealthy",
                "redisStatus": "healthy" if redis_health else "unhealthy",
                "uptime": self._get_system_uptime()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"error": str(e)}
    
    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        try:
            # Active agents count
            active_agents = self.db.query(Agent).filter(Agent.status == 'active').count()
            total_agents = self.db.query(Agent).count()
            
            # Basic metrics
            return {
                "activeAgents": active_agents,
                "totalAgents": total_agents,
                "successRate": 85.0,  # Placeholder
                "avgExecutionTime": 2.5,  # Placeholder
                "totalTokensUsed": 0,  # Placeholder
                "recentExecutions": 0  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error getting agent metrics: {e}")
            return {"error": str(e)}
    
    async def _get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics"""
        try:
            # Total workflows
            total_workflows = self.db.query(Workflow).count()
            completed_workflows = self.db.query(Workflow).filter(Workflow.status == 'completed').count()
            pending_workflows = self.db.query(Workflow).filter(Workflow.status == 'pending').count()
            
            # Workflow executions
            total_executions = self.db.query(WorkflowExecution).count()
            successful_executions = self.db.query(WorkflowExecution).filter(WorkflowExecution.status == 'completed').count()
            
            # Recent activity (last hour)
            since = datetime.now() - timedelta(hours=1)
            recent_workflows = self.db.query(Workflow).filter(Workflow.created_at >= since).count()
            
            return {
                "totalWorkflows": total_workflows,
                "completedWorkflows": completed_workflows,
                "pendingWorkflows": pending_workflows,
                "completionRate": round((completed_workflows / total_workflows * 100) if total_workflows > 0 else 0, 2),
                "totalExecutions": total_executions,
                "successfulExecutions": successful_executions,
                "successRate": round((successful_executions / total_executions * 100) if total_executions > 0 else 0, 2),
                "recentWorkflows": recent_workflows
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow metrics: {e}")
            return {"error": str(e)}
    
    async def _get_context_metrics(self) -> Dict[str, Any]:
        """Get context optimization metrics"""
        try:
            # Placeholder metrics for now
            return {
                "tokensSaved": 0,
                "avgCompressionRatio": 0.0,
                "totalOptimizations": 0,
                "efficiency": 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting context metrics: {e}")
            return {"error": str(e)}
    
    async def _get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning and memory metrics"""
        try:
            # Placeholder metrics for now
            return {
                "totalMemoryItems": 0,
                "recentMemoryItems": 0,
                "knowledgeNodes": 0,
                "activeCollaborations": 0,
                "totalCollaborations": 0,
                "knowledgeGrowth": 0,
                "memoryConsolidations": 0,
                "avgImprovement": 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting learning metrics: {e}")
            return {"error": str(e)}
    
    async def _check_database_health(self) -> bool:
        """Check if database is healthy"""
        try:
            self.db.execute(select(1))
            return True
        except:
            return False
    
    async def _check_redis_health(self) -> bool:
        """Check if Redis is healthy"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
        except:
            pass
        return False
    
    def _get_system_uptime(self) -> str:
        """Get system uptime"""
        try:
            uptime_seconds = psutil.boot_time()
            uptime = datetime.now() - datetime.fromtimestamp(uptime_seconds)
            return str(uptime).split('.')[0]  # Remove microseconds
        except:
            return "Unknown"
    
    async def get_agent_analytics(self, agent_id: int, period: str = "7d") -> Dict[str, Any]:
        """Get detailed analytics for a specific agent"""
        try:
            # Placeholder implementation
            return {
                "agentId": agent_id,
                "period": period,
                "successRate": 85.0,
                "avgExecutionTime": 2.5,
                "totalTokensUsed": 0,
                "totalExecutions": 0,
                "performanceTrend": []
            }
            
        except Exception as e:
            logger.error(f"Error getting agent analytics: {e}")
            return {"error": str(e)}
    
    async def track_agent_execution(
        self,
        agent_id: int,
        workflow_id: int,
        execution_time: float,
        tokens_used: int = 0,
        success: bool = True,
        error_message: str = None,
        context_optimization_applied: bool = False,
        memory_items_created: int = 0,
        collaboration_sessions: int = 0
    ):
        """Track agent execution for analytics"""
        try:
            # For now, just log the tracking
            logger.info(f"Agent {agent_id} executed workflow {workflow_id} in {execution_time}s")
            
            # Publish update to Redis for real-time dashboard
            if self.redis_client:
                update = {
                    "type": "agent_execution",
                    "agent_id": agent_id,
                    "workflow_id": workflow_id,
                    "execution_time": execution_time,
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
                self.redis_client.publish("dashboard_updates", json.dumps(update))
            
        except Exception as e:
            logger.error(f"Error tracking agent execution: {e}")
    
    async def track_context_optimization(
        self,
        original_tokens: int,
        optimized_tokens: int,
        optimization_type: str,
        pattern_used: str = None,
        execution_time: float = None
    ):
        """Track context optimization for analytics"""
        try:
            # For now, just log the tracking
            logger.info(f"Context optimization: {original_tokens} -> {optimized_tokens} tokens")
            
            # Publish update to Redis
            if self.redis_client:
                update = {
                    "type": "context_optimization",
                    "tokens_saved": original_tokens - optimized_tokens,
                    "compression_ratio": optimized_tokens / original_tokens,
                    "timestamp": datetime.now().isoformat()
                }
                self.redis_client.publish("dashboard_updates", json.dumps(update))
            
        except Exception as e:
            logger.error(f"Error tracking context optimization: {e}")
    
    async def track_learning_progress(
        self,
        agent_id: int,
        knowledge_items: int = 0,
        memory_consolidations: int = 0,
        performance_improvement: float = 0.0,
        knowledge_transfers: int = 0
    ):
        """Track learning progress for analytics"""
        try:
            # For now, just log the tracking
            logger.info(f"Learning progress for agent {agent_id}: {knowledge_items} items")
            
            # Publish update to Redis
            if self.redis_client:
                update = {
                    "type": "learning_progress",
                    "agent_id": agent_id,
                    "knowledge_items": knowledge_items,
                    "performance_improvement": performance_improvement,
                    "timestamp": datetime.now().isoformat()
                }
                self.redis_client.publish("dashboard_updates", json.dumps(update))
            
        except Exception as e:
            logger.error(f"Error tracking learning progress: {e}")
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for WebSocket updates"""
        try:
            # Get current system metrics
            system_health = await self._get_system_health()
            
            return {
                "systemHealth": system_health,
                "recentExecutions": 0,
                "recentOptimizations": 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {"error": str(e)}