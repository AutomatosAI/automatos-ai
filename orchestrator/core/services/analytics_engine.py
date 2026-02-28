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

from core.database.database import get_db
from core.models import (
    Agent, Workflow, WorkflowExecution,
    SystemMetrics
)

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """
    Calculates and aggregates analytics data for the dashboard
    """
    
    def __init__(self, redis_client: redis.Redis = None):
        # Use centralized Redis client
        from core.redis.client import get_redis_client
        self.redis_client = get_redis_client()
        if self.redis_client:
            logger.info("Redis connection established successfully")
        else:
            logger.warning("Redis client not initialized")

        from core.database.database import SessionLocal
        self.db = SessionLocal()

    def close(self):
        """Close the database session. Call on shutdown."""
        if self.db:
            self.db.close()
    
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
        """Get context optimization metrics from the database"""
        try:
            # Safe database query with error handling
            try:
                # Check if the context_optimization_metrics table exists
                from sqlalchemy import text, func, exc
                
                # Safer approach to check if table exists
                try:
                    # Query to count total optimizations with safer SQL
                    total_optimizations_query = text("""
                        SELECT COUNT(*) 
                        FROM information_schema.tables 
                        WHERE table_name = 'context_optimization_metrics'
                    """)
                    
                    table_exists = self.db.execute(total_optimizations_query).scalar() or 0
                    
                    if table_exists > 0:
                        # Table exists, get real metrics
                        tokens_saved_query = text("""
                            SELECT COALESCE(SUM(tokens_saved), 0)
                            FROM context_optimization_metrics
                        """)
                        
                        total_optimizations_query = text("""
                            SELECT COUNT(*)
                            FROM context_optimization_metrics
                        """)
                        
                        avg_compression_query = text("""
                            SELECT COALESCE(AVG(compression_ratio), 1.0)
                            FROM context_optimization_metrics
                        """)
                        
                        tokens_saved = self.db.execute(tokens_saved_query).scalar() or 0
                        total_optimizations = self.db.execute(total_optimizations_query).scalar() or 0
                        avg_compression = self.db.execute(avg_compression_query).scalar() or 1.0
                        
                        efficiency = 0.0
                        if total_optimizations > 0:
                            efficiency = min(((tokens_saved / max(1, total_optimizations)) * avg_compression) / 1000, 1.0) * 100
                            
                        return {
                            "tokensSaved": tokens_saved,
                            "avgCompressionRatio": avg_compression,
                            "totalOptimizations": total_optimizations,
                            "efficiency": efficiency
                        }
                    else:
                        # Table doesn't exist, return zeros
                        return {
                            "tokensSaved": 0, 
                            "avgCompressionRatio": 1.0,
                            "totalOptimizations": 0,
                            "efficiency": 0.0
                        }
                except exc.SQLAlchemyError as sql_error:
                    # SQL error occurred, return zeros
                    logger.warning(f"SQL error checking context metrics table: {sql_error}")
                    return {
                        "tokensSaved": 0, 
                        "avgCompressionRatio": 1.0,
                        "totalOptimizations": 0,
                        "efficiency": 0.0
                    }
                    
            except Exception as db_error:
                # Database query failed, log error and return zeros
                logger.warning(f"Failed to query context metrics from database: {db_error}")
                return {
                    "tokensSaved": 0, 
                    "avgCompressionRatio": 1.0,
                    "totalOptimizations": 0,
                    "efficiency": 0.0
                }
                
        except Exception as e:
            logger.error(f"Error getting context metrics: {e}")
            return {
                "tokensSaved": 0, 
                "avgCompressionRatio": 1.0,
                "totalOptimizations": 0,
                "efficiency": 0.0
            }
    
    async def _get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning and memory metrics from the database"""
        try:
            # Try to fetch real metrics from the database if available
            try:
                # Import required SQLAlchemy components
                from sqlalchemy import text, func
                
                # Query to get total memory items
                total_memory_query = text("""
                    SELECT COUNT(*) 
                    FROM memory_items
                """)
                
                # Query to get recent memory items (last 7 days)
                recent_memory_query = text("""
                    SELECT COUNT(*) 
                    FROM memory_items 
                    WHERE created_at > NOW() - INTERVAL '7 days'
                """)
                
                # Query to get knowledge nodes
                knowledge_nodes_query = text("""
                    SELECT COUNT(*) 
                    FROM knowledge_nodes
                """)
                
                # Query to get active collaborations
                active_collab_query = text("""
                    SELECT COUNT(*) 
                    FROM collaboration_sessions 
                    WHERE status = 'active'
                """)
                
                # Query to get total collaborations
                total_collab_query = text("""
                    SELECT COUNT(*) 
                    FROM collaboration_sessions
                """)
                
                # Query to get memory consolidations
                consolidations_query = text("""
                    SELECT COUNT(*) 
                    FROM learning_progress_tracking 
                    WHERE recorded_at > NOW() - INTERVAL '30 days'
                """)
                
                # Execute queries and get metrics
                total_memory = self.db.execute(total_memory_query).scalar() or 0
                recent_memory = self.db.execute(recent_memory_query).scalar() or 0
                knowledge_nodes = self.db.execute(knowledge_nodes_query).scalar() or 0
                active_collaborations = self.db.execute(active_collab_query).scalar() or 0
                total_collaborations = self.db.execute(total_collab_query).scalar() or 0
                memory_consolidations = self.db.execute(consolidations_query).scalar() or 0
                
                # Calculate knowledge growth (percentage increase in last 30 days)
                knowledge_growth_query = text("""
                    SELECT 
                        (COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM knowledge_nodes 
                                                   WHERE created_at < NOW() - INTERVAL '30 days'), 0)) - 100 
                    FROM knowledge_nodes 
                    WHERE created_at > NOW() - INTERVAL '30 days'
                """)
                
                knowledge_growth = self.db.execute(knowledge_growth_query).scalar() or 0
                
                # Calculate average improvement
                improvement_query = text("""
                    SELECT COALESCE(AVG(performance_improvement), 0.0) 
                    FROM learning_progress_tracking 
                    WHERE recorded_at > NOW() - INTERVAL '30 days'
                """)
                
                avg_improvement = self.db.execute(improvement_query).scalar() or 0.0
                
                # Return real metrics
                return {
                    "totalMemoryItems": total_memory,
                    "recentMemoryItems": recent_memory,
                    "knowledgeNodes": knowledge_nodes,
                    "activeCollaborations": active_collaborations,
                    "totalCollaborations": total_collaborations,
                    "knowledgeGrowth": knowledge_growth,
                    "memoryConsolidations": memory_consolidations,
                    "avgImprovement": avg_improvement
                }
                
            except Exception as db_error:
                # Database query failed, log error and return zeros
                logger.warning(f"Failed to query learning metrics from database: {db_error}")
                # Return zeros for consistent error handling
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
    
    async def _check_database_health(self) -> bool:
        """Check if database is healthy"""
        try:
            self.db.execute(select(1))
            return True
        except Exception:
            return False
    
    async def _check_redis_health(self) -> bool:
        """Check if Redis is healthy"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
        except Exception:
            pass
        return False
    
    def _get_system_uptime(self) -> str:
        """Get system uptime"""
        try:
            uptime_seconds = psutil.boot_time()
            uptime = datetime.now() - datetime.fromtimestamp(uptime_seconds)
            return str(uptime).split('.')[0]  # Remove microseconds
        except Exception:
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