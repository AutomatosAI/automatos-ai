"""
Analytics Engine for Real-time Dashboard
========================================

Aggregates REAL metrics from database tables for the monitoring dashboard.
NO MOCK DATA - All metrics are calculated from actual system data.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc, asc, distinct, case
from sqlalchemy.sql import text
import redis.asyncio as redis
from dataclasses import dataclass, asdict
import psutil

logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    """Real agent performance metrics"""
    agent_id: str
    name: str
    agent_type: str
    execution_count: int
    total_tokens_used: int
    success_rate: float
    avg_execution_time: float
    error_rate: float
    last_active: Optional[datetime]
    memory_items_count: int
    collaboration_count: int
    status: str

@dataclass
class TaskMetrics:
    """Real task execution metrics"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    avg_completion_time: float
    decomposition_depth: float
    subtask_success_rate: float
    collaboration_efficiency: float

@dataclass
class MemoryMetrics:
    """Real memory system metrics"""
    total_memories: int
    working_memory_count: int
    short_term_count: int
    long_term_count: int
    consolidation_rate: float
    knowledge_nodes: int
    knowledge_edges: int
    avg_importance_score: float
    memory_growth_rate: float

@dataclass
class ContextMetrics:
    """Real context optimization metrics"""
    total_optimizations: int
    avg_tokens_saved: float
    total_tokens_saved: int
    avg_information_density: float
    compression_ratio: float
    optimization_success_rate: float
    patterns_used: Dict[str, int]

@dataclass
class CollaborationMetrics:
    """Real collaboration metrics"""
    total_sessions: int
    active_sessions: int
    avg_agents_per_session: float
    shared_contexts_count: int
    avg_access_count: float
    collaboration_success_rate: float
    message_exchange_count: int

class AnalyticsEngine:
    """
    Real-time analytics engine that aggregates data from PostgreSQL
    """
    
    def __init__(self, db_session: Session, redis_client: Optional[redis.Redis] = None):
        self.db = db_session
        self.redis = redis_client
        self.cache = {}
        self.cache_ttl = 60  # Cache for 60 seconds
    
    async def get_agent_metrics(self, time_range: Optional[timedelta] = None) -> List[AgentMetrics]:
        """Get real agent metrics from database"""
        try:
            # Base query for agents
            query = self.db.query(
                'Agent.id',
                'Agent.name', 
                'Agent.agent_type',
                'Agent.execution_count',
                'Agent.total_tokens_used',
                'Agent.success_rate',
                'Agent.avg_execution_time',
                'Agent.status'
            ).from_statement(text("""
                SELECT 
                    a.id,
                    a.name,
                    a.agent_type,
                    COALESCE(a.execution_count, 0) as execution_count,
                    COALESCE(a.total_tokens_used, 0) as total_tokens_used,
                    COALESCE(a.success_rate, 0) as success_rate,
                    COALESCE(a.avg_execution_time, 0) as avg_execution_time,
                    a.status,
                    COUNT(DISTINCT m.id) as memory_items_count,
                    COUNT(DISTINCT cs.id) as collaboration_count,
                    MAX(ap.recorded_at) as last_active
                FROM agents a
                LEFT JOIN memory_items m ON m.agent_id = a.id
                LEFT JOIN collaboration_sessions cs ON cs.initiator_agent_id = a.id
                LEFT JOIN agent_performance ap ON ap.agent_id = a.id
                GROUP BY a.id, a.name, a.agent_type, a.execution_count, 
                         a.total_tokens_used, a.success_rate, a.avg_execution_time, a.status
            """))
            
            results = query.all()
            
            metrics = []
            for row in results:
                error_rate = 100 - row.success_rate if row.success_rate else 0
                
                metrics.append(AgentMetrics(
                    agent_id=str(row.id),
                    name=row.name,
                    agent_type=row.agent_type,
                    execution_count=row.execution_count or 0,
                    total_tokens_used=row.total_tokens_used or 0,
                    success_rate=row.success_rate or 0,
                    avg_execution_time=row.avg_execution_time or 0,
                    error_rate=error_rate,
                    last_active=row.last_active if hasattr(row, 'last_active') else None,
                    memory_items_count=row.memory_items_count if hasattr(row, 'memory_items_count') else 0,
                    collaboration_count=row.collaboration_count if hasattr(row, 'collaboration_count') else 0,
                    status=row.status
                ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting agent metrics: {e}")
            return []
    
    async def get_task_metrics(self, time_range: Optional[timedelta] = None) -> TaskMetrics:
        """Get real task execution metrics from database"""
        try:
            # Build time filter if specified
            time_filter = ""
            if time_range:
                cutoff = datetime.now() - time_range
                time_filter = f"WHERE td.created_at >= '{cutoff.isoformat()}'"
            
            # Query task decomposition and assignment stats
            task_stats = self.db.execute(text(f"""
                SELECT 
                    COUNT(DISTINCT td.id) as total_tasks,
                    COUNT(DISTINCT CASE WHEN td.status = 'completed' THEN td.id END) as completed_tasks,
                    COUNT(DISTINCT CASE WHEN td.status = 'failed' THEN td.id END) as failed_tasks,
                    COUNT(DISTINCT CASE WHEN td.status IN ('pending', 'in_progress') THEN td.id END) as pending_tasks,
                    AVG(EXTRACT(EPOCH FROM (td.updated_at - td.created_at))) as avg_completion_time,
                    AVG(COALESCE(json_array_length(td.subtasks), 0)) as avg_decomposition_depth,
                    COUNT(DISTINCT ta.id) as total_assignments,
                    COUNT(DISTINCT CASE WHEN ta.status = 'completed' THEN ta.id END) as completed_assignments
                FROM task_decompositions td
                LEFT JOIN task_assignments ta ON ta.decomposition_id = td.id
                {time_filter}
            """)).fetchone()
            
            # Calculate subtask success rate
            subtask_success_rate = 0
            if task_stats.total_assignments > 0:
                subtask_success_rate = (task_stats.completed_assignments / task_stats.total_assignments) * 100
            
            # Query collaboration efficiency
            collab_stats = self.db.execute(text(f"""
                SELECT 
                    AVG(json_array_length(cs.participants)) as avg_participants,
                    COUNT(CASE WHEN cs.status = 'completed' AND cs.result IS NOT NULL THEN 1 END) as successful_collabs,
                    COUNT(cs.id) as total_collabs
                FROM collaboration_sessions cs
                {time_filter.replace('td.created_at', 'cs.created_at') if time_filter else ''}
            """)).fetchone()
            
            collaboration_efficiency = 0
            if collab_stats.total_collabs > 0:
                collaboration_efficiency = (collab_stats.successful_collabs / collab_stats.total_collabs) * 100
            
            return TaskMetrics(
                total_tasks=task_stats.total_tasks or 0,
                completed_tasks=task_stats.completed_tasks or 0,
                failed_tasks=task_stats.failed_tasks or 0,
                pending_tasks=task_stats.pending_tasks or 0,
                avg_completion_time=round(task_stats.avg_completion_time or 0, 2),
                decomposition_depth=round(task_stats.avg_decomposition_depth or 0, 1),
                subtask_success_rate=round(subtask_success_rate, 1),
                collaboration_efficiency=round(collaboration_efficiency, 1)
            )
            
        except Exception as e:
            logger.error(f"Error getting task metrics: {e}")
            return TaskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    async def get_memory_metrics(self) -> MemoryMetrics:
        """Get real memory system metrics from database"""
        try:
            # Query memory items statistics
            memory_stats = self.db.execute(text("""
                SELECT 
                    COUNT(*) as total_memories,
                    COUNT(CASE WHEN memory_level = 'working' THEN 1 END) as working_count,
                    COUNT(CASE WHEN memory_level = 'short_term' THEN 1 END) as short_term_count,
                    COUNT(CASE WHEN memory_level = 'long_term' THEN 1 END) as long_term_count,
                    AVG(importance) as avg_importance,
                    COUNT(CASE WHEN consolidation_count > 0 THEN 1 END) as consolidated_memories
                FROM memory_items
            """)).fetchone()
            
            # Query knowledge graph statistics
            knowledge_stats = self.db.execute(text("""
                SELECT 
                    (SELECT COUNT(*) FROM knowledge_nodes) as node_count,
                    (SELECT COUNT(*) FROM knowledge_edges) as edge_count
            """)).fetchone()
            
            # Calculate memory growth rate (last 7 days)
            growth_stats = self.db.execute(text("""
                SELECT 
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as recent_memories,
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '14 days' 
                               AND created_at < NOW() - INTERVAL '7 days' THEN 1 END) as previous_memories
                FROM memory_items
            """)).fetchone()
            
            growth_rate = 0
            if growth_stats.previous_memories > 0:
                growth_rate = ((growth_stats.recent_memories - growth_stats.previous_memories) / 
                             growth_stats.previous_memories) * 100
            
            consolidation_rate = 0
            if memory_stats.total_memories > 0:
                consolidation_rate = (memory_stats.consolidated_memories / memory_stats.total_memories) * 100
            
            return MemoryMetrics(
                total_memories=memory_stats.total_memories or 0,
                working_memory_count=memory_stats.working_count or 0,
                short_term_count=memory_stats.short_term_count or 0,
                long_term_count=memory_stats.long_term_count or 0,
                consolidation_rate=round(consolidation_rate, 1),
                knowledge_nodes=knowledge_stats.node_count or 0,
                knowledge_edges=knowledge_stats.edge_count or 0,
                avg_importance_score=round(memory_stats.avg_importance or 0, 2),
                memory_growth_rate=round(growth_rate, 1)
            )
            
        except Exception as e:
            logger.error(f"Error getting memory metrics: {e}")
            return MemoryMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    async def get_context_metrics(self) -> ContextMetrics:
        """Get real context optimization metrics from database"""
        try:
            # Query context optimization statistics
            context_stats = self.db.execute(text("""
                SELECT 
                    COUNT(*) as total_optimizations,
                    AVG(tokens_saved) as avg_tokens_saved,
                    SUM(tokens_saved) as total_tokens_saved,
                    AVG(information_density) as avg_information_density,
                    AVG(original_tokens::float / NULLIF(optimized_tokens, 0)) as compression_ratio,
                    COUNT(CASE WHEN tokens_saved > 0 THEN 1 END) as successful_optimizations
                FROM context_optimizations
            """)).fetchone()
            
            # Query pattern usage
            pattern_usage = self.db.execute(text("""
                SELECT 
                    pattern_type,
                    COUNT(*) as usage_count
                FROM context_optimizations
                WHERE pattern_type IS NOT NULL
                GROUP BY pattern_type
                ORDER BY usage_count DESC
                LIMIT 10
            """)).fetchall()
            
            patterns = {row.pattern_type: row.usage_count for row in pattern_usage}
            
            success_rate = 0
            if context_stats.total_optimizations > 0:
                success_rate = (context_stats.successful_optimizations / context_stats.total_optimizations) * 100
            
            return ContextMetrics(
                total_optimizations=context_stats.total_optimizations or 0,
                avg_tokens_saved=round(context_stats.avg_tokens_saved or 0, 1),
                total_tokens_saved=context_stats.total_tokens_saved or 0,
                avg_information_density=round(context_stats.avg_information_density or 0, 3),
                compression_ratio=round(context_stats.compression_ratio or 1, 2),
                optimization_success_rate=round(success_rate, 1),
                patterns_used=patterns
            )
            
        except Exception as e:
            logger.error(f"Error getting context metrics: {e}")
            return ContextMetrics(0, 0, 0, 0, 1, 0, {})
    
    async def get_collaboration_metrics(self) -> CollaborationMetrics:
        """Get real collaboration metrics from database"""
        try:
            # Query collaboration session statistics
            collab_stats = self.db.execute(text("""
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_sessions,
                    AVG(json_array_length(participants)) as avg_participants,
                    COUNT(CASE WHEN status = 'completed' AND result IS NOT NULL THEN 1 END) as successful_sessions
                FROM collaboration_sessions
            """)).fetchone()
            
            # Query shared contexts statistics
            context_stats = self.db.execute(text("""
                SELECT 
                    COUNT(*) as shared_contexts,
                    AVG(access_count) as avg_access_count
                FROM shared_contexts
            """)).fetchone()
            
            # Query agent messages for collaboration
            message_stats = self.db.execute(text("""
                SELECT COUNT(*) as message_count
                FROM agent_messages
                WHERE session_id IS NOT NULL
            """)).fetchone()
            
            success_rate = 0
            if collab_stats.total_sessions > 0:
                success_rate = (collab_stats.successful_sessions / collab_stats.total_sessions) * 100
            
            return CollaborationMetrics(
                total_sessions=collab_stats.total_sessions or 0,
                active_sessions=collab_stats.active_sessions or 0,
                avg_agents_per_session=round(collab_stats.avg_participants or 0, 1),
                shared_contexts_count=context_stats.shared_contexts or 0,
                avg_access_count=round(context_stats.avg_access_count or 0, 1),
                collaboration_success_rate=round(success_rate, 1),
                message_exchange_count=message_stats.message_count or 0
            )
            
        except Exception as e:
            logger.error(f"Error getting collaboration metrics: {e}")
            return CollaborationMetrics(0, 0, 0, 0, 0, 0, 0)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get real system health metrics"""
        try:
            # CPU and Memory from psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Database health check
            try:
                db_health = self.db.execute(text("SELECT 1")).scalar()
                db_status = "healthy" if db_health == 1 else "unhealthy"
                
                # Get database connection count
                db_connections = self.db.execute(text("""
                    SELECT count(*) FROM pg_stat_activity 
                    WHERE state != 'idle' AND datname = current_database()
                """)).scalar()
            except:
                db_status = "unhealthy"
                db_connections = 0
            
            # Redis health check
            redis_status = "healthy"
            redis_memory = 0
            if self.redis:
                try:
                    await self.redis.ping()
                    info = await self.redis.info('memory')
                    redis_memory = round(info.get('used_memory_rss', 0) / (1024 * 1024), 1)  # MB
                except:
                    redis_status = "unhealthy"
            
            # API response time (measure a simple query)
            start_time = datetime.now()
            self.db.execute(text("SELECT COUNT(*) FROM agents")).scalar()
            api_latency = (datetime.now() - start_time).total_seconds() * 1000  # ms
            
            # Queue depth from workflow_executions
            queue_depth = self.db.execute(text("""
                SELECT COUNT(*) FROM workflow_executions
                WHERE status IN ('pending', 'queued')
            """)).scalar() or 0
            
            # Error rate (last hour)
            error_stats = self.db.execute(text("""
                SELECT 
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                    COUNT(*) as total
                FROM workflow_executions
                WHERE created_at >= NOW() - INTERVAL '1 hour'
            """)).fetchone()
            
            error_rate = 0
            if error_stats and error_stats.total > 0:
                error_rate = (error_stats.failed / error_stats.total) * 100
            
            # Determine overall health status
            health_score = 100
            if cpu_percent > 80:
                health_score -= 20
            if memory.percent > 85:
                health_score -= 20
            if db_status == "unhealthy":
                health_score -= 30
            if redis_status == "unhealthy":
                health_score -= 15
            if error_rate > 10:
                health_score -= 15
            
            if health_score >= 80:
                overall_status = "healthy"
                status_color = "green"
            elif health_score >= 60:
                overall_status = "warning"
                status_color = "yellow"
            else:
                overall_status = "critical"
                status_color = "red"
            
            return {
                "overall_status": overall_status,
                "status_color": status_color,
                "health_score": health_score,
                "cpu": {
                    "usage_percent": round(cpu_percent, 1),
                    "cores": psutil.cpu_count(),
                    "status": "healthy" if cpu_percent < 80 else "warning" if cpu_percent < 90 else "critical"
                },
                "memory": {
                    "usage_percent": round(memory.percent, 1),
                    "used_gb": round(memory.used / (1024**3), 1),
                    "total_gb": round(memory.total / (1024**3), 1),
                    "status": "healthy" if memory.percent < 85 else "warning" if memory.percent < 95 else "critical"
                },
                "disk": {
                    "usage_percent": round(disk.percent, 1),
                    "used_gb": round(disk.used / (1024**3), 1),
                    "total_gb": round(disk.total / (1024**3), 1),
                    "status": "healthy" if disk.percent < 80 else "warning" if disk.percent < 90 else "critical"
                },
                "database": {
                    "status": db_status,
                    "active_connections": db_connections,
                    "response_time_ms": round(api_latency, 1)
                },
                "redis": {
                    "status": redis_status,
                    "memory_mb": redis_memory
                },
                "queue": {
                    "depth": queue_depth,
                    "status": "healthy" if queue_depth < 50 else "warning" if queue_depth < 100 else "critical"
                },
                "error_rate": round(error_rate, 1),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "overall_status": "unknown",
                "status_color": "gray",
                "health_score": 0,
                "error": str(e)
            }
    
    async def get_dashboard_overview(self) -> Dict[str, Any]:
        """Get complete dashboard overview with all metrics"""
        try:
            # Run all metric queries in parallel
            agent_metrics_task = asyncio.create_task(self.get_agent_metrics())
            task_metrics_task = asyncio.create_task(self.get_task_metrics())
            memory_metrics_task = asyncio.create_task(self.get_memory_metrics())
            context_metrics_task = asyncio.create_task(self.get_context_metrics())
            collab_metrics_task = asyncio.create_task(self.get_collaboration_metrics())
            health_task = asyncio.create_task(self.get_system_health())
            
            # Wait for all tasks to complete
            agent_metrics = await agent_metrics_task
            task_metrics = await task_metrics_task
            memory_metrics = await memory_metrics_task
            context_metrics = await context_metrics_task
            collab_metrics = await collab_metrics_task
            system_health = await health_task
            
            # Calculate aggregate statistics
            active_agents = len([a for a in agent_metrics if a.status == 'active'])
            total_agents = len(agent_metrics)
            
            # Success rate across all agents
            avg_success_rate = 0
            if agent_metrics:
                avg_success_rate = sum(a.success_rate for a in agent_metrics) / len(agent_metrics)
            
            # Total tokens saved percentage
            token_savings_percent = 0
            if context_metrics.total_optimizations > 0:
                # Estimate original tokens
                original_tokens_estimate = context_metrics.total_tokens_saved * (context_metrics.compression_ratio / (context_metrics.compression_ratio - 1))
                if original_tokens_estimate > 0:
                    token_savings_percent = (context_metrics.total_tokens_saved / original_tokens_estimate) * 100
            
            return {
                "summary": {
                    "active_agents": active_agents,
                    "total_agents": total_agents,
                    "tasks_completed": task_metrics.completed_tasks,
                    "tasks_pending": task_metrics.pending_tasks,
                    "overall_success_rate": round(avg_success_rate, 1),
                    "system_health": system_health["health_score"],
                    "tokens_saved": context_metrics.total_tokens_saved,
                    "token_savings_percent": round(token_savings_percent, 1),
                    "memory_items": memory_metrics.total_memories,
                    "knowledge_graph_size": memory_metrics.knowledge_nodes + memory_metrics.knowledge_edges,
                    "active_collaborations": collab_metrics.active_sessions
                },
                "agents": [asdict(a) for a in agent_metrics[:10]],  # Top 10 agents
                "tasks": asdict(task_metrics),
                "memory": asdict(memory_metrics),
                "context": asdict(context_metrics),
                "collaboration": asdict(collab_metrics),
                "system_health": system_health,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_agent_activity_heatmap(self, days: int = 7) -> Dict[str, Any]:
        """Get agent activity heatmap data for the last N days"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            # Query hourly activity from agent_performance table
            activity_data = self.db.execute(text("""
                SELECT 
                    DATE_PART('dow', recorded_at) as day_of_week,
                    DATE_PART('hour', recorded_at) as hour_of_day,
                    COUNT(*) as activity_count,
                    AVG(tokens_used) as avg_tokens,
                    AVG(execution_time) as avg_time
                FROM agent_performance
                WHERE recorded_at >= :cutoff
                GROUP BY day_of_week, hour_of_day
                ORDER BY day_of_week, hour_of_day
            """), {"cutoff": cutoff}).fetchall()
            
            # Create heatmap matrix
            heatmap_matrix = [[0 for _ in range(24)] for _ in range(7)]
            max_activity = 0
            
            for row in activity_data:
                dow = int(row.day_of_week)
                hour = int(row.hour_of_day)
                heatmap_matrix[dow][hour] = row.activity_count
                max_activity = max(max_activity, row.activity_count)
            
            # Identify peak hours
            peak_hours = []
            for row in activity_data:
                if row.activity_count > max_activity * 0.8:  # Top 20% activity
                    peak_hours.append({
                        "day": int(row.day_of_week),
                        "hour": int(row.hour_of_day),
                        "activity": row.activity_count,
                        "avg_tokens": round(row.avg_tokens or 0, 0),
                        "avg_time": round(row.avg_time or 0, 1)
                    })
            
            return {
                "heatmap": heatmap_matrix,
                "peak_hours": peak_hours[:10],  # Top 10 peak hours
                "max_activity": max_activity,
                "days_analyzed": days,
                "data_points": len(activity_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting activity heatmap: {e}")
            return {"heatmap": [], "peak_hours": [], "error": str(e)}
    
    async def get_learning_progress_timeline(self, days: int = 30) -> Dict[str, Any]:
        """Get learning progress over time"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            # Query daily memory and knowledge growth
            daily_progress = self.db.execute(text("""
                WITH daily_data AS (
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as memories_created,
                        AVG(importance) as avg_importance,
                        COUNT(CASE WHEN memory_level = 'long_term' THEN 1 END) as long_term_memories
                    FROM memory_items
                    WHERE created_at >= :cutoff
                    GROUP BY DATE(created_at)
                ),
                knowledge_data AS (
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as nodes_created
                    FROM knowledge_nodes
                    WHERE created_at >= :cutoff
                    GROUP BY DATE(created_at)
                ),
                learning_data AS (
                    SELECT 
                        DATE(recorded_at) as date,
                        AVG(performance_score) as avg_performance
                    FROM learning_outcomes
                    WHERE recorded_at >= :cutoff
                    GROUP BY DATE(recorded_at)
                )
                SELECT 
                    COALESCE(d.date, k.date, l.date) as date,
                    COALESCE(d.memories_created, 0) as memories_created,
                    COALESCE(d.avg_importance, 0) as avg_importance,
                    COALESCE(d.long_term_memories, 0) as long_term_memories,
                    COALESCE(k.nodes_created, 0) as knowledge_nodes_created,
                    COALESCE(l.avg_performance, 0) as performance_score
                FROM daily_data d
                FULL OUTER JOIN knowledge_data k ON d.date = k.date
                FULL OUTER JOIN learning_data l ON d.date = l.date
                ORDER BY date
            """), {"cutoff": cutoff}).fetchall()
            
            timeline = []
            cumulative_memories = 0
            cumulative_knowledge = 0
            
            for row in daily_progress:
                cumulative_memories += row.memories_created
                cumulative_knowledge += row.knowledge_nodes_created
                
                timeline.append({
                    "date": row.date.isoformat(),
                    "memories_created": row.memories_created,
                    "cumulative_memories": cumulative_memories,
                    "long_term_ratio": round(row.long_term_memories / row.memories_created * 100, 1) if row.memories_created > 0 else 0,
                    "knowledge_nodes": row.knowledge_nodes_created,
                    "cumulative_knowledge": cumulative_knowledge,
                    "avg_importance": round(row.avg_importance, 2),
                    "performance_score": round(row.performance_score, 1)
                })
            
            # Calculate growth rate
            if len(timeline) >= 2:
                recent_growth = timeline[-1]["cumulative_memories"] - timeline[-8]["cumulative_memories"] if len(timeline) > 7 else timeline[0]["cumulative_memories"]
                previous_growth = timeline[-8]["cumulative_memories"] - timeline[-15]["cumulative_memories"] if len(timeline) > 14 else 0
                
                growth_trend = "increasing" if recent_growth > previous_growth else "decreasing" if recent_growth < previous_growth else "stable"
            else:
                growth_trend = "insufficient_data"
            
            return {
                "timeline": timeline,
                "total_memories": cumulative_memories,
                "total_knowledge_nodes": cumulative_knowledge,
                "growth_trend": growth_trend,
                "days_analyzed": days
            }
            
        except Exception as e:
            logger.error(f"Error getting learning progress: {e}")
            return {"timeline": [], "error": str(e)}
    
    async def publish_metric_update(self, metric_type: str, data: Dict[str, Any]):
        """Publish metric update to Redis for real-time dashboard updates"""
        if not self.redis:
            return
        
        try:
            message = json.dumps({
                "type": f"metric_update_{metric_type}",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            
            await self.redis.publish("dashboard_updates", message)
            logger.debug(f"Published {metric_type} update to Redis")
            
        except Exception as e:
            logger.error(f"Error publishing metric update: {e}")

