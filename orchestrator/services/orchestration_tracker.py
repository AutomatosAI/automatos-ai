"""
Orchestration Event Tracker
============================
Captures all orchestration events in real-time for learning, benchmarking,
and performance analysis. Proves the system gets smarter over time.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import statistics
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Tracks performance improvements over time"""
    workflow_type: str
    execution_number: int
    execution_time: float
    token_usage: int
    cost: float
    success_rate: float
    quality_score: float
    memory_hits: int
    cache_hits: int
    agent_reuse_rate: float
    learning_applied: bool
    timestamp: datetime

@dataclass
class LearningMetric:
    """Tracks learning and intelligence improvements"""
    pattern_id: str
    pattern_type: str
    occurrences: int
    success_improvements: float
    time_savings: float
    cost_savings: float
    confidence: float
    last_applied: datetime

class OrchestrationTracker:
    """
    Real-time orchestration event tracker that captures everything
    for benchmarking and proving system intelligence growth.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.current_workflow_id = None
        self.current_execution_id = None
        self.event_buffer = []
        self.metrics_cache = {}
        self.performance_history = []
        self.learning_metrics = {}
        
        # Benchmarking baselines
        self.baseline_metrics = {
            "avg_execution_time": None,
            "avg_token_usage": None,
            "avg_cost": None,
            "avg_success_rate": None,
            "baseline_established": False
        }
        
        logger.info("OrchestrationTracker initialized")
    
    async def start_tracking(self, workflow_id: int, execution_id: int):
        """Start tracking a new workflow execution"""
        self.current_workflow_id = workflow_id
        self.current_execution_id = execution_id
        self.event_buffer = []
        
        logger.info(f"📊 Started tracking workflow {workflow_id}, execution {execution_id}")
        
        # Record start event
        await self.track_event("execution_started", {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat()
        })
    
    async def track_event(self, event_type: str, event_data: Dict[str, Any]):
        """Track any orchestration event"""
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat(),
            "workflow_id": self.current_workflow_id,
            "execution_id": self.current_execution_id
        }
        
        self.event_buffer.append(event)
        
        # Send to API for persistence
        await self._store_event(event_type, event_data)
        
        # Update real-time metrics
        await self._update_metrics(event_type, event_data)
    
    async def track_memory_operation(
        self,
        operation_type: str,
        memory_type: str,
        agent_id: Optional[int] = None,
        items_count: int = 0,
        relevance_score: float = 0.0,
        hit_rate: float = 0.0
    ):
        """Track memory system operations with performance metrics"""
        await self.track_event("memory_operation", {
            "operation": operation_type,
            "memory_type": memory_type,
            "agent_id": agent_id,
            "items_count": items_count,
            "relevance_score": relevance_score,
            "hit_rate": hit_rate,
            "cache_benefit": hit_rate > 0.5  # Track when cache is beneficial
        })
        
        # Update memory efficiency metrics
        if "memory_efficiency" not in self.metrics_cache:
            self.metrics_cache["memory_efficiency"] = []
        
        self.metrics_cache["memory_efficiency"].append({
            "hit_rate": hit_rate,
            "relevance": relevance_score,
            "timestamp": datetime.now().isoformat()
        })
    
    async def track_agent_selection(
        self,
        task_id: str,
        selected_agent_id: int,
        selection_score: float,
        selection_time: float,
        alternative_agents: List[Dict[str, Any]]
    ):
        """Track agent selection decisions and efficiency"""
        await self.track_event("agent_selection", {
            "task_id": task_id,
            "selected_agent_id": selected_agent_id,
            "selection_score": selection_score,
            "selection_time": selection_time,
            "alternatives_considered": len(alternative_agents),
            "optimal_selection": selection_score > 0.8,
            "selection_efficiency": 1.0 / selection_time if selection_time > 0 else 0
        })
        
        # Track agent reuse patterns
        if "agent_reuse" not in self.metrics_cache:
            self.metrics_cache["agent_reuse"] = {}
        
        if selected_agent_id not in self.metrics_cache["agent_reuse"]:
            self.metrics_cache["agent_reuse"][selected_agent_id] = 0
        
        self.metrics_cache["agent_reuse"][selected_agent_id] += 1
    
    async def track_communication(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        priority: str,
        response_time: Optional[float] = None
    ):
        """Track inter-agent communication efficiency"""
        await self.track_event("communication", {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "type": message_type,
            "priority": priority,
            "response_time": response_time,
            "efficient_routing": response_time < 100 if response_time else False
        })
        
        # Track communication patterns
        if "communication_patterns" not in self.metrics_cache:
            self.metrics_cache["communication_patterns"] = []
        
        self.metrics_cache["communication_patterns"].append({
            "pair": f"{from_agent}->{to_agent}",
            "type": message_type,
            "response_time": response_time
        })
    
    async def track_learning_application(
        self,
        pattern_id: str,
        pattern_type: str,
        confidence: float,
        expected_improvement: float,
        actual_improvement: Optional[float] = None
    ):
        """Track when learning patterns are applied and their effectiveness"""
        await self.track_event("learning_applied", {
            "pattern_id": pattern_id,
            "pattern_type": pattern_type,
            "confidence": confidence,
            "expected_improvement": expected_improvement,
            "actual_improvement": actual_improvement,
            "learning_effective": actual_improvement > expected_improvement if actual_improvement else None
        })
        
        # Update learning metrics
        if pattern_id not in self.learning_metrics:
            self.learning_metrics[pattern_id] = LearningMetric(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                occurrences=0,
                success_improvements=0,
                time_savings=0,
                cost_savings=0,
                confidence=confidence,
                last_applied=datetime.now()
            )
        
        metric = self.learning_metrics[pattern_id]
        metric.occurrences += 1
        metric.confidence = (metric.confidence + confidence) / 2
        metric.last_applied = datetime.now()
        
        if actual_improvement:
            metric.success_improvements += actual_improvement
    
    async def track_task_completion(
        self,
        task_id: str,
        agent_id: int,
        execution_time: float,
        tokens_used: int,
        success: bool,
        quality_score: float,
        cost: float
    ):
        """Track task completion with performance metrics"""
        await self.track_event("task_completed", {
            "task_id": task_id,
            "agent_id": agent_id,
            "execution_time": execution_time,
            "tokens_used": tokens_used,
            "success": success,
            "quality_score": quality_score,
            "cost": cost,
            "efficiency_score": quality_score / cost if cost > 0 else 0
        })
        
        # Update performance tracking
        if "task_performance" not in self.metrics_cache:
            self.metrics_cache["task_performance"] = []
        
        self.metrics_cache["task_performance"].append({
            "execution_time": execution_time,
            "tokens": tokens_used,
            "cost": cost,
            "quality": quality_score,
            "success": success
        })
    
    async def complete_tracking(self) -> Dict[str, Any]:
        """Complete tracking and generate comprehensive metrics"""
        if not self.current_workflow_id:
            return {}
        
        # Calculate aggregate metrics
        metrics = await self._calculate_final_metrics()
        
        # Compare with baseline
        improvements = await self._calculate_improvements(metrics)
        
        # Generate learning insights
        insights = await self._generate_insights(metrics, improvements)
        
        # Store complete analysis
        await self.track_event("execution_completed", {
            "metrics": metrics,
            "improvements": improvements,
            "insights": insights,
            "total_events": len(self.event_buffer)
        })
        
        # Update baseline if this is better
        await self._update_baseline(metrics)
        
        # Add to performance history
        self.performance_history.append(PerformanceMetric(
            workflow_type=f"workflow_{self.current_workflow_id}",
            execution_number=len(self.performance_history) + 1,
            execution_time=metrics.get("total_execution_time", 0),
            token_usage=metrics.get("total_tokens", 0),
            cost=metrics.get("total_cost", 0),
            success_rate=metrics.get("success_rate", 0),
            quality_score=metrics.get("avg_quality", 0),
            memory_hits=metrics.get("memory_hit_rate", 0),
            cache_hits=metrics.get("cache_hits", 0),
            agent_reuse_rate=metrics.get("agent_reuse_rate", 0),
            learning_applied=len(self.learning_metrics) > 0,
            timestamp=datetime.now()
        ))
        
        logger.info(f"📈 Tracking complete: {improvements.get('overall_improvement', 0):.1f}% improvement")
        
        return {
            "workflow_id": self.current_workflow_id,
            "execution_id": self.current_execution_id,
            "metrics": metrics,
            "improvements": improvements,
            "insights": insights,
            "events_captured": len(self.event_buffer)
        }
    
    async def _calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive final metrics"""
        metrics = {
            "total_events": len(self.event_buffer),
            "start_time": self.event_buffer[0]["timestamp"] if self.event_buffer else None,
            "end_time": self.event_buffer[-1]["timestamp"] if self.event_buffer else None
        }
        
        # Task performance metrics
        if "task_performance" in self.metrics_cache:
            tasks = self.metrics_cache["task_performance"]
            metrics["total_tasks"] = len(tasks)
            metrics["successful_tasks"] = sum(1 for t in tasks if t["success"])
            metrics["success_rate"] = metrics["successful_tasks"] / metrics["total_tasks"] if metrics["total_tasks"] > 0 else 0
            metrics["total_execution_time"] = sum(t["execution_time"] for t in tasks)
            metrics["total_tokens"] = sum(t["tokens"] for t in tasks)
            metrics["total_cost"] = sum(t["cost"] for t in tasks)
            metrics["avg_quality"] = statistics.mean([t["quality"] for t in tasks]) if tasks else 0
            metrics["avg_task_time"] = statistics.mean([t["execution_time"] for t in tasks]) if tasks else 0
        
        # Memory efficiency metrics
        if "memory_efficiency" in self.metrics_cache:
            mem_data = self.metrics_cache["memory_efficiency"]
            metrics["memory_operations"] = len(mem_data)
            metrics["memory_hit_rate"] = statistics.mean([m["hit_rate"] for m in mem_data]) if mem_data else 0
            metrics["avg_relevance"] = statistics.mean([m["relevance"] for m in mem_data]) if mem_data else 0
        
        # Agent reuse metrics
        if "agent_reuse" in self.metrics_cache:
            reuse_data = self.metrics_cache["agent_reuse"]
            total_selections = sum(reuse_data.values())
            unique_agents = len(reuse_data)
            metrics["agent_reuse_rate"] = 1 - (unique_agents / total_selections) if total_selections > 0 else 0
            metrics["most_used_agent"] = max(reuse_data, key=reuse_data.get) if reuse_data else None
        
        # Communication efficiency
        if "communication_patterns" in self.metrics_cache:
            comm_data = self.metrics_cache["communication_patterns"]
            response_times = [c["response_time"] for c in comm_data if c["response_time"]]
            metrics["total_communications"] = len(comm_data)
            metrics["avg_response_time"] = statistics.mean(response_times) if response_times else 0
            metrics["communication_efficiency"] = sum(1 for rt in response_times if rt < 100) / len(response_times) if response_times else 0
        
        # Learning application metrics
        metrics["patterns_applied"] = len(self.learning_metrics)
        metrics["learning_confidence"] = statistics.mean([m.confidence for m in self.learning_metrics.values()]) if self.learning_metrics else 0
        
        return metrics
    
    async def _calculate_improvements(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvements compared to baseline"""
        improvements = {}
        
        if not self.baseline_metrics["baseline_established"]:
            improvements["status"] = "baseline_establishing"
            improvements["overall_improvement"] = 0
            return improvements
        
        # Calculate improvements for each metric
        if self.baseline_metrics["avg_execution_time"] and current_metrics.get("total_execution_time"):
            time_improvement = (self.baseline_metrics["avg_execution_time"] - current_metrics["total_execution_time"]) / self.baseline_metrics["avg_execution_time"]
            improvements["time_improvement"] = time_improvement * 100
        
        if self.baseline_metrics["avg_cost"] and current_metrics.get("total_cost"):
            cost_improvement = (self.baseline_metrics["avg_cost"] - current_metrics["total_cost"]) / self.baseline_metrics["avg_cost"]
            improvements["cost_savings"] = cost_improvement * 100
        
        if self.baseline_metrics["avg_token_usage"] and current_metrics.get("total_tokens"):
            token_improvement = (self.baseline_metrics["avg_token_usage"] - current_metrics["total_tokens"]) / self.baseline_metrics["avg_token_usage"]
            improvements["token_efficiency"] = token_improvement * 100
        
        if self.baseline_metrics["avg_success_rate"] and current_metrics.get("success_rate"):
            success_improvement = current_metrics["success_rate"] - self.baseline_metrics["avg_success_rate"]
            improvements["success_improvement"] = success_improvement * 100
        
        # Calculate overall improvement score
        improvement_values = [v for k, v in improvements.items() if k != "status" and isinstance(v, (int, float))]
        improvements["overall_improvement"] = statistics.mean(improvement_values) if improvement_values else 0
        
        # Determine status
        if improvements["overall_improvement"] > 10:
            improvements["status"] = "significant_improvement"
        elif improvements["overall_improvement"] > 0:
            improvements["status"] = "improving"
        elif improvements["overall_improvement"] < -10:
            improvements["status"] = "degrading"
        else:
            improvements["status"] = "stable"
        
        return improvements
    
    async def _generate_insights(self, metrics: Dict[str, Any], improvements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable insights from metrics"""
        insights = []
        
        # Time efficiency insight
        if improvements.get("time_improvement", 0) > 10:
            insights.append({
                "type": "efficiency",
                "title": "Significant Time Savings",
                "description": f"Execution time improved by {improvements['time_improvement']:.1f}% through learning and optimization",
                "impact": "high",
                "evidence": f"Current: {metrics.get('total_execution_time', 0):.1f}s vs Baseline: {self.baseline_metrics.get('avg_execution_time', 0):.1f}s"
            })
        
        # Cost optimization insight
        if improvements.get("cost_savings", 0) > 5:
            insights.append({
                "type": "cost",
                "title": "Cost Reduction Achieved",
                "description": f"Token usage optimized resulting in {improvements['cost_savings']:.1f}% cost savings",
                "impact": "high",
                "evidence": f"Saved ${metrics.get('total_cost', 0) * (improvements['cost_savings']/100):.2f} on this execution"
            })
        
        # Memory efficiency insight
        if metrics.get("memory_hit_rate", 0) > 0.7:
            insights.append({
                "type": "memory",
                "title": "Effective Memory Utilization",
                "description": f"Memory system achieving {metrics['memory_hit_rate']*100:.0f}% hit rate",
                "impact": "medium",
                "evidence": "High cache hit rate reduces redundant processing"
            })
        
        # Agent reuse insight
        if metrics.get("agent_reuse_rate", 0) > 0.5:
            insights.append({
                "type": "collaboration",
                "title": "Efficient Agent Collaboration",
                "description": f"Agent reuse rate of {metrics['agent_reuse_rate']*100:.0f}% shows effective task routing",
                "impact": "medium",
                "evidence": f"Most used agent: {metrics.get('most_used_agent', 'N/A')}"
            })
        
        # Learning effectiveness insight
        if metrics.get("patterns_applied", 0) > 0:
            insights.append({
                "type": "learning",
                "title": "Active Learning System",
                "description": f"Applied {metrics['patterns_applied']} learned patterns with {metrics.get('learning_confidence', 0)*100:.0f}% confidence",
                "impact": "high",
                "evidence": "System is actively learning and improving from past executions"
            })
        
        # Quality improvement insight
        if metrics.get("avg_quality", 0) > 0.8:
            insights.append({
                "type": "quality",
                "title": "High Quality Output",
                "description": f"Achieving {metrics['avg_quality']*100:.0f}% quality score",
                "impact": "high",
                "evidence": f"Success rate: {metrics.get('success_rate', 0)*100:.0f}%"
            })
        
        return insights
    
    async def _update_baseline(self, metrics: Dict[str, Any]):
        """Update baseline metrics if current performance is better"""
        if not self.baseline_metrics["baseline_established"]:
            # Establish initial baseline
            self.baseline_metrics["avg_execution_time"] = metrics.get("total_execution_time", 0)
            self.baseline_metrics["avg_token_usage"] = metrics.get("total_tokens", 0)
            self.baseline_metrics["avg_cost"] = metrics.get("total_cost", 0)
            self.baseline_metrics["avg_success_rate"] = metrics.get("success_rate", 0)
            self.baseline_metrics["baseline_established"] = True
            logger.info("📊 Baseline metrics established")
        else:
            # Update baseline with moving average
            alpha = 0.3  # Learning rate for exponential moving average
            
            if metrics.get("total_execution_time"):
                self.baseline_metrics["avg_execution_time"] = (
                    alpha * metrics["total_execution_time"] + 
                    (1 - alpha) * self.baseline_metrics["avg_execution_time"]
                )
            
            if metrics.get("total_tokens"):
                self.baseline_metrics["avg_token_usage"] = (
                    alpha * metrics["total_tokens"] + 
                    (1 - alpha) * self.baseline_metrics["avg_token_usage"]
                )
            
            if metrics.get("total_cost"):
                self.baseline_metrics["avg_cost"] = (
                    alpha * metrics["total_cost"] + 
                    (1 - alpha) * self.baseline_metrics["avg_cost"]
                )
            
            if metrics.get("success_rate"):
                self.baseline_metrics["avg_success_rate"] = (
                    alpha * metrics["success_rate"] + 
                    (1 - alpha) * self.baseline_metrics["avg_success_rate"]
                )
    
    async def _store_event(self, event_type: str, event_data: Dict[str, Any]):
        """Store event to API for persistence"""
        if not self.current_workflow_id:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/v1/workflows/{self.current_workflow_id}/store-orchestration-event",
                    json={
                        "event_type": event_type,
                        "event_data": event_data
                    }
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to store event: {response.status}")
        except Exception as e:
            logger.error(f"Error storing event: {e}")
    
    async def _update_metrics(self, event_type: str, event_data: Dict[str, Any]):
        """Update real-time metrics based on events"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/v1/workflows/{self.current_workflow_id}/store-orchestration-event",
                    json={
                        "event_type": "metric_update",
                        "event_data": {
                            f"last_{event_type}": datetime.now().isoformat(),
                            f"{event_type}_count": 1
                        }
                    }
                ) as response:
                    pass  # Metrics updated
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def get_performance_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """Get performance trends over recent executions"""
        if len(self.performance_history) < 2:
            return {"status": "insufficient_data"}
        
        recent = self.performance_history[-window_size:]
        
        trends = {
            "execution_time_trend": self._calculate_trend([p.execution_time for p in recent]),
            "cost_trend": self._calculate_trend([p.cost for p in recent]),
            "quality_trend": self._calculate_trend([p.quality_score for p in recent]),
            "success_trend": self._calculate_trend([p.success_rate for p in recent]),
            "learning_adoption": sum(1 for p in recent if p.learning_applied) / len(recent),
            "window_size": len(recent),
            "total_executions": len(self.performance_history)
        }
        
        # Determine overall trend
        positive_trends = sum(1 for k, v in trends.items() if k.endswith("_trend") and v < 0)  # Negative is good for time/cost
        if positive_trends >= 3:
            trends["overall_trend"] = "improving"
        elif positive_trends >= 2:
            trends["overall_trend"] = "stable"
        else:
            trends["overall_trend"] = "needs_attention"
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend coefficient (negative = improving for time/cost metrics)"""
        if len(values) < 2:
            return 0
        
        # Simple linear regression trend
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator

# Global tracker instance
orchestration_tracker = OrchestrationTracker()
