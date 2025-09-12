
"""
Context Constraints Management Module
Handles token limits, resource constraints, and usage optimization
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
import asyncio
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ContextConstraints:
    """Context constraints configuration"""
    max_tokens: int = 128000
    safety_buffer: float = 0.15
    max_memory_mb: int = 512
    max_session_duration: int = 3600  # seconds
    max_concurrent_sessions: int = 100

@dataclass
class UsageMetrics:
    """Usage metrics for monitoring"""
    token_usage: int
    memory_usage: float
    session_duration: int
    utilization_rate: float
    timestamp: datetime

class ContextConstraintsManager:
    """Advanced context constraints manager with intelligent resource allocation"""
    
    def __init__(self, constraints: Optional[ContextConstraints] = None):
        self.constraints = constraints or ContextConstraints()
        self.effective_capacity = int(self.constraints.max_tokens * (1 - self.constraints.safety_buffer))
        self.usage_history: deque = deque(maxlen=1000)
        self.session_usage: Dict[str, UsageMetrics] = {}
        self.active_sessions: Dict[str, datetime] = {}
        
        logger.info(f"Initialized ContextConstraintsManager with capacity: {self.effective_capacity} tokens")
    
    async def validate_context_addition(self, session_id: str, context_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate if new context can be added without violating constraints"""
        try:
            # Calculate current usage
            current_usage = await self.calculate_token_usage(session_id, context_data)
            memory_usage = await self.calculate_memory_usage(session_id)
            
            # Check token limits
            if current_usage > self.effective_capacity:
                return False, f"Context exceeds token limit: {current_usage}/{self.effective_capacity}"
            
            # Check memory limits
            if memory_usage > self.constraints.max_memory_mb:
                return False, f"Memory usage exceeds limit: {memory_usage:.2f}/{self.constraints.max_memory_mb}MB"
            
            # Check session duration
            if session_id in self.active_sessions:
                duration = (datetime.utcnow() - self.active_sessions[session_id]).total_seconds()
                if duration > self.constraints.max_session_duration:
                    return False, f"Session duration exceeds limit: {duration}/{self.constraints.max_session_duration}s"
            
            # Check concurrent sessions
            if len(self.active_sessions) >= self.constraints.max_concurrent_sessions:
                return False, f"Too many concurrent sessions: {len(self.active_sessions)}/{self.constraints.max_concurrent_sessions}"
            
            return True, "Context validation successful"
            
        except Exception as e:
            logger.error(f"Error validating context: {e}")
            return False, f"Validation error: {str(e)}"
    
    async def calculate_token_usage(self, session_id: str, context_data: Dict[str, Any]) -> int:
        """Calculate approximate token usage for context data"""
        try:
            # Convert to string and estimate tokens (rough approximation: 1 token ≈ 4 characters)
            context_str = json.dumps(context_data, default=str)
            estimated_tokens = len(context_str) // 4
            
            # Add existing session usage
            if session_id in self.session_usage:
                estimated_tokens += self.session_usage[session_id].token_usage
            
            return estimated_tokens
            
        except Exception as e:
            logger.error(f"Error calculating token usage: {e}")
            return 0
    
    async def calculate_memory_usage(self, session_id: str) -> float:
        """Calculate current memory usage in MB"""
        try:
            import sys
            
            # Calculate size of session data
            session_size = 0
            if session_id in self.session_usage:
                session_size = sys.getsizeof(self.session_usage[session_id])
            
            # Add usage history size
            history_size = sys.getsizeof(self.usage_history)
            
            # Convert to MB
            total_size_mb = (session_size + history_size) / (1024 * 1024)
            
            return total_size_mb
            
        except Exception as e:
            logger.error(f"Error calculating memory usage: {e}")
            return 0.0
    
    async def update_usage_metrics(self, session_id: str, context_data: Dict[str, Any]):
        """Update usage metrics for session"""
        try:
            token_usage = await self.calculate_token_usage(session_id, context_data)
            memory_usage = await self.calculate_memory_usage(session_id)
            
            # Calculate session duration
            session_start = self.active_sessions.get(session_id, datetime.utcnow())
            duration = (datetime.utcnow() - session_start).total_seconds()
            
            # Calculate utilization rate
            utilization_rate = (token_usage / self.effective_capacity) * 100
            
            # Create metrics
            metrics = UsageMetrics(
                token_usage=token_usage,
                memory_usage=memory_usage,
                session_duration=int(duration),
                utilization_rate=utilization_rate,
                timestamp=datetime.utcnow()
            )
            
            # Update session usage
            self.session_usage[session_id] = metrics
            self.usage_history.append(metrics)
            
            # Track active session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = datetime.utcnow()
            
            logger.debug(f"Updated usage metrics for session {session_id}: {utilization_rate:.1f}% utilization")
            
        except Exception as e:
            logger.error(f"Error updating usage metrics: {e}")
    
    async def get_usage_analysis(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive usage analysis for session"""
        try:
            if session_id not in self.session_usage:
                return {
                    "session_id": session_id,
                    "status": "not_found",
                    "token_usage": 0,
                    "utilization_rate": 0,
                    "remaining_capacity": self.effective_capacity,
                    "memory_usage": 0,
                    "session_duration": 0
                }
            
            metrics = self.session_usage[session_id]
            
            return {
                "session_id": session_id,
                "status": "active",
                "token_usage": metrics.token_usage,
                "utilization_rate": metrics.utilization_rate,
                "remaining_capacity": self.effective_capacity - metrics.token_usage,
                "memory_usage": metrics.memory_usage,
                "session_duration": metrics.session_duration,
                "efficiency_score": self._calculate_efficiency_score(metrics),
                "recommendations": self._generate_recommendations(metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting usage analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_efficiency_score(self, metrics: UsageMetrics) -> float:
        """Calculate efficiency score based on resource utilization"""
        try:
            # Balance between utilization and resource conservation
            utilization_score = min(metrics.utilization_rate / 100, 1.0)
            memory_score = 1.0 - min(metrics.memory_usage / self.constraints.max_memory_mb, 1.0)
            duration_score = 1.0 - min(metrics.session_duration / self.constraints.max_session_duration, 1.0)
            
            # Weighted average
            efficiency = (utilization_score * 0.4 + memory_score * 0.3 + duration_score * 0.3)
            return round(efficiency * 100, 2)
            
        except Exception as e:
            logger.error(f"Error calculating efficiency score: {e}")
            return 0.0
    
    def _generate_recommendations(self, metrics: UsageMetrics) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        try:
            if metrics.utilization_rate > 80:
                recommendations.append("High token usage detected. Consider context compression.")
            
            if metrics.memory_usage > self.constraints.max_memory_mb * 0.8:
                recommendations.append("High memory usage. Consider session cleanup.")
            
            if metrics.session_duration > self.constraints.max_session_duration * 0.8:
                recommendations.append("Long session detected. Consider session splitting.")
            
            if metrics.utilization_rate < 30:
                recommendations.append("Low utilization. Consider increasing batch size.")
            
            if not recommendations:
                recommendations.append("Resource usage is optimal.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations.")
        
        return recommendations
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions and optimize resource usage"""
        try:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            for session_id, start_time in self.active_sessions.items():
                if (current_time - start_time).total_seconds() > self.constraints.max_session_duration:
                    expired_sessions.append(session_id)
            
            # Clean up expired sessions
            for session_id in expired_sessions:
                await self.end_session(session_id)
            
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            return 0
    
    async def end_session(self, session_id: str):
        """End a session and clean up resources"""
        try:
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Archive usage metrics (keep for analysis)
            if session_id in self.session_usage:
                final_metrics = self.session_usage[session_id]
                self.usage_history.append(final_metrics)
                del self.session_usage[session_id]
            
            logger.debug(f"Ended session {session_id}")
            
        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health and performance metrics"""
        try:
            current_time = datetime.utcnow()
            
            # Calculate average metrics from recent history
            recent_metrics = [m for m in self.usage_history if (current_time - m.timestamp).total_seconds() < 3600]
            
            if recent_metrics:
                avg_utilization = sum(m.utilization_rate for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
                avg_duration = sum(m.session_duration for m in recent_metrics) / len(recent_metrics)
            else:
                avg_utilization = avg_memory = avg_duration = 0
            
            return {
                "system_status": "healthy" if avg_utilization < 90 else "warning",
                "active_sessions": len(self.active_sessions),
                "total_capacity": self.constraints.max_tokens,
                "effective_capacity": self.effective_capacity,
                "average_utilization": round(avg_utilization, 2),
                "average_memory_usage": round(avg_memory, 2),
                "average_session_duration": round(avg_duration, 2),
                "peak_utilization": max((m.utilization_rate for m in recent_metrics), default=0),
                "performance_trend": self._analyze_performance_trend(recent_metrics),
                "last_cleanup": current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"error": str(e)}
    
    def _analyze_performance_trend(self, metrics: List[UsageMetrics]) -> str:
        """Analyze performance trend over recent metrics"""
        if len(metrics) < 10:
            return "insufficient_data"
        
        try:
            # Compare first half vs second half
            mid_point = len(metrics) // 2
            first_half_avg = sum(m.utilization_rate for m in metrics[:mid_point]) / mid_point
            second_half_avg = sum(m.utilization_rate for m in metrics[mid_point:]) / (len(metrics) - mid_point)
            
            diff = second_half_avg - first_half_avg
            
            if diff > 10:
                return "increasing"
            elif diff < -10:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "unknown"
