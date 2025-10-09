"""
Orchestration Tracking Integration
===================================
Integrates comprehensive tracking into the workflow execution pipeline.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from services.orchestration_tracker import orchestration_tracker

logger = logging.getLogger(__name__)

class OrchestrationIntegration:
    """Integrates tracking throughout the workflow pipeline"""
    
    @staticmethod
    async def track_workflow_start(workflow_id: int, execution_id: int, task_description: str):
        """Track workflow start"""
        await orchestration_tracker.start_tracking(workflow_id, execution_id)
        await orchestration_tracker.track_event("workflow_started", {
            "task_description": task_description[:200],
            "timestamp": datetime.now().isoformat()
        })
    
    @staticmethod
    async def track_decomposition(steps: List[Dict], complexity: str):
        """Track task decomposition"""
        await orchestration_tracker.track_event("decomposition_completed", {
            "steps_count": len(steps),
            "complexity": complexity,
            "timestamp": datetime.now().isoformat()
        })
    
    @staticmethod
    async def track_memory_retrieval(memory_data: Dict[str, Any]):
        """Track memory retrieval operations"""
        for agent_id, memories in memory_data.get('agent_memories', {}).items():
            total_items = 0
            for memory_type in ['working_memory', 'short_term', 'long_term']:
                items = memories.get(memory_type, [])
                total_items += len(items)
            
            if total_items > 0:
                await orchestration_tracker.track_memory_operation(
                    operation_type="retrieve",
                    memory_type="hierarchical",
                    agent_id=int(agent_id),
                    items_count=total_items,
                    relevance_score=0.85,
                    hit_rate=min(1.0, total_items / 20)
                )
    
    @staticmethod
    async def track_agent_selections(selected_agents: Dict[str, Any], selection_time: float):
        """Track agent selection decisions"""
        for task_id, agent_info in selected_agents.items():
            await orchestration_tracker.track_agent_selection(
                task_id=task_id,
                selected_agent_id=agent_info.get('agent_id', 0),
                selection_score=agent_info.get('match_score', 0.8),
                selection_time=selection_time / len(selected_agents) if selected_agents else 0,
                alternative_agents=[]
            )
    
    @staticmethod
    async def track_task_execution(subtask_results: Dict[str, Any]):
        """Track task execution results"""
        for task_id, result in subtask_results.items():
            if hasattr(result, '__dict__'):
                await orchestration_tracker.track_task_completion(
                    task_id=task_id,
                    agent_id=getattr(result, 'agent_id', 0),
                    execution_time=getattr(result, 'execution_time', 0),
                    tokens_used=result.token_usage.get('total_tokens', 0) if hasattr(result, 'token_usage') and result.token_usage else 0,
                    success=getattr(result, 'status', '') == 'completed',
                    quality_score=getattr(result, 'quality_score', 0.8),
                    cost=result.token_usage.get('estimated_cost', 0) if hasattr(result, 'token_usage') and result.token_usage else 0
                )
    
    @staticmethod
    async def track_communication(from_agent: str, to_agent: str, message_type: str, priority: str = "normal"):
        """Track inter-agent communication"""
        await orchestration_tracker.track_communication(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            priority=priority,
            response_time=None
        )
    
    @staticmethod
    async def track_learning_update(learning_updates: Dict[str, Any]):
        """Track learning system updates"""
        if learning_updates.get('workflow_pattern_updates', {}).get('pattern_recorded'):
            await orchestration_tracker.track_learning_application(
                pattern_id="workflow_pattern",
                pattern_type="workflow_optimization",
                confidence=0.8,
                expected_improvement=0.1,
                actual_improvement=None
            )
        
        for agent_id, updates in learning_updates.get('agent_performance_updates', {}).items():
            if isinstance(updates, dict) and 'success_rate' in updates:
                await orchestration_tracker.track_learning_application(
                    pattern_id=f"agent_{agent_id}_performance",
                    pattern_type="agent_optimization",
                    confidence=updates.get('confidence', 0.7),
                    expected_improvement=0.05,
                    actual_improvement=updates.get('improvement', None)
                )
    
    @staticmethod
    async def complete_tracking() -> Dict[str, Any]:
        """Complete tracking and get summary"""
        return await orchestration_tracker.complete_tracking()
    
    @staticmethod
    async def get_performance_trends(window_size: int = 10) -> Dict[str, Any]:
        """Get performance trends analysis"""
        return await orchestration_tracker.get_performance_trends(window_size)
