"""
Learning System Updater
=======================

Closes the feedback loop by updating agent performance, learning from execution patterns,
and improving future workflow executions.

Learns from:
- Agent execution success/failure rates
- Context quality effectiveness
- Task decomposition accuracy
- Quality scores and recommendations

Week 4 - PRD-10 Implementation (FINAL COMPONENT)
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from database.models import Agent, WorkflowExecution, Workflow
from core.agent_execution_manager import SubtaskExecution, SubtaskStatus
from core.result_aggregator import AggregatedResults

logger = logging.getLogger(__name__)


class LearningSystemUpdater:
    """
    Updates the learning system based on workflow execution results.
    
    Learning Areas:
    1. Agent Performance - Success rates, token efficiency, execution times
    2. Workflow Patterns - Successful decomposition strategies
    3. Context Effectiveness - Which RAG sources lead to better results
    4. Quality Improvements - Identify and reinforce high-quality patterns
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
        
        # Learning parameters
        self.learning_rate = 0.1  # How quickly to adapt (0.1 = 10% weight to new data)
        self.min_executions_for_update = 3  # Minimum executions before updating patterns
    
    async def update_from_execution(
        self,
        workflow_id: int,
        execution_id: int,
        aggregated_results: AggregatedResults,
        subtask_executions: Dict[str, SubtaskExecution],
        decomposition_metadata: Dict[str, Any],
        context_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update learning system based on a single workflow execution.
        
        Args:
            workflow_id: Workflow ID
            execution_id: Execution ID
            aggregated_results: Quality scores and metrics
            subtask_executions: Individual subtask results
            decomposition_metadata: Task decomposition info
            context_metadata: Context engineering info
            
        Returns:
            Summary of learning updates applied
        """
        self.logger.info(f"🎓 Updating learning system from workflow {workflow_id} execution {execution_id}")
        
        updates = {
            "agent_performance_updates": {},
            "workflow_pattern_updates": {},
            "context_effectiveness_updates": {},
            "total_updates": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Update agent performance metrics
        agent_updates = await self._update_agent_performance(
            aggregated_results.agent_performance,
            subtask_executions
        )
        updates["agent_performance_updates"] = agent_updates
        updates["total_updates"] += len(agent_updates)
        
        # 2. Learn workflow patterns
        pattern_updates = await self._learn_workflow_patterns(
            workflow_id,
            aggregated_results.quality_scores.overall,
            decomposition_metadata
        )
        updates["workflow_pattern_updates"] = pattern_updates
        updates["total_updates"] += 1 if pattern_updates["pattern_recorded"] else 0
        
        # 3. Assess context effectiveness
        context_updates = await self._assess_context_effectiveness(
            context_metadata,
            aggregated_results.quality_scores.accuracy
        )
        updates["context_effectiveness_updates"] = context_updates
        
        self.logger.info(
            f"✅ Learning system updated: {updates['total_updates']} total updates, "
            f"{len(agent_updates)} agents improved"
        )
        
        return updates
    
    async def _update_agent_performance(
        self,
        agent_performance: Dict[str, Dict[str, Any]],
        subtask_executions: Dict[str, SubtaskExecution]
    ) -> Dict[str, Dict[str, Any]]:
        """Update agent performance metrics in database"""
        
        updates = {}
        
        for agent_name, perf_data in agent_performance.items():
            try:
                # Find agent by name
                agent = self.db.query(Agent).filter(
                    Agent.name == agent_name
                ).first()
                
                if not agent:
                    self.logger.warning(f"Agent '{agent_name}' not found in database")
                    continue
                
                # Get current metrics
                current_metrics = agent.performance_metrics or {}
                
                # Calculate new metrics with exponential moving average
                new_success_rate = perf_data["success_rate"]
                new_avg_tokens = perf_data["avg_tokens"]
                new_avg_time = perf_data["avg_time_ms"]
                
                # Update with learning rate
                if "success_rate" in current_metrics:
                    updated_success_rate = (
                        (1 - self.learning_rate) * current_metrics["success_rate"] +
                        self.learning_rate * new_success_rate
                    )
                else:
                    updated_success_rate = new_success_rate
                
                if "avg_tokens_per_task" in current_metrics:
                    updated_avg_tokens = (
                        (1 - self.learning_rate) * current_metrics["avg_tokens_per_task"] +
                        self.learning_rate * new_avg_tokens
                    )
                else:
                    updated_avg_tokens = new_avg_tokens
                
                if "avg_execution_time_ms" in current_metrics:
                    updated_avg_time = (
                        (1 - self.learning_rate) * current_metrics["avg_execution_time_ms"] +
                        self.learning_rate * new_avg_time
                    )
                else:
                    updated_avg_time = new_avg_time
                
                # Update agent metrics
                agent.performance_metrics = {
                    **current_metrics,
                    "success_rate": updated_success_rate,
                    "avg_tokens_per_task": updated_avg_tokens,
                    "avg_execution_time_ms": updated_avg_time,
                    "total_tasks_executed": current_metrics.get("total_tasks_executed", 0) + perf_data["total_tasks"],
                    "last_updated": datetime.now().isoformat(),
                    "learning_version": current_metrics.get("learning_version", 0) + 1
                }
                
                # Update quality scores if available
                agent.quality_score = updated_success_rate
                agent.reliability = updated_success_rate
                
                self.db.commit()
                
                updates[agent_name] = {
                    "previous_success_rate": current_metrics.get("success_rate", 0),
                    "new_success_rate": updated_success_rate,
                    "improvement": updated_success_rate - current_metrics.get("success_rate", 0),
                    "total_tasks": perf_data["total_tasks"]
                }
                
                self.logger.info(
                    f"✅ Updated agent '{agent_name}': "
                    f"success rate {updated_success_rate:.0%}, "
                    f"avg tokens {updated_avg_tokens:.0f}, "
                    f"avg time {updated_avg_time:.0f}ms"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to update agent '{agent_name}': {e}")
                continue
        
        return updates
    
    async def _learn_workflow_patterns(
        self,
        workflow_id: int,
        quality_score: float,
        decomposition_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn from successful workflow execution patterns"""
        
        try:
            workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
            
            if not workflow:
                return {"pattern_recorded": False, "reason": "Workflow not found"}
            
            # Update workflow success metrics
            workflow_def = workflow.workflow_definition or {}
            
            # Track execution history
            if "execution_history" not in workflow_def:
                workflow_def["execution_history"] = []
            
            # Add this execution to history
            execution_record = {
                "timestamp": datetime.now().isoformat(),
                "quality_score": quality_score,
                "llm_model": decomposition_metadata.get("llm_model"),
                "execution_strategy": decomposition_metadata.get("execution_strategy"),
                "num_subtasks": len(decomposition_metadata.get("subtasks", [])),
                "complexity": decomposition_metadata.get("complexity_assessment")
            }
            
            workflow_def["execution_history"].append(execution_record)
            
            # Keep only last 10 executions
            workflow_def["execution_history"] = workflow_def["execution_history"][-10:]
            
            # Calculate rolling average quality
            recent_scores = [
                ex["quality_score"]
                for ex in workflow_def["execution_history"]
                if "quality_score" in ex
            ]
            
            if recent_scores:
                workflow_def["avg_quality_score"] = sum(recent_scores) / len(recent_scores)
                workflow_def["total_executions"] = workflow_def.get("total_executions", 0) + 1
            
            # Learn patterns from high-quality executions
            if quality_score >= 0.8:
                if "successful_patterns" not in workflow_def:
                    workflow_def["successful_patterns"] = []
                
                pattern = {
                    "quality_score": quality_score,
                    "llm_model": decomposition_metadata.get("llm_model"),
                    "execution_strategy": decomposition_metadata.get("execution_strategy"),
                    "timestamp": datetime.now().isoformat()
                }
                
                workflow_def["successful_patterns"].append(pattern)
                workflow_def["successful_patterns"] = workflow_def["successful_patterns"][-5:]  # Keep last 5
            
            workflow.workflow_definition = workflow_def
            self.db.commit()
            
            return {
                "pattern_recorded": True,
                "quality_score": quality_score,
                "avg_quality": workflow_def.get("avg_quality_score", 0),
                "total_executions": workflow_def.get("total_executions", 1),
                "successful_patterns_count": len(workflow_def.get("successful_patterns", []))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to learn workflow patterns: {e}")
            return {"pattern_recorded": False, "reason": str(e)}
    
    async def _assess_context_effectiveness(
        self,
        context_metadata: Dict[str, Any],
        accuracy_score: float
    ) -> Dict[str, Any]:
        """Assess effectiveness of context engineering"""
        
        if not context_metadata or not context_metadata.get("is_real"):
            return {"assessed": False, "reason": "No context metadata"}
        
        summary = context_metadata.get("summary", {})
        
        # Simple assessment: correlate context quality with accuracy
        context_coverage = summary.get("context_coverage", 0)
        avg_context_quality = summary.get("avg_context_quality", 0)
        total_sources = summary.get("total_sources_retrieved", 0)
        
        # Determine if context was effective
        is_effective = accuracy_score >= 0.7 and total_sources > 0
        
        assessment = {
            "assessed": True,
            "is_effective": is_effective,
            "accuracy_score": accuracy_score,
            "context_coverage": context_coverage,
            "context_quality": avg_context_quality,
            "sources_used": total_sources,
            "recommendation": self._get_context_recommendation(
                is_effective,
                context_coverage,
                avg_context_quality,
                total_sources
            )
        }
        
        return assessment
    
    def _get_context_recommendation(
        self,
        is_effective: bool,
        coverage: float,
        quality: float,
        sources: int
    ) -> str:
        """Generate recommendation for context improvement"""
        
        if is_effective:
            return "✅ Context engineering is working well - maintain current settings"
        
        if coverage < 0.7:
            return "⚠️ Low context coverage - increase max_chunks in RAG settings"
        
        if quality < 0.6:
            return "⚠️ Low context quality - improve document quality or adjust similarity threshold"
        
        if sources < 2:
            return "⚠️ Too few sources - expand knowledge base or adjust search parameters"
        
        return "⚠️ Context effectiveness below target - review RAG configuration"
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning system state"""
        
        try:
            # Get agent performance stats
            agents = self.db.query(Agent).filter(Agent.status == "active").all()
            
            agent_stats = {
                "total_agents": len(agents),
                "agents_with_metrics": sum(1 for a in agents if a.performance_metrics),
                "avg_success_rate": sum(
                    a.performance_metrics.get("success_rate", 0)
                    for a in agents if a.performance_metrics
                ) / max(len([a for a in agents if a.performance_metrics]), 1),
                "total_tasks_executed": sum(
                    a.performance_metrics.get("total_tasks_executed", 0)
                    for a in agents if a.performance_metrics
                )
            }
            
            # Get workflow execution stats
            total_workflows = self.db.query(Workflow).count()
            workflows_with_history = self.db.query(Workflow).filter(
                Workflow.workflow_definition.isnot(None)
            ).count()
            
            # Get recent execution trends
            week_ago = datetime.now() - timedelta(days=7)
            recent_executions = self.db.query(WorkflowExecution).filter(
                WorkflowExecution.started_at >= week_ago
            ).count()
            
            completed_executions = self.db.query(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.started_at >= week_ago,
                    WorkflowExecution.status == "completed"
                )
            ).count()
            
            return {
                "agent_performance": agent_stats,
                "workflow_patterns": {
                    "total_workflows": total_workflows,
                    "workflows_with_history": workflows_with_history,
                    "learning_coverage": workflows_with_history / max(total_workflows, 1)
                },
                "recent_activity": {
                    "executions_last_7_days": recent_executions,
                    "completed_last_7_days": completed_executions,
                    "success_rate_7_days": completed_executions / max(recent_executions, 1) if recent_executions > 0 else 0
                },
                "learning_system_health": "healthy" if agent_stats["agents_with_metrics"] > 0 else "initializing",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get learning summary: {e}")
            return {
                "error": str(e),
                "learning_system_health": "error",
                "timestamp": datetime.now().isoformat()
            }

