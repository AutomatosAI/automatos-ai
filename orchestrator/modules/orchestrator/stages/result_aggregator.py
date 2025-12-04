"""
Result Aggregator
=================

Aggregates and scores workflow execution results using 5-dimension quality assessment.

Quality Dimensions:
1. Completeness - Did all subtasks complete successfully?
2. Accuracy - Quality of agent responses and context usage
3. Efficiency - Token usage and execution time
4. Reliability - Error rate and retry count
5. Coherence - Consistency across subtask results

Week 3 Part 2 - PRD-10 Implementation
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from modules.agents import SubtaskExecution, SubtaskStatus

# Import probability theory from shared math
try:
    from shared.math import ProbabilityTheory, ConfidenceInterval
    PROBABILITY_AVAILABLE = True
except ImportError:
    PROBABILITY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class QualityScores:
    """5-dimension quality scores"""
    completeness: float  # 0.0 - 1.0
    accuracy: float
    efficiency: float
    reliability: float
    coherence: float
    overall: float  # Weighted average
    
    # PHASE 4B: Add confidence intervals for uncertainty quantification
    overall_confidence_interval: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "efficiency": self.efficiency,
            "reliability": self.reliability,
            "coherence": self.coherence,
            "overall": self.overall
        }


@dataclass
class AggregatedResults:
    """Aggregated workflow execution results"""
    workflow_id: int
    execution_id: int
    quality_scores: QualityScores
    total_subtasks: int
    completed_subtasks: int
    failed_subtasks: int
    total_tokens_used: int
    total_execution_time_ms: int
    avg_context_quality: float
    total_retries: int
    agent_performance: Dict[str, Dict[str, Any]]  # agent_id -> performance metrics
    recommendations: List[str]
    timestamp: datetime


class ResultAggregator:
    """
    Aggregates workflow execution results and calculates quality scores.
    
    Uses 5-dimension quality assessment:
    - Completeness (30%): Task completion rate
    - Accuracy (25%): Context quality + response quality
    - Efficiency (20%): Token efficiency + time efficiency
    - Reliability (15%): Error rate + retry count
    - Coherence (10%): Response consistency
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality scoring weights
        self.weights = {
            "completeness": 0.30,
            "accuracy": 0.25,
            "efficiency": 0.20,
            "reliability": 0.15,
            "coherence": 0.10
        }
        
        # Efficiency benchmarks
        self.benchmarks = {
            "tokens_per_subtask": 500,  # Target tokens/subtask
            "time_per_subtask_ms": 5000  # Target 5s/subtask
        }
    
    def aggregate_results(
        self,
        workflow_id: int,
        execution_id: int,
        subtask_executions: Dict[str, SubtaskExecution],
        decomposition_metadata: Dict[str, Any],
        agent_selection_metadata: Dict[str, Any],
        context_engineering_metadata: Dict[str, Any],
        agent_execution_metadata: Dict[str, Any]
    ) -> AggregatedResults:
        """
        Aggregate all workflow execution results and calculate quality scores.
        
        Args:
            workflow_id: Workflow ID
            execution_id: Execution ID
            subtask_executions: Dict of subtask execution results
            decomposition_metadata: Task decomposition metadata
            agent_selection_metadata: Agent selection metadata
            context_engineering_metadata: Context engineering metadata
            agent_execution_metadata: Agent execution metadata
            
        Returns:
            AggregatedResults with quality scores and metrics
        """
        self.logger.info(f"📊 Aggregating results for workflow {workflow_id} execution {execution_id}")
        
        # 1. Calculate quality scores
        quality_scores = self._calculate_quality_scores(
            subtask_executions,
            context_engineering_metadata
        )
        
        # 2. Calculate aggregate metrics
        total_subtasks = len(subtask_executions)
        completed = sum(1 for e in subtask_executions.values() if e.status == SubtaskStatus.COMPLETED)
        failed = sum(1 for e in subtask_executions.values() if e.status == SubtaskStatus.FAILED)
        
        total_tokens = sum(e.tokens_used for e in subtask_executions.values())
        total_time = sum(e.execution_time_ms for e in subtask_executions.values())
        total_retries = sum(e.retry_count for e in subtask_executions.values())
        
        avg_context_quality = sum(e.context_quality for e in subtask_executions.values()) / max(total_subtasks, 1)
        
        # 3. Analyze agent performance
        agent_performance = self._analyze_agent_performance(subtask_executions)
        
        # 4. Generate recommendations
        recommendations = self._generate_recommendations(
            quality_scores,
            subtask_executions,
            agent_performance,
            decomposition_metadata,
            context_engineering_metadata
        )
        
        self.logger.info(
            f"✅ Aggregation complete: {quality_scores.overall:.0%} overall quality, "
            f"{completed}/{total_subtasks} completed"
        )
        
        return AggregatedResults(
            workflow_id=workflow_id,
            execution_id=execution_id,
            quality_scores=quality_scores,
            total_subtasks=total_subtasks,
            completed_subtasks=completed,
            failed_subtasks=failed,
            total_tokens_used=total_tokens,
            total_execution_time_ms=total_time,
            avg_context_quality=avg_context_quality,
            total_retries=total_retries,
            agent_performance=agent_performance,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _calculate_quality_scores(
        self,
        subtask_executions: Dict[str, SubtaskExecution],
        context_metadata: Dict[str, Any]
    ) -> QualityScores:
        """Calculate 5-dimension quality scores"""
        
        if not subtask_executions:
            return QualityScores(0, 0, 0, 0, 0, 0)
        
        # 1. Completeness Score (30%)
        completeness = self._calculate_completeness_score(subtask_executions)
        
        # 2. Accuracy Score (25%)
        accuracy = self._calculate_accuracy_score(subtask_executions, context_metadata)
        
        # 3. Efficiency Score (20%)
        efficiency = self._calculate_efficiency_score(subtask_executions)
        
        # 4. Reliability Score (15%)
        reliability = self._calculate_reliability_score(subtask_executions)
        
        # 5. Coherence Score (10%)
        coherence = self._calculate_coherence_score(subtask_executions)
        
        # Overall weighted score
        overall = (
            completeness * self.weights["completeness"] +
            accuracy * self.weights["accuracy"] +
            efficiency * self.weights["efficiency"] +
            reliability * self.weights["reliability"] +
            coherence * self.weights["coherence"]
        )
        
        # PHASE 4B: Calculate confidence interval for overall score
        confidence_interval = None
        if PROBABILITY_AVAILABLE and len(subtask_executions) >= 2:
            try:
                # Collect individual scores as samples
                scores = [completeness, accuracy, efficiency, reliability, coherence]
                ci = ProbabilityTheory.calculate_confidence_interval(scores, confidence_level=0.95)
                confidence_interval = {
                    "mean": ci.mean,
                    "lower_bound": ci.lower_bound,
                    "upper_bound": ci.upper_bound,
                    "confidence_level": ci.confidence_level
                }
                self.logger.debug(
                    f"📊 Quality Score Confidence Interval: "
                    f"{ci.mean:.2f} [{ci.lower_bound:.2f}, {ci.upper_bound:.2f}] @ {ci.confidence_level:.0%}"
                )
            except Exception as e:
                self.logger.warning(f"Could not calculate confidence interval: {e}")
        
        return QualityScores(
            completeness=completeness,
            accuracy=accuracy,
            efficiency=efficiency,
            reliability=reliability,
            coherence=coherence,
            overall=overall,
            overall_confidence_interval=confidence_interval
        )
    
    def _calculate_completeness_score(
        self,
        executions: Dict[str, SubtaskExecution]
    ) -> float:
        """
        Completeness: Did all subtasks complete successfully?
        Score = (completed subtasks / total subtasks)
        """
        total = len(executions)
        completed = sum(1 for e in executions.values() if e.status == SubtaskStatus.COMPLETED)
        return completed / total if total > 0 else 0
    
    def _calculate_accuracy_score(
        self,
        executions: Dict[str, SubtaskExecution],
        context_metadata: Dict[str, Any]
    ) -> float:
        """
        Accuracy: Quality of responses and context usage
        Score = average(context_quality, response_quality)
        """
        total = len(executions)
        
        # Context quality from context engineering
        context_quality = context_metadata.get("summary", {}).get("avg_context_quality", 0.7)
        
        # Response quality heuristic: based on response length and completion
        response_scores = []
        for exec in executions.values():
            if exec.status == SubtaskStatus.COMPLETED and exec.llm_response:
                # Simple heuristic: responses with >50 chars considered quality
                response_length = len(exec.llm_response or "")
                if response_length > 50:
                    response_scores.append(min(response_length / 500, 1.0))  # Cap at 500 chars
                else:
                    response_scores.append(0.5)  # Short response
            else:
                response_scores.append(0.0)  # Failed or no response
        
        response_quality = sum(response_scores) / max(len(response_scores), 1)
        
        # Weighted average
        return (context_quality * 0.6 + response_quality * 0.4)
    
    def _calculate_efficiency_score(
        self,
        executions: Dict[str, SubtaskExecution]
    ) -> float:
        """
        Efficiency: Token usage and execution time vs benchmarks
        Score = average(token_efficiency, time_efficiency)
        
        Uses inverse ratio scoring (benchmark/actual) so values never go negative.
        Perfect score (1.0) when actual ≤ benchmark, decreases as actual exceeds benchmark.
        """
        total = len(executions)
        if total == 0:
            return 0.0
        
        # Token efficiency - inverse ratio
        total_tokens = sum(e.tokens_used for e in executions.values())
        avg_tokens = total_tokens / total
        if avg_tokens > 0:
            # If avg_tokens ≤ benchmark: score = 1.0
            # If avg_tokens > benchmark: score = benchmark / avg_tokens (e.g., 500/1843 = 0.27)
            token_efficiency = min(1.0, self.benchmarks["tokens_per_subtask"] / avg_tokens)
        else:
            token_efficiency = 1.0
        
        # Time efficiency - inverse ratio
        total_time = sum(e.execution_time_ms for e in executions.values())
        avg_time = total_time / total
        if avg_time > 0:
            time_efficiency = min(1.0, self.benchmarks["time_per_subtask_ms"] / avg_time)
        else:
            time_efficiency = 1.0
        
        self.logger.debug(
            f"📊 Efficiency: tokens={token_efficiency:.2%} (avg {avg_tokens:.0f} vs benchmark {self.benchmarks['tokens_per_subtask']}), "
            f"time={time_efficiency:.2%} (avg {avg_time:.0f}ms vs benchmark {self.benchmarks['time_per_subtask_ms']}ms)"
        )
        
        return (token_efficiency * 0.5 + time_efficiency * 0.5)
    
    def _calculate_reliability_score(
        self,
        executions: Dict[str, SubtaskExecution]
    ) -> float:
        """
        Reliability: Error rate and retry count
        Score = (1 - error_rate) * (1 - retry_penalty)
        """
        total = len(executions)
        
        # Error rate
        failed = sum(1 for e in executions.values() if e.status == SubtaskStatus.FAILED)
        error_rate = failed / total if total > 0 else 0
        
        # Retry penalty
        total_retries = sum(e.retry_count for e in executions.values())
        retry_penalty = min(0.3, total_retries * 0.05)  # Max 30% penalty, 5% per retry
        
        return max(0, (1 - error_rate) * (1 - retry_penalty))
    
    def _calculate_coherence_score(
        self,
        executions: Dict[str, SubtaskExecution]
    ) -> float:
        """
        Coherence: Consistency across subtask results
        Score based on similar response patterns and agent consistency
        """
        # Simple heuristic: check if all completed tasks have responses
        completed = [e for e in executions.values() if e.status == SubtaskStatus.COMPLETED]
        
        if not completed:
            return 0.0
        
        # Check response consistency (all have responses)
        responses_present = sum(1 for e in completed if e.llm_response and len(e.llm_response) > 0)
        response_consistency = responses_present / len(completed) if completed else 0
        
        # Check token consistency (similar token usage across tasks)
        token_usages = [e.tokens_used for e in completed if e.tokens_used > 0]
        if token_usages:
            avg_tokens = sum(token_usages) / len(token_usages)
            variance = sum((t - avg_tokens) ** 2 for t in token_usages) / len(token_usages)
            std_dev = variance ** 0.5
            # Lower std_dev = more consistent
            token_consistency = max(0, 1 - (std_dev / (avg_tokens + 1)))
        else:
            token_consistency = 0.5
        
        return (response_consistency * 0.6 + token_consistency * 0.4)
    
    def _analyze_agent_performance(
        self,
        executions: Dict[str, SubtaskExecution]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by agent"""
        
        agent_stats = {}
        
        for exec in executions.values():
            agent_name = exec.agent_name
            if agent_name not in agent_stats:
                agent_stats[agent_name] = {
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "failed_tasks": 0,
                    "total_tokens": 0,
                    "total_time_ms": 0,
                    "total_retries": 0
                }
            
            stats = agent_stats[agent_name]
            stats["total_tasks"] += 1
            stats["total_tokens"] += exec.tokens_used
            stats["total_time_ms"] += exec.execution_time_ms
            stats["total_retries"] += exec.retry_count
            
            if exec.status == SubtaskStatus.COMPLETED:
                stats["completed_tasks"] += 1
            elif exec.status == SubtaskStatus.FAILED:
                stats["failed_tasks"] += 1
        
        # Calculate derived metrics
        for agent_name, stats in agent_stats.items():
            stats["success_rate"] = stats["completed_tasks"] / stats["total_tasks"] if stats["total_tasks"] > 0 else 0
            stats["avg_tokens"] = stats["total_tokens"] / stats["total_tasks"] if stats["total_tasks"] > 0 else 0
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_tasks"] if stats["total_tasks"] > 0 else 0
        
        return agent_stats
    
    def _generate_recommendations(
        self,
        quality_scores: QualityScores,
        executions: Dict[str, SubtaskExecution],
        agent_performance: Dict[str, Dict[str, Any]],
        decomposition_metadata: Dict[str, Any],
        context_metadata: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on results"""
        
        recommendations = []
        
        # Completeness recommendations
        if quality_scores.completeness < 0.8:
            failed_count = sum(1 for e in executions.values() if e.status == SubtaskStatus.FAILED)
            recommendations.append(
                f"⚠️ Low completeness ({quality_scores.completeness:.0%}): {failed_count} subtasks failed. "
                "Consider improving error handling or agent selection."
            )
        
        # Accuracy recommendations
        if quality_scores.accuracy < 0.7:
            recommendations.append(
                f"⚠️ Low accuracy ({quality_scores.accuracy:.0%}): "
                "Consider enhancing context quality or using more specialized agents."
            )
        
        # Efficiency recommendations
        if quality_scores.efficiency < 0.6:
            total_tokens = sum(e.tokens_used for e in executions.values())
            recommendations.append(
                f"⚠️ Low efficiency ({quality_scores.efficiency:.0%}): {total_tokens} tokens used. "
                "Consider optimizing prompts or reducing context size."
            )
        
        # Reliability recommendations
        if quality_scores.reliability < 0.8:
            total_retries = sum(e.retry_count for e in executions.values())
            if total_retries > 3:
                recommendations.append(
                    f"⚠️ High retry count ({total_retries} retries): "
                    "Consider improving agent configurations or LLM settings."
                )
        
        # Agent-specific recommendations
        for agent_name, stats in agent_performance.items():
            if stats["success_rate"] < 0.7:
                recommendations.append(
                    f"⚠️ Agent '{agent_name}' has low success rate ({stats['success_rate']:.0%}). "
                    "Consider retraining or replacing this agent."
                )
        
        # Success message if everything is good
        if quality_scores.overall >= 0.8 and not recommendations:
            recommendations.append(
                f"✅ Excellent execution quality ({quality_scores.overall:.0%})! All dimensions performing well."
            )
        
        return recommendations

