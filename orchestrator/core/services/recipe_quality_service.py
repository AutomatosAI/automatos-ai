"""
Recipe Quality Service - Stage 7 (Quality)
===========================================

Performs 5-dimensional quality assessment of recipe executions:
completeness, accuracy, efficiency, reliability, and cost.
Updates the recipe's quality_score field with the average score.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.core import RecipeExecution, WorkflowTemplate

logger = logging.getLogger(__name__)


class RecipeQualityService:
    """
    Performs 5-dimensional quality assessment of recipe executions.
    Updates the recipe's quality_score with a rolling average.
    """

    def __init__(self, db: Session):
        if db is None:
            raise ValueError("RecipeQualityService requires an injected DB session")
        self.db = db

    def assess_quality(
        self, execution_id: str, learnings: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Perform a 5-dimensional quality assessment of a recipe execution.

        Args:
            execution_id: The execution_id string (e.g. "exec-abc123def456")
            learnings: Optional learnings dict from RecipeLearningService.analyze_execution()
            workspace_id: Optional workspace UUID for isolation filtering

        Returns:
            Dict with quality_score, breakdown object, grade, and bottlenecks array.
        """
        # Fetch execution record
        query = self.db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id
        )
        if workspace_id:
            query = query.filter(RecipeExecution.workspace_id == workspace_id)
        execution = query.first()

        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        # Fetch the associated recipe
        recipe_query = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.id == execution.recipe_id
        )
        if workspace_id:
            recipe_query = recipe_query.filter(WorkflowTemplate.workspace_id == workspace_id)
        recipe = recipe_query.first()

        if not recipe:
            raise ValueError(f"Recipe not found for execution: {execution_id}")

        # Calculate 5 quality dimensions
        completeness = self._assess_completeness(execution, recipe)
        accuracy = self._assess_accuracy(execution, recipe)
        efficiency = self._assess_efficiency(execution, recipe)
        reliability = self._assess_reliability(execution, learnings)
        cost = self._assess_cost(execution, recipe)

        breakdown = {
            "completeness": completeness,
            "accuracy": accuracy,
            "efficiency": efficiency,
            "reliability": reliability,
            "cost": cost,
        }

        # Calculate average quality score
        quality_score = sum(breakdown.values()) / len(breakdown)

        # Assign grade
        grade = self._calculate_grade(quality_score)

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(execution)

        result = {
            "execution_id": execution_id,
            "assessed_at": datetime.utcnow().isoformat(),
            "quality_score": round(quality_score, 4),
            "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "grade": grade,
            "bottlenecks": bottlenecks,
        }

        # Update recipe's quality_score with rolling average
        self._update_quality_score(recipe, quality_score)

        return result

    def _assess_completeness(
        self, execution: RecipeExecution, recipe: WorkflowTemplate
    ) -> float:
        """
        Assess completeness: Did all steps execute and produce output?

        Scores based on:
        - Percentage of steps that completed
        - Whether expected outputs were produced
        - Whether all required steps ran
        """
        step_results = execution.step_results or []
        recipe_steps = recipe.steps or []
        total_expected = len(recipe_steps) if recipe_steps else len(step_results)

        if total_expected == 0:
            return 1.0 if execution.status == 'completed' else 0.0

        # Score: proportion of steps that completed
        completed_steps = sum(
            1 for sr in step_results
            if isinstance(sr, dict) and sr.get("status") == "completed"
        )
        step_completion_score = completed_steps / total_expected

        # Score: did steps produce output?
        steps_with_output = sum(
            1 for sr in step_results
            if isinstance(sr, dict) and sr.get("output")
        )
        output_score = steps_with_output / total_expected if total_expected > 0 else 0.0

        # Score: overall execution completion
        execution_score = 1.0 if execution.status == 'completed' else 0.3 if execution.status == 'running' else 0.0

        # Weighted average: step completion matters most
        return (step_completion_score * 0.5) + (output_score * 0.3) + (execution_score * 0.2)

    def _assess_accuracy(
        self, execution: RecipeExecution, recipe: WorkflowTemplate
    ) -> float:
        """
        Assess accuracy: Did the execution produce correct/valid results?

        Scores based on:
        - Whether output matches expected output schema
        - Absence of error messages in completed steps
        - Output data validity
        """
        step_results = execution.step_results or []
        total_steps = len(step_results)

        if total_steps == 0:
            return 0.5  # No data to assess

        # Score: proportion of steps without errors
        error_free_steps = sum(
            1 for sr in step_results
            if isinstance(sr, dict) and not sr.get("error")
        )
        error_free_score = error_free_steps / total_steps

        # Score: output schema conformance
        output_schema = recipe.outputs or {}
        output_data = execution.output_data or {}
        schema_score = 1.0
        if output_schema and isinstance(output_schema, dict):
            expected_fields = set(output_schema.keys())
            actual_fields = set(output_data.keys()) if isinstance(output_data, dict) else set()
            if expected_fields:
                schema_score = len(expected_fields & actual_fields) / len(expected_fields)

        # Score: no execution-level error
        no_error_score = 1.0 if not execution.error_message else 0.0

        return (error_free_score * 0.5) + (schema_score * 0.3) + (no_error_score * 0.2)

    def _assess_efficiency(
        self, execution: RecipeExecution, recipe: WorkflowTemplate
    ) -> float:
        """
        Assess efficiency: How well did the execution use time and resources?

        Scores based on:
        - Per-step timeout adherence
        - Total execution time vs configured timeout
        - Retry overhead
        """
        step_results = execution.step_results or []
        execution_config = recipe.execution_config or {}

        # Get configured timeouts — normalise ms or seconds to ms
        raw_step = execution_config.get("timeout_per_step") or execution_config.get("per_step_timeout") or 300
        raw_total = execution_config.get("total_timeout") or 600
        timeout_per_step = raw_step if raw_step >= 1000 else raw_step * 1000   # → ms
        total_timeout = raw_total if raw_total >= 1000 else raw_total * 1000

        # Score: per-step time efficiency
        step_time_scores: List[float] = []
        for sr in step_results:
            if not isinstance(sr, dict):
                continue
            duration = sr.get("duration_ms", 0)
            if duration <= 0:
                step_time_scores.append(0.5)  # Unknown duration
            elif duration <= timeout_per_step * 0.5:
                step_time_scores.append(1.0)  # Well within timeout
            elif duration <= timeout_per_step:
                # Linear scale from 1.0 to 0.5 as duration approaches timeout
                step_time_scores.append(1.0 - 0.5 * (duration / timeout_per_step))
            else:
                step_time_scores.append(0.0)  # Exceeded timeout

        step_time_score = (
            sum(step_time_scores) / len(step_time_scores)
            if step_time_scores else 0.5
        )

        # Score: total execution time efficiency
        total_duration = 0.0
        if execution.started_at and execution.completed_at:
            total_duration = (execution.completed_at - execution.started_at).total_seconds() * 1000
        else:
            total_duration = sum(
                sr.get("duration_ms", 0)
                for sr in step_results
                if isinstance(sr, dict)
            )

        if total_duration <= 0:
            total_time_score = 0.5
        elif total_duration <= total_timeout * 0.5:
            total_time_score = 1.0
        elif total_duration <= total_timeout:
            total_time_score = 1.0 - 0.5 * (total_duration / total_timeout)
        else:
            total_time_score = 0.0

        # Score: retry overhead (fewer retries = more efficient)
        total_retries = sum(
            sr.get("retries", 0)
            for sr in step_results
            if isinstance(sr, dict)
        )
        total_steps = len(step_results)
        if total_steps > 0:
            avg_retries = total_retries / total_steps
            retry_score = max(0.0, 1.0 - (avg_retries * 0.25))  # -25% per avg retry
        else:
            retry_score = 1.0

        return (step_time_score * 0.4) + (total_time_score * 0.3) + (retry_score * 0.3)

    def _assess_reliability(
        self, execution: RecipeExecution, learnings: Optional[Dict[str, Any]]
    ) -> float:
        """
        Assess reliability: How consistently does this execution succeed?

        Scores based on:
        - Execution status (completed vs failed)
        - Pattern severity from learnings
        - Retry frequency
        """
        # Base score from execution status
        status_scores = {
            "completed": 1.0,
            "running": 0.7,
            "pending": 0.5,
            "failed": 0.0,
            "cancelled": 0.2,
        }
        status_score = status_scores.get(execution.status, 0.3)

        # Score based on patterns from learnings
        pattern_score = 1.0
        if learnings and learnings.get("patterns"):
            patterns = learnings["patterns"]
            high_severity = sum(1 for p in patterns if p.get("severity") == "high")
            medium_severity = sum(1 for p in patterns if p.get("severity") == "medium")
            # Deduct for pattern severity
            pattern_score = max(0.0, 1.0 - (high_severity * 0.3) - (medium_severity * 0.1))

        # Score from step-level reliability
        step_results = execution.step_results or []
        total_steps = len(step_results)
        if total_steps > 0:
            failed_steps = sum(
                1 for sr in step_results
                if isinstance(sr, dict) and sr.get("status") == "failed"
            )
            step_reliability = 1.0 - (failed_steps / total_steps)
        else:
            step_reliability = status_score  # Fall back to status

        return (status_score * 0.4) + (pattern_score * 0.3) + (step_reliability * 0.3)

    def _assess_cost(
        self, execution: RecipeExecution, recipe: WorkflowTemplate
    ) -> float:
        """
        Assess cost efficiency: How well did the execution use resources vs results?

        Scores based on:
        - Number of retries (each retry costs tokens/time)
        - Step count efficiency (fewer steps for same result = better)
        - Execution metadata cost data if available
        """
        step_results = execution.step_results or []
        recipe_steps = recipe.steps or []
        total_steps = len(step_results)

        if total_steps == 0:
            return 0.5  # No data to assess

        # Score: retry cost (retries waste resources)
        total_retries = sum(
            sr.get("retries", 0)
            for sr in step_results
            if isinstance(sr, dict)
        )
        max_acceptable_retries = total_steps * 2  # 2 retries per step max
        if max_acceptable_retries > 0:
            retry_cost_score = max(0.0, 1.0 - (total_retries / max_acceptable_retries))
        else:
            retry_cost_score = 1.0

        # Score: step utilization (did all steps contribute?)
        useful_steps = sum(
            1 for sr in step_results
            if isinstance(sr, dict) and sr.get("status") == "completed" and sr.get("output")
        )
        utilization_score = useful_steps / total_steps if total_steps > 0 else 0.0

        # Score: actual vs expected step count
        expected_steps = len(recipe_steps) if recipe_steps else total_steps
        if expected_steps > 0:
            step_efficiency = min(1.0, expected_steps / max(total_steps, 1))
        else:
            step_efficiency = 1.0

        # Score: execution metadata cost if available
        metadata = execution.execution_metadata or {}
        if metadata.get("total_tokens"):
            # Normalize token usage (lower is better, assume 100k tokens as high end)
            token_score = max(0.0, 1.0 - (metadata["total_tokens"] / 100000))
        else:
            token_score = 0.7  # Neutral when no data available

        return (retry_cost_score * 0.3) + (utilization_score * 0.3) + (step_efficiency * 0.2) + (token_score * 0.2)

    def _calculate_grade(self, quality_score: float) -> str:
        """
        Assign a letter grade based on quality score.

        A: >= 0.9
        B: >= 0.75
        C: >= 0.6
        D: >= 0.4
        F: < 0.4
        """
        if quality_score >= 0.9:
            return "A"
        elif quality_score >= 0.75:
            return "B"
        elif quality_score >= 0.6:
            return "C"
        elif quality_score >= 0.4:
            return "D"
        else:
            return "F"

    def _identify_bottlenecks(self, execution: RecipeExecution) -> List[Dict[str, Any]]:
        """
        Identify execution bottlenecks: slowest steps and highest failure steps.

        Returns array of bottleneck objects with type, step_index, and details.
        """
        bottlenecks: List[Dict[str, Any]] = []
        step_results = execution.step_results or []

        if not step_results:
            return bottlenecks

        # Find slowest steps (top 3 by duration)
        timed_steps = [
            (idx, sr)
            for idx, sr in enumerate(step_results)
            if isinstance(sr, dict) and sr.get("duration_ms", 0) > 0
        ]
        timed_steps.sort(key=lambda x: x[1].get("duration_ms", 0), reverse=True)

        for idx, sr in timed_steps[:3]:
            duration_ms = sr.get("duration_ms", 0)
            if duration_ms > 5000:  # Only flag steps >5 seconds
                bottlenecks.append({
                    "type": "slow_step",
                    "step_index": idx,
                    "duration_ms": duration_ms,
                    "agent_id": sr.get("agent_id"),
                    "description": f"Step {idx + 1} took {duration_ms / 1000:.1f}s",
                })

        # Find steps with highest failure/retry rates
        for idx, sr in enumerate(step_results):
            if not isinstance(sr, dict):
                continue

            if sr.get("status") == "failed":
                bottlenecks.append({
                    "type": "failed_step",
                    "step_index": idx,
                    "error": sr.get("error"),
                    "agent_id": sr.get("agent_id"),
                    "description": f"Step {idx + 1} failed: {sr.get('error', 'Unknown error')}",
                })

            retries = sr.get("retries", 0)
            if retries >= 2:
                bottlenecks.append({
                    "type": "high_retry_step",
                    "step_index": idx,
                    "retries": retries,
                    "agent_id": sr.get("agent_id"),
                    "description": f"Step {idx + 1} required {retries} retries",
                })

        return bottlenecks

    def _update_quality_score(
        self, recipe: WorkflowTemplate, new_score: float
    ) -> None:
        """
        Update the recipe's quality_score with a rolling average.

        Uses exponential moving average to weight recent executions more heavily.
        Alpha = 0.3 means 30% weight on new score, 70% on existing average.
        """
        alpha = 0.3

        if recipe.quality_score is not None:
            updated_score = (alpha * new_score) + ((1 - alpha) * recipe.quality_score)
        else:
            updated_score = new_score

        recipe.quality_score = round(updated_score, 4)
        self.db.commit()

        logger.info(
            f"Updated quality_score for recipe {recipe.id}: "
            f"{recipe.quality_score} (grade: {self._calculate_grade(recipe.quality_score)})"
        )
