"""
Playbook Learning Service - Stage 6 (Learn)
============================================

Analyzes playbook executions to extract improvement patterns and generate
suggestions for prompt rewrites, model upgrades, and tool additions.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.core import RecipeExecution, WorkflowTemplate

logger = logging.getLogger(__name__)


class PlaybookLearningService:
    """
    Analyzes playbook executions and extracts improvement patterns.
    Stores results in the playbook's learning_data JSONB field.
    """

    def __init__(self, db: Optional[Session] = None):
        self.db = db or next(get_db())

    def analyze_execution(self, execution_id: str) -> Dict[str, Any]:
        """
        Analyze a completed execution to extract patterns and improvement suggestions.

        Args:
            execution_id: The execution_id string (e.g. "exec-abc123def456")

        Returns:
            Dict with patterns array, suggestions array, and performance_metrics object.
        """
        # Fetch execution record
        execution = self.db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id
        ).first()

        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        # Fetch the associated playbook
        playbook = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.id == execution.recipe_id
        ).first()

        if not playbook:
            raise ValueError(f"Playbook not found for execution: {execution_id}")

        # Extract patterns from execution trace
        patterns = self._extract_patterns(execution)

        # Generate improvement suggestions
        suggestions = self._generate_suggestions(execution, patterns)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(execution)

        result = {
            "execution_id": execution_id,
            "analyzed_at": datetime.utcnow().isoformat(),
            "patterns": patterns,
            "suggestions": suggestions,
            "performance_metrics": performance_metrics,
        }

        # Store results in playbook's learning_data
        self._update_learning_data(playbook, result)

        return result

    def _extract_patterns(self, execution: RecipeExecution) -> List[Dict[str, Any]]:
        """
        Analyze execution trace to extract recurring patterns.

        Examines step_results, status, and error_message to identify:
        - Failure patterns (e.g., "Always fails when input X is null")
        - Success patterns (e.g., "Step 2 succeeds when Step 1 produces JSON")
        - Timing patterns (e.g., "Step 3 takes >10s consistently")
        """
        patterns: List[Dict[str, Any]] = []
        step_results = execution.step_results or []

        # Pattern: Execution failed
        if execution.status == 'failed':
            pattern = {
                "type": "failure",
                "description": f"Execution failed: {execution.error_message or 'Unknown error'}",
                "severity": "high",
                "step": execution.current_step,
                "stage": execution.current_stage,
            }
            patterns.append(pattern)

        # Analyze individual step results
        for idx, step_result in enumerate(step_results):
            if not isinstance(step_result, dict):
                continue

            step_status = step_result.get("status", "unknown")
            step_error = step_result.get("error")
            step_duration = step_result.get("duration_ms")

            # Pattern: Step failure
            if step_status == "failed":
                patterns.append({
                    "type": "step_failure",
                    "description": f"Step {idx + 1} failed: {step_error or 'Unknown error'}",
                    "severity": "high",
                    "step_index": idx,
                    "agent_id": step_result.get("agent_id"),
                })

            # Pattern: Slow step (>30 seconds)
            if step_duration and step_duration > 30000:
                patterns.append({
                    "type": "slow_step",
                    "description": f"Step {idx + 1} took {step_duration / 1000:.1f}s (>30s threshold)",
                    "severity": "medium",
                    "step_index": idx,
                    "duration_ms": step_duration,
                })

            # Pattern: Empty output
            step_output = step_result.get("output")
            if step_status == "completed" and not step_output:
                patterns.append({
                    "type": "empty_output",
                    "description": f"Step {idx + 1} completed but produced no output",
                    "severity": "low",
                    "step_index": idx,
                })

            # Pattern: Retry needed
            retries = step_result.get("retries", 0)
            if retries > 0:
                patterns.append({
                    "type": "retry_needed",
                    "description": f"Step {idx + 1} required {retries} retries before completing",
                    "severity": "medium",
                    "step_index": idx,
                    "retries": retries,
                })

        # Pattern: Input data issues
        input_data = execution.input_data or {}
        playbook = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.id == execution.recipe_id
        ).first()
        if playbook and playbook.inputs:
            for param_name, param_config in playbook.inputs.items():
                if isinstance(param_config, dict) and param_config.get("required"):
                    if param_name not in input_data or input_data[param_name] is None:
                        patterns.append({
                            "type": "missing_input",
                            "description": f"Required input '{param_name}' was missing or null",
                            "severity": "high",
                            "parameter": param_name,
                        })

        return patterns

    def _generate_suggestions(
        self,
        execution: RecipeExecution,
        patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate improvement suggestions based on extracted patterns.

        Suggestion types:
        - prompt_rewrite: Suggest improved prompt templates
        - model_upgrade: Suggest using a more capable model
        - tool_addition: Suggest adding tools to steps
        """
        suggestions: List[Dict[str, Any]] = []

        for pattern in patterns:
            pattern_type = pattern.get("type")

            if pattern_type == "step_failure":
                # Suggest prompt rewrite for failing steps
                step_idx = pattern.get("step_index")
                suggestions.append({
                    "type": "prompt_rewrite",
                    "description": f"Consider rewriting the prompt for step {step_idx + 1} to handle the error case",
                    "step_index": step_idx,
                    "reason": pattern["description"],
                    "priority": "high",
                })
                # Suggest model upgrade for persistent failures
                suggestions.append({
                    "type": "model_upgrade",
                    "description": f"Consider upgrading the model for step {step_idx + 1} to improve reliability",
                    "step_index": step_idx,
                    "reason": pattern["description"],
                    "priority": "medium",
                })

            elif pattern_type == "slow_step":
                step_idx = pattern.get("step_index")
                suggestions.append({
                    "type": "prompt_rewrite",
                    "description": f"Simplify the prompt for step {step_idx + 1} to reduce execution time",
                    "step_index": step_idx,
                    "reason": pattern["description"],
                    "priority": "medium",
                })

            elif pattern_type == "empty_output":
                step_idx = pattern.get("step_index")
                suggestions.append({
                    "type": "prompt_rewrite",
                    "description": f"Add explicit output format instructions to step {step_idx + 1}",
                    "step_index": step_idx,
                    "reason": pattern["description"],
                    "priority": "low",
                })

            elif pattern_type == "retry_needed":
                step_idx = pattern.get("step_index")
                retries = pattern.get("retries", 0)
                if retries >= 3:
                    suggestions.append({
                        "type": "model_upgrade",
                        "description": f"Step {step_idx + 1} needs {retries} retries - consider a more capable model",
                        "step_index": step_idx,
                        "reason": pattern["description"],
                        "priority": "high",
                    })
                suggestions.append({
                    "type": "tool_addition",
                    "description": f"Add validation tools to step {step_idx + 1} to reduce retry needs",
                    "step_index": step_idx,
                    "reason": pattern["description"],
                    "priority": "medium",
                })

            elif pattern_type == "failure":
                # General execution failure
                suggestions.append({
                    "type": "prompt_rewrite",
                    "description": "Review all step prompts for robustness against edge cases",
                    "reason": pattern["description"],
                    "priority": "high",
                })

        return suggestions

    def _calculate_performance_metrics(
        self, execution: RecipeExecution
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics from the execution.

        Returns object with:
        - total_duration_ms: Total execution time
        - step_durations: Per-step timing breakdown
        - success_rate: Percentage of steps that completed successfully
        - total_retries: Sum of retries across all steps
        - error_count: Number of steps that errored
        """
        step_results = execution.step_results or []
        total_steps = len(step_results)

        step_durations: List[Dict[str, Any]] = []
        successful_steps = 0
        total_retries = 0
        error_count = 0
        total_duration_ms = 0

        for idx, step_result in enumerate(step_results):
            if not isinstance(step_result, dict):
                continue

            duration = step_result.get("duration_ms", 0)
            total_duration_ms += duration

            step_durations.append({
                "step_index": idx,
                "duration_ms": duration,
                "status": step_result.get("status", "unknown"),
            })

            if step_result.get("status") == "completed":
                successful_steps += 1
            elif step_result.get("status") == "failed":
                error_count += 1

            total_retries += step_result.get("retries", 0)

        # Calculate overall duration from timestamps if available
        if execution.started_at and execution.completed_at:
            overall_duration = (execution.completed_at - execution.started_at).total_seconds() * 1000
        else:
            overall_duration = total_duration_ms

        return {
            "total_duration_ms": overall_duration,
            "step_durations": step_durations,
            "success_rate": (successful_steps / total_steps) if total_steps > 0 else 0.0,
            "total_retries": total_retries,
            "error_count": error_count,
            "total_steps": total_steps,
            "successful_steps": successful_steps,
        }

    def _update_learning_data(
        self, playbook: WorkflowTemplate, analysis_result: Dict[str, Any]
    ) -> None:
        """
        Merge analysis results into the playbook's learning_data JSONB field.

        Maintains a rolling history of analyses, appending new entries
        and capping history to the most recent 50 entries.
        """
        existing_data = playbook.learning_data or {}

        # Initialize structure if empty
        if not existing_data.get("analyses"):
            existing_data["analyses"] = []

        # Append the new analysis
        existing_data["analyses"].append({
            "execution_id": analysis_result["execution_id"],
            "analyzed_at": analysis_result["analyzed_at"],
            "pattern_count": len(analysis_result["patterns"]),
            "suggestion_count": len(analysis_result["suggestions"]),
            "performance_metrics": analysis_result["performance_metrics"],
        })

        # Cap to last 50 analyses
        existing_data["analyses"] = existing_data["analyses"][-50:]

        # Update latest patterns and suggestions (overwrite with most recent)
        existing_data["latest_patterns"] = analysis_result["patterns"]
        existing_data["latest_suggestions"] = analysis_result["suggestions"]
        existing_data["latest_performance"] = analysis_result["performance_metrics"]
        existing_data["last_analyzed_at"] = analysis_result["analyzed_at"]

        # Update playbook
        playbook.learning_data = existing_data
        self.db.commit()
        logger.info(
            f"Updated learning_data for playbook {playbook.id} "
            f"({len(analysis_result['patterns'])} patterns, "
            f"{len(analysis_result['suggestions'])} suggestions)"
        )
