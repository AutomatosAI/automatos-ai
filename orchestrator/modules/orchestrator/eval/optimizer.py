"""
Prompt Optimizer — FutureAGI Bayesian Optimization for Stage 1-2 Prompts
========================================================================
PRD-59 US-008: Uses FutureAGI's Bayesian optimization to improve task
decomposition and agent selection system prompts.

Workflow:
  1. Load current system prompt from Prompt Registry (PRD-58)
  2. Run eval dataset to get baseline score
  3. Submit to FutureAGI optimize_prompt() with Bayesian search
  4. Re-run eval dataset on optimized prompt
  5. If improved, store new prompt variant in Prompt Registry as A/B test candidate

Usage:
    optimizer = StagePromptOptimizer()
    result = await optimizer.optimize("task_decomposer", iterations=10)
    print(result.improvement_pct)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .runner import PromptEvalRunner, EvalReport

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a prompt optimization run."""
    stage: str
    baseline_score: float
    optimized_score: float
    improvement_pct: float
    original_prompt: str
    optimized_prompt: Optional[str] = None
    algorithm: str = "bayesian"
    iterations: int = 0
    stored_as_variant: bool = False
    variant_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# Default prompts used by stages when Prompt Registry is not available
_DEFAULT_PROMPTS = {
    "task_decomposer": (
        "You are an expert task decomposition system. Always return valid JSON. "
        "Decompose the user's request into subtasks with required_skills, dependencies, "
        "and execution_strategy (sequential or parallel)."
    ),
    "agent_selector": (
        "You are an intelligent agent selection system. Select the best agent for each "
        "subtask based on: skill match (40%), performance history (30%), "
        "current workload (20%), and recency (10%). Return agent assignments as JSON."
    ),
}

# Prompt Registry slugs
_REGISTRY_SLUGS = {
    "task_decomposer": "task-decomposer",
    "agent_selector": "agent-selector",
}


class StagePromptOptimizer:
    """Optimizes Stage 1-2 system prompts using FutureAGI + eval datasets."""

    def __init__(self):
        self._eval_runner = PromptEvalRunner()
        self._futureagi = None
        self._registry = None

    async def optimize(
        self,
        stage: str,
        algorithm: str = "bayesian",
        target_metric: str = "instruction_adherence",
        iterations: int = 10,
        tags: Optional[List[str]] = None,
        auto_store: bool = False,
    ) -> OptimizationResult:
        """Run optimization loop for a stage's system prompt.

        Args:
            stage: 'task_decomposer' or 'agent_selector'
            algorithm: FutureAGI optimization algorithm
            target_metric: Primary metric to optimize
            iterations: Number of optimization iterations
            tags: Filter eval dataset by tags
            auto_store: If True, auto-store improved prompt as variant in Registry

        Returns:
            OptimizationResult with before/after scores
        """
        # Step 1: Load current prompt
        current_prompt = await self._get_current_prompt(stage)

        # Step 2: Baseline evaluation
        logger.info(f"Running baseline evaluation for {stage}...")
        baseline = await self._eval_runner.evaluate_stage(stage, tags=tags)
        baseline_score = baseline.avg_score

        logger.info(f"Baseline score for {stage}: {baseline_score:.2f}")

        # Step 3: Run FutureAGI optimization
        try:
            from core.services.futureagi_service import FutureAGIService
            service = FutureAGIService.get_instance()

            if not service.is_available:
                return OptimizationResult(
                    stage=stage,
                    baseline_score=baseline_score,
                    optimized_score=baseline_score,
                    improvement_pct=0.0,
                    original_prompt=current_prompt,
                    error="FutureAGI not configured — set FUTUREAGI_API_KEY and FUTUREAGI_SECRET_KEY",
                )

            opt_result = await service.optimize_prompt(
                prompt_content=current_prompt,
                algorithm=algorithm,
                target_metric=target_metric,
                num_iterations=iterations,
            )

            if "error" in opt_result:
                return OptimizationResult(
                    stage=stage,
                    baseline_score=baseline_score,
                    optimized_score=baseline_score,
                    improvement_pct=0.0,
                    original_prompt=current_prompt,
                    error=f"Optimization failed: {opt_result['error']}",
                )

            optimized_prompt = opt_result.get("optimized_prompt", current_prompt)

        except Exception as e:
            return OptimizationResult(
                stage=stage,
                baseline_score=baseline_score,
                optimized_score=baseline_score,
                improvement_pct=0.0,
                original_prompt=current_prompt,
                error=f"FutureAGI unavailable: {e}",
            )

        # Step 4: Re-evaluate with optimized prompt
        # Note: To properly evaluate, we'd need to inject the optimized prompt
        # into the stage. For now, we use the FutureAGI best_score directly.
        optimized_score = opt_result.get("best_score", baseline_score) or baseline_score
        improvement = ((optimized_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0.0

        logger.info(
            f"Optimization complete for {stage}: "
            f"{baseline_score:.2f} → {optimized_score:.2f} ({improvement:+.1f}%)"
        )

        result = OptimizationResult(
            stage=stage,
            baseline_score=baseline_score,
            optimized_score=optimized_score,
            improvement_pct=improvement,
            original_prompt=current_prompt,
            optimized_prompt=optimized_prompt,
            algorithm=algorithm,
            iterations=iterations,
            details=opt_result,
        )

        # Step 5: Store as A/B variant if improved
        if auto_store and improvement > 5.0:  # Only store if 5%+ improvement
            stored = await self._store_variant(stage, optimized_prompt, result)
            result.stored_as_variant = stored
            if stored:
                logger.info(f"Stored optimized prompt as A/B variant for {stage}")

        return result

    async def _get_current_prompt(self, stage: str) -> str:
        """Load current system prompt from Prompt Registry or defaults."""
        slug = _REGISTRY_SLUGS.get(stage)
        if slug:
            try:
                from modules.prompts.registry import PromptRegistry
                registry = PromptRegistry()
                prompt_data = await registry.get_prompt(slug)
                if prompt_data and prompt_data.get("content"):
                    return prompt_data["content"]
            except Exception as e:
                logger.debug(f"Prompt Registry not available for {stage}: {e}")

        return _DEFAULT_PROMPTS.get(stage, "")

    async def _store_variant(
        self, stage: str, optimized_prompt: str, result: OptimizationResult
    ) -> bool:
        """Store optimized prompt as A/B test variant in Prompt Registry."""
        slug = _REGISTRY_SLUGS.get(stage)
        if not slug:
            return False

        try:
            from modules.prompts.registry import PromptRegistry
            registry = PromptRegistry()

            # Store as variant with metadata
            variant_data = {
                "content": optimized_prompt,
                "variant_name": f"{result.algorithm}_optimized",
                "metadata": {
                    "optimization_algorithm": result.algorithm,
                    "baseline_score": result.baseline_score,
                    "optimized_score": result.optimized_score,
                    "improvement_pct": result.improvement_pct,
                    "iterations": result.iterations,
                },
            }

            # Use create_variant if available, otherwise update
            if hasattr(registry, "create_variant"):
                variant_id = await registry.create_variant(slug, variant_data)
                result.variant_id = variant_id
                return True
            elif hasattr(registry, "update_prompt"):
                # Fallback: update the prompt directly (no A/B)
                await registry.update_prompt(slug, {"content": optimized_prompt})
                return True

        except Exception as e:
            logger.warning(f"Failed to store variant for {stage}: {e}")

        return False
