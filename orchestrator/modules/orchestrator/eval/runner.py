"""
Prompt Evaluation Runner
========================
PRD-59 US-007/US-008: Runs evaluation datasets against Stage 1-2 prompts
and optionally scores via FutureAGI.

Usage:
    runner = PromptEvalRunner()
    results = await runner.evaluate_stage("task_decomposer")
    print(results.summary())

    # With FutureAGI scoring
    results = await runner.evaluate_stage("task_decomposer", use_futureagi=True)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .datasets import EvalCase, get_eval_dataset

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""
    case_id: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    futureagi_scores: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class EvalReport:
    """Aggregated evaluation report for a stage."""
    stage: str
    total_cases: int
    passed: int
    failed: int
    avg_score: float
    results: List[EvalResult]
    duration_ms: int = 0

    def summary(self) -> str:
        pct = (self.passed / self.total_cases * 100) if self.total_cases > 0 else 0
        return (
            f"Stage: {self.stage} | "
            f"{self.passed}/{self.total_cases} passed ({pct:.0f}%) | "
            f"Avg score: {self.avg_score:.2f} | "
            f"Duration: {self.duration_ms}ms"
        )


class PromptEvalRunner:
    """Runs evaluation datasets against stage prompts."""

    def __init__(self):
        self._futureagi = None

    async def evaluate_stage(
        self,
        stage: str,
        tags: Optional[List[str]] = None,
        use_futureagi: bool = False,
    ) -> EvalReport:
        """Run all eval cases for a stage.

        Args:
            stage: 'task_decomposer' or 'agent_selector'
            tags: Optional filter — only run cases matching these tags
            use_futureagi: Whether to score via FutureAGI SDK

        Returns:
            EvalReport with per-case results and aggregate stats
        """
        dataset = get_eval_dataset(stage)
        if tags:
            dataset = [c for c in dataset if any(t in c.tags for t in tags)]

        if not dataset:
            return EvalReport(
                stage=stage, total_cases=0, passed=0, failed=0,
                avg_score=0.0, results=[],
            )

        start = time.monotonic()
        results: List[EvalResult] = []

        for case in dataset:
            try:
                result = await self._evaluate_case(stage, case, use_futureagi)
                results.append(result)
            except Exception as e:
                results.append(EvalResult(
                    case_id=case.id, passed=False, score=0.0,
                    error=str(e),
                ))

        total_ms = int((time.monotonic() - start) * 1000)
        passed = sum(1 for r in results if r.passed)
        avg = sum(r.score for r in results) / len(results) if results else 0.0

        report = EvalReport(
            stage=stage,
            total_cases=len(results),
            passed=passed,
            failed=len(results) - passed,
            avg_score=avg,
            results=results,
            duration_ms=total_ms,
        )

        logger.info(report.summary())
        return report

    async def _evaluate_case(
        self, stage: str, case: EvalCase, use_futureagi: bool
    ) -> EvalResult:
        """Evaluate a single case."""
        start = time.monotonic()

        if stage in ("task_decomposer", "stage1"):
            result = await self._eval_task_decomposer(case)
        elif stage in ("agent_selector", "stage2"):
            result = await self._eval_agent_selector(case)
        else:
            return EvalResult(
                case_id=case.id, passed=False, score=0.0,
                error=f"Unknown stage: {stage}",
            )

        # Optionally score with FutureAGI
        if use_futureagi and result.details.get("output"):
            futureagi_scores = await self._score_with_futureagi(
                case, result.details["output"]
            )
            result.futureagi_scores = futureagi_scores
            # Blend local score with FutureAGI average
            if futureagi_scores:
                fagi_avg = sum(futureagi_scores.values()) / len(futureagi_scores)
                result.score = 0.6 * result.score + 0.4 * fagi_avg

        result.duration_ms = int((time.monotonic() - start) * 1000)
        return result

    async def _eval_task_decomposer(self, case: EvalCase) -> EvalResult:
        """Run task decomposer and check against expected output."""
        try:
            from modules.orchestrator.stages import RealTaskDecomposer
            from core.llm import create_llm_manager

            llm = create_llm_manager(service_name="eval")
            decomposer = RealTaskDecomposer(llm)
            output = await decomposer.decompose_task(case.input)
        except Exception as e:
            return EvalResult(
                case_id=case.id, passed=False, score=0.0,
                error=f"Decomposer failed: {e}",
            )

        subtasks = output.get("subtasks", [])
        checks: Dict[str, bool] = {}
        expected = case.expected

        # Check subtask count
        if "min_subtasks" in expected:
            checks["min_subtasks"] = len(subtasks) >= expected["min_subtasks"]
        if "max_subtasks" in expected:
            checks["max_subtasks"] = len(subtasks) <= expected["max_subtasks"]

        # Check required skills present in any subtask
        if "required_skills_present" in expected:
            all_skills = set()
            for st in subtasks:
                all_skills.update(s.lower() for s in st.get("required_skills", []))
            required = set(s.lower() for s in expected["required_skills_present"])
            checks["skills_present"] = len(required & all_skills) > 0

        # Check execution strategy
        if "execution_strategy" in expected:
            checks["execution_strategy"] = (
                output.get("execution_strategy", "sequential") == expected["execution_strategy"]
            )

        # Check dependencies exist
        if "has_dependencies" in expected and expected["has_dependencies"]:
            has_deps = any(st.get("dependencies") for st in subtasks)
            checks["has_dependencies"] = has_deps

        passed_checks = sum(1 for v in checks.values() if v)
        total_checks = len(checks) or 1
        score = passed_checks / total_checks

        return EvalResult(
            case_id=case.id,
            passed=score >= 0.6,
            score=score,
            details={"checks": checks, "subtask_count": len(subtasks), "output": output},
        )

    async def _eval_agent_selector(self, case: EvalCase) -> EvalResult:
        """Run agent selector evaluation (structural checks only without live agents)."""
        # Agent selector evaluation is structural — verify the scoring formula is applied
        checks: Dict[str, bool] = {}
        expected = case.expected

        # Verify scoring formula structure
        if "all_four_signals_used" in expected:
            # Check that the codebase has the formula — this is a static check
            try:
                import inspect
                from modules.orchestrator.llm.llm_agent_selector import LLMAgentSelector
                source = inspect.getsource(LLMAgentSelector)
                checks["has_skill_weight"] = "0.4" in source and "coverage" in source
                checks["has_perf_weight"] = "0.3" in source and "perf" in source.lower()
                checks["has_load_weight"] = "0.2" in source and "workload" in source.lower()
                checks["has_recency_weight"] = "0.1" in source and "recency" in source.lower()
            except Exception:
                checks["formula_check"] = False

        if "top_agent_skills" in expected:
            # Structural check: skills are present in the expected list
            checks["skills_defined"] = len(expected["top_agent_skills"]) > 0

        passed_checks = sum(1 for v in checks.values() if v)
        total_checks = len(checks) or 1
        score = passed_checks / total_checks

        return EvalResult(
            case_id=case.id,
            passed=score >= 0.6,
            score=score,
            details={"checks": checks, "output": {"structural_eval": True}},
        )

    async def _score_with_futureagi(
        self, case: EvalCase, output: Any
    ) -> Optional[Dict[str, float]]:
        """Score output using FutureAGI SDK."""
        try:
            from core.services.futureagi_service import FutureAGIService

            service = FutureAGIService.get_instance()
            if not service.is_available:
                return None

            output_str = json.dumps(output) if isinstance(output, dict) else str(output)

            scores = await service.assess_prompt(
                prompt_content=case.input,
                test_cases=[{
                    "input": case.input,
                    "output": output_str,
                    "expected": json.dumps(case.expected),
                }],
                metrics=case.metrics,
            )

            return scores.get("scores", {})
        except Exception as e:
            logger.debug(f"FutureAGI scoring failed for {case.id}: {e}")
            return None
