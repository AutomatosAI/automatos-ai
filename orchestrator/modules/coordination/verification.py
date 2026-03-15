"""
Verification Service — PRD-82A Sequential Mission Coordinator
==============================================================

Two-stage task output verification:
  1. Deterministic checks (free, fast) — short-circuit on must_pass failure
  2. Cross-model LLM judge — different model family than executor

Verdicts: pass | fail | partial

Source: PRD-103 (Verification Quality)
        PRD-82A Section 5 (cross-model principle), Section 11 (retry guardrails)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import Config
from core.llm import create_llm_manager
from modules.coordination.deterministic_checks import DeterministicChecker, DeterministicResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_PARTIAL = "partial"

# Scoring dimensions returned by the LLM judge
SCORE_DIMENSIONS = ("relevance", "completeness", "accuracy", "format_compliance")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    """Immutable result of task verification."""

    verdict: str  # pass | fail | partial
    scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 1.0
    deterministic_passed: bool = True
    deterministic_failures: List[str] = field(default_factory=list)
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Cross-model selection
# ---------------------------------------------------------------------------


def _parse_verifier_model_mapping() -> Dict[str, str]:
    """Parse the COORDINATOR_VERIFIER_MODEL_MAPPING config string into a dict."""
    raw = Config.COORDINATOR_VERIFIER_MODEL_MAPPING
    mapping: Dict[str, str] = {}
    if not raw:
        return mapping
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            family, model = pair.split("=", 1)
            mapping[family.strip().lower()] = model.strip()
    return mapping


def _select_verifier_model(executor_model: Optional[str]) -> str:
    """
    Select a verifier model from a different family than the executor.

    Detection: look for family keywords in the executor model string.
    Falls back to COORDINATOR_VERIFIER_FALLBACK_MODEL.
    """
    fallback = Config.COORDINATOR_VERIFIER_FALLBACK_MODEL
    if not executor_model:
        return fallback

    model_lower = executor_model.lower()
    mapping = _parse_verifier_model_mapping()

    # Match executor model to a family
    family_keywords = {
        "anthropic": ["claude", "anthropic"],
        "openai": ["gpt", "openai", "o1", "o3", "o4"],
        "google": ["gemini", "google", "palm"],
        "deepseek": ["deepseek"],
        "meta": ["llama", "meta"],
    }

    for family, keywords in family_keywords.items():
        if any(kw in model_lower for kw in keywords):
            verifier = mapping.get(family)
            if verifier:
                return verifier
            break

    return fallback


# ---------------------------------------------------------------------------
# LLM judge prompt
# ---------------------------------------------------------------------------

_VERIFIER_SYSTEM_PROMPT = """\
You are a verification judge for an AI agent platform. Your job is to evaluate \
whether a task's output meets its success criteria.

Rules:
- Score each dimension on a scale of 0.0 to 1.0.
- Be objective: shorter outputs are not worse if they meet criteria; \
longer outputs are not better if they don't.
- Use absolute scoring, not relative comparison.
- Return ONLY a single JSON object (no markdown, no explanation).
"""


def _build_judge_prompt(
    *,
    task_title: str,
    task_description: str,
    output: str,
    verification_criteria: Optional[List[Dict[str, Any]]],
    deterministic_result: DeterministicResult,
) -> str:
    """Build the user prompt for the LLM judge."""
    criteria_text = "None specified."
    if verification_criteria:
        criteria_lines = []
        for i, c in enumerate(verification_criteria, 1):
            must = " [MUST PASS]" if c.get("must_pass") else ""
            criteria_lines.append(
                f"  {i}. {c.get('type', 'unknown')}: {c.get('value', '')}{must}"
            )
        criteria_text = "\n".join(criteria_lines)

    det_text = "All passed." if deterministic_result.passed else (
        "Failures:\n" + "\n".join(
            f"  - [{f.check_type}] {f.description}"
            for f in deterministic_result.failures
        )
    )

    # Truncate output to avoid exceeding context limits
    max_output_chars = 12000
    output_display = output[:max_output_chars]
    if len(output) > max_output_chars:
        output_display += f"\n\n... (truncated, {len(output)} total characters)"

    return f"""\
## Task
**Title:** {task_title}
**Description:** {task_description}

## Verification Criteria
{criteria_text}

## Deterministic Check Results
{det_text}

## Agent Output
{output_display}

## Required JSON Output
Return ONLY a JSON object with this exact structure:
{{
  "relevance": 0.0-1.0,
  "completeness": 0.0-1.0,
  "accuracy": 0.0-1.0,
  "format_compliance": 0.0-1.0,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your assessment"
}}
"""


def _extract_judge_json(content: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM judge response (handles markdown blocks)."""
    import re

    if not content:
        return None

    # Try markdown block first
    block_match = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL)
    text = block_match.group(1).strip() if block_match else content.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: find first { ... } block
    brace_match = re.search(r"\{.*\}", content, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# VerificationService
# ---------------------------------------------------------------------------


class VerificationService:
    """
    Verifies task outputs using deterministic checks + cross-model LLM judge.

    Stateless — all data comes from arguments.
    """

    def __init__(self) -> None:
        self._checker = DeterministicChecker()

    async def verify_task(
        self,
        task_title: str,
        task_description: str,
        output: str,
        verification_criteria: Optional[List[Dict[str, Any]]],
        executor_model: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a task's output.

        Args:
            task_title: Human-readable task title.
            task_description: Full task description/instructions.
            output: The agent's output text.
            verification_criteria: List of criterion dicts from task spec.
            executor_model: Model used by the executing agent (for cross-model selection).

        Returns:
            VerificationResult with verdict, scores, and reasoning.
        """
        if not output or not output.strip():
            return VerificationResult(
                verdict=VERDICT_FAIL,
                reasoning="Task produced empty output.",
                deterministic_passed=False,
            )

        # Stage 1: Deterministic checks
        det_result = self._checker.check(output, verification_criteria)

        det_failure_descriptions = [f.description for f in det_result.failures]

        if det_result.short_circuited:
            logger.info(
                "Verification FAIL (deterministic short-circuit) for task '%s'",
                task_title,
            )
            return VerificationResult(
                verdict=VERDICT_FAIL,
                reasoning="Deterministic must_pass check failed: "
                + "; ".join(det_failure_descriptions),
                deterministic_passed=False,
                deterministic_failures=det_failure_descriptions,
            )

        # Stage 2: Cross-model LLM judge
        return await self._run_llm_judge(
            task_title=task_title,
            task_description=task_description,
            output=output,
            verification_criteria=verification_criteria,
            executor_model=executor_model,
            deterministic_result=det_result,
            deterministic_failures=det_failure_descriptions,
        )

    async def _run_llm_judge(
        self,
        *,
        task_title: str,
        task_description: str,
        output: str,
        verification_criteria: Optional[List[Dict[str, Any]]],
        executor_model: Optional[str],
        deterministic_result: DeterministicResult,
        deterministic_failures: List[str],
    ) -> VerificationResult:
        """Run the cross-model LLM judge with retry logic."""
        verifier_model = _select_verifier_model(executor_model)
        max_retries = Config.COORDINATOR_MAX_VERIFICATION_RETRIES

        prompt = _build_judge_prompt(
            task_title=task_title,
            task_description=task_description,
            output=output,
            verification_criteria=verification_criteria,
            deterministic_result=deterministic_result,
        )

        messages = [
            {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        last_error: Optional[str] = None

        for attempt in range(1, max_retries + 1):
            try:
                llm = create_llm_manager(
                    service_name="verifier",
                    model=verifier_model,
                )
                response = await llm.generate_response(messages)
                raw = _extract_judge_json(response.content)

                if raw is None:
                    last_error = f"LLM judge returned non-JSON on attempt {attempt}"
                    logger.warning(
                        "Verification LLM judge returned non-JSON (attempt %d/%d) for task '%s'",
                        attempt, max_retries, task_title,
                    )
                    continue

                # Extract scores
                scores: Dict[str, float] = {}
                for dim in SCORE_DIMENSIONS:
                    val = raw.get(dim)
                    if isinstance(val, (int, float)):
                        scores[dim] = max(0.0, min(1.0, float(val)))
                    else:
                        scores[dim] = 0.5  # Default if missing

                confidence = raw.get("confidence", 1.0)
                if not isinstance(confidence, (int, float)):
                    confidence = 1.0
                confidence = max(0.0, min(1.0, float(confidence)))

                reasoning = str(raw.get("reasoning", ""))

                # Extract token usage from response if available
                tokens_used = 0
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    if hasattr(usage, "total_tokens"):
                        tokens_used = usage.total_tokens
                    elif isinstance(usage, dict):
                        tokens_used = usage.get("total_tokens", 0)

                # Determine verdict
                verdict = self._compute_verdict(scores, confidence)

                # If deterministic checks had non-must_pass failures, factor in
                if not deterministic_result.passed and verdict == VERDICT_PASS:
                    # Downgrade to partial if deterministic checks failed
                    # but weren't must_pass
                    verdict = VERDICT_PARTIAL
                    reasoning = (
                        f"Deterministic check failures: {'; '.join(deterministic_failures)}. "
                        f"LLM judge assessment: {reasoning}"
                    )

                logger.info(
                    "Verification %s for task '%s' (model: %s, scores: %s, confidence: %.2f)",
                    verdict.upper(),
                    task_title,
                    verifier_model,
                    scores,
                    confidence,
                )

                return VerificationResult(
                    verdict=verdict,
                    scores=scores,
                    reasoning=reasoning,
                    confidence=confidence,
                    deterministic_passed=deterministic_result.passed,
                    deterministic_failures=deterministic_failures,
                    tokens_used=tokens_used,
                )

            except Exception:
                last_error = f"LLM judge call failed on attempt {attempt}"
                logger.error(
                    "Verification LLM judge error (attempt %d/%d) for task '%s'",
                    attempt, max_retries, task_title,
                    exc_info=True,
                )

        # All retries exhausted — return partial (escalate to human)
        logger.warning(
            "Verification LLM judge exhausted %d retries for task '%s': %s",
            max_retries, task_title, last_error,
        )
        return VerificationResult(
            verdict=VERDICT_PARTIAL,
            reasoning=f"LLM judge failed after {max_retries} attempts: {last_error}",
            deterministic_passed=deterministic_result.passed,
            deterministic_failures=deterministic_failures,
        )

    @staticmethod
    def _compute_verdict(
        scores: Dict[str, float],
        confidence: float,
    ) -> str:
        """
        Determine verdict from scores and confidence.

        Rules (PRD-103 Section 5.4):
          - If any score < fail_threshold → FAIL
          - If confidence < escalation_threshold → PARTIAL (escalate to human)
          - If all scores >= pass_threshold → PASS
          - Otherwise → PARTIAL
        """
        pass_threshold = Config.COORDINATOR_VERIFICATION_PASS_THRESHOLD
        fail_threshold = Config.COORDINATOR_VERIFICATION_FAIL_THRESHOLD
        confidence_escalation = Config.COORDINATOR_VERIFICATION_CONFIDENCE_ESCALATION

        score_values = list(scores.values())

        if any(s < fail_threshold for s in score_values):
            return VERDICT_FAIL

        if confidence < confidence_escalation:
            return VERDICT_PARTIAL

        if all(s >= pass_threshold for s in score_values):
            return VERDICT_PASS

        return VERDICT_PARTIAL
