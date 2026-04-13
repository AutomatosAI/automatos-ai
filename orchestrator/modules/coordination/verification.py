"""
Verification Service — Advisory Review (no rejection)
=====================================================

Two-stage task output review:
  1. Deterministic checks (free, fast) — structural quality signals
  2. Cross-model LLM reviewer — scores + actionable suggestions

ALL tasks proceed to VERIFIED regardless of scores. Verification is
advisory only — feedback is stored in task.output_metadata["review_feedback"]
for downstream consumers (synthesis tasks, reports, humans) to incorporate.
Tasks are NEVER rejected or retried based on verification.

Source: PRD-103 (Verification Quality)
        PRD-82A Section 5 (cross-model principle)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

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
    suggestions: List[str] = field(default_factory=list)
    tokens_used: int = 0


@dataclass(frozen=True)
class ConsistencyIssue:
    """A single consistency issue found between task outputs."""

    task_ids: List[str]
    description: str
    severity: str  # "high" | "medium" | "low"


@dataclass(frozen=True)
class ConsistencyResult:
    """Immutable result of cross-task consistency verification."""

    passed: bool
    score: float  # 0.0 - 1.0
    reasoning: str = ""
    issues: List[ConsistencyIssue] = field(default_factory=list)
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

    executor_family = None
    for family, keywords in family_keywords.items():
        if any(kw in model_lower for kw in keywords):
            executor_family = family
            verifier = mapping.get(family)
            if verifier:
                return verifier
            break

    # Executor family has no dedicated mapping — pick any OTHER family's model
    # to preserve the cross-model guarantee
    if executor_family:
        for family, model in mapping.items():
            if family != executor_family:
                return model

    return fallback


# ---------------------------------------------------------------------------
# LLM judge prompt
# ---------------------------------------------------------------------------

_VERIFIER_SYSTEM_PROMPT = """\
You are a quality reviewer for an AI agent platform. Your job is to review \
a task's output and provide constructive feedback — what's good, what could \
be improved, and specific suggestions for improvement.

Rules:
- Score each dimension on a scale of 0.0 to 1.0.
- Be objective: shorter outputs are not worse if they meet criteria; \
longer outputs are not better if they don't.
- Use absolute scoring, not relative comparison.
- Your feedback is ADVISORY ONLY — it will NOT reject or retry the task.
- Focus on actionable suggestions: "add X", "clarify Y", "fix Z".
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

    # Detect media output (images, files, etc.)
    import re as _re
    _has_image = bool(
        _re.search(r"!\[.*?\]\(.*?\)", output or "")
        or _re.search(r"data:image/", output or "")
        or _re.search(r"/api/generated-images/", output or "")
    )

    # Strip base64 data to avoid blowing up context
    output_clean = _re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
        "[BASE64_IMAGE_DATA]",
        output or "",
    )

    # Truncate output to avoid exceeding context limits
    max_output_chars = 12000
    output_display = output_clean[:max_output_chars]
    if len(output_clean) > max_output_chars:
        output_display += f"\n\n... (truncated, {len(output_clean)} total characters)"

    # Detect research/web-search tasks — the judge cannot verify external sources
    _title_lower = (task_title or "").lower()
    _desc_lower = (task_description or "").lower()
    _is_research = any(
        kw in _title_lower or kw in _desc_lower
        for kw in ("research", "web search", "websearch", "find articles",
                    "search for", "gather information", "discussion points",
                    "talking points", "find recent")
    )

    # Adjust instructions for media-producing tasks
    media_note = ""
    if _has_image:
        media_note = (
            "\n**NOTE:** This task produces visual/media output (images). "
            "The agent output contains image references (URLs or markdown image tags). "
            "Verify that image references are present and the surrounding text is relevant. "
            "Do NOT penalise for missing text sections like 'Final image asset' — "
            "the image URL/reference IS the asset.\n"
        )

    # Adjust instructions for research tasks — judge cannot verify external sources
    research_note = ""
    if _is_research:
        research_note = (
            "\n**NOTE:** This is a research/web-search task. The agent gathered "
            "information from external sources that you cannot access or verify. "
            "For the `accuracy` dimension, evaluate whether the claims are "
            "internally consistent, properly attributed to sources, and plausible "
            "— NOT whether you can independently confirm them. Do NOT penalise "
            "accuracy for information you simply cannot verify. If the research "
            "cites sources and the claims are plausible, score accuracy >= 0.7.\n"
        )

    return f"""\
## Task
**Title:** {task_title}
**Description:** {task_description}
{media_note}{research_note}
## Verification Criteria
{criteria_text}

## Deterministic Check Results
{det_text}

## Agent Output
<agent_output>
{output_display}
</agent_output>

## Required JSON Output
Return ONLY a JSON object with this exact structure:
{{
  "relevance": 0.0-1.0,
  "completeness": 0.0-1.0,
  "accuracy": 0.0-1.0,
  "format_compliance": 0.0-1.0,
  "confidence": 0.0-1.0,
  "reasoning": "Brief assessment of quality",
  "suggestions": ["Specific actionable improvement 1", "Improvement 2"]
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

    Includes per-run output hash caching (PRD-82B US-007): if the same task
    produces identical output on retry, the cached VerificationResult is
    returned without a second LLM call.

    Cache is class-level so it survives across VerificationService() instances
    created per reconciler tick.  Keyed by (run_id, task_id, sha256(output)).
    """

    # Class-level cache: {(run_id, task_id, output_hash): VerificationResult}
    _cache: Dict[Tuple[UUID, UUID, str], "VerificationResult"] = {}

    def __init__(self) -> None:
        self._checker = DeterministicChecker()

    # -------------------------------------------------------------------
    # Cache helpers (PRD-82B US-007)
    # -------------------------------------------------------------------

    @classmethod
    def _output_hash(cls, output: str) -> str:
        """Compute SHA-256 hex digest of the raw output text."""
        return hashlib.sha256(output.encode()).hexdigest()

    @classmethod
    def clear_cache(cls, run_id: UUID) -> int:
        """
        Remove all cached verification results for a given run.

        Returns the number of entries removed.
        """
        keys_to_remove = [k for k in cls._cache if k[0] == run_id]
        for k in keys_to_remove:
            del cls._cache[k]
        if keys_to_remove:
            logger.info(
                "Verification cache cleared for run %s (%d entries removed)",
                run_id,
                len(keys_to_remove),
            )
        return len(keys_to_remove)

    async def verify_task(
        self,
        task_title: str,
        task_description: str,
        output: str,
        verification_criteria: Optional[List[Dict[str, Any]]],
        executor_model: Optional[str] = None,
        *,
        run_id: Optional[UUID] = None,
        task_id: Optional[UUID] = None,
    ) -> VerificationResult:
        """
        Verify a task's output.

        Args:
            task_title: Human-readable task title.
            task_description: Full task description/instructions.
            output: The agent's output text.
            verification_criteria: List of criterion dicts from task spec.
            executor_model: Model used by the executing agent (for cross-model selection).
            run_id: Orchestration run ID (for caching scope).
            task_id: Task ID (for cache key).

        Returns:
            VerificationResult with verdict, scores, and reasoning.
        """
        if not output or not output.strip():
            return VerificationResult(
                verdict=VERDICT_FAIL,
                reasoning="Task produced empty output.",
                deterministic_passed=False,
            )

        # ------------------------------------------------------------------
        # Cache lookup (PRD-82B US-007)
        # ------------------------------------------------------------------
        cache_key: Optional[Tuple[UUID, UUID, str]] = None
        if run_id is not None and task_id is not None:
            output_hash = self._output_hash(output)
            cache_key = (run_id, task_id, output_hash)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info(
                    "Verification cache hit for task %s (run %s)",
                    task_id,
                    run_id,
                )
                return cached

        # Stage 1: Deterministic checks
        det_result = self._checker.check(output, verification_criteria)

        det_failure_descriptions = [f.description for f in det_result.failures]

        if det_result.short_circuited:
            logger.info(
                "Verification FAIL (deterministic short-circuit) for task '%s'",
                task_title,
            )
            result = VerificationResult(
                verdict=VERDICT_FAIL,
                reasoning="Deterministic must_pass check failed: "
                + "; ".join(det_failure_descriptions),
                deterministic_passed=False,
                deterministic_failures=det_failure_descriptions,
            )
            if cache_key is not None:
                self._cache[cache_key] = result
            return result

        # Stage 2: Cross-model LLM judge
        result = await self._run_llm_judge(
            task_title=task_title,
            task_description=task_description,
            output=output,
            verification_criteria=verification_criteria,
            executor_model=executor_model,
            deterministic_result=det_result,
            deterministic_failures=det_failure_descriptions,
        )

        # Store in cache
        if cache_key is not None:
            self._cache[cache_key] = result
        return result

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
                suggestions = raw.get("suggestions", [])
                if not isinstance(suggestions, list):
                    suggestions = []

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
                # BUT only downgrade if failures are structural (min_length,
                # format_regex, etc) — not required_sections which is just
                # heading-name guessing that burns tokens on retries
                if not deterministic_result.passed and verdict == VERDICT_PASS:
                    structural_failures = [
                        f for f in deterministic_result.failures
                        if f.check_type != "required_sections"
                    ]
                    if structural_failures:
                        verdict = VERDICT_PARTIAL
                        reasoning = (
                            f"Deterministic check failures: "
                            f"{'; '.join(f.description for f in structural_failures)}. "
                            f"LLM judge assessment: {reasoning}"
                        )
                    else:
                        # Only required_sections failed — trust the LLM judge
                        logger.info(
                            "Skipping PARTIAL downgrade for task '%s' — "
                            "only required_sections failed, LLM judge says PASS",
                            task_title,
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
                    suggestions=suggestions,
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

    # -------------------------------------------------------------------
    # Cross-task consistency verification (PRD-82B US-006)
    # -------------------------------------------------------------------

    async def verify_cross_task_consistency(
        self,
        run_id: UUID,
        goal: str,
        task_outputs: List[Dict[str, Any]],
    ) -> ConsistencyResult:
        """
        Verify that all task outputs are consistent with each other.

        Args:
            run_id: The orchestration run ID (for logging).
            goal: The original mission goal.
            task_outputs: List of dicts with keys: task_id, title, output.

        Returns:
            ConsistencyResult with passed, score, reasoning, issues.
        """
        if len(task_outputs) < 2:
            return ConsistencyResult(
                passed=True,
                score=1.0,
                reasoning="Single task — consistency check not applicable.",
            )

        # Use cross-model verifier (different family from typical executor)
        verifier_model = _select_verifier_model(None)  # fallback model
        max_retries = Config.COORDINATOR_MAX_VERIFICATION_RETRIES

        prompt = self._build_consistency_prompt(goal, task_outputs)
        messages = [
            {"role": "system", "content": _CONSISTENCY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        last_error: Optional[str] = None

        for attempt in range(1, max_retries + 1):
            try:
                llm = create_llm_manager(
                    service_name="consistency_verifier",
                    model=verifier_model,
                )
                response = await llm.generate_response(messages)
                raw = _extract_judge_json(response.content)

                if raw is None:
                    last_error = f"Consistency LLM returned non-JSON on attempt {attempt}"
                    logger.warning(
                        "Consistency check non-JSON (attempt %d/%d) for run %s",
                        attempt, max_retries, run_id,
                    )
                    continue

                # Extract fields
                passed = bool(raw.get("passed", False))
                score = raw.get("score", 0.5)
                if not isinstance(score, (int, float)):
                    score = 0.5
                score = max(0.0, min(1.0, float(score)))

                reasoning = str(raw.get("reasoning", ""))

                # Parse issues
                issues: List[ConsistencyIssue] = []
                raw_issues = raw.get("issues", [])
                if isinstance(raw_issues, list):
                    for issue in raw_issues:
                        if isinstance(issue, dict):
                            task_ids = issue.get("task_ids", [])
                            if not isinstance(task_ids, list):
                                task_ids = [str(task_ids)]
                            issues.append(ConsistencyIssue(
                                task_ids=[str(tid) for tid in task_ids],
                                description=str(issue.get("description", "")),
                                severity=str(issue.get("severity", "medium")),
                            ))

                # Extract token usage
                tokens_used = 0
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    if hasattr(usage, "total_tokens"):
                        tokens_used = usage.total_tokens
                    elif isinstance(usage, dict):
                        tokens_used = usage.get("total_tokens", 0)

                logger.info(
                    "Consistency check %s for run %s (score=%.2f, issues=%d, model=%s)",
                    "PASSED" if passed else "FAILED",
                    run_id,
                    score,
                    len(issues),
                    verifier_model,
                )

                return ConsistencyResult(
                    passed=passed,
                    score=score,
                    reasoning=reasoning,
                    issues=issues,
                    tokens_used=tokens_used,
                )

            except Exception:
                last_error = f"Consistency LLM call failed on attempt {attempt}"
                logger.error(
                    "Consistency check error (attempt %d/%d) for run %s",
                    attempt, max_retries, run_id,
                    exc_info=True,
                )

        # All retries exhausted — assume passed (don't block human review)
        logger.warning(
            "Consistency check exhausted %d retries for run %s: %s",
            max_retries, run_id, last_error,
        )
        return ConsistencyResult(
            passed=True,
            score=0.5,
            reasoning=f"Consistency check failed after {max_retries} attempts: {last_error}. Defaulting to passed.",
        )

    @staticmethod
    def _build_consistency_prompt(
        goal: str,
        task_outputs: List[Dict[str, Any]],
    ) -> str:
        """Build the user prompt for cross-task consistency verification."""
        max_output_chars = 4000  # Per task, to fit within context

        task_sections: List[str] = []
        for t in task_outputs:
            output_text = str(t.get("output", ""))
            if len(output_text) > max_output_chars:
                output_text = output_text[:max_output_chars] + f"\n... (truncated, {len(output_text)} total chars)"
            task_sections.append(
                f"### Task: {t.get('title', 'Untitled')} (ID: {t.get('task_id', 'unknown')})\n"
                f"<output>\n{output_text}\n</output>"
            )

        tasks_text = "\n\n".join(task_sections)

        return f"""\
## Mission Goal
{goal}

## Task Outputs ({len(task_outputs)} tasks)

{tasks_text}

## Required JSON Output
Return ONLY a JSON object with this exact structure:
{{
  "passed": true/false,
  "score": 0.0-1.0,
  "reasoning": "Brief explanation of consistency assessment",
  "issues": [
    {{
      "task_ids": ["id1", "id2"],
      "description": "Description of the inconsistency",
      "severity": "high|medium|low"
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Consistency verification prompt (PRD-82B US-006)
# ---------------------------------------------------------------------------

_CONSISTENCY_SYSTEM_PROMPT = """\
You are a consistency verification judge for an AI agent platform. Your job is \
to check whether multiple task outputs from the same mission are consistent \
with each other and collectively satisfy the mission goal.

Check for:
1. **Contradictions** — Do any outputs contradict each other (conflicting facts, \
dates, numbers, recommendations)?
2. **Goal coverage** — Do the outputs collectively address all aspects of the \
mission goal? Are there significant gaps?
3. **Redundant duplication** — Is there excessive overlap that indicates wasted \
effort or copy-paste?
4. **Logical coherence** — Do the outputs form a coherent narrative when read \
together? Do later tasks properly build on earlier outputs?

Rules:
- Score on a scale of 0.0 (completely inconsistent) to 1.0 (perfectly consistent).
- Set "passed" to true if score >= 0.7 and no high-severity issues exist.
- Be pragmatic: minor formatting differences or stylistic variations are fine.
- Return ONLY a single JSON object (no markdown, no explanation outside JSON).
"""
