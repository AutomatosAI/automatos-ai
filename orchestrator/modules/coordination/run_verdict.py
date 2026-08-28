"""
Run-level Verdict Service -- PRD-204 S6
=======================================

Extends the PRD-200 per-task judge (``modules/coordination/verification.py``)
to the RUN level: score the whole launched unit's output against the watch's
``success_criteria`` on a 6-dimension business rubric.

Dimensions (each 0-1, weighted mean -> ``watch.final_score``):
- ``business_usefulness``, ``completeness``, ``evidence_quality``,
  ``clarity``, ``actionability`` -- judged by the cross-model LLM;
- ``reliability`` -- rule-based mechanics folded in from
  ``PlaybookQualityService`` heuristics (tool failures / retries / step
  status over ``step_results``) for playbooks, and a task-state analogue for
  missions. Deliberately NOT an LLM opinion: the mechanics are already in
  the rows.

Scale (PRD-204 Section 8 Q8): 0-1 internal EVERYWHERE; only the notification
display edge (``watch_notifications.format_score_display``) formats x10.

Reuse (not re-derivation):
- cross-model selection: ``verification._select_verifier_model``;
- judge JSON extraction: ``verification._extract_judge_json``;
- output-hash caching: same class-level cache pattern as
  ``VerificationService`` -- keyed ``(watch_id, sha256(output))`` so a
  rescore of identical output (e.g. rerun that produced the same thing)
  costs zero LLM calls.

LLM cost attribution: the manager is created with ``request_type='watch'``
and ``_tracking_ctx['execution_id'] = 'watch-<id>'`` -- the recipe-executor
idiom -- so ``llm_usage`` shows supervision cost per watch, WITHOUT
polluting the watched execution's own rollup (which the S7 rerun estimate
sums).

Judge failure is fail-soft: after retries the verdict comes back
``judge_failed=True`` with ``score=None`` -- the S10 decision step then
closes by outcome (v1 semantics) instead of blocking the watch forever.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from config import Config
from modules.coordination.verification import (
    _extract_judge_json,
    _select_verifier_model,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

# The five judged dimensions + the mechanics-derived one. Weighted mean with
# equal weights (one knob, one place -- adjust here if the rubric evolves).
LLM_DIMENSIONS: Tuple[str, ...] = (
    "business_usefulness",
    "completeness",
    "evidence_quality",
    "clarity",
    "actionability",
)
MECHANICS_DIMENSION = "reliability"
ALL_DIMENSIONS: Tuple[str, ...] = LLM_DIMENSIONS + (MECHANICS_DIMENSION,)
DIMENSION_WEIGHTS: Dict[str, float] = {dim: 1.0 / len(ALL_DIMENSIONS) for dim in ALL_DIMENSIONS}

# Output text cap for the judge prompt (same guard as the task judge).
MAX_OUTPUT_CHARS = 12000
# Per-task / per-step excerpt cap when composing the run output bundle.
EXCERPT_CHARS = 600


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOutputBundle:
    """What the judge sees for one run-shaped target."""

    text: str                       # composed run output
    kind: str                       # 'mission' | 'playbook_execution' | 'board_task'
    terminal_state: str
    mechanics_reliability: float    # 0-1 rule-based
    executor_model: Optional[str] = None
    empty: bool = False


@dataclass(frozen=True)
class RunVerdict:
    """Immutable run-level verdict. ``score`` is None only on judge failure."""

    score: Optional[float]
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    caveats: List[str] = field(default_factory=list)
    confidence: float = 1.0
    tokens_used: int = 0
    cached: bool = False
    judge_failed: bool = False
    output_hash: str = ""

    def passes(self, threshold: float) -> bool:
        """At-or-above passes (0.79 fails a 0.80 bar; 0.80 passes it)."""
        return self.score is not None and self.score >= threshold

    def as_text(self) -> str:
        """The ``final_verdict`` text: score + reasoning + caveats."""
        parts: List[str] = []
        if self.judge_failed:
            parts.append(
                "Run-level scoring unavailable (judge failed after retries); "
                "verdict recorded by outcome."
            )
        elif self.score is not None:
            dims = ", ".join(
                f"{d}={self.dimension_scores.get(d, 0.0):.2f}" for d in ALL_DIMENSIONS
            )
            parts.append(f"Run score {self.score:.2f} ({dims}).")
        if self.reasoning:
            parts.append(self.reasoning.strip())
        if self.caveats:
            parts.append("Caveats: " + "; ".join(str(c) for c in self.caveats) + ".")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Pure scoring math (unit-testable without a DB or LLM)
# ---------------------------------------------------------------------------


def weighted_mean(dimension_scores: Dict[str, float]) -> float:
    """Weighted mean over ALL_DIMENSIONS, missing dims scored 0."""
    total = 0.0
    for dim in ALL_DIMENSIONS:
        raw = dimension_scores.get(dim, 0.0)
        clamped = max(0.0, min(1.0, float(raw)))
        total += clamped * DIMENSION_WEIGHTS[dim]
    return round(total, 4)


def mission_mechanics(task_states: List[Dict[str, Any]], run_state: str) -> float:
    """Mission analogue of PlaybookQualityService reliability: task success
    ratio + retry drag + outcome. Pure over ``[{state, attempts}]`` rows."""
    status_score = 1.0 if run_state == "completed" else 0.0
    if not task_states:
        return round(0.4 * status_score + 0.6 * 0.5, 4)

    done = sum(1 for t in task_states if t.get("state") in ("verified", "completed"))
    failed = sum(1 for t in task_states if t.get("state") == "failed")
    counted = max(1, done + failed)
    success_ratio = done / counted

    total = len(task_states)
    retries = sum(max(0, int(t.get("attempts") or 1) - 1) for t in task_states)
    retry_score = max(0.0, 1.0 - (retries / total) * 0.25)

    return round(0.4 * status_score + 0.3 * success_ratio + 0.3 * retry_score, 4)


def _hash_output(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def _compact(value: Any, limit: int = EXCERPT_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) > limit:
        return text[:limit] + f"... (truncated, {len(text)} chars)"
    return text


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_RUN_JUDGE_SYSTEM_PROMPT = """\
You are a business-quality reviewer for an AI agent platform. You review the
FINAL OUTPUT of a completed unit of work (a mission or a playbook run)
against the intent it was launched with, from the perspective of the person
who asked for it.

Rules:
- Score each dimension 0.0 to 1.0, absolute (not relative to other runs).
- business_usefulness: would the requester act on this as delivered?
- completeness: does it cover everything the intent asked for?
- evidence_quality: are claims grounded/attributed rather than asserted?
- clarity: is it immediately understandable without cleanup?
- actionability: are next steps / conclusions concrete and usable?
- Shorter is not worse; longer is not better.
- List concrete caveats the requester should know before trusting the output.
- Return ONLY a single JSON object (no markdown, no prose outside JSON).
"""


def build_run_judge_prompt(
    *,
    success_criteria: str,
    bundle: RunOutputBundle,
) -> str:
    """User prompt for the run-level judge. Pure -- golden-testable."""
    output_display = bundle.text[:MAX_OUTPUT_CHARS]
    if len(bundle.text) > MAX_OUTPUT_CHARS:
        output_display += f"\n... (truncated, {len(bundle.text)} total characters)"

    return f"""\
## What was asked (success criteria / intent)
{success_criteria}

## What ran
A {bundle.kind} that reached terminal state '{bundle.terminal_state}'.
Mechanical reliability (computed from step/task records, for context only --
do NOT return a reliability score): {bundle.mechanics_reliability:.2f}

## Run output
<run_output>
{output_display}
</run_output>

## Required JSON output
Return ONLY a JSON object with this exact structure:
{{
  "business_usefulness": 0.0-1.0,
  "completeness": 0.0-1.0,
  "evidence_quality": 0.0-1.0,
  "clarity": 0.0-1.0,
  "actionability": 0.0-1.0,
  "confidence": 0.0-1.0,
  "reasoning": "One-paragraph assessment against the intent",
  "caveats": ["Concrete caveat 1", "Caveat 2"]
}}
"""


# ---------------------------------------------------------------------------
# RunVerdictService
# ---------------------------------------------------------------------------


def _default_llm_factory(watch, verifier_model: str):
    """Build the judge LLM with watch cost attribution (recipe-executor
    idiom: set ``_tracking_ctx`` execution_id + request_type)."""
    from core.llm import create_llm_manager

    llm = create_llm_manager(
        service_name="watch_verdict",
        model=verifier_model,
        workspace_id=getattr(watch, "workspace_id", None),
        request_type="watch",
    )
    if hasattr(llm, "_tracking_ctx"):
        llm._tracking_ctx["request_type"] = "watch"
        llm._tracking_ctx["execution_id"] = f"watch-{getattr(watch, 'id', 'unknown')}"
    return llm


class RunVerdictService:
    """Scores a watch's run-level output. Stateless besides the class cache."""

    # {(watch_id_str, output_hash): RunVerdict} -- VerificationService pattern.
    _cache: Dict[Tuple[str, str], RunVerdict] = {}

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    @classmethod
    def clear_cache(cls, watch_id) -> int:
        keys = [k for k in cls._cache if k[0] == str(watch_id)]
        for k in keys:
            del cls._cache[k]
        return len(keys)

    # ------------------------------------------------------------------
    # Output collection (cheap DB reads, no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def collect_run_output(db: Session, watch) -> Optional[RunOutputBundle]:
        """Compose the judgeable output for the watch's CURRENT target.

        Returns None when the target row is gone (caller parks the watch).
        """
        target_type = getattr(watch, "target_type", None)
        if target_type == "mission":
            return RunVerdictService._collect_mission(db, watch)
        if target_type == "playbook_execution":
            return RunVerdictService._collect_playbook(db, watch)
        if target_type == "board_task":
            return RunVerdictService._collect_board_task(db, watch)
        return None

    @staticmethod
    def _collect_mission(db: Session, watch) -> Optional[RunOutputBundle]:
        from core.models.orchestration import OrchestrationRun, OrchestrationTask

        run = (
            db.query(OrchestrationRun)
            .filter(OrchestrationRun.id == watch.target_id)
            .first()
        )
        if run is None:
            return None

        tasks = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run.id)
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        sections: List[str] = []
        summary = getattr(run, "output_summary", None)
        if summary:
            sections.append("### Mission output summary\n" + _compact(summary, 4000))
        for t in tasks:
            output = getattr(t, "output", None)
            if output:
                sections.append(
                    f"### Task: {getattr(t, 'title', '')} "
                    f"[{getattr(t, 'state', '')}]\n{_compact(output)}"
                )

        task_states = [
            {"state": getattr(t, "state", None), "attempts": getattr(t, "attempt_number", 1) or 1}
            for t in tasks
        ]
        mechanics = mission_mechanics(task_states, getattr(run, "state", ""))
        text = "\n\n".join(sections)
        return RunOutputBundle(
            text=text,
            kind="mission",
            terminal_state=getattr(run, "state", "unknown"),
            mechanics_reliability=mechanics,
            executor_model=None,  # multi-agent run: no single executor family
            empty=not text.strip(),
        )

    @staticmethod
    def _collect_board_task(db: Session, watch) -> Optional[RunOutputBundle]:
        """PRD-224 US-002: compose a board task's recorded output for the SAME
        run-level judge missions/playbooks use -- result (+ any review feedback,
        or the error on a failed task). Returns None when the task row is gone
        (caller parks the watch), mirroring the mission/playbook collectors.
        """
        from core.models.core import BoardTask

        try:
            task_id = int(watch.target_id)
        except (TypeError, ValueError):
            return None
        task = db.query(BoardTask).filter(BoardTask.id == task_id).first()
        if task is None:
            return None

        status = getattr(task, "status", "") or ""
        sections: List[str] = []
        result = getattr(task, "result", None)
        if result:
            sections.append("### Result\n" + _compact(result, 8000))
        review_feedback = getattr(task, "review_feedback", None)
        if review_feedback:
            sections.append("### Review feedback\n" + _compact(review_feedback, 2000))
        error_message = getattr(task, "error_message", None)
        if error_message:
            sections.append("### Error\n" + _compact(error_message, 1000))

        # Rule-based reliability (the MECHANICS dimension, not an LLM opinion):
        # a 'done' task ran clean, 'failed' is a hard mechanical miss, anything
        # else terminal-with-output (e.g. reviewed) sits neutral -- the judge
        # scores the content quality separately.
        if status == "done":
            mechanics = 1.0
        elif status == "failed":
            mechanics = 0.0
        else:
            mechanics = 0.5

        text = "\n\n".join(sections)
        return RunOutputBundle(
            text=text,
            kind="board_task",
            terminal_state=status or "unknown",
            mechanics_reliability=mechanics,
            executor_model=None,  # single-agent ticket: executor known at run time
            empty=not text.strip(),
        )

    @staticmethod
    def _collect_playbook(db: Session, watch) -> Optional[RunOutputBundle]:
        from core.models.core import RecipeExecution

        execution = (
            db.query(RecipeExecution)
            .filter(RecipeExecution.execution_id == watch.target_id)
            .first()
        )
        if execution is None:
            return None

        output_data = execution.output_data or {}
        step_results = execution.step_results or []

        sections: List[str] = []
        final_output = output_data.get("final_output")
        if final_output:
            sections.append("### Final output\n" + _compact(final_output, 8000))
        if step_results:
            compact_steps = [
                {
                    "order": sr.get("order"),
                    "name": sr.get("agent_name") or sr.get("name"),
                    "status": sr.get("status"),
                    "error": (str(sr.get("error"))[:200] if sr.get("error") else None),
                    "retries": sr.get("retries", 0),
                }
                for sr in step_results
                if isinstance(sr, dict)
            ]
            sections.append("### Step record\n" + _compact(compact_steps, 2000))
        if execution.error_message:
            sections.append("### Error\n" + _compact(execution.error_message, 1000))

        # Reliability: reuse PlaybookQualityService's rule-based heuristics
        # over step_results (PRD-204 S6 -- reuse, don't re-derive). The
        # private call is deliberate and guarded; 0.5 = neutral on failure.
        try:
            from core.services.playbook_quality_service import PlaybookQualityService

            mechanics = float(
                PlaybookQualityService(db)._assess_reliability(execution, None)
            )
        except Exception:
            logger.warning(
                "[RunVerdict] reliability heuristic failed for %s -- neutral 0.5",
                watch.target_id,
                exc_info=True,
            )
            mechanics = 0.5

        # Cross-model guarantee where cheaply knowable: the run's primary
        # model from the llm_usage rollup.
        executor_model: Optional[str] = None
        try:
            from services.report_service import compute_execution_metrics

            executor_model = compute_execution_metrics(
                db,
                execution.workspace_id,
                execution_id=execution.execution_id,
            ).get("model")
        except Exception:
            logger.debug("[RunVerdict] executor model lookup failed", exc_info=True)

        text = "\n\n".join(sections)
        return RunOutputBundle(
            text=text,
            kind="playbook_execution",
            terminal_state=execution.status or "unknown",
            mechanics_reliability=max(0.0, min(1.0, mechanics)),
            executor_model=executor_model,
            empty=not text.strip(),
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def score_run(
        self,
        db: Session,
        watch,
        *,
        bundle: Optional[RunOutputBundle] = None,
        llm_factory: Optional[Callable[[Any, str], Any]] = None,
    ) -> Optional[RunVerdict]:
        """Score the watch's current target output. None = target missing.

        ``bundle`` and ``llm_factory`` are injection seams for tests (judge
        always stubbed; CI never calls a live model from this suite).
        """
        if bundle is None:
            bundle = self.collect_run_output(db, watch)
        if bundle is None:
            return None

        output_hash = _hash_output(bundle.text)
        cache_key = (str(getattr(watch, "id", "")), output_hash)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("[RunVerdict] cache hit for watch %s", watch.id)
            return RunVerdict(
                score=cached.score,
                dimension_scores=dict(cached.dimension_scores),
                reasoning=cached.reasoning,
                caveats=list(cached.caveats),
                confidence=cached.confidence,
                tokens_used=0,
                cached=True,
                judge_failed=cached.judge_failed,
                output_hash=output_hash,
            )

        if bundle.empty:
            # Deterministic floor -- no LLM call for nothing to judge.
            dims = {dim: 0.0 for dim in LLM_DIMENSIONS}
            dims[MECHANICS_DIMENSION] = bundle.mechanics_reliability
            verdict = RunVerdict(
                score=weighted_mean(dims),
                dimension_scores=dims,
                reasoning=(
                    f"The {bundle.kind} reached '{bundle.terminal_state}' but "
                    "produced no judgeable output."
                ),
                caveats=["No run output was found to score."],
                output_hash=output_hash,
            )
            self._cache[cache_key] = verdict
            return verdict

        verdict = await self._run_judge(
            watch, bundle, output_hash, llm_factory or _default_llm_factory
        )
        self._cache[cache_key] = verdict
        return verdict

    async def _run_judge(
        self,
        watch,
        bundle: RunOutputBundle,
        output_hash: str,
        llm_factory: Callable[[Any, str], Any],
    ) -> RunVerdict:
        verifier_model = _select_verifier_model(bundle.executor_model)
        criteria = (
            getattr(watch, "success_criteria", None)
            or getattr(watch, "title", None)
            or "Deliver the launched work successfully."
        )
        prompt = build_run_judge_prompt(success_criteria=criteria, bundle=bundle)
        messages = [
            {"role": "system", "content": _RUN_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        max_retries = Config.COORDINATOR_MAX_VERIFICATION_RETRIES
        last_error: Optional[str] = None

        for attempt in range(1, max_retries + 1):
            try:
                llm = llm_factory(watch, verifier_model)
                response = await llm.generate_response(messages)
                raw = _extract_judge_json(getattr(response, "content", None))
                if raw is None:
                    last_error = f"non-JSON judge response (attempt {attempt})"
                    logger.warning(
                        "[RunVerdict] non-JSON judge response for watch %s "
                        "(attempt %d/%d)",
                        watch.id,
                        attempt,
                        max_retries,
                    )
                    continue

                dims: Dict[str, float] = {}
                for dim in LLM_DIMENSIONS:
                    val = raw.get(dim)
                    dims[dim] = (
                        max(0.0, min(1.0, float(val)))
                        if isinstance(val, (int, float))
                        else 0.5
                    )
                dims[MECHANICS_DIMENSION] = bundle.mechanics_reliability

                confidence = raw.get("confidence", 1.0)
                confidence = (
                    max(0.0, min(1.0, float(confidence)))
                    if isinstance(confidence, (int, float))
                    else 1.0
                )
                caveats = raw.get("caveats", [])
                if not isinstance(caveats, list):
                    caveats = []

                tokens_used = 0
                usage = getattr(response, "usage", None)
                if usage:
                    if hasattr(usage, "total_tokens"):
                        tokens_used = usage.total_tokens
                    elif isinstance(usage, dict):
                        tokens_used = usage.get("total_tokens", 0)

                score = weighted_mean(dims)
                logger.info(
                    "[RunVerdict] watch %s scored %.4f (model=%s, dims=%s)",
                    watch.id,
                    score,
                    verifier_model,
                    dims,
                )
                return RunVerdict(
                    score=score,
                    dimension_scores=dims,
                    reasoning=str(raw.get("reasoning", "")),
                    caveats=[str(c) for c in caveats],
                    confidence=confidence,
                    tokens_used=tokens_used,
                    output_hash=output_hash,
                )
            except Exception as exc:  # noqa: BLE001 -- judged fail-soft below
                last_error = f"judge call failed on attempt {attempt}: {exc}"
                logger.error(
                    "[RunVerdict] judge error for watch %s (attempt %d/%d)",
                    watch.id,
                    attempt,
                    max_retries,
                    exc_info=True,
                )

        logger.warning(
            "[RunVerdict] judge exhausted %d retries for watch %s: %s",
            max_retries,
            watch.id,
            last_error,
        )
        return RunVerdict(
            score=None,
            reasoning=f"Judge failed after {max_retries} attempts: {last_error}",
            judge_failed=True,
            output_hash=output_hash,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def apply_verdict(db: Session, watch, verdict: RunVerdict):
        """Write ``final_score``/``final_verdict`` + the idempotent SCORED
        event. Joins the caller's transaction (flush via ingest)."""
        from core.models.watch_enums import WatchEventType
        from services.watch_service import WatchService

        if verdict.score is not None:
            watch.final_score = verdict.score
        watch.final_verdict = verdict.as_text()

        return WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.SCORED.value,
            event_key=f"scored:{watch.target_type}:{watch.target_id}:{verdict.output_hash[:12]}",
            summary=verdict.reasoning[:500] if verdict.reasoning else "Run scored",
            snapshot={
                "dimension_scores": verdict.dimension_scores,
                "confidence": verdict.confidence,
                "caveats": verdict.caveats,
                "judge_failed": verdict.judge_failed,
                "cached": verdict.cached,
            },
            score=verdict.score,
        )
