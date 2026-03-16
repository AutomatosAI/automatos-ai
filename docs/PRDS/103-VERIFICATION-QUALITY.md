# PRD-103 — Verification & Quality

**Version:** 1.0
**Type:** Research + Design
**Status:** Complete — Ready for Peer Review
**Priority:** P0
**Dependencies:** PRD-100 (Research Master), PRD-101 (Mission Schema — `success_criteria` JSONB), PRD-102 (Coordinator Architecture — verify step in lifecycle)
**Blocks:** PRD-106 (Outcome Telemetry — verifier_score feeds telemetry)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-15

---

## 1. Problem Statement

### 1.1 The Gap

Automatos has **no automated output verification**. The platform can execute tasks, but it cannot answer: "Did this agent's output actually satisfy the success criteria?" Without verification, the coordinator (PRD-102) is blind — it cannot decide whether to retry, continue, or escalate.

### 1.2 Existing Quality Systems

| System | Location | Scope | Limitation |
|---|---|---|---|
| `recipe_quality_service.py` — 5-dimension scoring (completeness, accuracy, efficiency, reliability, cost) → A-F grade | `orchestrator/core/services/` | Recipe executions only | Not wired to mission tasks, no success criteria matching |
| `quality_assessor.py` (Stage 7) — 5-dimension weighted scoring (completeness, coherence, accuracy, professionalism, clarity) | `orchestrator/modules/orchestrator/stages/` | Per-execution quality | Assesses general quality, not against specific success criteria |
| FutureAGI live traffic scoring — `completeness`, `is_helpful`, `is_concise` via `agent-opt-worker` | `orchestrator/core/services/futureagi_service.py` | Prompt quality eval | Evaluates prompts, not task outcomes |
| RAG quality scorer — `avg_similarity`, `source_diversity`, `coverage`, `freshness` | `orchestrator/modules/search/` | RAG-specific metrics | Only for retrieval-augmented tasks |
| `agent_reports` with 1-5 star grading (`grade` SMALLINT) | `orchestrator/core/models/core.py` | Human manual grading | After-the-fact, not automated, not mission-integrated |
| `BoardTask.result` (TEXT) + `error_message` | `orchestrator/core/models/board.py` | Free-text result | No structured quality signal |

**Decision: Verification is a new, separate system.** It does NOT unify existing scoring. Existing systems serve different scopes: `recipe_quality_service` = recipe rolling averages (stays), `quality_assessor` = pipeline stage quality (stays), `report grading` = human stars on reports (stays). Verification evaluates mission task outputs against explicit `success_criteria`. Merging these would break existing consumers and conflate different evaluation contexts.

### 1.3 What This PRD Delivers

A **VerificationService** that:
1. Takes a task output + its `success_criteria` JSONB (from PRD-101's `orchestration_tasks`)
2. Runs deterministic checks first (format, length, required sections) — zero LLM cost
3. Evaluates remaining criteria using LLM-as-judge with a different model than the executor
4. Produces a structured score (per-criterion + aggregate + confidence)
5. Returns a pass/fail/partial verdict to the coordinator for decision-making
6. Feeds results to telemetry (PRD-106) for learning

---

## 2. Prior Art: Verification Patterns

### 2.1 Overview

Six systems and patterns were studied to inform verification design. The core challenge: how do you reliably assess whether an LLM's output satisfies requirements, without hallucinating quality?

### 2.2 System-by-System Analysis

#### LLM-as-Judge (Zheng et al. 2023, MT-Bench/Arena; Raschka 2024)

MT-Bench achieved **80%+ agreement with human evaluators** on objective tasks using rubric-based absolute scoring. The key findings:

- **Rubric-based absolute scoring** (1-5 Likert per dimension) is more stable than pairwise comparison: ~9% score flip rate vs 35% for pairwise under prompt manipulation (Raschka 2024).
- **Position bias** exists: LLMs prefer the first option in pairwise comparisons. Absolute scoring eliminates this.
- **Verbosity bias**: LLMs rate longer outputs higher. Counter with explicit rubric instructions.
- **Self-preference bias** (arxiv:2410.21819, 2024): LLMs rate their own outputs higher due to lower perplexity of self-generated text.

**What we adopt:** Rubric-based absolute scoring (not pairwise). Each success criterion becomes a rubric item with a 1-5 scale. This scales to single outputs (no reference output needed) and produces stable scores.

**What we reject:** Pairwise comparison (requires O(n^2) comparisons, doesn't work for single-output evaluation).

#### Constitutional AI Critique (Anthropic 2022, arxiv:2212.08073)

Constitutional AI evaluates outputs against a set of principles (the "constitution"). The critic identifies violations and suggests revisions. The key insight: **principles as evaluation rubric** — each success criterion can be framed as a constitutional principle the output must satisfy.

**What we adopt:** The principle-based evaluation framing. Success criteria are expressed as principles: "The output MUST cover all 6 EU AI Act risk categories" becomes a constitutional check.

**What we reject:** The revision cycle (Constitutional AI revises the output; we only evaluate, the agent retries if needed).

#### OpenAI Evals Framework (openai/evals)

OpenAI Evals composes evaluators: deterministic checks (exact match, regex, JSON schema) + model-graded checks. The key pattern: **deterministic-first, LLM-second**. Many criteria can be checked without LLM: word count, format compliance, required sections present, URL validity.

**What we adopt:** Deterministic-first pipeline. Check format, length, schema, required sections BEFORE calling the LLM judge. This dramatically reduces verification cost (deterministic checks are free) and catches obvious failures immediately.

**What we reject:** The full Evals framework infrastructure (we integrate with FutureAGI instead).

#### DeepEval (confident-ai/deepeval)

DeepEval's **DAG evaluation pattern** uses decision-tree evaluation with conditional branching: check format → check completeness → check accuracy → check quality. Each node can be deterministic or LLM-based. Failed early checks skip expensive later checks.

**What we adopt:** DAG evaluation pipeline. If format check fails, skip LLM quality assessment (no point evaluating quality of a malformed output). This is the "short-circuit" pattern.

**What we reject:** DeepEval's Pytest-like test framework (over-engineered for our in-process verification).

#### RAGAS (explodinggradients/ragas)

RAGAS provides specialized metrics for RAG tasks: faithfulness (95% human agreement), answer relevancy, context precision/recall. The key insight: **task-type-specific verification dimensions**.

**What we adopt:** For RAG-heavy mission tasks (task_type = "research"), add faithfulness and source grounding as verification dimensions alongside generic quality dimensions.

**What we reject:** Using RAGAS as the verification framework (too specialized; we need general-purpose verification).

#### FutureAGI (Existing — futureagi_service.py)

FutureAGI is already integrated via the `agent-opt-worker` HTTP proxy:

| Capability | Endpoint | Status | Reusable? |
|---|---|---|---|
| Prompt assessment | `POST /assess` | Production | Partially — prompt quality, not task output |
| Live traffic scoring | `POST /score` | Production | Yes — same pattern: input + output + metrics → scores |
| Prompt optimization | `POST /optimize` | Production | No — optimization, not evaluation |
| Safety check | `POST /safety` | Production | Yes — safety is one verification dimension |

**What we adopt:** Extend the `agent-opt-worker` with a new `POST /verify-task` endpoint. Same infrastructure, same HTTP proxy pattern, new verification logic.

### 2.3 Architectural Decisions Summary

| Decision | Choice | Source | Rationale |
|---|---|---|---|
| **Scoring method** | Rubric-based absolute scoring (1-5 Likert) | Zheng et al. 2023, Raschka 2024 | 80%+ human agreement; 9% flip rate (vs 35% pairwise); scales to single outputs |
| **Judge model** | Always different model from executor | arxiv:2410.21819 | Self-preference bias empirically demonstrated; cross-model eliminates correlation |
| **Evaluation pipeline** | Deterministic → LLM (DAG with short-circuit) | OpenAI Evals, DeepEval | Free deterministic checks first; skip expensive LLM if format fails |
| **Criteria framing** | Constitutional principles from success_criteria | Anthropic 2022 | Structured, evaluable, maps directly to task definition |
| **Infrastructure** | FutureAGI worker extension (POST /verify-task) | Existing integration | Zero new infrastructure; same proxy pattern |
| **Cost target** | Verification ≤ 15% of task generation cost | Industry benchmarks | Single judge (not ensemble) with deterministic pre-filtering |

---

## 3. VerificationService Interface

### 3.1 Core Interface

```python
@dataclass(frozen=True)
class CriterionScore:
    """Score for a single success criterion."""
    criterion: str
    score: float  # 0.0 - 1.0
    met: bool  # True if score >= threshold
    must_pass: bool  # From success_criteria definition
    reasoning: str  # Judge's reasoning (for retry feedback)
    check_type: str  # "deterministic" or "llm"


@dataclass(frozen=True)
class VerificationResult:
    """Immutable result from task verification."""
    task_id: int
    verdict: str  # "pass" | "fail" | "partial"
    confidence: float  # 0.0 - 1.0
    aggregate_score: float  # 0.0 - 1.0 (weighted mean of criterion scores)
    criteria_scores: list[CriterionScore]
    quality_scores: dict[str, float]  # Generic quality dimensions
    tokens_used: int
    cost_usd: float
    verifier_model: str
    verification_time_ms: float


class VerificationService:
    """
    Evaluates mission task outputs against success criteria.

    Pipeline:
    1. Deterministic checks (format, length, schema) — free
    2. LLM evaluation (quality, accuracy, completeness) — costs tokens
    3. Aggregate scores and produce verdict
    """

    def __init__(
        self,
        futureagi_client: FutureAGIClient,
        default_verifier_model: str = "openai/gpt-4o-mini",
        default_threshold: float = 0.7,
    ) -> None: ...

    async def verify_task(
        self,
        task: OrchestrationTask,
        output: str,
        verifier_model: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        """
        Verify a task's output against its success criteria.

        1. Run deterministic checks — if any must_pass fails, short-circuit to FAIL
        2. Run LLM verification for remaining criteria
        3. Aggregate scores, compute verdict
        4. Store results on task and emit event
        """
        ...

    async def verify_cross_task_consistency(
        self,
        run_id: int,
        task_ids: list[int],
    ) -> VerificationResult:
        """
        Check consistency between related task outputs.
        e.g., research findings and analysis report agree on risk levels.
        Called before mission-level review, not per-task.
        """
        ...
```

### 3.2 Verification Pipeline

```
Task output received
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 1: Deterministic Checks       │
│  For each criterion:                │
│    - Format check (regex, schema)   │
│    - Length check (min/max words)    │
│    - Required sections (headers)    │
│    - URL validity                   │
│    - JSON conformance               │
│                                     │
│  If any must_pass deterministic     │
│  check fails → FAIL immediately     │
│  (zero LLM cost)                    │
└──────────────┬──────────────────────┘
               │ all deterministic passed (or N/A)
               ▼
┌─────────────────────────────────────┐
│ Stage 2: LLM Verification           │
│  Build rubric from remaining        │
│  success criteria                   │
│  Call verifier model (different     │
│  from executor) via FutureAGI       │
│  worker POST /verify-task           │
│                                     │
│  Parse structured scores            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Stage 3: Aggregation                │
│  Combine deterministic + LLM scores │
│  Weighted mean (by criterion weight)│
│  Compute verdict:                   │
│    - ALL must_pass met + avg ≥ 0.7  │
│      → PASS                         │
│    - ALL must_pass met + avg < 0.7  │
│      → PARTIAL                      │
│    - ANY must_pass failed           │
│      → FAIL                         │
│  Confidence = min(criteria scores)  │
│  Low confidence (<0.5) → escalate   │
│  to human regardless of verdict     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Stage 4: Store & Signal             │
│  Update orchestration_tasks:        │
│    verifier_score = aggregate       │
│    verification_status = verdict    │
│  Emit orchestration_event:          │
│    type = "verification_completed"  │
│    data = full VerificationResult   │
│  Signal coordinator:                │
│    PASS → advance to next task      │
│    PARTIAL → coordinator decides    │
│    FAIL → retry with feedback       │
└─────────────────────────────────────┘
```

---

## 4. Deterministic Check Registry

### 4.1 Check Types

```python
class DeterministicCheckType(StrEnum):
    """Types of checks that don't require LLM evaluation."""
    FORMAT_REGEX = "format_regex"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    REQUIRED_SECTIONS = "required_sections"
    JSON_SCHEMA = "json_schema"
    URL_VALID = "url_valid"
    CONTAINS_KEYWORDS = "contains_keywords"
    WORD_COUNT_RANGE = "word_count_range"


class DeterministicChecker:
    """
    Registry of deterministic checks.
    Each check takes (output, criterion_params) and returns (passed, reasoning).
    """

    _CHECKS: dict[str, Callable] = {
        "format_regex": _check_regex,
        "min_length": _check_min_length,
        "max_length": _check_max_length,
        "required_sections": _check_required_sections,
        "json_schema": _check_json_schema,
        "url_valid": _check_url_valid,
        "contains_keywords": _check_keywords,
        "word_count_range": _check_word_count,
    }

    def can_check_deterministically(self, criterion: dict) -> bool:
        """
        Determine if a criterion can be checked without LLM.
        Heuristic based on criterion text patterns.
        """
        text = criterion["criterion"].lower()
        patterns = {
            r"must have .+ section": "required_sections",
            r"at least \d+ words": "min_length",
            r"no more than \d+ words": "max_length",
            r"must include .+ header": "required_sections",
            r"valid json": "json_schema",
            r"format: ": "format_regex",
        }
        return any(re.search(p, text) for p in patterns)

    async def check(
        self, output: str, criterion: dict
    ) -> Optional[CriterionScore]:
        """
        Run deterministic check if applicable.
        Returns None if no deterministic check applies.
        """
        check_type = self._infer_check_type(criterion)
        if check_type is None:
            return None

        checker = self._CHECKS[check_type]
        passed, reasoning = checker(output, criterion)

        return CriterionScore(
            criterion=criterion["criterion"],
            score=1.0 if passed else 0.0,
            met=passed,
            must_pass=criterion.get("must_pass", False),
            reasoning=reasoning,
            check_type="deterministic",
        )
```

### 4.2 Example: Required Sections Check

```python
def _check_required_sections(output: str, criterion: dict) -> tuple[bool, str]:
    """
    Check if output contains required section headers.
    Parses criterion text like: "Has Executive Summary, Findings, and Recommendations sections"
    """
    # Extract section names from criterion text
    text = criterion["criterion"]
    # Pattern: "Has X, Y, and Z sections" or "Must include X, Y, Z headers"
    section_match = re.findall(
        r'(?:has|include|contain)s?\s+(.+?)(?:\s+sections?|\s+headers?)',
        text, re.IGNORECASE
    )

    if not section_match:
        return True, "Could not parse required sections from criterion"

    # Split "X, Y, and Z" into ["X", "Y", "Z"]
    sections_text = section_match[0]
    sections = [
        s.strip().lower()
        for s in re.split(r',\s*(?:and\s+)?|(?:and\s+)', sections_text)
        if s.strip()
    ]

    # Check each section exists as a heading in output
    output_lower = output.lower()
    missing = []
    for section in sections:
        # Check for markdown headers or plain text headers
        patterns = [
            f"# {section}",
            f"## {section}",
            f"### {section}",
            f"{section}:",
            f"**{section}**",
        ]
        if not any(p in output_lower for p in patterns):
            missing.append(section)

    if missing:
        return False, f"Missing sections: {', '.join(missing)}"
    return True, f"All {len(sections)} required sections found"
```

---

## 5. LLM Verification Protocol

### 5.1 Model Selection

**Rule: verifier model MUST differ from executor model.**

Self-preference bias (arxiv:2410.21819) is empirically demonstrated — LLMs systematically rate their own outputs higher due to lower perplexity of self-generated text. Cross-model verification eliminates this correlation.

```python
VERIFIER_MODEL_SELECTION: dict[str, str] = {
    # If executor used this family → verifier uses this
    "anthropic": "openai/gpt-4o-mini",      # Claude executed → GPT verifies
    "openai": "anthropic/claude-haiku-4-5-20251001",  # GPT executed → Claude verifies
    "google": "openai/gpt-4o-mini",          # Gemini executed → GPT verifies
    "deepseek": "anthropic/claude-haiku-4-5-20251001",  # DeepSeek → Claude verifies
    "meta": "openai/gpt-4o-mini",            # Llama executed → GPT verifies
    "default": "openai/gpt-4o-mini",         # Fallback: GPT-4o-mini (cheap, good judge)
}

def select_verifier_model(executor_model: str) -> str:
    """Select a verifier model from a different family than the executor."""
    provider = executor_model.split("/")[0] if "/" in executor_model else "default"
    return VERIFIER_MODEL_SELECTION.get(provider, VERIFIER_MODEL_SELECTION["default"])
```

**Cost rationale:** GPT-4o-mini and Claude Haiku are ~10-20x cheaper than their full counterparts. Using them as judges keeps verification cost at ~10-15% of task generation cost. MT-Bench showed cheaper models are adequate judges for rubric-based evaluation.

### 5.2 Verifier Prompt Template

```python
VERIFIER_PROMPT = """You are a task output evaluator. Your job is to assess whether an agent's output satisfies specific success criteria.

## Task Description
{task_description}

## Success Criteria (your rubric)
{criteria_rubric}

## Agent's Output
{task_output}

## Instructions
For EACH criterion, provide:
1. A score from 1-5:
   1 = Not met at all
   2 = Partially addressed but major gaps
   3 = Addressed with some gaps or quality issues
   4 = Well addressed with minor issues
   5 = Fully met, high quality
2. Whether the criterion is MET (score >= 4) or NOT MET (score < 4)
3. Brief reasoning (1-2 sentences)

IMPORTANT RULES:
- Score based on the RUBRIC CRITERIA, not your general impression
- Shorter outputs are not worse if they meet the criteria
- Longer outputs are not better if they don't meet the criteria
- Base accuracy judgments on the output's internal consistency and specificity
- If you're uncertain about a factual claim, score it 3 (uncertain) not 1 (wrong)
- Report your confidence level (0.0-1.0) for each score

## Output Format (JSON)
```json
{{
    "criteria_scores": [
        {{
            "criterion": "exact criterion text",
            "score": 4,
            "normalized_score": 0.75,
            "met": true,
            "confidence": 0.85,
            "reasoning": "Brief explanation"
        }}
    ],
    "quality_scores": {{
        "completeness": 0.8,
        "accuracy": 0.7,
        "clarity": 0.9,
        "relevance": 0.85
    }},
    "overall_assessment": "Brief summary of strengths and weaknesses"
}}
```
"""
```

### 5.3 Score Normalization

LLM judges return 1-5 Likert scores. These are normalized to 0.0-1.0 for storage and comparison:

```python
def normalize_likert(score: int) -> float:
    """Normalize 1-5 Likert to 0.0-1.0."""
    return (score - 1) / 4.0
    # 1 → 0.0, 2 → 0.25, 3 → 0.5, 4 → 0.75, 5 → 1.0
```

### 5.4 Verdict Computation

```python
def compute_verdict(
    criteria_scores: list[CriterionScore],
    threshold: float = 0.7,
    confidence_escalation_threshold: float = 0.5,
) -> tuple[str, float, float]:
    """
    Compute pass/fail/partial verdict from criterion scores.

    Returns: (verdict, aggregate_score, confidence)

    Rules:
    1. If ANY must_pass criterion is NOT met → FAIL
    2. If ALL must_pass met AND aggregate >= threshold → PASS
    3. If ALL must_pass met AND aggregate < threshold → PARTIAL
    4. If confidence < 0.5 → override verdict to PARTIAL (escalate to human)
    """
    # Check must_pass criteria
    must_pass_failed = [
        c for c in criteria_scores
        if c.must_pass and not c.met
    ]
    if must_pass_failed:
        return "fail", _weighted_mean(criteria_scores), _min_confidence(criteria_scores)

    # Compute weighted aggregate
    aggregate = _weighted_mean(criteria_scores)
    confidence = _min_confidence(criteria_scores)

    # Low confidence → escalate regardless
    if confidence < confidence_escalation_threshold:
        return "partial", aggregate, confidence

    if aggregate >= threshold:
        return "pass", aggregate, confidence
    else:
        return "partial", aggregate, confidence


def _weighted_mean(scores: list[CriterionScore]) -> float:
    """Weighted mean of criterion scores."""
    # Weights come from success_criteria definition
    total_weight = sum(getattr(s, 'weight', 1.0) for s in scores)
    if total_weight == 0:
        return 0.0
    return sum(s.score * getattr(s, 'weight', 1.0) for s in scores) / total_weight
```

---

## 6. FutureAGI Worker Extension

### 6.1 New Endpoint: POST /verify-task

```python
# Request schema
{
    "task_description": "Research EU AI Act requirements",
    "success_criteria": [
        {
            "criterion": "Covers all 6 risk categories",
            "weight": 0.4,
            "must_pass": true
        },
        {
            "criterion": "Includes implementation timelines",
            "weight": 0.3,
            "must_pass": false
        },
        {
            "criterion": "Cites specific articles",
            "weight": 0.3,
            "must_pass": false
        }
    ],
    "task_output": "... agent's output ...",
    "verifier_model": "openai/gpt-4o-mini",
    "executor_model": "anthropic/claude-sonnet-4-20250514",
    "threshold": 0.7
}

# Response schema
{
    "verdict": "pass",
    "confidence": 0.82,
    "aggregate_score": 0.78,
    "criteria_scores": [
        {
            "criterion": "Covers all 6 risk categories",
            "score": 0.85,
            "met": true,
            "must_pass": true,
            "confidence": 0.9,
            "reasoning": "All 6 categories identified: unacceptable, high, limited, minimal, general purpose, transparency obligations",
            "check_type": "llm"
        },
        {
            "criterion": "Includes implementation timelines",
            "score": 0.70,
            "met": true,
            "must_pass": false,
            "confidence": 0.75,
            "reasoning": "Key dates mentioned (Aug 2025, Feb 2025) but missing some secondary deadlines",
            "check_type": "llm"
        },
        {
            "criterion": "Cites specific articles",
            "score": 0.80,
            "met": true,
            "must_pass": false,
            "confidence": 0.80,
            "reasoning": "References Articles 5, 6, 9, 52 specifically. Could cite more.",
            "check_type": "llm"
        }
    ],
    "quality_scores": {
        "completeness": 0.82,
        "accuracy": 0.78,
        "clarity": 0.85,
        "relevance": 0.90
    },
    "tokens_used": 1250,
    "cost_usd": 0.003,
    "verifier_model": "openai/gpt-4o-mini",
    "verification_time_ms": 2340
}
```

### 6.2 Worker Implementation

The `agent-opt-worker` service gets a new route that follows the existing pattern:

```python
# In agent-opt-worker service
@app.post("/verify-task")
async def verify_task(request: VerifyTaskRequest) -> VerifyTaskResponse:
    """
    Verify task output against success criteria.

    Pipeline:
    1. Deterministic checks (local, no LLM)
    2. LLM evaluation (via OpenRouter, different model from executor)
    3. Score aggregation and verdict
    """
    checker = DeterministicChecker()
    criteria_scores = []

    # Stage 1: Deterministic checks
    for criterion in request.success_criteria:
        det_result = await checker.check(request.task_output, criterion)
        if det_result is not None:
            criteria_scores.append(det_result)
            # Short-circuit: must_pass deterministic failure → immediate FAIL
            if det_result.must_pass and not det_result.met:
                return _build_fail_response(criteria_scores, request)

    # Stage 2: LLM verification for remaining criteria
    remaining = [
        c for c in request.success_criteria
        if c["criterion"] not in {cs.criterion for cs in criteria_scores}
    ]

    llm_scores = []
    if remaining:
        llm_scores = await _llm_verify(
            task_description=request.task_description,
            criteria=remaining,
            output=request.task_output,
            model=request.verifier_model,
        )
        criteria_scores.extend(llm_scores)

    # Stage 3: Verdict
    verdict, aggregate, confidence = compute_verdict(
        criteria_scores, request.threshold
    )

    return VerifyTaskResponse(
        verdict=verdict,
        confidence=confidence,
        aggregate_score=aggregate,
        criteria_scores=criteria_scores,
        quality_scores=_extract_quality_scores(llm_scores),
        tokens_used=sum(s.tokens_used for s in llm_scores if hasattr(s, 'tokens_used')),
        cost_usd=_compute_cost(llm_scores, request.verifier_model),
        verifier_model=request.verifier_model,
    )
```

---

## 7. Coordinator Integration

### 7.1 How Verification Drives Coordinator Decisions

```python
# In MissionReconciler (PRD-102)
async def _handle_task_output(
    self, run: OrchestrationRun, task: OrchestrationTask
) -> None:
    """Called when a task submits output (transitions to VERIFYING)."""

    # 1. Select verifier model (different from executor)
    verifier_model = select_verifier_model(task.model_used)

    # 2. Run verification
    result = await self._verification_service.verify_task(
        task=task,
        output=task.result_text,
        verifier_model=verifier_model,
        threshold=run.config.get("verification_threshold", 0.7),
    )

    # 3. Store verification result
    task.verifier_score = result.aggregate_score
    task.verification_status = result.verdict

    # 4. Emit event
    await self._emit_event(run.id, "verification_completed", {
        "task_id": task.id,
        "verdict": result.verdict,
        "score": result.aggregate_score,
        "confidence": result.confidence,
    })

    # 5. Coordinator decision based on verdict
    match result.verdict:
        case "pass":
            await self._transition_task(task, TaskState.COMPLETED)
            await self._on_task_completed(run.id, task.id)

        case "partial":
            if result.confidence < 0.5:
                # Low confidence → human must decide
                await self._transition_task(task, TaskState.AWAITING_HUMAN)
            elif task.attempt_count < task.max_retries:
                # Retry with verifier feedback
                await self._retry_with_feedback(run, task, result)
            else:
                # Max retries exhausted → human decides
                await self._transition_task(task, TaskState.AWAITING_HUMAN)

        case "fail":
            if task.attempt_count < task.max_retries:
                await self._retry_with_feedback(run, task, result)
            else:
                await self._transition_task(task, TaskState.FAILED)
```

### 7.2 Retry-with-Feedback Protocol

When verification fails but retries remain, the verifier's reasoning is injected into the agent's next prompt:

```python
async def _retry_with_feedback(
    self, run: OrchestrationRun, task: OrchestrationTask,
    result: VerificationResult
) -> None:
    """
    Build retry context from verification feedback.
    The agent receives specific, actionable feedback on what to fix.
    """
    failed_criteria = [
        {
            "criterion": c.criterion,
            "score": c.score,
            "reasoning": c.reasoning,
        }
        for c in result.criteria_scores
        if not c.met
    ]

    task.retry_context = {
        "attempt": task.attempt_count + 1,
        "max_attempts": task.max_retries,
        "previous_score": result.aggregate_score,
        "feedback": f"Your previous output scored {result.aggregate_score:.0%}. "
                    f"The following criteria were not met:",
        "failed_criteria": failed_criteria,
        "verifier_summary": result.quality_scores,
    }

    task.attempt_count += 1
    delay_s = min(10 * (2 ** (task.attempt_count - 1)), 300)

    await self._transition_task(task, TaskState.AWAITING_RETRY)
    await self._schedule_retry(task, delay_s)
```

---

## 8. Verification Timing Strategy

### 8.1 Inline vs Batch

| Task Position | Strategy | Rationale |
|---|---|---|
| Task with dependents (critical path) | **Inline** — verify immediately | Blocks next task; fast feedback enables quick retry |
| Terminal task (no dependents) | **Inline** — verify immediately | Still needed for mission completion assessment |
| Cross-task consistency | **Batch** — after all related tasks complete | Requires multiple outputs to compare |

**Decision: All per-task verification is inline.** The 2-3 second latency of an LLM verification call is negligible compared to the minutes a task takes to execute. Batch is only for cross-task consistency (optional, post-completion).

### 8.2 Verification Cost Model

| Component | Cost | When |
|---|---|---|
| Deterministic checks | $0.00 | Always (before LLM) |
| Single LLM judge call | ~$0.003-0.01 per task | When deterministic checks pass |
| Cross-task consistency | ~$0.005 per pair | Optional, after related tasks complete |

**For a typical 4-task mission ($2-4 total):**
- Verification cost: ~$0.012-0.04 (4 judge calls)
- As % of mission cost: ~1-2%
- With retries (assume 1 retry): ~2-4%

This is well within the 10-30% industry benchmark, primarily because deterministic checks filter out failures that would have required expensive LLM evaluation.

### 8.3 Verification Bypass Rules

| Condition | Action | Rationale |
|---|---|---|
| Task cost < $0.05 | Skip LLM verification, deterministic only | Cost of verification would exceed task cost |
| task_type = "simple" | Deterministic only | Simple tasks (formatting, routing) don't need LLM quality assessment |
| All criteria are deterministic | Skip LLM entirely | No subjective criteria to evaluate |
| Mission config `skip_verification = true` | Skip entirely | User explicitly opts out (autonomy mode with high trust) |

---

## 9. Configurable Thresholds

### 9.1 Threshold Hierarchy

```
Workspace defaults (from workspace settings)
  └─ Mission overrides (from mission config)
       └─ Task overrides (from success_criteria per task)
```

### 9.2 Default Configuration

```python
DEFAULT_VERIFICATION_CONFIG = {
    "pass_threshold": 0.7,           # Aggregate score >= 0.7 → PASS
    "confidence_escalation": 0.5,    # Confidence < 0.5 → escalate to human
    "must_pass_score": 0.75,         # Individual must_pass criterion threshold (Likert 4/5)
    "max_verification_cost_pct": 0.30,  # Max verification cost as % of task cost
    "skip_verification_below_usd": 0.05,  # Skip LLM verification for cheap tasks
    "cross_task_consistency": True,   # Enable cross-task consistency checks
    "verifier_model": "auto",        # Auto-select based on executor model
}
```

---

## 10. Cross-Task Consistency Checking

### 10.1 When to Check

Cross-task consistency is checked when two or more tasks share a topic or produce related outputs. The coordinator identifies related task pairs based on dependency edges and task descriptions.

### 10.2 Consistency Verifier Prompt

```python
CONSISTENCY_PROMPT = """You are checking consistency between two related task outputs.

## Task A: {task_a_title}
{task_a_output_summary}

## Task B: {task_b_title}
{task_b_output_summary}

## Check
Do these outputs contradict each other on any factual claims, conclusions, or recommendations?

Respond with JSON:
```json
{{
    "consistent": true/false,
    "contradictions": [
        {{
            "topic": "what they disagree on",
            "task_a_claim": "what Task A says",
            "task_b_claim": "what Task B says",
            "severity": "high|medium|low"
        }}
    ],
    "confidence": 0.85
}}
```
"""
```

---

## 11. Schema Integration (PRD-101 Fields)

### 11.1 Fields on orchestration_tasks

| Field | Type | Set By | Notes |
|---|---|---|---|
| `verifier_score` | FLOAT | VerificationService | 0.0-1.0 aggregate score |
| `verification_status` | VARCHAR(20) | VerificationService | pass/fail/partial |
| `success_criteria` | JSONB | MissionPlanner | Input criteria for verification |

### 11.2 Events Emitted

| Event Type | Data | Emitted When |
|---|---|---|
| `verification_started` | `{task_id, verifier_model}` | Verification begins |
| `verification_completed` | `{task_id, verdict, score, confidence, criteria_scores}` | Verification finishes |
| `verification_failed` | `{task_id, error}` | Verification service error (not task failure) |
| `human_review_requested` | `{task_id, reason, score, confidence}` | Low confidence → escalate |

---

## 12. Acceptance Criteria

### Must Have

- [x] **VerificationService interface** — method signatures with type hints for `verify_task()` and `verify_cross_task_consistency()`
- [x] **Verifier prompt template** — bias mitigations (self-preference via cross-model, verbosity via rubric instructions, position via absolute scoring)
- [x] **Deterministic check registry** — 8 check types (format, length, schema, sections, URL, keywords, word count, JSON)
- [x] **LLM verification protocol** — model selection (cross-family), rubric construction from success_criteria, scoring output schema
- [x] **FutureAGI worker extension** — `POST /verify-task` endpoint spec with request/response schemas
- [x] **Verdict taxonomy** — pass/fail/partial with confidence scores, must_pass enforcement, threshold configuration
- [x] **PRD-101 schema integration** — `verifier_score`, `verification_status` fields on `orchestration_tasks`
- [x] **PRD-102 coordinator integration** — verification verdict drives coordinator decisions (retry with feedback, continue, escalate)
- [x] **Retry-with-feedback protocol** — verifier reasoning injected into agent retry prompt
- [x] **Cost model** — verification at 1-4% of task cost with deterministic pre-filtering

### Should Have

- [x] **Cross-task consistency checking** — consistency verifier for related task outputs
- [x] **Configurable thresholds** — workspace defaults, mission overrides, task overrides
- [x] **Verification bypass rules** — skip for cheap tasks, simple tasks, explicit opt-out

### Nice to Have

- [ ] Multi-judge ensemble design (2-3 judges for high-stakes tasks)
- [ ] Verification result caching (don't re-verify unchanged outputs on retry)
- [ ] Human calibration tracking (override rate → threshold adjustment)

---

## 13. Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | **LLM-as-judge unreliability on subjective tasks** | Medium | High | Rubric-based scoring (9% flip rate). Default to human review for low confidence. Track override rate. |
| 2 | **Verification cost exceeding task cost** | Medium | Low | Deterministic checks first. Skip LLM for cheap tasks. Single-judge default. Budget: ≤15% of task cost. |
| 3 | **Self-preference bias** | High | High | ENFORCED cross-model verification. Never same model for execution and verification. |
| 4 | **False positives (bad output passes)** | High | Medium | Multiple criteria checked independently. Any must_pass failure = task fails. Human review for high-stakes missions. |
| 5 | **False negatives (good output fails)** | Medium | Medium | Confidence scores + human escalation. Track false negative rate via override data. Adjust thresholds from telemetry (PRD-106). |
| 6 | **Verifier prompt engineering is hard** | High | High | Start with simple rubric template. Iterate based on override data. FutureAGI team's eval expertise. |
| 7 | **Three existing scoring systems cause confusion** | Medium | Medium | Verification is NEW and SEPARATE — different scope (mission task success criteria). Do not unify. Document distinction. |

---

## 14. Dependencies

| Dependency | Direction | Notes |
|---|---|---|
| PRD-101 (Mission Schema) | Uses | `success_criteria` JSONB, `verifier_score`, `verification_status` fields |
| PRD-102 (Coordinator) | Blocked by | Coordinator triggers verification; this PRD defines the interface |
| PRD-105 (Budget) | Uses | Verification cost included in mission budget |
| PRD-106 (Telemetry) | Feeds | `verifier_score` is a first-class telemetry field |
| `agent-opt-worker` | Extension | New `/verify-task` endpoint on existing service |
| OpenRouter | Uses | Cross-model verification requires calling different provider |

---

## Appendix: Research Sources

| Source | What It Informed |
|--------|-----------------|
| Zheng et al. 2023, "Judging LLM-as-a-Judge" (arxiv:2306.05685) | Rubric-based absolute scoring, 80%+ human agreement, position bias |
| Raschka 2024, "LLM Judge Stability" | 9% flip rate for rubric-based vs 35% for pairwise |
| Self-preference bias study (arxiv:2410.21819, 2024) | Cross-model verification mandate |
| Anthropic 2022, Constitutional AI (arxiv:2212.08073) | Principle-based evaluation framing |
| OpenAI Evals (openai/evals) | Deterministic-first, LLM-second pipeline |
| DeepEval (confident-ai/deepeval) | DAG evaluation with short-circuit on failure |
| RAGAS (explodinggradients/ragas) | Task-type-specific verification dimensions |
| Automatos FutureAGI integration | Existing worker pattern for extension |
| Automatos recipe_quality_service.py, quality_assessor.py | Existing quality systems — separate scope, don't unify |
