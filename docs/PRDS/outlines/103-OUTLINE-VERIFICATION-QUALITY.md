# PRD-103 Outline: Verification & Quality

**Type:** Research + Design
**Status:** Outline (Loop 0)
**Depends On:** PRD-100 (Research Master), PRD-101 (Mission Schema — `success_criteria` JSONB), PRD-102 (Coordinator Architecture — verify step in lifecycle)
**Blocks:** PRD-106 (Outcome Telemetry — verifier_score feeds telemetry)

---

## Section 1: Problem Statement

### Why This PRD Exists

Automatos has **no automated output verification**. Today:

| What Exists | Limitation |
|---|---|
| `agent_reports` with 1-5 star grading (`grade` SMALLINT) | Human-only, manual, after-the-fact |
| `recipe_quality_service.py` — 5-dimension scoring (completeness, accuracy, efficiency, reliability, cost) → 0-1.0 → A-F grade | Recipe-scoped only, not wired to mission tasks |
| `quality_assessor.py` (Stage 7) — 5-dimension weighted scoring (completeness, coherence, accuracy, professionalism, clarity) | Per-execution, not against success criteria |
| FutureAGI live traffic scoring — `completeness`, `is_helpful`, `is_concise` via `agent-opt-worker` | Prompt-quality eval, not task-outcome verification |
| RAG quality scorer — `avg_similarity`, `source_diversity`, `coverage`, `freshness` | RAG-specific, not general-purpose |
| `heartbeat_results` JSONB — `findings[]`, `actions_taken[]`, `tokens_used`, `cost` | Captures what happened, not whether it was good |
| `BoardTask.result` (TEXT) + `error_message` | Free-text, no structured quality signal |

### The Verification Gap

Missions (PRD-100 Section 3) require a **verify step** between task execution and human review. The coordinator (PRD-102) needs a signal: "Did this agent's output actually satisfy the success criteria?" Without this:

1. **Human bottleneck** — every task output requires manual review with no pre-screening
2. **No quality gate** — bad outputs flow to dependent tasks, cascading failures
3. **No learning signal** — PRD-106 telemetry has no `verifier_score` to learn from
4. **Coordinator is blind** — cannot decide whether to retry, continue, or escalate

### What This PRD Delivers

A **VerificationService** that:
1. Takes a task output + its `success_criteria` JSONB (from PRD-101's `mission_tasks`)
2. Evaluates output against each criterion using LLM-as-judge
3. Produces a structured score (per-criterion + aggregate)
4. Returns a pass/fail/partial verdict with confidence
5. Feeds results to the coordinator for decision-making and to telemetry for learning

---

## Section 2: Prior Art Research Targets

### Systems to Study (each gets dedicated research)

| System/Pattern | Source | Focus Areas | Key Question |
|---|---|---|---|
| **LLM-as-Judge (MT-Bench/Arena)** | Zheng et al. 2023 (arxiv:2306.05685); Arena-Hard (LMSYS 2024) | Pairwise vs absolute scoring, position bias mitigation, 80%+ human agreement on objective tasks, 50-70% on subjective | Should we use rubric-based absolute scoring (scales to single outputs) or pairwise comparison? |
| **Constitutional AI Critique** | Anthropic 2022 (arxiv:2212.08073) | Principle-based self-critique, revision cycles, constitutional principles as evaluation rubric | Should success criteria be expressed as constitutional principles that the verifier checks? |
| **OpenAI Evals Framework** | `openai/evals` GitHub | Modular evaluators, deterministic + model-graded composition, different model for grading vs generation | Should we compose deterministic checks (format, length) with LLM checks (quality, accuracy)? |
| **DeepEval** | `confident-ai/deepeval` GitHub | G-Eval (simple custom metrics), DAG (decision-tree evaluation), 14+ pre-built metrics, Pytest-like test framework | Should we adopt DeepEval's DAG pattern for structured evaluation pipelines? |
| **RAGAS** | `explodinggradients/ragas` GitHub | Faithfulness (95% human agreement), answer relevancy, context precision/recall, rubric-based criteria scoring | Should we use RAGAS metrics for RAG-heavy mission tasks? |
| **FutureAGI (existing)** | `orchestrator/core/services/futureagi_service.py` | Already integrated: assess, optimize, safety, live scoring via `agent-opt-worker`. Extensible via new worker endpoints | How do we extend FutureAGI's existing `/score` pattern for mission task verification? |
| **Existing Quality Scoring** | `recipe_quality_service.py`, `quality_assessor.py`, `report_service.py` | 3 scoring systems already built with overlapping dimensions. Report grading (1-5 stars) is human-only | Should verification unify these scoring approaches or add a fourth? |

### Key Patterns Discovered in Research

**Rubric-based absolute scoring is the right default (Zheng et al. 2023, Raschka 2024):** Pairwise comparison requires O(n²) comparisons and doesn't work for evaluating single outputs. Rubric-based scoring (1-5 Likert per dimension) is more stable (~9% score flip rate vs 35% for pairwise under manipulation) and scales to evaluating individual task outputs. MT-Bench achieved 80%+ agreement with human evaluators on objective tasks using this approach.

**Self-preference bias requires cross-model verification (arxiv:2410.21819, 2024):** LLMs systematically rate their own outputs higher due to lower perplexity of self-generated text. Mission verification MUST use a different model than the one that executed the task. Using a smaller/cheaper model as judge reduces correlation with the generator AND saves cost.

**Deterministic-first, LLM-second (OpenAI Evals pattern):** Many success criteria can be checked deterministically: word count, format compliance, required sections present, URL validity, JSON schema conformance. LLM judge should only evaluate what deterministic checks cannot: quality, accuracy, completeness of reasoning. This dramatically reduces verification cost.

**DeepEval's DAG is the right evaluation architecture:** Decision-tree evaluation with conditional branching maps directly to success criteria checking: check format → check completeness → check accuracy → check quality. Each node can be deterministic or LLM-based. Failed early checks skip expensive later checks.

**Verification cost budget: 10-30% of generation cost (industry benchmarks):** Single LLM judge run adds ~10-15% of generation cost. Robust evaluation (position-swapped double-check) adds ~20-30%. Ensemble judging (3+ models) adds 50-100%. For missions, single-judge with escalation is the right tradeoff.

---

## Section 3: Verification Taxonomy

### Types of Verification (by what they check)

| Type | What It Checks | Method | Example |
|---|---|---|---|
| **Format Compliance** | Output matches required structure | Deterministic (regex, JSON schema, section headers) | "Report must have Executive Summary, Findings, Recommendations sections" |
| **Completeness** | All required elements present | Hybrid (deterministic count + LLM assessment) | "Must cover all 5 EU AI Act risk categories" |
| **Factual Accuracy** | Claims are grounded in sources | LLM-as-judge with source verification | "All cited regulations must exist and be correctly described" |
| **Success Criteria Match** | Output satisfies each stated criterion | LLM-as-judge against criteria rubric | "Must identify at least 3 compliance gaps with severity ratings" |
| **Cross-Agent Consistency** | Multiple agents' outputs don't contradict | LLM comparison of related task outputs | "Research findings and compliance report must agree on risk levels" |
| **Quality Threshold** | Output meets minimum quality bar | LLM scoring on dimensions (clarity, depth, professionalism) | "Writing quality score ≥ 3.5/5" |

### Verification Granularity

| Level | Scope | When | Cost |
|---|---|---|---|
| **Per-criterion** | Individual success criterion | After task execution | 1 LLM call per criterion |
| **Per-task** | Aggregate across all criteria for one task | After per-criterion checks | 0 additional (aggregation) |
| **Cross-task** | Consistency between related task outputs | After dependent tasks complete | 1 LLM call per pair |
| **Per-mission** | Overall mission quality assessment | Before human review | 1 LLM call (summary) |

---

## Section 4: Key Design Questions

### Q1: Same Model or Different Model for Verification?

**Answer (from research): ALWAYS different model.**

Self-preference bias is empirically demonstrated (arxiv:2410.21819). Options:
- **Cheaper model judges expensive model** — GPT-4o-mini verifies Claude output. Saves cost, reduces correlation.
- **Same-tier different provider** — Claude verifies GPT-4o output (or vice versa). Maximizes independence.
- **Configurable per mission** — user picks verifier model, defaults to cheaper cross-provider.

**Design question for PRD:** Should `mission_runs` have a `verifier_model` field, or should this be workspace-level config?

### Q2: Scoring Rubric Design

**Options:**
- **Fixed rubric** — same 5 dimensions for every task (completeness, accuracy, relevance, clarity, format)
- **Dynamic rubric** — generated from `success_criteria` JSONB per task
- **Hybrid** — fixed quality dimensions + per-task criteria dimensions

**Recommendation:** Hybrid. Fixed quality dimensions provide baseline comparability across missions. Per-task criteria dimensions ensure specific requirements are checked.

### Q3: Pass/Fail Threshold

**Options:**
- **Hard threshold** — score ≥ 0.7 passes (configurable per mission)
- **Per-criterion threshold** — each criterion must individually pass
- **Weighted aggregate** — some criteria are must-pass (blockers), others are nice-to-have
- **Confidence-gated** — low confidence scores → human review, high confidence → auto-pass/fail

**Design question for PRD:** What is the default threshold? Should it be per-workspace, per-mission, or per-task?

### Q4: Human Override Flow

When verification fails:
1. **Auto-retry** — coordinator retries with feedback from verifier (continuation, not retry — Symphony pattern)
2. **Escalate to human** — flag for review with verifier's reasoning
3. **Accept with warning** — proceed but mark as low-confidence

**Design question for PRD:** How many auto-retries before escalation? Should the coordinator modify the prompt based on verifier feedback?

### Q5: Verification Cost Control

Verification adds 10-30% to task cost. For a 5-task mission at $2, that's $0.20-0.60.

**Options:**
- **Always verify** — every task gets verified
- **Smart verification** — only verify tasks above cost threshold or with downstream dependencies
- **Sample verification** — verify a random subset for learning, verify all for high-stakes missions
- **Tiered** — deterministic checks always, LLM verification only for tasks with subjective criteria

**Design question for PRD:** Should verification be opt-out (default on) or opt-in (default off)?

### Q6: Batch vs Inline Verification

- **Inline** — verify immediately after each task completes. Blocks next task if dependency. Enables fast retry.
- **Batch** — verify all tasks after mission completes. Cheaper (can batch LLM calls). Delays feedback.
- **Hybrid** — verify critical-path tasks inline, verify leaf tasks in batch.

**Recommendation:** Inline for tasks with dependents, batch for terminal tasks.

---

## Section 5: FutureAGI Integration

### What Already Exists

FutureAGI is integrated via `futureagi_service.py` → `agent-opt-worker` HTTP proxy:

| Capability | Endpoint | Status | Reusable for Verification? |
|---|---|---|---|
| Prompt assessment | `POST /assess` | ✅ Production | Partially — assesses prompt quality, not task output quality |
| Live traffic scoring | `POST /score` | ✅ Production | Yes — same pattern: input + output + metrics → scores |
| Prompt optimization | `POST /optimize` | ✅ Production | No — optimizes prompts, not evaluates outputs |
| Safety check | `POST /safety` | ✅ Production | Yes — safety is one verification dimension |

### Extension Strategy

**New worker endpoint: `POST /verify-task`**

```python
# Request
{
    "task_description": "Research EU AI Act requirements",
    "success_criteria": [
        "Covers all 6 risk categories",
        "Includes implementation timelines",
        "Cites specific articles"
    ],
    "task_output": "... agent's output ...",
    "verifier_model": "gpt-4o-mini",  # different from executor
    "scoring_mode": "rubric"  # or "binary", "criteria_match"
}

# Response
{
    "verdict": "pass",  # pass | fail | partial
    "confidence": 0.87,
    "aggregate_score": 0.82,
    "criteria_scores": [
        {"criterion": "Covers all 6 risk categories", "score": 0.9, "met": true, "reasoning": "..."},
        {"criterion": "Includes implementation timelines", "score": 0.7, "met": true, "reasoning": "..."},
        {"criterion": "Cites specific articles", "score": 0.85, "met": true, "reasoning": "..."}
    ],
    "quality_scores": {
        "completeness": 0.85,
        "accuracy": 0.80,
        "clarity": 0.78,
        "relevance": 0.90
    },
    "tokens_used": 1250,
    "cost": 0.003
}
```

**Key design: the verifier prompt**

The verifier prompt is the most critical design artifact. It must:
1. Present success criteria as a rubric (not a checklist — rubrics produce better LLM judgments)
2. Instruct the verifier to reason before scoring (chain-of-thought improves accuracy)
3. Explicitly instruct against position bias and verbosity bias
4. Request confidence alongside scores
5. Use a structured output format (JSON) for reliable parsing

### Data Flow

```
Task Execution Complete
    │
    ▼
VerificationService.verify_task(task_id)
    │
    ├─ Step 1: Deterministic checks (format, length, required sections)
    │   └─ If fails → return FAIL immediately (no LLM cost)
    │
    ├─ Step 2: LLM verification via FutureAGI worker
    │   └─ POST /verify-task with criteria + output
    │
    ├─ Step 3: Store results
    │   └─ mission_events (type="verification", data=scores)
    │   └─ mission_tasks.verification_score, verification_status
    │
    └─ Step 4: Signal coordinator
        ├─ PASS → coordinator advances to next task
        ├─ PARTIAL → coordinator decides: retry with feedback or escalate
        └─ FAIL → coordinator retries (with verifier reasoning) or escalates
```

---

## Section 6: Acceptance Criteria for Full PRD

### Must Have

- [ ] `VerificationService` interface definition with method signatures
- [ ] Verifier prompt template with bias mitigations (position, verbosity, self-preference)
- [ ] Deterministic check registry (format, length, schema, required sections)
- [ ] LLM verification protocol: model selection, rubric construction from success_criteria, scoring output schema
- [ ] FutureAGI worker extension design (`POST /verify-task` endpoint spec)
- [ ] Verdict taxonomy: pass/fail/partial with confidence scores
- [ ] Integration with PRD-101 schema: which fields on `mission_tasks` store verification results
- [ ] Integration with PRD-102 coordinator: how verification verdict drives coordinator decisions (retry, continue, escalate)
- [ ] Retry-with-feedback protocol: how verifier reasoning is fed back to the executing agent
- [ ] Cost model: expected verification cost as % of task cost, with optimization strategies
- [ ] Cross-task consistency checking design (for tasks with shared context)

### Should Have

- [ ] Mapping of existing quality scoring systems (recipe_quality, quality_assessor, report grading) to verification dimensions
- [ ] Configurable threshold design (workspace-level defaults, mission-level overrides, task-level overrides)
- [ ] Verification bypass rules (e.g., skip verification for tasks under $0.10)

### Nice to Have

- [ ] Multi-judge ensemble design (when to use 2-3 judges for high-stakes tasks)
- [ ] Verification result caching (don't re-verify unchanged outputs on retry)
- [ ] Human-in-the-loop calibration: track human override rate to calibrate thresholds

---

## Section 7: Risks & Dependencies

### Risks

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | **LLM-as-judge unreliability on subjective tasks** | Medium | High | Use rubric-based scoring (9% flip rate vs 35% pairwise). Default to human review for low-confidence scores. Track human override rate to measure reliability. |
| 2 | **Verification cost exceeding task cost** | Medium | Medium | Deterministic checks first (zero LLM cost). Skip LLM verification for cheap tasks. Single-judge default, ensemble only for high-stakes. Budget: verification ≤ 30% of task cost. |
| 3 | **Self-preference bias corrupting scores** | High | High | ENFORCE cross-model verification. Never use same model for execution and verification. Default: cheaper model from different provider. |
| 4 | **False positives (bad output passes)** | High | Medium | Multiple criteria checked independently. Any "must-pass" criterion failing = task fails regardless of aggregate. Human review for missions above cost threshold. |
| 5 | **False negatives (good output fails)** | Medium | Medium | Confidence scores + human escalation path. Track false negative rate via human override data. Adjust thresholds based on telemetry (PRD-106). |
| 6 | **Verifier prompt engineering is hard** | High | High | Start with simple rubric template. Iterate based on human override data. Constitutional AI principles as fallback. FutureAGI's existing eval expertise. |
| 7 | **Three existing quality scoring systems create confusion** | Medium | Medium | PRD-103 must either unify them or clearly delineate: recipe_quality = recipe-scoped, quality_assessor = per-execution, verification = against success criteria. |

### Dependencies

| Dependency | From | What's Needed |
|---|---|---|
| `success_criteria` JSONB schema | PRD-101 | Verification input — must be structured enough for rubric generation |
| Coordinator verify step | PRD-102 | Where in the lifecycle verification is called, what the coordinator does with results |
| `verifier_score` telemetry field | PRD-106 | Verification scores must be stored in a format telemetry can aggregate |
| `agent-opt-worker` extensibility | Existing infra | New `/verify-task` endpoint on the worker service |
| Cross-model API access | Existing infra (OpenRouter) | Must be able to call a different model for verification than was used for execution |

### Cross-PRD Connections

- **PRD-104 (Ephemeral Agents):** Contractor agents have no memory/personality — verification is even more important as quality signal since there's no agent reputation to rely on
- **PRD-105 (Budget):** Verification cost must be included in mission budget estimation. Budget enforcement must account for verification retries.
- **PRD-107 (Context Interface):** Verifier may need access to shared context to assess cross-agent consistency. Context interface must support read-only verification access.
