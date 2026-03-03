# PRD-69: Agent Intelligence Layer — Instincts, Evaluation & Strategic Context

**Version:** 1.0
**Status:** Draft
**Priority:** P1
**Author:** Gar Kavanagh + Auto CTO
**Created:** 2026-03-03
**Updated:** 2026-03-03
**Dependencies:** PRD-68 (Progressive Complexity Routing — IN PROGRESS), PRD-50 (Universal Router — COMPLETE), PRD-10 (Workflow Engine — COMPLETE), PRD-05 (Memory & Knowledge — COMPLETE), PRD-06 (Monitoring & Analytics — COMPLETE)
**Branch:** `feat/agent-intelligence-layer`

---

## Executive Summary

Automatos has sophisticated infrastructure: a 7-tier routing engine, a 9-stage orchestration pipeline, a 5-level memory hierarchy, 850+ tools via Composio, and a Progressive Complexity model (PRD-68) that routes atoms through organisms. But the **intelligence layer** — the system that makes all of this learn, evaluate, and improve — is largely stubbed.

Concrete evidence:

| Component | Status | Evidence |
|-----------|--------|----------|
| `PlaybookMiner` | Returns hardcoded demo sequences | `modules/learning/playbooks/miner.py:29-34` — `_fetch_sequences()` returns 4 static lists |
| `feedback/` directory | Empty | `modules/learning/feedback/__init__.py` — 0 lines |
| `patterns/` directory | Empty | `modules/learning/patterns/__init__.py` — 0 lines |
| Agent evaluation fields | 8 of 10 never populated | `core/models/core.py` — `specialization_score`, `reliability_score`, `adaptation_rate`, `collaboration_score` etc. are declared but never written to outside seed data |
| Memory hierarchy | Exists but not integrated into routing | Mem0 stores memories, but `LearningSystemUpdater` doesn't read them back to influence routing or tool selection |
| Cost tracking | Comprehensive but passive | `analytics_engine.py`, `llm_analytics.py` track spend — but cost data never feeds back into model routing or budget enforcement |
| Context compaction | Token-count-based | `context_guard.py` compacts at 80% token usage regardless of workflow phase — mid-reasoning compaction loses critical chain-of-thought |

The `affaan-m/everything-claude-code` repository (58k+ stars, MIT license) has solved many of these problems for the Claude Code CLI: instinct-based learning, structured agent handoffs, eval harnesses, iterative retrieval, and strategic compaction. This PRD adapts 10 of those patterns into Automatos's multi-tenant SaaS architecture.

### What We're Building

An intelligence layer that makes Automatos **learn from every execution and get smarter over time**:

1. **Instinct-Based Learning** — Observe execution patterns, build confidence, auto-promote proven instincts into routing rules and skills
2. **Agent Pipeline Orchestration** — Structured handoff documents and verdict systems so multi-agent workflows pass real context instead of dumping raw output
3. **Cost-Aware Evaluation** — Route complexity tiers to appropriate models, enforce budgets, and measure agent quality with pass@k scoring
4. **Strategic Context Management** — Iterative retrieval that refines queries and phase-aware compaction that respects workflow boundaries

### What We're NOT Building

- A new routing engine (PRD-68 handles that)
- A new memory system (Mem0 integration is complete)
- A new pipeline architecture (PRD-59 pipeline.py is solid)
- Human-in-the-loop eval UI (Phase 3 future work)

---

## 1. Current State

### 1.1 Learning System — Wired but Empty

```
modules/learning/
├── engine/
│   └── core.py          ← LearningSystemUpdater — updates agent performance metrics,
│                            but only writes avg_response_time and total_executions.
│                            8 other eval fields are Column(Float, nullable=True) in core.py
│                            and NEVER get written to.
├── patterns/
│   └── __init__.py      ← EMPTY. Intended for pattern recognition.
├── feedback/
│   └── __init__.py      ← EMPTY. Intended for feedback loops.
├── playbooks/
│   └── miner.py         ← PlaybookMiner._fetch_sequences() returns 4 hardcoded lists.
│                            mine() counts n-gram frequencies over those 4 lists.
└── tests/
    ├── test_learning_system.py
    └── test_playbook_miner.py
```

The `LearningSystemUpdater.update_from_execution()` is called from the pipeline but its downstream effects are minimal — it updates `total_executions`, `avg_response_time`, and `success_rate` on the Agent model. The remaining evaluation fields (`specialization_score`, `reliability_score`, `adaptation_rate`, `collaboration_score`, `avg_quality_score`, `performance_score`, `readiness_score`, `discriminatory_power`) are never computed.

### 1.2 Pipeline — Composable but No Inter-Agent Context

`modules/orchestrator/pipeline.py` defines a composable stage executor with `StageStatus`, `ErrorStrategy`, and SSE event emission. Stages receive a `WorkflowContext` and return a `StageResult`. But when multiple agents participate in a workflow:

- Agent B receives Agent A's raw text output, not structured context
- No verdict system exists — there's no way for an agent to say "this needs more work" vs. "ship it"
- Pipeline progress is emitted as SSE events but with stage-level granularity, not reasoning-step granularity

### 1.3 Cost Tracking — Tracks but Doesn't Act

The platform tracks LLM spend per workspace/agent/model through `llm_analytics.py`, `analytics_engine.py`, and `statistics.py`. But:

- No budget limits exist — a runaway workflow can burn unlimited tokens
- Cost data doesn't influence model selection — PRD-68's complexity tiers map to pipeline depth but not model cost
- No quality evaluation — we track whether executions complete, not whether they're *good*

### 1.4 Context Engineering — One-Shot Retrieval, Token-Based Compaction

`modules/orchestrator/stages/context_engineering.py` does single-pass RAG retrieval: one query → one set of results → inject into prompt. No refinement loop if results are poor.

`core/context_guard.py` compacts at 80% token usage using a flat strategy: summarize older turns, keep recent ones. It doesn't know about workflow phases — compaction mid-reasoning is as likely as compaction between phases.

---

## 2. Feature Area A: Instinct-Based Learning

**Adapted from:** `everything-claude-code/skills/continuous-learning-v2/SKILL.md`, `commands/evolve.md`, `commands/skill-create.md`

### Concept

An **instinct** is an observed execution pattern that the system learns through repetition and reinforcement. Unlike hardcoded rules, instincts:

- Start with low confidence (0.3) and strengthen through repeated observation
- Decay over time if not reinforced (prevents stale patterns)
- Auto-promote into concrete routing rules or skills at high confidence (≥ 0.8)
- Are workspace-scoped (tenant A's patterns don't bleed into tenant B)

### A.1 Instinct Data Model

**New file:** `orchestrator/core/models/instincts.py`

```python
class Instinct(Base):
    __tablename__ = "instincts"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)

    # What was observed
    trigger_pattern = Column(String, nullable=False)     # e.g. "user asks about calendar"
    trigger_type = Column(String, nullable=False)         # "intent" | "tool_sequence" | "error_recovery" | "routing"

    # What worked
    successful_action = Column(String, nullable=False)    # e.g. "route to composio:google_calendar"
    action_metadata = Column(JSON, nullable=True)         # tool params, agent config, model used

    # Confidence tracking
    confidence = Column(Float, default=0.3, nullable=False)
    observation_count = Column(Integer, default=1, nullable=False)
    success_count = Column(Integer, default=1, nullable=False)
    failure_count = Column(Integer, default=0, nullable=False)
    last_observed = Column(DateTime, nullable=False)
    last_reinforced = Column(DateTime, nullable=True)

    # Lifecycle
    status = Column(String, default="active", nullable=False)  # "active" | "promoted" | "decayed" | "archived"
    promoted_to = Column(String, nullable=True)            # "routing_rule:id" | "skill:id" when promoted

    # Audit
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
```

**Migration:** Add `instincts` table. Alembic migration in `orchestrator/alembic/versions/`.

### A.2 Observation Pipeline

**New file:** `orchestrator/modules/learning/patterns/instinct_observer.py`

The observer hooks into `LearningSystemUpdater.update_from_execution()` and extracts instincts from completed workflow executions:

```
Execution completes
  → LearningSystemUpdater.update_from_execution()
    → InstinctObserver.observe(execution_data)
      → Extract patterns:
          - Intent → tool mapping (which tools solved which intents)
          - Error → recovery mapping (what retries/fallbacks worked)
          - Agent → task affinity (which agents excel at which task types)
          - Routing decisions (which complexity tier was right)
      → For each pattern:
          - If matching instinct exists: reinforce (bump confidence, update counts)
          - If no match: create new instinct at confidence 0.3
```

**Key design decisions:**
- Observer is **read-only on the execution path** — it processes data after execution completes, never blocks the hot path
- Pattern extraction uses the existing `SubtaskExecution` metadata (tool calls, durations, quality scores) — no new data collection needed
- Deduplication by `(workspace_id, trigger_type, trigger_pattern, successful_action)` compound uniqueness

### A.3 Confidence Scoring Engine

**New file:** `orchestrator/modules/learning/feedback/confidence.py`

```
confidence_new = (
    base_confidence                          # 0.3 for new instincts
    + (success_rate × reinforcement_weight)  # success_count / observation_count × 0.1
    - (time_decay × decay_rate)              # days_since_last_reinforced × 0.005
)

Clamped to [0.1, 0.95]
```

**Confidence lifecycle:**
- **0.3** — New observation, unverified
- **0.5** — Seen 3+ times with >70% success rate
- **0.7** — Reliable pattern, starts influencing suggestions
- **0.8** — Promotion threshold — auto-promoted to routing rule or skill
- **0.9** — Maximum practical confidence (never reaches 1.0 — always room for learning)
- **< 0.2** — Decayed, marked `status="decayed"` and excluded from queries

**Decay schedule:** Confidence decreases by 0.005 per day without reinforcement. An instinct that isn't reinforced for 60 days decays from 0.8 → 0.5. After 120 days without reinforcement, it decays below 0.2 and is archived.

### A.4 Auto-Promotion (Instincts → Skills/Routing Rules)

**New file:** `orchestrator/modules/learning/patterns/instinct_promoter.py`

When an instinct reaches confidence ≥ 0.8 with ≥ 10 observations:

| Trigger Type | Promotes To | Effect |
|-------------|-------------|--------|
| `intent` | Routing rule in `routing` table | AutoBrain uses this to bypass regex matching |
| `tool_sequence` | Playbook template | Suggested as a workflow recipe |
| `error_recovery` | Error handler in pipeline | Auto-retry with the learned fallback |
| `routing` | Complexity tier override | Overrides default tier for known patterns |

Promotion is **reversible** — if the promoted rule starts failing (tracked via the same observation pipeline), the instinct is demoted back to `active` status and confidence is reduced.

### A.5 Evolve Endpoint

**Adapted from:** `everything-claude-code/commands/evolve.md`

**New API endpoint:** `POST /api/v1/learning/evolve`

Clusters related instincts and proposes higher-order patterns:

```
Request:
  POST /api/v1/learning/evolve
  { "workspace_id": 123, "min_confidence": 0.5 }

Response:
  {
    "clusters": [
      {
        "theme": "Calendar management",
        "instincts": [instinct_id_1, instinct_id_2, instinct_id_3],
        "proposed_skill": {
          "name": "calendar_assistant",
          "description": "Route calendar-related queries to Google Calendar tools",
          "trigger_patterns": ["schedule meeting", "check calendar", "free time"],
          "confidence": 0.85  // avg of cluster
        },
        "action": "promote_to_skill"  // or "merge_instincts" or "archive_weak"
      }
    ],
    "total_instincts_analyzed": 47,
    "clusters_found": 3
  }
```

Clustering uses LLM-based semantic grouping (not k-means — patterns are natural language, not vectors). The LLM receives instinct descriptions and groups by functional theme.

### A.6 Populate Agent Evaluation Fields

**Modify:** `orchestrator/modules/learning/engine/core.py`

Wire the 8 unused evaluation fields on the Agent model:

| Field | Computation | Source Data |
|-------|------------|-------------|
| `avg_quality_score` | Weighted moving average of `QualityAssessor` scores | `quality_assessor.py` stage output |
| `specialization_score` | Entropy of task-type distribution (low entropy = specialist) | `subtask_executions` — count tasks by type per agent |
| `reliability_score` | Success rate weighted by recency | `success_count / total` with exponential recency weighting |
| `adaptation_rate` | Improvement slope over last 20 executions | Linear regression on quality scores over time |
| `collaboration_score` | Success rate in multi-agent workflows vs. solo | Compare quality when agent works alone vs. with others |
| `performance_score` | Composite: 40% quality + 30% speed + 30% token efficiency | Blend of existing metrics |
| `readiness_score` | Current availability × reliability × recency | Is this agent warmed up and reliable right now? |
| `discriminatory_power` | Variance in quality across task types | High = good at some things, bad at others. Low = consistent. |

These fields feed into `agent_selector.py` and `llm_agent_selector.py` for smarter agent assignment.

---

## 3. Feature Area B: Agent Pipeline Orchestration

**Adapted from:** `everything-claude-code/agents/planner.md`, `agents/code-reviewer.md`

### B.1 Handoff Document Schema

**Modify:** `orchestrator/modules/orchestrator/pipeline.py`

When Agent A completes a stage and Agent B picks up, the pipeline currently passes raw text output. Replace with a structured `HandoffDocument`:

```python
@dataclass
class HandoffDocument:
    """Structured context passed between pipeline stages/agents."""

    # Source
    from_stage: str                    # e.g. "task_decomposer"
    from_agent_id: Optional[int]       # Agent that produced this

    # Core content
    summary: str                       # 1-2 sentence executive summary
    detailed_output: str               # Full output text

    # Structured data
    key_decisions: List[str]           # Decisions made in this stage
    open_questions: List[str]          # Unresolved questions for next stage
    artifacts: Dict[str, Any]          # Named outputs (code, data, configs)

    # Quality signals
    confidence: float                  # How confident is this output (0-1)
    quality_score: Optional[float]     # From QualityAssessor if available

    # Context for next stage
    suggested_approach: Optional[str]  # How the next stage should proceed
    blockers: List[str]                # Known blockers

    # Metadata
    token_cost: int                    # Tokens consumed producing this
    duration_ms: int                   # Wall-clock time
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

The `WorkflowContext` (already the pipeline's shared state) gains a `handoff_chain: List[HandoffDocument]` field. Each stage appends its handoff document. Downstream stages can read the full chain or just the most recent handoff.

### B.2 Pipeline Verdict System

**Modify:** `orchestrator/modules/orchestrator/pipeline.py`

Add a verdict protocol so stages can express outcomes beyond success/failure:

```python
class StageVerdict(Enum):
    CONTINUE = "continue"       # Output is good, proceed to next stage
    NEEDS_WORK = "needs_work"   # Output needs revision — loop back to this stage
    BLOCKED = "blocked"         # Cannot proceed — missing data/tools/permissions
    SHIP = "ship"               # Output is final — skip remaining stages and deliver
    ESCALATE = "escalate"       # Needs human review before proceeding
```

**Pipeline behavior per verdict:**

| Verdict | Pipeline Action |
|---------|----------------|
| `CONTINUE` | Advance to next stage (current behavior) |
| `NEEDS_WORK` | Re-execute current stage with feedback from `HandoffDocument.open_questions`. Max 2 retries before `ESCALATE`. |
| `BLOCKED` | Log blocker, emit SSE event, skip to next stage with `ErrorStrategy.SKIP` |
| `SHIP` | Short-circuit remaining stages, deliver result immediately |
| `ESCALATE` | Pause pipeline, emit SSE event for human review. Resume on API call. |

This replaces the current binary `StageStatus.COMPLETED` / `StageStatus.FAILED` for inter-stage communication. `StageStatus` remains for lifecycle tracking (pending/running/completed/failed).

### B.3 Pipeline Visualization via SSE

**Modify:** `orchestrator/modules/orchestrator/pipeline.py`

Extend existing SSE event emission with richer data:

```json
{
  "event": "pipeline:stage_verdict",
  "data": {
    "stage": "quality_assessor",
    "verdict": "needs_work",
    "confidence": 0.6,
    "open_questions": ["Missing error handling for edge case X"],
    "retry_count": 1,
    "handoff_summary": "Code review found 2 issues, requesting revision"
  }
}
```

Current SSE events emit `stage_started` and `stage_completed`. Add `stage_verdict`, `stage_retry`, `pipeline_short_circuit` (for SHIP), and `pipeline_escalation` event types.

### B.4 Context Mode Switching

**Adapted from:** `everything-claude-code/contexts/dev.md`, `contexts/review.md`, `contexts/research.md`

**New file:** `orchestrator/consumers/chatbot/context_modes.py`

Context modes adjust the system prompt, tool selection, and evaluation criteria based on the user's current intent:

| Mode | System Prompt Emphasis | Tool Priority | Eval Criteria |
|------|----------------------|---------------|---------------|
| `dev` | Build features, write code, execute tasks | Code tools, Composio actions, platform tools | Correctness, completion |
| `review` | Analyze existing work, find issues, suggest improvements | CodeGraph search, RAG, analytics | Thoroughness, accuracy |
| `research` | Gather information, compare options, summarize findings | Web search, RAG, document retrieval | Breadth, source quality |
| `plan` | Create plans, estimate effort, identify risks | Platform awareness, memory, analytics | Completeness, feasibility |

**Activation:** Auto-detected by AutoBrain based on intent (e.g., "review this PR" → review mode, "build a webhook handler" → dev mode). Can also be set explicitly via chat command `/mode review` or API parameter.

**Integration point:** Context modes inject a mode-specific system prompt prefix before the personality prompt in `service.py`. They also adjust `tool_hints` (PRD-68) to prioritize mode-relevant tools.

---

## 4. Feature Area C: Cost-Aware Evaluation

**Adapted from:** `everything-claude-code/rules/common/performance.md`, `skills/eval-harness/SKILL.md`

### C.1 Complexity-Based Model Routing

**Modify:** `orchestrator/core/llm/manager.py`, `orchestrator/consumers/chatbot/service.py`

PRD-68 defines 5 complexity tiers (ATOM → ORGANISM) and maps them to pipeline depth. This feature adds **model cost tiers** to that mapping:

| Complexity | Pipeline | Model Category | Typical Models | Cost/1K tokens |
|-----------|----------|---------------|----------------|----------------|
| ATOM | Direct response | `economy` | Haiku, GPT-4o-mini, Gemini Flash | $0.001-0.005 |
| MOLECULE | Single agent | `standard` | Sonnet, GPT-4o, Gemini Pro | $0.01-0.03 |
| CELL | Agent + tools | `standard` | Sonnet, GPT-4o, Gemini Pro | $0.01-0.03 |
| ORGAN | Multi-agent | `premium` | Opus, GPT-4.5, Gemini Ultra | $0.05-0.15 |
| ORGANISM | Full swarm | `premium` | Opus, GPT-4.5, Gemini Ultra | $0.05-0.15 |

**Implementation:** Add `model_category` field to `ComplexityAssessment` (PRD-68's dataclass). `LLMManager.get_model()` accepts a `category` parameter and selects from the workspace's configured models for that tier. Falls back to `config.LLM_MODEL` if no category-specific model is configured.

**Workspace model configuration:** New `workspace_model_preferences` table:

```python
class WorkspaceModelPreference(Base):
    __tablename__ = "workspace_model_preferences"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    category = Column(String, nullable=False)  # "economy" | "standard" | "premium"
    model_id = Column(String, nullable=False)   # OpenRouter model ID
    max_tokens = Column(Integer, nullable=True)  # Override for this category

    __table_args__ = (UniqueConstraint("workspace_id", "category"),)
```

### C.2 Budget Tracking with Alerts

**New file:** `orchestrator/core/services/cost_governor.py`

```
Workspace budget configuration:
  - daily_limit_usd: float (default: no limit)
  - monthly_limit_usd: float (default: no limit)
  - alert_threshold_pct: float (default: 0.8 = alert at 80% spend)
  - overage_action: "alert" | "downgrade" | "block"

On each LLM call:
  1. CostGovernor.check_budget(workspace_id, estimated_cost)
  2. If under budget → proceed
  3. If over alert threshold → emit warning SSE event
  4. If over limit:
     - "alert": proceed but notify admin
     - "downgrade": force model_category down one tier (premium→standard→economy)
     - "block": reject the request with budget_exceeded error
```

Budget data sources: existing `llm_usage` table tracked by `analytics_engine.py`. No new tracking needed — just aggregation + enforcement.

### C.3 Eval Harness with pass@k Scoring

**Adapted from:** `everything-claude-code/skills/eval-harness/SKILL.md`

**New file:** `orchestrator/modules/orchestrator/stages/eval_harness.py`

For high-complexity tasks (ORGAN/ORGANISM), optionally run the agent pipeline k times and select the best result:

```
eval_harness(task, k=3):
  results = []
  for i in range(k):
    result = await pipeline.execute(task, temperature=0.7 + i*0.1)
    score = quality_assessor.score(result)
    results.append((result, score))

  # Select best by quality score
  best = max(results, key=lambda r: r[1])

  # Record pass@k metrics
  pass_at_1 = 1 if results[0][1] >= threshold else 0
  pass_at_k = 1 if any(r[1] >= threshold for r in results) else 0

  return best, {"pass@1": pass_at_1, "pass@k": pass_at_k, "scores": [r[1] for r in results]}
```

**When to use:** Controlled by workspace setting. Default is `k=1` (no eval harness). Premium workspaces can set `k=3` for critical workflows. Budget enforcement (C.2) applies — k runs cost k× tokens.

**Quality threshold:** Configurable per workspace, default 0.7 (on the 0-1 scale from `quality_assessor.py`).

---

## 5. Feature Area D: Strategic Context Management

**Adapted from:** `everything-claude-code/skills/iterative-retrieval/SKILL.md`, `skills/strategic-compact/SKILL.md`

### D.1 Iterative Retrieval

**Modify:** `orchestrator/modules/orchestrator/stages/context_engineering.py`

Replace single-pass RAG with a DISPATCH → EVALUATE → REFINE → LOOP cycle:

```
Cycle 1:
  DISPATCH: Generate initial retrieval query from user intent
  EVALUATE: Score retrieved chunks (relevance 0-1, coverage 0-1)
  → If relevance ≥ 0.7 AND coverage ≥ 0.7: DONE
  REFINE: Generate refined query based on what's missing
  → Loop (max 3 cycles)

Cycle 2:
  DISPATCH: Refined query (e.g., add specificity, broaden scope, change keywords)
  EVALUATE: Score new chunks + previous chunks combined
  → If thresholds met: DONE
  REFINE: Further refinement or different retrieval source

Cycle 3 (terminal):
  DISPATCH: Final attempt with maximum breadth
  EVALUATE: Accept whatever we have
  → Return best available context regardless of score
```

**Scoring functions:**
- **Relevance:** Cosine similarity between query embedding and chunk embeddings (already computed by RAG pipeline)
- **Coverage:** LLM-assessed — "Does this context contain enough information to answer the query?" Binary yes/no with a brief explanation

**Performance guard:** Each cycle adds ~500ms (one RAG query + one LLM coverage check). Max 3 cycles = ~1.5s additional latency. Only activated for CELL+ complexity (ATOM and MOLECULE use single-pass retrieval).

### D.2 Phase-Aware Compaction

**Modify:** `orchestrator/core/context_guard.py`

Current behavior: Compact at 80% token usage regardless of what's happening in the pipeline.

New behavior: Compact at **workflow phase boundaries**, preserving the full context within each phase.

```
Phase boundary detection:
  - Between pipeline stages (StageResult emitted)
  - Between agent turns in multi-agent workflows
  - Between user messages in conversational flows
  - NEVER mid-reasoning (within a single LLM call's context)

Compaction strategy by content type:
  - Tool call results: Summarize to key outputs (keep structured data, drop verbose logs)
  - Agent reasoning: Preserve final conclusions, compress intermediate reasoning
  - User messages: Keep all (they're the source of truth)
  - System prompts: Never compact (they define behavior)
  - Handoff documents: Keep summary + key_decisions, drop detailed_output after 2 phases
```

**Integration with HandoffDocument (B.1):** Handoff documents provide natural compaction units. After a stage completes, its handoff document's `summary` replaces the full stage output in the conversation context. The `detailed_output` is available in the `handoff_chain` if a downstream stage needs to look back.

**Token budget allocation:**

```
Total context budget: model_max_tokens × 0.80 (keep 20% for response)

Allocation:
  - System prompt: fixed (measured at startup)
  - Current phase context: up to 50% of remaining budget
  - Handoff chain summaries: up to 25% of remaining budget
  - Memory/RAG context: up to 25% of remaining budget

If over budget:
  1. Compress oldest handoff chain entries (keep only summary)
  2. Reduce RAG context (fewer chunks)
  3. Summarize current phase's older turns
  4. NEVER truncate current reasoning turn
```

---

## 6. Phased Implementation

### Phase 1 — MVP (Weeks 1-3)

Core learning loop + basic pipeline improvements + cost routing.

| Feature | Scope | Files |
|---------|-------|-------|
| A.1 | Instinct data model + migration | `core/models/instincts.py` (NEW), alembic migration |
| A.2 | Observation pipeline (intent→tool + routing patterns only) | `modules/learning/patterns/instinct_observer.py` (NEW), `modules/learning/engine/core.py` (MODIFY) |
| A.3 | Confidence scoring (basic: success rate + time decay) | `modules/learning/feedback/confidence.py` (NEW) |
| B.1 | HandoffDocument schema + wire into pipeline | `modules/orchestrator/pipeline.py` (MODIFY) |
| B.2 | CONTINUE/SHIP verdicts only (no NEEDS_WORK loops yet) | `modules/orchestrator/pipeline.py` (MODIFY) |
| C.1 | Model category on ComplexityAssessment + LLMManager routing | `core/llm/manager.py` (MODIFY), `consumers/chatbot/service.py` (MODIFY) |
| D.1 | Simplified iterative retrieval (2 cycles max, relevance-only scoring) | `modules/orchestrator/stages/context_engineering.py` (MODIFY) |

**Phase 1 success criteria:**
- Instincts are being created from real executions
- HandoffDocuments are passing between pipeline stages
- ATOM tasks use economy models, ORGANISM tasks use premium models
- RAG retrieval does at least 1 refinement cycle when initial results score below 0.7

### Phase 2 — Enhancement (Weeks 4-6)

Auto-promotion, full verdict system, budget enforcement, phase-aware compaction.

| Feature | Scope | Files |
|---------|-------|-------|
| A.4 | Auto-promotion (instincts → routing rules) | `modules/learning/patterns/instinct_promoter.py` (NEW) |
| A.5 | Evolve endpoint (cluster + propose skills) | `api/learning.py` (MODIFY) |
| A.6 | Populate all 8 agent evaluation fields | `modules/learning/engine/core.py` (MODIFY) |
| B.2+ | NEEDS_WORK + BLOCKED + ESCALATE verdicts with retry loops | `modules/orchestrator/pipeline.py` (MODIFY) |
| B.3 | Enriched SSE events for pipeline visualization | `modules/orchestrator/pipeline.py` (MODIFY) |
| B.4 | Context modes (dev/review/research/plan) | `consumers/chatbot/context_modes.py` (NEW), `consumers/chatbot/service.py` (MODIFY) |
| C.2 | Budget tracking + alerts + auto-downgrade | `core/services/cost_governor.py` (NEW), `core/models/workspaces.py` (MODIFY) |
| C.3 | Eval harness with pass@k scoring | `modules/orchestrator/stages/eval_harness.py` (NEW) |
| D.1+ | Full iterative retrieval (3 cycles, coverage scoring) | `modules/orchestrator/stages/context_engineering.py` (MODIFY) |
| D.2 | Phase-aware compaction | `core/context_guard.py` (MODIFY) |

**Phase 2 success criteria:**
- High-confidence instincts auto-promote to routing rules and are used by AutoBrain
- Agent eval fields are populated and visible in agent analytics
- Budget limits enforce model downgrades when workspace spend exceeds threshold
- Context compaction never occurs mid-reasoning

### Phase 3 — Future Work (Not Scoped)

| Feature | Description |
|---------|-------------|
| Multi-model trust rules | Per-model confidence tracking — "trust Opus for code review, use Haiku for summarization" |
| Instinct marketplace | Share high-confidence instincts across workspaces (opt-in) |
| Human-in-loop eval | UI for reviewing ESCALATE verdicts and providing feedback |
| Context mode UI | Frontend mode selector with visual indicator |
| Eval dashboard | Visualize pass@k metrics, agent performance trends, instinct lifecycle |
| Cross-workspace instinct federation | Platform-level instincts learned from aggregate patterns (privacy-preserving) |

---

## 7. File Impact Table

### New Files

| File | Feature | Description |
|------|---------|-------------|
| `orchestrator/core/models/instincts.py` | A.1 | Instinct SQLAlchemy model |
| `orchestrator/modules/learning/patterns/instinct_observer.py` | A.2 | Observation pipeline — extracts instincts from executions |
| `orchestrator/modules/learning/feedback/confidence.py` | A.3 | Confidence scoring + decay engine |
| `orchestrator/modules/learning/patterns/instinct_promoter.py` | A.4 | Auto-promotion logic |
| `orchestrator/consumers/chatbot/context_modes.py` | B.4 | Context mode definitions + switching |
| `orchestrator/core/services/cost_governor.py` | C.2 | Budget tracking + enforcement |
| `orchestrator/modules/orchestrator/stages/eval_harness.py` | C.3 | pass@k evaluation harness |
| `orchestrator/alembic/versions/xxx_add_instincts_table.py` | A.1 | Database migration |
| `orchestrator/alembic/versions/xxx_add_workspace_model_prefs.py` | C.1 | Database migration |

### Modified Files

| File | Feature | Change |
|------|---------|--------|
| `orchestrator/modules/learning/engine/core.py` | A.2, A.6 | Wire InstinctObserver into update loop; compute all 8 eval fields |
| `orchestrator/modules/orchestrator/pipeline.py` | B.1, B.2, B.3 | Add HandoffDocument, StageVerdict, enriched SSE events |
| `orchestrator/core/llm/manager.py` | C.1 | Add `category` parameter to model selection |
| `orchestrator/consumers/chatbot/service.py` | C.1, B.4 | Route complexity tier to model category; integrate context modes |
| `orchestrator/modules/orchestrator/stages/context_engineering.py` | D.1 | Iterative retrieval loop with refinement |
| `orchestrator/core/context_guard.py` | D.2 | Phase-aware compaction strategy |
| `orchestrator/api/learning.py` | A.5 | Add `/evolve` endpoint |
| `orchestrator/core/models/workspaces.py` | C.2 | Add budget fields to workspace model |

### everything-claude-code Cross-References

| Automatos Feature | everything-claude-code Source | Adaptation Notes |
|-------------------|-------------------------------|------------------|
| A.1-A.4 (Instincts) | `skills/continuous-learning-v2/SKILL.md` | File-based → PostgreSQL; single-user → multi-tenant with workspace scoping |
| A.5 (Evolve) | `commands/evolve.md` | CLI command → REST API endpoint; local clustering → LLM-based semantic grouping |
| A.6 (Eval fields) | `commands/skill-create.md` | Skill quality metrics → agent performance metrics mapped to existing DB columns |
| B.1-B.2 (Handoffs/Verdicts) | `agents/planner.md`, `agents/code-reviewer.md` | Agent-specific handoff → generic pipeline protocol; text verdicts → enum-based system |
| B.4 (Context modes) | `contexts/dev.md`, `contexts/review.md`, `contexts/research.md` | Static context files → dynamic mode switching with tool priority adjustment |
| C.1-C.2 (Cost routing) | `rules/common/performance.md` | Manual model selection guidance → automated complexity-based routing with budget enforcement |
| C.3 (Eval harness) | `skills/eval-harness/SKILL.md` | Local eval script → pipeline-integrated stage with quality_assessor scoring |
| D.1 (Iterative retrieval) | `skills/iterative-retrieval/SKILL.md` | File-based retrieval → RAG pipeline integration with embedding-based relevance scoring |
| D.2 (Strategic compaction) | `skills/strategic-compact/SKILL.md` | Token-based → phase-boundary-based; conversation context → pipeline HandoffDocument chain |

---

## 8. API Surface

### New Endpoints

```
POST   /api/v1/learning/evolve              — Cluster instincts, propose skills (A.5)
GET    /api/v1/learning/instincts            — List instincts for workspace (with filters)
GET    /api/v1/learning/instincts/:id        — Get instinct details
DELETE /api/v1/learning/instincts/:id        — Archive an instinct
POST   /api/v1/learning/instincts/:id/demote — Manually demote a promoted instinct

GET    /api/v1/workspace/budget              — Current budget status + spend
PUT    /api/v1/workspace/budget              — Set budget limits
GET    /api/v1/workspace/model-preferences   — Model category preferences
PUT    /api/v1/workspace/model-preferences   — Set model per category
```

### Modified Endpoints

```
GET    /api/v1/agents/:id                    — Now includes all 8 eval fields populated (A.6)
GET    /api/v1/analytics/agents              — Agent analytics includes eval field trends
POST   /api/v1/chat                          — Accepts optional `context_mode` parameter (B.4)
```

---

## 9. Database Changes

### New Tables

**`instincts`** — See A.1 for full schema. Indexed on `(workspace_id, status)` and `(workspace_id, trigger_type, confidence DESC)`.

**`workspace_model_preferences`** — See C.1 for schema. One row per workspace per category.

### Modified Tables

**`workspaces`** — Add columns:
- `budget_daily_limit_usd` (Float, nullable, default null = unlimited)
- `budget_monthly_limit_usd` (Float, nullable, default null = unlimited)
- `budget_alert_threshold_pct` (Float, default 0.8)
- `budget_overage_action` (String, default "alert")

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Instinct observation adds latency to execution path | Medium | Medium | Observer runs async after execution completes — zero impact on user-facing latency |
| Bad instincts get promoted and degrade routing | Low | High | Promotion requires ≥10 observations + ≥0.8 confidence. Promoted rules are monitored and auto-demoted on failure. |
| Iterative retrieval adds 1-1.5s to context engineering | High | Low | Only activated for CELL+ complexity. ATOM/MOLECULE use single-pass (no regression for simple queries). |
| Budget enforcement blocks critical workflows | Low | High | Default overage action is "alert" not "block". Admin can always override. Budget applies per-workspace, not per-request. |
| Phase-aware compaction is more complex than token-based | Medium | Medium | Implement incrementally: Phase 1 keeps current 80% threshold, Phase 2 adds phase boundary detection. Fallback to token-based if boundary detection fails. |
| eval harness (pass@k) multiplies cost by k | Medium | Medium | Default k=1 (no overhead). Only enabled explicitly per workspace. Budget enforcement (C.2) caps total spend regardless. |

---

## 11. Success Metrics

### Learning (Feature Area A)
- **Instinct creation rate:** ≥ 5 new instincts per 100 executions within first week
- **Promotion rate:** ≥ 10% of instincts reach promotion threshold within 30 days
- **Routing accuracy improvement:** Promoted instincts should show ≥ 90% success rate post-promotion
- **Agent eval field population:** All 8 fields populated for every agent with ≥ 5 executions

### Pipeline (Feature Area B)
- **Handoff utilization:** ≥ 80% of multi-stage workflows use structured HandoffDocuments
- **SHIP verdict frequency:** ≥ 15% of pipelines short-circuit via SHIP (indicating efficiency)
- **NEEDS_WORK resolution rate:** ≥ 70% of NEEDS_WORK verdicts resolve within max retries

### Cost (Feature Area C)
- **Model cost reduction:** ≥ 30% reduction in average cost per ATOM query (economy models vs. current)
- **Budget adherence:** 0 uncontrolled spend events after budget enforcement is active
- **Eval quality:** pass@3 shows ≥ 20% improvement over pass@1 for ORGAN+ tasks

### Context (Feature Area D)
- **Retrieval quality:** Iterative retrieval achieves ≥ 0.7 relevance on ≥ 85% of CELL+ queries (vs. current ~60% estimated)
- **Compaction safety:** 0 instances of mid-reasoning compaction after D.2 is deployed
- **Token efficiency:** ≥ 15% reduction in wasted context tokens through phase-aware allocation
