# PRD-82C: Parallel Execution, Intelligent Decomposition & Budget Governance

**Status:** Draft
**Date:** 2026-03-24
**Authors:** Gerard Kavanagh + Claude
**Dependencies:** PRD-82A (built), PRD-82B (partial), PRD-102, PRD-104, PRD-105, PRD-106
**Branch:** TBD

---

## TL;DR

Missions today are sequential playbooks with an LLM-generated plan. PRD-82C makes them what they were designed to be: parallel, budget-aware, intelligently decomposed multi-agent operations. This PRD consolidates all unfinished work from 82A/B and unbuilt specs from PRDs 102-106 into one deliverable.

---

## 1. Problem Statement

### 1.1 What's broken

A user asked for a 4,000-word research paper. The planner generated a single "write the paper" task. The agent hit `max_tokens`, output was truncated mid-sentence, verification failed. The user had to start over from scratch.

This failure pattern is systemic, not incidental:

| Problem | Root Cause | Impact |
|---------|-----------|--------|
| Single massive tasks | Planner prompt says "sequential tasks" with no decomposition guidance | Truncation, verification failure, wasted tokens |
| No parallel execution | `has_active_task()` hard-blocks dispatch regardless of `max_concurrent` | 3x slower than necessary for parallelizable work |
| No budget enforcement | `can_afford()` exists but never called; soft warning at 150% only | 402 errors mid-mission when credits run out |
| No synthesis step | `TaskType.SYNTHESIS` enum defined, zero executor logic | Parallel outputs can't be merged |
| Templates are sequential | All 4 templates chain `depends_on = [previous_task]` | Even template-matched missions run serially |
| No task sizing intelligence | Budget estimate = `num_tasks * 2000` tokens flat | Large tasks get same budget as small ones |

### 1.2 What works (don't break it)

These components are wired and tested — 82C builds ON them:

- **State machine & transitions** (orchestration_enums.py) — 11 task states, 10 run states, strict transition graph
- **Dependency DAG** (OrchestrationTaskDependency table + DependencyResolver) — validates acyclicity, resolves ready tasks
- **Agent matcher** (agent_matcher.py) — 5-factor weighted scoring, 0.4 threshold
- **Verification service** (verification.py) — deterministic checks + cross-model LLM judge + caching
- **Cross-task consistency** (verification.py) — ConsistencyResult/Issue, runs on mission finalization
- **Stall detection** (reconciler.py) — 60s/300s timeouts, recovery to QUEUED
- **Shared context field** (PRD-108) — per-mission Qdrant collection, inject/query/decay/reinforce
- **Run Trace UI** (mission-dag-canvas.tsx) — DAG visualization, activity feed, status badges
- **Optimistic locking** — version_id on all state transitions prevents double-dispatch
- **Event sourcing** — append-only orchestration_events for audit trail

### 1.3 What this PRD delivers

When 82C ships, a "write a research paper" mission will:
1. **Decompose** into parallel research tasks + sequential drafting + synthesis merge + review
2. **Dispatch** up to 3 tasks concurrently (configurable per mission)
3. **Enforce budget** — refuse to dispatch if budget would be exceeded
4. **Auto-generate synthesis** — merge parallel outputs into unified document
5. **Size tasks intelligently** — no single task exceeds token limits

---

## 2. User Stories

### US-001: Parallel Task Dispatch
**As a** mission coordinator, **I want** independent tasks to execute concurrently **so that** missions complete faster and agents aren't idle.

**Acceptance criteria:**
- Dispatcher respects `max_concurrent` field on OrchestrationRun (default: 3)
- Independent tasks (no shared dependencies) dispatch simultaneously
- Tasks with unmet dependencies remain PENDING until upstream VERIFIED
- DAG visualization updates in real-time showing parallel branches
- No regression: sequential missions (max_concurrent=1) still work

### US-002: Intelligent Decomposition with Parallel Groups
**As a** planner, **I want** to generate task DAGs with parallel branches **so that** independent work happens simultaneously.

**Acceptance criteria:**
- Planner system prompt instructs LLM to identify parallelizable subtasks
- LLM output includes `parallel_group` field on tasks (tasks in same group have no interdependencies)
- Templates generate parallel groups (e.g., 3 research tasks in parallel, all feeding into 1 synthesis)
- Validation rejects plans where parallel-grouped tasks have dependencies on each other
- Tasks within a parallel group get the same `sequence_number`

### US-003: Synthesis Task Auto-Generation
**As a** coordinator, **I want** synthesis tasks to be automatically created when parallel branches converge **so that** parallel outputs are merged coherently.

**Acceptance criteria:**
- When 2+ tasks share a downstream dependent, a SYNTHESIS task is auto-inserted
- Synthesis task receives all upstream outputs in `input_context`
- Synthesis prompt instructs agent to merge, reconcile contradictions, and produce unified output
- `TaskType.SYNTHESIS` tasks use verification criteria: coherence check + completeness check
- If planner explicitly includes a synthesis task, auto-generation is skipped

### US-004: Budget Admission Gate
**As a** platform operator, **I want** missions to stop dispatching when budget is exhausted **so that** users don't get surprise 402 errors.

**Acceptance criteria:**
- Pre-dispatch check: `can_afford(task_estimated_tokens)` must pass before dispatch
- Graduated response per PRD-105:
  - HEALTHY (<50%): dispatch normally
  - WARNING (50-80%): dispatch with reduced max_tokens
  - CRITICAL (80-100%): dispatch only must-complete tasks (synthesis, review)
  - EXCEEDED (>100%): pause mission, notify user, await resume or cancel
- Budget displayed on mission detail page with visual indicator
- Pre-mission estimate shown at plan approval: "Estimated cost: ~X tokens across Y tasks"

### US-005: Task Sizing & Complexity-Aware Decomposition
**As a** planner, **I want** to size tasks based on complexity **so that** no single task exceeds model token limits.

**Acceptance criteria:**
- Planner prompt includes guidance: "No task should require more than 4,000 words of output"
- For content tasks: sections decomposed individually (e.g., "Write Section 3: Prior Art" not "Write the paper")
- Token budget per task estimated by complexity tier:
  - LIGHT (search, lookup): 1,000 tokens
  - MEDIUM (analysis, short draft): 4,000 tokens
  - HEAVY (long-form writing, code generation): 8,000 tokens
  - SYNTHESIS (merge parallel outputs): 6,000 tokens
- Task max_tokens set from complexity tier, not global default
- If estimated output > model max_tokens, planner must split into subtasks

### US-006: Template Parallel Groups
**As a** template author, **I want** templates to define parallel task groups **so that** common mission types exploit concurrency.

**Acceptance criteria:**
- `TaskTemplate` gains `parallel_group: Optional[str]` field
- Tasks in same `parallel_group` share no dependencies and dispatch concurrently
- Updated templates:
  - `content_pipeline`: Research + Source Gathering (parallel) -> Outline -> Section drafts (parallel) -> Synthesis -> Edit -> Review
  - `research_and_report`: Topic research tasks (parallel) -> Analysis -> Synthesis -> Draft -> Review
  - `competitive_analysis`: Per-competitor research (parallel) -> Synthesis -> Report -> Review
  - `data_investigation`: Data gathering tasks (parallel) -> Analysis -> Report
- All templates include at least one SYNTHESIS task after parallel convergence

### US-007: Mission Budget Display
**As a** user, **I want** to see token usage and budget status on the mission detail page **so that** I know how much a mission is costing.

**Acceptance criteria:**
- Budget bar: used / estimated tokens with color coding (green/amber/red/exceeded)
- Per-task token breakdown visible in task detail
- Budget warning banner when WARNING threshold crossed
- Pre-approval: estimated token cost shown alongside plan
- Post-completion: total tokens used, cost estimate in USD (if pricing available)

### US-008: Complexity Detection for Decomposition Strategy
**As a** planner, **I want** to detect mission complexity **so that** simple goals get simple plans and complex goals get properly decomposed.

**Acceptance criteria:**
- Goal analysis classifies into complexity tiers: SIMPLE (3-5 tasks), MODERATE (5-10 tasks), COMPLEX (10-20 tasks)
- Complexity signals: word count of goal, number of deliverables mentioned, domain breadth, attachment count
- SIMPLE missions: max_concurrent=1 (sequential is fine)
- MODERATE missions: max_concurrent=2
- COMPLEX missions: max_concurrent=3
- User can override max_concurrent at plan approval

---

## 3. Architecture

### 3.1 Parallel Dispatch (dispatcher.py)

**Current:** `has_active_task()` → if ANY task active, skip dispatch.

**New:** `count_active_tasks()` → if active_count >= run.max_concurrent, skip dispatch. Otherwise dispatch up to (max_concurrent - active_count) ready tasks per tick.

```python
# BEFORE (sequential gate)
if cls.has_active_task(db, run.id):
    return DispatchResult(dispatched=False, skipped_reason="active_task_exists")

# AFTER (parallel gate)
active_count = cls.count_active_tasks(db, run.id)
dispatch_slots = run.max_concurrent - active_count
if dispatch_slots <= 0:
    return DispatchResult(dispatched=False, skipped_reason="max_concurrent_reached")

ready_tasks = DependencyResolver.get_ready_tasks(db, run.id)
dispatched = []
for task in ready_tasks[:dispatch_slots]:
    result = cls._dispatch_single(db, run, task, agents)
    if result.dispatched:
        dispatched.append(result)

# Budget gate (before each dispatch)
if not budget_manager.can_afford(run.id, task_token_estimate):
    if budget_status == BudgetStatus.EXCEEDED:
        pause_mission(db, run)
        return DispatchResult(dispatched=False, skipped_reason="budget_exceeded")
```

**Key change:** `dispatch_next()` becomes `dispatch_ready()` and can return multiple `DispatchResult`s.

### 3.2 Coordinator Tick Update (coordinator_service.py)

**Current:** `_process_run()` dispatches one task, then reconciles.

**New:** `_process_run()` dispatches up to `max_concurrent` tasks, executes them concurrently via `asyncio.gather()`, then reconciles.

```python
async def _process_run(self, db, run):
    # 1. Budget check
    budget_status = self._check_budget_status(db, run)
    if budget_status == BudgetStatus.EXCEEDED:
        self._pause_for_budget(db, run)
        return

    # 2. Dispatch ready tasks (up to max_concurrent)
    dispatch_results = MissionDispatcher.dispatch_ready(db, run, agents)

    # 3. Execute dispatched tasks concurrently
    if dispatch_results:
        executions = [
            self._execute_task(db, run, r.task, r.agent_id)
            for r in dispatch_results if r.dispatched
        ]
        await asyncio.gather(*executions, return_exceptions=True)

    # 4. Reconcile (verify completed, detect stalls, check completion)
    await MissionReconciler.reconcile(db, run)
```

### 3.3 Planner System Prompt Update (planner.py)

**Current prompt says:** "Tasks execute sequentially (one at a time)."

**New prompt:**

```
You are a mission planner for an AI agent platform. Your job is to decompose
a user's goal into a DAG (directed acyclic graph) of tasks that can be
executed by the available agents.

Rules:
- Each task must be atomic — completable by ONE agent in ONE execution.
- No single task should produce more than 4,000 words of output.
- If a deliverable requires more, split into section-level tasks.
- Identify tasks that can run IN PARALLEL (no shared dependencies).
  Assign the same sequence_number to parallel tasks.
- After parallel branches, include a SYNTHESIS task that merges outputs.
- Every task must specify an agent_role matching one of the available agents.
- Assign a complexity tier to each task: light, medium, heavy, or synthesis.
- The plan MUST contain between 3 and 20 tasks inclusive.
- Return ONLY a single JSON object (no markdown, no explanation).
```

### 3.4 Updated JSON Schema

```json
{
  "tasks": [
    {
      "temp_id": "task_1",
      "title": "Research prior art on shared memory architectures",
      "description": "Detailed instructions...",
      "agent_role": "researcher",
      "sequence_number": 1,
      "task_type": "llm_generation",
      "complexity": "medium",
      "parallel_group": "research",
      "verification_criteria": [
        {"type": "min_length", "value": 500, "must_pass": true}
      ],
      "required_tools": [],
      "dependencies": []
    },
    {
      "temp_id": "task_2",
      "title": "Research competitive analysis of agent frameworks",
      "description": "...",
      "agent_role": "researcher",
      "sequence_number": 1,
      "task_type": "llm_generation",
      "complexity": "medium",
      "parallel_group": "research",
      "dependencies": []
    },
    {
      "temp_id": "task_3",
      "title": "Synthesize research findings into unified brief",
      "description": "Merge outputs from task_1 and task_2...",
      "agent_role": "writer",
      "sequence_number": 2,
      "task_type": "synthesis",
      "complexity": "synthesis",
      "dependencies": ["task_1", "task_2"]
    }
  ]
}
```

**New fields:**
- `complexity`: "light" | "medium" | "heavy" | "synthesis" — drives token budget per task
- `parallel_group`: Optional string — tasks sharing a group have no interdependencies

### 3.5 Budget Admission Gate (dispatcher.py)

Integrates `TokenBudgetManager.can_afford()` into dispatch flow:

```python
# Per-task budget estimation based on complexity
COMPLEXITY_TOKEN_BUDGET = {
    "light":     1_000,
    "medium":    4_000,
    "heavy":     8_000,
    "synthesis": 6_000,
}

def _pre_dispatch_budget_check(self, db, run, task) -> BudgetDecision:
    estimated = COMPLEXITY_TOKEN_BUDGET.get(task.complexity, 4_000)
    budget_status = self.budget_manager.get_status(run.id)

    if budget_status == BudgetStatus.EXCEEDED:
        return BudgetDecision.BLOCK  # Pause mission

    if budget_status == BudgetStatus.CRITICAL:
        # Only allow synthesis and review tasks
        if task.task_type not in (TaskType.SYNTHESIS, TaskType.REVIEW):
            return BudgetDecision.DEFER

    if not self.budget_manager.can_afford(run.id, estimated):
        return BudgetDecision.BLOCK

    return BudgetDecision.ALLOW
```

**Budget lifecycle:**
1. **Plan approval** — show estimated tokens: `sum(COMPLEXITY_TOKEN_BUDGET[t.complexity] for t in tasks)`
2. **Pre-dispatch** — `can_afford()` check with graduated response
3. **Post-execution** — reconcile actual vs estimated, update `run.tokens_used`
4. **Warning at 80%** — emit event, show banner on UI
5. **Hard stop at 100%** — pause mission, user decides: add budget, cancel, or force-continue

### 3.6 Synthesis Task Executor

New logic in `_execute_task()` for SYNTHESIS tasks:

```python
if task.task_type == TaskType.SYNTHESIS:
    # Collect all upstream outputs
    upstream_outputs = self._collect_upstream_outputs(db, task)

    # Build synthesis-specific prompt
    prompt = self._build_synthesis_prompt(task, upstream_outputs)

    # Synthesis verification: coherence + completeness + no contradictions
    task.verification_criteria = [
        {"type": "required_sections", "value": task.expected_sections, "must_pass": True},
        {"type": "min_length", "value": sum(len(o) for o in upstream_outputs) * 0.5, "must_pass": False},
    ]
```

Synthesis prompt template:
```
You are merging the outputs of {n} parallel tasks into a single coherent document.

## Upstream Outputs
{for each upstream task: title + output}

## Instructions
{task.description}

## Rules
- Include ALL substantive content from upstream outputs. Do not drop findings.
- Resolve any contradictions between sources — note where sources disagree.
- Maintain consistent voice, terminology, and formatting throughout.
- Produce a unified document, not a concatenation of sections.
```

### 3.7 Template Updates (templates.py)

Add `parallel_group` and `complexity` to `TaskTemplate`:

```python
@dataclass(frozen=True)
class TaskTemplate:
    sequence: int
    agent_role: str
    title_pattern: str
    description_pattern: str
    task_type: str = TaskType.LLM_GENERATION.value
    complexity: str = "medium"
    parallel_group: Optional[str] = None
    verification_criteria: List[Dict] = field(default_factory=list)
    depends_on: Optional[List[str]] = None  # explicit deps, replaces auto-chaining
```

**Revised `content_pipeline` template:**

```
Sequence 1 (parallel_group="research"):
  - "Research background and sources for: {goal}" (complexity: medium)
  - "Gather data, statistics, and references for: {goal}" (complexity: medium)

Sequence 2:
  - "Synthesize research into structured outline for: {goal}" (type: synthesis, complexity: synthesis)
    depends_on: [task_1, task_2]

Sequence 3 (parallel_group="drafting"):
  - "Draft introduction and background sections" (complexity: heavy)
  - "Draft methodology/approach sections" (complexity: heavy)
  - "Draft analysis/results sections" (complexity: heavy)
    all depend_on: [task_3]

Sequence 4:
  - "Synthesize section drafts into complete document" (type: synthesis, complexity: synthesis)
    depends_on: [task_4, task_5, task_6]

Sequence 5:
  - "Edit and review final document" (type: review, complexity: medium)
    depends_on: [task_7]
```

This transforms a "write a research paper" mission from:
- **Before:** 1 agent, 1 task, truncated output, failed verification
- **After:** 2 parallel researchers -> outline synthesis -> 3 parallel drafters -> document synthesis -> review

### 3.8 Complexity Detection (planner.py)

New function called before decomposition:

```python
def _detect_complexity(goal: str, attachments: List) -> ComplexityTier:
    signals = {
        "word_count": len(goal.split()),
        "deliverable_count": _count_deliverables(goal),  # keywords: report, app, analysis, etc.
        "domain_breadth": _estimate_domains(goal),        # distinct topic areas mentioned
        "attachment_count": len(attachments),
    }

    score = (
        (1 if signals["word_count"] > 100 else 0) +
        (1 if signals["deliverable_count"] > 1 else 0) +
        (1 if signals["domain_breadth"] > 2 else 0) +
        (1 if signals["attachment_count"] > 0 else 0)
    )

    if score >= 3:
        return ComplexityTier.COMPLEX   # max_concurrent=3, 10-20 tasks
    elif score >= 1:
        return ComplexityTier.MODERATE  # max_concurrent=2, 5-10 tasks
    else:
        return ComplexityTier.SIMPLE    # max_concurrent=1, 3-5 tasks
```

`max_concurrent` set during planning, overridable at plan approval.

---

## 4. 82A/B Gap Closure

These items were scaffolded in 82A/B but never wired. 82C closes them:

### 4.1 Wire `max_concurrent` (82A gap)
- **Current:** Field on OrchestrationRun, server_default=1, never read
- **Fix:** Dispatcher reads `run.max_concurrent` in dispatch gate
- **Set by:** Planner sets based on complexity detection; user can override at approval

### 4.2 Wire `TaskType.SYNTHESIS` (82B gap)
- **Current:** Enum value exists, never generated or handled
- **Fix:** Planner generates SYNTHESIS tasks; coordinator has synthesis-specific prompt builder; templates include synthesis tasks after parallel convergence

### 4.3 Wire `TokenBudgetManager.can_afford()` (82B gap)
- **Current:** Method exists in token_budget_manager.py, never called
- **Fix:** Dispatcher calls `can_afford()` before every dispatch; graduated response (allow/defer/block)

### 4.4 Wire complexity-aware budget estimation (82B gap)
- **Current:** `TOKENS_PER_TASK_ESTIMATE = 2000` flat for all tasks
- **Fix:** Per-task estimate from `complexity` field: light=1000, medium=4000, heavy=8000, synthesis=6000

### 4.5 Update templates for parallel groups (82B gap)
- **Current:** All 4 templates chain `depends_on = [previous_task]`
- **Fix:** Templates use `parallel_group` and explicit `depends_on` for DAG structure

### 4.6 Dispatcher picks all ready tasks (82B gap)
- **Current:** `DependencyResolver.get_ready_tasks()` returns multiple, dispatcher takes `[0]` only
- **Fix:** Dispatcher iterates ready tasks up to available dispatch slots

---

## 5. What's NOT in 82C (deferred to 82D)

- **Ephemeral/contractor agents** (PRD-104) — significant new subsystem, decouple from parallel dispatch
- **Cross-mission knowledge transfer** — requires persistent knowledge graph design
- **Outcome telemetry dashboards** (PRD-106) — metadata columns can be added but dashboards are 82D
- **Model routing optimization** — static role->model mapping is sufficient for 82C
- **Prompt coaching / guidance engine** — learning layer, not execution layer
- **Tool policy layering** (PRD-105 Section 4) — workspace > mission > task > agent narrowing

---

## 6. Implementation Plan

### Phase 1: Parallel Dispatch (Core)
**Files:** `dispatcher.py`, `coordinator_service.py`, `orchestration_enums.py`

1. Replace `has_active_task()` with `count_active_tasks()` in dispatcher
2. `dispatch_next()` → `dispatch_ready()` returning `List[DispatchResult]`
3. Coordinator `_process_run()` executes dispatched tasks via `asyncio.gather()`
4. Wire `run.max_concurrent` into dispatch gate (read from DB, default 3)
5. Add RUNNING → RUNNING transition guard (multiple tasks running is valid)

**Test:** Create mission with 2 independent tasks, verify both dispatch on same tick.

### Phase 2: Intelligent Decomposition
**Files:** `planner.py`, `templates.py`

1. Update `_SYSTEM_PROMPT` — remove "sequential" language, add parallel group guidance
2. Add `complexity` and `parallel_group` to output schema
3. Add `_detect_complexity()` function — set `max_concurrent` during planning
4. Update `_validate_plan()` — verify parallel_group tasks have no cross-dependencies
5. Update all 4 templates with parallel groups and synthesis tasks
6. Add `render_template()` support for `parallel_group` and explicit `depends_on`
7. Synthesis task auto-insertion: if parallel group converges without explicit synthesis, inject one

**Test:** Submit "write a research paper" goal, verify plan has parallel research + synthesis.

### Phase 3: Budget Governance
**Files:** `dispatcher.py`, `coordinator_service.py`, `token_budget_manager.py`

1. Add `_pre_dispatch_budget_check()` to dispatcher
2. Wire `TokenBudgetManager.can_afford()` into dispatch flow
3. Add graduated response: HEALTHY/WARNING/CRITICAL/EXCEEDED
4. Pause mission on EXCEEDED — emit event, set run state to PAUSED
5. Add complexity-aware token estimates replacing flat 2000/task
6. Pre-approval budget display: estimated tokens shown in plan response

**Test:** Create mission with low budget, verify it pauses at threshold instead of 402.

### Phase 4: Synthesis Executor
**Files:** `coordinator_service.py`, `dispatcher.py`

1. Add `_build_synthesis_prompt()` — merges upstream outputs with reconciliation instructions
2. Synthesis-specific verification criteria: coherence + completeness
3. `_execute_task()` detects `TaskType.SYNTHESIS` and uses synthesis prompt builder
4. Auto-synthesis injection in planner: detect parallel convergence without explicit synthesis

**Test:** 2 parallel research tasks → synthesis task merges both outputs coherently.

### Phase 5: Frontend Updates
**Files:** `mission-detail-page.tsx`, `use-missions-api.ts`, `mission-dag-canvas.tsx`

1. Budget bar component: used/estimated tokens, color-coded
2. Budget warning banner when WARNING threshold crossed
3. Pre-approval: show estimated cost alongside plan
4. DAG canvas: parallel tasks rendered side-by-side (not just linear chain)
5. `max_concurrent` override control at plan approval
6. Per-task token usage in task detail panel

### Phase 6: Template Expansion
**Files:** `templates.py`

1. Rewrite all 4 templates with parallel groups
2. Add 2 new templates:
   - `coding_task`: Spec -> Implement + Tests (parallel) -> Review -> Deploy
   - `multi_document`: Per-document analysis (parallel) -> Synthesis -> Report
3. Template selection uses complexity tier to choose task count range

---

## 7. Validation Criteria

### 7.1 The Research Paper Test (must pass)

Submit the exact mission from `log.md`: "Write a technical research paper titled 'Shared Semantic Fields for Multi-Agent Coordination'..."

**Expected behavior:**
1. Planner detects COMPLEX (long goal, multiple sections, multiple deliverables)
2. Plan contains 8-12 tasks with parallel groups:
   - Group "research": 2-3 parallel research tasks (prior art, experiment data, competitive landscape)
   - Synthesis: merge research into brief
   - Group "drafting": 3-4 parallel section drafts (each < 4000 words)
   - Synthesis: merge sections into complete paper
   - Review: edit pass
3. max_concurrent = 3 (auto-detected from COMPLEX tier)
4. Budget estimate shown at approval (~50,000 tokens)
5. Research tasks dispatch simultaneously on first tick
6. No truncation — each task produces < 4000 words
7. Synthesis tasks merge parallel outputs
8. Final output: complete 3,000-4,000 word paper
9. Budget tracked throughout, no 402 surprises

### 7.2 Regression Tests

- Simple goal ("summarize this document") → 3-5 sequential tasks, max_concurrent=1
- Template match ("write a blog post about X") → content_pipeline template with parallel groups
- Budget exceeded → mission pauses, user notified, can resume or cancel
- Agent failure mid-parallel → failed task retries, siblings continue unblocked
- Replan after failure → generates replacement subtree only, preserves completed work

### 7.3 Performance Targets

| Metric | Target |
|--------|--------|
| Parallel mission speedup vs sequential | >= 2x for missions with 2+ parallel groups |
| Budget estimation accuracy | Within 50% of actual (improves with telemetry in 82D) |
| No single task > 4000 words output | 100% for content missions |
| Plan generation time | < 15s including complexity detection |
| Dispatch latency per tick | < 2s for up to 3 concurrent dispatches |

---

## 8. Data Model Changes

### 8.1 OrchestrationTask additions

```sql
ALTER TABLE orchestration_tasks
    ADD COLUMN complexity VARCHAR(10) DEFAULT 'medium',
    ADD COLUMN parallel_group VARCHAR(50) DEFAULT NULL,
    ADD COLUMN estimated_tokens INTEGER DEFAULT 4000;
```

### 8.2 OrchestrationRun additions

```sql
-- No new columns needed. max_concurrent already exists (default 1 → change default to 3).
-- token_budget_estimate already exists.
-- tokens_used already exists.

ALTER TABLE orchestration_runs
    ALTER COLUMN max_concurrent SET DEFAULT 3;
```

### 8.3 New enum values

```python
class BudgetStatus(str, Enum):
    HEALTHY = "healthy"      # <50%
    WARNING = "warning"      # 50-80%
    CRITICAL = "critical"    # 80-100%
    EXCEEDED = "exceeded"    # >100%

class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
```

---

## 9. API Changes

### 9.1 Plan approval response (existing endpoint, enriched response)

```json
{
  "id": "...",
  "state": "awaiting_approval",
  "plan": {
    "tasks": [...],
    "estimated_tokens": 48000,
    "estimated_cost_usd": null,
    "max_concurrent": 3,
    "complexity_tier": "complex",
    "parallel_groups": ["research", "drafting"],
    "has_synthesis_tasks": true
  }
}
```

### 9.2 Mission detail response (existing endpoint, enriched)

```json
{
  "id": "...",
  "budget": {
    "estimated_tokens": 48000,
    "used_tokens": 23500,
    "status": "warning",
    "percentage": 49
  },
  "tasks": [
    {
      "id": "...",
      "parallel_group": "research",
      "complexity": "medium",
      "estimated_tokens": 4000,
      "tokens_used": 3200
    }
  ]
}
```

### 9.3 Approve with overrides (existing endpoint, new body fields)

```json
POST /api/missions/{id}/approve
{
  "max_concurrent": 2,
  "token_budget_override": 60000
}
```

---

## 10. Risk & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Parallel tasks write conflicting outputs to shared field | Medium | Medium | Field dedup by content_hash already exists; synthesis prompt handles contradictions |
| LLM planner ignores parallel_group instructions | Medium | Low | Validation catches; fallback to sequential if no parallel groups |
| Budget estimation wildly inaccurate | High | Medium | Conservative defaults (2x actual); user override; soft-then-hard enforcement |
| Agent contention (same agent assigned to 2 parallel tasks) | Medium | Low | Agent matcher checks availability (busy=0.5 score); prefer different agents for parallel tasks |
| Synthesis quality poor (just concatenates) | Medium | High | Verification checks coherence; retry with feedback; cross-model judge |
| Parallel execution increases DB contention | Low | Medium | Optimistic locking already handles; version_id prevents double-state |

---

## 11. Migration Path

**This is NOT a breaking change.** Existing missions continue working:

1. Default `max_concurrent` changes from 1 → 3, but existing running missions keep their stored value
2. Old-format plans (no parallel_group, no complexity) are treated as sequential with "medium" complexity
3. Templates with no parallel_group fall back to sequential chaining
4. Budget gate defaults to HEALTHY if no estimate exists (no blocking)
5. Synthesis auto-insertion only triggers when parallel_group detected

**Rollback:** Set `max_concurrent=1` on all runs to revert to sequential behavior. Budget gate can be disabled via config flag `BUDGET_HARD_ENFORCEMENT_ENABLED=false`.

---

## 12. Verification & Review Gates

### 12.1 The 82A/B Problem

In 82A/B, Ralph built scaffolding (models, enums, classes) that passed code review because the code existed syntactically. But the code was never called. `max_concurrent` was a column nobody read. `TaskType.SYNTHESIS` was an enum nobody generated. `can_afford()` was a method nobody invoked.

**Root cause:** Review checked "does the code exist?" not "is the code reachable from the execution path?"

### 12.2 Wiring Verification Tests (MANDATORY per phase)

Every phase must include **wiring tests** — integration tests that prove the new code is called during actual mission execution. These are not unit tests of isolated functions. They trace the full call path.

#### Phase 1: Parallel Dispatch — Wiring Tests

```python
# test_parallel_dispatch_wiring.py

def test_dispatch_reads_max_concurrent():
    """Prove dispatcher uses run.max_concurrent, not hardcoded 1."""
    run = create_test_run(max_concurrent=2)
    task_a = create_test_task(run, sequence=1, dependencies=[])
    task_b = create_test_task(run, sequence=1, dependencies=[])

    results = MissionDispatcher.dispatch_ready(db, run, agents)
    assert len([r for r in results if r.dispatched]) == 2  # BOTH dispatched

def test_sequential_still_works():
    """Regression: max_concurrent=1 dispatches one at a time."""
    run = create_test_run(max_concurrent=1)
    task_a = create_test_task(run, sequence=1, dependencies=[])
    task_b = create_test_task(run, sequence=1, dependencies=[])

    results = MissionDispatcher.dispatch_ready(db, run, agents)
    assert len([r for r in results if r.dispatched]) == 1

def test_dependency_blocks_dispatch():
    """Tasks with unmet deps stay PENDING even with free dispatch slots."""
    run = create_test_run(max_concurrent=3)
    task_a = create_test_task(run, sequence=1, dependencies=[])
    task_b = create_test_task(run, sequence=2, dependencies=[task_a])

    results = MissionDispatcher.dispatch_ready(db, run, agents)
    dispatched_ids = [r.task.id for r in results if r.dispatched]
    assert task_a.id in dispatched_ids
    assert task_b.id not in dispatched_ids  # blocked by dependency

def test_coordinator_executes_parallel():
    """Prove coordinator runs dispatched tasks via asyncio.gather, not sequentially."""
    # Instrument _execute_task with timing
    # Two tasks dispatched → both start within 1s of each other (not 60s+ apart)
```

#### Phase 2: Intelligent Decomposition — Wiring Tests

```python
# test_decomposition_wiring.py

def test_planner_generates_parallel_groups():
    """Prove planner output includes parallel_group on at least some tasks."""
    result = planner.decompose("Write a research paper on AI coordination", ws_id, agents)
    groups = [t.parallel_group for t in result.tasks if t.parallel_group]
    assert len(groups) >= 2  # At least one parallel group with 2+ tasks

def test_planner_generates_synthesis_tasks():
    """Prove planner creates SYNTHESIS tasks after parallel convergence."""
    result = planner.decompose("Write a research paper on AI coordination", ws_id, agents)
    synthesis_tasks = [t for t in result.tasks if t.task_type == TaskType.SYNTHESIS.value]
    assert len(synthesis_tasks) >= 1

def test_no_task_exceeds_heavy_complexity():
    """Prove planner sizes tasks — no single task should be unbounded."""
    result = planner.decompose("Write a 10,000 word technical report", ws_id, agents)
    for task in result.tasks:
        assert task.complexity in ("light", "medium", "heavy", "synthesis")

def test_complexity_detection_sets_max_concurrent():
    """Prove complexity detection influences max_concurrent on the run."""
    # Complex goal → max_concurrent >= 2
    result = planner.decompose(COMPLEX_GOAL, ws_id, agents)
    assert result.max_concurrent >= 2

    # Simple goal → max_concurrent == 1
    result = planner.decompose("Summarize this document", ws_id, agents)
    assert result.max_concurrent == 1

def test_parallel_group_tasks_have_no_cross_deps():
    """Prove validation rejects plans where parallel-grouped tasks depend on each other."""
    # Inject a bad plan with cross-dependency within parallel group
    bad_plan = {...}  # task_1 and task_2 in group "research", task_2 depends on task_1
    errors = planner._validate_plan(bad_plan, agents)
    assert any("parallel_group" in e for e in errors)
```

#### Phase 3: Budget Governance — Wiring Tests

```python
# test_budget_wiring.py

def test_can_afford_called_before_dispatch():
    """Prove dispatcher calls can_afford() — not just that the method exists."""
    run = create_test_run(token_budget_estimate=1000, tokens_used=950)
    task = create_test_task(run, complexity="heavy")  # 8000 tokens estimated

    results = MissionDispatcher.dispatch_ready(db, run, agents)
    assert not any(r.dispatched for r in results)  # BLOCKED by budget

def test_budget_exceeded_pauses_mission():
    """Prove mission transitions to PAUSED when budget exceeded."""
    run = create_test_run(token_budget_estimate=5000, tokens_used=5500)

    coordinator._process_run(db, run)

    db.refresh(run)
    assert run.state == RunState.PAUSED.value

def test_critical_budget_allows_synthesis_only():
    """At 80-100%, only synthesis and review tasks dispatch."""
    run = create_test_run(token_budget_estimate=10000, tokens_used=8500)
    synthesis_task = create_test_task(run, task_type=TaskType.SYNTHESIS)
    heavy_task = create_test_task(run, task_type=TaskType.LLM_GENERATION, complexity="heavy")

    results = MissionDispatcher.dispatch_ready(db, run, agents)
    dispatched_types = [r.task.task_type for r in results if r.dispatched]
    assert TaskType.SYNTHESIS.value in dispatched_types
    assert TaskType.LLM_GENERATION.value not in dispatched_types

def test_budget_estimate_uses_complexity_not_flat():
    """Prove budget estimation is complexity-aware, not 2000*task_count."""
    result = planner.decompose(COMPLEX_GOAL, ws_id, agents)
    # Should NOT be len(tasks) * 2000
    flat_estimate = len(result.tasks) * 2000
    assert result.token_estimate != flat_estimate
```

#### Phase 4: Synthesis Executor — Wiring Tests

```python
# test_synthesis_wiring.py

def test_synthesis_task_receives_all_upstream_outputs():
    """Prove synthesis prompt includes ALL upstream outputs, not just last."""
    # Create 3 completed parallel tasks → 1 synthesis task
    upstream_outputs = ["Research finding A", "Research finding B", "Research finding C"]
    prompt = coordinator._build_synthesis_prompt(synthesis_task, upstream_outputs)
    for output in upstream_outputs:
        assert output in prompt  # ALL present, none dropped

def test_synthesis_auto_inserted_when_missing():
    """If planner creates parallel group without synthesis, auto-insert one."""
    plan_without_synthesis = {
        "tasks": [
            {"temp_id": "t1", "parallel_group": "research", "dependencies": []},
            {"temp_id": "t2", "parallel_group": "research", "dependencies": []},
            {"temp_id": "t3", "dependencies": ["t1", "t2"]},  # NOT marked synthesis
        ]
    }
    fixed_plan = planner._ensure_synthesis_tasks(plan_without_synthesis)
    synthesis_tasks = [t for t in fixed_plan["tasks"] if t["task_type"] == "synthesis"]
    assert len(synthesis_tasks) >= 1
```

### 12.3 Review Checklist (per phase, before merge)

Every phase PR must include this checklist. Reviewer must verify each item:

```markdown
## 82C Wiring Review Checklist

### Call-path verification (reviewer must trace, not just grep)
- [ ] New function/method is called from the coordinator tick loop (trace: tick → _process_run → ... → new code)
- [ ] No dead imports (every import is used in an active code path)
- [ ] No dead parameters (every new parameter is read and influences behavior)
- [ ] No default-only fields (new DB columns are written AND read somewhere)

### Integration test verification
- [ ] Wiring tests pass (not just unit tests of isolated functions)
- [ ] At least one test proves "old behavior would have done X, new behavior does Y"
- [ ] At least one test proves the feature works end-to-end through coordinator tick

### Regression verification
- [ ] Sequential missions (max_concurrent=1) still work
- [ ] Existing templates still match and render
- [ ] Missions without budget estimates don't crash
- [ ] Single-task missions complete normally

### Code quality
- [ ] No scaffolding without wiring (if you define it, call it)
- [ ] No TODO/FIXME for critical path items (those are bugs, not todos)
- [ ] Config values in config.py, not hardcoded
```

### 12.4 Phase Gate Reviews

Each phase has a gate before the next phase starts:

| Gate | Criteria | Reviewer |
|------|----------|----------|
| Phase 1 → 2 | Parallel dispatch wiring tests green. Manual test: 2 tasks dispatch on same tick. | Human (Gerard) |
| Phase 2 → 3 | "Write a paper" goal decomposes into parallel groups + synthesis. Plan has max_concurrent > 1. | Human (Gerard) |
| Phase 3 → 4 | Low-budget mission pauses at threshold. `can_afford()` provably called (test + log). | Human (Gerard) |
| Phase 4 → 5 | Synthesis task merges 2+ upstream outputs. Verification passes on merged output. | Human (Gerard) |
| Phase 5 → 6 | Budget bar renders. DAG shows parallel branches. Override controls work. | Human (Gerard) |
| Phase 6 → UAT | All templates render parallel groups. New templates match expected goals. | Human (Gerard) |

### 12.5 User Acceptance Tests (after all phases)

These are the final "does it actually work" tests run by Gerard:

**Test 1: Research Paper Mission**
- Input: The exact PRD-108 paper prompt from `log.md`
- Expected: Parallel research → synthesis → parallel drafting → synthesis → review → complete paper
- Pass criteria: Paper is 3,000-4,000 words, no truncation, all sections present, budget tracked

**Test 2: Simple Mission**
- Input: "Summarize this PDF" with attachment
- Expected: 3-4 sequential tasks, max_concurrent=1, completes quickly
- Pass criteria: Regression — simple missions don't over-decompose

**Test 3: Budget Limit Mission**
- Input: Complex goal with intentionally low budget override
- Expected: Mission pauses at budget threshold, user can resume or cancel
- Pass criteria: No 402 errors, clear budget UI, graceful pause

**Test 4: App Building Mission** (stretch)
- Input: "Build a simple todo app with React frontend and Express backend"
- Expected: Parallel spec + research → architecture → parallel implementation → synthesis → review
- Pass criteria: Outputs contain workable code, synthesis merges frontend + backend coherently

---

## 13. References

- PRD-82A: Sequential Mission Coordinator (built) — `docs/PRDS/82A-SEQUENTIAL-MISSION-COORDINATOR.md`
- PRD-82 Research: Orchestration Readiness — `docs/PRDS/82-RESEARCH-ORCHESTRATION-READINESS.md`
- PRD-102: Coordinator Architecture — `docs/PRDS/102-COORDINATOR-ARCHITECTURE.md`
- PRD-104: Ephemeral Agents & Model Selection — `docs/PRDS/104-EPHEMERAL-AGENTS-MODEL-SELECTION.md`
- PRD-105: Budget & Governance — `docs/PRDS/105-BUDGET-GOVERNANCE.md`
- PRD-106: Outcome Telemetry — `docs/PRDS/106-OUTCOME-TELEMETRY.md`
- PRD-108: Memory Field Prototype — `docs/PRDS/108-MEMORY-FIELD-PROTOTYPE.md`
