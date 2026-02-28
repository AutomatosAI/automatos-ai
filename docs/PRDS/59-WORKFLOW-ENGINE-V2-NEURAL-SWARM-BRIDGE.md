# PRD-59: Workflow Engine V2 — From 9-Stage Pipeline to Neural Swarm Architecture

**Version:** 1.0
**Status:** Draft
**Date:** February 18, 2026
**Author:** Claude Code (with Gavin Kavanagh)
**Prerequisites:** PRD-10 (Workflow Orchestration), PRD-16 (LLM-Driven Orchestrator), PRD-50 (Universal Router), PRD-51 (Orchestrator Unification), PRD-56 (Infrastructure Scaling), PRD-58 (Prompt Management)

---

## Executive Summary

The Automatos 9-stage workflow engine runs end-to-end but produces inconsistent results because **Stages 1-2 (Task Decomposition + Agent Selection) are only as good as their prompts, and those prompts have never been evaluated or optimized**. When decomposition is wrong, everything downstream fails — the best execution engine in the world can't fix a badly sliced task.

This PRD does three things:

1. **Stabilize the current engine** — Fix the 6 critical issues that make the 9-stage pipeline unreliable today (Stage 7 scores metadata not outputs, learning loop doesn't close, context optimization disabled, etc.)

2. **Evolve to dynamic stages** — Replace the fixed 9-stage sequence with a stage selector that skips unnecessary stages for simple tasks and adds inter-agent negotiation stages for complex ones. Not every task needs all 9 stages.

3. **Bridge to distributed execution** — Introduce the `TaskRunner` abstraction (from PRD-56) so each agent subtask can run as an independent worker today (asyncio), a queued job tomorrow (Redis/ARQ), and a K8s pod next quarter — without changing any orchestration logic. This is the path to microagent swarms.

### Why Now

The platform has evolved massively since the original 9-stage design (PRD-10, November 2025):

| Then (PRD-10) | Now (February 2026) |
|---|---|
| ~20 LLM models | 350+ models via OpenRouter + 8 providers |
| Basic agent skills | 40+ skill categories, personas, marketplace |
| No tool integration | 400+ MCP tools, Composio, tool catalog |
| Simple memory | 4-tier hierarchical + Mem0 persistent memory |
| No routing | Universal Router with 4-tier classification (PRD-50) |
| Hardcoded prompts | Prompt Registry with FutureAGI evaluation (PRD-58) |
| Single process | TaskRunner abstraction ready for K8s (PRD-56) |

The engine's *infrastructure* has grown 10x. The engine's *orchestration logic* hasn't kept up.

### Relationship to Existing PRDs

| PRD | Relationship |
|---|---|
| PRD-10 (Workflow Engine) | Original 8-stage design. PRD-59 is its successor. |
| PRD-16 (LLM-Driven Orchestrator) | Proposed LLM-first stages. PRD-59 implements the Master Orchestrator pattern selectively. |
| PRD-50 (Universal Router) | Router sits *above* the engine. PRD-59 focuses on what happens *after* routing decides to orchestrate. |
| PRD-51 (Orchestrator Unification) | Unifies tool loading + execution paths. PRD-59 depends on this being clean. |
| PRD-56 (Infrastructure Scaling) | `TaskRunner` abstraction. PRD-59 integrates it as the execution layer. |
| PRD-58 (Prompt Management) | Prompt Registry + FutureAGI. PRD-59 depends on optimized Stage 1-2 prompts. |
| PRD-04 (Inter-Agent Communication) | SharedContext + Redis pub/sub. PRD-59 evolves this into field-based coordination. |

---

## Part 1: Current State Audit

### What Works

| Component | Status | Evidence |
|---|---|---|
| Stage 1: Task Decomposition | **Working** | `RealTaskDecomposer` makes real LLM calls, returns JSON with subtasks, validates dependency graph via `GraphTheory` |
| Stage 2: Agent Selection | **Working** | `LLMAgentSelector` does batch LLM selection for all subtasks in one call |
| Stage 3: Context Engineering | **Partial** | RAG retrieval works when documents exist; CodeGraph works when indexed; **mathematical optimization (knapsack/MMR) is DISABLED** |
| Stage 4: Agent Execution | **Working** | `AgentExecutionManager` runs parallel groups via asyncio, tools work, `SharedContextManager` passes results between agents |
| Stage 5: Result Aggregation | **Working** | 5-dimension heuristic scoring (completeness, accuracy, efficiency, reliability, coherence) |
| Stage 6: Learning Update | **Partial** | Updates `agent.performance_metrics` in DB, but `LLMAgentSelector` (the live path) may not read these metrics back |
| Stage 7: Quality Assessment | **Broken** | Always uses heuristics (`use_llm=False`), scores a metadata summary string — not the actual agent outputs |
| Stage 8: Memory Storage | **Working** | Mem0 storage works; hierarchical consolidation stubs (collective memory returns `[]`) |
| Stage 9: Response Generation | **Working** | Builds structured output_data, stores analytics |

### The 6 Critical Issues

#### Issue 1: Stage 7 scores metadata, not outputs
**Location:** `api/workflows.py:2234-2248`
```python
quality_assessor = OutputQualityAssessor(
    llm_client=None,  # Will use heuristic assessment
    use_llm=False
)
```
The heuristic assesses a *summary string* containing subtask count, token count, and status. It does NOT evaluate the actual LLM responses. Stage 7 scores are meaningless — they'll hover around 0.65-0.75 regardless of output quality.

**Impact:** Quality gate doesn't work. Bad outputs pass. Good outputs don't score higher.

#### Issue 2: Learning loop doesn't fully close
**Location:** `modules/learning/engine/core.py` → `api/workflows.py:1663`

Stage 6's `LearningSystemUpdater` writes updated `performance_metrics` to the Agent DB record (exponential moving average, learning rate 0.1). But the **live execution path** uses `LLMAgentSelector` (a batch LLM call), which constructs a prompt about available agents — it's unclear whether it includes each agent's `performance_metrics.success_rate` in that prompt. If not, the learning loop writes data that nothing reads.

**Impact:** The system doesn't get smarter over time. Agent selection doesn't improve from experience.

#### Issue 3: Context optimization disabled
**Location:** `modules/orchestrator/stages/context_engineering.py:170`
```
"ℹ️ Mathematical optimization DISABLED - using basic RAG only"
```
The Shannon Entropy filtering, MMR diversity selection, and Knapsack token budget optimization — the mathematical foundations described in the Platform Guide — are all disabled. Context engineering falls back to basic RAG retrieval.

**Impact:** Context quality is unoptimized. Token budgets are not managed. The mathematical differentiation described in the ebook doesn't actually run.

#### Issue 4: `requires_context` field ignored
**Location:** `api/workflows.py:1898`
```python
subtasks_needing_context = steps  # FIX: Force all subtasks to get context engineering
```
Stage 1's decomposer returns a `requires_context` field per subtask, but the execution path ignores it and forces all subtasks through context engineering. This wastes tokens on subtasks that don't need context and adds latency.

**Impact:** Unnecessary RAG calls. Wasted tokens. Slower execution.

#### Issue 5: Two disconnected orchestrator implementations
**Location:** `api/workflows.py` vs `modules/orchestrator/service.py`

`execute_workflow_with_progress()` (2700+ lines inline in workflows.py) is the live path. `EnhancedOrchestratorService.execute_workflow()` (in service.py) is a cleaner implementation with 4-dimensional agent scoring, LLM quality assessment, and `WorkflowMemoryIntegrator` — but it's **disconnected** (import commented out in `api/orchestrator.py:23,30`).

**Impact:** The better implementation isn't used. Bug fixes happen in the wrong place.

#### Issue 6: No graph metadata propagation
**Location:** `modules/agents/execution/execution_manager.py:351-357`

The execution manager looks for `subtask['graph_metadata']` but the decomposer puts graph analysis in `result["graph_analysis"]` (top-level), not per-subtask. Graph dependency information computed in Stage 1 never reaches Stage 4.

**Impact:** Parallel execution grouping works (via `parallel_groups`), but individual subtasks don't know their position in the dependency graph.

---

## Part 2: Should We Keep 9 Stages?

### Analysis

The 9 stages emerged organically — PRD-10 had 8, PRD-16 grew it to 9. The number isn't grounded in theory. Looking at the **actual execution flow**, what matters is:

```
DECIDE what to do     →  Stages 1-2 (Decompose + Select)
PREPARE context       →  Stage 3 (Context Engineering)
DO the work           →  Stage 4 (Agent Execution)
EVALUATE the result   →  Stages 5, 7 (Aggregate + Quality)
LEARN from it         →  Stages 6, 8 (Learn + Store)
RESPOND               →  Stage 9 (Response)
```

That's **5 phases**, not 9 stages. Some tasks need all 5. A simple single-agent chat response needs only phases 3-4 (prepare + do). A recipe with pre-assigned agents skips phases 1-2 entirely (already decided).

### Recommendation: Dynamic Phase Selection

Replace the fixed 9-stage sequence with **5 phases** that expand into the specific stages needed:

```
Phase       Stages (expanded as needed)              Skip When
──────      ────────────────────────────              ─────────
PLAN        1. Task Decomposition                    RECIPE mode (steps pre-defined)
            2. Agent Selection                       RECIPE mode (agents pre-assigned)
            2b. Inter-Agent Negotiation (NEW)        Single-agent task

PREPARE     3. Context Engineering                   Agent doesn't need context
            3b. Prompt Optimization (NEW, PRD-58)    Prompt already evaluated

EXECUTE     4. Agent Execution                       Never skipped
            4b. Inter-Agent Coordination (PRD-04)    Single-agent task

EVALUATE    5. Result Aggregation                    Single subtask
            6. Quality Assessment (fixed)            Below quality threshold

LEARN       7. Learning Update                       Execution failed
            8. Memory Storage                        Transient/test execution
            9. Response Generation                   Never skipped
```

### What's New

**Stage 2b: Inter-Agent Negotiation** — Before execution, selected agents review the task plan and can propose adjustments. From PRD-04's collaborative problem-solving algorithm. Only triggers for 3+ agent workflows.

**Stage 3b: Prompt Optimization** — If PRD-58's Prompt Registry has evaluated the relevant system prompt and FutureAGI has an optimized variant, use it. Check `PromptRegistry.get_ab_variant()` for active A/B tests.

**Stage 4b: Inter-Agent Coordination** — During parallel execution, agents write findings to `SharedContextManager`. This is already partially implemented but formalized here as a distinct coordination step between parallel groups.

### The Phase Selector

```python
class PhaseSelector:
    """
    Determines which phases and stages to execute based on:
    - Execution mode (AUTONOMOUS / RECIPE / HYBRID)
    - Task complexity (from ComplexityAnalyzer)
    - Number of agents involved
    - Available context sources
    """

    def select_phases(
        self,
        mode: ExecutionMode,
        complexity: ComplexityLevel,
        agent_count: int,
        has_context_sources: bool,
    ) -> list[Phase]:
        phases = []

        # PLAN phase
        if mode == ExecutionMode.AUTONOMOUS:
            phases.append(Phase.PLAN_FULL)      # Stages 1, 2, 2b
        elif mode == ExecutionMode.HYBRID:
            phases.append(Phase.PLAN_PARTIAL)   # Stage 2 only
        # RECIPE mode: skip PLAN entirely

        # PREPARE phase
        if has_context_sources or complexity >= ComplexityLevel.MOLECULE:
            phases.append(Phase.PREPARE)         # Stages 3, 3b

        # EXECUTE phase (always)
        if agent_count > 1:
            phases.append(Phase.EXECUTE_MULTI)   # Stages 4, 4b
        else:
            phases.append(Phase.EXECUTE_SINGLE)  # Stage 4 only

        # EVALUATE phase
        if agent_count > 1 or complexity >= ComplexityLevel.CELL:
            phases.append(Phase.EVALUATE_FULL)   # Stages 5, 6
        else:
            phases.append(Phase.EVALUATE_LIGHT)  # Stage 5 only (no LLM quality)

        # LEARN phase (always, but scope varies)
        phases.append(Phase.LEARN)               # Stages 7, 8, 9

        return phases
```

This maps directly to the **Progressive Complexity Model** from the Platform Guide:

| Complexity Level | Phases Used | Estimated Stages | Token Budget |
|---|---|---|---|
| Atom (simple task) | EXECUTE + LEARN | 3 stages (4, 9, partial 8) | 50-200 |
| Molecule (needs examples) | PREPARE + EXECUTE + LEARN | 5 stages (3, 4, 5, 8, 9) | 500-2,000 |
| Cell (agent memory) | PREPARE + EXECUTE + EVALUATE + LEARN | 7 stages (3, 4, 5, 6, 7, 8, 9) | 2,000-4,000 |
| Organ (multi-agent) | All 5 phases | All stages including 2b, 4b | 4,000-8,000 |
| Organism (enterprise) | All 5 phases + meta-learning | All stages + cross-workflow learning | 8,000-16,000 |

---

## Part 3: The Fixes (Priority Order)

### Fix 1: Make Stage 7 evaluate real outputs
**Priority:** CRITICAL
**Effort:** 2 days

```python
# BEFORE (workflows.py:2234):
quality_assessor = OutputQualityAssessor(llm_client=None, use_llm=False)
quality_summary = f"Workflow execution with {total_subtasks} subtasks..."

# AFTER:
llm_manager = create_llm_manager(service_name="orchestrator")
quality_assessor = OutputQualityAssessor(
    llm_client=llm_manager,
    use_llm=True
)

# Build quality input from ACTUAL agent outputs, not metadata
actual_outputs = []
for step in execution_results:
    if step.get("result"):
        actual_outputs.append({
            "subtask": step["description"],
            "agent": step["agent_name"],
            "output": step["result"][:2000],  # Truncate for token budget
            "tokens_used": step.get("tokens_used", 0),
        })

quality_summary = json.dumps({
    "task": workflow.name,
    "subtask_outputs": actual_outputs,
    "overall_success_rate": completed_count / total_count,
})
```

**Validation:** Run a workflow, verify Stage 7 score changes meaningfully between a good and bad execution.

### Fix 2: Close the learning loop (Stage 6 → Stage 2)
**Priority:** CRITICAL
**Effort:** 1 day

Verify and fix that `LLMAgentSelector` includes agent performance data in its selection prompt.

**Location:** `core/llm/llm_agent_selector.py`

The agent selection prompt must include for each candidate agent:
```
Agent: {name}
  Skills: {skills}
  Success Rate: {performance_metrics.success_rate}  ← THIS
  Tasks Completed: {performance_metrics.total_tasks_executed}  ← THIS
  Avg Quality: {quality_score}  ← THIS
```

If the selector prompt doesn't include these, the LLM has no performance data to reason about. The learning loop writes to `/dev/null`.

**Formula (existing, from PRD-10):**
```
Agent Score = 0.4 × skill_match + 0.3 × performance_history + 0.2 × (1 - load) + 0.1 × recency
```

The LLM should receive these 4 signals for each candidate agent.

### Fix 3: Re-enable context optimization
**Priority:** HIGH
**Effort:** 3 days

**Location:** `modules/orchestrator/stages/context_engineering.py`

Re-enable the mathematical optimization pipeline:

```
Input → Retrieve (RAG + Memory + CodeGraph)
      → Filter (Shannon Entropy > 4.0 threshold)
      → Diversify (MMR with λ=0.7)
      → Optimize (Knapsack for token budget)
      → Assemble (Progressive Complexity level)
      → Output
```

**Shannon Entropy Filter:**
```
H(X) = -Σ p(xᵢ) × log₂(p(xᵢ))
```
Remove context items with H(X) < 4.0 (low information content — boilerplate, repetitive text).

**MMR Diversity Selection:**
```
MMR = arg max [λ × Sim(dᵢ, q) - (1-λ) × max Sim(dᵢ, dⱼ)]
```
Where λ=0.7 (70% relevance, 30% diversity). Prevents redundant context items.

**Knapsack Token Budget:**
```
Maximize: Σ value(cᵢ) × selected(cᵢ)
Subject to: Σ tokens(cᵢ) × selected(cᵢ) ≤ token_budget
```
Where `value(cᵢ)` = cosine_similarity × information_density.

**Why it was disabled:** Likely the `ContextOptimizer` class had an initialization issue or missing dependency. Debug, fix, re-enable behind a feature flag (`ENABLE_CONTEXT_OPTIMIZATION=true`).

### Fix 4: Respect `requires_context` from decomposer
**Priority:** HIGH
**Effort:** 0.5 days

```python
# BEFORE (workflows.py:1898):
subtasks_needing_context = steps  # FIX: Force all subtasks to get context engineering

# AFTER:
subtasks_needing_context = [
    step for step in steps
    if step.get("requires_context", True)  # Default to True for safety
]
subtasks_skipping_context = [
    step for step in steps
    if not step.get("requires_context", True)
]
if subtasks_skipping_context:
    logger.info(f"Skipping context engineering for {len(subtasks_skipping_context)} subtasks")
```

### Fix 5: Unify orchestrator implementations
**Priority:** HIGH
**Effort:** 5 days

Extract the inline 2700-line `execute_workflow_with_progress()` into a proper service class that uses the stage components from `modules/orchestrator/stages/`. Either:

**Option A:** Refactor `execute_workflow_with_progress()` to delegate to `EnhancedOrchestratorService` (clean, but risk of breaking the working path)

**Option B:** Gradually replace inline stage logic with calls to the stage components (safer, incremental)

Recommend **Option B** — replace one stage at a time, test between each.

### Fix 6: Propagate graph metadata to subtasks
**Priority:** MEDIUM
**Effort:** 0.5 days

```python
# In execute_workflow_with_progress(), after decomposition:
graph_analysis = decomposition_result.get("graph_analysis", {})
for step in steps:
    step["graph_metadata"] = {
        "parallel_group": next(
            (i for i, group in enumerate(graph_analysis.get("parallel_groups", []))
             if step["subtask_id"] in group),
            0
        ),
        "dependencies": step.get("dependencies", []),
        "execution_order": graph_analysis.get("execution_order", []),
    }
```

---

## Part 4: Stage 1-2 Optimization Strategy (PRD-58 Integration)

### Why Stages 1-2 Are the Highest Leverage

```
Stage 1 (Decompose) quality = 70%
Stage 2 (Select) quality = 80%
Combined = 0.7 × 0.8 = 56% correct pipeline setup
→ 44% of workflows start with the wrong plan or wrong agents
```

Improving Stage 1 to 90% and Stage 2 to 90%:
```
Combined = 0.9 × 0.9 = 81% correct pipeline setup
→ Only 19% suboptimal starts (57% reduction in errors)
```

### PRD-58 Integration Plan

Once the Prompt Registry (PRD-58 Phase 1A) is live:

1. **Evaluate current Stage 1-2 prompts** — Run FutureAGI evaluation on `task-decomposer` and `agent-selector` slugs against a test dataset of 30+ real workflow requests

2. **Optimize with FutureAGI** — Use Bayesian optimization (10 iterations) to improve both prompts. Target: +15% instruction adherence on decomposer, +20% task completion on selector

3. **A/B test** — Route 20% traffic to optimized prompts, measure quality score differences in Stage 7 (now that it evaluates real outputs)

4. **Activate** — When optimized prompts show statistically significant improvement, activate them

### Test Dataset for Stage 1 (Task Decomposer)

```json
{
    "prompt_slug": "task-decomposer",
    "test_cases": [
        {
            "id": "decompose-001",
            "input": "Build a REST API for user authentication with JWT tokens",
            "expected_output": {
                "subtasks": [
                    {"description": "Design auth database schema", "agent_type": "data_analyst"},
                    {"description": "Implement user registration endpoint", "agent_type": "developer"},
                    {"description": "Implement login with JWT generation", "agent_type": "developer"},
                    {"description": "Add middleware for token verification", "agent_type": "developer"},
                    {"description": "Write integration tests", "agent_type": "tester"}
                ],
                "execution_strategy": "hybrid"
            },
            "tags": ["multi-step", "development"]
        },
        {
            "id": "decompose-002",
            "input": "Summarize this quarterly report and email it to the team",
            "expected_output": {
                "subtasks": [
                    {"description": "Read and summarize the quarterly report", "agent_type": "analyst"},
                    {"description": "Draft email with summary", "agent_type": "writer"},
                    {"description": "Send email to team distribution list", "agent_type": "communication"}
                ],
                "execution_strategy": "sequential"
            },
            "tags": ["simple", "communication"]
        }
    ]
}
```

---

## Part 5: The TaskRunner Bridge (PRD-56 Integration)

### Architecture: From Monolith to Distributed

```
CURRENT (monolith):
  execute_workflow_with_progress()
    └── Stage 4: AgentExecutionManager
          └── asyncio.gather(subtask_1, subtask_2, subtask_3)
                All running in the same Python process

PHASE 1 (TaskRunner abstraction, 0 behavior change):
  execute_workflow_with_progress()
    └── Stage 4: AgentExecutionManager
          └── task_runner.submit(AgentTask) × N
                └── LocalTaskRunner: asyncio.create_task()  ← same as today

PHASE 2 (queued workers):
  execute_workflow_with_progress()
    └── Stage 4: AgentExecutionManager
          └── task_runner.submit(AgentTask) × N
                └── QueuedTaskRunner: Redis RPUSH → ARQ worker picks up
                      Workers run in separate Railway containers

PHASE 3 (K8s pods):
  execute_workflow_with_progress()
    └── Stage 4: AgentExecutionManager
          └── task_runner.submit(AgentTask) × N
                └── KubernetesTaskRunner: kubectl create job
                      Each subtask = one ephemeral pod
                      KEDA scales pods from zero based on queue depth
```

### AgentTask Model (from PRD-56)

```python
@dataclass
class AgentTask:
    task_id: UUID
    workspace_id: UUID
    execution_id: UUID

    # What to do
    task_type: str           # "workflow_subtask" | "chat_agent" | "recipe_step"
    agent_id: int
    agent_config: dict       # model_config, persona, system_prompt
    subtask: dict            # description, requirements, context

    # Resources
    priority: str            # "low" | "normal" | "high" | "critical"
    timeout_seconds: int     # default 300
    max_retries: int         # default 2
    resources: TaskResources # cpu_millicores, memory_mb

    # Context
    shared_context_id: UUID  # Reference to SharedContextManager store
    memory_scope: str        # Mem0 user_id for memory retrieval
    tools: list[str]         # Available tool names
```

### TaskRunner Interface

```python
class TaskRunner(ABC):
    @abstractmethod
    async def submit(self, task: AgentTask) -> TaskHandle:
        """Submit a task for execution. Returns immediately."""

    @abstractmethod
    async def get_result(self, handle: TaskHandle, timeout: float = 300) -> TaskResult:
        """Wait for and return the task result."""

    @abstractmethod
    async def cancel(self, handle: TaskHandle) -> bool:
        """Cancel a running task."""

    @abstractmethod
    async def get_status(self, handle: TaskHandle) -> TaskStatus:
        """Check task status without blocking."""
```

### Implementation Priority

```
Week 1-2:  LocalTaskRunner (same behavior, new interface)
Week 3-4:  QueuedTaskRunner (Redis + ARQ workers)
Month 2-3: KubernetesTaskRunner (K8s Jobs)
```

The `LocalTaskRunner` is a pure refactor — zero behavior change, just wrapping the existing `asyncio.gather()` in the TaskRunner interface. This is the keystone that unlocks everything.

---

## Part 6: Path to Neural Swarm Architecture

### From SharedContextManager to Neural Field

The `SharedContextManager` in Stage 4 is the embryo of the neural field from the Context-Engineering research. Currently it's an in-memory dict scoped to an execution. Here's how it evolves:

```
CURRENT:
  SharedContextManager = dict[str, Any]  # In-memory, dies with process

PHASE 1 (Redis-backed):
  SharedContextManager → Redis hash
  Key: swarm:{execution_id}:field
  Agents read/write from any process
  TTL: execution duration + 1 hour (temporal decay)

PHASE 2 (Vector field):
  SharedContextManager → Redis hash + pgvector embeddings
  Each agent's contribution is embedded
  Other agents retrieve by semantic similarity, not key lookup
  This IS the neural field Ψ(x,t) — continuous semantic space

PHASE 3 (Field dynamics):
  Add field equations from Context-Engineering research:

  Ψ(x, t+dt) = Ψ(x, t) + dt × [
    -∇V(Ψ)           # Gradient of potential (task objective pulls field)
    + D∇²Ψ           # Diffusion (knowledge spreads between agents)
    + Σᵢ δ(x-xᵢ)Aᵢ  # Agent contributions (point sources)
  ]

  Where:
  - Ψ(x,t) ∈ ℝⁿ is the field state (embedding space)
  - V(Ψ) is the task potential (objective function)
  - D is the diffusion coefficient (knowledge sharing rate)
  - Aᵢ is agent i's output (injected at position xᵢ)
```

### From Learning Update to Attractor Dynamics

Stage 6 currently uses exponential moving average:
```
updated_rate = 0.9 × current_rate + 0.1 × new_rate
```

This can evolve into attractor dynamics:

```python
# Current: simple EMA
agent.success_rate = 0.9 * agent.success_rate + 0.1 * execution_success

# Evolved: attractor dynamics
# Successful patterns become attractors that pull future selections
class AttractorLandscape:
    def update(self, agent_id: int, task_type: str, quality: float):
        """
        Quality > threshold → strengthen attractor (agent gets selected more)
        Quality < threshold → weaken attractor (agent gets selected less)

        dA/dt = α × (Q - Q_threshold) × A + noise

        Where:
        - A is attractor strength
        - Q is quality score
        - Q_threshold is the quality threshold (0.7)
        - α is learning rate (0.1)
        - noise prevents getting stuck in local optima
        """
        attractor_key = f"{agent_id}:{task_type}"
        current = self.attractors.get(attractor_key, 0.5)
        delta = self.alpha * (quality - self.threshold) * current
        noise = random.gauss(0, 0.01)
        self.attractors[attractor_key] = max(0, min(1, current + delta + noise))
```

### From Fixed Pipeline to Swarm Orchestration

The end state — microagents in K8s with neural field shared consciousness:

```
User Request
     │
     ▼
Universal Router (PRD-50)
     │
     ├─ Simple task → Single agent pod (Atom/Molecule complexity)
     │
     ├─ Complex task → Swarm Orchestrator
     │      │
     │      ├─ Phase Selector chooses stages
     │      ├─ Task Decomposer splits into subtasks
     │      ├─ Agent Selector chooses specialist models
     │      │
     │      └─ KubernetesTaskRunner.submit(AgentTask × N)
     │              │
     │              ├─ Pod 1: Claude Opus (research)     ─┐
     │              ├─ Pod 2: GPT-4 (code generation)    ─┤─ Neural Field
     │              ├─ Pod 3: Gemini (analysis)           ─┤  (Redis + pgvector)
     │              └─ Pod 4: DeepSeek (validation)       ─┘
     │                      │
     │                      ▼
     │              Field Coherence Check
     │              (do agents agree? Is the field stable?)
     │                      │
     │                      ▼
     │              Result Synthesis
     │
     └─ Workflow → Recipe executor (pre-defined steps)
```

Key properties of the swarm:
- **Ephemeral:** Pods spin up for a task and die after. KEDA scales from zero.
- **Heterogeneous:** Different models for different subtasks (Claude for research, GPT-4 for code, etc.)
- **Field-coordinated:** Agents don't message each other. They read/write to the shared neural field. Knowledge propagates via field dynamics, not explicit routing.
- **Self-improving:** Attractor dynamics mean the system converges on optimal agent-model-task combinations over time.

---

## Part 7: Implementation Plan

### Phase 1: Stabilize (Weeks 1-3)
*Goal: Make the current 9-stage pipeline reliable*

| # | Task | Effort | Depends On | Files |
|---|------|--------|-----------|-------|
| 1.1 | Fix Stage 7: evaluate real outputs (LLM-based) | 2 days | — | `api/workflows.py`, `stages/quality_assessor.py` |
| 1.2 | Close learning loop: verify Stage 6 → Stage 2 data flow | 1 day | — | `core/llm/llm_agent_selector.py`, `learning/engine/core.py` |
| 1.3 | Re-enable context optimization (entropy + MMR + knapsack) | 3 days | — | `stages/context_engineering.py`, `search/optimization/` |
| 1.4 | Respect `requires_context` from decomposer | 0.5 days | — | `api/workflows.py` |
| 1.5 | Propagate graph metadata to subtasks | 0.5 days | — | `api/workflows.py` |
| 1.6 | Integrate PRD-58 prompts for Stages 1-2 | 2 days | PRD-58 Phase 1A | `stages/task_decomposer.py`, `llm_agent_selector.py` |
| 1.7 | Run FutureAGI evaluation on Stage 1-2 prompts | 1 day | 1.6 | Eval datasets |
| 1.8 | Optimize Stage 1-2 prompts via FutureAGI | 2 days | 1.7 | Prompt versions |

**Phase 1 total: ~12 days**

### Phase 2: Dynamic Phases (Weeks 4-6)
*Goal: Replace fixed 9-stage sequence with PhaseSelector*

| # | Task | Effort | Depends On | Files |
|---|------|--------|-----------|-------|
| 2.1 | Build `PhaseSelector` class | 2 days | Phase 1 | `modules/orchestrator/phase_selector.py` (NEW) |
| 2.2 | Extract stages into composable pipeline | 3 days | 2.1 | `modules/orchestrator/pipeline.py` (NEW) |
| 2.3 | Wire `execute_workflow_with_progress()` to use pipeline | 3 days | 2.2 | `api/workflows.py` |
| 2.4 | Add Stage 2b: Inter-Agent Negotiation | 2 days | 2.2 | `stages/agent_negotiation.py` (NEW) |
| 2.5 | Add Stage 3b: Prompt Optimization check | 1 day | 2.2, PRD-58 | `stages/prompt_optimization.py` (NEW) |
| 2.6 | SSE streaming for dynamic phases | 2 days | 2.3 | `consumers/workflows/streaming.py` |

**Phase 2 total: ~13 days**

### Phase 3: TaskRunner Bridge (Weeks 7-10)
*Goal: Extract execution into TaskRunner interface*

| # | Task | Effort | Depends On | Files |
|---|------|--------|-----------|-------|
| 3.1 | Define `TaskRunner` interface + `AgentTask` model | 1 day | — | `core/task_runner/base.py` (NEW) |
| 3.2 | Implement `LocalTaskRunner` (wraps current asyncio) | 2 days | 3.1 | `core/task_runner/local.py` (NEW) |
| 3.3 | Refactor `AgentExecutionManager` to use TaskRunner | 3 days | 3.2 | `modules/agents/execution/execution_manager.py` |
| 3.4 | Move `SharedContextManager` to Redis | 2 days | 3.3 | `modules/orchestrator/shared_context.py` |
| 3.5 | Implement `QueuedTaskRunner` (Redis + ARQ) | 5 days | 3.3, 3.4 | `core/task_runner/queued.py` (NEW) |
| 3.6 | Deploy worker containers on Railway | 2 days | 3.5 | `Dockerfile.worker`, Railway config |

**Phase 3 total: ~15 days**

### Phase 4: Neural Field Prototype (Months 3-4)
*Goal: Implement field-based agent coordination*

| # | Task | Effort | Depends On | Files |
|---|------|--------|-----------|-------|
| 4.1 | Extend SharedContext with vector embeddings | 3 days | Phase 3 | `core/neural_field/field_store.py` (NEW) |
| 4.2 | Implement field read (semantic retrieval) | 2 days | 4.1 | `core/neural_field/field_reader.py` (NEW) |
| 4.3 | Implement field write (contribution + diffusion) | 2 days | 4.1 | `core/neural_field/field_writer.py` (NEW) |
| 4.4 | Replace explicit agent messaging with field ops | 3 days | 4.2, 4.3 | `execution_manager.py` |
| 4.5 | Implement attractor dynamics for learning | 3 days | 4.4 | `core/neural_field/attractor.py` (NEW) |
| 4.6 | Field coherence metric (do agents agree?) | 2 days | 4.4 | `core/neural_field/coherence.py` (NEW) |

**Phase 4 total: ~15 days**

### Phase 5: K8s Microagent Swarms (Months 4-6)
*Goal: Full distributed execution*

| # | Task | Effort | Depends On | Files |
|---|------|--------|-----------|-------|
| 5.1 | Implement `KubernetesTaskRunner` | 5 days | Phase 3 | `core/task_runner/kubernetes.py` (NEW) |
| 5.2 | KEDA ScaledJob configuration | 2 days | 5.1 | K8s manifests |
| 5.3 | Workspace namespace isolation | 2 days | 5.1 | K8s RBAC |
| 5.4 | Multi-model pod selection (right model per subtask) | 3 days | 5.1 | `core/task_runner/model_router.py` (NEW) |
| 5.5 | Neural field across pods (Redis + pgvector) | 3 days | Phase 4, 5.1 | Field store adaptation |
| 5.6 | Swarm monitoring dashboard | 5 days | 5.5 | Frontend + API |

**Phase 5 total: ~20 days**

---

## Part 8: Mathematical Foundations Reference

### Context Engineering (used in Stage 3)

**Shannon Entropy** — Filter low-information content:
```
H(X) = -Σᵢ p(xᵢ) × log₂(p(xᵢ))
```
Threshold: H(X) > 4.0 bits for inclusion.

**Cosine Similarity** — Semantic relevance:
```
cos(θ) = (A · B) / (||A|| × ||B||)
```
Threshold: cos(θ) > 0.7 for relevant context.

**MMR (Maximal Marginal Relevance)** — Balance relevance and diversity:
```
MMR = arg max [λ × Sim(dᵢ, q) - (1-λ) × max Sim(dᵢ, dⱼ)]
```
λ = 0.7 (70% relevance, 30% diversity).

**Knapsack Optimization** — Maximize information within token budget:
```
Maximize: Σ value(cᵢ) × selected(cᵢ)
Subject to: Σ tokens(cᵢ) × selected(cᵢ) ≤ budget
Time complexity: O(n × budget)
```

### Agent Selection (used in Stage 2)

**Multi-dimensional scoring:**
```
Score = 0.4 × skill_match + 0.3 × performance_history + 0.2 × (1 - load) + 0.1 × recency
```

**Exponential Moving Average for learning:**
```
μₜ = (1 - α) × μₜ₋₁ + α × xₜ
```
Where α = 0.1 (learning rate).

### Quality Assessment (used in Stage 7)

**Weighted quality score:**
```
Q = 0.30 × completeness + 0.25 × accuracy + 0.20 × efficiency + 0.15 × reliability + 0.10 × coherence
```

**Confidence interval (from ProbabilityTheory):**
```
CI = x̄ ± z × (σ / √n)
```
Where z = 1.96 for 95% confidence.

### Neural Field Dynamics (Phase 4-5)

**Field evolution equation:**
```
∂Ψ/∂t = -∇V(Ψ) + D∇²Ψ + Σᵢ δ(x-xᵢ)Aᵢ
```
Where:
- Ψ(x,t) ∈ ℝⁿ — field state in embedding space
- V(Ψ) — task potential (objective function gradient)
- D — diffusion coefficient (knowledge sharing rate, tunable)
- Aᵢ — agent i's contribution (injected at semantic position xᵢ)

**Field coherence metric:**
```
C = 1 - (1/N²) × Σᵢ Σⱼ ||Ψᵢ - Ψⱼ||²
```
C → 1 when all agents converge (agreement). C → 0 when agents diverge (conflict).

**Attractor dynamics:**
```
dA/dt = α × (Q - Q_threshold) × A + η
```
Where:
- A — attractor strength for (agent, task_type) pair
- Q — observed quality score
- Q_threshold = 0.7 (minimum acceptable)
- α = 0.1 (learning rate)
- η ~ N(0, 0.01) — noise term (prevents local optima)

---

## Part 9: New Files

| # | File | Purpose | Phase |
|---|------|---------|-------|
| 1 | `modules/orchestrator/phase_selector.py` | Dynamic phase selection based on complexity + mode | 2 |
| 2 | `modules/orchestrator/pipeline.py` | Composable stage pipeline executor | 2 |
| 3 | `modules/orchestrator/stages/agent_negotiation.py` | Stage 2b: Inter-agent task review | 2 |
| 4 | `modules/orchestrator/stages/prompt_optimization.py` | Stage 3b: PRD-58 prompt variant check | 2 |
| 5 | `core/task_runner/base.py` | TaskRunner interface + AgentTask model | 3 |
| 6 | `core/task_runner/local.py` | LocalTaskRunner (asyncio wrapper) | 3 |
| 7 | `core/task_runner/queued.py` | QueuedTaskRunner (Redis + ARQ) | 3 |
| 8 | `core/task_runner/factory.py` | TaskRunner factory (env-based selection) | 3 |
| 9 | `core/neural_field/field_store.py` | Redis + pgvector neural field storage | 4 |
| 10 | `core/neural_field/field_reader.py` | Semantic field retrieval | 4 |
| 11 | `core/neural_field/field_writer.py` | Field contribution + diffusion | 4 |
| 12 | `core/neural_field/attractor.py` | Attractor dynamics for learning | 4 |
| 13 | `core/neural_field/coherence.py` | Field coherence metric | 4 |
| 14 | `core/task_runner/kubernetes.py` | KubernetesTaskRunner (K8s Jobs) | 5 |

## Modified Files

| # | File | Changes | Phase |
|---|------|---------|-------|
| 1 | `api/workflows.py` | Stage 7 fix, context fix, graph propagation, pipeline integration | 1, 2 |
| 2 | `modules/orchestrator/stages/context_engineering.py` | Re-enable optimization | 1 |
| 3 | `modules/orchestrator/stages/quality_assessor.py` | Enable LLM assessment | 1 |
| 4 | `core/llm/llm_agent_selector.py` | Include performance metrics in prompt | 1 |
| 5 | `modules/learning/engine/core.py` | Verify metric persistence | 1 |
| 6 | `modules/agents/execution/execution_manager.py` | TaskRunner integration | 3 |
| 7 | `consumers/workflows/streaming.py` | Dynamic phase SSE events | 2 |

---

## Success Metrics

### Phase 1 (Stabilize)

| Metric | Current | Target |
|---|---|---|
| Stage 7 quality score variance | ~0 (always 0.65-0.75) | Meaningful range (0.3-0.95) |
| Stage 1-2 prompt eval score | Unknown | > 85% instruction adherence |
| Learning loop closure | Unverified | Agent with 10+ executions selected 20% more for matching tasks |
| Context token waste | Unknown (all subtasks get context) | 30% reduction via `requires_context` gating |

### Phase 2 (Dynamic Phases)

| Metric | Current | Target |
|---|---|---|
| Simple task latency | 3-10s (all 9 stages) | < 2s (3 stages for Atom tasks) |
| Token cost per simple task | ~12,000 tokens | < 3,000 tokens (skip PLAN + EVALUATE) |
| Multi-agent coordination quality | N/A | Measurable coherence score > 0.7 |

### Phase 3 (TaskRunner)

| Metric | Current | Target |
|---|---|---|
| Max concurrent subtasks | ~3 (asyncio, single process) | 10+ (worker pool) |
| Execution isolation | None (shared process) | Full (separate workers) |
| Failure blast radius | All subtasks die | Only failed subtask retries |

### Phase 5 (K8s Swarms)

| Metric | Target |
|---|---|
| Scale-to-zero time | < 30 seconds |
| Pod spin-up latency | < 10 seconds |
| Max concurrent agents per workspace | 20+ |
| Field coherence on multi-agent tasks | > 0.75 |
| Cost per workflow (10 subtasks) | < $0.15 compute + LLM costs |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Re-enabling context optimization introduces regressions | Degraded context quality | Feature flag `ENABLE_CONTEXT_OPTIMIZATION`. A/B test before full rollout. |
| LLM-based Stage 7 adds cost | ~$0.005/evaluation | Only use LLM quality for Organ+ complexity. Heuristic for Atom/Molecule. |
| TaskRunner refactor breaks execution | Workflows stop working | `LocalTaskRunner` is a pure wrapper — zero behavior change. Integration tests. |
| Redis-backed SharedContext adds latency | Slower inter-agent coordination | Redis HSET/HGET is < 1ms. Net effect is negligible. |
| K8s adds infrastructure complexity | Ops burden | Start with managed K8s (GKE Autopilot). KEDA handles scaling automatically. |
| Neural field math is too theoretical | Wasted effort | Phase 4 is optional. Phases 1-3 deliver concrete value independently. |

---

## Open Questions

1. **PRD-58 timeline:** Phase 1 fixes depend on PRD-58's Prompt Registry for Stage 1-2 optimization. Is PRD-58 Phase 1A (registry + seeding) on track?

2. **Context optimizer debug:** Why was the mathematical optimization disabled? Is it a dependency issue, a performance issue, or a quality issue? Need to investigate before re-enabling.

3. **QueuedTaskRunner infrastructure:** Should workers run on the same Railway project (service-level scaling) or a separate compute provider (e.g., Fly.io, Modal)?

4. **Neural field complexity:** Is the field dynamics math from Context-Engineering ready for implementation, or does it need more research? The diffusion equation requires discretization choices (grid resolution, time step) that affect both accuracy and performance.

5. **Backward compatibility:** When we switch from fixed 9-stage to dynamic phases, do existing workflow execution records need migration? The `WorkflowExecution.input_data` JSON stores per-stage metadata keyed by stage name.

---

## Glossary

| Term | Definition |
|---|---|
| **Neural Field** | Continuous semantic vector space shared by multiple agents. Replaces explicit message-passing with implicit field dynamics. |
| **Attractor** | A stable pattern in the learning landscape that the system converges toward. High-quality agent-task combinations become attractors. |
| **Field Coherence** | Measure of agreement between agents' contributions to the shared field. High coherence = agents are aligned. |
| **Progressive Complexity** | Automatos' 5-level hierarchy: Atom → Molecule → Cell → Organ → Organism. Each level adds context sophistication only when needed. |
| **TaskRunner** | Abstract interface for task execution. Implementations: LocalTaskRunner (asyncio), QueuedTaskRunner (Redis), KubernetesTaskRunner (K8s Jobs). |
| **Phase Selector** | Component that determines which workflow phases and stages to execute based on task complexity and execution mode. |
| **MMR** | Maximal Marginal Relevance. Algorithm that balances relevance and diversity when selecting context items. |
| **Knapsack** | Optimization algorithm that maximizes information value within a token budget constraint. |
| **KEDA** | Kubernetes Event-Driven Autoscaler. Scales pods from zero based on queue depth. |
