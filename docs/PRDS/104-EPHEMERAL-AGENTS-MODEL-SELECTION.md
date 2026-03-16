# PRD-104 — Ephemeral Agents & Model Selection

**Version:** 1.0
**Type:** Research + Design
**Status:** Complete — Ready for Peer Review
**Priority:** P0
**Dependencies:** PRD-100 (Research Master), PRD-101 (Mission Schema — `contractor_config` JSONB), PRD-102 (Coordinator Architecture — agent assignment)
**Blocks:** PRD-82C (Parallel Execution + Budget + Contractors)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-15

---

## 1. Problem Statement

### 1.1 The Gap

Automatos agents are **permanent residents**. Every agent occupies a row in the `agents` table (45+ columns), has a heartbeat config, skills, tool assignments, persona, voice profile, and semantic embeddings. Creating one means a DB write, tool resolution, optional LLM verification, and caching. Deleting one cascades through 11 dependent tables.

This is correct for **roster agents** — permanent team members with personality, memory, and ongoing responsibilities.

It is **completely wrong for mission work.** When the coordinator decomposes a goal into 4 subtasks, it needs focused agents in <100ms each, executing in parallel, reporting results, and disappearing. No persona. No heartbeat. No marketplace category. No voice profile.

### 1.2 What This PRD Delivers

1. **Contractor Agent Lifecycle** — spawn, configure, execute, report, destroy — with <100ms in-memory creation
2. **Model-Per-Role Strategy** — which models for which agent roles, with cost/quality tradeoffs
3. **Dynamic Tool Scoping** — coordinator specifies tools per contractor, no DB assignment needed
4. **Mission-Scoped Memory** — contractors share mission context but nothing persists after
5. **Auto-Cleanup** — TTL-based and mission-completion-based destruction
6. **Integration Design** — how contractors flow through existing `AgentFactory.execute_with_prompt()`

---

## 2. Prior Art: Ephemeral Agent Patterns

### 2.1 System-by-System Analysis

#### Agent Zero (frdel/agent-zero)

Agent Zero spawns subordinates via `Agent(number+1, fresh_config, SHARED_context)` — only customization is `profile` (a prompt directory). Key characteristics:
- Single subordinate at a time (linked list, not fan-out)
- Memory is SHARED (same FAISS index) — no isolation
- Conversation sealing via `history.new_topic()` — progressive compression (50% current / 30% topics / 20% bulks)
- Utility model for compression — cheap model handles internal coordination
- No timeouts, no budgets, no explicit destruction

**What we adopt:** Conversation sealing pattern (progressive context compression). Utility model for coordination overhead.

**What we reject:** Single-subordinate limitation (we need parallel fan-out). Shared memory (we need mission-scoped isolation). No lifecycle management.

#### AutoGen (microsoft/autogen)

AutoGen's agent is fully described by a config dict: `(name, system_message, llm_config, tools, description)`. The `GroupChatManager` selects speakers via LLM or deterministic rules. Swarm handoff priority: tool-returned agent → OnCondition → AFTER_WORK fallback. `context_variables` dict is shared mutable state.

**What we adopt:** Agent-as-config-dict pattern — this IS our contractor config model. Swarm handoff priority ordering for coordinator task transitions. `context_variables` as mission-scoped shared state.

**What we reject:** No explicit cleanup (Python GC). LLM-based speaker selection (expensive, non-deterministic).

#### Kubernetes Jobs

K8s Jobs define the infrastructure pattern for ephemeral workloads:
- `ttlSecondsAfterFinished` = auto-cleanup after completion
- `activeDeadlineSeconds` = hard timeout (overrides retries)
- `backoffLimit` = retry cap with exponential backoff
- `podFailurePolicy` rules: FailJob (fatal error codes), Ignore (infra disruption), Count (normal retry)
- Artifact preservation: write results to external storage BEFORE container exit

**What we adopt:** TTL-based cleanup. Hard timeout. Backoff limit. Failure classification (fatal vs retryable). Result persistence before destruction.

**What we reject:** Full container lifecycle management (premature — Phase 3/K8s scope).

### 2.2 Model Routing Research

#### RouteLLM (ICLR 2025, UC Berkeley/Anyscale/Canva)

Matrix factorization router achieves **75% cost reduction at 95% quality** on MT-Bench. Key finding: math/reasoning tasks need expensive models; conversational/summarization routes cheap. Static role-based mapping captures 80% of the routing value without any ML infrastructure.

#### BudgetMLAgent (AIMLSystems 2024)

Cascade pattern: free model → cheap model → expensive model, escalating only when output quality is insufficient. Achieved **96% cost reduction** vs single GPT-4 agent. Proves cascade/escalation is viable for multi-agent budgets.

#### OpenRouter (Existing Infrastructure)

Already integrated with 340 models. Provider routing params available:
- `sort`: `'price'` | `'throughput'` | `'latency'`
- `max_price`: Cost ceiling per call
- `preferred_min_throughput`: Min tokens/sec

**Decision: Static role→model mapping for v1.** No ML routing. RouteLLM proves static mapping captures 80% of value. OpenRouter's `sort`/`max_price` params are the v1 selection interface. Telemetry (PRD-106) provides data for future dynamic routing.

---

## 3. Contractor Agent Lifecycle

### 3.1 State Machine

```
SPAWNING → READY → EXECUTING → REPORTING → CLEANUP → DESTROYED
    │                  │            │
    └── SPAWN_FAILED   │            │
                       └── FAILED ──┘
```

| State | Duration | What Happens |
|---|---|---|
| SPAWNING | <100ms | In-memory `AgentRuntime` created from `contractor_config` |
| READY | <10ms | Tools resolved, context prepared |
| EXECUTING | 3-300s | Agent running LLM calls via `execute_with_prompt()` |
| REPORTING | <100ms | Result written to `orchestration_tasks.result_reference` |
| CLEANUP | <50ms | Evict from `active_agents`, delete Redis keys, soft-delete DB row |
| DESTROYED | Terminal | Agent no longer exists |
| SPAWN_FAILED | Terminal | Config validation failed, tools unavailable, etc. |
| FAILED | Terminal | Execution crashed, timeout, or max retries exhausted |

### 3.2 Contractor Config Schema

The coordinator specifies contractor configuration in `orchestration_tasks.contractor_config` JSONB (defined in PRD-101):

```python
@dataclass(frozen=True)
class ContractorConfig:
    """
    Minimal config for an ephemeral contractor agent.
    Maps to AutoGen's agent-as-config-dict pattern.
    """
    role: str                          # "researcher", "analyst", "writer", "reviewer", "coder", "simple"
    model: Optional[str] = None        # Explicit model override. None → use role default.
    tools: list[str] = field(default_factory=list)  # Tool names to include
    system_prompt_override: Optional[str] = None     # Custom system prompt. None → auto-generate.
    max_tokens: int = 4096             # Max output tokens per LLM call
    max_tool_iterations: int = 10      # Max tool loop iterations (matches AgentFactory default)
    timeout_s: int = 300               # Hard timeout (K8s activeDeadlineSeconds pattern)
    ttl_s: int = 3600                  # Time-to-live after creation (cleanup safety net)


# JSONB schema for orchestration_tasks.contractor_config
CONTRACTOR_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "enum": ["researcher", "analyst", "writer", "reviewer", "coder", "simple"]},
        "model": {"type": ["string", "null"]},
        "tools": {"type": "array", "items": {"type": "string"}},
        "system_prompt_override": {"type": ["string", "null"]},
        "max_tokens": {"type": "integer", "default": 4096},
        "max_tool_iterations": {"type": "integer", "default": 10},
        "timeout_s": {"type": "integer", "default": 300},
        "ttl_s": {"type": "integer", "default": 3600},
    },
    "required": ["role"],
}
```

### 3.3 DB Record Strategy: Hybrid

**Decision: In-memory execution + async DB audit row.**

| Phase | Storage | Latency |
|---|---|---|
| Creation | In-memory `AgentRuntime` only | <50ms |
| During execution | In-memory; async DB write of minimal audit row | DB write: ~200ms (non-blocking) |
| After completion | `orchestration_tasks` has the result; audit row has agent metadata | Already written |
| Cleanup | Evict from `active_agents`; soft-delete audit row (`is_active = False`) | <50ms |

The audit row is minimal:
```sql
INSERT INTO agents (
    name, workspace_id, agent_type, model_config, is_ephemeral,
    mission_id, expires_at, created_at, is_active
) VALUES (
    'contractor-{mission_id}-{task_order}',  -- name
    :workspace_id,
    'contractor',
    :model_config_json,
    TRUE,
    :mission_id,
    NOW() + INTERVAL ':ttl_s seconds',
    NOW(),
    TRUE
);
```

**Why not skip the DB row entirely?** Board tasks need `assigned_agent_id` for display. The admin UI should show active contractors. The telemetry system (PRD-106) needs `agent_id` for attribution. The async write doesn't block execution.

---

## 4. AgentFactory Integration

### 4.1 New Method: create_ephemeral_agent()

```python
class AgentFactory:
    # ... existing methods unchanged ...

    async def create_ephemeral_agent(
        self,
        workspace_id: int,
        mission_id: int,
        config: ContractorConfig,
        task_context: dict,
    ) -> AgentRuntime:
        """
        Create a lightweight ephemeral agent for mission work.

        Unlike create_agent() which does DB write + LLM verification (~500ms),
        this creates an in-memory AgentRuntime in <50ms.

        The AgentRuntime is compatible with execute_with_prompt() — same
        tool loop, same retry logic, same response synthesis.
        """
        # 1. Resolve model from role mapping (or use explicit override)
        model = config.model or ROLE_MODEL_DEFAULTS[config.role]

        # 2. Resolve tools from explicit list (NOT from DB agent_tools)
        tools = await self._resolve_explicit_tools(config.tools, workspace_id)

        # 3. Build system prompt
        system_prompt = config.system_prompt_override or self._build_contractor_prompt(
            role=config.role,
            task_context=task_context,
        )

        # 4. Create in-memory runtime
        runtime = AgentRuntime(
            id=None,  # No DB ID yet
            name=f"contractor-{mission_id}-{task_context.get('task_order', 0)}",
            workspace_id=workspace_id,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            max_tokens=config.max_tokens,
            max_tool_iterations=config.max_tool_iterations,
        )

        # 5. Async DB audit row (non-blocking)
        asyncio.create_task(
            self._write_contractor_audit_row(runtime, workspace_id, mission_id, config)
        )

        return runtime

    async def _resolve_explicit_tools(
        self, tool_names: list[str], workspace_id: int
    ) -> list[dict]:
        """
        Resolve tools by name from ToolRegistry + ActionRegistry.
        Unlike get_tools_for_agent() which queries agent_tools DB table,
        this takes an explicit list of tool names.
        """
        all_tools = await self._get_all_available_tools(workspace_id)
        resolved = []
        for name in tool_names:
            tool = all_tools.get(name)
            if tool is None:
                raise ValueError(f"Tool '{name}' not found in workspace {workspace_id}")
            resolved.append(tool)
        return resolved

    async def cleanup_ephemeral_agents(self, mission_id: int) -> int:
        """
        Cleanup all contractors for a completed/cancelled mission.
        Returns count of agents cleaned up.
        """
        # 1. Evict from in-memory cache
        evicted = 0
        for key in list(self.active_agents.keys()):
            agent = self.active_agents[key]
            if getattr(agent, '_mission_id', None) == mission_id:
                del self.active_agents[key]
                evicted += 1

        # 2. Soft-delete DB audit rows
        await self._db.execute(
            update(Agent)
            .where(
                Agent.mission_id == mission_id,
                Agent.is_ephemeral == True,
            )
            .values(is_active=False)
        )

        return evicted
```

### 4.2 What Does NOT Change

- `execute_with_prompt()` tool loop (lines ~838-862) — same 10-iteration tool loop
- `_execute_tool_calls()` (lines ~958-1028) — same tool dispatch
- `unified_executor.execute_tool()` — same prefix-based routing
- Heartbeat tick pattern — roster agents continue unchanged
- Agent API endpoints — contractors created by coordinator, not user API

### 4.3 Hard Constraint: No Sub-Contractors

**Contractors cannot spawn sub-contractors.** This is architectural, not a simplification:

1. **Bounded cost:** Sub-contractors create unbounded agent trees. Budget enforcement becomes impossible — the coordinator can't pre-estimate cost for a tree of unknown depth.
2. **Observability:** The coordinator must see every executing agent. Sub-contractors would be invisible to the reconciliation tick.
3. **Debugging:** Flat coordinator→contractor traces (2 levels) are tractable. N-level traces are exponentially harder.
4. **Alternative:** If a task is too complex, the coordinator should decompose it into smaller tasks (replanning per PRD-102 Section 9), not delegate decomposition to the contractor.

---

## 5. Model-Per-Role Strategy

### 5.1 Role Taxonomy

| Role | Task Types | Model Tier | Rationale |
|---|---|---|---|
| `researcher` | Web search, document analysis, data gathering | Mid-tier + large context | Process lots of text, synthesize findings |
| `analyst` | Data analysis, comparison, structured output | Mid-tier | Good reasoning, structured generation |
| `writer` | Reports, documentation, content creation | Mid-tier | Good prose at high volume |
| `coder` | Code generation, debugging, refactoring | Top-tier or specialized | Code quality is critical |
| `reviewer` | Quality review, fact-checking, verification | Mid-tier, **different family from coder** | Cognitive diversity catches different bugs |
| `simple` | Classification, formatting, routing, extraction | Cheap | Minimal reasoning needed |

### 5.2 Default Model Mapping

```python
ROLE_MODEL_DEFAULTS: dict[str, str] = {
    "researcher": "anthropic/claude-sonnet-4-20250514",
    "analyst": "google/gemini-2.5-pro-preview-03-25",
    "writer": "anthropic/claude-sonnet-4-20250514",
    "coder": "anthropic/claude-sonnet-4-20250514",
    "reviewer": "openai/gpt-4o",  # Different family from coder
    "simple": "anthropic/claude-haiku-4-5-20251001",
}

# Model tier classification for scoring (PRD-102 AgentMatcher)
MODEL_TIERS: dict[str, str] = {
    "anthropic/claude-opus-4-20250514": "top",
    "anthropic/claude-sonnet-4-20250514": "mid",
    "anthropic/claude-haiku-4-5-20251001": "cheap",
    "openai/gpt-4o": "mid",
    "openai/gpt-4o-mini": "cheap",
    "google/gemini-2.5-pro-preview-03-25": "mid",
    "deepseek/deepseek-chat": "mid",
    # ... more models
}

# OpenRouter provider params per role
ROLE_OPENROUTER_PARAMS: dict[str, dict] = {
    "researcher": {"sort": "throughput", "max_price": "0.01"},
    "analyst": {"sort": "throughput"},
    "writer": {"sort": "throughput"},
    "coder": {"sort": "quality"},
    "reviewer": {"sort": "quality", "max_price": "0.005"},
    "simple": {"sort": "price", "max_price": "0.001"},
}
```

### 5.3 Cognitive Diversity Enforcement

**Hard rule: reviewer model MUST be from a different model family than the task executor.**

This isn't a preference — it's a quality requirement. Different model families have different failure modes. A Claude-generated analysis reviewed by Claude misses the same blind spots. A GPT review catches different issues.

```python
def enforce_cognitive_diversity(
    executor_model: str, reviewer_model: str
) -> str:
    """
    Ensure reviewer uses a different model family.
    Returns corrected reviewer model if needed.
    """
    executor_family = executor_model.split("/")[0]
    reviewer_family = reviewer_model.split("/")[0]

    if executor_family == reviewer_family:
        # Swap to a different family
        alternatives = {
            "anthropic": "openai/gpt-4o",
            "openai": "anthropic/claude-sonnet-4-20250514",
            "google": "openai/gpt-4o",
            "deepseek": "anthropic/claude-sonnet-4-20250514",
        }
        return alternatives.get(executor_family, "openai/gpt-4o")

    return reviewer_model
```

### 5.4 User Override Surface

Users can override model selection at mission creation:

```json
{
    "config": {
        "model_preferences": {
            "researcher": "google/gemini-2.5-pro-preview-03-25",
            "coder": "anthropic/claude-opus-4-20250514",
            "reviewer": "openai/gpt-4o"
        }
    }
}
```

Override priority: user preference > mission config > workspace defaults > role defaults.

### 5.5 Cost Estimation

```python
async def estimate_mission_cost(
    tasks: list[TaskSpec],
    model_overrides: dict[str, str] = {},
) -> MissionCostEstimate:
    """
    Pre-execution cost estimate for human approval.
    Uses OpenRouter pricing + average token counts per role.
    """
    ROLE_AVG_TOKENS: dict[str, tuple[int, int]] = {
        # (avg_input_tokens, avg_output_tokens)
        "researcher": (8000, 4000),
        "analyst": (6000, 3000),
        "writer": (5000, 4000),
        "coder": (8000, 5000),
        "reviewer": (6000, 2000),
        "simple": (2000, 500),
    }

    total = 0.0
    breakdown = []
    for task in tasks:
        model = model_overrides.get(task.task_type) or ROLE_MODEL_DEFAULTS[task.task_type]
        pricing = await get_model_pricing(model)  # From llm_models table
        avg_in, avg_out = ROLE_AVG_TOKENS.get(task.task_type, (5000, 3000))

        cost = (avg_in * pricing.input_per_1k / 1000) + (avg_out * pricing.output_per_1k / 1000)
        total += cost
        breakdown.append(TaskCostEstimate(
            task_order=task.task_order,
            model=model,
            estimated_cost_usd=cost,
        ))

    # Add verification overhead (PRD-103: ~2-4% of task cost)
    verification_overhead = total * 0.03
    # Add coordination overhead (PRD-102: 1-2 LLM calls at ~$0.05)
    coordination_overhead = 0.10

    return MissionCostEstimate(
        task_breakdown=breakdown,
        total_task_cost=total,
        verification_cost=verification_overhead,
        coordination_cost=coordination_overhead,
        total_estimated=total + verification_overhead + coordination_overhead,
    )
```

---

## 6. Memory Isolation

### 6.1 What Contractors Can Access

| Memory Layer | Access | Rationale |
|---|---|---|
| **Mission context** (prior task results) | READ | Injected by coordinator via task prompt. Contractor sees outputs from earlier tasks. |
| **Shared mission context** (`SharedContextPort`) | READ/WRITE | Via PRD-107 interface. Contractors inject findings; later agents query them. |
| **Redis session memory** | NONE | Mission-scoped, not session-scoped. Contractors have no chat history. |
| **Postgres short-term memory** | NONE | No L2 memory for ephemeral agents. |
| **Mem0 long-term memory** | NONE | Contractors do not read or write to Mem0. Mission-scoped only. |
| **RAG / document search** | READ | Via workspace tools (`workspace_read_file`, `platform_search_documents`). |
| **NL2SQL** | NONE | No workspace data queries for contractors. |

### 6.2 How Context Flows to Contractors

```python
def _build_task_prompt(
    self, task: OrchestrationTask, prior_results: list[TaskResult]
) -> str:
    """
    Build the prompt for a contractor agent.
    Includes mission context + task instructions + prior task outputs.
    """
    sections = [
        f"# Mission Goal\n{task.run.goal}\n",
        f"# Your Task\n{task.description}\n",
        f"# Success Criteria\n",
    ]

    for criterion in task.success_criteria:
        must = " (REQUIRED)" if criterion.get("must_pass") else ""
        sections.append(f"- {criterion['criterion']}{must}")

    if prior_results:
        sections.append("\n# Context from Prior Tasks\n")
        for result in prior_results:
            sections.append(f"## Task {result.task_order}: {result.title}\n")
            sections.append(f"{result.output_summary}\n")

    if task.retry_context:
        sections.append("\n# Feedback from Previous Attempt\n")
        sections.append(f"Attempt {task.retry_context['attempt']} of {task.retry_context['max_attempts']}\n")
        sections.append(f"{task.retry_context['feedback']}\n")
        for fc in task.retry_context.get("failed_criteria", []):
            sections.append(f"- {fc['criterion']}: {fc['reasoning']}\n")

    return "\n".join(sections)
```

---

## 7. Cleanup Automation

### 7.1 Three Cleanup Triggers

| Trigger | When | Action |
|---|---|---|
| **Mission completion** | All tasks terminal (completed/failed/cancelled) | `cleanup_ephemeral_agents(mission_id)` |
| **TTL expiry** | `expires_at` timestamp passed | Periodic GC sweep (every 5 min) |
| **Explicit cancel** | Human cancels mission | Same as mission completion |

### 7.2 GC Sweep

```python
async def gc_expired_contractors(self) -> int:
    """
    Periodic sweep to catch contractors whose TTL expired.
    Runs every 5 minutes via scheduler.
    Safety net — mission completion cleanup should handle most cases.
    """
    expired = await self._db.execute(
        select(Agent)
        .where(
            Agent.is_ephemeral == True,
            Agent.is_active == True,
            Agent.expires_at < datetime.utcnow(),
        )
    )

    cleaned = 0
    for agent in expired:
        # Evict from memory
        self.active_agents.pop(agent.id, None)
        # Soft-delete
        agent.is_active = False
        cleaned += 1

    return cleaned
```

### 7.3 What Persists After Cleanup

| Data | Location | Persists? |
|---|---|---|
| Task output | `orchestration_tasks.result_reference` → workspace file | Yes |
| Execution trace | `orchestration_events` | Yes |
| Cost/token metrics | `llm_usage` rows with `mission_task_id` | Yes |
| Verifier score | `orchestration_tasks.verifier_score` | Yes |
| Agent DB row | `agents` table (soft-deleted) | Yes (queryable for audit) |
| In-memory runtime | `AgentFactory.active_agents` | No (evicted) |
| Redis keys | Contractor-specific Redis entries | No (expired or deleted) |

---

## 8. Concurrency Control

### 8.1 Limits

| Limit | Default | Configurable | Enforcement |
|---|---|---|---|
| Max concurrent contractors per mission | 3 | Yes (mission config) | Dispatcher checks before spawn |
| Max concurrent contractors per workspace | 5 | Yes (workspace settings) | Matches `heartbeat_service.max_concurrent_per_workspace` |
| Max total contractors per mission | 20 | No (hard limit) | Validation in plan decomposition |

### 8.2 Backpressure

When all contractor slots are full, the coordinator queues tasks in `queued` state. The next tick's dispatch phase picks them up when a slot opens.

```python
async def _dispatch_phase(self, run: OrchestrationRun) -> None:
    # ... (from PRD-102 Section 4.2)

    # Check workspace-level contractor limit
    workspace_contractors = await self._count_active_contractors(run.workspace_id)
    workspace_limit = await self._get_workspace_contractor_limit(run.workspace_id)
    workspace_slots = max(0, workspace_limit - workspace_contractors)

    # Use the more restrictive limit
    effective_slots = min(slots_available, workspace_slots)

    for task in ready_tasks[:effective_slots]:
        await self._dispatcher.dispatch_task(run, task)
```

---

## 9. Failure Classification

Following K8s `podFailurePolicy` and Prefect's CRASHED/FAILED distinction:

| Failure Type | Examples | Retryable? | Strategy |
|---|---|---|---|
| **Infrastructure** (CRASHED) | LLM timeout, rate limit 429, network error, OOM | Yes (auto) | Same config, exponential backoff |
| **Config** (FATAL) | Invalid model, tool not found, auth failure | No | Fail immediately, report to coordinator |
| **Quality** (FAILED) | Verifier rejects output | Yes (auto) | Same or different model, with verifier feedback |
| **Budget** (FATAL) | Budget exhausted pre-call | No | Coordinator decides (downgrade, pause, abort) |
| **Timeout** (CRASHED) | `activeDeadlineSeconds` exceeded | Yes (auto) | Retry with longer timeout or simpler instructions |

```python
class FailureType(StrEnum):
    INFRASTRUCTURE = "infrastructure"  # Retryable
    CONFIG = "config"                  # Fatal
    QUALITY = "quality"                # Retryable with feedback
    BUDGET = "budget"                  # Fatal (coordinator decides)
    TIMEOUT = "timeout"                # Retryable

def classify_failure(error: Exception) -> FailureType:
    """Classify an execution failure for retry decision."""
    if isinstance(error, asyncio.TimeoutError):
        return FailureType.TIMEOUT
    if isinstance(error, BudgetExceededError):
        return FailureType.BUDGET
    if isinstance(error, (ValueError, ConfigError)):
        return FailureType.CONFIG
    if isinstance(error, (ConnectionError, HTTPError)):
        status = getattr(error, 'status_code', None)
        if status == 429:  # Rate limited
            return FailureType.INFRASTRUCTURE
        if status and 500 <= status < 600:
            return FailureType.INFRASTRUCTURE
        return FailureType.INFRASTRUCTURE
    # Default: infrastructure (retryable)
    return FailureType.INFRASTRUCTURE
```

---

## 10. Acceptance Criteria

### Must Have

- [x] **Contractor lifecycle state machine** — 7 states with transitions, triggers, error handling, cleanup for all terminals
- [x] **`create_ephemeral_agent()` method** — signature with all params, defaults, validation, <100ms creation
- [x] **Model-per-role mapping** — 6 roles, tier assignments, OpenRouter config per role, cost estimates
- [x] **Cognitive diversity enforcement** — reviewer MUST use different model family from executor
- [x] **Tool scoping** — coordinator specifies tools explicitly, `_resolve_explicit_tools()` validates against registry
- [x] **Memory isolation** — what contractors can access (mission context, shared port) vs cannot (Mem0, session memory)
- [x] **Cleanup automation** — mission completion, TTL expiry, GC sweep — all three triggers
- [x] **AgentFactory integration** — `create_ephemeral_agent()`, `cleanup_ephemeral_agents()`, `_resolve_explicit_tools()`
- [x] **Contractor config schema** — JSONB schema matching `orchestration_tasks.contractor_config`
- [x] **Failure classification** — 5 failure types with retry/fatal determination
- [x] **Concurrency control** — per-mission and per-workspace limits with backpressure
- [x] **No sub-contractors** — hard constraint documented with rationale
- [x] **Cost estimation** — pre-execution cost projection per role using model pricing

### Should Have

- [ ] Board integration — how contractor tasks appear on kanban
- [ ] Cascade pattern design (BudgetMLAgent) — escalate model tier on quality failure

### Nice to Have

- [ ] Contractor performance profiling — latency breakdown (spawn, execute, cleanup)
- [ ] Pre-warmed LLM connection pool per model

---

## 11. Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | **Spawn overhead too high** | High | Medium | Hybrid: in-memory first, async DB. Pre-warm LLM connections. |
| 2 | **Model routing accuracy** — wrong model degrades quality | Medium | Medium | Static mapping (conservative). PRD-106 telemetry detects model-quality correlation. |
| 3 | **Contractor quality** — no personality or memory | High | Medium | Rich system prompts from ContextService (mission context, role instructions, success criteria). Quality comes from prompt, not persistence. |
| 4 | **Cleanup failures** — resource leaks | Medium | Medium | Defense in depth: mission completion + TTL + GC sweep. |
| 5 | **Tool scoping edge cases** | Medium | Low | Validate tool names at spawn time. Fail fast. |
| 6 | **Unbounded parallelism** — overwhelm LLM rate limits | High | High | Hard caps: per-mission (3), per-workspace (5). Queue excess. |
| 7 | **Cost blowout** — parallel expensive models | High | Medium | PRD-105 budget gate runs pre-check before each spawn. |
| 8 | **Model deprecation mid-mission** | Low | Low | 3-model fallback chain per role. OpenRouter handles within-model fallback. |

---

## 12. Dependencies

| Dependency | Direction | Notes |
|---|---|---|
| PRD-101 (Mission Schema) | Uses | `contractor_config` JSONB, `mission_id` FK on agents table |
| PRD-102 (Coordinator) | Blocked by | Coordinator decides when/what contractors to spawn |
| PRD-103 (Verification) | Informs | Verification cost affects model selection for reviewer role |
| PRD-105 (Budget) | Uses | Budget gate wraps contractor creation |
| PRD-106 (Telemetry) | Feeds | Per-contractor metrics: model, tokens, cost, duration, score |
| PRD-107 (Context Interface) | Informs | Context interface determines how contractors receive mission context |
| `AgentFactory` | Extension | New methods: `create_ephemeral_agent()`, `cleanup_ephemeral_agents()` |
| `tool_router.py` | Extension | New `_resolve_explicit_tools()` path for contractor tool resolution |

---

## Appendix: Research Sources

| Source | What It Informed |
|--------|-----------------|
| Agent Zero (frdel/agent-zero) | Conversation sealing, utility model, shared memory limitations |
| AutoGen (microsoft/autogen) | Agent-as-config-dict, Swarm handoff priority, context_variables |
| Kubernetes Jobs (kubernetes.io) | TTL cleanup, hard timeout, backoff limit, pod failure policy |
| RouteLLM (ICLR 2025, arxiv:2406.18665) | 75% cost reduction with static routing, role→tier mapping |
| BudgetMLAgent (AIMLSystems 2024) | Cascade pattern, 96% cost reduction |
| OpenRouter (openrouter.ai) | Provider routing params, 340 model catalog, Auto Router |
| Automatos AgentFactory (agent_factory.py) | execute_with_prompt() accepts AgentRuntime, tool loop pattern |
| Automatos heartbeat_service.py | _agent_tick() pattern, max_concurrent_per_workspace |
| Automatos config.py | PREMIUM_MODELS, BUDGET_MODELS, OpenRouter config |
