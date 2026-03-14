# PRD-104 Outline: Ephemeral Agents & Model Selection

**Type:** Research + Design
**Status:** Outline (Loop 0)
**Depends On:** PRD-100 (Research Master), PRD-101 (Mission Schema — `contractor_config` JSONB), PRD-102 (Coordinator Architecture — agent assignment in lifecycle)
**Blocks:** PRD-82C (Parallel Execution + Budget + Contractors)

---

## Section 1: Problem Statement

### Why This PRD Exists

Automatos agents are **permanent residents**. Every agent lives as a row in the `agents` table (45+ columns), has a heartbeat config, skills, tool assignments, persona, voice profile, and semantic embeddings. Creating an agent means a DB write, tool resolution queries, optional LLM verification, and caching in `AgentFactory.active_agents`. Deleting one requires cascading through 11 dependent tables.

This is correct for **roster agents** — permanent team members with personality, memory, and ongoing responsibilities.

It is **completely wrong for mission work.** When a coordinator decomposes "Research EU AI Act compliance" into 4 subtasks, it needs to spawn 4 focused agents in <100ms each, execute them (possibly in parallel), collect results, and destroy them. No persona. No heartbeat. No marketplace category. No voice profile.

### The Gap

| What Exists | What's Missing |
|---|---|
| `agents` table — 45+ columns, all loaded on every instantiation | Lightweight agent config: just `(role, model, tools, prompt)` |
| `AgentFactory.create_agent()` — DB write + LLM verification (~500ms) | Fast in-memory agent creation (<100ms) |
| `agent_type` column — supports custom strings but no lifecycle semantics | Mission-scoped lifecycle: spawn → execute → report → destroy |
| `model_config` JSON — per-agent, manually set | Per-role model selection: coordinator picks model based on task type |
| Tool resolution — DB-backed via `get_tools_for_agent()` | Dynamic tool scoping: coordinator assigns tools per mission task |
| No cleanup automation — agents are hard-deleted manually | Auto-cleanup on mission completion or TTL expiry |
| No concurrent agent limit per mission | Bounded parallelism: max N agents executing simultaneously |

### What This PRD Delivers

1. **Contractor Agent Lifecycle** — spawn, configure, execute, report, destroy — with <100ms creation
2. **Model-Per-Role Strategy** — which models for which agent roles, with cost/quality tradeoffs
3. **Dynamic Tool Scoping** — coordinator specifies tools per contractor, no DB assignment needed
4. **Mission-Scoped Memory** — contractors share mission context but nothing persists after
5. **Auto-Cleanup** — TTL-based and mission-completion-based destruction
6. **Integration Design** — how contractors flow through existing `AgentFactory.execute_with_prompt()`

---

## Section 2: Prior Art Research Targets

### Systems to Study (each gets dedicated research in the full PRD)

| System | What to Study | Why It Matters |
|---|---|---|
| **Agent Zero** (frdel/agent-zero) | `call_subordinate` delegation, conversation sealing, utility model, memory sharing | Only hierarchical agent system with progressive context compression. Sealing pattern directly applicable. |
| **AutoGen** (microsoft/autogen) | `ConversableAgent` constructor, `GroupChatManager` speaker selection, Swarm handoff priority, `context_variables` shared state | Most mature spawning API. Agent-as-config-dict pattern is exactly what we need. Swarm handoffs inform how coordinator transfers control. |
| **Kubernetes Jobs** | Job lifecycle (create → active → complete → cleanup), `ttlSecondsAfterFinished`, `activeDeadlineSeconds`, `backoffLimit`, `podFailurePolicy` | Infrastructure pattern for ephemeral workloads. TTL cleanup, timeout enforcement, retry policies — all map to contractor lifecycle. Future Phase 3 will containerize agents. |
| **Martian Model Router** | Model Mapping interpretability technique, expert orchestration architecture, automatic model indexing | Most advanced model routing. Uses interpretability (weight matrix analysis) to predict model performance without running it. Validates per-prompt routing is feasible. |
| **Unify.ai** | Neural network router, per-prompt model selection, custom router training on user data, 10-minute benchmark refresh | Proves custom-trained routers outperform default. Their "train on your own eval data" approach maps to our telemetry → model selection loop (PRD-106). |
| **OpenRouter** | Provider routing algorithm (inverse-square price weighting), Auto Router (Not Diamond), `sort`/`max_price`/`preferred_min_throughput` params, capability-based routing, Exacto endpoints | We already use OpenRouter with 340 models. Their provider config params are our v1 model selection interface. Auto Router is free fallback for uncertain cases. |
| **RouteLLM** (ICLR 2025) | Matrix factorization router, 75% cost reduction at 95% quality, MT-Bench/MMLU/GSM8K benchmarks | Academic validation that model routing works. Identifies which task types need expensive models (math/reasoning) vs cheap ones (conversational). |
| **BudgetMLAgent** (AIMLSystems 2024) | Cascade pattern (free → cheap → expensive), 96% cost reduction vs GPT-4 single-agent | Proves cascade/escalation pattern for multi-agent budgets. Relevant to PRD-105 budget enforcement. |

### Key Research from Loop 0

From the research agents' findings:

**Agent Zero:**
- Spawns subordinates via `Agent(number+1, fresh_config, SHARED_context)` — only customization is `profile` (prompt directory)
- Single subordinate at a time (linked list, not fan-out) — we need parallel fan-out
- Memory is SHARED (same FAISS index) — we need mission-scoped isolation
- Conversation sealing via `history.new_topic()` — progressive compression (50% current / 30% topics / 20% bulks). **Adopt this.**
- Utility model for compression — cheap model handles internal coordination. **Adopt this.**
- No timeouts, no budgets, no explicit destruction — we must have all three

**AutoGen:**
- Agent = `(name, system_message, llm_config, tools, description)` — fully described by a config dict. **This is our contractor config model.**
- `GroupChatManager` selects speaker via LLM or deterministic rules — our coordinator uses deterministic assignment
- Swarm handoff priority: tool-returned agent → OnCondition → AFTER_WORK fallback. **Adopt priority ordering for coordinator task transitions.**
- `context_variables` dict is shared mutable state across agents — maps to our mission-scoped context
- No explicit cleanup — agents are Python objects, GC'd. We need TTL enforcement.

**Kubernetes Jobs:**
- `ttlSecondsAfterFinished` = auto-cleanup. **Adopt for contractor TTL.**
- `activeDeadlineSeconds` = hard timeout, takes precedence over retries. **Adopt for contractor timeout.**
- `backoffLimit` = retry cap with exponential backoff. **Adopt for contractor retry policy.**
- `podFailurePolicy` rules: FailJob (fatal error codes), Ignore (infra disruption), Count (normal retry). **Adopt for contractor failure classification.**
- Artifact preservation: write results to external storage BEFORE exit. **Our pattern: contractor writes to mission_tasks.result before cleanup.**

**Model Routing:**
- Static role-based mapping captures 80% of value — no ML needed for v1
- OpenRouter's `sort`, `max_price`, `preferred_min_throughput` params are the v1 interface
- Cognitive diversity: reviewer MUST use different model family than coder (different failure modes)
- RouteLLM: 75% cost reduction at 95% quality on MT-Bench. Math/reasoning needs expensive models; conversational routes cheap.
- BudgetMLAgent: cascade pattern (free → cheap → expensive) achieves 96% cost reduction

---

## Section 3: Contractor Agent Lifecycle

### Lifecycle States

```
SPAWNING → READY → EXECUTING → REPORTING → CLEANUP → DESTROYED
    │                  │            │
    └── FAILED ←───────┴────────────┘
```

### Research Questions for Full PRD

1. **Spawn latency budget:** What's the acceptable time from coordinator decision to contractor executing? Target: <100ms for in-memory, <500ms with DB audit row.
2. **What persists after destruction?** Mission context says: `mission_tasks.result` (output), `mission_events` (execution trace), `mission_tasks.cost`/`tokens` (telemetry). Agent row itself is deleted or marked expired.
3. **Parallel fan-out limit:** How many contractors can execute simultaneously per mission? Per workspace? Must integrate with heartbeat_service's existing `max_concurrent_per_workspace = 5`.
4. **Failure classification:** Which errors are retryable (LLM timeout, rate limit) vs fatal (invalid config, auth failure, budget exhausted)? Adopt K8s podFailurePolicy pattern.
5. **Memory during execution:** Contractor gets mission context (prior task results, shared findings) but does NOT write to Mem0 long-term memory. Mission-scoped Redis or in-memory only.

### Key Design Decision: DB Row or In-Memory Only?

| Approach | Pros | Cons |
|---|---|---|
| **In-memory only** | <50ms spawn, no migration, no cleanup query | No audit trail, lost on crash, invisible to admin UI |
| **DB row with `is_ephemeral=True`** | Audit trail, visible in admin, survives crash, queryable | ~200-500ms spawn overhead, migration needed, cleanup job needed |
| **Hybrid (in-memory + async DB write)** | Fast spawn + eventual audit trail | Complexity, possible inconsistency if crash before write |

**Recommendation from research:** Hybrid. Create AgentRuntime in-memory for immediate execution. Async-write a minimal DB row (just `id, name, agent_type='contractor', workspace_id, mission_id, model_config, is_ephemeral=True, expires_at`). If crash occurs before async write, mission_events table captures the execution trace anyway.

### Integration Points

| Component | How Contractors Use It |
|---|---|
| `AgentFactory.execute_with_prompt()` | Primary execution path — contractors use same tool loop, retry logic, response synthesis |
| `get_tools_for_agent()` | Needs new `explicit_tools` param — contractor tools specified by coordinator, not DB lookup |
| `ContextService.build_context()` | New `ContextMode.MISSION_EXECUTION` mode — mission-specific system prompt + task context |
| `unified_executor.execute_tool()` | No change — same dispatch by prefix (`platform_*`, `workspace_*`, `composio_execute`) |
| `inter_agent.py` | Optional — contractors could use Redis pub/sub for real-time coordination within a mission |

---

## Section 4: Model-Per-Role Strategy

### Research Targets for Full PRD

1. **Role taxonomy:** Define the standard roles a coordinator can assign: `planner`, `researcher`, `coder`, `reviewer`, `writer`, `analyst`, `simple` (formatting/routing)
2. **Model tier mapping:** For each role, what model tier is appropriate? Based on RouteLLM and BudgetMLAgent findings:

| Role | Model Tier | Rationale | Example Models |
|---|---|---|---|
| `planner` | Mid-tier | Good reasoning, runs once per mission | Sonnet 4.6, GPT-4o |
| `researcher` | Mid-tier + large context | Process lots of text, synthesize | Gemini 3 Pro, Sonnet 4.6 |
| `coder` | Top-tier or specialized | Code quality is critical | Opus 4.6, GPT-4 |
| `reviewer` | Different family from coder | Cognitive diversity catches different bugs | GPT-5.1 (if coder=Claude), DeepSeek |
| `writer` | Mid-tier | Good prose, high volume | Sonnet 4.6 |
| `analyst` | Mid-tier + structured output | Data analysis, table generation | GPT-4o, Gemini 3 Pro |
| `simple` | Cheap | Classification, formatting, routing | Haiku 4.5, GPT-4o-mini |

3. **OpenRouter integration:** Use `provider` object params per role:
   - `sort`: `'price'` for simple, `'throughput'` for coder, `'latency'` for reviewer
   - `max_price`: Cost ceiling per role tier
   - `preferred_min_throughput`: Min tokens/sec for interactive roles
4. **Cascade/escalation:** Start cheap, escalate on low confidence. BudgetMLAgent pattern: free → $0.50/M → $15/M → $60/M
5. **Cognitive diversity enforcement:** Reviewer model family MUST differ from coder model family. This is a hard constraint, not a preference.

### Key Design Questions

1. **Where is the role→model mapping stored?** Options: (a) hardcoded in coordinator prompt, (b) workspace-level config table, (c) mission-level override. Recommendation: workspace config with mission override.
2. **Who decides the model?** Coordinator LLM suggests, but mapping table constrains. Coordinator can't assign Opus to a `simple` role.
3. **How does cost estimation work pre-execution?** Use OpenRouter's pricing API + average token counts per role to estimate mission cost before human approval.
4. **What about user model preferences?** PRD-100 says users can set "model preferences per role." How does this compose with workspace defaults and coordinator selection?
5. **Fallback when preferred model is unavailable?** OpenRouter handles provider fallback, but what if the model itself is deprecated? Need a fallback chain per role.

### Existing Codebase Touchpoints

- `agents.model_config` JSON — per-agent model settings. Contractors get this from role mapping, not DB.
- `config.py:LLM_MODEL` — system default. Contractors should NOT use this — they use role-specific models.
- `config.py:OPENROUTER_BASE_URL` — all contractor LLM calls route through OpenRouter.
- `AgentFactory._create_llm_manager()` — creates LLM client from config. Must support contractor model configs without DB lookup.

---

## Section 5: Key Design Questions

### Contractor Lifecycle

1. **DB record for contractors or ephemeral only?** Hybrid: in-memory execution + async DB audit row. See Section 3 analysis.
2. **Memory scope: mission-only or none?** Mission-only. Contractor sees prior task results from same mission (injected by coordinator). Does NOT read/write Mem0. Does NOT accumulate short-term memory across tasks (single-shot).
3. **Tool assignment: inherit from mission or custom per task?** Custom per task. Coordinator specifies `tools: ["platform_search_web", "workspace_read_file"]` in `contractor_config` JSONB (defined in PRD-101's `mission_tasks`).
4. **Spawn latency budget:** <100ms for in-memory creation. Actual LLM call (first response) adds 1-5s. Total spawn-to-first-output: <6s.
5. **What triggers cleanup?** Three triggers: (a) mission marked complete/failed, (b) TTL expiry (`expires_at`), (c) explicit coordinator `CLEANUP` command. Cleanup = evict from `active_agents` + soft-delete DB row if exists.

### Model Selection

6. **Static mapping vs dynamic routing for v1?** Static role-based mapping. No per-prompt ML routing in v1. OpenRouter's Auto Router as optional fallback.
7. **Cost tracking granularity:** Per-contractor-per-task. Each `execute_with_prompt()` already returns `tokens_used` and model info. Aggregate at mission level.
8. **User override surface:** Mission creation UI shows recommended models per role. User can override any role's model. Override stored in `mission_runs.config` JSONB.

### Integration

9. **How do contractors appear on the board?** Each contractor task creates a `board_task` with `source_type='mission'` and the mission's project label. Contractor agent_id is set on `board_tasks.assigned_agent_id` (requires minimal DB row).
10. **How does the coordinator communicate with contractors?** NOT via inter_agent Redis pub/sub. Coordinator calls `execute_with_prompt()` directly and awaits result. Simpler, debuggable, matches heartbeat tick pattern.
11. **Can a contractor spawn sub-contractors?** No. Contractors are leaf nodes. Only the coordinator decomposes and assigns. This prevents unbounded recursion and keeps the system observable.

---

## Section 6: Integration with AgentFactory

### Current Flow (Roster Agents)

```
create_agent() → DB write → LLM verify → cache in active_agents → execute_with_prompt()
                  ~500ms      ~2s            ~1ms                    ~3-30s
```

### Proposed Flow (Contractor Agents)

```
create_ephemeral_agent() → in-memory AgentRuntime → execute_with_prompt() → cleanup()
                            ~50ms                     ~3-30s                 ~10ms
         ↓ (async, non-blocking)
    DB audit row write (~200ms)
```

### Required Changes to AgentFactory

| Change | Scope | Risk |
|---|---|---|
| New `create_ephemeral_agent()` method | ~50 lines, new method | Low — doesn't touch existing paths |
| `get_tools_for_agent()` accepts `explicit_tools` param | ~10 lines, new code path | Low — existing path unchanged when param absent |
| `execute_with_prompt()` accepts `AgentRuntime` directly (already does!) | 0 lines — line 711 already checks `isinstance(agent, AgentRuntime)` | None |
| New `cleanup_ephemeral_agents(mission_id)` method | ~30 lines, new method | Low — only affects ephemeral agents |
| New `ContextMode.MISSION_EXECUTION` | ~20 lines in ContextService | Low — additive |

### What Does NOT Change

- `execute_with_prompt()` tool loop (lines 838-862) — same 10-iteration tool loop
- `_execute_tool_calls()` (lines 958-1028) — same tool dispatch
- `unified_executor.execute_tool()` — same prefix-based routing
- Heartbeat tick pattern — roster agents continue unchanged
- Agent API endpoints — contractors created by coordinator, not user API

---

## Section 7: Acceptance Criteria for Full PRD

### PRD-104 is done when:

1. **Contractor lifecycle is fully specified** — state machine with transitions, triggers, error handling, and cleanup for all terminal states
2. **Creation API defined** — `create_ephemeral_agent()` method signature with all params, defaults, and validation rules
3. **Model-per-role mapping defined** — role taxonomy, model tier assignments, OpenRouter config per role, cost estimates
4. **Tool scoping designed** — how coordinator specifies contractor tools, how `get_tools_for_agent()` handles explicit tool lists
5. **Memory isolation designed** — what mission context a contractor receives, what it can read/write, what persists after destruction
6. **Cleanup automation designed** — TTL-based, mission-completion-based, and explicit cleanup triggers with failure modes
7. **AgentFactory integration specified** — exact methods to add/modify, with pseudocode and interface signatures
8. **Board integration specified** — how contractor tasks appear on kanban, what `board_tasks` columns are set
9. **Cost estimation model** — how to predict mission cost before execution based on role→model mapping and historical averages
10. **Failure handling specified** — retryable vs fatal errors, escalation to coordinator, budget-exhaustion behavior
11. **Concurrency control specified** — max parallel contractors per mission, per workspace, backpressure strategy
12. **Data model defined** — `contractor_config` JSONB schema (referenced in PRD-101's `mission_tasks`), any new columns on `agents` table
13. **Prior art cited** — every design decision references specific systems/papers studied, with tradeoff analysis

---

## Section 8: Risks & Dependencies

### Risks

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | **Spawn overhead too high** — DB write + LLM init adds unacceptable latency for parallel fan-out | High | Medium | Hybrid approach: in-memory first, async DB. Pre-warm LLM connections. Pool LLM managers per model. |
| 2 | **Model routing accuracy** — wrong model for role degrades output quality without clear signal | Medium | Medium | Start with static mapping (conservative). Add telemetry to PRD-106 to detect model-quality correlation. Iterate. |
| 3 | **Contractor agent quality** — without personality, memory, or training, contractors may produce worse output than roster agents | High | Medium | Contractors get rich system prompts from ContextService (mission context, role instructions, success criteria). Quality comes from prompt, not persistence. |
| 4 | **Cleanup failures** — missed cleanup leads to resource leaks (memory, DB rows, Redis keys) | Medium | Medium | Defense in depth: TTL expiry (hard cap), mission-completion cleanup, periodic GC sweep. Belt and suspenders. |
| 5 | **Tool scoping complexity** — dynamic tool lists create edge cases in tool execution (missing tools, wrong params) | Medium | Low | Validate tool names against ToolRegistry at spawn time. Fail fast with clear error. |
| 6 | **Unbounded parallelism** — too many contractors overwhelm LLM rate limits or DB connections | High | High | Hard caps: per-mission (configurable, default 5), per-workspace (matches heartbeat limit). Queue excess tasks. |
| 7 | **Cost blowout** — parallel contractors with expensive models exceed budget before coordinator can react | High | Medium | PRD-105 budget enforcement runs pre-check before each contractor spawn. Coordinator holds budget lock. |
| 8 | **Model deprecation** — OpenRouter model removed mid-mission | Low | Low | Fallback chain per role (3 models deep). OpenRouter's provider routing handles within-model fallback. |

### Dependencies

| Dependency | Direction | What's Needed |
|---|---|---|
| **PRD-101** (Mission Schema) | Blocks 104 | `mission_tasks.contractor_config` JSONB schema must be defined. Contractor needs `mission_id` FK. |
| **PRD-102** (Coordinator) | Blocks 104 | Coordinator decides WHEN and WHAT contractors to spawn. Contractor lifecycle is called BY coordinator. |
| **PRD-103** (Verification) | Informs 104 | Verification cost budget (10-30% of task cost) affects model selection for `reviewer` role. |
| **PRD-105** (Budget) | Uses 104 | Budget enforcement wraps contractor creation — checks remaining budget before spawn. |
| **PRD-106** (Telemetry) | Uses 104 | Telemetry captures per-contractor metrics: model, tokens, cost, duration, verifier_score. |
| **PRD-107** (Context Interface) | Informs 104 | Context interface determines how contractors receive/share mission context. Phase 2 = direct injection. Phase 3 = field read/write. |

### Cross-PRD Discoveries

- `AgentFactory.execute_with_prompt()` already accepts `AgentRuntime` directly (line 711) — no modification needed for the execution path
- `heartbeat_service._agent_tick()` pattern is the structural template for contractor execution — same flow: load context → build prompt → call factory → collect result
- `inter_agent.py`'s `AgentCommunicationProtocol` and `CollaborativeReasoner` exist but are NOT wired into live code — future consideration for contractor-to-contractor coordination within a mission
- `SharedContextManager` has 2h Redis TTL — insufficient for multi-day missions. Contractors on short missions are fine, but PRD-107 must address this for the context interface.
- Existing `model_usage_stats` JSON on agents table tracks lifetime metrics — contractors need per-execution metrics instead (already in `execute_with_prompt()` return value)

---

## Appendix: Research Sources

### Systems Studied
- **Agent Zero** — github.com/frdel/agent-zero — delegation model, conversation sealing, utility model
- **AutoGen 0.2 & 0.4+** — github.com/microsoft/autogen — ConversableAgent, GroupChatManager, Swarm handoffs
- **Kubernetes Jobs** — kubernetes.io/docs/concepts/workloads/controllers/job/ — lifecycle, TTL, backoff, pod failure policy
- **Martian** — withmartian.com — Model Mapping, expert orchestration AI architecture
- **Unify.ai** — unify.ai — neural network router, custom router training, 10-min benchmark refresh
- **OpenRouter** — openrouter.ai — provider routing, Auto Router (Not Diamond), capability-based routing, Exacto endpoints

### Papers & Benchmarks
- **RouteLLM** (ICLR 2025, UC Berkeley/Anyscale/Canva) — matrix factorization router, 75% cost reduction, arxiv.org/abs/2406.18665
- **RouterArena** (2025) — benchmark of 12 routers, arxiv.org/html/2510.00202v1
- **BudgetMLAgent** (AIMLSystems 2024) — cascade pattern, 96% cost reduction, dl.acm.org/doi/10.1145/3703412.3703416
- **Claude Haiku 4.5 announcement** — Anthropic — "Sonnet orchestrates Haiku team" pattern

### Automatos Codebase Files
- `orchestrator/modules/agents/factory/agent_factory.py` — AgentFactory (1,425 lines), execute_with_prompt(), tool loop
- `orchestrator/services/heartbeat_service.py` — _agent_tick() pattern, rate limiting, concurrent execution
- `orchestrator/core/models/core.py:172-280` — Agent table schema (45+ columns)
- `orchestrator/modules/tools/tool_router.py:129-250` — get_tools_for_agent(), tool resolution
- `orchestrator/modules/agents/communication/inter_agent.py` — Redis pub/sub, shared context, consensus
- `orchestrator/config.py:144-226` — OpenRouter config, LLM model defaults
- `orchestrator/api/agents.py:404-483` — Agent creation API endpoint
