# PRD-102 Outline: Coordinator Architecture

**Type:** Research + Design
**Status:** Outline (Loop 0)
**Depends On:** PRD-100 (Research Master), PRD-101 (Mission Schema)
**Blocks:** PRD-103 (Verification), PRD-104 (Ephemeral Agents), PRD-107 (Context Interface)

---

## Section 1: Problem Statement

### Why This PRD Exists

Automatos has **no coordination layer**. The closest existing component is `heartbeat_service.py:_orchestrator_tick_llm()` (line 382), which runs a 5-iteration tool loop with an 8,000-token budget and `dispatcher_only` tools — it does health checks and reporting, not goal decomposition or agent dispatch.

### The Coordination Gap

| What Exists | What's Missing |
|---|---|
| `_orchestrator_tick_llm()` — LLM tool loop for workspace health checks | Goal decomposition: breaking "Research EU AI Act compliance" into subtasks |
| `AgentFactory.execute_with_prompt()` — per-agent execution with 10-iteration tool loop | Parallel dispatch: running independent subtasks concurrently via `asyncio.gather` |
| `AgentCommunicationProtocol` — Redis pub/sub messaging (built, not wired to heartbeat ticks) | Cross-task data flow: passing Task 1's output as input to Task 2 |
| `BoardTask` with `assigned_agent_id` — manual task assignment | Automatic agent selection: matching task requirements to agent capabilities |
| `SharedContextManager` — in-process shared state with Redis backing (2h TTL) | Mission state machine: tracking plan → execute → verify → review lifecycle |
| `TaskReconciler` — stall detection for `recipe_executions` only | Mission-scoped stall detection, dependency-aware retry, escalation on failure |
| `ContextMode.HEARTBEAT_ORCHESTRATOR` — 8k tokens, 5 sections, dispatcher tools | `ContextMode.COORDINATOR` — full tools, mission context section, no token cap |

### What This PRD Delivers

The architecture for a **CoordinatorService** that:
1. Takes a natural language goal + autonomy settings
2. Decomposes it into a dependency graph of 3-20 tasks (using PRD-101's `mission_tasks` schema)
3. Assigns each task to a roster agent or contractor agent
4. Dispatches tasks respecting dependency ordering
5. Monitors execution, handles failures (continuation vs retry)
6. Triggers verification (PRD-103) and human review gates
7. Detects mission completion and offers "save as routine"

---

## Section 2: Prior Art Research Targets

### Systems to Study (each gets dedicated research)

| System/Pattern | Source | Focus Areas | Key Question |
|---|---|---|---|
| **Blackboard Architecture** | Nii 1986 (AI Magazine); LbMAS (arxiv:2507.01701, 2025) | Shared state as coordination medium, knowledge source preconditions, event-driven activation, conflict resolution | Should the mission state object act as a blackboard that agents read/write to? |
| **HTN Planning** | Nau et al. JAIR 2003 (SHOP2); ChatHTN (arxiv:2505.11814, 2025); Hsiao et al. (arxiv:2511.07568, 2025) | Compound→primitive decomposition, method libraries, partial-order task networks, LLM as decomposition engine | Should we maintain decomposition templates that the LLM fills gaps in (ChatHTN hybrid)? |
| **BDI Agents** | Rao & Georgeff ICMAS 1995; ChatBDI (AAMAS 2025) | Belief-Desire-Intention cycle, intention commitment prevents thrashing, plan failure propagation, bold vs cautious reconsideration | Should the coordinator use BDI's intention model to prevent premature replanning? |
| **Symphony** | `openai/symphony` SPEC.md | WORKFLOW.md policy-as-code, reconciliation loop (dispatch + reconcile phases), continuation vs retry, workpad as progress checkpoint | Should we adopt the two-phase tick (dispatch new + reconcile running) and continuation vs retry distinction? |
| **CrewAI** | `crewAIInc/crewAI` | Sequential vs hierarchical process, `context=[task_a, task_b]` dependency declaration, guardrail validation pattern, `async_execution` + join | Should we adopt CrewAI's explicit `context=` dependency pattern for data flow between tasks? |
| **AutoGen** | `microsoft/autogen` | GroupChat turn-based coordination, Swarm handoff-based routing, termination composition (`\|` / `&`), nested execution isolation | Should agents explicitly hand off to the next agent (Swarm pattern) or should the coordinator always decide? |
| **LangGraph** | LangChain ecosystem | Typed state schema, deterministic conditional routing, checkpointing at every superstep, `interrupt()` for human review, Send API for dynamic parallelism | Should we adopt LangGraph's typed state + checkpoint-per-step model for mission durability? |
| **Automatos Codebase** | `heartbeat_service.py`, `inter_agent.py`, `context/service.py`, `agent_factory.py`, `task_reconciler.py` | What exists today that the coordinator builds on vs replaces |

### Key Patterns Discovered in Research

**Blackboard as mission state (Nii 1986, LbMAS 2025):** The mission state object (PRD-101's `mission_runs` + `mission_tasks`) acts as a blackboard — agents write results to it, the coordinator reads it to decide next actions. LbMAS (2025) showed 5% improvement over static multi-agent systems using this pattern with LLMs. Key adoption: event-driven activation (agent activates when its dependencies complete on the blackboard) over polling.

**HTN decomposition with LLM gap-filling (ChatHTN 2025):** The coordinator maintains a library of decomposition templates for known mission types. For novel goals, the LLM generates a decomposition. ChatHTN proved this hybrid is provably sound — the symbolic structure validates the LLM's output. Hsiao et al. (2025) showed hand-coded HTNs enable 20-70B models to outperform 120B baselines, confirming structure improves LLM planning.

**BDI intention commitment (Rao & Georgeff 1995):** Once the coordinator commits to a plan, it should not replan on every tick — only when a significant belief change occurs (task failure, budget exceeded, new user input). The bold/cautious spectrum from Kinny & Georgeff maps to the autonomy toggle: `approve` mode = cautious (human gates), `autonomous` mode = bolder (replan only on failure).

**Two-phase reconciliation tick (Symphony):** Every coordinator tick runs: (1) dispatch phase — find tasks whose dependencies are met and assign them; (2) reconcile phase — check running tasks for stalls, external state changes, or completion. This separation is cleaner than a single monolithic loop.

**Continuation vs retry (Symphony):** A task that completed normally but the mission isn't done → continuation (near-zero delay, resume from workspace). A task that failed → retry (exponential backoff). The `attempt_count` on `mission_tasks` tracks retries separately from continuations. Critical distinction for AI agents where "done with my part" ≠ "mission complete."

**Typed state + checkpointing (LangGraph):** The coordinator's state should be a typed schema (the `mission_runs` + `mission_tasks` tables from PRD-101) with a checkpoint after every state transition. This enables crash recovery — coordinator restarts, reads last state from DB, resumes.

**Explicit dependency declarations (CrewAI):** `task_inputs` JSONB (from PRD-101) maps to CrewAI's `context=[task_a, task_b]` — explicit, declarative, queryable. The scheduler resolves "which tasks are ready?" by checking `task_inputs` references against completed task IDs.

---

## Section 3: Coordinator Responsibilities

### 3.1 Plan Decomposition

The coordinator takes a natural language goal and produces a task graph:

```
User: "Research EU AI Act compliance for our product"
         │
         ▼
   COORDINATOR (LLM call with ContextMode.COORDINATOR)
   - Receives: goal, available agents (roster), workspace context
   - Produces: mission_tasks[] with dependency edges (task_inputs JSONB)
   - Output format: structured JSON matching mission_tasks schema
```

**Decomposition strategy (HTN-inspired hybrid):**
1. Check template library for matching mission type (exact match or semantic similarity)
2. If template found → use it, let LLM customize parameters (agent assignments, specific instructions)
3. If no template → LLM generates full decomposition from scratch
4. Validate decomposition: no cycles in dependency graph, all referenced agents exist, budget estimate within limits

**Key design question:** How much planning capability do current LLMs actually have? Research must benchmark decomposition quality across models (cheap models for simple missions, expensive models for complex ones).

### 3.2 Agent Assignment

For each task in the plan:

| Assignment Strategy | When Used |
|---|---|
| **Roster match** | Task requirements match a roster agent's skills/tools. Preferred — agent has memory, personality, history. |
| **Contractor spawn** | No roster agent matches, or task needs a specialist model. Ephemeral — mission-scoped lifecycle (PRD-104). |
| **User override** | In `approve` mode, user can reassign agents before execution starts. |

**Matching algorithm:** Compare task requirements (tools needed, model preference, domain) against agent capabilities from DB (`agents.skills`, `agent_tools`, `agents.model`). Score and rank. Deterministic, not LLM-based — CrewAI's "LLM-as-manager" approach is non-deterministic and untestable.

### 3.3 Progress Monitoring

The coordinator monitors via the two-phase tick (Symphony pattern):

**Phase A — Dispatch:**
1. Query `mission_tasks` where `status = 'pending'`
2. For each: check if all `task_inputs.__parents__` tasks are in terminal success state
3. If ready: transition to `scheduled`, create/update `BoardTask`, invoke `AgentFactory.execute_with_prompt()` (or queue for agent tick)
4. Respect concurrency limits (configurable per mission)

**Phase B — Reconcile:**
1. Query `mission_tasks` where `status = 'running'`
2. Check for stalls (elapsed > stall timeout) → handle per continuation/retry logic
3. Check for completed tasks → emit `TASK_COMPLETED` event, update mission state
4. Check if all tasks done → advance mission to `verifying` phase
5. Check budget → if approaching limit, emit `BUDGET_WARNING` event

### 3.4 Failure Handling

**Continuation vs retry (Symphony-inspired):**

| Scenario | Action | Delay |
|---|---|---|
| Agent completed normally, mission not done | Continuation — dispatch next dependent tasks | Immediate |
| Agent failed (error, timeout, tool crash) | Retry — same agent, exponential backoff | `min(10s × 2^(attempt-1), 5min)` |
| Agent failed, max retries exhausted | Escalate — try different agent or model | Immediate, different assignment |
| All alternatives exhausted | Mission failed — notify user | — |
| Budget exceeded mid-task | Pause mission — notify user for budget increase or cancellation | — |

**BDI-inspired reconsideration policy:**
- Do NOT replan on every tick (bold agent behavior for stable missions)
- Replan triggers: task failure after all retries, user sends new instructions, budget warning
- Replanning increments `mission_runs.plan_version` and emits `PLAN_REVISED` event

### 3.5 Human Review Gates

Two human interaction points:

1. **Plan approval** (`approve` mode): After decomposition, coordinator presents plan to user. User can approve, modify, or reject. Mission stays in `awaiting_approval` until human acts.
2. **Result review** (all modes): After verification (PRD-103), mission enters `awaiting_review`. User accepts, rejects (with feedback for specific tasks), or sends back for rework.

### 3.6 Mission Completion

A mission is complete when:
1. All `mission_tasks` are in terminal state (`verified` or `human_accepted`)
2. Verification (PRD-103) has run and scored all outputs
3. Human has reviewed (or autonomy mode and all verifications passed)
4. Budget accounting is finalized
5. User is offered "save as routine?" → creates `workflow_recipe` from mission structure

---

## Section 4: Key Design Questions

### Q1: LLM-Driven vs Rule-Based Planning?

**Options:**
- **Pure LLM:** Coordinator sends goal + available agents to LLM, gets back a task graph. Flexible but non-deterministic.
- **Pure rule-based:** Predefined templates for every mission type. Deterministic but brittle — can't handle novel goals.
- **Hybrid (recommended — ChatHTN pattern):** Template library for known patterns + LLM for novel goals + LLM for customizing templates. Validate all plans against structural rules (no cycles, valid agents, budget estimate).

**Research needed:** Benchmark decomposition quality. Give 10 mission goals to GPT-4o, Claude Sonnet, DeepSeek, Qwen — measure: task count, dependency correctness, instruction clarity, time to plan.

### Q2: Stateful vs Stateless Coordinator?

**Options:**
- **Stateful (in-process):** Coordinator holds mission state in memory, writes to DB periodically. Fast but lost on crash.
- **Stateless (DB-driven, recommended):** Coordinator reads state from DB on every tick, writes back after actions. Slower but crash-recoverable. Matches LangGraph's checkpoint model and Symphony's "restart recovery via tracker + filesystem."

**Recommendation:** Stateless. The `mission_runs`/`mission_tasks` tables from PRD-101 ARE the state. Coordinator reconstructs its understanding on every tick by querying them. This is why PRD-101's schema design is critical.

### Q3: How Does the Coordinator Use ContextService?

**New context mode needed:** `ContextMode.COORDINATOR`

| Section | Content |
|---|---|
| `identity` | Coordinator agent identity (role: mission coordinator) |
| `mission_context` (NEW) | Current mission: goal, plan, task statuses, agent assignments, budget status |
| `agent_roster` (NEW) | Available agents with their skills, tools, models, recent success rates |
| `platform_actions` | Full platform tools including new mission management tools |
| `task_context` | Current tick's focus: which tasks need dispatch, which are stalled |
| `datetime_context` | Current time for scheduling decisions |

**Token budget:** No cap (or 128k+ cap). Coordinator needs to see full mission context to make good decisions.

### Q4: Coordinator Prompt Design

The coordinator prompt must encode:
- **Role:** "You are a mission coordinator. Your job is to decompose goals, assign agents, and monitor execution."
- **Available actions:** Structured tool definitions for mission management
- **Current state:** Injected via `mission_context` section
- **Decision framework:** When to dispatch, when to wait, when to replan, when to escalate

**Research needed:** Test prompt designs. The `WORKFLOW.md` pattern (Symphony) of state-specific instructions is compelling — coordinator prompt could have sections for each mission state (`planning`, `executing`, `verifying`, `reviewing`).

### Q5: Replanning Triggers

When should the coordinator revise its plan?

| Trigger | Action |
|---|---|
| Task fails after max retries | Replan: remove failed task, find alternative path or substitute agent |
| User sends new instructions mid-mission | Replan: incorporate new requirements, may add/remove tasks |
| Budget warning (>80% spent) | Replan: cut remaining tasks to essentials, use cheaper models |
| Verification rejects a task output | Replan: retry with different instructions or different agent |
| Agent discovers new information | Replan: add tasks discovered during execution (dynamic task creation) |

**Key constraint:** Replanning must not discard completed work. Only pending/scheduled tasks can be modified. Running tasks continue unless explicitly cancelled.

### Q6: Where Does the Coordinator Live in the Module Hierarchy?

**Options:**
- `orchestrator/services/coordinator_service.py` — alongside `heartbeat_service.py` and `task_reconciler.py`
- `orchestrator/modules/coordination/coordinator.py` — new module

**Recommendation:** `orchestrator/services/coordinator_service.py` as the service, with supporting classes in `orchestrator/modules/coordination/` (planner, dispatcher, reconciler). The service registers its tick on the shared `UnifiedScheduler` like heartbeat does.

---

## Section 5: Integration Points

### How the Coordinator Calls Existing Components

| Existing Component | How Coordinator Uses It |
|---|---|
| `AgentFactory.execute_with_prompt()` | Dispatches each mission task to its assigned agent. Coordinator passes `context_mode=ContextMode.TASK_EXECUTION`, `prompt=task_instructions`. |
| `ContextService.build_context()` | Coordinator builds its own context with `ContextMode.COORDINATOR`. Also used when building agent context for task dispatch. |
| `get_tools_for_agent()` (tool_router.py:140) | Resolves tools for task agents. Coordinator may need its own tool set (mission management tools). |
| `UnifiedToolExecutor.execute_tool()` | Coordinator's own tool loop uses this for platform actions (create board task, update mission status). |
| `BoardTask` model (core/models/board.py) | Coordinator creates board tasks with `source_type='mission'`, `source_id=mission_run_id`. Links via `mission_tasks.board_task_id`. |
| `HeartbeatService._agent_tick()` | Agent ticks pick up board tasks assigned to them. Coordinator assigns tasks → heartbeat delivers them. Alternative: coordinator calls `execute_with_prompt` directly for immediate dispatch. |
| `TaskReconciler` | Extended to watch `mission_tasks` alongside `recipe_executions`. Coordinator handles escalation on max-retry failure. |
| `AgentCommunicationProtocol` | Coordinator broadcasts mission context updates to assigned agents via Redis pub/sub. Optional — only if agents need real-time coordination during execution. |
| `SharedContextManager` | Stores mission-scoped shared context (accumulated results from completed tasks). Agents read it to get sibling task outputs. |
| `workflow_recipes` table | "Save as routine" converts mission structure to recipe steps. |

### New Components the Coordinator Introduces

| Component | Purpose |
|---|---|
| `CoordinatorService` | Main service: tick loop, plan generation, dispatch, reconciliation |
| `MissionPlanner` | LLM-powered decomposition: goal → task graph. Template matching + LLM generation. |
| `MissionDispatcher` | Resolves ready tasks, assigns agents, calls `execute_with_prompt` or creates board tasks |
| `MissionReconciler` | Extends `TaskReconciler` pattern for mission-scoped stall detection and dependency-aware retry |
| `ContextMode.COORDINATOR` | New context mode with `mission_context` and `agent_roster` sections |
| `ContextMode.VERIFIER` | New context mode for verification agents (PRD-103) |
| `platform_create_mission` | Platform tool: user creates mission from chat |
| `platform_approve_plan` | Platform tool: user approves/modifies coordinator's plan |
| `platform_mission_status` | Platform tool: user checks mission progress |
| API endpoints | `POST /missions`, `GET /missions/{id}`, `POST /missions/{id}/approve`, `POST /missions/{id}/review` |

### Files That Must Be Modified

| File | Change |
|---|---|
| `orchestrator/modules/context/modes.py` | Add `COORDINATOR` and `VERIFIER` to `ContextMode` enum and `MODE_CONFIGS` |
| `orchestrator/modules/context/service.py` | Add `mission_context` and `agent_roster` section renderers |
| `orchestrator/services/task_reconciler.py` | Extend `_tick` to query `mission_tasks` alongside `recipe_executions` |
| `orchestrator/modules/tools/platform_actions.py` | Register mission management action definitions |
| `orchestrator/modules/tools/execution/platform_executor.py` | Add handlers for mission tools |
| `orchestrator/core/models/core.py` (or new `mission.py`) | Import mission models (defined in PRD-101) |
| `orchestrator/api/` | New `missions.py` router for mission API endpoints |
| `alembic/versions/` | Migration for any coordinator-specific columns (most schema is PRD-101) |

---

## Section 6: Acceptance Criteria for Full PRD-102

The complete PRD-102 is done when:

- [ ] **Prior art comparison table** — Blackboard, HTN, BDI, Symphony, CrewAI, AutoGen, LangGraph compared across: coordination model, state management, planning approach, failure handling, human review, scalability. With explicit "what we adopt" and "what we reject" per system.
- [ ] **CoordinatorService architecture** — Class diagram with `CoordinatorService`, `MissionPlanner`, `MissionDispatcher`, `MissionReconciler`. Method signatures for all public methods.
- [ ] **Coordinator tick algorithm** — Pseudocode for the two-phase tick: dispatch + reconcile. Include concurrency handling, error paths, and budget checks.
- [ ] **Decomposition pipeline** — How goal → task graph works: template matching, LLM generation, validation. Include prompt templates for the coordinator LLM call.
- [ ] **Agent assignment algorithm** — Pseudocode for matching task requirements to agent capabilities. Scoring function, fallback to contractor spawn.
- [ ] **Failure handling specification** — Continuation vs retry vs escalation decision tree. State machine transitions on each failure type.
- [ ] **ContextMode.COORDINATOR definition** — Sections, token budget, tool loading strategy. Include `mission_context` section renderer pseudocode.
- [ ] **Replanning specification** — Triggers, constraints (don't discard completed work), plan versioning.
- [ ] **Integration tests specification** — Key scenarios: simple 3-task sequential mission, parallel mission with join, mission with task failure and retry, mission with replanning, budget-exceeded mission.
- [ ] **API endpoint specification** — OpenAPI-style definitions for mission CRUD + approval + review endpoints.
- [ ] **Sequence diagrams** — At least 3: happy path mission, mission with failure/retry, mission with human review rejection.

---

## Section 7: Risks & Dependencies

### Risks

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | Coordinator complexity — too many responsibilities in one service | High | Split into focused classes: Planner, Dispatcher, Reconciler. Coordinator is the orchestrator, not the doer. |
| 2 | LLM planning reliability — decomposition quality varies by model and prompt | High | Template library for common patterns (ChatHTN hybrid). Validate all plans structurally before execution. Benchmark decomposition quality across models. |
| 3 | Cost of coordination calls — coordinator LLM calls add overhead per mission | Medium | Use cheap models for coordination (Haiku-class). Coordinator prompt should be concise. Template matching avoids LLM call entirely for known patterns. |
| 4 | Tick frequency tradeoff — too fast = wasted cycles, too slow = delayed dispatch | Medium | Start with 5-second tick (Symphony default). Make configurable. Consider event-driven activation for specific transitions (task completion → immediate dispatch of dependent tasks). |
| 5 | Parallel dispatch race conditions — two tasks complete simultaneously, both trigger dependent task | Medium | Use DB-level locking or `SELECT ... FOR UPDATE` when transitioning task status. Only one dispatch per tick per task. |
| 6 | Replanning destroys progress — bad replan discards valid completed work | High | Immutable completed tasks. Replanning only modifies `pending`/`scheduled` tasks. `plan_version` increments on every replan for audit trail. |
| 7 | Agent unavailability — assigned agent is offline or overloaded | Medium | Coordinator checks agent availability before dispatch. Fallback: reassign to different agent or spawn contractor. Stall detection catches unresponsive agents. |
| 8 | Circular dependencies in task graph — LLM generates impossible plan | Low | Validate DAG structure (topological sort) before accepting any plan. Reject plans with cycles. |
| 9 | Coordinator becomes single point of failure | Medium | Stateless design (DB-driven) means any instance can take over. No in-process state to lose. |
| 10 | Over-engineering the first version | High | PRD-100 Risk #3: "Start sequential-only. No parallel, no dynamic replanning. Get lifecycle right first." Phase the implementation: sequential missions first (82A/B), then parallel + replanning (82C). |

### Dependencies

| Dependency | Direction | Notes |
|---|---|---|
| PRD-101 (Mission Schema) | **Blocked by 101** | Coordinator reads/writes `mission_runs`, `mission_tasks`, `mission_events`. Cannot build coordinator without schema. |
| PRD-103 (Verification) | **Blocks 103** | Coordinator triggers verification phase. Verification PRD needs to know coordinator's handoff interface. |
| PRD-104 (Ephemeral Agents) | **Blocks 104** | Coordinator spawns contractor agents. Contractor PRD needs coordinator's spawn interface. |
| PRD-105 (Budget) | Uses 105 | Coordinator enforces budget limits defined in PRD-105. Can start with simple budget checks, enhance later. |
| PRD-106 (Telemetry) | Feeds 106 | Coordinator emits `mission_events` that telemetry queries. Event schema must support telemetry aggregation. |
| PRD-107 (Context Interface) | **Blocks 107** | Context interface must abstract how coordinator gets/sets context. Coordinator is the primary consumer. |
| Existing `HeartbeatService` | **Integration** | Coordinator registers its tick alongside heartbeat. Must not conflict with heartbeat's scheduling. |
| Existing `AgentFactory` | **Integration** | Coordinator dispatches via `execute_with_prompt()`. No changes needed to AgentFactory. |
| Existing `TaskReconciler` | **Extension** | Must extend to cover mission tasks. Could be a new `MissionReconciler` or an extension of existing class. |
| Existing `ContextService` | **Extension** | Must add `COORDINATOR` mode and `mission_context` section. Non-breaking — adds new mode, doesn't modify existing ones. |

---

## Appendix: Research Summary Matrix

| Aspect | Blackboard | HTN Planning | BDI | Symphony | CrewAI | AutoGen | LangGraph |
|---|---|---|---|---|---|---|---|
| **Coordination model** | Shared state + event-driven KS activation | Hierarchical decomposition of compound tasks into primitives | Belief-Desire-Intention deliberation cycle | Reconciliation loop (dispatch + reconcile) with policy-as-code | Sequential or hierarchical (LLM-as-manager) process | Turn-based group chat with LLM speaker selection | Typed state graph with deterministic conditional edges |
| **State management** | Blackboard data structure (shared, hierarchical) | World state updated at each primitive step | Belief base (agent's model of world) | External tracker (Linear) + workspace filesystem | In-memory crew state; Flows add SQLite persistence | In-memory message list (ephemeral) | Typed schema + pluggable checkpointers (Postgres, SQLite) |
| **Planning approach** | Opportunistic — no predetermined path | Method library for known decompositions; backtracking for alternatives | Plan library indexed by triggering events; LLM can generate plans dynamically | No planning — work comes from external tracker | LLM-as-manager in hierarchical mode; AgentPlanner pre-generates steps | No planning — conversation-driven emergence | Graph defined at compile time; conditional routing for branching |
| **Failure handling** | KS produces competing hypotheses; control resolves conflicts | Backtrack and try alternative method | Plan failure propagation with alternative plan selection; bold/cautious reconsideration | Continuation (1s) vs retry (exponential backoff); workspace preserved | Guardrail retry loop (max 3); soft failure — proceeds with bad output | No built-in failure handling | Checkpoint enables resume from last successful step |
| **Human review** | Not built-in | Not built-in | Not built-in (agent is autonomous) | PR review is the human gate; no mid-execution review | `human_input=True` per task; `@human_feedback` in Flows | `human_input_mode` on UserProxyAgent | `interrupt()` pauses execution; resume with human input |
| **What we adopt** | Mission state as blackboard; event-driven task activation; explicit conflict resolution | Template library + LLM gap-filling (ChatHTN); partial-order task networks for parallelism | Intention commitment (don't replan every tick); bold/cautious spectrum maps to autonomy toggle; plan failure propagation | Two-phase tick (dispatch + reconcile); continuation vs retry; WORKFLOW.md state-specific instructions | `context=[]` dependency declarations; guardrail validation pattern; `async_execution` + join | Swarm handoff pattern; termination condition composition | Typed state schema; checkpoint per step; `interrupt()` for human review; Send API for dynamic parallelism |
| **What we reject** | BB1 control blackboard (overkill for 3-20 tasks); distributed blackboard partitioning | Full formal HTN domain model (too rigid); hand-authored methods only | Static plan library (LLM replaces); symbolic brittleness (LLM handles fuzzy preconditions) | Linear-specific coupling; single-agent-per-task; no multi-agent coordination | LLM-as-manager for delegation (non-deterministic); soft guardrail failure; no dynamic task creation | LLM-based speaker selection per turn (expensive, non-deterministic); magic-string termination; ephemeral state | Full boilerplate burden; static graph compilation; LangSmith lock-in |
