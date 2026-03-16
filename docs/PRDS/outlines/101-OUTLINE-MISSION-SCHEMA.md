# PRD-101 Outline: Mission Schema & Data Model

**Type:** Research + Design
**Status:** Outline (Loop 0)
**Depends On:** PRD-100 (Research Master)
**Blocks:** PRD-102, 103, 104, 105, 106, 107, 108

---

## Section 1: Problem Statement

### Why This PRD Exists

Automatos has a complete execution infrastructure (340 LLMs, 850 tools, 11 channels, 5-layer memory, board, reports, recipes) but **no orchestration persistence layer**. There are no `orchestration_runs` or `orchestration_tasks` tables. No task graph with dependencies. No structured execution trace.

### The Gap

| What Exists | What's Missing |
|---|---|
| `board_tasks` — flat kanban with `parent_task_id` | DAG-structured task dependencies with trigger rules |
| `recipe_executions` — single-recipe tracking | Multi-step, multi-agent mission runs with shared context |
| `heartbeat_results` — per-tick outcome logging | Per-mission execution trace with cost/token/time attribution |
| `task_reconciler` — stall detection for recipes | Mission-aware reconciliation with continuation vs retry distinction |
| `board_tasks.planning_data` JSONB — reserved stub | Actual planning data: decomposition, dependency edges, agent assignments |

### What This PRD Delivers

The foundational data model that every other Phase 2 PRD builds on:
- `mission_runs` table — the top-level orchestration unit
- `mission_tasks` table — individual work units within a mission, with dependency edges
- `mission_events` table — append-only event log for state reconstruction and audit
- Status enums and state machine definitions
- Integration strategy with existing `board_tasks`, `agents`, `workspaces`, `workflow_recipes`

---

## Section 2: Prior Art Research Targets

### Systems to Study (each gets dedicated research)

| System | Repo/Source | Focus Areas | Key Question |
|---|---|---|---|
| **Temporal** | `temporalio/temporal` | Event sourcing model, MutableState as cache, activity_map, workflow branching, retry without history events | Should we use event sourcing with mutable state as a derived cache? |
| **Prefect** | `PrefectHQ/prefect` | `task_inputs` JSON as dependency graph, named sub-states, server-enforced state transitions via composable rule chains, `ResultRecordMetadata` for deferred results | Should we store dependency edges as JSON on task rows (like `task_inputs`) or in a separate edges table? |
| **Airflow** | `apache/airflow` | Trigger rules system (all_success, one_failed, etc.), XCom for data passing, DAG run vs task instance separation | Which trigger rules do we need for AI agent orchestration? |
| **Dagster** | `dagster-io/dagster` | Append-only event log as ground truth, IO manager pattern for result storage, content-addressed snapshots, tags as extension point, step state reconstructed from events | Should step/task state be a column or derived from event log? |
| **Symphony** | `openai/symphony` | WORKFLOW.md policy-as-code, continuation vs retry distinction, reconciliation-before-dispatch, tracker-driven (no orchestrator DB), workspace isolation | Should we adopt continuation vs retry as distinct code paths? |

### Key Patterns Discovered in Research

**Event sourcing (Temporal, Dagster):** Both use an append-only event log as ground truth, with denormalized status columns as derived caches. Temporal's `MutableState` and Dagster's `runs.status` are both reconstructable from their event logs. This provides full audit trail and replay capability.

**Dependencies as data, not schema (Prefect):** Prefect stores dependency edges as `task_inputs: dict[str, list[RunInput]]` JSON on each task run row — no separate edges table. Graph is reconstructed at query time. Trade-off: simpler writes, heavier reads for large DAGs.

**Continuation vs retry (Symphony):** Normal worker exit → 1s re-check (continuation). Failure → exponential backoff retry. The `attempt` counter differs. This distinction matters for AI agents: an agent that completed its turn cleanly but the mission isn't done yet is fundamentally different from an agent that crashed.

**Named sub-states (Prefect):** `(type=SCHEDULED, name="AwaitingConcurrencySlot")` avoids enum proliferation while providing rich semantic state. Directly applicable to mission tasks needing states like `(RUNNING, "CallingTool")` or `(SCHEDULED, "AwaitingHumanApproval")`.

**Tags as extension point (Dagster):** `run_tags` table stores schedule names, sensor names, parent run IDs, retry indicators as key-value pairs rather than dedicated columns. Keeps core schema stable while enabling rich metadata.

---

## Section 3: Data Model Requirements

### Entities Needed

#### 1. `mission_runs` — Top-Level Orchestration Unit

| Field | Type | Purpose |
|---|---|---|
| `id` | UUID PK | |
| `workspace_id` | UUID FK→workspaces | Tenant isolation |
| `title` | VARCHAR(500) | User-facing mission name |
| `goal` | TEXT | Natural language goal from user |
| `status` | VARCHAR(30) | State machine status (see below) |
| `status_name` | VARCHAR(50) | Named sub-state for rich semantics |
| `autonomy_mode` | VARCHAR(20) | `approve` (default) or `autonomous` |
| `created_by_user_id` | INT FK→users | Who initiated |
| `plan` | JSONB | Coordinator's decomposition plan |
| `plan_version` | INT | Incremented on replan |
| `budget_config` | JSONB | `{ max_cost_usd, max_tokens, max_duration_s, max_tool_calls }` |
| `budget_spent` | JSONB | `{ cost_usd, tokens_in, tokens_out, tool_calls, duration_s }` |
| `recipe_id` | INT FK→workflow_recipes NULL | If created from a recipe template |
| `saved_as_recipe_id` | INT FK→workflow_recipes NULL | If mission was saved as routine |
| `metadata` | JSONB | Extensible — model preferences, tags, labels |
| `started_at` | TIMESTAMPTZ | |
| `completed_at` | TIMESTAMPTZ | |
| `created_at` | TIMESTAMPTZ | |
| `updated_at` | TIMESTAMPTZ | |

**Status enum:** `planning` → `awaiting_approval` → `executing` → `verifying` → `awaiting_review` → `completed` | `failed` | `cancelled`

#### 2. `mission_tasks` — Individual Work Units

| Field | Type | Purpose |
|---|---|---|
| `id` | UUID PK | |
| `mission_run_id` | UUID FK→mission_runs CASCADE | Parent mission |
| `workspace_id` | UUID FK→workspaces | Denormalized for query perf |
| `title` | VARCHAR(500) | Task description |
| `instructions` | TEXT | Detailed instructions for agent |
| `status` | VARCHAR(30) | Task-level state machine |
| `status_name` | VARCHAR(50) | Named sub-state |
| `sequence_number` | INT | Ordering within mission |
| `assigned_agent_id` | INT FK→agents NULL | Roster agent |
| `contractor_config` | JSONB NULL | If ephemeral agent: `{ role, model, tools, system_prompt }` |
| `board_task_id` | INT FK→board_tasks NULL | Link to visible board task |
| `task_inputs` | JSONB | Dependency edges: `{ "param_name": [{"type": "task_output", "task_id": "..."}], "__parents__": [...] }` |
| `success_criteria` | JSONB | What "done" means: `{ criteria: [...], verification_model }` |
| `result` | TEXT | Agent output |
| `result_metadata` | JSONB | `{ storage_key, format, size_bytes }` for large results |
| `error_message` | TEXT | |
| `attempt_count` | INT DEFAULT 1 | Current attempt |
| `max_attempts` | INT DEFAULT 3 | Max retries |
| `tokens_in` | INT | |
| `tokens_out` | INT | |
| `cost_usd` | FLOAT | |
| `model_used` | VARCHAR(100) | Actual model that executed |
| `tools_used` | JSONB | `["tool_name_1", "tool_name_2"]` |
| `verifier_score` | FLOAT NULL | 0.0-1.0 from verification |
| `verifier_feedback` | TEXT NULL | |
| `human_accepted` | BOOL NULL | NULL=not reviewed, true/false=reviewed |
| `started_at` | TIMESTAMPTZ | |
| `completed_at` | TIMESTAMPTZ | |
| `created_at` | TIMESTAMPTZ | |
| `updated_at` | TIMESTAMPTZ | |

**Status enum:** `pending` → `scheduled` → `running` → `completed` | `failed` → `verifying` → `verified` | `rejected` → `awaiting_review`

**Dependency model (Prefect-inspired):** `task_inputs` JSONB stores edges. Format:
```json
{
  "research_output": [{"type": "task_output", "task_id": "uuid-of-task-1", "output_key": "result"}],
  "__parents__": [{"task_id": "uuid-of-task-1"}]
}
```
A task is schedulable when all tasks referenced in `task_inputs` (both data deps and `__parents__`) are in a terminal success state. This avoids a separate edges table while keeping dependency resolution simple.

#### 3. `mission_events` — Append-Only Event Log

| Field | Type | Purpose |
|---|---|---|
| `id` | BIGSERIAL PK | Monotonic ordering |
| `mission_run_id` | UUID FK→mission_runs | |
| `mission_task_id` | UUID FK→mission_tasks NULL | NULL for run-level events |
| `event_type` | VARCHAR(50) | See enum below |
| `event_data` | JSONB | Event-specific payload |
| `actor` | VARCHAR(30) | `coordinator`, `agent`, `verifier`, `user`, `system` |
| `actor_id` | VARCHAR(255) | Agent ID, user ID, or system identifier |
| `created_at` | TIMESTAMPTZ | |

**Event types:**
- Run-level: `MISSION_CREATED`, `PLAN_GENERATED`, `PLAN_APPROVED`, `PLAN_REVISED`, `EXECUTION_STARTED`, `BUDGET_WARNING`, `BUDGET_EXCEEDED`, `MISSION_COMPLETED`, `MISSION_FAILED`, `MISSION_CANCELLED`
- Task-level: `TASK_CREATED`, `TASK_SCHEDULED`, `TASK_STARTED`, `TASK_OUTPUT`, `TASK_COMPLETED`, `TASK_FAILED`, `TASK_RETRY_SCHEDULED`, `TASK_VERIFICATION_STARTED`, `TASK_VERIFIED`, `TASK_REJECTED`, `TASK_HUMAN_ACCEPTED`, `TASK_HUMAN_REJECTED`

**Design decision:** Events supplement, not replace, the status columns on `mission_runs` and `mission_tasks`. Status columns are the "MutableState" cache (Temporal pattern) for fast queries. Events are the audit log for debugging, replay, and telemetry extraction.

#### 4. ~~`mission_tags`~~ — DEFERRED

> **Decision (review 2026-03-15):** Cut from v1. The `metadata` JSONB field on `mission_runs` provides the same extensibility without a join table. Add a dedicated tags table only if JSONB query performance becomes a bottleneck (unlikely at expected mission volume). Dagster's pattern is designed for thousands of runs — we'll have tens to hundreds.

---

## Section 4: Key Design Questions

These must be answered in the full PRD-101:

### Q1: Event Sourcing vs CRUD with Audit Log?

**Options:**
- **Full event sourcing (Temporal/Dagster):** Events are ground truth, status is derived. Enables replay, full audit, time-travel queries. Cost: more complex writes, status must be reconstructed.
- **CRUD + append-only event log (recommended):** Status columns are the primary source for queries. Events are an audit/telemetry log. Simpler queries, still full audit trail. Status update + event insert happen in same transaction.

**Recommendation:** CRUD + event log. We don't need replay capability (agents aren't deterministic). We need fast status queries (dashboard, reconciliation) and full audit trail (debugging, telemetry).

### Q2: Dependencies — JSON on Task Row vs Edges Table?

**Options:**
- **JSON on task row (Prefect pattern):** `task_inputs JSONB` stores edges inline. Simpler writes. Graph reconstructed at query time. No join needed for "get task with its deps."
- **Separate edges table:** `mission_task_edges(from_task_id, to_task_id, edge_type)`. Easier "find all tasks blocked by X" queries. More normalized. Additional table to maintain.
- **Hybrid (recommended):** Store edges in `task_inputs` JSONB for write simplicity, but use a PostgreSQL function or materialized view for "blocked tasks" queries if performance requires it.

**Recommendation:** JSON on task row (Prefect pattern). AI missions will have 3-20 tasks, not thousands. The simpler write path matters more than query optimization at this scale.

### Q3: State Machine — How Many States?

**Design constraints:**
- Must distinguish "waiting for dependencies" from "ready to execute" from "running"
- Must support verification as a distinct phase
- Must support human review as a gate
- Must distinguish continuation (clean exit, more work needed) from retry (failure)

**Proposed task states:** `pending` (created, deps not met) → `scheduled` (deps met, ready for agent) → `running` → `completed` (agent done) → `verifying` → `verified` / `rejected` → `awaiting_review` (human gate)

**Named sub-states for richness:** `(status=pending, status_name="awaiting_dependency")`, `(status=running, status_name="calling_tool")`, `(status=scheduled, status_name="awaiting_concurrency_slot")`

### Q4: Board Task Integration — Mirror, Link, or Replace?

**Options:**
- **Mirror:** Create a `board_task` for every `mission_task`, keep them in sync. Users see missions on the board. Risk: dual-write consistency.
- **Link:** `mission_tasks.board_task_id` FK. Create board task on mission creation, update it when mission task changes. Board is the user-facing view; mission_tasks is the orchestration view.
- **Replace:** Don't use board_tasks for missions. Missions have their own UI.

**Recommendation:** Link. Create a `board_task` per `mission_task` with `source_type='mission'` and `source_id=mission_run_id`. The board becomes a unified view of all work (manual tasks, recipe tasks, mission tasks). The heartbeat agent tick already picks up board_tasks — this keeps missions visible to the existing system.

### Q5: How Do Mission Tasks Relate to Recipe Steps?

When a successful mission is "saved as routine," the `mission_tasks` structure must map to `workflow_recipes.steps` JSONB. The full PRD must define this mapping:
- Which mission_task fields map to recipe step fields
- How dependency edges translate to recipe step ordering
- Whether contractor_config is preserved in the recipe

---

## Section 5: Existing Codebase Touchpoints

Every file/table the schema must integrate with:

### Database Tables (verified via schema audit)

| Table | Integration Point |
|---|---|
| `workspaces` | `mission_runs.workspace_id` FK. `settings.orchestrator` may need mission config. |
| `agents` | `mission_tasks.assigned_agent_id` FK. Agent config drives tool/model selection. |
| `board_tasks` | `mission_tasks.board_task_id` FK. Must extend `source_type` enum to include `'mission'`. `parent_task_id` can group mission tasks under a parent board task. |
| `workflow_recipes` | `mission_runs.recipe_id` FK (created from recipe). `mission_runs.saved_as_recipe_id` FK (saved as recipe). |
| `recipe_executions` | No direct FK, but reconciliation logic should handle both recipe and mission stalls. |
| `heartbeat_results` | Mission tasks executed via agent ticks may generate heartbeat results. Link via `mission_events`. |
| `agent_reports` | Agents executing mission tasks may auto-generate reports. Link via `mission_task_id` tag or event. |
| `memory_short_term` | Mission context may be stored as `content_type='mission_context'`. |
| `llm_usage` | Token/cost tracking per mission task should cross-reference `llm_usage` for billing. |

### Code Files (verified to exist)

| File | Why It Matters |
|---|---|
| `orchestrator/core/models/core.py` | Contains Agent, Skill, Workflow, BoardTask models. New mission models go here or in a new `mission.py`. |
| `orchestrator/core/models/board.py` | BoardTask query helpers. Must add mission-aware queries. |
| `orchestrator/services/heartbeat_service.py` | Agent tick picks up board_tasks. Must understand mission-sourced tasks. |
| `orchestrator/services/task_reconciler.py` | Currently only reconciles `recipe_executions`. Must extend to `mission_runs`/`mission_tasks`. |
| `orchestrator/modules/agents/factory/agent_factory.py` | `execute_with_prompt()` — the execution path for mission tasks. Contractor agents need to work through this. |
| `orchestrator/modules/tools/tool_router.py` | `get_tools_for_agent()` — tool assignment for mission task agents. |
| `orchestrator/modules/context/service.py` | ContextService has 8 modes. May need `COORDINATOR` and `VERIFIER` modes for PRD-102. |
| `orchestrator/modules/memory/unified_memory_service.py` | 5-layer memory. Mission context flows through here. |
| `alembic/versions/` | New migration required for all new tables. Must follow existing naming: `prd101_mission_schema.py`. |

---

## Section 6: Acceptance Criteria for Full PRD-101

The complete PRD-101 is done when:

- [ ] **Prior art comparison table** — Temporal, Prefect, Airflow, Dagster, Symphony approaches compared across: state machine, dependency model, event sourcing, result passing, reconciliation. With explicit "what we adopt" and "what we reject" per system.
- [ ] **Complete SQLAlchemy models** — Python code for `MissionRun`, `MissionTask`, `MissionEvent`, `MissionTag` with all fields, types, constraints, indices, and relationships.
- [ ] **Alembic migration script** — Complete `prd101_mission_schema.py` migration creating all tables, indices, and constraints.
- [ ] **State machine specification** — Formal state diagrams for both `mission_runs` and `mission_tasks` with all valid transitions, guard conditions, and event types emitted per transition.
- [ ] **Dependency resolution algorithm** — Pseudocode for "given a mission_run_id, which tasks are schedulable?" including handling of `task_inputs` JSONB parsing.
- [ ] **Board task integration spec** — How `mission_tasks` map to `board_tasks` (creation, status sync, deletion). Include code for the sync function.
- [ ] **Recipe conversion spec** — How a completed mission maps to `workflow_recipes.steps` JSONB for "save as routine" functionality.
- [ ] **Event schema catalog** — Every event type with its `event_data` JSONB schema defined.
- [ ] **Query patterns** — SQL/ORM queries for: dashboard (all missions for workspace), mission detail (tasks + events), schedulable tasks, stall detection, cost aggregation.
- [ ] **Migration strategy** — How to deploy without breaking existing board_tasks, recipe_executions, or heartbeat flows.

---

## Section 7: Risks & Dependencies

### Risks

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | Schema too complex for Phase 2 scope | High | Start with mission_runs + mission_tasks only. Add events table if needed. mission_tags can wait. |
| 2 | Board task sync creates dual-write bugs | Medium | Use database triggers or application-level transaction wrapping. Test sync edge cases exhaustively. |
| 3 | JSONB dependency edges become a query bottleneck | Low (at expected scale) | AI missions will have 3-20 tasks. Monitor query performance. Add materialized view if needed. |
| 4 | State machine is wrong — too many or too few states | Medium | Study actual mission execution flows before finalizing. The verification and human review phases are the most uncertain. |
| 5 | Event log table grows unbounded | Medium | Add retention policy. Index on `(mission_run_id, created_at)`. Consider partitioning by month if volume is high. |
| 6 | Recipe conversion loses fidelity | Medium | Not all mission patterns map cleanly to sequential recipe steps. Define what's convertible and what's not. |

### Dependencies

| Dependency | Direction | Notes |
|---|---|---|
| PRD-102 (Coordinator) | **Blocked by 101** | Coordinator creates and updates mission_runs/tasks. Needs the schema. |
| PRD-103 (Verification) | **Blocked by 101** | Verifier reads mission_tasks.success_criteria, writes verifier_score/feedback. |
| PRD-104 (Ephemeral Agents) | **Blocked by 101** | Contractor config lives in mission_tasks.contractor_config JSONB. |
| PRD-105 (Budget) | **Blocked by 101** | Budget tracking uses mission_runs.budget_config/budget_spent. |
| PRD-106 (Telemetry) | **Blocked by 101** | Telemetry extraction queries mission_events + mission_tasks cost/token fields. |
| PRD-107 (Context Interface) | Loosely coupled | Context interface wraps ContextService, doesn't directly touch mission schema. |
| PRD-108 (Memory Field) | Loosely coupled | Memory field prototype may use mission context but doesn't depend on schema. |
| Existing `board_tasks` | **Integration** | Must not break existing board functionality. New `source_type='mission'` value. |
| Existing `task_reconciler` | **Extension** | Must extend to handle mission stall detection alongside recipe stall detection. |

---

## Appendix: Research Summary Matrix

| Aspect | Temporal | Prefect | Airflow | Dagster | Symphony |
|---|---|---|---|---|---|
| **State storage** | MutableState blob (derived from events) | Denormalized columns on run/task rows | status column on dag_run/task_instance | status column on runs (derived from events) | In-memory only (tracker is source of truth) |
| **Event sourcing** | Full (history_node is ground truth) | Partial (state rows are append-only history) | None (status is mutable) | Full (event_logs is ground truth) | None |
| **Dependency model** | Implicit in Command sequence | `task_inputs` JSONB on task row | Static DAG + trigger rules | Static graph from asset/op definitions | Tracker-driven (external) |
| **Result passing** | In history events | ResultRecordMetadata reference in state | XCom (key-value store) | IO Manager (pluggable storage) | Workspace filesystem |
| **Retry model** | Activity retry without events; final outcome only | Server-enforced via orchestration rules | try_number on same task_instance row | STEP_UP_FOR_RETRY event, parent_run_id chain | Continuation (1s) vs exponential backoff |
| **Reconciliation** | Shard-based, range_id fencing | Server-side state rules | Scheduler polling loop | Daemon heartbeats + run monitoring | Reconcile-before-dispatch on every tick |
| **Tags/metadata** | Memo fields, search attributes | tags + labels on run/task | conf JSONB on dag_run | run_tags table (key-value) | WORKFLOW.md config |
| **What we adopt** | Event log as audit (not ground truth), MutableState-as-cache concept | `task_inputs` JSONB for deps, named sub-states, composable state rules | Trigger rules concept (simplified) | Append-only events for audit, tags table, content-addressed snapshots | Continuation vs retry distinction, reconciliation-before-dispatch |
| **What we reject** | Full event sourcing (too complex for our scale), blob-serialized state | No edges table (we agree — JSON is fine at our scale) | Static DAG requirement, polling scheduler, XCom for results | Step state only in events (we want denormalized status columns too) | No persistent DB (we need durable state) |
