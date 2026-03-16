# PRD-101 — Mission Schema & Data Model

**Version:** 1.0
**Type:** Research + Design
**Status:** Complete — Ready for Peer Review
**Priority:** P0
**Dependencies:** PRD-100 (Research Master)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-14

---

## 1. Problem Statement

### 1.1 The Gap

Automatos has a production-grade foundation: 340 LLMs, 850 tools, 11 channel adapters, 5-layer memory, a Kanban board, agent reports, scheduled heartbeats, and a recipe engine. Users can do single-agent Tasks and scheduled Routines today. What they cannot do is describe a complex goal — "Research EU AI Act compliance for our product" — and have the system decompose it into subtasks, assign agents, execute with verification, and track everything on the board.

This is the **Mission** gap identified in PRD-100 (Section 3). Every piece of infrastructure exists except the data layer that makes coordinated multi-agent execution persistent, traceable, and recoverable.

Specifically, the platform has **no**:

| Missing Component | Why It Matters |
|-------------------|----------------|
| `orchestration_runs` table | No way to record a mission's goal, plan, config, cost, or status |
| `orchestration_tasks` table | No way to track subtasks with dependencies, assignment, and verification |
| `orchestration_events` table | No audit trail for state transitions — debugging requires log-grepping |
| Task dependency graph | No DAG, no topological sort, no "what's ready to run?" query |
| State machine for task lifecycle | Board tasks have 4 statuses; missions need 10+ with defined transitions |
| Integration contracts | No defined mapping from orchestration tables → board_tasks, agent_reports, recipes |

Without this data layer, the Coordinator (PRD-102), Verifier (PRD-103), Ephemeral Agents (PRD-104), Budget Enforcement (PRD-105), and Telemetry (PRD-106) have nothing to read from or write to. **The schema is the foundation everything else stands on.**

### 1.2 What This PRD Delivers

This document is the research and design specification for the mission data layer. It delivers:

- [x] **Prior art analysis** (Section 2) — five production systems studied: Temporal, Prefect, Airflow, Dagster, Symphony. Key patterns extracted: dual-write state tracking, DB-authoritative scheduling, continuation vs retry, infrastructure vs quality failure classification.
- [x] **State machine design** (Section 3) — two-level state model (StateType + StateName), full transition tables for runs and tasks, continuation vs retry semantics, stall detection algorithm, board task status mapping, concurrency safety via optimistic locking.
- [x] **`orchestration_runs` table** (Section 4) — 20 columns, JSONB schemas for plan and config, 7 indexes, Alembic migration pseudocode.
- [x] **`orchestration_tasks` table** (Section 5) — 25 columns, join table for dependencies with trigger rules, topological sort algorithm using `graphlib.TopologicalSorter`, board task bridge functions, Alembic migration.
- [x] **`orchestration_events` table** (Section 6) — append-only audit trail, 30+ event types, BRIN indexes for time-series queries, retention policy, connection to PRD-106 telemetry.
- [x] **Integration contracts** (Section 7) — field mappings to board_tasks, agent_reports, and recipes. ER diagram. Migration safety guarantees. Workspace isolation pattern.
- [x] **Full SQL DDL** (Section 12) — copy-pasteable CREATE TABLE statements for all 5 tables + existing table alterations.
- [x] **SQLAlchemy models** (Section 13) — Python model classes matching existing codebase conventions, with enums, transition tables, and board status mapping.

### 1.3 What This PRD Does NOT Cover

These are explicitly out of scope — each has its own research PRD:

| Out of Scope | Covered By |
|-------------|------------|
| How the coordinator decomposes goals into tasks | PRD-102 (Coordinator Architecture) |
| How verification/scoring works | PRD-103 (Verification & Quality) |
| Ephemeral "contractor" agent lifecycle | PRD-104 (Ephemeral Agents & Model Selection) |
| Budget enforcement and approval gates | PRD-105 (Budget & Governance) |
| Outcome telemetry queries and learning | PRD-106 (Outcome Telemetry & Learning Foundation) |
| Context interface for Phase 3 swap | PRD-107 (Context Interface Abstraction) |
| API endpoints and service layer | PRD-82A (Implementation PRD) |
| Frontend changes beyond existing board | PRD-82A (Implementation PRD) |

This PRD designs the **tables, state machine, and integration contracts**. PRD-82A will take these designs and produce the Alembic migration, SQLAlchemy models, API endpoints, and service layer code.

### 1.4 Design Philosophy

Five principles guided every decision in this document:

1. **Research before building.** Every architectural choice in Sections 3-7 cites a production system that validated the pattern at scale. No design-by-vibes.
2. **DB-authoritative.** The database is the single source of truth for mission state. The coordinator re-derives "what's ready to run?" from DB on every tick. No in-memory state to lose on crash.
3. **Dual-write for state.** Current state denormalized on the row for O(1) dashboard queries. Every transition also appended to `orchestration_events` for audit trail. Both in one transaction.
4. **Additive-only integration.** Three new tables. Zero changes to existing table structures. New optional columns and indexes on existing tables added via `CREATE INDEX CONCURRENTLY`. No downtime. No backfill.
5. **Schema enables, implementation decides.** The schema supports parallel execution, budget enforcement, and verification scoring — but whether those features are built sequentially or in parallel is an implementation decision for PRD-82A through 82D.

---

## 2. Prior Art: DAG Execution Patterns

### 2.1 Overview

Mission orchestration requires storing runs (missions), tasks (subtasks), dependencies between tasks, and state transitions. Before designing our schema, we studied five production systems that solve related problems at scale. Each takes a fundamentally different approach to the same core challenge: how do you persist the state of a multi-step execution with dependencies?

The systems studied:
- **Temporal** — workflow-as-code with deterministic replay over an append-only event history
- **Prefect** — flow/task runs with denormalized state + append-only state history tables
- **Apache Airflow** — DAG runs with DB-authoritative scheduling and trigger rules
- **Dagster** — event-sourced execution with the event log as the system of record
- **OpenAI Symphony** — tracker-as-coordinator with in-memory orchestration and policy-as-code

### 2.2 Comparison Table

| Dimension | Temporal | Prefect | Airflow | Dagster | Symphony |
|-----------|----------|---------|---------|---------|----------|
| **Run model** | `executions` table; state stored as serialized protobuf blob; identity = `(namespace, workflow_id, run_id)` | `flow_run` table; state denormalized inline (`state_type`, `state_name`) + FK to `flow_run_state` history | `dag_run` table; `state` column (queued/running/success/failed); unique on `(dag_id, run_id)` | `runs` table; `run_body` TEXT = full serialized `DagsterRun` JSON; denormalized `start_time`/`end_time` | No persistent run table — Linear issue is the run record; in-memory claim state only |
| **Task model** | `activity_info_maps` table; one row per in-flight activity; serialized `PendingActivityInfo` proto blob | `task_run` table; explicit columns for state, timing, retry policy, cache, task_inputs | `task_instance` table; composite key `(dag_id, run_id, task_id, map_index)`; no FK to DAG model (by design) | No task table — step state derived from `event_logs` entries keyed by `(run_id, step_key)` | No task table — each Linear issue = one task; agent workspace = execution context |
| **Dependency model** | Implicit in workflow code — replay-derived; no stored DAG | `task_inputs` JSON column — typed refs to upstream `TaskRunResult` / `Parameter` / `Constant` | Serialized in `serialized_dag` table; trigger rules evaluate upstream states at scheduling time | Declared in `GraphDefinition` dependency dict; stored in serialized `JobSnapshot` | Linear issue dependencies (if any); mostly single-task-per-issue model |
| **State machine** | Workflow: `RUNNING → COMPLETED/FAILED/CANCELED/TERMINATED/CONTINUED_AS_NEW/TIMED_OUT/PAUSED`. Activity: `SCHEDULED → STARTED → (completed via event)` | 9 StateTypes: `SCHEDULED → PENDING → RUNNING → COMPLETED/FAILED/CRASHED/CANCELLED`. Sub-states: `AwaitingRetry`, `Cached`, `Late`, `Suspended` | DAG: `queued → running → success/failed`. Task: `scheduled → queued → running → success/failed/skipped/upstream_failed` + `up_for_retry`, `deferred`, `removed` | Run: `NOT_STARTED → QUEUED → STARTING → STARTED → SUCCESS/FAILURE/CANCELED`. Steps: event-derived `STEP_START → STEP_SUCCESS/STEP_FAILURE/STEP_SKIPPED` | Claim: `Unclaimed → Claimed → Running → RetryQueued/Released`. Work phases: `PreparingWorkspace → BuildingPrompt → LaunchingAgent → StreamingTurn → Succeeded/Failed/TimedOut/Stalled` |
| **Failure handling** | 4 timeout types (schedule-to-close, schedule-to-start, start-to-close, heartbeat). Retry policy with exponential backoff. Heartbeat checkpointing for long activities | Retries with configurable delay + jitter. `CRASHED` state for infrastructure failures (distinct from `FAILED`). Cache-based skip on re-execution | `retries` + `retry_delay` per task. `up_for_retry` state. Trigger rules cascade failure downstream (`upstream_failed`). `retry_exponential_backoff` flag | Retry via `RetryRequested` event; step re-enters execution. IO managers externalize data so retries don't lose upstream results | Continuation (clean exit) = 1s delay, same workspace, no attempt increment. Retry (failure) = exponential backoff `10s × 2^(attempt-1)`, fresh branch possible |
| **Inter-task data** | Activity results stored as events in history; workflow code reads them via SDK await | Results stored as `Artifact` records; `task_run_state._data` JSON or external storage via `result_artifact_id` | XCom table: `(dag_id, run_id, task_id, key) → value`. Small values only; pluggable backend for large payloads | IO managers: `handle_output()` serializes to external storage, `load_input()` deserializes. No in-process passing by default | No inter-task data passing — single-agent-per-issue model. Agent reads its own prior commits/workpad |
| **Event/audit log** | `history_node` table — append-only event history; all state reconstructable via deterministic replay | `flow_run_state` / `task_run_state` tables — append-only state history per entity | `log` table for task logs; state changes tracked on `task_instance` row directly (mutable) | `event_logs` table — THE system of record. All state derived from this append-only stream | No persistent log — in-memory token accounting per session. Linear comments serve as audit trail |

### 2.3 System-by-System Analysis

#### Temporal

Temporal's defining architectural choice is **deterministic replay over explicit state storage**. The database schema is deliberately opaque — most domain state lives in serialized protobuf blobs in the `executions` table, not in queryable columns. Dependencies between activities are never stored; they're encoded implicitly in deterministic workflow code. When a workflow needs to resume, Temporal replays the entire event history (`history_node` table) against the same code to reconstruct exact execution state.

This works brilliantly for Temporal's use case (long-running business processes with complex branching) but is explicitly **wrong for our needs**. We need queryable dependency structure ("show me all tasks blocked by task X"), human-readable state ("what's the mission doing right now?"), and dashboard visibility — all of which require explicit, denormalized columns rather than opaque blobs.

**What we adopt:** The execution chain concept (`workflow_id` persists across retries/continuations while `run_id` is unique per attempt) maps well to our mission model — a mission ID persists while individual task attempts get their own IDs. The four distinct timeout types (schedule-to-close, schedule-to-start, start-to-close, heartbeat) inform our timeout design for agent tasks.

**What we reject:** Serialized blob storage, deterministic replay, implicit dependency encoding. Our users need to see and query mission state directly.

**Source:** PostgreSQL schema at `temporalio/temporal/schema/postgresql/v12/temporal/schema.sql`; proto definitions in `temporal/api/workflow/v1/message.proto` and `temporal/api/history/v1/message.proto`.

#### Prefect

Prefect's key innovation is **dual-write state tracking**: current state is denormalized inline on the run/task row (`state_type`, `state_name`, `state_timestamp`) for O(1) query performance, while every state transition is also written as an immutable row in `flow_run_state` / `task_run_state` history tables. This gives you both fast current-state queries and a complete audit trail.

The dependency model stores task inputs as a JSON column (`task_inputs`) with typed references to upstream task runs, parameters, or constants — rather than a separate edges table. This is compact and self-contained per task but makes "find all downstream tasks of X" require scanning all tasks' `task_inputs` columns.

Prefect's 9-state model with sub-states (`Cached`, `AwaitingRetry`, `Late`, `Suspended`) is the richest among the systems studied. The distinction between `FAILED` (code error) and `CRASHED` (infrastructure failure) is particularly useful — it answers "should we retry?" differently based on failure type.

**What we adopt:** The dual-write pattern (denormalized current state + append-only event log) is the strongest architectural pattern across all systems studied. The `CRASHED` vs `FAILED` distinction maps directly to our agent execution — an agent hitting a timeout is different from an agent producing wrong output. The `empirical_policy` JSON for retry configuration per task is a clean pattern.

**What we reject:** JSON-encoded dependency graph in `task_inputs` — at our scale it works, but a join table is more queryable for "find blocked tasks." The `dynamic_key` uniqueness approach is over-engineered for our use case.

**Source:** ORM models at `PrefectHQ/prefect/src/prefect/server/database/orm_models.py`; state definitions at `src/prefect/server/schemas/states.py`.

#### Apache Airflow

Airflow's defining principle is **the database is the single source of truth**. The scheduler holds no authoritative state in memory — it reads and writes `task_instance` rows for every scheduling decision, using pessimistic locking (`SELECT FOR UPDATE`) to coordinate multiple schedulers in HA mode. Dependencies exist only in the serialized DAG definition, and the scheduler re-evaluates trigger rules against current DB state on every tick.

Airflow's trigger rule system is the most sophisticated dependency model studied. Beyond simple "all predecessors must succeed," it supports `one_success` (fire on first upstream success without waiting), `none_failed` (tolerates skips but not failures), `all_done` (fires regardless of upstream state), and 9 other rules. This enables complex conditional execution patterns.

The XCom mechanism for inter-task data is deliberately constrained — small values only, with pluggable backends for large payloads. This separation of metadata passing (XCom) from bulk data transfer (external storage) is a pattern worth adopting.

**What we adopt:** DB-authoritative scheduling — our coordinator should re-derive "what's ready to run" from DB state, not trust in-memory queues. The trigger rule concept (though we'll start with just `all_success` and `all_done`). The XCom pattern — task outputs stored separately from task metadata, with the task row pointing to the output location. The deliberate omission of ORM foreign keys between high-write tables to avoid lock contention.

**What we reject:** The `logical_date` / data interval concept is specific to batch processing, not mission orchestration. The `pool` / `pool_slots` resource scheduling is more complexity than we need initially.

**Source:** Models at `apache/airflow/airflow-core/src/airflow/models/dagrun.py` and `taskinstance.py`; trigger rules documented at `airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html`.

#### Dagster

Dagster's architecture is the purest event-sourcing model studied. The `event_logs` table is the **sole system of record** — there is no `steps` or `ops` table. Step state is entirely derived from the sequence of events for a given `(run_id, step_key)` pair. The `runs` table stores a `run_body` TEXT column containing the full serialized `DagsterRun` object, with denormalized columns (`status`, `start_time`, `end_time`) maintained as query-performance shortcuts.

IO managers decouple inter-op data passing from execution logic — outputs are serialized to external storage (S3, database, filesystem) and deserialized for downstream consumers. This means no data flows between ops in-process, making retry and re-execution safe by default.

The asset materialization model — tracking what data assets were produced, when, and by which run — is a novel concept that maps to our mission outputs. A completed research task produces a "research artifact" that downstream tasks consume.

**What we adopt:** The event log as an append-only audit trail (though not as the sole system of record — we'll use Prefect's dual-write pattern). The concept of separating task output storage from task metadata. The `run_tags` key-value table for flexible metadata without schema changes.

**What we reject:** Full event sourcing as the primary state model — it makes simple queries ("which tasks are currently running?") require event stream scanning. The serialized `run_body` blob is the same anti-pattern as Temporal's approach for our needs.

**Source:** Schema at `dagster-io/dagster/python_modules/dagster/dagster/_core/storage/runs/schema.py` and `event_log/schema.py`; DagsterRunStatus enum at `dagster/_core/storage/dagster_run.py`.

#### OpenAI Symphony

Symphony takes the most radical approach: **no persistent orchestration database at all**. The Linear issue tracker is the coordinator — Symphony polls it for eligible issues, claims them in memory, and dispatches coding agents. All durable state lives in Linear (issue status, comments, PR links) and Git (branches, commits, workpad files). The orchestrator is deliberately stateless and recovers from restart by re-polling.

The continuation vs retry distinction is Symphony's most valuable contribution. A clean agent exit (task still in progress) triggers immediate continuation — same workspace, same thread, no backoff. An abnormal exit (failure, timeout, stall) triggers exponential backoff retry with a fresh branch. This prevents thrashing on failures while keeping normal multi-turn work fast. The `attempt` counter is passed to the agent via the WORKFLOW.md template so the agent knows whether it's continuing or retrying.

The WORKFLOW.md policy-as-code pattern — runtime configuration (concurrency, timeouts, active states, hooks) in YAML front matter, agent prompt template in Markdown body — is elegant for teams that want version-controlled orchestration policy.

**What we adopt:** The continuation vs retry distinction — our coordinator should handle "agent needs more turns" differently from "agent failed." The concept of passing attempt context to agents so they can adapt behavior. Lifecycle hooks (before_run, after_run) with asymmetric failure semantics — pre-hooks abort, post-hooks are best-effort. Stall detection via elapsed time since last event.

**What we reject:** No persistent storage — we need queryable mission history, cost tracking, and dashboard visibility. Linear-as-coordinator — we have our own board and need the coordinator to be a first-class service. In-memory-only orchestration state.

**Source:** `openai/symphony/SPEC.md` for architecture; `openai/symphony/elixir/WORKFLOW.md` for policy-as-code reference implementation.

### 2.4 Architectural Decisions Informed by Prior Art

Based on this analysis, our mission schema adopts the following patterns:

| Decision | Pattern | Source | Rationale |
|----------|---------|--------|-----------|
| **State storage** | Dual-write: denormalized current state on row + append-only event log | Prefect | Fast queries for dashboards AND complete audit trail for debugging |
| **Dependency storage** | Explicit join table (`orchestration_task_dependencies`) | Airflow (conceptual) | Queryable in both directions: "what blocks X?" and "what does X block?" |
| **Scheduling authority** | DB-authoritative — coordinator re-derives ready tasks from DB state each tick | Airflow | Crash-safe; no in-memory state to lose; supports future HA coordinator |
| **Failure classification** | Distinguish infrastructure failure from output quality failure | Prefect (`CRASHED` vs `FAILED`) | Different retry strategies: infra failure → retry same task; quality failure → retry with different model or escalate |
| **Continuation vs retry** | Separate continuation (more turns needed) from retry (something broke) | Symphony | Prevents backoff on normal multi-turn agent work while protecting against failure loops |
| **Inter-task data** | Separate output storage from task metadata; task row references output location | Dagster (IO managers), Airflow (XCom) | Keeps task table lean; outputs can be large (research reports, analysis docs) |
| **Timeout model** | Multiple timeout types per task (total deadline, per-attempt, stall detection) | Temporal (4 timeouts), Symphony (stall detection) | Different failure modes need different timeouts |
| **Run identity** | Mission ID persists across retries; task attempts get unique IDs | Temporal (execution chains) | Users track missions by stable ID; system tracks individual attempts for cost/debugging |
| **Flexible metadata** | Tags/labels as key-value pairs in a separate table or JSONB column | Dagster (`run_tags`), Airflow (`conf`) | Extensible without schema migration; supports filtering, grouping, search |
| **Trigger rules** | Start with `all_success` (default) and `all_done`; add more later | Airflow | Simple cases first; the framework supports richer rules when needed |

### 2.5 What We Explicitly Avoid

1. **Serialized blob storage** (Temporal, Dagster `run_body`) — our users need to query mission state from dashboards and APIs without deserialization
2. **Full event sourcing as primary state model** (Dagster) — adds query complexity for common operations; we use events as audit trail, not source of truth
3. **Implicit dependency encoding** (Temporal) — we can't replay LLM calls deterministically; dependencies must be explicit and queryable
4. **In-memory-only orchestration** (Symphony) — we need persistent mission history for cost tracking, learning (PRD-106), and user review
5. **Tracker-as-coordinator** (Symphony) — our board is a visibility layer, not the control plane; the coordinator service owns execution logic

---

## 3. State Machine Design

### 3.1 Design Philosophy

The state machine must serve three audiences simultaneously:

1. **The coordinator** — needs to know what's ready to run, what's blocked, and what failed
2. **The dashboard** — needs human-readable status that maps to the existing board_tasks UI
3. **The debugger** — needs a complete transition history to answer "what happened?"

We adopt a **two-level state model** inspired by Prefect's architecture: a small, stable `StateType` enum drives orchestration logic, while a richer `StateName` provides user-facing detail. This lets us add display states (e.g., `awaiting_payment`) without touching coordinator code.

We also adopt the **dual-write pattern** validated in Section 2: every state transition updates the denormalized current-state column on the row (fast queries) AND appends an immutable event to `orchestration_events` (audit trail). Both writes occur in a single database transaction. This is the same pattern Prefect uses at significantly larger scale than our target (~100-500 concurrent runs).

**Why not full event sourcing?** We don't need deterministic replay (Temporal's use case). Our agents are non-deterministic LLMs — replaying orchestration code wouldn't reproduce the same results. Event sourcing adds projection maintenance, snapshot management, and eventual consistency complexity that isn't justified at our scale. The hybrid approach gives us O(1) current-state queries and a complete audit trail without the overhead.

**Why not pure CRUD?** Airflow's mutable-only approach makes debugging "why did this task get stuck?" require grepping application logs. We need structured transition history for mission observability, telemetry (PRD-106), and human review.

### 3.2 State Definitions

#### Run States (orchestration_runs)

| StateType | StateName | Terminal? | Description | Triggered By |
|-----------|-----------|-----------|-------------|-------------|
| PENDING | `pending` | No | Run created, plan not yet approved | System (on mission creation) |
| PENDING | `planning` | No | Coordinator is decomposing the goal into tasks | Coordinator |
| PENDING | `awaiting_approval` | No | Plan ready, waiting for human to approve | Coordinator (after planning) |
| RUNNING | `running` | No | Tasks are being executed | Human (approves plan) or System (autonomy mode) |
| PAUSED | `paused` | No | Human paused execution | Human |
| PAUSED | `budget_exceeded` | No | Hard budget cap hit, waiting for human decision | System (budget check) |
| TERMINAL | `completed` | Yes | All tasks passed verification, human accepted | Human (accepts) or System (auto-accept mode) |
| TERMINAL | `failed` | Yes | Unrecoverable failure (max retries exhausted, human rejected) | System or Human |
| TERMINAL | `cancelled` | Yes | Human cancelled the mission | Human |

#### Task States (orchestration_tasks)

| StateType | StateName | Terminal? | Description | Triggered By |
|-----------|-----------|-----------|-------------|-------------|
| PENDING | `pending` | No | Task created, dependencies not yet met | Coordinator (during planning) |
| PENDING | `queued` | No | Dependencies met, waiting for agent slot | Dependency resolver |
| PENDING | `awaiting_retry` | No | Failed, scheduled for retry after backoff | System (retry logic) |
| RUNNING | `assigned` | No | Agent selected, execution starting | Coordinator |
| RUNNING | `running` | No | Agent actively working (LLM calls in progress) | Agent |
| RUNNING | `continuing` | No | Agent exited cleanly, needs more turns | Agent (clean exit) |
| PAUSED | `verifying` | No | Output submitted, verifier evaluating | Agent (submits output) |
| PAUSED | `awaiting_human` | No | Verifier or coordinator requested human review | Verifier or Coordinator |
| TERMINAL | `completed` | Yes | Passed verification (or human accepted) | Verifier or Human |
| TERMINAL | `failed` | Yes | Max retries exhausted or human rejected | System or Human |
| TERMINAL | `cancelled` | Yes | Parent run cancelled or human cancelled task | Run state change or Human |
| TERMINAL | `skipped` | Yes | Dependency failed with `all_done` trigger rule; task not needed | Dependency resolver |

#### StateType Mapping

Orchestration code switches on `StateType` (4 values, stable). Display and logging use `StateName` (extensible).

```python
from enum import StrEnum

class StateType(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINAL = "terminal"

class RunState(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    PAUSED = "paused"
    BUDGET_EXCEEDED = "budget_exceeded"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskState(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    AWAITING_RETRY = "awaiting_retry"
    ASSIGNED = "assigned"
    RUNNING = "running"
    CONTINUING = "continuing"
    VERIFYING = "verifying"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

RUN_STATE_TYPE: dict[RunState, StateType] = {
    RunState.PENDING: StateType.PENDING,
    RunState.PLANNING: StateType.PENDING,
    RunState.AWAITING_APPROVAL: StateType.PENDING,
    RunState.RUNNING: StateType.RUNNING,
    RunState.PAUSED: StateType.PAUSED,
    RunState.BUDGET_EXCEEDED: StateType.PAUSED,
    RunState.COMPLETED: StateType.TERMINAL,
    RunState.FAILED: StateType.TERMINAL,
    RunState.CANCELLED: StateType.TERMINAL,
}

TASK_STATE_TYPE: dict[TaskState, StateType] = {
    TaskState.PENDING: StateType.PENDING,
    TaskState.QUEUED: StateType.PENDING,
    TaskState.AWAITING_RETRY: StateType.PENDING,
    TaskState.ASSIGNED: StateType.RUNNING,
    TaskState.RUNNING: StateType.RUNNING,
    TaskState.CONTINUING: StateType.RUNNING,
    TaskState.VERIFYING: StateType.PAUSED,
    TaskState.AWAITING_HUMAN: StateType.PAUSED,
    TaskState.COMPLETED: StateType.TERMINAL,
    TaskState.FAILED: StateType.TERMINAL,
    TaskState.CANCELLED: StateType.TERMINAL,
    TaskState.SKIPPED: StateType.TERMINAL,
}

TERMINAL_RUN_STATES = frozenset(
    s for s, t in RUN_STATE_TYPE.items() if t == StateType.TERMINAL
)
TERMINAL_TASK_STATES = frozenset(
    s for s, t in TASK_STATE_TYPE.items() if t == StateType.TERMINAL
)
```

### 3.3 Transition Diagrams

#### Run State Transitions

```
                            ┌─────────────┐
                            │   pending    │
                            └──────┬──────┘
                                   │ coordinator starts planning
                                   ▼
                            ┌─────────────┐
                            │  planning    │
                            └──────┬──────┘
                                   │ plan ready
                                   ▼
                    ┌───────────────────────────────┐
                    │      awaiting_approval         │◄──── human pauses
                    └───────────┬───────────────────┘       │
                  human approves│    human rejects           │
             or autonomy mode   │         │                  │
                                ▼         ▼                  │
                         ┌──────────┐  ┌────────┐            │
              ┌─────────►│ running  │  │ failed │            │
              │          └────┬─────┘  └────────┘            │
              │               │                              │
              │    ┌──────────┼──────────┐                   │
              │    │          │          │                    │
              │    ▼          ▼          ▼                    │
              │ ┌──────┐  ┌──────────┐  ┌───────────────┐    │
              │ │paused│  │completed │  │budget_exceeded│    │
              │ └──┬───┘  └──────────┘  └───────┬───────┘    │
              │    │                            │            │
              │    └────────── resume ───────────┘            │
              │         (human continues)                    │
              └──────────────────────────────────────────────┘

         Any non-terminal state ──── human cancels ────► cancelled
```

#### Task State Transitions

```
                        ┌─────────┐
                        │ pending │
                        └────┬────┘
                             │ dependencies met
                             ▼
                        ┌─────────┐
                        │ queued  │
                        └────┬────┘
                             │ agent assigned
                             ▼
                        ┌──────────┐
                        │ assigned │
                        └────┬─────┘
                             │ execution begins
                             ▼
                        ┌─────────┐
              ┌────────►│ running │◄────────────────────┐
              │         └────┬────┘                     │
              │              │                          │
              │   ┌──────────┼───────────┐              │
              │   │          │           │              │
              │   ▼          ▼           ▼              │
              │ ┌────────┐ ┌──────────┐ ┌───────────┐   │
              │ │crashed │ │continuing│ │ output    │   │
              │ │(failed)│ │(clean    │ │ submitted │   │
              │ └───┬────┘ │ exit)    │ └─────┬─────┘   │
              │     │      └────┬─────┘       │         │
              │     │           │             ▼         │
              │     │    1s delay,      ┌───────────┐   │
              │     │    attempt=same   │ verifying │   │
              │     │           │       └─────┬─────┘   │
              │     │           │             │         │
              │     │           │    ┌────────┼────────┐│
              │     │           │    │        │        ││
              │     │           │    ▼        ▼        ▼│
              │     │           │ ┌──────┐ ┌───────┐ ┌──────────────┐
              │     │           │ │passed│ │failed │ │awaiting_human│
              │     │           │ └──┬───┘ └───┬───┘ └──────┬───────┘
              │     │           │    │         │            │
              │     │           │    ▼         ▼            │
              │     │           │ completed  retry?         │
              │     │           │         ┌───┴───┐    human decides
              │     │           │        yes      no       │
              │     │           │         │       │    ┌───┴───┐
              │     │           │         ▼       ▼   approve reject
              │     │           │  ┌──────────┐ failed  │      │
              │     │           │  │awaiting_ │         │      │
              │     │           │  │retry     │         ▼      ▼
              │     │           │  └────┬─────┘     completed failed
              │     │           │       │
              │     └───────────┼───────┘
              │                 │  backoff expires
              └─────────────────┘

         Any non-terminal state ──── run cancelled ────► cancelled
         Dependency failed + trigger=all_success ──────► skipped
```

### 3.4 Transition Tables

#### Run Transitions

| From | To | Trigger | Actor | Side Effects |
|------|----|---------|-------|-------------|
| `pending` | `planning` | Mission created | Coordinator | Emit `run_started` event |
| `planning` | `awaiting_approval` | Plan decomposition complete | Coordinator | Create `orchestration_tasks` rows; emit `plan_ready` event |
| `planning` | `running` | Plan complete + autonomy mode | Coordinator | Create tasks + begin execution; emit `plan_ready` + `run_started` |
| `awaiting_approval` | `running` | Human approves plan | Human (API) | Begin task execution; emit `human_approved` event |
| `awaiting_approval` | `failed` | Human rejects plan | Human (API) | Emit `human_rejected` event |
| `running` | `completed` | All tasks terminal + all passed | Coordinator | Set `completed_at`; emit `run_completed`; offer "save as routine" |
| `running` | `failed` | Unrecoverable task failure or budget exceeded without override | Coordinator | Set `completed_at`; emit `run_failed`; cancel remaining tasks |
| `running` | `paused` | Human pauses | Human (API) | Pause all non-terminal tasks; emit `run_paused` |
| `running` | `budget_exceeded` | Cost exceeds hard cap | System (budget check) | Pause all non-terminal tasks; emit `budget_exceeded` |
| `paused` | `running` | Human resumes | Human (API) | Resume paused tasks; emit `run_resumed` |
| `budget_exceeded` | `running` | Human increases budget | Human (API) | Resume tasks; emit `budget_increased` |
| `budget_exceeded` | `cancelled` | Human cancels | Human (API) | Cancel all tasks; emit `run_cancelled` |
| Any non-terminal | `cancelled` | Human cancels | Human (API) | Cancel all non-terminal tasks; emit `run_cancelled` |

#### Task Transitions

| From | To | Trigger | Actor | Side Effects |
|------|----|---------|-------|-------------|
| `pending` | `queued` | All dependencies in terminal success state | Dependency resolver | Emit `task_queued` |
| `pending` | `skipped` | Dependency failed + trigger rule = `all_success` | Dependency resolver | Emit `task_skipped` |
| `queued` | `assigned` | Agent selected by coordinator | Coordinator | Set `agent_id`; create board_task; emit `task_assigned` |
| `assigned` | `running` | Agent begins execution | Agent | Set `started_at`; update board_task → `in_progress`; emit `task_started` |
| `running` | `continuing` | Agent exits cleanly, needs more turns | Agent (clean exit) | Emit `task_continuing`; schedule continuation (1s delay, same attempt) |
| `continuing` | `running` | Continuation dispatched | System (timer) | Emit `task_resumed` |
| `running` | `verifying` | Agent submits output | Agent | Store output reference; update board_task → `review`; emit `task_output_submitted` |
| `verifying` | `completed` | Verifier passes output | Verifier agent | Set `verifier_score`; update board_task → `done`; emit `verification_passed` |
| `verifying` | `awaiting_human` | Verifier score below threshold or verifier uncertain | Verifier agent | Emit `human_review_requested` |
| `verifying` | `awaiting_retry` | Verifier fails output + retries remaining | Verifier agent | Set `verifier_score`; emit `verification_failed`; schedule retry with backoff |
| `verifying` | `failed` | Verifier fails output + no retries remaining | Verifier agent | Set `verifier_score`; update board_task → `done` (with error); emit `task_failed` |
| `awaiting_human` | `completed` | Human approves | Human (API) | Update board_task → `done`; emit `human_approved` |
| `awaiting_human` | `awaiting_retry` | Human rejects + retries remaining | Human (API) | Emit `human_rejected`; schedule retry |
| `awaiting_human` | `failed` | Human rejects + no retries | Human (API) | Update board_task → `done` (with error); emit `human_rejected` + `task_failed` |
| `awaiting_retry` | `assigned` | Backoff timer expires | System (timer) | Increment `attempt_count`; emit `task_retrying` |
| `running` | `failed` | Infrastructure failure (timeout, crash, OOM) | System (reconciler) | Update board_task → `done` (with error); emit `task_crashed` |
| `running` | `awaiting_retry` | Infrastructure failure + retries remaining | System (reconciler) | Emit `task_crashed`; schedule retry with backoff |
| Any non-terminal | `cancelled` | Parent run cancelled | Run state change | Update board_task → `done` (with error); emit `task_cancelled` |

### 3.5 Continuation vs Retry (from Symphony)

The distinction between continuation and retry is critical for AI agent tasks. An agent researching a topic may need 5 LLM turns — each "exit" between turns is a continuation, not a failure.

| Dimension | Continuation | Retry |
|-----------|-------------|-------|
| **Trigger** | Agent exits cleanly, work incomplete | Infrastructure failure, verification failure, or timeout |
| **Attempt counter** | Unchanged (same attempt) | Incremented |
| **Delay** | 1 second (fixed) | Exponential backoff: `min(10s × 2^(attempt-1), 5min)` |
| **Agent context** | Same agent, workspace preserved, prior output available | Same or different agent, fresh prompt with attempt number |
| **State sequence** | `running → continuing → running` | `running → awaiting_retry → assigned → running` |
| **Board task status** | Stays `in_progress` | Briefly shows retry status, then back to `in_progress` |
| **Budget impact** | Counts toward task budget | Counts toward task budget (coordinator may switch to cheaper model on retry) |
| **Max turns** | Configurable per task (default: 10, matching existing AgentFactory tool loop) | Configurable per task (default: 3) |

**Backoff progression for retries:**

| Attempt | Delay | Cumulative Wait |
|---------|-------|-----------------|
| 1 | 10s | 10s |
| 2 | 20s | 30s |
| 3 | 40s | 70s |
| 4 | 80s | 150s (2.5min) |
| 5 | 160s | 310s (5.2min) |
| 6+ | 300s (cap) | +5min each |

### 3.6 Failure Classification

Following Prefect's `CRASHED` vs `FAILED` distinction, adapted for AI agent execution:

| Failure Type | Cause | Retryable? | Retry Strategy | Example |
|-------------|-------|-----------|----------------|---------|
| **Infrastructure failure** (CRASHED equivalent) | Agent timeout, OOM, network error, provider outage | Yes (auto) | Same task, exponential backoff | OpenRouter returns 503; agent process killed |
| **Quality failure** (FAILED equivalent) | Verifier rejects output, wrong format, incomplete work | Yes (auto) | Same or different model, with failure context in prompt | Research report missing 2 of 5 required sections |
| **Human rejection** | Human reviews and rejects | Conditional | Only if human chooses "retry" vs "fail" | Human says "this analysis is wrong, try again" |
| **Budget exhaustion** | Task cost exceeds per-task or per-run budget | No (requires human) | Human must increase budget | Task used $5 of $3 budget |
| **Dependency failure** | Upstream task failed with `all_success` trigger | No | Task skipped | Research task failed → analysis task can't proceed |
| **Cancellation** | Human or system cancels | No | N/A | User abandons mission |

**Key design decision:** Infrastructure failures bypass verification (no point judging output from a crashed agent). Quality failures always go through verification. This matches Prefect's pattern where `CRASHED` bypasses orchestration rules via `force=True`.

### 3.7 Stall Detection

Adapted from Symphony's reconciliation loop and the existing `task_reconciler.py`:

| Detection | Threshold | Action |
|-----------|-----------|--------|
| Task in `running` with no heartbeat/event | `TASK_STALL_TIMEOUT` (default: 5 min) | Transition to `awaiting_retry` (if retries remain) or `failed` |
| Task in `assigned` with no start event | `TASK_ASSIGN_TIMEOUT` (default: 2 min) | Re-queue: transition back to `queued` for reassignment |
| Task in `verifying` with no verdict | `VERIFY_TIMEOUT` (default: 3 min) | Escalate to `awaiting_human` |
| Run in `running` with all tasks terminal but not resolved | `RUN_RESOLVE_TIMEOUT` (default: 1 min) | Coordinator re-evaluates run completion |

**Implementation:** Extend the existing `task_reconciler.py` pattern. The reconciler runs on a tick (via APScheduler, matching the existing heartbeat infrastructure) and queries for stalled entities using the denormalized state column + `updated_at` timestamp. This is the DB-authoritative scheduling pattern from Airflow — the reconciler re-derives "what needs attention" from DB state each tick, with no in-memory state to lose on crash.

```python
# Pseudocode for orchestration reconciler tick
async def reconcile_tick(session: AsyncSession):
    now = utcnow()

    # Stalled running tasks
    stalled = await session.execute(
        select(OrchestrationTask)
        .where(
            OrchestrationTask.state == TaskState.RUNNING,
            OrchestrationTask.updated_at < now - timedelta(seconds=TASK_STALL_TIMEOUT),
        )
        .with_for_update(skip_locked=True)  # skip tasks being processed by another tick
    )
    for task in stalled.scalars():
        if task.attempt_count < task.max_retries:
            await transition_task(session, task, TaskState.AWAITING_RETRY,
                                  reason="stall_detected")
        else:
            await transition_task(session, task, TaskState.FAILED,
                                  reason="stall_detected_max_retries")

    # Ready-to-run tasks (dependencies resolved)
    queued = await session.execute(
        select(OrchestrationTask)
        .where(OrchestrationTask.state == TaskState.QUEUED)
    )
    for task in queued.scalars():
        if await all_dependencies_met(session, task):
            await assign_agent(session, task)
```

### 3.8 Board Task Status Mapping

The existing `board_tasks` table has 5 statuses: `inbox`, `assigned`, `in_progress`, `review`, `done`. Every orchestration task creates a corresponding board_task for UI visibility. The mapping:

| Orchestration Task State | Board Task Status | Notes |
|-------------------------|-------------------|-------|
| `pending` | (no board_task yet) | Board task created on assignment |
| `queued` | (no board_task yet) | Board task created on assignment |
| `awaiting_retry` | `assigned` | Waiting to be re-dispatched |
| `assigned` | `assigned` | Agent selected |
| `running` | `in_progress` | Sets `started_at` |
| `continuing` | `in_progress` | Stays in progress during multi-turn |
| `verifying` | `review` | Output under evaluation |
| `awaiting_human` | `review` | Human decision needed |
| `completed` | `done` | Sets `completed_at` |
| `failed` | `done` | Sets `completed_at` + `error_message` |
| `cancelled` | `done` | Sets `completed_at` + `error_message` |
| `skipped` | `done` | Sets `completed_at` + result = "skipped: dependency failed" |

**Integration mechanism:** Board tasks are linked via `source_type='orchestration'` and `source_id=<orchestration_run_id>` (existing fields on `board_tasks`). The `orchestration_tasks` table holds a `board_task_id` FK for direct reference. State synchronization is performed as a side effect of the `transition_task()` function — every orchestration state change updates the corresponding board_task status in the same transaction.

### 3.9 Concurrency Safety

State transitions must be safe under concurrent access. Two scenarios matter:

1. **Coordinator and agent racing on the same task** — coordinator tries to cancel while agent submits output
2. **Reconciler and agent racing** — reconciler detects stall while agent is about to report completion

**Approach: Optimistic locking with `version_id_col`**

```python
class OrchestrationTask(Base):
    __tablename__ = "orchestration_tasks"

    id = mapped_column(UUID, primary_key=True, server_default=func.gen_random_uuid())
    state = mapped_column(sa.Enum(TaskState), nullable=False, default=TaskState.PENDING)
    version_id = mapped_column(Integer, nullable=False, default=1)

    __mapper_args__ = {"version_id_col": version_id}
```

Every `UPDATE` includes `WHERE version_id = <loaded_value>` and increments the version. If another transaction changed the row, SQLAlchemy raises `StaleDataError`. The transition function catches this and returns a conflict result rather than silently corrupting state.

```python
from sqlalchemy.orm.exc import StaleDataError

async def transition_task(
    session: AsyncSession,
    task: OrchestrationTask,
    to_state: TaskState,
    *,
    reason: str | None = None,
    metadata: dict | None = None,
) -> tuple[bool, OrchestrationTask]:
    """Atomically transition a task state. Returns (success, task)."""
    from_state = task.state

    if to_state not in ALLOWED_TASK_TRANSITIONS.get(from_state, set()):
        raise InvalidTransition(f"{from_state} → {to_state} not allowed")

    if to_state in TERMINAL_TASK_STATES and from_state in TERMINAL_TASK_STATES:
        raise InvalidTransition("Cannot transition between terminal states")

    task.state = to_state
    task.updated_at = utcnow()

    # Side effects
    if to_state == TaskState.RUNNING and task.started_at is None:
        task.started_at = utcnow()
    if TASK_STATE_TYPE[to_state] == StateType.TERMINAL and task.completed_at is None:
        task.completed_at = utcnow()

    # Append event (dual-write)
    event = OrchestrationEvent(
        run_id=task.run_id,
        task_id=task.id,
        event_type=f"task_{to_state}",
        payload={"from": from_state, "to": to_state, "reason": reason, **(metadata or {})},
    )
    session.add(event)

    # Sync board task
    if task.board_task_id:
        board_status = TASK_TO_BOARD_STATUS[to_state]
        await sync_board_task(session, task.board_task_id, board_status, task)

    try:
        await session.flush()
        return True, task
    except StaleDataError:
        await session.rollback()
        return False, await session.get(OrchestrationTask, task.id)
```

**For claim-style operations** (assigning an agent to a queued task), use `SELECT FOR UPDATE SKIP LOCKED` to prevent two coordinators from claiming the same task:

```python
# Claim next queued task for agent
task = await session.execute(
    select(OrchestrationTask)
    .where(OrchestrationTask.state == TaskState.QUEUED)
    .order_by(OrchestrationTask.created_at)
    .limit(1)
    .with_for_update(skip_locked=True)
)
```

### 3.10 Transition Enforcement

**No external library needed.** The transition rules are a ~30-line dict. Python state machine libraries (pytransitions, python-statemachine) don't integrate with SQLAlchemy and would add a dependency for ~10 states. We enforce transitions in application code via the `transition_task()` / `transition_run()` functions. All state changes must go through these functions — never set `.state` directly.

```python
ALLOWED_TASK_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING:        frozenset({TaskState.QUEUED, TaskState.SKIPPED, TaskState.CANCELLED}),
    TaskState.QUEUED:         frozenset({TaskState.ASSIGNED, TaskState.CANCELLED}),
    TaskState.ASSIGNED:       frozenset({TaskState.RUNNING, TaskState.CANCELLED}),
    TaskState.RUNNING:        frozenset({TaskState.CONTINUING, TaskState.VERIFYING,
                                         TaskState.AWAITING_RETRY, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.CONTINUING:     frozenset({TaskState.RUNNING, TaskState.CANCELLED}),
    TaskState.VERIFYING:      frozenset({TaskState.COMPLETED, TaskState.AWAITING_RETRY,
                                         TaskState.AWAITING_HUMAN, TaskState.FAILED}),
    TaskState.AWAITING_HUMAN: frozenset({TaskState.COMPLETED, TaskState.AWAITING_RETRY, TaskState.FAILED}),
    TaskState.AWAITING_RETRY: frozenset({TaskState.ASSIGNED, TaskState.CANCELLED}),
    # Terminal states have no outgoing transitions
    TaskState.COMPLETED:      frozenset(),
    TaskState.FAILED:         frozenset(),
    TaskState.CANCELLED:      frozenset(),
    TaskState.SKIPPED:        frozenset(),
}

ALLOWED_RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.PENDING:            frozenset({RunState.PLANNING, RunState.CANCELLED}),
    RunState.PLANNING:           frozenset({RunState.AWAITING_APPROVAL, RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}),
    RunState.AWAITING_APPROVAL:  frozenset({RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}),
    RunState.RUNNING:            frozenset({RunState.COMPLETED, RunState.FAILED, RunState.PAUSED,
                                            RunState.BUDGET_EXCEEDED, RunState.CANCELLED}),
    RunState.PAUSED:             frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.BUDGET_EXCEEDED:    frozenset({RunState.RUNNING, RunState.CANCELLED}),
    # Terminal states
    RunState.COMPLETED:          frozenset(),
    RunState.FAILED:             frozenset(),
    RunState.CANCELLED:          frozenset(),
}
```

### 3.11 Key Design Decisions Summary

| Decision | Choice | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| State tracking | Hybrid dual-write (CRUD + event log) | Pure event sourcing (Temporal/Dagster), pure CRUD (Airflow) | O(1) queries + audit trail, no projection maintenance overhead. Validated by Prefect at larger scale. |
| State model | Two-level (StateType + StateName) | Flat enum, hierarchical states | Stable orchestration code (4 StateTypes) + extensible display (add states without touching coordinator). Inspired by Prefect's StateType/state_name pattern. |
| Continuation vs retry | Distinct paths with different semantics | Single retry mechanism for both | AI agents frequently need multiple turns (continuation). Conflating this with failure retry causes unnecessary backoff and attempt inflation. Adopted from Symphony. |
| Failure classification | Infrastructure vs quality, separate handling | Single "failed" state | Infrastructure failure → auto-retry same config. Quality failure → retry with different model or escalate to human. Adapted from Prefect's CRASHED vs FAILED. |
| Concurrency control | Optimistic locking (version_id_col) + SELECT FOR UPDATE for claims | Pessimistic locking everywhere, eventual consistency | Low contention (state changes are seconds apart). Optimistic = no lock held during slow operations. Pessimistic only for claim-style dequeuing. |
| Transition enforcement | Application-level dict + function | DB triggers, state machine library | ~100 lines, no dependency, testable, integrated with dual-write and board_task sync. Libraries don't integrate with SQLAlchemy. |
| Board task mapping | Orchestration owns lifecycle, syncs to board_task as side effect | Board task as source of truth, separate UI table | Existing UI gets mission visibility for free. No new frontend work needed for basic mission tracking. |
| Stall detection | DB-authoritative reconciler on tick (extending existing task_reconciler pattern) | In-memory timeouts, heartbeat-only | Crash-safe — reconciler re-derives state from DB each tick. Matches existing infrastructure (APScheduler + task_reconciler.py). |

---

## 4. Data Model: orchestration_runs

The `orchestration_runs` table is the top-level record for every mission. It stores the user's original goal, the coordinator's decomposition plan, execution configuration, and aggregate tracking metrics. One row = one mission attempt.

### 4.1 Design Principles

1. **Denormalized current state** — `state` column for O(1) dashboard queries (dual-write pattern from Section 2.4)
2. **Immutable goal, mutable plan** — the user's original `goal` never changes; the `plan` JSONB evolves during planning
3. **JSONB for extensible config** — autonomy level, budget caps, model preferences stored as structured JSON, not as N columns that require migrations for every new setting
4. **Workspace isolation** — every query must filter by `workspace_id` (FK → `workspaces.id`)
5. **Match existing patterns** — UUID primary key, `server_default=func.now()` timestamps, `ondelete='CASCADE'` for workspace FK (consistent with `board_tasks`, `agent_reports`)

### 4.2 Column Definitions

| Column | Type | Nullable | Default | Constraint | Description |
|--------|------|----------|---------|------------|-------------|
| `id` | `UUID` | No | `gen_random_uuid()` | PK | Stable mission identifier; persists across retries |
| `workspace_id` | `UUID` | No | — | FK → `workspaces.id` ON DELETE CASCADE | Multi-tenant isolation |
| `title` | `VARCHAR(500)` | No | — | — | Human-readable mission title (coordinator-generated or user-provided) |
| `description` | `TEXT` | Yes | `NULL` | — | Optional extended description |
| `goal` | `TEXT` | No | — | — | Original user input, verbatim. Never modified after creation. |
| `state` | `VARCHAR(30)` | No | `'pending'` | — | Current `RunState` value (see Section 3.2). Denormalized for fast queries. |
| `state_type` | `VARCHAR(10)` | No | `'pending'` | — | Current `StateType` value. Stable enum for coordinator logic. |
| `plan` | `JSONB` | Yes | `NULL` | — | Coordinator's decomposition — task list with descriptions, dependencies, agent assignments. Populated during `planning` state. Schema in Section 4.3. |
| `config` | `JSONB` | No | `'{}'` | — | Mission configuration — autonomy level, budget, model preferences, timeout overrides. Schema in Section 4.4. |
| `result_summary` | `TEXT` | Yes | `NULL` | — | Coordinator-generated summary of mission outcome (for completed missions) |
| `error_message` | `TEXT` | Yes | `NULL` | — | Failure reason (for failed/cancelled missions) |
| `created_by` | `VARCHAR(255)` | No | — | — | User ID (Clerk) or `'system'` for auto-triggered missions. String type matches `board_tasks.created_by_id` pattern. |
| `coordinator_agent_id` | `INTEGER` | Yes | `NULL` | FK → `agents.id` ON DELETE SET NULL | Roster agent acting as coordinator, or NULL if using system coordinator |
| `total_tokens` | `INTEGER` | No | `0` | `CHECK (total_tokens >= 0)` | Aggregate token usage across all tasks |
| `total_cost` | `NUMERIC(10,6)` | No | `0` | `CHECK (total_cost >= 0)` | Aggregate cost in USD across all tasks |
| `task_count` | `INTEGER` | No | `0` | `CHECK (task_count >= 0)` | Total tasks in this mission (denormalized for dashboard) |
| `tasks_completed` | `INTEGER` | No | `0` | `CHECK (tasks_completed >= 0)` | Tasks in terminal success state (denormalized) |
| `tasks_failed` | `INTEGER` | No | `0` | `CHECK (tasks_failed >= 0)` | Tasks in terminal failure state (denormalized) |
| `started_at` | `TIMESTAMPTZ` | Yes | `NULL` | — | When first task began execution (state → `running`) |
| `completed_at` | `TIMESTAMPTZ` | Yes | `NULL` | — | When mission reached terminal state |
| `duration_ms` | `INTEGER` | Yes | `NULL` | — | `completed_at - started_at` in milliseconds. Computed on completion. |
| `version_id` | `INTEGER` | No | `1` | — | Optimistic locking counter (SQLAlchemy `version_id_col`) |
| `created_at` | `TIMESTAMPTZ` | No | `NOW()` | — | Row creation timestamp |
| `updated_at` | `TIMESTAMPTZ` | No | `NOW()` | — | Last modification timestamp (auto-updated) |

**Why `NUMERIC(10,6)` for cost?** LLM API calls cost fractions of a cent. `FLOAT` introduces rounding errors on aggregation (`SUM` of 1000 tasks at $0.003 each). `NUMERIC` is exact. 10 digits with 6 decimal places supports up to $9,999.999999 per mission — more than sufficient.

**Why denormalized task counts?** Dashboard queries like "show all running missions with progress" would otherwise require `JOIN + GROUP BY` on potentially large task tables. The coordinator updates these counters atomically when task states change (same transaction as the dual-write event).

### 4.3 Plan JSONB Schema

The `plan` column stores the coordinator's task decomposition. It's populated during the `planning` state and serves as the blueprint for task creation.

```json
{
  "version": 1,
  "strategy": "sequential",
  "reasoning": "The user wants EU AI Act compliance research. This decomposes into 4 sequential phases: research requirements, analyze product, write report, review report.",
  "tasks": [
    {
      "temp_id": "t1",
      "title": "Research EU AI Act requirements",
      "description": "Identify all requirements from the EU AI Act relevant to our product category",
      "task_type": "research",
      "suggested_agent": {
        "type": "roster",
        "agent_id": 42,
        "agent_name": "Researcher"
      },
      "suggested_model": "anthropic/claude-sonnet-4-6",
      "tools_needed": ["web_search", "document_analysis"],
      "depends_on": [],
      "estimated_tokens": 15000,
      "estimated_cost": 0.045,
      "success_criteria": "Comprehensive list of requirements with article references"
    },
    {
      "temp_id": "t2",
      "title": "Analyze product against requirements",
      "description": "Map each EU AI Act requirement to our product's current compliance status",
      "task_type": "analysis",
      "suggested_agent": {
        "type": "roster",
        "agent_id": 42,
        "agent_name": "Researcher"
      },
      "suggested_model": null,
      "tools_needed": ["workspace_read_file", "workspace_grep"],
      "depends_on": ["t1"],
      "estimated_tokens": 20000,
      "estimated_cost": 0.060,
      "success_criteria": "Gap analysis table with compliance status per requirement"
    }
  ],
  "total_estimated_tokens": 75000,
  "total_estimated_cost": 0.225
}
```

**Design notes:**
- `temp_id` is a coordinator-assigned identifier used during planning. Real `orchestration_tasks.id` UUIDs replace these after approval.
- `depends_on` references `temp_id` values (resolved to real task IDs on task creation).
- `suggested_agent` and `suggested_model` are hints — the coordinator may override based on availability or budget.
- `strategy` is informational: `"sequential"`, `"parallel"`, or `"mixed"`. The actual execution order is determined by dependency resolution.
- The plan is **immutable after approval**. Re-planning creates a new version (increment `version`), logged as an event.

### 4.4 Config JSONB Schema

The `config` column stores mission-level settings. Modeled after `workflow_recipes.execution_config` — same pattern of structured JSON for runtime configuration.

```json
{
  "autonomy": {
    "level": "approve",
    "auto_approve_threshold": null
  },
  "budget": {
    "soft_limit_usd": 2.00,
    "hard_limit_usd": 5.00,
    "warn_at_percent": 80
  },
  "model_preferences": {
    "planner": "anthropic/claude-sonnet-4-6",
    "researcher": null,
    "writer": null,
    "reviewer": "anthropic/claude-haiku-4-5-20251001",
    "verifier": "anthropic/claude-haiku-4-5-20251001"
  },
  "timeouts": {
    "task_stall_seconds": 300,
    "task_assign_seconds": 120,
    "verify_seconds": 180,
    "run_max_duration_seconds": 3600
  },
  "retry": {
    "max_retries_per_task": 3,
    "max_continuations_per_task": 10,
    "backoff_base_seconds": 10,
    "backoff_max_seconds": 300
  },
  "notifications": {
    "on_completion": true,
    "on_failure": true,
    "on_budget_warning": true,
    "channel": "slack"
  }
}
```

**Autonomy levels:**
| Level | Behavior |
|-------|----------|
| `"approve"` (default) | Coordinator shows plan → human approves → execution begins |
| `"autonomous"` | Plan auto-approved if estimated cost ≤ `auto_approve_threshold`. Otherwise, falls back to `approve`. |
| `"full_auto"` | No human gates. System runs to completion or budget exhaustion. Requires explicit opt-in. |

**Why JSONB instead of columns?** Config evolves faster than schema. Adding "notification preferences" or "priority scheduling" shouldn't require an Alembic migration. The trade-off is weaker type enforcement at the DB level — mitigated by Pydantic validation on the API layer (same pattern used by `workflow_recipes.execution_config` and `agents.configuration`).

### 4.5 Indexes

```sql
-- Primary query: "show my active missions" (dashboard)
CREATE INDEX ix_orch_runs_workspace_state
    ON orchestration_runs (workspace_id, state_type)
    WHERE state_type != 'terminal';

-- Query: "find mission by ID" (detail view)
-- PK index covers this

-- Query: "recent completed missions" (history)
CREATE INDEX ix_orch_runs_workspace_completed
    ON orchestration_runs (workspace_id, completed_at DESC)
    WHERE state_type = 'terminal';

-- Query: "missions by creator" (user activity)
CREATE INDEX ix_orch_runs_created_by
    ON orchestration_runs (workspace_id, created_by);

-- Query: "stale runs" (reconciler)
CREATE INDEX ix_orch_runs_state_updated
    ON orchestration_runs (state, updated_at)
    WHERE state_type NOT IN ('terminal');
```

**Partial indexes** (`WHERE state_type != 'terminal'`) keep the index small — most runs will be terminal over time. Active runs (the ones queried by dashboards and reconcilers) stay in a compact index.

### 4.6 Example INSERT

```sql
-- User submits: "Research EU AI Act compliance for our product"
INSERT INTO orchestration_runs (
    workspace_id,
    title,
    goal,
    state,
    state_type,
    config,
    created_by
) VALUES (
    '550e8400-e29b-41d4-a716-446655440000',           -- workspace_id
    'EU AI Act Compliance Research',                     -- title (coordinator-generated)
    'Research EU AI Act compliance for our product',     -- goal (user's exact input)
    'pending',                                           -- state
    'pending',                                           -- state_type
    '{
      "autonomy": {"level": "approve"},
      "budget": {"soft_limit_usd": 2.0, "hard_limit_usd": 5.0, "warn_at_percent": 80},
      "model_preferences": {},
      "timeouts": {},
      "retry": {"max_retries_per_task": 3}
    }'::jsonb,                                           -- config (defaults merged with user prefs)
    'user_2abc123'                                       -- created_by (Clerk user ID)
)
RETURNING id, created_at;

-- Returns: id = 'a1b2c3d4-...', created_at = '2026-03-14T22:30:00Z'
-- Next: Coordinator transitions state → 'planning' and begins decomposition
```

### 4.7 Alembic Migration

```python
"""PRD-101: Create orchestration_runs table

Mission-level execution records for the Mission Mode coordinator.
Stores user goals, coordinator plans, execution config, and aggregate metrics.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "prd101_orchestration_runs"
down_revision = None  # Set to latest migration at implementation time
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orchestration_runs",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"),
                  primary_key=True),
        sa.Column("workspace_id", UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("goal", sa.Text, nullable=False),
        sa.Column("state", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("state_type", sa.String(10), nullable=False, server_default="pending"),
        sa.Column("plan", JSONB, nullable=True),
        sa.Column("config", JSONB, nullable=False, server_default="{}"),
        sa.Column("result_summary", sa.Text, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("coordinator_agent_id", sa.Integer, nullable=True),
        sa.Column("total_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_cost", sa.Numeric(10, 6), nullable=False, server_default="0"),
        sa.Column("task_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tasks_completed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tasks_failed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("version_id", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        # Constraints
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["coordinator_agent_id"], ["agents.id"], ondelete="SET NULL"),
        sa.CheckConstraint("total_tokens >= 0", name="ck_orch_runs_tokens_positive"),
        sa.CheckConstraint("total_cost >= 0", name="ck_orch_runs_cost_positive"),
        sa.CheckConstraint("task_count >= 0", name="ck_orch_runs_task_count_positive"),
        sa.CheckConstraint("tasks_completed >= 0", name="ck_orch_runs_tasks_completed_positive"),
        sa.CheckConstraint("tasks_failed >= 0", name="ck_orch_runs_tasks_failed_positive"),
    )

    # Indexes
    op.create_index(
        "ix_orch_runs_workspace_state",
        "orchestration_runs",
        ["workspace_id", "state_type"],
        postgresql_where=sa.text("state_type != 'terminal'"),
    )
    op.create_index(
        "ix_orch_runs_workspace_completed",
        "orchestration_runs",
        ["workspace_id", sa.text("completed_at DESC")],
        postgresql_where=sa.text("state_type = 'terminal'"),
    )
    op.create_index(
        "ix_orch_runs_created_by",
        "orchestration_runs",
        ["workspace_id", "created_by"],
    )
    op.create_index(
        "ix_orch_runs_state_updated",
        "orchestration_runs",
        ["state", "updated_at"],
        postgresql_where=sa.text("state_type != 'terminal'"),
    )

    # Table comment
    op.execute(
        "COMMENT ON TABLE orchestration_runs IS "
        "'Mission-level execution records (PRD-101). One row per mission attempt.'"
    )


def downgrade() -> None:
    op.drop_index("ix_orch_runs_state_updated", table_name="orchestration_runs")
    op.drop_index("ix_orch_runs_created_by", table_name="orchestration_runs")
    op.drop_index("ix_orch_runs_workspace_completed", table_name="orchestration_runs")
    op.drop_index("ix_orch_runs_workspace_state", table_name="orchestration_runs")
    op.drop_table("orchestration_runs")
```

### 4.8 Design Decisions

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Primary key type | UUID | Integer (SERIAL) | Missions may be created from multiple sources (API, chatbot, scheduler). UUID avoids coordination for ID generation. Matches `workspaces.id` and `agent_reports.id` patterns. |
| Goal storage | Immutable `TEXT` column | Part of `config` JSONB | The goal is the user's contract with the system. It should never be buried in JSON or accidentally modified. Separate column enables full-text search. |
| Plan storage | JSONB column on runs table | Separate `orchestration_plans` table | Plans are 1:1 with runs and always loaded together. A separate table adds a JOIN for every plan read with no queryability benefit (we never query across plans). |
| Cost tracking | `NUMERIC(10,6)` | `FLOAT`, `INTEGER` (cents) | `FLOAT` accumulates rounding errors. Integer cents loses sub-cent precision (common in LLM billing). `NUMERIC` is exact and supports both. |
| Denormalized counters | `task_count`, `tasks_completed`, `tasks_failed` | Computed via `COUNT(*)` query | Dashboard shows "3/7 tasks complete" — this query runs on every page load. Denormalized counters avoid a JOIN + GROUP BY on every render. Updated atomically with task state transitions. |
| Config extensibility | JSONB with Pydantic validation | Typed columns | Config changes (new autonomy levels, new notification types) shouldn't require migrations. Same pattern as `agents.configuration` and `workflow_recipes.execution_config`. |
| `created_by` type | `VARCHAR(255)` | `INTEGER` FK → `users.id` | Matches `board_tasks.created_by_id` pattern. Clerk user IDs are strings (`user_2abc...`). Supports `'system'` for auto-triggered missions without a nullable FK. |
| `coordinator_agent_id` | Nullable FK → `agents.id` | Required FK, separate `coordinator_type` column | Most missions use the system coordinator (no specific agent). When a roster agent coordinates, reference it. `SET NULL` on agent deletion — the run record survives. |

---

## 5. Data Model: orchestration_tasks

The `orchestration_tasks` table records every subtask within a mission. Each row tracks assignment, execution, verification, and result for a single unit of work. Tasks reference their parent run, their assigned agent, and their corresponding board task (for UI visibility). Dependencies between tasks are stored in a separate join table (`orchestration_task_dependencies`) — not as an array column — for referential integrity and clean scheduling queries.

### 5.1 Design Principles

1. **Join table for dependencies** — PostgreSQL's own documentation warns that "searching for specific array elements can be a sign of database misdesign" and recommends a separate table. A `task_dependencies` join table gives us FK enforcement, B-tree indexes for both directions (upstream/downstream), and trivial addition of edge metadata (`dependency_type`). At our scale (5-50 tasks per mission), the extra JOIN is negligible — and the "find ready tasks" query is cleaner than `unnest()` or `jsonb_array_elements()`.
2. **Board task bridge** — every `orchestration_task` creates a `board_task` with `source_type='orchestration'` and `source_id` set to the run ID. This gives us free dashboard visibility without new UI components. The `board_task_id` FK on `orchestration_tasks` links back for updates.
3. **Two-level state** — `state` (rich display) and `state_type` (stable orchestration logic) from Section 3.2, same pattern as `orchestration_runs`.
4. **Continuation vs retry** — `attempt_number` tracks retry attempts (backoff, fresh start). `continuation_count` tracks clean continuation turns (same attempt, 1s delay). Both capped by `config.retry` on the parent run.
5. **Output stored externally** — large task outputs go to `output_ref` (workspace file path or report ID), not inline. Only `output_summary` (≤2000 chars) is stored on the row for dashboard display. This follows the pattern all 5 studied systems use — none store large outputs on the task row.
6. **Match existing conventions** — UUID primary key (matches `orchestration_runs`), `NUMERIC(10,6)` for cost, `TIMESTAMPTZ` timestamps with `server_default`, optimistic locking via `version_id`.

### 5.2 Column Definitions

| Column | Type | Nullable | Default | Constraint | Description |
|--------|------|----------|---------|------------|-------------|
| `id` | `UUID` | No | `gen_random_uuid()` | PK | Stable task identifier |
| `run_id` | `UUID` | No | — | FK → `orchestration_runs.id` ON DELETE CASCADE | Parent mission |
| `workspace_id` | `UUID` | No | — | FK → `workspaces.id` ON DELETE CASCADE | Denormalized for query efficiency (avoids JOIN to runs for workspace filtering) |
| `sequence_number` | `SMALLINT` | No | — | — | Position in plan order (1-based). Stable after planning — used for display, not execution order. |
| `title` | `VARCHAR(500)` | No | — | — | Human-readable task title (from coordinator plan) |
| `description` | `TEXT` | Yes | `NULL` | — | Detailed task description / instructions for the agent |
| `task_type` | `VARCHAR(30)` | No | — | — | `TaskType` enum: `research`, `analysis`, `writing`, `coding`, `verification`, `review`, `synthesis`, `other` |
| `state` | `VARCHAR(30)` | No | `'pending'` | — | Current `TaskState` value (Section 3.2) |
| `state_type` | `VARCHAR(10)` | No | `'pending'` | — | Current `StateType` value. Stable enum for coordinator logic. |
| `trigger_rule` | `VARCHAR(30)` | No | `'all_success'` | — | When this task becomes ready. Values: `all_success` (default), `all_done`, `none_failed`, `always`. Inspired by Airflow's trigger rules — we adopt the 4 most relevant for agent orchestration. |
| `agent_id` | `INTEGER` | Yes | `NULL` | FK → `agents.id` ON DELETE SET NULL | Assigned roster agent. NULL if contractor or unassigned. |
| `agent_type` | `VARCHAR(20)` | Yes | `NULL` | — | `roster` (permanent agent) or `contractor` (ephemeral, mission-scoped). NULL when unassigned. |
| `model_override` | `VARCHAR(255)` | Yes | `NULL` | — | LLM model override for this task. NULL = use run-level `config.model_preferences` or agent default. |
| `tools_requested` | `JSONB` | Yes | `NULL` | — | Array of tool names the coordinator wants available for this task. Hint, not enforcement — agent's assigned tools take precedence. |
| `success_criteria` | `TEXT` | Yes | `NULL` | — | Plain-text description of what constitutes success. Used by the verifier (PRD-103). |
| `output_summary` | `VARCHAR(2000)` | Yes | `NULL` | — | Truncated output for dashboard display. Written by agent or coordinator on completion. |
| `output_ref` | `VARCHAR(500)` | Yes | `NULL` | — | Reference to full output: workspace file path (`/reports/{agent}/{slug}.md`) or `agent_reports.id`. |
| `verifier_score` | `NUMERIC(3,2)` | Yes | `NULL` | `CHECK (verifier_score >= 0 AND verifier_score <= 1)` | Verification quality score (0.00–1.00). Written by verifier agent (PRD-103). |
| `verified_by` | `VARCHAR(255)` | Yes | `NULL` | — | Who verified: agent ID, `'human'`, or `'auto'`. |
| `error_message` | `TEXT` | Yes | `NULL` | — | Failure reason (for failed/cancelled tasks) |
| `attempt_number` | `SMALLINT` | No | `1` | `CHECK (attempt_number >= 1)` | Current retry attempt (incremented on retry, not on continuation) |
| `continuation_count` | `SMALLINT` | No | `0` | `CHECK (continuation_count >= 0)` | Number of continuation turns within current attempt |
| `tokens_used` | `INTEGER` | No | `0` | `CHECK (tokens_used >= 0)` | Total tokens consumed across all attempts |
| `cost` | `NUMERIC(10,6)` | No | `0` | `CHECK (cost >= 0)` | Total cost in USD across all attempts |
| `board_task_id` | `INTEGER` | Yes | `NULL` | FK → `board_tasks.id` ON DELETE SET NULL | Corresponding board task for UI visibility. Created when task is planned. |
| `started_at` | `TIMESTAMPTZ` | Yes | `NULL` | — | When agent began execution (state → `running`) |
| `completed_at` | `TIMESTAMPTZ` | Yes | `NULL` | — | When task reached terminal state |
| `duration_ms` | `INTEGER` | Yes | `NULL` | — | `completed_at - started_at` in milliseconds |
| `version_id` | `INTEGER` | No | `1` | — | Optimistic locking counter (SQLAlchemy `version_id_col`) |
| `created_at` | `TIMESTAMPTZ` | No | `NOW()` | — | Row creation timestamp |
| `updated_at` | `TIMESTAMPTZ` | No | `NOW()` | — | Last modification timestamp |

**Why denormalize `workspace_id`?** The dashboard query "show all tasks for my workspace" would otherwise require a JOIN to `orchestration_runs`. Since workspace_id never changes for a task, denormalizing avoids the JOIN on every task list render.

**Why `SMALLINT` for attempt/continuation?** A task that retries 255+ times or continues 65535+ turns has a bug, not a workload. `SMALLINT` (2 bytes, max 32767) is more than sufficient and saves 2 bytes per row vs `INTEGER`.

**Why `NUMERIC(3,2)` for verifier_score?** Scores are 0.00 to 1.00. `NUMERIC(3,2)` stores exactly two decimal places with no floating-point rounding. `FLOAT` would work but invites `0.6999...` display issues.

### 5.3 Trigger Rules

Inspired by Airflow's trigger rule system (13 rules), we adopt the 4 most relevant for LLM agent orchestration. Each rule defines when a task's dependencies are considered "met" and the task can transition from `pending` to `queued`.

| Rule | Semantics | Use Case |
|------|-----------|----------|
| `all_success` (default) | All upstream tasks must be in `completed` state | Standard pipeline: next agent runs only after all prerequisites succeed |
| `all_done` | All upstream tasks must be in any terminal state (`completed`, `failed`, `cancelled`, `skipped`) | Join/aggregation nodes that collect results regardless of individual success |
| `none_failed` | All upstream tasks must be terminal AND none may be `failed` (skipped/cancelled are OK) | Parallel fan-out where some branches are optional but hard failures should block |
| `always` | Skip dependency evaluation entirely — task is immediately `queued` when created | Cleanup, notification, or cost-tracking tasks that must always run |

**Why only 4 rules?** Airflow's `ONE_SUCCESS`, `ONE_FAILED`, `ALL_FAILED`, etc. are designed for complex ETL branching with thousands of tasks. Our missions have 5-50 tasks with human oversight. Four rules cover every pattern we need:
- Sequential pipeline → `all_success`
- Parallel research with synthesis → `all_success` on the synthesis task
- Error-tolerant aggregation → `all_done`
- Optional branches → `none_failed`
- Guaranteed cleanup → `always`

If we discover a need for `one_success` (race pattern) or others, adding them requires only a new enum value and a case in the trigger rule evaluator — no schema change.

### 5.4 Task Dependencies (Join Table)

```sql
CREATE TABLE orchestration_task_dependencies (
    task_id          UUID NOT NULL,
    depends_on_id    UUID NOT NULL,
    dependency_type  VARCHAR(20) NOT NULL DEFAULT 'data',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (task_id, depends_on_id),
    FOREIGN KEY (task_id) REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_id) REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
    CHECK (task_id != depends_on_id)
);

CREATE INDEX ix_task_deps_depends_on ON orchestration_task_dependencies (depends_on_id);
```

**Dependency types:**

| Type | Semantics |
|------|-----------|
| `data` (default) | Downstream task consumes upstream task's output. The coordinator passes the output reference to the downstream agent's context. |
| `ordering` | Downstream task must wait for upstream to complete, but does not consume its output. Used for side-effect ordering (e.g., "write to DB before reading from DB"). |

**Why a join table instead of an array column?**

| Criterion | Join Table (chosen) | `UUID[]` Array | JSONB Array |
|-----------|-------------------|----------------|-------------|
| FK enforcement | ✅ DB-enforced | ❌ None | ❌ None |
| Self-referencing cycle prevention | ✅ `CHECK (task_id != depends_on_id)` | ❌ App-only | ❌ App-only |
| Edge metadata | ✅ Add columns | ❌ Requires schema change | ⚠️ Add JSON keys |
| "Find ready tasks" query | ✅ Standard `NOT EXISTS` + JOIN | ⚠️ `unnest()` + JOIN | ⚠️ `jsonb_array_elements()` + JOIN |
| Index type | B-tree (cheap) | GIN (overkill at scale) | GIN (overkill at scale) |
| "What blocks task X?" | ✅ Single index lookup | ❌ Full-table scan | ❌ Full-table scan |
| "What does task X block?" | ✅ Single index lookup | ⚠️ `ANY()` scan | ⚠️ `jsonb_path_query()` |
| PostgreSQL recommendation | ✅ Preferred | ❌ "Sign of misdesign" | ❌ Not for relational edges |

**Cycle detection** happens at planning time in Python (via `graphlib.TopologicalSorter.prepare()`) before rows are inserted. The `CHECK (task_id != depends_on_id)` constraint catches self-references at the DB level; multi-node cycles are caught by the topological sort.

### 5.5 Dependency Resolution Algorithm

We use Python's `graphlib.TopologicalSorter` (stdlib since 3.9), which implements Kahn's algorithm internally with incremental update support.

**At planning time — validate the DAG:**

```python
from graphlib import TopologicalSorter, CycleError

def validate_task_graph(tasks: list[dict]) -> list[str]:
    """
    Validate that the coordinator's plan has no circular dependencies.
    Returns topological order or raises ValueError with cycle path.

    tasks: list of {"temp_id": "t1", "depends_on": ["t0"], ...} from plan JSONB
    """
    ts = TopologicalSorter()
    for task in tasks:
        ts.add(task["temp_id"], *task.get("depends_on", []))

    try:
        ts.prepare()
    except CycleError as e:
        cycle_path = " → ".join(e.args[1])
        raise ValueError(f"Circular dependency detected: {cycle_path}")

    return list(ts.static_order())
```

**At runtime — find ready tasks and react to completions:**

```python
from graphlib import TopologicalSorter

class DependencyResolver:
    """
    Tracks task completion and determines which tasks are ready to execute.
    Initialized from DB state on coordinator startup — crash-safe.
    """

    def __init__(self, tasks: dict[str, list[str]], completed: set[str]):
        """
        tasks: {task_id: [dependency_task_id, ...]} — from DB
        completed: set of task_ids already in terminal state — from DB
        """
        self._ts = TopologicalSorter()
        for task_id, deps in tasks.items():
            if task_id not in completed:
                remaining_deps = [d for d in deps if d not in completed]
                self._ts.add(task_id, *remaining_deps)
        self._ts.prepare()  # safe — cycles validated at planning time

    def get_ready(self) -> tuple[str, ...]:
        """Return task IDs with all dependencies met."""
        return self._ts.get_ready()

    def mark_done(self, task_id: str) -> tuple[str, ...]:
        """Mark task complete. Returns newly-unblocked task IDs."""
        self._ts.done(task_id)
        return self._ts.get_ready()

    def is_complete(self) -> bool:
        """True when all tasks are done."""
        return not self._ts.is_active()
```

**Crash safety:** The resolver is reconstructed from DB state on every coordinator tick (same pattern as Airflow's DB-authoritative scheduling). No in-memory state survives across restarts. The coordinator queries:

```sql
-- Load task graph from DB
SELECT t.id, array_agg(d.depends_on_id) AS deps
FROM orchestration_tasks t
LEFT JOIN orchestration_task_dependencies d ON d.task_id = t.id
WHERE t.run_id = $1
GROUP BY t.id;
```

Then builds the resolver with completed tasks excluded. This is O(N) where N = number of tasks in the mission (5-50). Rebuilding from scratch on every tick is trivially fast at this scale.

**Edge cases:**

| Scenario | Behavior |
|----------|----------|
| Empty mission (0 tasks) | `get_ready()` returns empty tuple. `is_complete()` returns True immediately. |
| Single task, no deps | Task is immediately ready. |
| Fully parallel (no deps) | All tasks returned by first `get_ready()`. |
| Diamond (A→B, A→C, B→D, C→D) | A first, then B+C in parallel, then D after both complete. |
| Self-reference | Caught by DB constraint `CHECK (task_id != depends_on_id)`. |
| Multi-node cycle | Caught by `TopologicalSorter.prepare()` at planning time. |

### 5.6 Trigger Rule Evaluation

The "find ready tasks" query combines dependency resolution with trigger rule evaluation. For `all_success` (the default and most common), a task is ready when all its upstream dependencies have `state = 'completed'`. Other rules evaluate different terminal state combinations.

```sql
-- Find tasks ready to execute in a mission
-- This is the coordinator's core scheduling query
SELECT t.id, t.title, t.trigger_rule
FROM orchestration_tasks t
WHERE t.run_id = $1
  AND t.state = 'pending'
  AND (
    -- Rule: ALWAYS — skip dependency check
    t.trigger_rule = 'always'
    OR
    -- No dependencies at all
    NOT EXISTS (
      SELECT 1 FROM orchestration_task_dependencies d WHERE d.task_id = t.id
    )
    OR
    -- Has dependencies — evaluate trigger rule
    CASE t.trigger_rule
      -- ALL_SUCCESS: every upstream must be 'completed'
      WHEN 'all_success' THEN NOT EXISTS (
        SELECT 1
        FROM orchestration_task_dependencies d
        JOIN orchestration_tasks dep ON dep.id = d.depends_on_id
        WHERE d.task_id = t.id
          AND dep.state != 'completed'
      )
      -- ALL_DONE: every upstream must be in any terminal state
      WHEN 'all_done' THEN NOT EXISTS (
        SELECT 1
        FROM orchestration_task_dependencies d
        JOIN orchestration_tasks dep ON dep.id = d.depends_on_id
        WHERE d.task_id = t.id
          AND dep.state_type != 'terminal'
      )
      -- NONE_FAILED: all upstream terminal, none 'failed'
      WHEN 'none_failed' THEN NOT EXISTS (
        SELECT 1
        FROM orchestration_task_dependencies d
        JOIN orchestration_tasks dep ON dep.id = d.depends_on_id
        WHERE d.task_id = t.id
          AND (dep.state_type != 'terminal' OR dep.state = 'failed')
      )
      ELSE FALSE  -- unknown rule, don't schedule
    END
  );
```

**In practice, the coordinator uses the Python `DependencyResolver` (Section 5.5) rather than this SQL for the common `all_success` case.** The SQL version is provided for:
- Reconciler/stall detection (runs on APScheduler, independent of coordinator)
- Debugging ("why isn't this task running?")
- Dashboard queries ("show me blocked tasks")

**Cascade states:** When a task fails and downstream tasks have `trigger_rule = 'all_success'`, the coordinator cascades them to `skipped` state. This is done in Python (loop over downstream tasks, check trigger rule, set state + emit event) rather than as a DB trigger — keeping side effects explicit and debuggable.

### 5.7 Indexes

```sql
-- Primary query: "tasks in this mission" (mission detail view)
CREATE INDEX ix_orch_tasks_run_id
    ON orchestration_tasks (run_id);

-- Query: "pending tasks in this mission" (coordinator scheduling)
CREATE INDEX ix_orch_tasks_run_state
    ON orchestration_tasks (run_id, state_type)
    WHERE state_type != 'terminal';

-- Query: "tasks assigned to this agent" (agent workload view)
CREATE INDEX ix_orch_tasks_agent
    ON orchestration_tasks (agent_id, state)
    WHERE agent_id IS NOT NULL;

-- Query: "tasks for this workspace" (dashboard)
CREATE INDEX ix_orch_tasks_workspace
    ON orchestration_tasks (workspace_id, state_type);

-- Query: "board task link" (reverse lookup from board UI)
CREATE INDEX ix_orch_tasks_board_task
    ON orchestration_tasks (board_task_id)
    WHERE board_task_id IS NOT NULL;

-- Query: "stale tasks" (reconciler)
CREATE INDEX ix_orch_tasks_state_updated
    ON orchestration_tasks (state, updated_at)
    WHERE state_type NOT IN ('terminal', 'pending');
```

**Partial indexes** on active states keep indexes compact. Terminal tasks accumulate over time but are rarely queried for scheduling.

### 5.8 Example: Creating Tasks from a Plan

After the coordinator generates a plan and the human approves, tasks are created in a single transaction:

```sql
-- Transaction: create tasks + dependencies + board tasks for a mission

-- 1. Create orchestration tasks
INSERT INTO orchestration_tasks (
    id, run_id, workspace_id, sequence_number, title, description,
    task_type, state, state_type, trigger_rule,
    agent_id, agent_type, model_override, tools_requested, success_criteria
) VALUES
    -- Task 1: Research (no dependencies)
    ('11111111-0000-0000-0000-000000000001',
     'aaaaaaaa-0000-0000-0000-000000000001',
     '550e8400-e29b-41d4-a716-446655440000',
     1, 'Research EU AI Act requirements',
     'Identify all requirements from the EU AI Act relevant to our product category',
     'research', 'pending', 'pending', 'all_success',
     42, 'roster', NULL, '["web_search", "document_analysis"]',
     'Comprehensive list of requirements with article references'),
    -- Task 2: Analysis (depends on Task 1)
    ('11111111-0000-0000-0000-000000000002',
     'aaaaaaaa-0000-0000-0000-000000000001',
     '550e8400-e29b-41d4-a716-446655440000',
     2, 'Analyze product against requirements',
     'Map each EU AI Act requirement to our product compliance status',
     'analysis', 'pending', 'pending', 'all_success',
     42, 'roster', NULL, '["workspace_read_file", "workspace_grep"]',
     'Gap analysis table with compliance status per requirement'),
    -- Task 3: Write report (depends on Tasks 1 + 2)
    ('11111111-0000-0000-0000-000000000003',
     'aaaaaaaa-0000-0000-0000-000000000001',
     '550e8400-e29b-41d4-a716-446655440000',
     3, 'Write compliance report',
     'Synthesize research and analysis into a compliance report',
     'writing', 'pending', 'pending', 'all_success',
     NULL, 'contractor', 'anthropic/claude-sonnet-4-6', NULL,
     'Professional report covering all identified requirements'),
    -- Task 4: Review (depends on Task 3, always runs)
    ('11111111-0000-0000-0000-000000000004',
     'aaaaaaaa-0000-0000-0000-000000000001',
     '550e8400-e29b-41d4-a716-446655440000',
     4, 'Review and score report',
     'Verify report quality, completeness, and accuracy',
     'verification', 'pending', 'pending', 'all_success',
     NULL, 'contractor', 'anthropic/claude-haiku-4-5-20251001', NULL,
     'Score ≥ 0.7 on quality, completeness, and accuracy dimensions');

-- 2. Create dependencies
INSERT INTO orchestration_task_dependencies (task_id, depends_on_id, dependency_type) VALUES
    ('11111111-0000-0000-0000-000000000002', '11111111-0000-0000-0000-000000000001', 'data'),
    ('11111111-0000-0000-0000-000000000003', '11111111-0000-0000-0000-000000000001', 'data'),
    ('11111111-0000-0000-0000-000000000003', '11111111-0000-0000-0000-000000000002', 'data'),
    ('11111111-0000-0000-0000-000000000004', '11111111-0000-0000-0000-000000000003', 'data');

-- 3. Create board tasks (for UI visibility)
INSERT INTO board_tasks (
    workspace_id, title, description, status, priority,
    source_type, source_id, assigned_agent_id, created_by_type, created_by_id
) VALUES
    ('550e8400-e29b-41d4-a716-446655440000',
     'Research EU AI Act requirements',
     'Identify all requirements from the EU AI Act relevant to our product category',
     'inbox', 'medium', 'orchestration',
     'aaaaaaaa-0000-0000-0000-000000000001',
     42, 'orchestration', 'system')
RETURNING id;
-- Link board_task.id back to orchestration_task.board_task_id

-- 4. Update run task counts
UPDATE orchestration_runs
SET task_count = 4, state = 'running', state_type = 'running',
    started_at = NOW()
WHERE id = 'aaaaaaaa-0000-0000-0000-000000000001';
```

### 5.9 Board Task Mapping

Every `orchestration_task` creates a corresponding `board_task` for UI visibility. The mapping:

| orchestration_tasks field | board_tasks field | Notes |
|---------------------------|-------------------|-------|
| `title` | `title` | Direct copy |
| `description` | `description` | Direct copy |
| `workspace_id` | `workspace_id` | Same FK |
| `agent_id` | `assigned_agent_id` | Roster agent; NULL for contractors |
| — | `source_type` | Always `'orchestration'` |
| `run_id` (as string) | `source_id` | Links board task back to mission |
| — | `created_by_type` | `'orchestration'` |
| — | `created_by_id` | `'system'` |
| `state` → mapped | `status` | See mapping below |

**State → Board Status mapping:**

| orchestration_task state | board_task status |
|--------------------------|-------------------|
| `pending`, `queued`, `awaiting_retry` | `inbox` |
| `assigned` | `assigned` |
| `running`, `continuing` | `in_progress` |
| `verifying`, `awaiting_human` | `review` |
| `completed` | `done` |
| `failed`, `cancelled`, `skipped` | `done` (with `error_message` set) |

The coordinator updates the board task status atomically with the orchestration task state change (same transaction as the dual-write event pattern from Section 3.1).

### 5.10 Alembic Migration

```python
"""PRD-101: Create orchestration_tasks and orchestration_task_dependencies tables

Task-level execution records and dependency edges for Mission Mode.
Each task tracks assignment, execution, verification, and links to board_tasks for UI.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "prd101_orchestration_tasks"
down_revision = "prd101_orchestration_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orchestration_tasks",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"),
                  primary_key=True),
        sa.Column("run_id", UUID(as_uuid=True), nullable=False),
        sa.Column("workspace_id", UUID(as_uuid=True), nullable=False),
        sa.Column("sequence_number", sa.SmallInteger, nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("task_type", sa.String(30), nullable=False),
        sa.Column("state", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("state_type", sa.String(10), nullable=False, server_default="pending"),
        sa.Column("trigger_rule", sa.String(30), nullable=False, server_default="all_success"),
        sa.Column("agent_id", sa.Integer, nullable=True),
        sa.Column("agent_type", sa.String(20), nullable=True),
        sa.Column("model_override", sa.String(255), nullable=True),
        sa.Column("tools_requested", JSONB, nullable=True),
        sa.Column("success_criteria", sa.Text, nullable=True),
        sa.Column("output_summary", sa.String(2000), nullable=True),
        sa.Column("output_ref", sa.String(500), nullable=True),
        sa.Column("verifier_score", sa.Numeric(3, 2), nullable=True),
        sa.Column("verified_by", sa.String(255), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("attempt_number", sa.SmallInteger, nullable=False, server_default="1"),
        sa.Column("continuation_count", sa.SmallInteger, nullable=False, server_default="0"),
        sa.Column("tokens_used", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost", sa.Numeric(10, 6), nullable=False, server_default="0"),
        sa.Column("board_task_id", sa.Integer, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("version_id", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        # Foreign keys
        sa.ForeignKeyConstraint(["run_id"], ["orchestration_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["board_task_id"], ["board_tasks.id"], ondelete="SET NULL"),
        # Constraints
        sa.CheckConstraint("verifier_score >= 0 AND verifier_score <= 1",
                          name="ck_orch_tasks_score_range"),
        sa.CheckConstraint("attempt_number >= 1", name="ck_orch_tasks_attempt_positive"),
        sa.CheckConstraint("continuation_count >= 0", name="ck_orch_tasks_continuation_positive"),
        sa.CheckConstraint("tokens_used >= 0", name="ck_orch_tasks_tokens_positive"),
        sa.CheckConstraint("cost >= 0", name="ck_orch_tasks_cost_positive"),
    )

    # Indexes
    op.create_index("ix_orch_tasks_run_id", "orchestration_tasks", ["run_id"])
    op.create_index(
        "ix_orch_tasks_run_state", "orchestration_tasks",
        ["run_id", "state_type"],
        postgresql_where=sa.text("state_type != 'terminal'"),
    )
    op.create_index(
        "ix_orch_tasks_agent", "orchestration_tasks",
        ["agent_id", "state"],
        postgresql_where=sa.text("agent_id IS NOT NULL"),
    )
    op.create_index("ix_orch_tasks_workspace", "orchestration_tasks",
                    ["workspace_id", "state_type"])
    op.create_index(
        "ix_orch_tasks_board_task", "orchestration_tasks",
        ["board_task_id"],
        postgresql_where=sa.text("board_task_id IS NOT NULL"),
    )
    op.create_index(
        "ix_orch_tasks_state_updated", "orchestration_tasks",
        ["state", "updated_at"],
        postgresql_where=sa.text("state_type NOT IN ('terminal', 'pending')"),
    )

    # Dependencies join table
    op.create_table(
        "orchestration_task_dependencies",
        sa.Column("task_id", UUID(as_uuid=True), nullable=False),
        sa.Column("depends_on_id", UUID(as_uuid=True), nullable=False),
        sa.Column("dependency_type", sa.String(20), nullable=False, server_default="data"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        # Composite PK
        sa.PrimaryKeyConstraint("task_id", "depends_on_id"),
        # FKs
        sa.ForeignKeyConstraint(["task_id"], ["orchestration_tasks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["depends_on_id"], ["orchestration_tasks.id"],
                               ondelete="CASCADE"),
        # Prevent self-reference
        sa.CheckConstraint("task_id != depends_on_id", name="ck_task_deps_no_self_ref"),
    )
    op.create_index("ix_task_deps_depends_on", "orchestration_task_dependencies",
                    ["depends_on_id"])

    # Table comments
    op.execute(
        "COMMENT ON TABLE orchestration_tasks IS "
        "'Task-level execution records within a mission (PRD-101).'"
    )
    op.execute(
        "COMMENT ON TABLE orchestration_task_dependencies IS "
        "'DAG edges between orchestration tasks. Join table for dependency resolution (PRD-101).'"
    )


def downgrade() -> None:
    op.drop_index("ix_task_deps_depends_on", table_name="orchestration_task_dependencies")
    op.drop_table("orchestration_task_dependencies")
    op.drop_index("ix_orch_tasks_state_updated", table_name="orchestration_tasks")
    op.drop_index("ix_orch_tasks_board_task", table_name="orchestration_tasks")
    op.drop_index("ix_orch_tasks_workspace", table_name="orchestration_tasks")
    op.drop_index("ix_orch_tasks_agent", table_name="orchestration_tasks")
    op.drop_index("ix_orch_tasks_run_state", table_name="orchestration_tasks")
    op.drop_index("ix_orch_tasks_run_id", table_name="orchestration_tasks")
    op.drop_table("orchestration_tasks")
```

### 5.11 Design Decisions

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Dependency storage | Join table (`orchestration_task_dependencies`) | `UUID[]` array column, JSONB array | PostgreSQL docs recommend against arrays for relationship storage. Join table gives FK enforcement, B-tree indexes, edge metadata, and cleaner queries. At 5-50 tasks per mission, the extra JOIN is free. |
| Dependency direction | `task_id` depends on `depends_on_id` | Reverse (downstream_id, upstream_id) | "This task depends on that task" reads naturally. Matches `graphlib.TopologicalSorter.add(node, *predecessors)` convention. |
| Trigger rules | 4 rules (all_success, all_done, none_failed, always) | Full Airflow set (13 rules) | 4 rules cover all agent orchestration patterns. Adding more is a code change, not a schema change. YAGNI. |
| Output storage | `output_summary` (2000 chars) + `output_ref` (path) | Full output as TEXT column | All 5 studied systems avoid inline output storage. Large agent outputs (reports, code) go to workspace files or reports table. Summary is for dashboard display only. |
| Board task integration | FK `board_task_id` on orchestration_tasks | Reverse FK on board_tasks | orchestration_tasks owns the relationship. Board tasks are created first, then linked. `SET NULL` on board task deletion preserves orchestration history. |
| Workspace_id denormalization | Denormalized on tasks | JOIN to runs | Avoids JOIN on every workspace-filtered task query. workspace_id is immutable — no consistency risk. |
| Cycle detection | Python `graphlib.TopologicalSorter` at planning time | DB trigger, recursive CTE | Cycles are a planning error, not a runtime condition. Detecting at planning time with a clear error message ("circular dependency: A → B → C → A") is better UX than a DB constraint error. |
| Dependency resolution at runtime | Python `DependencyResolver` rebuilt from DB state | Pure SQL query | Python resolver uses `graphlib` incremental updates (O(out-degree) per completion). SQL query is provided for reconciler/debugging. Both derive from the same DB state — crash-safe. |
| `attempt_number` vs `continuation_count` | Separate columns | Single `attempts` counter | Continuation (clean exit, 1s delay, same workspace) and retry (failure, backoff, fresh start) are fundamentally different operations (Symphony research). Conflating them makes it impossible to distinguish "agent needed 5 turns" from "agent failed 5 times." |
| Verifier score type | `NUMERIC(3,2)` | `FLOAT`, `INTEGER` (0-100) | Consistent with `NUMERIC(10,6)` for cost. Exact decimal arithmetic. 0.00–1.00 is a standard scoring range that avoids the "is 7/10 good or bad?" ambiguity of integer scales. |

---

## 6. Event Log & Audit Trail

The `orchestration_events` table is the append-only audit trail for every state change, decision, and notable occurrence in a mission's lifecycle. It is the second half of the **dual-write pattern** established in Section 2.4: every state transition writes the new state to the entity row (fast queries) AND appends an immutable event (complete history). Both writes occur in a single database transaction.

### 6.1 Design Philosophy

**Lightweight event sourcing, not full event sourcing.** We log events for observability, debugging, and telemetry — not for state reconstruction. The `orchestration_runs` and `orchestration_tasks` tables hold the authoritative current state. Events answer "what happened and when?" without being the source of truth for "what is the current state?"

This is the same trade-off Prefect makes with its `flow_run_state` / `task_run_state` history tables alongside denormalized state on the run row. It avoids the projection maintenance and snapshot management overhead of full event sourcing (Dagster/Temporal) while giving us a complete audit trail that pure CRUD (Airflow's mutable `task_instance`) cannot provide.

**Why not just application logs?** Structured events in PostgreSQL are queryable, joinable, and indexable. "Show me every failure in the last 24 hours with the failing agent and retry count" is a SQL query, not a log grep. Events also serve as the raw data feed for PRD-106 (Outcome Telemetry & Learning Foundation) — every event is a data point for pattern analysis.

**Existing patterns in Automatos:** The codebase already has three audit log tables that inform our design:

| Table | Pattern | What We Adopt |
|-------|---------|---------------|
| `skill_audit_log` (PRD-22) | `action` + `action_details` JSON + `status` + `execution_time_ms` | Structured action with flexible JSON payload; timing metadata |
| `permission_audit_logs` (PRD-17) | `action` + `details` JSON + `user_id` + `timestamp` | Actor identification; indexed action column |
| `heartbeat_results` | `findings[]` + `actions_taken[]` JSONB arrays + `source_type`/`source_id` | Append-only event accumulation; source attribution |

Our `orchestration_events` table combines the best of these: typed events (from audit logs), flexible JSONB payload (from all three), actor tracking (from permission logs), and append-only semantics (from heartbeat_results).

### 6.2 Event Type Taxonomy

Event types follow the `{entity}_{lifecycle}` naming convention from Temporal (e.g., `ActivityTaskScheduled`, `ActivityTaskCompleted`), adapted to our domain. Types are grouped by entity for readability but stored as a flat enum.

#### Run Events

| Event Type | Trigger | Payload Fields |
|-----------|---------|----------------|
| `run_created` | Mission submitted by user or system | `goal`, `config_summary`, `autonomy_level` |
| `run_planning_started` | Coordinator begins decomposition | `coordinator_model` |
| `run_plan_ready` | Coordinator produces task plan | `task_count`, `estimated_cost`, `strategy` |
| `run_approved` | Human approves plan (or auto-approved) | `approved_by`, `modifications` (if human edited plan) |
| `run_rejected` | Human rejects plan | `rejected_by`, `reason` |
| `run_started` | First task begins execution | — |
| `run_paused` | Human pauses mission | `paused_by`, `reason` |
| `run_resumed` | Human resumes mission | `resumed_by` |
| `run_budget_warning` | Cost exceeds soft limit | `current_cost`, `soft_limit`, `percent_used` |
| `run_budget_exceeded` | Cost exceeds hard limit | `current_cost`, `hard_limit` |
| `run_budget_increased` | Human increases budget cap | `old_limit`, `new_limit`, `increased_by` |
| `run_completed` | All tasks terminal + success | `total_cost`, `total_tokens`, `duration_ms`, `tasks_completed`, `tasks_failed` |
| `run_failed` | Unrecoverable failure | `reason`, `failing_task_id`, `total_cost` |
| `run_cancelled` | Human cancels mission | `cancelled_by`, `tasks_remaining` |

#### Task Events

| Event Type | Trigger | Payload Fields |
|-----------|---------|----------------|
| `task_created` | Coordinator creates task from plan | `task_type`, `trigger_rule`, `depends_on` (task IDs) |
| `task_queued` | Dependencies met, ready for assignment | `unblocked_by` (task ID that completed last) |
| `task_assigned` | Coordinator assigns agent | `agent_id`, `agent_type`, `model`, `board_task_id` |
| `task_started` | Agent begins execution | `attempt_number` |
| `task_continuing` | Agent exits cleanly, needs more turns | `continuation_count`, `tokens_this_turn` |
| `task_resumed` | Continuation dispatched | `continuation_count` |
| `task_output_submitted` | Agent submits result | `output_ref`, `output_summary_length`, `tokens_used` |
| `task_verification_started` | Verifier begins evaluation | `verifier_agent_id`, `verifier_model` |
| `task_verification_passed` | Verifier approves output | `score`, `verifier_feedback` |
| `task_verification_failed` | Verifier rejects output | `score`, `verifier_feedback`, `retries_remaining` |
| `task_human_review_requested` | Escalated to human | `reason`, `score` |
| `task_human_approved` | Human approves output | `approved_by` |
| `task_human_rejected` | Human rejects output | `rejected_by`, `reason`, `retries_remaining` |
| `task_retrying` | Retry scheduled after failure | `attempt_number`, `backoff_seconds`, `failure_type` (`infrastructure` or `quality`) |
| `task_crashed` | Infrastructure failure detected | `error_type`, `error_message`, `duration_ms` |
| `task_failed` | Max retries exhausted or unrecoverable | `reason`, `total_attempts`, `total_cost` |
| `task_skipped` | Dependency failed + trigger rule prevents execution | `skipped_because`, `failed_dependency_id` |
| `task_cancelled` | Parent run cancelled | `cancelled_by` |

#### System Events

| Event Type | Trigger | Payload Fields |
|-----------|---------|----------------|
| `stall_detected` | Reconciler finds stalled task/run | `entity_type`, `entity_id`, `stalled_state`, `stalled_since`, `action_taken` |
| `model_fallback` | Primary model unavailable, falling back | `task_id`, `requested_model`, `fallback_model`, `reason` |
| `cost_snapshot` | Periodic cost aggregation | `run_id`, `total_cost`, `total_tokens`, `by_task` (breakdown) |

**Why 30+ event types instead of a generic "state_changed"?** Typed events enable:
1. **Targeted queries** — `WHERE event_type = 'task_crashed'` is faster than `WHERE payload->>'type' = 'crash'`
2. **Payload validation** — each event type has a known payload schema (enforceable via Pydantic on write)
3. **Downstream processing** — PRD-106 telemetry can subscribe to specific event types without parsing payloads
4. **Dashboard widgets** — "recent failures" widget queries `task_failed` + `task_crashed` directly

**Why not Temporal's 59 event types?** Temporal models internal execution machinery (workflow task scheduling, timer management, deterministic replay checkpoints). We don't have an execution replay engine — our events track business-level lifecycle changes visible to users and the coordinator.

### 6.3 Column Definitions

| Column | Type | Nullable | Default | Constraint | Description |
|--------|------|----------|---------|------------|-------------|
| `id` | `BIGSERIAL` | No | auto-increment | PK | Sequential event ID. `BIGINT` not `UUID` — events are high-volume append-only where sequential IDs are cheaper and provide natural ordering. |
| `run_id` | `UUID` | No | — | FK → `orchestration_runs.id` ON DELETE CASCADE | Which mission this event belongs to |
| `task_id` | `UUID` | Yes | `NULL` | FK → `orchestration_tasks.id` ON DELETE CASCADE | Which task (NULL for run-level events like `run_created`, `run_completed`) |
| `event_type` | `VARCHAR(50)` | No | — | — | Event type from taxonomy (Section 6.2). Indexed for filtering. |
| `payload` | `JSONB` | No | `'{}'` | — | Event-specific data. Schema varies by event_type (see payload fields in Section 6.2). |
| `actor_type` | `VARCHAR(20)` | No | `'system'` | — | Who triggered the event: `system`, `coordinator`, `agent`, `verifier`, `human`, `reconciler` |
| `actor_id` | `VARCHAR(255)` | Yes | `NULL` | — | ID of the actor: agent ID, user ID (Clerk), or NULL for system-triggered events |
| `created_at` | `TIMESTAMPTZ` | No | `NOW()` | — | When the event occurred. Indexed for time-range queries. |

**Why `BIGSERIAL` instead of `UUID`?**
- Events are append-only, never referenced by ID from other tables — no need for globally-unique identifiers
- Sequential `BIGINT` is 8 bytes vs UUID's 16 bytes — saves 8 bytes per row at potentially millions of rows
- B-tree indexes on `BIGINT` are more compact and cache-friendly
- Natural ordering: `ORDER BY id` = insertion order without consulting `created_at`
- Matches Dagster's `event_logs` (integer PK) and Airflow's `log` table (serial PK)

**Why `VARCHAR(50)` instead of a PostgreSQL ENUM for event_type?** Adding new event types to a PostgreSQL ENUM requires `ALTER TYPE ... ADD VALUE` — a DDL operation that can't be rolled back in a transaction. `VARCHAR` with application-layer validation (Pydantic enum) is simpler to evolve. Same rationale as the `state` columns on runs/tasks.

**Why no `workspace_id`?** Events always belong to a run, and runs have `workspace_id`. For workspace-filtered event queries, JOIN to `orchestration_runs`. This avoids denormalizing workspace_id onto every event row (unlike tasks, where the denormalization saves a frequent JOIN). Event queries are less frequent and typically already filtered by `run_id`.

### 6.4 Event Immutability Contract

Events are **append-only**. Once written, an event row is never updated or deleted by application code. This contract enables:

1. **Trustworthy audit trail** — "what happened" is never rewritten after the fact
2. **Safe concurrent reads** — no locking needed for event queries
3. **Simple replication** — append-only tables replicate cleanly to read replicas or analytics databases
4. **PRD-106 compatibility** — telemetry pipelines can process events exactly once with a high-water-mark cursor (last processed `id`)

**Enforcement:** No `UPDATE` or `DELETE` statements against `orchestration_events` in application code. The retention policy (Section 6.7) is the only mechanism that removes rows, and it operates at the DBA/cron level, not application level.

**No `updated_at` column.** Unlike runs and tasks, events have no mutable state. A single `created_at` timestamp is sufficient. Adding `updated_at` would signal that updates are expected — the opposite of our intent.

### 6.5 Event Creation Pattern

Events are created as a side effect of state transitions, inside the same database transaction. This is already implemented in the `transition_task()` function from Section 3.9:

```python
from enum import StrEnum

class EventType(StrEnum):
    # Run events
    RUN_CREATED = "run_created"
    RUN_PLANNING_STARTED = "run_planning_started"
    RUN_PLAN_READY = "run_plan_ready"
    RUN_APPROVED = "run_approved"
    RUN_REJECTED = "run_rejected"
    RUN_STARTED = "run_started"
    RUN_PAUSED = "run_paused"
    RUN_RESUMED = "run_resumed"
    RUN_BUDGET_WARNING = "run_budget_warning"
    RUN_BUDGET_EXCEEDED = "run_budget_exceeded"
    RUN_BUDGET_INCREASED = "run_budget_increased"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"

    # Task events
    TASK_CREATED = "task_created"
    TASK_QUEUED = "task_queued"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_CONTINUING = "task_continuing"
    TASK_RESUMED = "task_resumed"
    TASK_OUTPUT_SUBMITTED = "task_output_submitted"
    TASK_VERIFICATION_STARTED = "task_verification_started"
    TASK_VERIFICATION_PASSED = "task_verification_passed"
    TASK_VERIFICATION_FAILED = "task_verification_failed"
    TASK_HUMAN_REVIEW_REQUESTED = "task_human_review_requested"
    TASK_HUMAN_APPROVED = "task_human_approved"
    TASK_HUMAN_REJECTED = "task_human_rejected"
    TASK_RETRYING = "task_retrying"
    TASK_CRASHED = "task_crashed"
    TASK_FAILED = "task_failed"
    TASK_SKIPPED = "task_skipped"
    TASK_CANCELLED = "task_cancelled"

    # System events
    STALL_DETECTED = "stall_detected"
    MODEL_FALLBACK = "model_fallback"
    COST_SNAPSHOT = "cost_snapshot"


class ActorType(StrEnum):
    SYSTEM = "system"
    COORDINATOR = "coordinator"
    AGENT = "agent"
    VERIFIER = "verifier"
    HUMAN = "human"
    RECONCILER = "reconciler"


def emit_event(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    event_type: EventType,
    task_id: uuid.UUID | None = None,
    payload: dict | None = None,
    actor_type: ActorType = ActorType.SYSTEM,
    actor_id: str | None = None,
) -> None:
    """
    Append an immutable event to the orchestration_events table.

    MUST be called within an existing transaction — never commits on its own.
    The caller (transition_task, transition_run, or coordinator logic)
    owns the transaction boundary.
    """
    event = OrchestrationEvent(
        run_id=run_id,
        task_id=task_id,
        event_type=event_type.value,
        payload=payload or {},
        actor_type=actor_type.value,
        actor_id=actor_id,
    )
    session.add(event)
```

**Integration with `transition_task()` (from Section 3.9):**

The `transition_task()` function already emits events via `OrchestrationEvent(...)`. The `emit_event()` helper standardizes this pattern with enum validation and consistent actor tracking. The existing code in Section 3.9 line `event_type=f"task_{to_state}"` is replaced with explicit `EventType` enum values — some state transitions emit events that don't map 1:1 to the state name (e.g., entering `failed` from `running` emits `task_crashed`, but entering `failed` from `verifying` emits `task_failed`).

**Events that don't correspond to state transitions:**

Not every event is a state change. Some events are emitted mid-state:

| Event | State During Emission | Notes |
|-------|-----------------------|-------|
| `run_budget_warning` | `running` (unchanged) | Soft limit hit — informational, no state change |
| `task_verification_started` | `verifying` (unchanged) | Tracking when verifier begins, not a state change |
| `stall_detected` | Various | Reconciler observation before it acts |
| `model_fallback` | `running` (unchanged) | Model substitution during execution |
| `cost_snapshot` | `running` (unchanged) | Periodic aggregation |

### 6.6 Query Examples

The event table is designed for three primary query patterns: **timeline reconstruction**, **failure analysis**, and **performance metrics extraction**.

#### Timeline Reconstruction

"Show me everything that happened in mission X, in order."

```sql
SELECT
    e.id,
    e.event_type,
    e.task_id,
    t.title AS task_title,
    e.actor_type,
    e.actor_id,
    e.payload,
    e.created_at
FROM orchestration_events e
LEFT JOIN orchestration_tasks t ON t.id = e.task_id
WHERE e.run_id = $1
ORDER BY e.id;  -- id = insertion order, cheaper than ORDER BY created_at
```

#### Failure Analysis

"Find all failures across my workspace in the last 24 hours."

```sql
SELECT
    e.run_id,
    r.title AS mission_title,
    e.task_id,
    t.title AS task_title,
    e.event_type,
    e.payload->>'error_message' AS error,
    e.payload->>'failure_type' AS failure_type,
    e.created_at
FROM orchestration_events e
JOIN orchestration_runs r ON r.id = e.run_id
LEFT JOIN orchestration_tasks t ON t.id = e.task_id
WHERE r.workspace_id = $1
  AND e.event_type IN ('task_crashed', 'task_failed', 'run_failed')
  AND e.created_at >= NOW() - INTERVAL '24 hours'
ORDER BY e.created_at DESC;
```

#### Performance Metrics

"Calculate average time from task assignment to completion for each task type."

```sql
WITH task_timings AS (
    SELECT
        t.task_type,
        t.id AS task_id,
        assign_evt.created_at AS assigned_at,
        complete_evt.created_at AS completed_at,
        EXTRACT(EPOCH FROM (complete_evt.created_at - assign_evt.created_at)) * 1000 AS duration_ms
    FROM orchestration_tasks t
    JOIN orchestration_events assign_evt
        ON assign_evt.task_id = t.id AND assign_evt.event_type = 'task_assigned'
    JOIN orchestration_events complete_evt
        ON complete_evt.task_id = t.id AND complete_evt.event_type IN ('task_verification_passed', 'task_human_approved')
    WHERE t.workspace_id = $1
      AND t.state = 'completed'
)
SELECT
    task_type,
    COUNT(*) AS tasks,
    ROUND(AVG(duration_ms)) AS avg_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms)) AS p95_ms,
    ROUND(MIN(duration_ms)) AS min_ms,
    ROUND(MAX(duration_ms)) AS max_ms
FROM task_timings
GROUP BY task_type
ORDER BY avg_ms DESC;
```

#### Retry Analysis

"Which tasks are retried most often, and what's the success rate after retry?"

```sql
SELECT
    t.task_type,
    t.title,
    COUNT(*) FILTER (WHERE e.event_type = 'task_retrying') AS retry_count,
    COUNT(*) FILTER (WHERE e.event_type = 'task_verification_passed') AS eventual_successes,
    COUNT(*) FILTER (WHERE e.event_type = 'task_failed') AS eventual_failures,
    ROUND(
        COUNT(*) FILTER (WHERE e.event_type = 'task_verification_passed')::numeric /
        NULLIF(COUNT(*) FILTER (WHERE e.event_type IN ('task_verification_passed', 'task_failed')), 0),
        2
    ) AS success_rate
FROM orchestration_events e
JOIN orchestration_tasks t ON t.id = e.task_id
WHERE t.workspace_id = $1
  AND e.event_type IN ('task_retrying', 'task_verification_passed', 'task_failed')
GROUP BY t.id, t.task_type, t.title
HAVING COUNT(*) FILTER (WHERE e.event_type = 'task_retrying') > 0
ORDER BY retry_count DESC;
```

#### PRD-106 Telemetry Feed

"Stream events since cursor for telemetry pipeline processing."

```sql
-- Cursor-based polling: process events since last high-water mark
SELECT id, run_id, task_id, event_type, payload, actor_type, actor_id, created_at
FROM orchestration_events
WHERE id > $1  -- $1 = last processed event ID (high-water mark)
ORDER BY id
LIMIT 1000;    -- batch size
```

This cursor pattern (sequential `BIGSERIAL` ID as cursor) is why events use `BIGSERIAL` instead of `UUID`. The telemetry pipeline stores its last processed ID and polls for new events — no complex change-data-capture infrastructure needed.

### 6.7 Retention Policy

Events accumulate indefinitely if not managed. At our expected scale (10-50 events per mission, ~100 missions/day = ~2,500 events/day = ~1M events/year), storage is manageable but querying old events degrades performance without maintenance.

**Three-tier retention strategy:**

| Tier | Age | Storage | Access Pattern |
|------|-----|---------|----------------|
| **Hot** | 0–30 days | `orchestration_events` table (PostgreSQL) | Real-time queries, dashboard, debugging |
| **Warm** | 30–180 days | Same table, but excluded from partial indexes | Historical analysis, PRD-106 pattern mining |
| **Cold** | 180+ days | Archived (export to S3/object storage as JSONL, then DELETE) | Compliance/audit only, rare access |

**Implementation approach: pg_cron + batched DELETE**

At our projected volume (~1M events/year), table partitioning (pg_partman) is overkill. A simple scheduled cleanup job is sufficient:

```sql
-- Run weekly via pg_cron or APScheduler
-- Archive old events to export table, then delete
WITH archived AS (
    DELETE FROM orchestration_events
    WHERE created_at < NOW() - INTERVAL '180 days'
    RETURNING *
)
INSERT INTO orchestration_events_archive
SELECT * FROM archived;
```

**When to upgrade to partitioning:** If event volume exceeds ~10M events/month (e.g., 100+ missions/day with 50+ events each), switch to `PARTITION BY RANGE (created_at)` with monthly partitions and pg_partman for automated partition management. The table schema supports this transition — `created_at` is already `NOT NULL` and indexed.

**Archive table schema:** Identical to `orchestration_events` but without foreign key constraints (the referenced runs/tasks may be deleted independently). Used only for compliance queries.

### 6.8 Indexes

```sql
-- Primary query: "timeline for a mission" (mission detail view)
-- Also used by transition_task() to emit events within a run context
CREATE INDEX ix_orch_events_run_id
    ON orchestration_events (run_id, id);

-- Query: "events for a specific task" (task detail view, debugging)
CREATE INDEX ix_orch_events_task_id
    ON orchestration_events (task_id, id)
    WHERE task_id IS NOT NULL;

-- Query: "recent failures" (dashboard widget, alerting)
CREATE INDEX ix_orch_events_type_created
    ON orchestration_events (event_type, created_at DESC);

-- Query: "telemetry cursor" (PRD-106 pipeline, sequential scan from last ID)
-- PK index on id covers this — sequential reads on a BIGSERIAL PK are optimal

-- Query: "time-range scans" (retention cleanup, analytics)
-- BRIN index: compact, effective for append-only time-ordered data
CREATE INDEX ix_orch_events_created_brin
    ON orchestration_events USING BRIN (created_at)
    WITH (pages_per_range = 32);
```

**Why BRIN for `created_at`?** Events are insert-ordered and `created_at` correlates perfectly with physical row order. A BRIN index is ~1000x smaller than a B-tree for the same column on append-only tables. It supports time-range scans (retention cleanup, "last 24 hours" queries) efficiently. The tradeoff is slightly less precise than B-tree — acceptable for time-range filtering where exact row targeting isn't needed.

**Why `(run_id, id)` instead of `(run_id, created_at)`?** The `id` column (BIGSERIAL) provides insertion ordering identical to `created_at` but without timezone comparison overhead. For `ORDER BY` within a run, `id` is strictly monotonic — faster to sort and more compact in the index.

### 6.9 SQLAlchemy Model

```python
from sqlalchemy import BigInteger, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
import uuid

from orchestrator.core.database.base import Base


class OrchestrationEvent(Base):
    """
    Append-only event log for mission lifecycle tracking (PRD-101).

    Events are NEVER updated or deleted by application code.
    Every state transition in orchestration_runs/tasks emits an event
    in the same transaction (dual-write pattern, Section 2.4).

    Serves as raw data feed for PRD-106 (Outcome Telemetry).
    """
    __tablename__ = "orchestration_events"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("orchestration_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("orchestration_tasks.id", ondelete="CASCADE"),
        nullable=True,
    )
    event_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
    )
    payload: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}",
    )
    actor_type: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="system",
    )
    actor_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # No version_id — events are immutable
    # No updated_at — events are never modified
    # No relationships defined — events are write-heavy, read via raw queries
```

### 6.10 Alembic Migration

```python
"""PRD-101: Create orchestration_events table

Append-only event log for mission lifecycle tracking.
Second half of the dual-write pattern — every state transition
on runs/tasks emits an event in the same transaction.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "prd101_orchestration_events"
down_revision = "prd101_orchestration_tasks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orchestration_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("run_id", UUID(as_uuid=True), nullable=False),
        sa.Column("task_id", UUID(as_uuid=True), nullable=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("payload", JSONB, nullable=False, server_default="{}"),
        sa.Column("actor_type", sa.String(20), nullable=False, server_default="system"),
        sa.Column("actor_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        # Foreign keys
        sa.ForeignKeyConstraint(["run_id"], ["orchestration_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["task_id"], ["orchestration_tasks.id"], ondelete="CASCADE"),
    )

    # Indexes
    op.create_index("ix_orch_events_run_id", "orchestration_events", ["run_id", "id"])
    op.create_index(
        "ix_orch_events_task_id", "orchestration_events",
        ["task_id", "id"],
        postgresql_where=sa.text("task_id IS NOT NULL"),
    )
    op.create_index(
        "ix_orch_events_type_created", "orchestration_events",
        ["event_type", sa.text("created_at DESC")],
    )
    op.execute(
        "CREATE INDEX ix_orch_events_created_brin "
        "ON orchestration_events USING BRIN (created_at) "
        "WITH (pages_per_range = 32)"
    )

    # Table comment
    op.execute(
        "COMMENT ON TABLE orchestration_events IS "
        "'Append-only event log for mission lifecycle (PRD-101). "
        "Never UPDATE or DELETE from application code.'"
    )

    # Archive table (identical schema, no FKs)
    op.create_table(
        "orchestration_events_archive",
        sa.Column("id", sa.BigInteger, primary_key=True),  # NOT autoincrement — preserves original IDs
        sa.Column("run_id", UUID(as_uuid=True), nullable=False),
        sa.Column("task_id", UUID(as_uuid=True), nullable=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("payload", JSONB, nullable=False, server_default="{}"),
        sa.Column("actor_type", sa.String(20), nullable=False, server_default="system"),
        sa.Column("actor_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_orch_events_archive_run", "orchestration_events_archive", ["run_id"])
    op.create_index("ix_orch_events_archive_created", "orchestration_events_archive", ["created_at"])

    op.execute(
        "COMMENT ON TABLE orchestration_events_archive IS "
        "'Cold storage for orchestration events older than 180 days (PRD-101).'"
    )


def downgrade() -> None:
    op.drop_index("ix_orch_events_archive_created", table_name="orchestration_events_archive")
    op.drop_index("ix_orch_events_archive_run", table_name="orchestration_events_archive")
    op.drop_table("orchestration_events_archive")
    op.execute("DROP INDEX IF EXISTS ix_orch_events_created_brin")
    op.drop_index("ix_orch_events_type_created", table_name="orchestration_events")
    op.drop_index("ix_orch_events_task_id", table_name="orchestration_events")
    op.drop_index("ix_orch_events_run_id", table_name="orchestration_events")
    op.drop_table("orchestration_events")
```

### 6.11 Connection to PRD-106 (Outcome Telemetry)

The `orchestration_events` table is the **primary data source** for PRD-106 (Outcome Telemetry & Learning Foundation). Every event is a structured data point that the telemetry pipeline can process for pattern analysis.

**What PRD-106 will extract from events:**

| Analysis | Events Used | Insight |
|----------|-------------|---------|
| **Agent effectiveness** | `task_assigned` + `task_verification_passed`/`task_failed` | Which agents succeed at which task types? |
| **Model cost-effectiveness** | `task_assigned` (model) + `cost_snapshot` | Does claude-sonnet at 10x the cost produce measurably better results than haiku? |
| **Retry patterns** | `task_retrying` + `task_crashed` | Which failure types are transient (worth retrying) vs persistent? |
| **Task duration distribution** | `task_started` + `task_verification_passed` | How long do different task types actually take? |
| **Verification accuracy** | `task_verification_passed` + `task_human_approved`/`task_human_rejected` | Does the verifier's judgment match human judgment? |
| **Mission bottlenecks** | Timeline reconstruction per run | Which tasks consistently delay mission completion? |
| **Budget accuracy** | `run_plan_ready` (estimated_cost) + `run_completed` (total_cost) | How accurate are the coordinator's cost estimates? |

**Design constraint for PRD-106:** The event schema must remain stable — adding new event types is fine, but changing existing payload schemas or event type names breaks downstream telemetry queries. New payload fields should be additive (never remove or rename existing fields).

### 6.12 Design Decisions

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Primary key | `BIGSERIAL` | UUID | Events are high-volume, append-only, never externally referenced. Sequential integers are 8 bytes vs 16, more compact in indexes, and provide natural ordering. Matches Dagster (`event_logs`) and Airflow (`log`) patterns. |
| Event type storage | `VARCHAR(50)` with Python enum | PostgreSQL ENUM type | `ALTER TYPE ADD VALUE` can't be rolled back in a transaction. VARCHAR with application-layer validation is simpler to evolve. New event types don't need a migration. |
| Payload structure | Flat `JSONB` column | Typed columns per event | 30+ event types × 3-5 fields each = 100+ columns. JSONB keeps the table lean. Payload schema is validated by Pydantic on write (same pattern as `orchestration_runs.config`). |
| Actor tracking | `actor_type` + `actor_id` | Single `actor` string | Separating type from ID enables queries like "find all human actions" (`WHERE actor_type = 'human'`) without parsing a composite string. |
| Workspace filtering | JOIN to `orchestration_runs` | Denormalize `workspace_id` | Unlike tasks (which need workspace filtering on every list query), event queries are typically scoped to a run_id. Denormalizing workspace_id onto every event row wastes 16 bytes/row with minimal query benefit. |
| Retention | pg_cron batched DELETE + archive table | pg_partman, application-layer TTL | At ~1M events/year, partitioning overhead isn't justified. Batched DELETE with `RETURNING` → archive is simple, transactional, and handles our scale. Upgrade path to pg_partman is documented if volume grows 10x. |
| Archive strategy | Separate table (same schema, no FKs) | S3 export only | Keeping archived events in PostgreSQL enables historical queries without object storage tooling. The archive table has minimal indexes (run_id + created_at only) to reduce write overhead. |
| `created_at` index | BRIN | B-tree | Append-only data with correlated physical order is the ideal BRIN use case — ~1000x smaller than B-tree with comparable query performance for range scans. |
| No `workspace_id` column | JOIN for workspace queries | Denormalize | Events are queried by run_id (detail view) or event_type (alerts). Workspace-level event queries are rare and tolerate a JOIN. Saves 16 bytes × millions of rows. |
| Immutability enforcement | Application convention + table comment | DB trigger blocking UPDATE/DELETE | DB trigger adds overhead on every INSERT (trigger evaluation). Convention is sufficient when all writes go through `emit_event()`. The table comment documents the contract. |

---

## 7. Integration with Existing Schema

The orchestration tables don't exist in isolation — they must integrate cleanly with four existing systems: the Kanban board (`board_tasks`), agent reports (`agent_reports`), recipes (`workflow_recipes`), and workspace isolation. This section defines every integration point, the data flow between tables, and the migration strategy that adds three new tables without breaking anything.

### 7.1 Entity Relationship Diagram

```
┌──────────────┐          ┌─────────────────────────┐          ┌──────────────┐
│  workspaces  │          │   orchestration_runs     │          │    agents    │
│──────────────│          │─────────────────────────│          │──────────────│
│ id (UUID) PK │◄────FK───│ workspace_id             │          │ id (INT) PK  │
│ ...          │          │ id (UUID) PK             │──FK────► │ ...          │
└──────────────┘          │ coordinator_agent_id ────┘          └──────┬───────┘
       ▲                  │ state, plan, config...   │                 │
       │                  └──────────┬───────────────┘                 │
       │                             │ 1:N                             │
       │                             ▼                                 │
       │                  ┌─────────────────────────┐                  │
       │                  │   orchestration_tasks    │                  │
       │             FK───│ workspace_id             │                  │
       │                  │ id (UUID) PK             │──FK─────────────┘
       │                  │ run_id ──────────────FK──┘    agent_id
       │                  │ board_task_id ───FK──┐
       │                  │ state, output_ref... │
       │                  └──────┬───────┬───────┘
       │                         │       │
       │                    1:N  │       │  M:N
       │                         ▼       ▼
       │           ┌──────────────┐  ┌───────────────────────────────┐
       │           │ orchestration│  │ orchestration_task_dependencies│
       │           │ _events      │  │───────────────────────────────│
       │           │──────────────│  │ task_id (FK → tasks)          │
       │           │ run_id (FK)  │  │ depends_on_id (FK → tasks)    │
       │           │ task_id (FK) │  │ dependency_type                │
       │           │ event_type   │  └───────────────────────────────┘
       │           │ payload JSONB│
       │           └──────────────┘
       │
       │            ┌──────────────────┐          ┌──────────────────────┐
       │            │   board_tasks    │          │   workflow_recipes   │
       │            │──────────────────│          │──────────────────────│
       ├───FK───────│ workspace_id     │          │ id (INT) PK          │
       │            │ id (INT) PK      │◄─ ─ ─ ─ │ originated_from_     │
       │            │ source_type      │   (save  │   mission_run_id     │
       │            │ source_id        │  as      │ steps, inputs...     │
       │            │ parent_task_id──┐│ routine) └──────────────────────┘
       │            │ planning_data   ││
       │            └─────────────────┘│           ┌──────────────────────┐
       │                    ▲          └──────────►│ board_tasks (child)  │
       │                    │                      │ parent_task_id = ↑   │
       │                    │ FK                   └──────────────────────┘
       │                    │
       │            ┌──────────────────┐
       │            │  agent_reports   │
       │            │──────────────────│
       └───FK───────│ workspace_id     │
                    │ agent_id (FK)    │
                    │ orchestration_   │
                    │   task_id (FK)   │  ◄── NEW COLUMN
                    │ report_type      │
                    │ file_path        │
                    └──────────────────┘
```

### 7.2 Board Task Integration

Every orchestration run creates one **parent** board task (the mission), and every orchestration task creates one **child** board task (linked via `parent_task_id`). This gives the existing Kanban UI mission visibility without new frontend components.

#### 7.2.1 Source Type Values

The `board_tasks.source_type` column currently holds two values:

| Value | Created By | Count Today |
|-------|-----------|-------------|
| `user` (default) | Manual creation via API | Most tasks |
| `recipe` | `board_task_bridge.py` during recipe execution | Recipe-triggered tasks |

Orchestration adds two new values:

| Value | Created By | Description |
|-------|-----------|-------------|
| `orchestration` | Coordinator on mission start | One per `orchestration_runs` row — the parent task |
| `orchestration_task` | Coordinator during task planning | One per `orchestration_tasks` row — child tasks |

No migration needed — `source_type` is `VARCHAR(30)`, not a database enum.

#### 7.2.2 Field Mapping: orchestration_runs → board_tasks (Parent)

| BoardTask Field | Source | Example Value |
|----------------|--------|---------------|
| `workspace_id` | `orchestration_runs.workspace_id` | `550e8400-...` |
| `title` | `"Mission: " + orchestration_runs.title` | `"Mission: EU AI Act Compliance"` |
| `description` | `orchestration_runs.goal` (verbatim user input) | `"Research EU AI Act compliance for our product"` |
| `status` | Mapped from `orchestration_runs.state` (see Section 3.8 pattern) | `in_progress` |
| `priority` | `orchestration_runs.config.priority` or `'high'` | `high` |
| `review_mode` | `'manual'` (missions always need human review) | `manual` |
| `assigned_agent_id` | `orchestration_runs.coordinator_agent_id` | `42` or `NULL` |
| `created_by_type` | `'orchestration'` | `orchestration` |
| `created_by_id` | `orchestration_runs.created_by` | `user_clerk_abc` |
| `parent_task_id` | `NULL` (this IS the parent) | `NULL` |
| `source_type` | `'orchestration'` | `orchestration` |
| `source_id` | `str(orchestration_runs.id)` | `"a1b2c3d4-..."` |
| `tags` | `['mission', orchestration_runs.state]` | `['mission', 'running']` |
| `planning_data` | See below | JSONB |
| `started_at` | `orchestration_runs.started_at` | Timestamp |

**`planning_data` JSONB for mission parent:**

```json
{
  "run_id": "a1b2c3d4-...",
  "strategy": "sequential",
  "task_progress": {"completed": 2, "total": 4, "failed": 0},
  "budget": {"spent": 0.12, "limit": 5.00},
  "current_task": "Analyze product against requirements"
}
```

#### 7.2.3 Field Mapping: orchestration_tasks → board_tasks (Child)

| BoardTask Field | Source | Example Value |
|----------------|--------|---------------|
| `workspace_id` | `orchestration_tasks.workspace_id` | `550e8400-...` |
| `title` | `orchestration_tasks.title` | `"Research EU AI Act requirements"` |
| `description` | `orchestration_tasks.description` | Detailed task instructions |
| `status` | Mapped from `orchestration_tasks.state` (Section 3.8) | `assigned` |
| `priority` | Inherited from parent or `'medium'` | `medium` |
| `review_mode` | `'auto'` unless task has `success_criteria` → `'manual'` | `auto` |
| `assigned_agent_id` | `orchestration_tasks.agent_id` | `42` |
| `created_by_type` | `'orchestration'` | `orchestration` |
| `parent_task_id` | Parent mission's `board_tasks.id` | `1234` |
| `source_type` | `'orchestration_task'` | `orchestration_task` |
| `source_id` | `str(orchestration_tasks.id)` | `"e5f6g7h8-..."` |
| `tags` | `['mission_task', orchestration_tasks.task_type]` | `['mission_task', 'research']` |
| `planning_data` | See below | JSONB |
| `result` | `orchestration_tasks.output_summary` (≤2000 chars, truncated to 4000 by board) | Summary text |
| `error_message` | `orchestration_tasks.error_message` | Failure reason |

**`planning_data` JSONB for task child:**

```json
{
  "task_id": "e5f6g7h8-...",
  "run_id": "a1b2c3d4-...",
  "sequence": 1,
  "task_type": "research",
  "attempt": 1,
  "continuation": 0,
  "dependencies": ["depends on: Task 0 (always)"],
  "trigger_rule": "all_success"
}
```

#### 7.2.4 State Synchronization

Board task status is updated as a **side effect of `transition_task()` and `transition_run()`** — in the same database transaction as the state change and event emission (the dual-write from Section 3.3). This is not a separate sync job; it's atomic.

```python
# Inside transition_task() — already defined in Section 3.9
async def transition_task(session, task, to_state, *, reason=None, metadata=None):
    # ... validate transition, update state, emit event ...

    # Sync board task status (same transaction)
    if task.board_task_id:
        board_task = await session.get(BoardTask, task.board_task_id)
        if board_task:
            board_task.status = TASK_STATE_TO_BOARD_STATUS[to_state]
            if to_state == TaskState.RUNNING and not board_task.started_at:
                board_task.started_at = utcnow()
            if TASK_STATE_TYPE[to_state] == StateType.TERMINAL:
                board_task.completed_at = utcnow()
                if to_state == TaskState.COMPLETED:
                    board_task.result = task.output_summary[:4000] if task.output_summary else None
                elif to_state in (TaskState.FAILED, TaskState.CANCELLED):
                    board_task.error_message = task.error_message

    # Update parent mission board task progress
    await _update_mission_board_progress(session, task.run_id)
```

#### 7.2.5 Board Task Bridge Functions

Following the pattern in `orchestrator/services/board_task_bridge.py`, add these functions to the same file (or a new `orchestration_board_bridge.py`):

```python
def create_mission_board_task(
    db: Session,
    run: OrchestrationRun,
) -> Optional[int]:
    """Create parent BoardTask for a mission. Returns board_task.id or None if exists."""
    existing = db.query(BoardTask.id).filter(
        BoardTask.source_type == 'orchestration',
        BoardTask.source_id == str(run.id),
    ).first()
    if existing:
        return existing[0]

    task = BoardTask(
        workspace_id=run.workspace_id,
        title=f"Mission: {run.title}",
        description=run.goal,
        status='in_progress',
        priority='high',
        review_mode='manual',
        coordinator_agent_id=run.coordinator_agent_id,
        created_by_type='orchestration',
        created_by_id=run.created_by,
        source_type='orchestration',
        source_id=str(run.id),
        tags=['mission'],
        planning_data={
            'run_id': str(run.id),
            'strategy': (run.plan or {}).get('strategy', 'unknown'),
            'task_progress': {'completed': 0, 'total': run.task_count, 'failed': 0},
        },
    )
    db.add(task)
    db.flush()  # Get id without committing (caller manages transaction)
    return task.id


def create_task_board_task(
    db: Session,
    task: OrchestrationTask,
    parent_board_task_id: int,
) -> Optional[int]:
    """Create child BoardTask for an orchestration task. Returns board_task.id."""
    existing = db.query(BoardTask.id).filter(
        BoardTask.source_type == 'orchestration_task',
        BoardTask.source_id == str(task.id),
    ).first()
    if existing:
        return existing[0]

    bt = BoardTask(
        workspace_id=task.workspace_id,
        title=task.title,
        description=task.description,
        status='inbox',
        priority='medium',
        review_mode='auto',
        assigned_agent_id=task.agent_id,
        created_by_type='orchestration',
        parent_task_id=parent_board_task_id,
        source_type='orchestration_task',
        source_id=str(task.id),
        tags=['mission_task', task.task_type],
        planning_data={
            'task_id': str(task.id),
            'run_id': str(task.run_id),
            'sequence': task.sequence_number,
            'task_type': task.task_type,
        },
    )
    db.add(bt)
    db.flush()
    return bt.id
```

#### 7.2.6 New Indexes on board_tasks

Two partial unique indexes prevent duplicate board tasks for the same orchestration entity:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS uq_board_tasks_orchestration_run
    ON board_tasks(source_id) WHERE source_type = 'orchestration';

CREATE UNIQUE INDEX IF NOT EXISTS uq_board_tasks_orchestration_task
    ON board_tasks(source_id) WHERE source_type = 'orchestration_task';
```

These follow the existing pattern from PRD-72:
```sql
-- Already exists:
CREATE UNIQUE INDEX uq_board_tasks_recipe_exec
    ON board_tasks(source_id) WHERE source_type = 'recipe';
```

#### 7.2.7 parent_task_id Usage

The `board_tasks.parent_task_id` column exists today (`INTEGER FK → board_tasks.id ON DELETE SET NULL`) but is **unused by recipes or manual tasks**. The API accepts it (`POST /api/v1/tasks` body, `GET /api/v1/tasks?parent_task_id=N` filter) but no feature populates it.

Orchestration is the first consumer: every task-level board task has `parent_task_id` pointing to the mission-level board task. This enables:
- Dashboard query: "show all tasks in this mission" → `WHERE parent_task_id = :mission_board_id`
- Tree rendering: parent + children hierarchy in the Kanban UI
- Cascade behavior: already defined as `ON DELETE SET NULL` — if mission board task is manually deleted, child tasks become orphans (acceptable; orchestration tables are the source of truth)

### 7.3 Agent Reports Integration

PRD-76 established the `agent_reports` table for structured report metadata with workspace file storage. Orchestration task completion should auto-generate reports using the same system.

#### 7.3.1 New FK on agent_reports

Add one nullable column:

```sql
ALTER TABLE agent_reports
    ADD COLUMN IF NOT EXISTS orchestration_task_id UUID
    REFERENCES orchestration_tasks(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS ix_agent_reports_orch_task
    ON agent_reports(orchestration_task_id)
    WHERE orchestration_task_id IS NOT NULL;
```

This parallels the existing `heartbeat_result_id` FK — both are optional context references that link a report to its trigger:

| FK Column | Links To | Populated When |
|-----------|----------|---------------|
| `heartbeat_result_id` | `heartbeat_results.id` | Report created during heartbeat tick |
| `orchestration_task_id` | `orchestration_tasks.id` | Report created on task completion |

Both can be NULL (standalone report). Both use `ON DELETE SET NULL` (report survives source deletion).

#### 7.3.2 Report Creation Flow

When an orchestration task reaches a terminal state with output, a report is auto-created via the existing `ReportService.create_report()`:

```python
# Called from transition_task() when to_state in TERMINAL_TASK_STATES
async def _auto_create_task_report(
    session: AsyncSession,
    task: OrchestrationTask,
    run: OrchestrationRun,
):
    """Auto-generate an agent_report for a completed orchestration task."""
    if not task.output_summary and not task.output_ref:
        return  # No output → no report

    agent = await session.get(Agent, task.agent_id) if task.agent_id else None

    report_type = _task_type_to_report_type(task.task_type, task.state)
    status = "ok" if task.state == TaskState.COMPLETED else "warning"

    await report_service.create_report(
        workspace_id=task.workspace_id,
        agent_id=task.agent_id,          # nullable — matches PRD-76 Phase 2
        agent_name=agent.name if agent else "Coordinator",
        title=f"{run.title} — {task.title}",
        content=task.output_summary or f"See full output: {task.output_ref}",
        report_type=report_type,
        status=status,
        summary=task.output_summary[:500] if task.output_summary else None,
        metrics={
            "tokens_used": task.tokens_used,
            "cost_usd": float(task.cost),
            "duration_ms": task.duration_ms,
            "attempt_number": task.attempt_number,
            "verifier_score": float(task.verifier_score) if task.verifier_score else None,
        },
        orchestration_task_id=task.id,   # NEW FK
    )
```

#### 7.3.3 Report Type Mapping

Existing `report_type` values: `standup`, `research`, `incident`, `summary`, `delivery`, `audit`.

Orchestration task types map to report types:

| Task Type (`orchestration_tasks.task_type`) | Report Type | Rationale |
|---------------------------------------------|-------------|-----------|
| `research` | `research` | Direct mapping |
| `analysis` | `research` | Analysis is a form of research |
| `writing` | `delivery` | Writing produces a deliverable |
| `coding` | `delivery` | Code output is a deliverable |
| `verification` | `audit` | Verification is quality audit |
| `review` | `audit` | Review is quality audit |
| `synthesis` | `summary` | Synthesis produces a summary |
| `other` | `delivery` | Default to deliverable |

For failed tasks, `report_type` is always `incident` (regardless of task type) to surface failures in the incident filter.

No new `report_type` enum values are needed — the existing six cover all orchestration task types.

#### 7.3.4 Mission-Level Summary Report

When a run reaches a terminal state (`completed`, `failed`, `cancelled`), the coordinator creates one summary report for the entire mission:

```python
async def _create_mission_summary_report(
    session: AsyncSession,
    run: OrchestrationRun,
):
    """Create a summary report for the entire mission."""
    tasks = await session.execute(
        select(OrchestrationTask)
        .where(OrchestrationTask.run_id == run.id)
        .order_by(OrchestrationTask.sequence_number)
    )

    # Build markdown summary
    lines = [f"# Mission: {run.title}\n", f"**Goal:** {run.goal}\n"]
    for t in tasks.scalars():
        status_icon = "✅" if t.state == "completed" else "❌" if t.state == "failed" else "⏭️"
        lines.append(f"- {status_icon} **{t.title}** — {t.state} ({t.duration_ms or 0}ms, ${float(t.cost):.4f})")

    lines.append(f"\n**Total cost:** ${float(run.total_cost):.4f}")
    lines.append(f"**Total tokens:** {run.total_tokens}")
    lines.append(f"**Duration:** {run.duration_ms}ms")

    await report_service.create_report(
        workspace_id=run.workspace_id,
        agent_id=run.coordinator_agent_id,
        agent_name="Mission Coordinator",
        title=f"Mission Complete: {run.title}",
        content="\n".join(lines),
        report_type="summary",
        status="ok" if run.state == "completed" else "warning",
        summary=run.result_summary[:500] if run.result_summary else None,
        metrics={
            "total_tokens": run.total_tokens,
            "total_cost": float(run.total_cost),
            "duration_ms": run.duration_ms,
            "tasks_completed": run.tasks_completed,
            "tasks_failed": run.tasks_failed,
            "task_count": run.task_count,
        },
    )
```

### 7.4 Recipe Integration ("Save as Routine")

A successfully completed mission can be converted into a repeatable recipe (`workflow_recipes` row). This is the "Save as Routine?" button from PRD-100 Section 3.

#### 7.4.1 Conversion Flow

```
User clicks "Save as Routine" on completed mission
        │
        ▼
   API: POST /api/v1/missions/{run_id}/save-as-recipe
        │
        ▼
   Read orchestration_runs + orchestration_tasks + orchestration_task_dependencies
        │
        ▼
   Transform mission structure → recipe structure
        │
        ▼
   INSERT workflow_recipes row
        │
        ▼
   Return recipe_id → user can edit, schedule, or run immediately
```

#### 7.4.2 Field Mapping: orchestration_runs → workflow_recipes

| orchestration_runs Field | workflow_recipes Field | Transformation |
|--------------------------|----------------------|----------------|
| `title` | `name` | Direct copy |
| `goal` | `description` | Direct copy (user's original intent) |
| `id` | — (stored in `template_definition.originated_from`) | Lineage tracking |
| `plan.strategy` | `execution_config.mode` | `"sequential"` → `"sequential"`, `"parallel"` → `"parallel"` |
| `plan.tasks` | `steps` | See task→step mapping below |
| `config.budget` | `execution_config.budget` | Copy budget config |
| `config.model_preferences` | Per-step model config | Distributed to steps |
| `config.retry` | `execution_config.max_retries` | Copy retry policy |
| `workspace_id` | `workspace_id` | Direct copy |
| `created_by` | `created_by` | Direct copy |
| — | `template_id` | Auto-generated: `"mission-{run_id[:8]}-{timestamp}"` |
| — | `owner_type` | `'workspace'` |
| — | `tags` | `['from_mission', 'auto_generated']` |

#### 7.4.3 Field Mapping: orchestration_tasks → recipe steps

Each orchestration task becomes a recipe step. Dependencies are flattened to `pass_to` references:

```json
{
  "steps": [
    {
      "step_id": "s1",
      "order": 1,
      "agent_id": 42,
      "prompt_template": "Research EU AI Act requirements relevant to our product category.\n\nSuccess criteria: {success_criteria}",
      "task_type": "research",
      "model_override": "anthropic/claude-sonnet-4-6",
      "tools_needed": ["web_search", "document_analysis"],
      "pass_to": null,
      "error_handling": {"max_retries": 3, "on_failure": "stop"},
      "originated_from_task_id": "e5f6g7h8-..."
    },
    {
      "step_id": "s2",
      "order": 2,
      "agent_id": 42,
      "prompt_template": "Analyze our product against the requirements from step s1.\n\nInput from previous step: {{s1.output}}",
      "task_type": "analysis",
      "depends_on": ["s1"],
      "pass_to": "s3",
      "error_handling": {"max_retries": 2, "on_failure": "stop"}
    }
  ]
}
```

| orchestration_tasks Field | Recipe Step Field | Notes |
|---------------------------|------------------|-------|
| `sequence_number` | `order` | Direct mapping |
| `title` + `description` | `prompt_template` | Combined into agent instructions |
| `agent_id` | `agent_id` | Roster agent preserved; contractor agents replaced with `null` (user must assign) |
| `model_override` | `model_override` | Preserved if set |
| `task_type` | `task_type` | New field on steps (not in current recipe schema) |
| `tools_requested` | `tools_needed` | Hint list preserved |
| `success_criteria` | Embedded in `prompt_template` | Injected as template variable |
| Dependencies (join table) | `depends_on` array | Simplified to step_id references |
| `trigger_rule` | `error_handling.on_failure` | `all_success` → `"stop"`, `all_done` → `"continue"`, `always` → `"always_run"` |

#### 7.4.4 Limitations

1. **Contractor agents don't survive conversion.** Ephemeral agents are mission-scoped — the recipe stores `agent_id: null` with a note that the user must assign a roster agent or let the coordinator pick one.
2. **DAG → sequence flattening.** Complex parallel DAGs are topologically sorted into a linear sequence for the recipe's `order` field. Parallel execution is noted in `execution_config.mode: "parallel"` but individual step parallelism is lost. This is acceptable — recipes are simpler than missions by design.
3. **No input parameterization.** The first conversion is a snapshot — the user's original goal is hardcoded in the description. To make the recipe truly reusable, the user must edit the steps to add `{{variable}}` placeholders in the `prompt_template` fields and define corresponding `inputs` schema. This is a manual step, not automated.

#### 7.4.5 `execution_config` JSONB for Mission-Derived Recipes

```json
{
  "mode": "sequential",
  "max_retries": 3,
  "timeout_per_step": 300,
  "total_timeout": 1800,
  "budget": {
    "max_cost_usd": 5.00,
    "max_tokens": 100000
  },
  "originated_from": {
    "mission_run_id": "a1b2c3d4-...",
    "actual_cost": 0.45,
    "actual_tokens": 23000,
    "actual_duration_ms": 45000,
    "completed_at": "2026-03-15T14:30:00Z"
  }
}
```

The `originated_from` block preserves lineage — when the recipe runs, the system can compare actual performance against the originating mission to detect drift.

### 7.5 Workspace Isolation

All orchestration queries MUST filter by `workspace_id`. This is enforced at the application layer (no PostgreSQL RLS), matching the pattern used by every other table in the platform.

#### 7.5.1 Auth Pattern

Workspace ID is resolved from the request via `hybrid.py`:

1. Header: `X-Workspace-ID` (preferred)
2. Header: `X-Workspace` (fallback)
3. Query param: `workspace_id`
4. Environment: `WORKSPACE_ID` / `DEFAULT_WORKSPACE_ID`

The `_user_has_workspace_access()` function validates the user owns or is a member of the requested workspace, preventing `X-Workspace-ID` spoofing.

#### 7.5.2 Query Pattern

Every API endpoint follows this pattern (from `board_tasks.py`):

```python
# List: Always scope to workspace
runs = db.query(OrchestrationRun).filter(
    OrchestrationRun.workspace_id == ctx.workspace_id
).order_by(OrchestrationRun.created_at.desc())

# Get single: Filter on BOTH id AND workspace_id
run = db.query(OrchestrationRun).filter(
    OrchestrationRun.id == run_id,
    OrchestrationRun.workspace_id == ctx.workspace_id,
).first()
if not run:
    raise HTTPException(status_code=404, detail="Run not found")
```

**Critical:** Never filter on `id` alone — always include `workspace_id` in the WHERE clause for single-record lookups. This prevents a user in workspace A from accessing workspace B's missions by guessing UUIDs.

#### 7.5.3 Denormalized workspace_id on orchestration_tasks

As noted in Section 5.1, `orchestration_tasks` denormalizes `workspace_id` (copied from the parent run) to avoid a JOIN on the most common query: "show all tasks for my workspace." The denormalization is safe because:
- workspace_id never changes on a run
- The coordinator copies it at task creation time
- The FK constraint on `orchestration_runs.workspace_id` ensures the workspace exists

#### 7.5.4 Cascade Delete

All orchestration tables use `ON DELETE CASCADE` for the workspace FK:

```sql
workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE
```

When a workspace is deleted, all its runs, tasks, dependencies, events, board tasks, and reports are cascade-deleted. This matches the pattern on `board_tasks`, `agent_reports`, `workflow_recipes`, and every other workspace-scoped table.

#### 7.5.5 No Admin Override Needed

Unlike agents and skills (which have marketplace items visible across workspaces), orchestration is strictly workspace-scoped. There is no `admin_all_workspaces` bypass for mission data — missions belong to exactly one workspace, always.

### 7.6 Index Summary

All indexes across the orchestration tables and their integration points:

#### New Tables (from Sections 4-6)

| Table | Index | Type | Purpose |
|-------|-------|------|---------|
| `orchestration_runs` | `ix_orch_runs_ws_created` | B-tree (`workspace_id`, `created_at DESC`) | Dashboard timeline: "my recent missions" |
| `orchestration_runs` | `ix_orch_runs_active` | Partial B-tree (`workspace_id`) WHERE `state_type IN ('scheduled','running')` | Reconciler: "find active runs" |
| `orchestration_runs` | `ix_orch_runs_created_by` | B-tree (`created_by`) | "My missions" filter |
| `orchestration_tasks` | `ix_orch_tasks_run` | B-tree (`run_id`) | Task list for a mission |
| `orchestration_tasks` | `ix_orch_tasks_ws_state` | B-tree (`workspace_id`, `state`) | Dashboard: "tasks by status" |
| `orchestration_tasks` | `ix_orch_tasks_agent` | B-tree (`agent_id`) WHERE `agent_id IS NOT NULL` | "Tasks assigned to this agent" |
| `orchestration_tasks` | `ix_orch_tasks_board` | B-tree (`board_task_id`) WHERE `board_task_id IS NOT NULL` | Reverse lookup: board → orchestration |
| `orchestration_task_dependencies` | PK covers (`task_id`, `depends_on_id`) | B-tree (composite PK) | Dependency lookup in both directions |
| `orchestration_task_dependencies` | `ix_orch_deps_reverse` | B-tree (`depends_on_id`) | "What depends on this task?" (downstream lookup) |
| `orchestration_events` | `ix_orch_events_run` | B-tree (`run_id`) | Event timeline for a mission |
| `orchestration_events` | `ix_orch_events_task` | B-tree (`task_id`) WHERE `task_id IS NOT NULL` | Events for a specific task |
| `orchestration_events` | `ix_orch_events_created` | BRIN (`created_at`) | Time-range queries on append-only data |
| `orchestration_events` | `ix_orch_events_type` | B-tree (`event_type`) | Filter by event type |

#### Existing Tables (new indexes)

| Table | Index | Type | Purpose |
|-------|-------|------|---------|
| `board_tasks` | `uq_board_tasks_orchestration_run` | Unique partial (`source_id`) WHERE `source_type = 'orchestration'` | Idempotent mission board task creation |
| `board_tasks` | `uq_board_tasks_orchestration_task` | Unique partial (`source_id`) WHERE `source_type = 'orchestration_task'` | Idempotent task board task creation |
| `agent_reports` | `ix_agent_reports_orch_task` | B-tree (`orchestration_task_id`) WHERE `orchestration_task_id IS NOT NULL` | "Reports from this task" |

### 7.7 Migration Safety

The orchestration migration creates **only new tables and adds optional columns/indexes to existing tables**. No existing columns are modified or removed.

#### 7.7.1 Migration Strategy

One migration file: `prd101_orchestration_tables.py` (following the PRD-prefixed naming convention from recent migrations like `prd76_agent_reports.py`, `prd77_agent_scheduled_tasks.py`).

```python
revision = "prd101_orchestration_tables"
down_revision = None  # Standalone — safe to run in any order
```

**Standalone migration** (`down_revision = None`) — same pattern as prd76, prd77, prd79. No dependency on other migrations. Safe to run against any database state.

#### 7.7.2 Safety Guarantees

| Risk | Mitigation |
|------|-----------|
| Table already exists (re-run) | `CREATE TABLE IF NOT EXISTS` on all tables |
| Index already exists | `CREATE INDEX IF NOT EXISTS` on all indexes |
| FK target doesn't exist | Tables created in dependency order: runs → tasks → dependencies → events |
| Existing `board_tasks` rows | No column changes to `board_tasks` — only new indexes added |
| Existing `agent_reports` rows | New `orchestration_task_id` column is `NULL` by default — no backfill needed |
| Downgrade safety | `DROP TABLE IF EXISTS` in reverse dependency order: events → dependencies → tasks → runs |
| Production lock time | `CREATE INDEX CONCURRENTLY` for indexes on existing tables (board_tasks, agent_reports) to avoid locking |

#### 7.7.3 Changes to Existing Tables

| Table | Change | Risk Level | Notes |
|-------|--------|-----------|-------|
| `board_tasks` | 2 new partial unique indexes | **Low** — additive only | Use `CREATE INDEX CONCURRENTLY` to avoid table lock |
| `agent_reports` | 1 new nullable column + 1 partial index | **Low** — nullable column, no default | `ALTER TABLE ADD COLUMN IF NOT EXISTS` is fast (metadata-only for nullable columns in PostgreSQL) |

No changes to `agents`, `workspaces`, `workflow_recipes`, or any other existing table. The `workflow_recipes` table gains no new columns — the "save as routine" flow creates a new row using existing columns, with mission lineage stored in the `execution_config` JSONB.

#### 7.7.4 Raw SQL Style

Following the codebase convention (prd76, prd79), the migration uses `op.execute()` with raw SQL rather than `op.create_table()`:

```python
def upgrade() -> None:
    # 1. New tables (raw SQL — clean FK syntax, IF NOT EXISTS)
    op.execute("""
        CREATE TABLE IF NOT EXISTS orchestration_runs ( ... );
        CREATE TABLE IF NOT EXISTS orchestration_tasks ( ... );
        CREATE TABLE IF NOT EXISTS orchestration_task_dependencies ( ... );
        CREATE TABLE IF NOT EXISTS orchestration_events ( ... );
    """)

    # 2. Indexes on new tables
    op.execute(""" CREATE INDEX IF NOT EXISTS ... """)

    # 3. Alterations to existing tables (CONCURRENTLY for production safety)
    op.execute("""
        ALTER TABLE agent_reports
            ADD COLUMN IF NOT EXISTS orchestration_task_id UUID
            REFERENCES orchestration_tasks(id) ON DELETE SET NULL;
    """)

    # Note: CONCURRENTLY indexes must be created outside transaction
    # op.execute("COMMIT")  -- if needed for CONCURRENTLY
    # op.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS ...")


def downgrade() -> None:
    op.execute("ALTER TABLE agent_reports DROP COLUMN IF EXISTS orchestration_task_id;")
    op.execute("DROP TABLE IF EXISTS orchestration_events;")
    op.execute("DROP TABLE IF EXISTS orchestration_task_dependencies;")
    op.execute("DROP TABLE IF EXISTS orchestration_tasks;")
    op.execute("DROP TABLE IF EXISTS orchestration_runs;")
```

### 7.8 Design Decisions Summary

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Board task granularity | 1 parent (mission) + N children (tasks) | 1 flat board task per mission | Children use existing `parent_task_id` FK for hierarchy. Individual task visibility matches PRD-100's "every task visible on board" requirement. |
| Board task sync | Side effect in `transition_task()` | Separate sync job / CDC | Same-transaction update is atomic and simple. No eventual consistency lag. Matches recipe bridge pattern. |
| `source_type` values | `orchestration` + `orchestration_task` | Single `orchestration` value for both | Separate values enable "show only missions" vs "show only tasks" filters. Partial unique indexes need distinct values. |
| Report auto-creation | On terminal state in `transition_task()` | Agent calls `platform_submit_report` explicitly | Auto-creation ensures every task gets a report. Agent can still call `platform_submit_report` for richer content — the auto-report is a fallback. |
| Report FK | New `orchestration_task_id` on `agent_reports` | Reuse `heartbeat_result_id` | Explicit FK enables "reports for this mission" queries. Heartbeat and orchestration are separate execution contexts. |
| Recipe conversion | Snapshot with manual editing | Parameterized template auto-generation | First version is a snapshot — auto-parameterization requires prompt analysis that's out of scope for PRD-101. Users edit the recipe to add variables. |
| Workspace isolation | Application-layer filtering only | PostgreSQL RLS | Matches existing pattern across all tables. RLS would be a platform-wide decision, not per-feature. |
| Migration strategy | Standalone (`down_revision = None`) | Chained to previous migration | Standalone migrations are the recent convention (prd76+). Avoids merge head conflicts. |
| Existing table changes | Additive only (new columns, indexes) | Modify existing columns | Zero risk to existing functionality. No backfill migrations. No downtime. |

---

## 8. Open Questions

These are unresolved design decisions surfaced during research. Each needs input before or during implementation (PRD-82A).

### 8.1 Schema Design Questions

| # | Question | Context | Options | Recommendation |
|---|----------|---------|---------|----------------|
| Q1 | **Should `orchestration_events` use BIGSERIAL or UUID for PK?** | BIGSERIAL is more natural for append-only tables (monotonic, BRIN-friendly, smaller indexes). UUID matches every other table in the codebase. | (a) BIGSERIAL — optimal for append-only workload (b) UUID — consistency with codebase | BIGSERIAL — event tables are different from entity tables. Append-only semantics favor monotonic keys. BRIN indexes on BIGSERIAL are ~100x smaller than B-tree on UUID. |
| Q2 | **Should `plan` JSONB on `orchestration_runs` be immutable after planning phase?** | Currently the design allows plan mutation during execution (e.g., coordinator adds tasks dynamically). But immutable plans are easier to reason about and debug. | (a) Immutable after `RUNNING` — changes create a new plan version (b) Mutable — coordinator updates in place (c) Append-only plan history (array of versions) | Mutable for v1 — dynamic replanning is a Phase 2 feature (PRD-102) and immutability would block it. Revisit when PRD-102 defines coordinator behavior. |
| Q3 | **How should large task outputs be stored?** | Section 5.1 says outputs go to `output_ref` (workspace file path or report ID). But the workspace file system is per-agent, and missions span multiple agents. | (a) Workspace files under `/missions/{run_id}/` directory (b) Agent reports via `platform_submit_report` (c) S3 directly (d) JSONB column with size limit | (a) or (b) — workspace files for raw output, agent reports for structured results. PRD-76 already built the report pipeline. Avoid S3 for v1 complexity. |
| Q4 | **Should `orchestration_task_dependencies` support weighted edges?** | Current design has `dependency_type` (strict/soft). Weighted edges could express "80% confidence this dependency is needed" for AI-planned graphs. | (a) Keep simple — strict/soft only (b) Add `weight` FLOAT column | (a) — YAGNI. AI-planned dependency confidence is a PRD-102 concern. The schema can add a column later without migration risk. |

### 8.2 Integration Questions

| # | Question | Context | Options | Recommendation |
|---|----------|---------|---------|----------------|
| Q5 | **Should board task creation be synchronous (in transition_task) or async (event-driven)?** | Section 7.2.5 defines synchronous board task creation inside `transition_task()`. This couples orchestration to board logic. An event-driven approach (listen to `orchestration_events`) decouples them but adds eventual consistency. | (a) Synchronous — same transaction, guaranteed consistency (b) Async — event consumer creates board tasks | (a) for v1 — the board task bridge is 3 functions and matches the existing `board_task_bridge.py` pattern from recipes. Decoupling adds complexity without benefit at our scale. |
| Q6 | **Should recipe conversion (Section 7.4) preserve agent assignments or parameterize them?** | A mission assigns specific roster agents. When converted to a recipe, should it lock to those agents or use role-based placeholders? | (a) Snapshot — lock agent IDs (b) Parameterize — convert to role placeholders like `{researcher}` (c) Snapshot with manual editing | (c) — snapshot first, let users edit. Auto-parameterization requires prompt analysis that's out of scope. |
| Q7 | **Should `workspace_id` on `orchestration_tasks` be denormalized?** | Section 7.5.3 argues for denormalization (avoids JOIN to `orchestration_runs` on every task query). But it's redundant data. | Already decided: Yes, denormalize. | Confirmed — every task query needs workspace scoping. The JOIN cost is small but avoidable, and the pattern matches `board_tasks` which also denormalizes `workspace_id`. |

### 8.3 Operational Questions

| # | Question | Context | Options | Recommendation |
|---|----------|---------|---------|----------------|
| Q8 | **What's the event retention period?** | Section 6.7 proposes 90 days hot + archive. But we don't have an archive mechanism today. | (a) 90 days, then DELETE (b) 90 days, then archive to S3/cold storage (c) Keep everything (small scale) | (c) for v1 — at <1000 missions/month, event volume is negligible. Revisit when PRD-106 telemetry analysis defines data retention needs. |
| Q9 | **Should the stall detection reconciler be a new service or extend `task_reconciler.py`?** | Section 3.7 proposes extending the existing reconciler. But orchestration stall detection has different logic than recipe task reconciliation. | (a) Extend `task_reconciler.py` — add orchestration-specific tick (b) New `orchestration_reconciler.py` — separate concerns | (b) — separate file, same APScheduler infrastructure. The reconciliation logic is different enough to warrant its own module. Register it as a new scheduled tick alongside the existing one. |
| Q10 | **Should we add a `tags` JSONB column or a separate `orchestration_run_tags` table?** | Section 2.4 notes Dagster uses `run_tags` table for filterable metadata. JSONB is simpler but harder to index for arbitrary key lookups. | (a) JSONB column on `orchestration_runs` (b) Separate tags table with `(run_id, key, value)` | (a) for v1 — GIN index on JSONB handles our query patterns. Separate table is warranted only if we need cross-run tag aggregation queries, which is a PRD-106 concern. |

---

## 9. Risk Register

| # | Risk | Impact | Likelihood | Mitigation | Detected During |
|---|------|--------|------------|------------|-----------------|
| R1 | **JSONB schema drift** — `plan` and `config` JSONB columns evolve without validation, leading to coordinator crashes on old data | High | High | Pydantic models with `model_validate()` on read. Version field in plan JSONB enables migration logic. Always add fields as optional with defaults. | Section 4.3 plan schema design |
| R2 | **Denormalized counter desync** — `task_count`, `tasks_completed`, `tasks_failed` on `orchestration_runs` drift from actual `orchestration_tasks` counts due to bugs or partial transaction failures | Medium | Medium | (1) Counters updated in same transaction as task state change. (2) Reconciler periodically re-derives counts from `SELECT COUNT(*) ... GROUP BY state` and corrects drift. (3) Dashboard queries can fall back to live COUNT if discrepancy detected. | Section 4.2 denormalized counters |
| R3 | **Optimistic locking contention** — high-frequency state transitions cause `StaleDataError` storms, especially during parallel task execution with fast completion | Medium | Low | At our scale (5-50 tasks/mission), state changes are seconds apart — contention is minimal. `transition_task()` returns `(False, refreshed_task)` on conflict — caller retries once with fresh state. If contention grows, switch hot-path transitions to `SELECT FOR UPDATE SKIP LOCKED`. | Section 3.9 concurrency safety |
| R4 | **Board task coupling** — synchronous board task updates inside `transition_task()` add latency and create a failure coupling between orchestration and board systems | Medium | Low | Board task sync is a simple `UPDATE` on an already-loaded row (same transaction, no extra round-trip). If board_task_id is NULL (task not yet planned), sync is skipped. Fallback: if board sync fails, orchestration state still commits — board is eventually consistent via reconciler. | Section 7.2.4 state synchronization |
| R5 | **Event table growth** — append-only `orchestration_events` grows unbounded if retention policy isn't implemented, degrading query performance on time-range scans | Low | High | BRIN index on `created_at` keeps time-range scans efficient regardless of table size. At projected volume (~1M events/year), PostgreSQL handles this comfortably. Archive strategy (Section 6.7) is documented but deferred to PRD-106 which defines retention needs. Monitor table size via `pg_total_relation_size()`. | Section 6.7 retention policy |
| R6 | **Migration lock on `board_tasks`** — adding indexes to the existing `board_tasks` table during migration could lock the table in production | High | Medium | Use `CREATE INDEX CONCURRENTLY` for all indexes on existing tables. This requires the index creation to run outside a transaction (`op.execute("COMMIT")` before `CREATE INDEX CONCURRENTLY`). Test migration on staging with production-sized data before deploying. | Section 7.7.4 raw SQL style |
| R7 | **Trigger rule complexity creep** — starting with 4 trigger rules invites requests for more (`one_success`, `one_failed`, branching logic), increasing coordinator complexity | Low | Medium | 4 rules are stored as `VARCHAR(30)` — adding new values requires only a code change in the trigger rule evaluator, not a schema migration. Gate new rules behind feature flags. Document that complex conditional logic belongs in the coordinator's planning prompt (PRD-102), not in trigger rules. | Section 5.3 trigger rules |
| R8 | **Recipe conversion loses fidelity** — converting a mission DAG to a recipe flattens parallel execution into sequential steps, losing the execution structure that made the mission succeed | Medium | High | Document the limitation explicitly (Section 7.4.4). Recipe `execution_config.mode` preserves the strategy hint. Full DAG-aware recipes are a future enhancement — the recipe engine would need to understand dependencies natively (not in scope for PRD-101 or 82A). | Section 7.4.4 limitations |
| R9 | **Circular dependency at runtime** — coordinator dynamically adds tasks during execution that create a cycle not caught at planning time | High | Low | `graphlib.TopologicalSorter.prepare()` is called after every dynamic task addition, not just at initial planning. If a cycle is detected, the new task is rejected with an error event. The `CHECK (task_id != depends_on_id)` constraint catches self-references at the DB level. | Section 5.5 cycle detection |
| R10 | **`output_ref` path breakage** — task output stored at a workspace file path is moved or deleted, leaving `output_ref` pointing to nothing | Low | Medium | `output_ref` is a reference, not a guarantee. The report service (PRD-76) also stores content in `agent_reports` — the report is the durable artifact, the file is a convenience copy. Downstream tasks receive `output_summary` (inline text), not the file path. | Section 5.1 output storage |

---

## 10. Implementation Acceptance Criteria

These are testable criteria for PRD-82A (the implementation PRD that follows this research document). The schema is "done" when all criteria pass.

### 10.1 Database Schema

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| AC-1 | `orchestration_runs` table exists with all 22 columns from Section 4.2 | `\d orchestration_runs` matches column definitions |
| AC-2 | `orchestration_tasks` table exists with all 27 columns from Section 5.2 | `\d orchestration_tasks` matches column definitions |
| AC-3 | `orchestration_task_dependencies` table exists with composite PK and self-reference CHECK | `\d orchestration_task_dependencies` shows PK + constraint |
| AC-4 | `orchestration_events` table exists with BIGSERIAL PK and all 8 columns from Section 6.3 | `\d orchestration_events` matches column definitions |
| AC-5 | `orchestration_events_archive` table exists (identical schema, no FKs) | `\d orchestration_events_archive` |
| AC-6 | All FK constraints enforced: runs→workspaces, runs→agents, tasks→runs, tasks→workspaces, tasks→agents, tasks→board_tasks, deps→tasks (both directions), events→runs, events→tasks | `SELECT conname FROM pg_constraint WHERE conrelid = ...` |
| AC-7 | All CHECK constraints enforced: non-negative tokens/cost/counts, score range 0-1, attempt ≥ 1, no self-referencing deps | `INSERT` violating each CHECK returns error |
| AC-8 | All indexes from Section 7.6 exist, including partial indexes with correct WHERE clauses | `\di` + `pg_indexes` query for WHERE clause text |
| AC-9 | `agent_reports.orchestration_task_id` column exists (nullable UUID FK → orchestration_tasks) | `\d agent_reports` shows new column |
| AC-10 | Partial unique indexes on `board_tasks` for `source_type = 'orchestration'` and `source_type = 'orchestration_task'` | Duplicate INSERT returns unique violation |

### 10.2 SQLAlchemy Models

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| AC-11 | `OrchestrationRun` model importable from `orchestrator.core.models` | `from orchestrator.core.models import OrchestrationRun` succeeds |
| AC-12 | `OrchestrationTask` model importable with relationships to run, agent, board_task | Model introspection shows relationships |
| AC-13 | `OrchestrationTaskDependency` model importable with composite PK | Model metadata shows composite PK |
| AC-14 | `OrchestrationEvent` model importable with BIGSERIAL PK | Model metadata shows autoincrement Integer PK |
| AC-15 | `version_id_col` configured on OrchestrationRun and OrchestrationTask for optimistic locking | `__mapper_args__` includes `version_id_col` |
| AC-16 | All Python enums (`StateType`, `RunState`, `TaskState`, `EventType`, `ActorType`, `TaskType`, `TriggerRule`) defined and importable | Import and iterate all enum values |

### 10.3 State Machine

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| AC-17 | `transition_task()` enforces allowed transitions from Section 3.10 — invalid transitions raise `InvalidTransition` | Unit test: attempt every invalid transition, assert error |
| AC-18 | `transition_run()` enforces allowed transitions — invalid transitions raise `InvalidTransition` | Unit test: attempt every invalid transition, assert error |
| AC-19 | Every state transition emits an `OrchestrationEvent` in the same transaction | Unit test: transition + query events in same session, assert event exists |
| AC-20 | Terminal state transitions set `completed_at` and `duration_ms` | Unit test: transition to completed/failed/cancelled, assert timestamps set |
| AC-21 | `StaleDataError` on concurrent modification returns `(False, refreshed_task)` instead of raising | Unit test: load task in two sessions, modify in one, attempt transition in other |
| AC-22 | Board task status synced on every orchestration task state change (per mapping in Section 3.8) | Integration test: transition task, query board_task, assert status matches |

### 10.4 Integration

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| AC-23 | `create_mission_board_task()` creates parent board_task with `source_type='orchestration'` | Integration test: create run, call function, query board_tasks |
| AC-24 | `create_task_board_task()` creates child board_task with `parent_task_id` linking to mission board_task | Integration test: verify `parent_task_id` is set |
| AC-25 | Duplicate board_task creation is idempotent (returns existing ID) | Call `create_mission_board_task()` twice, assert same board_task.id |
| AC-26 | Auto-report creation on task completion writes to `agent_reports` with `orchestration_task_id` FK | Integration test: complete task, query `agent_reports` for matching FK |
| AC-27 | Workspace isolation: query for run_id belonging to workspace A with workspace B context returns 404 | Integration test: cross-workspace access attempt fails |

### 10.5 Migration

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| AC-28 | Migration runs successfully on empty database | `alembic upgrade head` on fresh DB |
| AC-29 | Migration runs successfully on production-like database (existing board_tasks, agent_reports, agents) | `alembic upgrade head` on staging with data |
| AC-30 | Downgrade cleanly removes all orchestration tables and the `agent_reports.orchestration_task_id` column | `alembic downgrade -1` + verify tables gone |
| AC-31 | Migration is re-runnable (`IF NOT EXISTS` on all CREATE statements) | Run migration twice without error |
| AC-32 | Existing board_tasks and agent_reports rows unaffected by migration | Row count before = row count after |

---

## 11. Dependencies & Sequencing

### 11.1 What Must Be Built First

The schema is the foundation — everything else depends on it:

```
PRD-101 (this document — schema design)
    │
    ▼
PRD-82A (implementation: Alembic migration, SQLAlchemy models, API endpoints)
    │
    ├── PRD-102 (Coordinator) — reads/writes orchestration_runs, creates orchestration_tasks
    ├── PRD-103 (Verification) — writes verifier_score, verified_by on orchestration_tasks
    ├── PRD-104 (Ephemeral Agents) — writes agent_type='contractor' on orchestration_tasks
    ├── PRD-105 (Budget) — reads/writes total_cost, config.budget on orchestration_runs
    └── PRD-106 (Telemetry) — reads orchestration_events for pattern analysis
```

### 11.2 Implementation Order for PRD-82A

Within the implementation PRD, build in this order:

| Phase | Deliverable | Depends On | Can Parallelize With |
|-------|------------|------------|---------------------|
| **1. Migration** | Alembic migration file creating all 4 tables + archive table + existing table alterations | Nothing | — |
| **2. Models** | SQLAlchemy model classes in `orchestrator/core/models/orchestration.py` | Phase 1 (tables must exist) | — |
| **3. Enums** | Python enums (`StateType`, `RunState`, `TaskState`, etc.) in `orchestrator/core/models/orchestration_enums.py` | Nothing (pure Python) | Phase 1 |
| **4. State machine** | `transition_task()`, `transition_run()`, `emit_event()` in `orchestrator/services/orchestration_state.py` | Phase 2, 3 | — |
| **5. Board bridge** | `create_mission_board_task()`, `create_task_board_task()` in `orchestrator/services/orchestration_board_bridge.py` | Phase 2 | Phase 4 |
| **6. Dependency resolver** | `DependencyResolver` class, `validate_task_graph()` in `orchestrator/services/orchestration_deps.py` | Phase 3 (uses TaskState enum) | Phase 4, 5 |
| **7. API endpoints** | CRUD endpoints for runs, tasks, events in `orchestrator/api/missions.py` | Phase 2, 4, 5, 6 | — |

### 11.3 What Can Be Deferred

These features are designed in this PRD but can be implemented incrementally:

| Feature | Section | Defer Until | Rationale |
|---------|---------|-------------|-----------|
| Trigger rules beyond `all_success` | 5.3 | PRD-82C (parallel execution) | Sequential missions only need `all_success`. Other rules enable parallel patterns. |
| Recipe conversion ("Save as Routine") | 7.4 | PRD-82D or later | Nice-to-have. Core mission execution works without it. |
| Event archive/retention | 6.7 | PRD-106 (telemetry defines needs) | At <1M events/year, no urgency. |
| `orchestration_events_archive` table | 6.10 | Same as above | Table exists (migration creates it) but archival job is deferred. |
| Report auto-creation | 7.3.2 | PRD-82B (coordinator builds this flow) | Depends on coordinator knowing when tasks complete. |
| Stall detection reconciler | 3.7 | PRD-82B (coordinator) | Reconciler is part of coordinator's tick loop. |

### 11.4 Cross-PRD Interface Contracts

These are the columns/fields that downstream PRDs will write to. The schema must support them even if this PRD doesn't populate them:

| Column | Written By | PRD |
|--------|-----------|-----|
| `orchestration_tasks.verifier_score` | Verifier agent | PRD-103 |
| `orchestration_tasks.verified_by` | Verifier agent or human | PRD-103 |
| `orchestration_tasks.agent_type = 'contractor'` | Coordinator | PRD-104 |
| `orchestration_tasks.model_override` | Coordinator (model routing) | PRD-104 |
| `orchestration_runs.total_cost` / `total_tokens` | Budget enforcer | PRD-105 |
| `orchestration_runs.config.budget.*` | User / Budget enforcer | PRD-105 |
| `orchestration_events` (all event types) | Telemetry pipeline reads | PRD-106 |

---

## 12. Appendix: Full SQL DDL

Complete `CREATE TABLE` statements ready to convert to Alembic. These follow the codebase convention of raw SQL via `op.execute()`.

```sql
-- ============================================================
-- PRD-101: Mission Schema & Data Model
-- Complete DDL for orchestration tables
-- ============================================================

-- 1. orchestration_runs — mission-level execution records
CREATE TABLE IF NOT EXISTS orchestration_runs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id            UUID            NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    title                   VARCHAR(500)    NOT NULL,
    description             TEXT,
    goal                    TEXT            NOT NULL,
    state                   VARCHAR(30)     NOT NULL DEFAULT 'pending',
    state_type              VARCHAR(10)     NOT NULL DEFAULT 'pending',
    plan                    JSONB,
    config                  JSONB           NOT NULL DEFAULT '{}',
    result_summary          TEXT,
    error_message           TEXT,
    created_by              VARCHAR(255)    NOT NULL,
    coordinator_agent_id    INTEGER         REFERENCES agents(id) ON DELETE SET NULL,
    total_tokens            INTEGER         NOT NULL DEFAULT 0,
    total_cost              NUMERIC(10,6)   NOT NULL DEFAULT 0,
    task_count              INTEGER         NOT NULL DEFAULT 0,
    tasks_completed         INTEGER         NOT NULL DEFAULT 0,
    tasks_failed            INTEGER         NOT NULL DEFAULT 0,
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    duration_ms             INTEGER,
    version_id              INTEGER         NOT NULL DEFAULT 1,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT ck_orch_runs_tokens_positive          CHECK (total_tokens >= 0),
    CONSTRAINT ck_orch_runs_cost_positive             CHECK (total_cost >= 0),
    CONSTRAINT ck_orch_runs_task_count_positive        CHECK (task_count >= 0),
    CONSTRAINT ck_orch_runs_tasks_completed_positive   CHECK (tasks_completed >= 0),
    CONSTRAINT ck_orch_runs_tasks_failed_positive      CHECK (tasks_failed >= 0)
);

COMMENT ON TABLE orchestration_runs IS
    'Mission-level execution records (PRD-101). One row per mission attempt.';

-- Indexes
CREATE INDEX IF NOT EXISTS ix_orch_runs_workspace_state
    ON orchestration_runs (workspace_id, state_type)
    WHERE state_type != 'terminal';

CREATE INDEX IF NOT EXISTS ix_orch_runs_workspace_completed
    ON orchestration_runs (workspace_id, completed_at DESC)
    WHERE state_type = 'terminal';

CREATE INDEX IF NOT EXISTS ix_orch_runs_created_by
    ON orchestration_runs (workspace_id, created_by);

CREATE INDEX IF NOT EXISTS ix_orch_runs_state_updated
    ON orchestration_runs (state, updated_at)
    WHERE state_type != 'terminal';


-- 2. orchestration_tasks — task-level execution records
CREATE TABLE IF NOT EXISTS orchestration_tasks (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID            NOT NULL REFERENCES orchestration_runs(id) ON DELETE CASCADE,
    workspace_id        UUID            NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    sequence_number     SMALLINT        NOT NULL,
    title               VARCHAR(500)    NOT NULL,
    description         TEXT,
    task_type           VARCHAR(30)     NOT NULL,
    state               VARCHAR(30)     NOT NULL DEFAULT 'pending',
    state_type          VARCHAR(10)     NOT NULL DEFAULT 'pending',
    trigger_rule        VARCHAR(30)     NOT NULL DEFAULT 'all_success',
    agent_id            INTEGER         REFERENCES agents(id) ON DELETE SET NULL,
    agent_type          VARCHAR(20),
    model_override      VARCHAR(255),
    tools_requested     JSONB,
    success_criteria    TEXT,
    output_summary      VARCHAR(2000),
    output_ref          VARCHAR(500),
    verifier_score      NUMERIC(3,2),
    verified_by         VARCHAR(255),
    error_message       TEXT,
    attempt_number      SMALLINT        NOT NULL DEFAULT 1,
    continuation_count  SMALLINT        NOT NULL DEFAULT 0,
    tokens_used         INTEGER         NOT NULL DEFAULT 0,
    cost                NUMERIC(10,6)   NOT NULL DEFAULT 0,
    board_task_id       INTEGER         REFERENCES board_tasks(id) ON DELETE SET NULL,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    duration_ms         INTEGER,
    version_id          INTEGER         NOT NULL DEFAULT 1,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT ck_orch_tasks_score_range         CHECK (verifier_score >= 0 AND verifier_score <= 1),
    CONSTRAINT ck_orch_tasks_attempt_positive     CHECK (attempt_number >= 1),
    CONSTRAINT ck_orch_tasks_continuation_positive CHECK (continuation_count >= 0),
    CONSTRAINT ck_orch_tasks_tokens_positive       CHECK (tokens_used >= 0),
    CONSTRAINT ck_orch_tasks_cost_positive         CHECK (cost >= 0)
);

COMMENT ON TABLE orchestration_tasks IS
    'Task-level execution records within a mission (PRD-101).';

-- Indexes
CREATE INDEX IF NOT EXISTS ix_orch_tasks_run_id
    ON orchestration_tasks (run_id);

CREATE INDEX IF NOT EXISTS ix_orch_tasks_run_state
    ON orchestration_tasks (run_id, state_type)
    WHERE state_type != 'terminal';

CREATE INDEX IF NOT EXISTS ix_orch_tasks_agent
    ON orchestration_tasks (agent_id, state)
    WHERE agent_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_orch_tasks_workspace
    ON orchestration_tasks (workspace_id, state_type);

CREATE INDEX IF NOT EXISTS ix_orch_tasks_board_task
    ON orchestration_tasks (board_task_id)
    WHERE board_task_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_orch_tasks_state_updated
    ON orchestration_tasks (state, updated_at)
    WHERE state_type NOT IN ('terminal', 'pending');


-- 3. orchestration_task_dependencies — DAG edges
CREATE TABLE IF NOT EXISTS orchestration_task_dependencies (
    task_id          UUID            NOT NULL REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
    depends_on_id    UUID            NOT NULL REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
    dependency_type  VARCHAR(20)     NOT NULL DEFAULT 'data',
    created_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (task_id, depends_on_id),
    CONSTRAINT ck_task_deps_no_self_ref CHECK (task_id != depends_on_id)
);

COMMENT ON TABLE orchestration_task_dependencies IS
    'DAG edges between orchestration tasks. Join table for dependency resolution (PRD-101).';

CREATE INDEX IF NOT EXISTS ix_task_deps_depends_on
    ON orchestration_task_dependencies (depends_on_id);


-- 4. orchestration_events — append-only audit trail
CREATE TABLE IF NOT EXISTS orchestration_events (
    id              BIGSERIAL       PRIMARY KEY,
    run_id          UUID            NOT NULL REFERENCES orchestration_runs(id) ON DELETE CASCADE,
    task_id         UUID            REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
    event_type      VARCHAR(50)     NOT NULL,
    payload         JSONB           NOT NULL DEFAULT '{}',
    actor_type      VARCHAR(20)     NOT NULL DEFAULT 'system',
    actor_id        VARCHAR(255),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE orchestration_events IS
    'Append-only event log for mission lifecycle (PRD-101). Never UPDATE or DELETE from application code.';

-- Indexes
CREATE INDEX IF NOT EXISTS ix_orch_events_run_id
    ON orchestration_events (run_id, id);

CREATE INDEX IF NOT EXISTS ix_orch_events_task_id
    ON orchestration_events (task_id, id)
    WHERE task_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_orch_events_type_created
    ON orchestration_events (event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_orch_events_created_brin
    ON orchestration_events USING BRIN (created_at)
    WITH (pages_per_range = 32);


-- 5. orchestration_events_archive — cold storage (no FKs)
CREATE TABLE IF NOT EXISTS orchestration_events_archive (
    id              BIGINT          PRIMARY KEY,  -- preserves original IDs
    run_id          UUID            NOT NULL,
    task_id         UUID,
    event_type      VARCHAR(50)     NOT NULL,
    payload         JSONB           NOT NULL DEFAULT '{}',
    actor_type      VARCHAR(20)     NOT NULL DEFAULT 'system',
    actor_id        VARCHAR(255),
    created_at      TIMESTAMPTZ     NOT NULL
);

COMMENT ON TABLE orchestration_events_archive IS
    'Cold storage for orchestration events older than 180 days (PRD-101).';

CREATE INDEX IF NOT EXISTS ix_orch_events_archive_run
    ON orchestration_events_archive (run_id);

CREATE INDEX IF NOT EXISTS ix_orch_events_archive_created
    ON orchestration_events_archive (created_at);


-- 6. Alterations to existing tables

-- agent_reports: link reports to orchestration tasks
ALTER TABLE agent_reports
    ADD COLUMN IF NOT EXISTS orchestration_task_id UUID
    REFERENCES orchestration_tasks(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS ix_agent_reports_orch_task
    ON agent_reports (orchestration_task_id)
    WHERE orchestration_task_id IS NOT NULL;

-- board_tasks: idempotent mission/task board task creation
-- NOTE: Use CREATE INDEX CONCURRENTLY in production to avoid table lock
CREATE UNIQUE INDEX IF NOT EXISTS uq_board_tasks_orchestration_run
    ON board_tasks (source_id) WHERE source_type = 'orchestration';

CREATE UNIQUE INDEX IF NOT EXISTS uq_board_tasks_orchestration_task
    ON board_tasks (source_id) WHERE source_type = 'orchestration_task';
```

---

## 13. Appendix: SQLAlchemy Models

Complete Python model classes matching the DDL above. These follow the codebase conventions documented in the model audit: `Base` from `core.database.base`, `PGUUID(as_uuid=True)` for UUIDs, `server_default` for all defaults, `DateTime(timezone=True)` for timestamps, no PostgreSQL ENUM types.

```python
"""PRD-101: Mission Schema & Data Model — SQLAlchemy Models.

Orchestration tables for Mission Mode: runs, tasks, dependencies, events.
All models use the project's established conventions:
- Base from core.database.base
- PGUUID(as_uuid=True) for UUID columns
- VARCHAR for enum columns (no DB ENUM types)
- JSONB for structured data with server_default='{}'
- DateTime(timezone=True) with server_default=func.now()
- extend_existing=True in __table_args__
"""

from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship

from core.database.base import Base


# ---------------------------------------------------------------------------
# orchestration_runs
# ---------------------------------------------------------------------------

class OrchestrationRun(Base):
    """Mission-level execution record. One row per mission attempt."""

    __tablename__ = "orchestration_runs"
    __table_args__ = (
        Index(
            "ix_orch_runs_workspace_state",
            "workspace_id", "state_type",
            postgresql_where="state_type != 'terminal'",
        ),
        Index(
            "ix_orch_runs_workspace_completed",
            "workspace_id", "completed_at",
            postgresql_where="state_type = 'terminal'",
        ),
        Index("ix_orch_runs_created_by", "workspace_id", "created_by"),
        Index(
            "ix_orch_runs_state_updated",
            "state", "updated_at",
            postgresql_where="state_type != 'terminal'",
        ),
        CheckConstraint("total_tokens >= 0", name="ck_orch_runs_tokens_positive"),
        CheckConstraint("total_cost >= 0", name="ck_orch_runs_cost_positive"),
        CheckConstraint("task_count >= 0", name="ck_orch_runs_task_count_positive"),
        CheckConstraint("tasks_completed >= 0", name="ck_orch_runs_tasks_completed_positive"),
        CheckConstraint("tasks_failed >= 0", name="ck_orch_runs_tasks_failed_positive"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    goal = Column(Text, nullable=False)
    state = Column(String(30), nullable=False, default="pending", server_default="pending")
    state_type = Column(String(10), nullable=False, default="pending", server_default="pending")
    plan = Column(JSONB, nullable=True)
    config = Column(JSONB, nullable=False, default=dict, server_default="{}")
    result_summary = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_by = Column(String(255), nullable=False)
    coordinator_agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )
    total_tokens = Column(Integer, nullable=False, default=0, server_default="0")
    total_cost = Column(Numeric(10, 6), nullable=False, default=0, server_default="0")
    task_count = Column(Integer, nullable=False, default=0, server_default="0")
    tasks_completed = Column(Integer, nullable=False, default=0, server_default="0")
    tasks_failed = Column(Integer, nullable=False, default=0, server_default="0")
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)
    version_id = Column(Integer, nullable=False, default=1, server_default="1")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False,
    )

    __mapper_args__ = {"version_id_col": version_id}

    # Relationships
    workspace = relationship("Workspace", foreign_keys=[workspace_id])
    coordinator_agent = relationship("Agent", foreign_keys=[coordinator_agent_id])
    tasks = relationship(
        "OrchestrationTask", back_populates="run", cascade="all, delete-orphan",
    )
    events = relationship(
        "OrchestrationEvent", back_populates="run", cascade="all, delete-orphan",
    )


# ---------------------------------------------------------------------------
# orchestration_tasks
# ---------------------------------------------------------------------------

class OrchestrationTask(Base):
    """Task-level execution record within a mission."""

    __tablename__ = "orchestration_tasks"
    __table_args__ = (
        Index("ix_orch_tasks_run_id", "run_id"),
        Index(
            "ix_orch_tasks_run_state", "run_id", "state_type",
            postgresql_where="state_type != 'terminal'",
        ),
        Index(
            "ix_orch_tasks_agent", "agent_id", "state",
            postgresql_where="agent_id IS NOT NULL",
        ),
        Index("ix_orch_tasks_workspace", "workspace_id", "state_type"),
        Index(
            "ix_orch_tasks_board_task", "board_task_id",
            postgresql_where="board_task_id IS NOT NULL",
        ),
        Index(
            "ix_orch_tasks_state_updated", "state", "updated_at",
            postgresql_where="state_type NOT IN ('terminal', 'pending')",
        ),
        CheckConstraint(
            "verifier_score >= 0 AND verifier_score <= 1",
            name="ck_orch_tasks_score_range",
        ),
        CheckConstraint("attempt_number >= 1", name="ck_orch_tasks_attempt_positive"),
        CheckConstraint("continuation_count >= 0", name="ck_orch_tasks_continuation_positive"),
        CheckConstraint("tokens_used >= 0", name="ck_orch_tasks_tokens_positive"),
        CheckConstraint("cost >= 0", name="ck_orch_tasks_cost_positive"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("orchestration_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    workspace_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    sequence_number = Column(SmallInteger, nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(String(30), nullable=False)
    state = Column(String(30), nullable=False, default="pending", server_default="pending")
    state_type = Column(String(10), nullable=False, default="pending", server_default="pending")
    trigger_rule = Column(String(30), nullable=False, default="all_success", server_default="all_success")
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    agent_type = Column(String(20), nullable=True)
    model_override = Column(String(255), nullable=True)
    tools_requested = Column(JSONB, nullable=True)
    success_criteria = Column(Text, nullable=True)
    output_summary = Column(String(2000), nullable=True)
    output_ref = Column(String(500), nullable=True)
    verifier_score = Column(Numeric(3, 2), nullable=True)
    verified_by = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)
    attempt_number = Column(SmallInteger, nullable=False, default=1, server_default="1")
    continuation_count = Column(SmallInteger, nullable=False, default=0, server_default="0")
    tokens_used = Column(Integer, nullable=False, default=0, server_default="0")
    cost = Column(Numeric(10, 6), nullable=False, default=0, server_default="0")
    board_task_id = Column(
        Integer, ForeignKey("board_tasks.id", ondelete="SET NULL"), nullable=True,
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)
    version_id = Column(Integer, nullable=False, default=1, server_default="1")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False,
    )

    __mapper_args__ = {"version_id_col": version_id}

    # Relationships
    run = relationship("OrchestrationRun", back_populates="tasks")
    workspace = relationship("Workspace", foreign_keys=[workspace_id])
    agent = relationship("Agent", foreign_keys=[agent_id])
    board_task = relationship("BoardTask", foreign_keys=[board_task_id])
    events = relationship(
        "OrchestrationEvent", back_populates="task", cascade="all, delete-orphan",
    )


# ---------------------------------------------------------------------------
# orchestration_task_dependencies
# ---------------------------------------------------------------------------

class OrchestrationTaskDependency(Base):
    """DAG edge: task_id depends on depends_on_id."""

    __tablename__ = "orchestration_task_dependencies"
    __table_args__ = (
        Index("ix_task_deps_depends_on", "depends_on_id"),
        CheckConstraint("task_id != depends_on_id", name="ck_task_deps_no_self_ref"),
        {"extend_existing": True},
    )

    task_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("orchestration_tasks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    depends_on_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("orchestration_tasks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    dependency_type = Column(String(20), nullable=False, default="data", server_default="data")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    task = relationship(
        "OrchestrationTask", foreign_keys=[task_id],
        backref="downstream_deps",
    )
    depends_on = relationship(
        "OrchestrationTask", foreign_keys=[depends_on_id],
        backref="upstream_deps",
    )


# ---------------------------------------------------------------------------
# orchestration_events
# ---------------------------------------------------------------------------

class OrchestrationEvent(Base):
    """Append-only event log for mission lifecycle tracking.

    Events are NEVER updated or deleted by application code.
    Every state transition emits an event in the same transaction
    (dual-write pattern, PRD-101 Section 2.4).
    """

    __tablename__ = "orchestration_events"
    __table_args__ = (
        Index("ix_orch_events_run_id", "run_id", "id"),
        Index(
            "ix_orch_events_task_id", "task_id", "id",
            postgresql_where="task_id IS NOT NULL",
        ),
        Index("ix_orch_events_type_created", "event_type", "created_at"),
        {"extend_existing": True},
    )
    # NOTE: BRIN index on created_at must be created via raw SQL in migration:
    # CREATE INDEX ix_orch_events_created_brin ON orchestration_events USING BRIN (created_at)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("orchestration_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    task_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("orchestration_tasks.id", ondelete="CASCADE"),
        nullable=True,
    )
    event_type = Column(String(50), nullable=False)
    payload = Column(JSONB, nullable=False, default=dict, server_default="{}")
    actor_type = Column(String(20), nullable=False, default="system", server_default="system")
    actor_id = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # No version_id — events are immutable
    # No updated_at — events are never modified

    # Relationships
    run = relationship("OrchestrationRun", back_populates="events")
    task = relationship("OrchestrationTask", back_populates="events")
```

### 13.1 Model Registration

Add to `orchestrator/core/models/__init__.py`:

```python
from .orchestration import (
    OrchestrationRun,
    OrchestrationTask,
    OrchestrationTaskDependency,
    OrchestrationEvent,
)
```

### 13.2 Enum Definitions

Place in `orchestrator/core/models/orchestration_enums.py` (separate file to avoid circular imports):

```python
"""PRD-101: Orchestration enums for state machines, event types, and task classification."""

from enum import StrEnum


class StateType(StrEnum):
    """Stable orchestration logic enum (4 values). Coordinator switches on this."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINAL = "terminal"


class RunState(StrEnum):
    """Rich display state for orchestration_runs."""
    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    PAUSED = "paused"
    BUDGET_EXCEEDED = "budget_exceeded"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskState(StrEnum):
    """Rich display state for orchestration_tasks."""
    PENDING = "pending"
    QUEUED = "queued"
    AWAITING_RETRY = "awaiting_retry"
    ASSIGNED = "assigned"
    RUNNING = "running"
    CONTINUING = "continuing"
    VERIFYING = "verifying"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TaskType(StrEnum):
    """Classification of orchestration task work."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    CODING = "coding"
    VERIFICATION = "verification"
    REVIEW = "review"
    SYNTHESIS = "synthesis"
    OTHER = "other"


class TriggerRule(StrEnum):
    """When a task's dependencies are considered met."""
    ALL_SUCCESS = "all_success"
    ALL_DONE = "all_done"
    NONE_FAILED = "none_failed"
    ALWAYS = "always"


class AgentType(StrEnum):
    """Whether an agent is permanent or mission-scoped."""
    ROSTER = "roster"
    CONTRACTOR = "contractor"


class EventType(StrEnum):
    """Typed event categories for orchestration_events."""
    # Run lifecycle
    RUN_CREATED = "run_created"
    RUN_PLANNING_STARTED = "run_planning_started"
    RUN_PLAN_READY = "run_plan_ready"
    RUN_APPROVED = "run_approved"
    RUN_REJECTED = "run_rejected"
    RUN_STARTED = "run_started"
    RUN_PAUSED = "run_paused"
    RUN_RESUMED = "run_resumed"
    RUN_BUDGET_WARNING = "run_budget_warning"
    RUN_BUDGET_EXCEEDED = "run_budget_exceeded"
    RUN_BUDGET_INCREASED = "run_budget_increased"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"

    # Task lifecycle
    TASK_CREATED = "task_created"
    TASK_QUEUED = "task_queued"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_CONTINUING = "task_continuing"
    TASK_RESUMED = "task_resumed"
    TASK_OUTPUT_SUBMITTED = "task_output_submitted"
    TASK_VERIFICATION_STARTED = "task_verification_started"
    TASK_VERIFICATION_PASSED = "task_verification_passed"
    TASK_VERIFICATION_FAILED = "task_verification_failed"
    TASK_HUMAN_REVIEW_REQUESTED = "task_human_review_requested"
    TASK_HUMAN_APPROVED = "task_human_approved"
    TASK_HUMAN_REJECTED = "task_human_rejected"
    TASK_RETRYING = "task_retrying"
    TASK_CRASHED = "task_crashed"
    TASK_FAILED = "task_failed"
    TASK_SKIPPED = "task_skipped"
    TASK_CANCELLED = "task_cancelled"

    # System
    STALL_DETECTED = "stall_detected"
    MODEL_FALLBACK = "model_fallback"
    COST_SNAPSHOT = "cost_snapshot"


class ActorType(StrEnum):
    """Who triggered an orchestration event."""
    SYSTEM = "system"
    COORDINATOR = "coordinator"
    AGENT = "agent"
    VERIFIER = "verifier"
    HUMAN = "human"
    RECONCILER = "reconciler"


# ---- State mappings ----

RUN_STATE_TYPE: dict[RunState, StateType] = {
    RunState.PENDING: StateType.PENDING,
    RunState.PLANNING: StateType.PENDING,
    RunState.AWAITING_APPROVAL: StateType.PENDING,
    RunState.RUNNING: StateType.RUNNING,
    RunState.PAUSED: StateType.PAUSED,
    RunState.BUDGET_EXCEEDED: StateType.PAUSED,
    RunState.COMPLETED: StateType.TERMINAL,
    RunState.FAILED: StateType.TERMINAL,
    RunState.CANCELLED: StateType.TERMINAL,
}

TASK_STATE_TYPE: dict[TaskState, StateType] = {
    TaskState.PENDING: StateType.PENDING,
    TaskState.QUEUED: StateType.PENDING,
    TaskState.AWAITING_RETRY: StateType.PENDING,
    TaskState.ASSIGNED: StateType.RUNNING,
    TaskState.RUNNING: StateType.RUNNING,
    TaskState.CONTINUING: StateType.RUNNING,
    TaskState.VERIFYING: StateType.PAUSED,
    TaskState.AWAITING_HUMAN: StateType.PAUSED,
    TaskState.COMPLETED: StateType.TERMINAL,
    TaskState.FAILED: StateType.TERMINAL,
    TaskState.CANCELLED: StateType.TERMINAL,
    TaskState.SKIPPED: StateType.TERMINAL,
}

TERMINAL_RUN_STATES = frozenset(
    s for s, t in RUN_STATE_TYPE.items() if t == StateType.TERMINAL
)
TERMINAL_TASK_STATES = frozenset(
    s for s, t in TASK_STATE_TYPE.items() if t == StateType.TERMINAL
)

# ---- Transition tables ----

ALLOWED_RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.PENDING:           frozenset({RunState.PLANNING, RunState.CANCELLED}),
    RunState.PLANNING:          frozenset({RunState.AWAITING_APPROVAL, RunState.RUNNING,
                                           RunState.FAILED, RunState.CANCELLED}),
    RunState.AWAITING_APPROVAL: frozenset({RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}),
    RunState.RUNNING:           frozenset({RunState.COMPLETED, RunState.FAILED, RunState.PAUSED,
                                           RunState.BUDGET_EXCEEDED, RunState.CANCELLED}),
    RunState.PAUSED:            frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.BUDGET_EXCEEDED:   frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.COMPLETED:         frozenset(),
    RunState.FAILED:            frozenset(),
    RunState.CANCELLED:         frozenset(),
}

ALLOWED_TASK_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING:        frozenset({TaskState.QUEUED, TaskState.SKIPPED, TaskState.CANCELLED}),
    TaskState.QUEUED:         frozenset({TaskState.ASSIGNED, TaskState.CANCELLED}),
    TaskState.ASSIGNED:       frozenset({TaskState.RUNNING, TaskState.CANCELLED}),
    TaskState.RUNNING:        frozenset({TaskState.CONTINUING, TaskState.VERIFYING,
                                         TaskState.AWAITING_RETRY, TaskState.FAILED,
                                         TaskState.CANCELLED}),
    TaskState.CONTINUING:     frozenset({TaskState.RUNNING, TaskState.CANCELLED}),
    TaskState.VERIFYING:      frozenset({TaskState.COMPLETED, TaskState.AWAITING_RETRY,
                                         TaskState.AWAITING_HUMAN, TaskState.FAILED}),
    TaskState.AWAITING_HUMAN: frozenset({TaskState.COMPLETED, TaskState.AWAITING_RETRY,
                                         TaskState.FAILED}),
    TaskState.AWAITING_RETRY: frozenset({TaskState.ASSIGNED, TaskState.CANCELLED}),
    TaskState.COMPLETED:      frozenset(),
    TaskState.FAILED:         frozenset(),
    TaskState.CANCELLED:      frozenset(),
    TaskState.SKIPPED:        frozenset(),
}

# ---- Board task status mapping ----

TASK_STATE_TO_BOARD_STATUS: dict[TaskState, str] = {
    TaskState.PENDING:        "inbox",
    TaskState.QUEUED:         "inbox",
    TaskState.AWAITING_RETRY: "assigned",
    TaskState.ASSIGNED:       "assigned",
    TaskState.RUNNING:        "in_progress",
    TaskState.CONTINUING:     "in_progress",
    TaskState.VERIFYING:      "review",
    TaskState.AWAITING_HUMAN: "review",
    TaskState.COMPLETED:      "done",
    TaskState.FAILED:         "done",
    TaskState.CANCELLED:      "done",
    TaskState.SKIPPED:        "done",
}
```
