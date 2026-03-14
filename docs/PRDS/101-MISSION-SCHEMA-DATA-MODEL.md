# PRD-101 — Mission Schema & Data Model

**Version:** 0.1 (Draft — sections added incrementally)
**Type:** Research + Design
**Status:** In Progress
**Priority:** P0
**Dependencies:** PRD-100 (Research Master)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-14

---

## 1. Problem Statement

_(US-007 — written after all research sections are complete)_

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

_(US-004)_

---

## 6. Event Log & Audit Trail

_(US-005)_

---

## 7. Integration with Existing Schema

_(US-006)_

---

## 8. Open Questions

_(US-007)_

---

## 9. Risk Register

_(US-008)_

---

## 10. Implementation Acceptance Criteria

_(US-008)_

---

## 11. Dependencies & Sequencing

_(US-008)_

---

## 12. Appendix: Full SQL DDL

_(US-008)_

---

## 13. Appendix: SQLAlchemy Models

_(US-008)_
