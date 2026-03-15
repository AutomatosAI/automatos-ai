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
