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

_(US-002)_

---

## 4. Data Model: orchestration_runs

_(US-003)_

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
