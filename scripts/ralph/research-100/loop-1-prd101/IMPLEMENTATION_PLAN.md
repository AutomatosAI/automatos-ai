# Loop 1: PRD-101 — Mission Schema & Data Model

## Progress

- [x] US-001: Research DAG execution patterns
- [x] US-002: Research state machine patterns for task lifecycle
- [x] US-003: Design orchestration_runs table
- [x] US-004: Design orchestration_tasks table
- [x] US-005: Design orchestration_events table
- [ ] US-006: Design integration with existing tables
- [ ] US-007: Write PRD introduction and problem statement
- [ ] US-008: Write PRD conclusion, risks, and acceptance criteria

## Discoveries

- **Dual-write pattern** (Prefect) is the strongest state storage approach — denormalized current state for fast queries + append-only event log for audit trail
- **Continuation vs retry** (Symphony) is a critical distinction for agent-based systems — must be modeled in the schema
- **DB-authoritative scheduling** (Airflow) is the safest pattern — coordinator re-derives ready tasks from DB, no in-memory state to lose
- **Infrastructure failure vs quality failure** needs different handling — Prefect's CRASHED vs FAILED distinction maps to our agent timeout vs bad output
- All 5 systems avoid storing large task outputs in the task table — separate output storage referenced by ID or path

- **Two-level state model** (StateType + StateName) inspired by Prefect allows adding display states without touching orchestration logic
- **Continuation vs retry** adopted from Symphony — critical for multi-turn agent work (continuation = 1s delay, same attempt; retry = exponential backoff, attempt incremented)
- **Optimistic locking** via SQLAlchemy `version_id_col` is sufficient for our concurrency needs; SELECT FOR UPDATE only for claim-style dequeuing
- **Board task mapping** via `source_type='orchestration'` field already exists — no new board_tasks columns needed for basic integration
- **Stall detection** extends existing `task_reconciler.py` pattern — DB-authoritative, crash-safe, runs on APScheduler tick
- **No state machine library needed** — ~100 lines of Python (enums + transition dict + transition function) covers all requirements

- **Join table wins over array column** — PostgreSQL docs explicitly warn arrays for relational edges are "a sign of database misdesign." FK enforcement, B-tree indexes, and edge metadata make the join table strictly better at our scale.
- **4 trigger rules from Airflow's 13** — `all_success`, `all_done`, `none_failed`, `always` cover all agent orchestration patterns. Others (`one_success`, `all_failed`) are for complex ETL branching we don't need.
- **`graphlib.TopologicalSorter`** (stdlib 3.9+) is purpose-built for our use case: cycle detection at planning time, incremental "task completed → get ready tasks" at runtime, zero dependencies.
- **Temporal stores NO dependency data** — dependencies are implicit in code order, reconstructed via deterministic replay. Explicitly the wrong model for queryable task graphs.
- **Prefect's `task_inputs`** is lineage metadata only — dependency resolution happens in-process via `future.result()` blocking. The JSON column is never queried for scheduling.
- **`source_type='orchestration'`** does not yet exist in BoardTask — safe to add. Existing values: `user`, `recipe`.
- **`parent_task_id`** on BoardTask is defined but unused — could be used for mission task hierarchy in future but not needed now.

## Cross-PRD Dependencies Found

- **PRD-106 (Telemetry):** The orchestration_events table IS the raw data for outcome analysis. Event schema design must anticipate telemetry queries.
- **PRD-103 (Verification):** Verification score storage on tasks needs a float column + verifier agent reference. Quality failure vs infra failure distinction affects verification design.
- **PRD-104 (Ephemeral Agents):** Contractor agent lifecycle needs a reference in orchestration_tasks — agent_type enum (roster/contractor) and potentially a contractor config JSONB.
- **PRD-102 (Coordinator):** Symphony's continuation vs retry pattern should influence coordinator's retry logic design.
