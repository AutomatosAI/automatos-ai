# Loop 0: Design Outlines for PRDs 101-108

## Progress

- [x] US-001: Design PRD-101 outline (Mission Schema & Data Model)
- [ ] US-002: Design PRD-102 outline (Coordinator Architecture)
- [ ] US-003: Design PRD-103 outline (Verification & Quality)
- [ ] US-004: Design PRD-104 outline (Ephemeral Agents & Model Selection)
- [ ] US-005: Design PRD-105 outline (Budget & Governance)
- [ ] US-006: Design PRD-106 outline (Outcome Telemetry & Learning)
- [ ] US-007: Design PRD-107 outline (Context Interface Abstraction)
- [ ] US-008: Design PRD-108 outline (Memory Field Prototype)

## Discoveries

- Temporal and Dagster both use full event sourcing; we adopt the hybrid approach (CRUD + audit events) for simplicity
- Prefect's `task_inputs` JSONB pattern is ideal for our scale (3-20 tasks per mission) — avoids separate edges table
- Symphony's continuation vs retry distinction is critical for AI agents and should be adopted
- Dagster's `run_tags` table pattern keeps schema stable while enabling rich metadata
- Existing `board_tasks.planning_data` JSONB and `source_type` column were designed for this extension
- `board_tasks.parent_task_id` already supports subtask trees — missions can use this
- `task_reconciler.py` currently only handles `recipe_executions` — must extend for missions

## Cross-PRD Dependencies Found

- PRD-102 (Coordinator) will need `COORDINATOR` context mode in ContextService — note for PRD-107
- PRD-103 (Verification) needs `success_criteria` JSONB schema defined in PRD-101's mission_tasks
- PRD-104 (Ephemeral Agents) needs `contractor_config` JSONB schema defined in PRD-101's mission_tasks
- PRD-105 (Budget) needs `budget_config`/`budget_spent` JSONB schemas from PRD-101's mission_runs
- PRD-106 (Telemetry) will query mission_events + cost/token fields — schema must support efficient aggregation
- Existing `llm_usage` table tracks per-call costs — mission tasks should reference or aggregate these
