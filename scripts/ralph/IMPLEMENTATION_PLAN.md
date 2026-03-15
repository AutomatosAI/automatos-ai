# PRD-82A Implementation Plan — Sequential Mission Coordinator

> **Scope**: Backend (orchestrator) | **Risk**: Medium (new subsystem, no existing code changes) | **Branch**: `ralph/82a-sequential-mission-coordinator`

## Phase 1: Schema & Models

- [x] US-001: Create orchestration enums module (`orchestrator/core/models/orchestration_enums.py`) — StateType, RunState(10), TaskState(11), EventType(39), ActorType, TaskType, TriggerRule, FailureReasonCode(8) StrEnums. Transition dicts, terminal frozensets, BOARD_STATUS_MAP. Note: Python 3.10 — used `(str, Enum)` pattern instead of `StrEnum`.
- [x] US-002: Create OrchestrationRun model (`orchestrator/core/models/orchestration.py`) — mission execution row with output_summary JSONB, token tracking, version_id optimistic locking. Uses SQLAlchemy 1.x Column style matching existing codebase patterns. CHECK constraint on state column validates against RunState enum values.
- [x] US-003: Create OrchestrationTask model (`orchestrator/core/models/orchestration.py`) — task row with 24 columns: failure_reason_code, token tracking, version_id optimistic locking. Composite index on (run_id, sequence_number). Partial index on active (non-terminal) states for coordinator tick queries. CHECK constraint validates TaskState enum values.
- [x] US-004: Create OrchestrationTaskDependency + OrchestrationEvent models (`orchestrator/core/models/orchestration.py`) — DAG edge table with unique constraint on (task_id, depends_on_task_id), trigger_rule defaulting to 'all_success'. Append-only OrchestrationEvent with NO version_id, composite index on (run_id, created_at) for timeline queries. Both follow existing Column style.
- [x] US-005: Create Alembic migration (`alembic/versions/prd82a_orchestration_tables.py`) — CREATE 4 tables (orchestration_runs, orchestration_tasks, orchestration_task_dependencies, orchestration_events), ALTER board_tasks (add orchestration_run_id, orchestration_task_id FKs), ALTER agent_reports (add orchestration_task_id FK). All columns with idempotent IF NOT EXISTS guards. Partial indexes on active states, composite indexes on run+sequence. CHECK constraints match Python enums exactly (10 RunState, 11 TaskState). Downgrade drops FKs first, then tables in reverse dependency order. Note: indexes on existing tables use IF NOT EXISTS (not CONCURRENTLY) since they run inside alembic's transaction.
- [x] US-006: Register models in `__init__.py` — import and export all 4 models (OrchestrationRun, OrchestrationTask, OrchestrationTaskDependency, OrchestrationEvent), 8 enums (StateType, RunState, TaskState, EventType, ActorType, TaskType, TriggerRule, FailureReasonCode), and 5 mapping objects (ALLOWED_TASK_TRANSITIONS, ALLOWED_RUN_TRANSITIONS, TERMINAL_RUN_STATES, TERMINAL_TASK_STATES, BOARD_STATUS_MAP). Verified no circular imports.

## Phase 2: State Machine & Board Bridge

- [x] US-007: Create state transition service (`orchestrator/services/orchestration_state.py`) — transition_task(), transition_run(), emit_event(). Dual-write pattern: state change + OrchestrationEvent in same flush. Optimistic lock: catches StaleDataError → raises ConflictError. InvalidTransitionError for disallowed transitions. Timestamp updates: started_at on RUNNING, completed_at on terminal states. Internal helpers map TaskState/RunState to EventType for event creation. No internal commit — caller manages transaction.
- [x] US-008: Create board bridge service (`orchestrator/services/orchestration_board_bridge.py`) — create_mission_board_task(), create_task_board_task(), sync_board_status(). Uses BOARD_STATUS_MAP → _ORCHESTRATION_TO_BOARD_STATUS secondary mapping (PRD terms like "backlog"/"blocked" → kanban terms like "inbox"/"review"). Also added orchestration_run_id and orchestration_task_id FK columns to BoardTask ORM model in core.py (migration already had them, ORM was missing them).
- [x] US-009: Create dependency resolver (`orchestrator/services/orchestration_deps.py`) — DependencyResolver with graphlib. validate_task_graph() checks cycles + invalid refs via TopologicalSorter, get_ready_tasks() queries pending tasks whose all upstream deps are verified, get_topological_order() returns execution order. Custom exceptions: CyclicDependencyError, InvalidDependencyError. Batch queries deps and upstream states to minimize DB round-trips.

## Phase 3: Coordinator

- [x] US-010: Create coordination package + AgentMatcher (`orchestrator/modules/coordination/agent_matcher.py`) — deterministic scoring (5 weights, threshold 0.4). Immutable MatchResult dataclass. Batch queries for tool assignments and busy agents. Skill matching hierarchy: exact name (1.0), substring (0.75), tag (0.5), none (0.0). History placeholder at 0.5 until 82B.
- [ ] US-011: Create MissionPlanner (`orchestrator/modules/coordination/planner.py`) — LLM decomposition + PlanValidator. 3 retry attempts on validation failure.
- [ ] US-012: Create MissionDispatcher (`orchestrator/modules/coordination/dispatcher.py`) — sequential dispatch, optimistic claim pattern, execute_with_prompt() integration.
- [ ] US-013: Create MissionReconciler (`orchestrator/modules/coordination/reconciler.py`) — stall detection (60s/300s), completion check, failure check.
- [ ] US-014: Create CoordinatorService (`orchestrator/services/coordinator_service.py`) — 5s tick, lifecycle methods (create/approve/reject/review/pause/resume/cancel), output_summary builder.
- [ ] US-015: Register coordinator tick on scheduler — alongside heartbeat tick, skip if no active missions.
- [ ] US-016: Add COORDINATOR context mode + MissionContextSection — 128k budget, mission state + task statuses in prompt.
- [ ] US-017: Create AgentRosterSection — render available agents with capabilities for planner.

## Phase 4: Verification

- [ ] US-018: Create DeterministicChecker (`orchestrator/modules/coordination/deterministic_checks.py`) — 8 check types, must_pass short-circuit.
- [ ] US-019: Create VerificationService (`orchestrator/modules/coordination/verification.py`) — deterministic first, then cross-model LLM judge. Pass/fail/partial verdicts.
- [ ] US-020: Wire verification into reconciler — completed→verifying→verified flow, retry with feedback, token tracking.

## Phase 5: API

- [ ] US-021: Create missions REST API router (`orchestrator/api/missions.py`) — 9 endpoints, auth, workspace isolation, Pydantic models.
- [ ] US-022: Mount missions router in main app.

## Discovered Issues

(None yet)
