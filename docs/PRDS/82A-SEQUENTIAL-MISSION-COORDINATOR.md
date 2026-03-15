# PRD-82A — Sequential Mission Coordinator

**Version:** 1.1
**Type:** Implementation
**Status:** Draft
**Priority:** P0
**Research Base:** PRDs 101 (Schema), 102 (Coordinator), 103 (Verification)
**Author:** Gerard Kavanagh + Claude
**Reviewer:** GPT 5.4 (CTO review, 2026-03-15)
**Date:** 2026-03-15

---

## 1. Goal

Ship the minimum viable mission: user says "do this project" → coordinator decomposes into tasks → assigns roster agents → executes sequentially → verifies each output → human reviews. One task at a time. No contractors, no budget gates, no parallel dispatch.

## 2. What Ships

| Component | Source PRD | Description |
|-----------|-----------|-------------|
| 4 DB tables + 2 table alterations | 101 | `orchestration_runs`, `orchestration_tasks`, `orchestration_task_dependencies`, `orchestration_events` + FK on `board_tasks` and `agent_reports`. `orchestration_archive` deferred to 82B (no archival needed until volume warrants it). |
| SQLAlchemy models + enums | 101 | Full ORM layer with state machine, optimistic locking, transition validation |
| State machine service | 101 | `transition_task()`, `transition_run()`, `emit_event()` — dual-write pattern |
| Board task bridge | 101 | Parent mission task + child per-task cards on kanban |
| Dependency resolver | 101 | DAG validation + topological ordering via `graphlib` |
| CoordinatorService | 102 | Stateless 5s tick: dispatch phase + reconcile phase |
| MissionPlanner | 102 | LLM decomposition + structural validation (templates deferred to 82B) |
| MissionDispatcher | 102 | Sequential dispatch via `execute_with_prompt()` (max 1 concurrent) |
| MissionReconciler | 102 | Stall detection, completion check, failure handling |
| AgentMatcher | 102 | Deterministic scoring — roster agents only |
| VerificationService | 103 | Deterministic checks + single cross-model LLM judge |
| `ContextMode.COORDINATOR` | 102 | New context mode + `MissionContextSection` + `AgentRosterSection` |
| REST API | 102 | `POST/GET /api/missions`, lifecycle endpoints |

## 3. What Does NOT Ship (Deferred)

| Deferred | Target PRD | Why |
|----------|-----------|-----|
| Parallel dispatch (`max_concurrent > 1`) | 82C | Get lifecycle right first |
| Contractor/ephemeral agents | 82C | Needs AgentFactory changes |
| Budget enforcement gates | 82C | Sequential missions are cheap; budget matters at scale |
| Telemetry pipeline wiring | 82B | Schema captures events; querying them is 82B |
| Template matching for decomposition | 82B | LLM-only decomposition works for v1 |
| Replanning on failure | 82B | Sequential retry is sufficient for v1 |
| "Save as routine" conversion | 82B | Needs UI work |
| Complexity detection ("this should be a mission") | 82D | Requires telemetry data first |
| `orchestration_archive` table | 82B | No archival needed until volume warrants it |
| FutureAGI `/verify-task` worker endpoint | 82C | Verification runs in-process for v1 |
| Cross-task consistency verification | 82B | Only matters with parallel execution |
| Model-per-role selection | 82C | Roster agents use their configured model |
| Trigger rules beyond `all_success` | 82C | Sequential chains only need `all_success` |

---

## 4. State Machine

### 4.1 Canonical State Enums

**RunState** (top-level mission states):

| State | StateType | Description |
|-------|-----------|-------------|
| `pending` | `INITIAL` | Created, not yet planned |
| `planning` | `ACTIVE` | LLM decomposing goal into tasks |
| `awaiting_approval` | `BLOCKED` | Plan ready, waiting human approval |
| `running` | `ACTIVE` | Tasks dispatching and executing |
| `paused` | `BLOCKED` | User paused — no new dispatches |
| `verifying` | `ACTIVE` | All tasks done, final verification pass |
| `awaiting_human` | `BLOCKED` | Verified, waiting human review |
| `completed` | `TERMINAL` | Human accepted |
| `failed` | `TERMINAL` | Unrecoverable failure |
| `cancelled` | `TERMINAL` | User cancelled |

**TaskState** (per-task states):

| State | StateType | Description |
|-------|-----------|-------------|
| `pending` | `INITIAL` | Not yet ready (deps unmet) |
| `queued` | `ACTIVE` | Dependencies met, awaiting dispatch |
| `assigned` | `ACTIVE` | Agent selected, dispatching |
| `running` | `ACTIVE` | Agent executing |
| `completed` | `ACTIVE` | Agent finished — **NOT terminal, awaiting verification** |
| `verifying` | `ACTIVE` | Verification in progress |
| `verified` | `TERMINAL` | Passed verification |
| `failed` | `TERMINAL` | Failed after max retries |
| `skipped` | `TERMINAL` | Skipped (dependency failed, mission cancelled) |
| `stalled` | `ACTIVE` | Stall detected, pending re-dispatch |
| `retrying` | `ACTIVE` | Failed verification, retrying with feedback |

### 4.2 Task State Transitions

```
pending → queued          # dependencies met
queued → assigned         # agent selected
assigned → running        # execute_with_prompt() called
assigned → stalled        # no response within 60s
running → completed       # agent returns result
running → stalled         # no progress within 300s
running → failed          # agent error / exception
completed → verifying     # verification started
verifying → verified      # verification PASS
verifying → retrying      # verification FAIL (retries remaining)
verifying → failed        # verification FAIL (no retries left)
retrying → assigned       # re-dispatched with feedback
stalled → assigned        # re-dispatched
* → skipped              # mission cancelled or dependency failed
```

### 4.3 Board Task Status Mapping

Board tasks are for kanban visibility. **`completed` ≠ `done`** — a task is only `done` on the board after verification passes.

| TaskState | Board Status | Rationale |
|-----------|-------------|-----------|
| `pending` | `backlog` | Not ready yet |
| `queued` | `todo` | Ready to dispatch |
| `assigned` | `in_progress` | Agent claimed |
| `running` | `in_progress` | Agent working |
| `completed` | `in_review` | Output received, verification pending |
| `verifying` | `in_review` | Verification running |
| `verified` | `done` | Only VERIFIED maps to done |
| `failed` | `blocked` | Failed |
| `stalled` | `blocked` | Stalled |
| `retrying` | `in_progress` | Retrying with feedback |
| `skipped` | `cancelled` | Skipped |

### 4.4 Dispatch Claim Pattern

To prevent double-dispatch under concurrent ticks or multi-instance deployment:

```sql
-- Claim a queued task atomically
UPDATE orchestration_tasks
SET state = 'assigned',
    assigned_agent_id = :agent_id,
    version_id = version_id + 1,
    updated_at = NOW()
WHERE id = :task_id
  AND state = 'queued'
  AND version_id = :expected_version
RETURNING id;
-- Returns 0 rows if already claimed → skip, no error
```

This is optimistic locking (not `SELECT FOR UPDATE SKIP LOCKED`) because:
- Sequential dispatch means at most 1 task claimed per tick
- `version_id` check is sufficient — if another instance claimed it, the UPDATE returns 0 rows
- No row-level locks held, no deadlock risk

---

## 5. Design Principles (from PRD-102)

1. **DB-authoritative, stateless coordinator.** No in-memory state. Every tick reads from DB, writes back. Any instance can take over after a crash.
2. **Dual-write pattern.** State change on row + append-only event in same transaction. Never one without the other.
3. **Direct dispatch, not board-task pickup.** Coordinator calls `execute_with_prompt()` directly. Board tasks exist for kanban visibility only — the heartbeat tick does NOT pick up mission tasks.
4. **Cross-model verification.** Verifier MUST use a different model family than executor. Self-preference bias is empirically demonstrated.
5. **Deterministic-first verification.** Check format/length/sections before burning an LLM call.
6. **Continuation vs retry.** Clean agent exit → continuation (1s, same attempt). Failure → retry (exponential backoff, attempt incremented).
7. **Optimistic locking.** `version_id` column on runs and tasks. `StaleDataError` → conflict response, not crash.

---

## 6. Mission Output Summary

When all tasks reach `verified`, the coordinator generates a **mission output summary** before entering `awaiting_human`. This is what the human reviews — not raw task outputs.

**Structure** (stored in `orchestration_runs.output_summary` JSONB column):

```json
{
  "goal": "Research EU AI Act compliance for our product",
  "tasks_completed": 4,
  "tasks_failed": 0,
  "total_duration_seconds": 342,
  "task_summaries": [
    {
      "sequence": 1,
      "title": "Research EU AI Act requirements",
      "agent": "Researcher",
      "verdict": "pass",
      "output_excerpt": "First 500 chars of verified output...",
      "output_location": "/reports/researcher/2026-03-15_eu-ai-act.md"
    }
  ],
  "generated_at": "2026-03-15T14:30:00Z"
}
```

The summary is **generated by the coordinator** (not an LLM call) — it's a structured aggregation of task results. The human review endpoint shows this summary and allows accept/reject per task.

---

## 7. Planner vs Validator Separation

The **MissionPlanner** and **VerificationService** are separate concerns with different responsibilities:

| Aspect | MissionPlanner | VerificationService |
|--------|---------------|-------------------|
| **When** | Before execution (plan phase) | After execution (verify phase) |
| **Input** | Goal string + agent roster | Task output + task spec |
| **Output** | `DecompositionResult` (tasks + deps) | `VerificationResult` (verdict + scores) |
| **LLM role** | Decomposition (structured output) | Judgment (pass/fail/partial) |
| **Model** | Coordinator's configured model | DIFFERENT model family than executor |
| **Deterministic path** | DAG validation, task count limits | Format checks, length, required sections |
| **File** | `coordination/planner.py` | `coordination/verification.py` |

**PlanValidator** is a sub-component of `MissionPlanner` (not a separate service). It validates the plan structure:
- DAG is acyclic (via `graphlib`)
- All referenced agents exist in roster
- Task count within bounds (3-20 for v1)
- No orphan tasks (all tasks reachable from root)
- Dependencies reference valid task IDs

Plan validation is **deterministic** — no LLM needed. If structural validation fails, the plan is rejected immediately and the planner retries (up to 3 attempts) before failing the mission.

---

## 8. Failure Classification

Tasks fail for different reasons. A structured `failure_reason_code` on `orchestration_tasks` enables targeted retry policies and debugging.

**FailureReasonCode** enum:

| Code | Description | Retry? |
|------|-------------|--------|
| `agent_error` | Agent raised an exception | Yes (up to max_retries) |
| `agent_timeout` | Agent exceeded stall threshold | Yes (re-dispatch) |
| `verification_fail` | Output failed verification | Yes (with feedback) |
| `verification_reject` | Human rejected during review | Yes (with feedback) |
| `no_agent_available` | No roster agent matched | No — surface to user |
| `dependency_failed` | Upstream task failed | No — skip |
| `cancelled` | Mission cancelled | No |
| `max_retries_exhausted` | All retries consumed | No — fail mission |

Stored as: `orchestration_tasks.failure_reason_code VARCHAR(50) NULL` — only populated when task enters `failed` or `skipped` state.

---

## 9. Budget Tracking (Soft)

No hard budget enforcement in 82A — sequential missions are cheap. But we **track** costs for visibility and to inform 82C's hard gates.

**Fields on `orchestration_runs`:**
- `token_budget_estimate INTEGER NULL` — planner's pre-execution estimate (total input + output tokens)
- `tokens_used INTEGER DEFAULT 0` — actual tokens consumed (updated per task completion)

**Fields on `orchestration_tasks`:**
- `tokens_used INTEGER DEFAULT 0` — tokens consumed by this task's execution + verification

**Warning behavior:** When `tokens_used > token_budget_estimate * 1.5`, emit an `EventType.BUDGET_WARNING` event. No enforcement — just telemetry for 82B dashboards.

---

## 10. Naming Convention

| Layer | Term | Example |
|-------|------|---------|
| Database | `orchestration_*` | `orchestration_runs`, `orchestration_tasks` |
| Python models | `Orchestration*` | `OrchestrationRun`, `OrchestrationTask` |
| API endpoints | `/api/missions` | `POST /api/missions`, `GET /api/missions/{id}` |
| UX/Frontend | "Mission" | "Create Mission", "Mission Status" |
| Events | `orchestration.*` | `orchestration.task.completed` |

**Rule:** Database and backend code use `orchestration`. API routes and UI use `mission`. The mapping happens at the API layer — endpoint handlers translate between the two. No aliasing, no dual names in the same layer.

---

## 11. Verification Retry Guardrails

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_retries` (per task) | 3 | Enough to incorporate feedback, not enough to burn budget |
| `max_verification_retries` | 2 | Verification itself can fail (LLM error) — 2 retries max |
| `retry_backoff` | Exponential: 5s, 15s, 45s | Give agent time, don't flood |

**Failure class policies:**

| Failure Class | Policy |
|--------------|--------|
| `agent_error` | Retry immediately (likely transient) |
| `agent_timeout` | Re-dispatch to SAME agent (stall recovery) |
| `verification_fail` | Retry with verifier reasoning injected into agent prompt |
| `verification_reject` (human) | Retry with human feedback injected |
| `no_agent_available` | Fail task immediately — no retry (roster problem, not execution) |
| `dependency_failed` | Skip task — no retry (upstream must be fixed first) |

**Escalation:** After `max_retries` exhausted → task enters `failed` → mission enters `failed` → emit `orchestration.mission.failed` event. No automatic replanning in 82A (deferred to 82B).

---

## 12. Implementation Phases

### Phase 1: Schema & Models (~Ralph)

**Goal:** Tables exist, models compile, migration runs clean.

#### Files to CREATE

| # | File | What |
|---|------|------|
| 1 | `orchestrator/core/models/orchestration_enums.py` | `StateType`, `RunState` (10), `TaskState` (11), `EventType` (30+, including `BUDGET_WARNING`), `ActorType`, `TaskType`, `TriggerRule`, `FailureReasonCode` (8) StrEnums. `RUN_STATE_TYPE`/`TASK_STATE_TYPE` mappings. `TERMINAL_*_STATES` frozensets. `ALLOWED_TASK_TRANSITIONS`/`ALLOWED_RUN_TRANSITIONS` dicts. `BOARD_STATUS_MAP` dict. Source: PRD-101 Section 3.2 + 3.10, PRD-82A Sections 4 + 8 |
| 2 | `orchestrator/core/models/orchestration.py` | SQLAlchemy models: `OrchestrationRun` (+ `output_summary` JSONB, `token_budget_estimate` INT, `tokens_used` INT), `OrchestrationTask` (+ `failure_reason_code` VARCHAR(50), `tokens_used` INT), `OrchestrationTaskDependency`, `OrchestrationEvent`. All with `version_id_col` for optimistic locking. `OrchestrationArchive` deferred to 82B. Source: PRD-101 Section 13 |
| 3 | `alembic/versions/xxxx_prd101_orchestration_tables.py` | Single migration: CREATE 4 tables, ALTER `board_tasks` (add `orchestration_run_id`, `orchestration_task_id` FKs), ALTER `agent_reports` (add `orchestration_task_id` FK). All `CREATE INDEX CONCURRENTLY` for existing tables. Source: PRD-101 Section 12 |

#### Files to MODIFY

| # | File | Change |
|---|------|--------|
| 4 | `orchestrator/core/models/__init__.py` | Import and export new models |

#### Acceptance Criteria — Phase 1

- [ ] `alembic upgrade head` succeeds on clean DB
- [ ] `alembic downgrade -1` succeeds (drops tables, removes FKs)
- [ ] All 4 tables created with correct columns, types, constraints
- [ ] `board_tasks.orchestration_run_id` and `orchestration_task_id` FKs exist (nullable)
- [ ] `agent_reports.orchestration_task_id` FK exists (nullable)
- [ ] All indexes created (including partial indexes on active states)
- [ ] Python enums match DB CHECK constraints
- [ ] `ALLOWED_TASK_TRANSITIONS` and `ALLOWED_RUN_TRANSITIONS` are complete
- [ ] Models can be imported without circular dependencies

### Phase 2: State Machine & Board Bridge (~Ralph)

**Goal:** State transitions work, events are emitted, board tasks sync.

#### Files to CREATE

| # | File | What |
|---|------|------|
| 5 | `orchestrator/services/orchestration_state.py` | `transition_task(task, new_state, actor, reason)` — validates transition, updates `state`/`state_type`/timestamps, emits event, syncs board task, all in one transaction. `transition_run()` same pattern. `emit_event()` for non-transition events. Handles `StaleDataError` from optimistic locking. Source: PRD-101 Section 3.9 |
| 6 | `orchestrator/services/orchestration_board_bridge.py` | `create_mission_board_task(run)` — creates parent board task with `source_type='orchestration'`. `create_task_board_task(task)` — creates child with `source_type='orchestration_task'`. `sync_board_status(task)` — maps orchestration state → board status. Source: PRD-101 Section 7.2 |
| 7 | `orchestrator/services/orchestration_deps.py` | `DependencyResolver` — wraps `graphlib.TopologicalSorter`. `validate_task_graph(tasks, deps)` — checks DAG (no cycles), all refs valid. `get_ready_tasks(run_id)` — finds tasks where all parents are terminal success. Source: PRD-101 Section 5.5 |

#### Acceptance Criteria — Phase 2

- [ ] `transition_task(task, RUNNING)` updates state, state_type, started_at, emits event
- [ ] Invalid transitions raise `InvalidTransitionError` (e.g., PENDING → COMPLETED)
- [ ] Optimistic locking: concurrent transitions on same task → one succeeds, one raises `StaleDataError`
- [ ] Board task created for mission with `source_type='orchestration'`
- [ ] Board task created per task with `source_type='orchestration_task'`
- [ ] Board status syncs when task state changes (RUNNING → `in_progress`, COMPLETED → `in_review`, VERIFIED → `done`)
- [ ] DAG validation catches cycles
- [ ] `get_ready_tasks()` returns only tasks whose ALL parents are terminal success

### Phase 3: Coordinator Service (~Interactive)

**Goal:** Coordinator creates missions, decomposes goals, dispatches sequentially, reconciles.

#### Files to CREATE

| # | File | What |
|---|------|------|
| 8 | `orchestrator/modules/coordination/__init__.py` | Package |
| 9 | `orchestrator/modules/coordination/planner.py` | `MissionPlanner.decompose(goal, workspace_id, agents, config)` → `DecompositionResult`. LLM call with structured output → validate DAG → estimate costs. Template matching deferred. Source: PRD-102 Section 5 |
| 10 | `orchestrator/modules/coordination/dispatcher.py` | `MissionDispatcher.dispatch_task(run, task)` → select agent via AgentMatcher → create board task → transition to ASSIGNED → `execute_with_prompt()` → on completion, transition to VERIFYING. Sequential: only dispatches if no other task is RUNNING/ASSIGNED. Source: PRD-102 Section 6 |
| 11 | `orchestrator/modules/coordination/reconciler.py` | `MissionReconciler.reconcile(run)` → check running tasks for stalls → check if all tasks terminal → advance mission state. Stall thresholds: ASSIGNED 60s, RUNNING 300s (configurable). Source: PRD-102 Section 4.3-4.5 |
| 12 | `orchestrator/modules/coordination/agent_matcher.py` | `AgentMatcher.match(task_spec, agents)` → deterministic scoring (tool_coverage 0.35, skill_match 0.25, model_fit 0.15, availability 0.10, history 0.15). Threshold 0.4. Returns best match or None. Source: PRD-102 Section 6.2 |
| 13 | `orchestrator/services/coordinator_service.py` | Main service. `register_tick()` on shared scheduler (5s). `tick()` = dispatch phase + reconcile phase for all active missions. Lifecycle: `create_mission`, `approve_plan`, `reject_plan`, `review_mission`, `pause_mission`, `resume_mission`, `cancel_mission`. Source: PRD-102 Section 3.3 |

#### Files to MODIFY

| # | File | Change |
|---|------|--------|
| 14 | `orchestrator/modules/context/modes.py` | Add `COORDINATOR` to `ContextMode` enum and `MODE_CONFIGS`. Sections: identity, mission_context, agent_roster, platform_actions, task_context, datetime_context. `token_budget=131072` (128k — coordinator needs full mission context + agent roster + task history), `tool_loading=FULL`. Source: PRD-102 Section 7.1 |
| 15 | `orchestrator/modules/context/service.py` | Register `MissionContextSection` and `AgentRosterSection` renderers |
| 16 | Startup/scheduler registration | Register coordinator tick alongside heartbeat tick |

#### Files to CREATE (context sections)

| # | File | What |
|---|------|------|
| 17 | `orchestrator/modules/context/sections/mission_context.py` | Renders current mission state: goal, plan summary, task statuses, agent assignments, budget status. Source: PRD-102 Section 7.2 |
| 18 | `orchestrator/modules/context/sections/agent_roster.py` | Renders available agents: id, name, skills, tools, model, recent success rate. Source: PRD-102 Section 7.2 |

#### Acceptance Criteria — Phase 3

- [ ] `create_mission("Research EU AI Act compliance", config)` → creates `orchestration_run` in `pending` state
- [ ] Planner decomposes goal into 3-20 tasks with dependency edges
- [ ] Plan validation rejects cycles, invalid agent refs, empty plans
- [ ] In approve mode: mission waits in `awaiting_approval` until `approve_plan()` called
- [ ] In autonomy mode: mission auto-advances to `running`
- [ ] Dispatcher assigns task to highest-scoring roster agent
- [ ] Dispatcher calls `AgentFactory.execute_with_prompt()` directly (not board task pickup)
- [ ] Only 1 task runs at a time (sequential enforcement)
- [ ] On task completion: dependency resolver queues next ready task
- [ ] Stall detection: ASSIGNED task with no progress after 60s → re-dispatch
- [ ] Stall detection: RUNNING task exceeding 300s → mark stalled
- [ ] When all tasks terminal success → mission transitions to `verifying`
- [ ] When any task fails after max retries → mission transitions to `failed`
- [ ] `pause_mission()` stops dispatching new tasks (running task continues)
- [ ] `cancel_mission()` transitions to `cancelled`, stops all pending tasks
- [ ] Coordinator context mode builds system prompt with mission state + agent roster
- [ ] Board task visibility: parent mission card + child task cards created and synced
- [ ] Dispatch uses optimistic claim pattern (Section 4.4) — concurrent claims don't double-dispatch
- [ ] Mission output summary generated when all tasks verified (Section 6)
- [ ] `failure_reason_code` populated on task failure (Section 8)
- [ ] Token tracking: `tokens_used` updated per task completion (Section 9)

### Phase 4: Verification (~Ralph)

**Goal:** Task outputs verified before mission advances.

#### Files to CREATE

| # | File | What |
|---|------|------|
| 19 | `orchestrator/modules/coordination/verification.py` | `VerificationService.verify_task(task)` → deterministic checks → LLM judge → `VerificationResult(verdict, scores, reasoning)`. Cross-model: if task used Claude, verifier uses GPT-4o (or vice versa). Source: PRD-103 Sections 3-5 |
| 20 | `orchestrator/modules/coordination/deterministic_checks.py` | `DeterministicChecker` with 8 check types: format_regex, min_length, max_length, required_sections, json_schema, url_valid, contains_keywords, word_count_range. Short-circuits: if must_pass deterministic check fails → FAIL immediately. Source: PRD-103 Section 4 |

#### Files to MODIFY

| # | File | Change |
|---|------|--------|
| 21 | `orchestrator/modules/coordination/reconciler.py` | After task COMPLETED → call `VerificationService.verify_task()` → if PASS: transition to VERIFIED → if FAIL: retry with verifier feedback (up to max_retries) → if PARTIAL: escalate to human |

#### Acceptance Criteria — Phase 4

- [ ] Deterministic check failure → immediate FAIL verdict (no LLM cost)
- [ ] LLM verifier uses different model family than executor
- [ ] Verification result stored: `mission_events` with type `verification_completed`
- [ ] PASS verdict → task transitions to `verified`
- [ ] FAIL verdict → retry with verifier reasoning injected into agent prompt
- [ ] PARTIAL verdict (low confidence) → escalate to human review
- [ ] After all tasks verified → mission transitions to `awaiting_human`
- [ ] Human review: accept → `completed`, reject with feedback → specific tasks re-queued

### Phase 5: API Endpoints (~Ralph)

**Goal:** Frontend can create and manage missions.

#### Files to CREATE

| # | File | What |
|---|------|------|
| 22 | `orchestrator/api/missions.py` | REST router. Source: PRD-102 Section 10 |

#### Endpoints

| Method | Path | Action |
|--------|------|--------|
| `POST` | `/api/missions` | Create mission (goal, config) |
| `GET` | `/api/missions` | List missions for workspace (paginated, filterable by state) |
| `GET` | `/api/missions/{id}` | Get mission detail (run + tasks + events) |
| `POST` | `/api/missions/{id}/approve` | Approve plan (optional modifications) |
| `POST` | `/api/missions/{id}/reject` | Reject plan (with reason) |
| `POST` | `/api/missions/{id}/review` | Submit human review (accept/reject per task) |
| `POST` | `/api/missions/{id}/pause` | Pause mission |
| `POST` | `/api/missions/{id}/resume` | Resume mission |
| `POST` | `/api/missions/{id}/cancel` | Cancel mission |

#### Files to MODIFY

| # | File | Change |
|---|------|--------|
| 23 | Router registration (main app) | Mount `/api/missions` router |

#### Acceptance Criteria — Phase 5

- [ ] All 9 endpoints return correct responses
- [ ] Workspace isolation: missions only visible to owning workspace
- [ ] Auth: all endpoints require valid Clerk token
- [ ] `GET /api/missions/{id}` returns full task graph with statuses
- [ ] `POST /api/missions/{id}/approve` with modifications updates plan
- [ ] `POST /api/missions/{id}/review` with per-task feedback re-queues specific tasks

---

## 13. Build Plan

| Phase | Builder | Estimated Files | Dependencies |
|-------|---------|----------------|--------------|
| **Phase 1: Schema** | Ralph | 4 | None |
| **Phase 2: State Machine** | Ralph | 3 | Phase 1 |
| **Phase 3: Coordinator** | Interactive (Claude + Gerard) | 11 | Phase 2 |
| **Phase 4: Verification** | Ralph | 2 + 1 modify | Phase 3 |
| **Phase 5: API** | Ralph | 1 + 1 modify | Phase 3 |

**Total: 22 new files, 5 modified files**

Phases 4 and 5 can run in parallel after Phase 3.

---

## 14. Key Integration Points

### How the Coordinator Uses Existing Systems

| System | How It's Used | File |
|--------|--------------|------|
| `AgentFactory.execute_with_prompt()` | Dispatches each task to assigned agent | `agent_factory.py` |
| `ContextService.build_context()` | Builds coordinator prompt with `ContextMode.COORDINATOR` | `context/service.py` |
| `get_tools_for_agent()` | Resolves tools for task agents (unchanged) | `tool_router.py:140` |
| `UnifiedToolExecutor.execute_tool()` | Coordinator's own tool loop for platform actions | `unified_executor.py` |
| `BoardTask` model | Creates kanban cards for mission + tasks | `core/models/board.py` |
| `UnifiedScheduler` | Registers 5s coordinator tick alongside heartbeat | `heartbeat_service.py` pattern |

### What Does NOT Change

- `AgentFactory` — no modifications needed (already accepts `AgentRuntime`)
- `execute_with_prompt()` tool loop — same 10-iteration loop
- `HeartbeatService` — continues unchanged for routine/recipe work
- `TaskReconciler` — continues for recipe executions only
- All existing API endpoints — no breaking changes

---

## 15. Sequence Diagram: Happy Path

```
User                 API              CoordinatorService        MissionPlanner         AgentFactory
  │                   │                      │                       │                     │
  ├── POST /missions ─►                      │                       │                     │
  │                   ├── create_mission() ──►│                       │                     │
  │                   │                      ├── decompose() ────────►│                     │
  │                   │                      │                       ├── LLM call           │
  │                   │                      │                       ├── validate DAG       │
  │                   │                      │◄── DecompositionResult─┤                     │
  │                   │                      ├── create tasks in DB   │                     │
  │                   │                      ├── create board tasks   │                     │
  │                   │                      ├── → awaiting_approval  │                     │
  │◄── 201 {run_id} ─┤                      │                       │                     │
  │                   │                      │                       │                     │
  ├── POST approve ──►                      │                       │                     │
  │                   ├── approve_plan() ───►│                       │                     │
  │                   │                      ├── → running            │                     │
  │                   │                      │                       │                     │
  │                   │          [tick @ 5s]  │                       │                     │
  │                   │                      ├── dispatch phase       │                     │
  │                   │                      │   find queued task #1  │                     │
  │                   │                      │   AgentMatcher.match() │                     │
  │                   │                      │   → ASSIGNED           │                     │
  │                   │                      ├── execute_with_prompt() ──────────────────────►
  │                   │                      │                       │                     ├─► LLM
  │                   │                      │                       │                     │◄─ result
  │                   │                      │                       │                     │
  │                   │          [tick @ 5s]  │                       │                     │
  │                   │                      ├── reconcile phase      │                     │
  │                   │                      │   task #1 COMPLETED    │                     │
  │                   │                      │   verify_task()        │                     │
  │                   │                      │   → VERIFIED           │                     │
  │                   │                      │   _on_task_completed() │                     │
  │                   │                      │   queue task #2        │                     │
  │                   │                      │                       │                     │
  │                   │          [... repeat for tasks 2-N ...]       │                     │
  │                   │                      │                       │                     │
  │                   │                      ├── all tasks verified   │                     │
  │                   │                      ├── build output_summary │                     │
  │                   │                      ├── → awaiting_human     │                     │
  │                   │                      │                       │                     │
  ├── POST review ───►                      │                       │                     │
  │                   ├── review_mission() ─►│                       │                     │
  │                   │                      ├── → completed          │                     │
  │◄── 200 {result} ─┤                      │                       │                     │
```

---

## 16. Testing Strategy

### Unit Tests (per Phase)

| Phase | Test Focus | Key Scenarios |
|-------|-----------|---------------|
| 1 | Models + enums | State mappings correct, transition dicts complete, model instantiation |
| 2 | State machine | Valid transitions succeed, invalid transitions raise, optimistic lock conflict, board sync |
| 3 | Coordinator | Decomposition → valid DAG, sequential dispatch (only 1 running), stall detection triggers, dependency resolution queues next |
| 4 | Verification | Deterministic checks short-circuit, cross-model selection, pass/fail/partial verdicts |
| 5 | API | CRUD, lifecycle state transitions via HTTP, auth, workspace isolation |

### Integration Tests

| Scenario | What It Tests |
|----------|--------------|
| **3-task sequential mission (happy path)** | Full lifecycle: create → decompose → approve → execute 3 tasks sequentially → verify each → human accept |
| **Task failure with retry** | Task fails → retry with verifier feedback → succeeds on retry 2 → mission completes |
| **Task failure after max retries** | Task fails 3x → mission fails → correct events emitted |
| **Human rejects specific task** | Mission verified → human rejects task 2 → task 2 re-queued → re-executed → re-verified → human accepts |
| **Stall detection** | Task assigned but agent doesn't respond → stall detected → re-dispatched |
| **Pause/resume** | Running mission paused → current task continues → no new tasks dispatched → resume → next task dispatches |
| **Cancel** | Running mission cancelled → running task continues to completion → no new tasks → mission state = cancelled |

---

## 17. Risks

| # | Risk | Mitigation |
|---|------|------------|
| 1 | LLM decomposition quality varies by model | Structural validation catches bad plans (cycles, missing agents, too many/few tasks). Human approval gate for v1. |
| 2 | Agent matching returns no match (no roster agent has required tools) | Return clear error: "No agent available with tools [X, Y]. Create an agent with these tools or remove the requirement." |
| 3 | Coordinator tick conflicts with heartbeat tick | Separate scheduler entries. Coordinator processes missions only. Heartbeat processes routine/recipe only. No overlap. |
| 4 | `execute_with_prompt()` hangs forever | Reconciler stall detection catches it at 300s threshold. Mark stalled, re-dispatch or fail. |
| 5 | Migration on production DB with existing data | All new tables (`CREATE TABLE IF NOT EXISTS`), all FKs nullable, all indexes `CONCURRENTLY`. Zero-downtime. |
| 6 | Board task status gets out of sync | Dual-write in same transaction. If board update fails, entire transition rolls back. |

---

## 18. Success Criteria

82A is **done** when:

1. A user can create a mission from the API with a natural language goal
2. The coordinator decomposes it into 3-10 tasks with dependencies
3. The user approves the plan (or auto-approves in autonomy mode)
4. Tasks execute sequentially via roster agents
5. Each task output is verified (deterministic + LLM)
6. Failed tasks retry with verifier feedback
7. Completed mission enters human review
8. Human can accept, reject specific tasks, or reject all
9. The full lifecycle is visible on the kanban board
10. All state transitions emit events to `orchestration_events`
