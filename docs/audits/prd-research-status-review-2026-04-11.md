# PRD Research Status Review — 82* and 100-108

**Date:** 2026-04-11
**Author:** Claude (Opus 4.6) session
**Scope:** Audit what's been researched, designed, and built across the Mission Mode research track (PRDs 100-108) and its implementation children (82A/B/C). Ground truth = codebase greps, not PRD headers.
**Sibling audit:** [auto-agent-cross-tenant-flaw.md](./auto-agent-cross-tenant-flaw.md)

---

## TL;DR

Nine research PRDs (100-108) fed three implementation PRDs (82A/B/C). What's actually built on disk:

| PRD | Type | Status | Notes |
|---|---|---|---|
| **100** Research: Autonomous Operating Layer | Research / roadmap | **N/A (research)** | Master vision doc. Phase 2 = missions; Phase 3 = neural field. |
| **101** Mission Schema & Data Model | Design-for-build | **BUILT via 82A** | Tables + enums + state machine shipped. Schema matches spec. |
| **102** Coordinator Architecture | Design-for-build | **BUILT via 82A** | CoordinatorService, Planner, Dispatcher, Reconciler, AgentMatcher all exist. |
| **103** Verification & Quality | Design-for-build | **BUILT via 82A** (partial) | VerificationService + DeterministicChecker in `modules/coordination/`. No dedicated `/verify-task` worker endpoint — runs in-process. |
| **104** Ephemeral Agents & Model Selection | Design-for-build | **NOT BUILT** | No `ContractorConfig`, no `create_ephemeral_agent`, no role-based model map, no lifecycle SM. Blocks 82C+. |
| **105** Budget & Governance | Design-for-build | **PARTIAL via 82C** | `budget_config`/`budget_spent` JSONB columns on `orchestration_runs` + `_pre_dispatch_budget_check` wired in dispatcher. No `MissionBudgetManager` class, no `BudgetExceededError`, no 4-tier tool policy, no `mission_task_id` FK on `llm_usage`. |
| **106** Outcome Telemetry | Design-for-build | **NOT BUILT** | No `TelemetryService`, no `mission_events` table, no `mv_model_performance` MV, no `MISSION_TASKS_TOTAL` metric. Events are persisted via `orchestration_events` (101's append log), not 106's design. |
| **107** Context Interface Abstraction | Design-for-build | **PARTIAL** | `core/ports/context.py` exists; `modules/context/factory.py` + `SHARED_CONTEXT_BACKEND` config var exists. 10-caller migration status unverified. |
| **108** Memory Field Prototype | Design-for-build | **PARTIAL** | `VectorFieldSharedContext` adapter shipped in `modules/context/adapters/vector_field.py`. NO Qdrant service, NO `experiment_results` table, NO 30-run experiment harness, NO Phase 3 gate report. |
| **82** Orchestration Readiness | Research | **N/A** | Split doc. Produced 82A/B/C/D. |
| **82A** Sequential Mission Coordinator | Implementation | **BUILT** | All expected modules present. Verification uses in-process path (not 103's worker endpoint). |
| **82B** Mission Intelligence Layer | Implementation | **PARTIAL** | Replan support + archive migrations shipped. Templates shipped (`templates.py`, `TEMPLATE_REGISTRY`). Uncertainty on: telemetry API endpoints, history scoring actually wired (vs stub), cross-task consistency, save-as-routine. |
| **82C** Parallel Execution + Budget Decomp | Implementation | **BUILT (core path)** | `dispatch_ready`, `asyncio.gather` in `_process_run`, `parallel_group`/`complexity`/`estimated_tokens` columns, `_pre_dispatch_budget_check`, dedicated tests (`test_82c_wiring`, `test_budget_gate`, `test_coordinator_parallel`, `test_synthesis_*`). |
| **82D** (Guidance + Contractors) | — | **NOT STARTED** | Mentioned in 82's split plan; no file on disk. This is where PRD-104 would land. |

Bottom line: the **sequential + parallel coordinator pipeline works** (82A/82C core). The **governance/observability/intelligence** layers (104 contractors, 105 budget manager, 106 telemetry, 82B polish, 82D guidance) are either partial or absent. The current Mission Zero failures (`agent_error` on task dc3d11fe, no real agents) are a symptom of **missing 104 + incomplete 82B roster/matcher wiring**, not coordinator bugs.

---

## Findings matrix — what shipped, what didn't

Legend: ✅ built and grep-verified · 🟡 partial / stubbed / design differs from spec · ❌ not on disk

### Research & data model

| PRD-101 deliverable | Status | Evidence |
|---|---|---|
| `orchestration_runs` table | ✅ | `core/models/orchestration.py`, `alembic/versions/prd82a_orchestration_tables.py` |
| `orchestration_tasks` table | ✅ | same |
| `orchestration_task_dependencies` | ✅ | same |
| `orchestration_events` (append log) | ✅ | same; 18+ `emit_event()` call sites in `coordinator_service.py` |
| `RunState` / `TaskState` enums + `ALLOWED_*_TRANSITIONS` | ✅ | `core/models/orchestration_enums.py` — 10 run states, 11 task states, including `REPLANNING` |
| `FailureReasonCode` StrEnum | ✅ | same |
| `BOARD_STATUS_MAP` | ✅ | same |
| `budget_config` / `budget_spent` JSONB on runs | ✅ | `core/models/orchestration.py:115-116` |

### Coordinator (PRD-102)

| Deliverable | Status | Evidence |
|---|---|---|
| `CoordinatorService` with two-phase tick | ✅ | `services/coordinator_service.py` — 2000+ lines, `_process_run` at :524 |
| `MissionPlanner` + `replan()` | ✅ | `modules/coordination/planner.py:296`, `planner.py:305 async def replan` |
| `MissionDispatcher` | ✅ | `modules/coordination/dispatcher.py:76` |
| `MissionReconciler` | ✅ | `modules/coordination/reconciler.py:83` |
| `AgentMatcher` | ✅ | `modules/coordination/agent_matcher.py:101` |
| Template-based decomposition | ✅ | `modules/coordination/templates.py` — `TEMPLATE_REGISTRY` with at least `research_and_report` |
| `_detect_complexity` | ✅ | in planner (82C addition) |
| `ContextMode.COORDINATOR` + `MissionContextSection` | 🟡 | Not grep-verified — tasked as follow-up |
| HTN template library | 🟡 | templates.py has registry; depth of coverage unverified |

### Verification (PRD-103)

| Deliverable | Status | Evidence |
|---|---|---|
| `VerificationService` | ✅ | `modules/coordination/verification.py:302` |
| `DeterministicChecker` | ✅ | `modules/coordination/deterministic_checks.py:64` |
| `VERIFIER_MODEL_SELECTION` cross-family map | ❓ | No grep hit for the constant name |
| `POST /verify-task` worker endpoint | ❌ | No hit. Verification runs in-process inside coordinator, not as a worker route. This is a **spec drift** — acceptable but worth documenting. |
| Adaptive thresholds | ❓ | Unverified |

### Ephemeral agents (PRD-104) — THE GAP

| Deliverable | Status | Evidence |
|---|---|---|
| `ContractorConfig` dataclass | ❌ | no hit |
| `create_ephemeral_agent()` function | ❌ | no hit |
| Role-based model mapping (6 roles × defaults) | ❌ | no hit |
| `enforce_cognitive_diversity()` | ❌ | no hit |
| Lifecycle state machine (SPAWNING→…→DESTROYED) | ❌ | no hit |
| Concurrency limits (3/5/20) | ❌ | no hit |
| Cleanup triggers (TTL, GC sweep) | ❌ | no hit |
| `AgentRuntime` reference | 🟡 | `agent_factory.py:155` has a class named `AgentRuntime` — this is **pre-existing** roster runtime, NOT the PRD-104 ephemeral runtime. PRD-104 outline's claim "already accepts AgentRuntime" refers to this class; the contractor path it envisions is still unbuilt. |

**Consequence:** Mission Zero seeds placeholder roster agents because 104's contractor path doesn't exist. The `agent_error` failure on task `dc3d11fe…` happens because the matcher assigns a real-but-underspecified roster agent (no tools/persona/model for the task shape) and `execute_with_prompt` blows up at runtime.

### Budget (PRD-105)

| Deliverable | Status | Evidence |
|---|---|---|
| `MissionBudgetManager` class | ❌ | no hit |
| `BudgetExceededError` | ❌ | no hit |
| `_pre_dispatch_budget_check` | ✅ | `dispatcher.py:440`, called at `:593` |
| `BudgetStatus` enum | ✅ | `orchestration_enums.py` |
| `BudgetDecision` response | 🟡 | used at `dispatcher.py:593` — structure unverified vs spec |
| 4-tier tool policy (workspace→mission→task→agent) in `get_tools_for_agent()` | ❌ | no hit |
| `mission_task_id` FK on `llm_usage` | ❌ | no hit in `usage_tracker.py` or migrations |
| Plan-limit enforcement via `Workspace.plan_limits` | ❓ | unverified |

So 82C built the admission-gate **mechanism** (dispatcher hook + column) but skipped 105's governance **class** and its `llm_usage` integration. Cost attribution to tasks is impossible until the FK lands.

### Telemetry (PRD-106)

| Deliverable | Status | Evidence |
|---|---|---|
| `TelemetryService` / `TelemetryEvent` | ❌ | no hit |
| `mission_events` table (separate from 101's `orchestration_events`) | ❌ | only `orchestration_events` exists |
| Summary columns on runs/tasks (total_tokens, cost_usd, verifier_score) | 🟡 | Partial — some on `orchestration_tasks`; full 106 column list not verified |
| `mv_model_performance` materialized view | ❌ | no hit |
| `MISSION_TASKS_TOTAL` / `MISSION_COST_USD` Prom metrics | ❌ | no hit |
| `/api/missions/{id}/telemetry` + `/api/analytics/mission-outcomes` | ❓ | `api/missions.py` has `get_mission_events` but other telemetry endpoints unverified |
| Attribute allowlist / PII prevention | ❌ | no hit |

106 is the **biggest unshipped design doc in the stack**. The 101 event log covers coordinator events, but 106's model/cost/accuracy analytics layer is not built. No ML feedback loop possible yet.

### Context abstraction + field (PRD-107 + PRD-108)

| Deliverable | Status | Evidence |
|---|---|---|
| `core/ports/context.py` (ABCs) | ✅ | file exists |
| `modules/context/factory.py` | ✅ | exists; reads `SHARED_CONTEXT_BACKEND` (named differently from PRD — `CONTEXT_BACKEND` → `SHARED_CONTEXT_BACKEND`) |
| `SHARED_CONTEXT_BACKEND` config var | ✅ | `config.py:424`, defaults to `"vector_field"` |
| `DefaultContextProvider` / `RedisSharedContext` adapters | 🟡 | `modules/context/adapters/redis_context.py` exists; 10-caller migration unverified |
| `VectorFieldSharedContext` (108 prototype) | ✅ | `modules/context/adapters/vector_field.py:43` |
| Qdrant Railway service | ❓ | needs Railway inspection |
| `experiment_results` table | ❌ | no hit |
| 30-run controlled experiment harness | ❌ | no hit |
| Decay / reinforce / measure_stability methods | ❓ | file exists but internal API not verified |
| Phase 3 go/no-go gate report | ❌ | no hit |

The port abstraction and vector-field adapter shipped. The **scientific experiment** (the whole reason PRD-108 exists) did not — so there is no data to support promoting neural-field to the default, and no criteria to reject it either. The `SHARED_CONTEXT_BACKEND=vector_field` default is effectively **an undefended architectural choice** made without the experiment PRD-108 committed to running.

### 82A/82B/82C implementation

| Piece | Status | Evidence |
|---|---|---|
| 82A tables + enums + state machine | ✅ | prd82a_orchestration_tables.py |
| 82A CoordinatorService + 5s tick | ✅ | coordinator_service.py |
| 82A Planner/Dispatcher/Reconciler/Matcher/Verification/DeterministicChecker | ✅ | all six files in `modules/coordination/` |
| 82A 9 REST endpoints at `/api/missions` | 🟡 | `api/missions.py` exists; endpoint count unverified |
| 82A board bridge | ✅ | `services/orchestration_board_bridge.py` |
| 82A dependency resolver | ✅ | `services/orchestration_deps.py` (implied by MEMORY) |
| 82A "Phase 6 Chat Mission Mode" | ❓ | **Not in 82A PRD** (explicitly deferred). MEMORY says Ralph built it overnight. Spec drift — verify what was actually built against 82C's mission-conversation design. |
| 82B `templates.py` + `TEMPLATE_REGISTRY` | ✅ | templates.py 1357 lines, registry present |
| 82B `replan()` on planner | ✅ | `planner.py:305` |
| 82B `RunState.REPLANNING` | ✅ | `orchestration_enums.py` |
| 82B `OrchestrationArchive` | ✅ | `prd82b_us009_orchestration_archive.py` |
| 82B `_score_history` actually wired (not 0.5 stub) | ❓ | needs inspection |
| 82B cross-task consistency | ❓ | unverified |
| 82B telemetry endpoints | 🟡 | `get_mission_events` only — `/cost`, `/stats`, `/agents/{id}/mission-history` unverified |
| 82B save-as-routine | ❓ | unverified |
| 82B verification hash cache + adaptive thresholds | ❓ | unverified |
| 82C `count_active_tasks` + `dispatch_ready` | ✅ | dispatcher.py |
| 82C `_process_run` with `asyncio.gather` | ✅ | `coordinator_service.py:524`, `:581` |
| 82C `_pre_dispatch_budget_check` | ✅ | `dispatcher.py:440` |
| 82C `parallel_group` + `complexity` + `estimated_tokens` columns | ✅ | `prd82c_parallel_schema.py` |
| 82C synthesis task auto-insertion | ✅ | `test_synthesis_auto_insert.py` exists |
| 82C `TaskType.SYNTHESIS` executor branch | ✅ | `test_synthesis_executor.py` exists (presumes implementation) |
| 82C wiring tests | ✅ | `test_82c_wiring.py`, `test_dispatcher_parallel.py`, `test_coordinator_parallel.py`, `test_budget_gate.py` — 82C's mandate was followed |
| **82D Guidance/Contractors** | ❌ | no file — this is where PRD-104 contractors and 105 tool policy would land |

---

## Session findings recap (for the record)

These are the concrete items uncovered in this session beyond the PRD audit itself. They are not PRD-level but they inform the build/no-build decisions above.

### 1. Auto agent cross-tenant flaw
Full writeup: [`auto-agent-cross-tenant-flaw.md`](./auto-agent-cross-tenant-flaw.md). Summary: `get_default_agent_id` in `api/chat.py:264-322` has a terminal `return 1` fallback that silently routes any workspace without Composio-assigned agents to admin's agent #1. The Orchestrator settings form (`api/workspaces.py:272-316`) saves `workspace.settings["orchestrator"]` JSONB that **no downstream code reads** — it's an inert UI. Quick fix for demo: no code change (admin's first-branch query succeeds). Proper fix: hidden per-workspace Auto agent row (`slug=auto-{workspace_id}`, `is_system_agent=True`) — 5-step deploy order documented in the audit.

### 2. PRD-104 16-gap analysis
Organised by severity:
- **4 blockers:** `AgentRuntime` undefined in PRD (fixed — it's the 82A roster class, repurposed name); PRD-101 migrations (✅ shipped, no longer a blocker); PRD-107 `SharedContextPort` dependency (✅ shipped); PRD-105 Budget Gate dependency (🟡 partial — admission gate wired, Manager class missing).
- **8 unknowns:** role-specific prompt templates (highest value, missing); tool catalog surface at decomposition time; tool name canonicalization across 3 registries; `llm_models` table state for cost pricing; <100ms spawn target realism; in-flight contractors on coordinator restart; mission context token budget; `planner` role dropped between outline and PRD.
- **4 handwaves:** async audit row race; **roster-vs-contractor coexistence (urgent product decision)**; testing story for ephemeral spawns; board integration misclassified as Should-Have.

**Gap #14 (coexistence) is the single decision that unblocks both the Auto fix AND Mission Zero.** Until we pick "missions use ephemeral contractors always" vs "missions use roster agents + contractors are a premium tier" vs "coordinator picks per-task based on complexity," we cannot write correct seeding code for Mission Zero or correct fallback code for Auto.

### 3. Mission Zero `agent_error` evidence
Log truth (not my earlier fiction): Mission Zero plan created (8 tasks), user approved, dispatcher assigned 3, task `dc3d11fe-dbb3-473a-9b30-c0c570956582` ("Extract compliance and technical assurance evidence") failed with `agent_error`, reconciler treated as fatal, 4 downstream tasks skipped, run → `failed` at 12:36:04. **Root cause is execution-time, not seeding** — needs log filter on that task ID and check of the agent's tool set vs task requirements. Unrelated to 82C coordinator correctness.

### 4. Spec drift documented
- **Verification:** PRD-103 designs a worker endpoint `POST /verify-task`; actual code runs verification in-process inside the coordinator. Acceptable for sequential path; becomes a bottleneck once parallel missions dispatch 5+ concurrent verifications.
- **Context backend name:** PRD-107 designs `CONTEXT_BACKEND` env var; actual `config.SHARED_CONTEXT_BACKEND`. Minor — worth renaming for consistency or updating the PRD.
- **Mission Chat Mode:** 82A defers to 82C; 82C defers to 82D; MEMORY says Ralph shipped it overnight. Needs reconciliation against at least one spec.

---

## Recommended next actions — ordered by unblocking value

### Blocker tier — do before any more mission work

1. **Answer Gap #14 (roster vs contractor coexistence).** One-page decision doc. Until this is written, every "fix Mission Zero" and "fix Auto" attempt is a coin flip. Options: (a) missions always use ephemeral contractors (ship PRD-104); (b) missions use roster agents only (kill PRD-104 entirely); (c) hybrid — coordinator picks per-task. Pick one with rationale, land it as an addendum to 104.
2. **Debug the `agent_error` on task `dc3d11fe…`.** Railway log filter by task id, then DB check `orchestration_tasks.result_ref` / error field. This is the ground-truth signal for "does the current 82A/B/C pipeline survive a real mission?" Without it, every other metric is theatrical.

### High tier — ship before next demo

3. **PRD-104 implementation (if Gap #14 answer = ship it).** Scope: `ContractorConfig`, `create_ephemeral_agent`, role→model mapping, cognitive diversity swap. Drop the K8s/lifecycle/cleanup/TTL pieces for v1 — just ship the spawn-config pattern so missions can use role-appropriate model+prompt combinations without polluting the agent roster.
4. **PRD-105 `MissionBudgetManager` class + `mission_task_id` FK.** The dispatcher hook is live; the manager and cost attribution are missing. Without the FK on `llm_usage`, nobody can answer "what did this mission cost." 1-day job.
5. **Auto agent proper fix (from sibling audit).** Depends on Gap #14 outcome. If contractors ship, Auto becomes "spawn a contractor per chat message." If roster-only, Auto becomes a hidden `is_system_agent=True` row per workspace.

### Medium tier — debt / observability

6. **PRD-106 `TelemetryService` + `mission_events` schema.** Without 106, the ML/learning feedback loop for 110-116 is dead. But 110-116 aren't on anyone's near-term roadmap, so 106 can wait if it has to.
7. **PRD-108 experiment harness.** The Qdrant adapter shipped, the default is `vector_field`, and nobody ran the 30-run A/B. Either run the experiment and publish the gate report, or fall the default back to `redis` and mark PRD-108 as "experiment skipped, prototype retained."
8. **82B gaps:** verify/complete `_score_history` wiring, cross-task consistency, save-as-routine, telemetry endpoint count. Confirm PRD-82B v1.0 "production learnings" patches (rebalanced weights, `_LARGE_CONTEXT_MODELS`, `must_pass=false` default) actually landed.
9. **Reconcile "Phase 6 Chat Mission Mode".** Either add it to 82A as a phase 6 addendum or move it into 82C Chapter 5. Code without a spec is worse than no code.

### Low tier — cleanup

10. PRD-103 worker endpoint decision: keep in-process or extract to `agent-opt-worker`? Only matters once parallelism exceeds ~3 concurrent verifications.
11. Config var rename: `SHARED_CONTEXT_BACKEND` → `CONTEXT_BACKEND` to match PRD-107, or update the PRD.
12. Tool policy layering (105 §4-tier): design is clean, implementation is absent, not urgent until a mission burns a workspace budget.

---

## Traceability map

```
100 (roadmap)
 ├── 101 (schema) ────────────► 82A ✅ built
 ├── 102 (coordinator) ────────► 82A ✅ built
 ├── 103 (verification) ───────► 82A ✅ in-process (spec drift on worker endpoint)
 ├── 104 (ephemeral agents) ───► 82D ❌ not started
 ├── 105 (budget) ─────────────► 82C 🟡 admission gate only
 ├── 106 (telemetry) ──────────► ⟂  ❌ no impl PRD, not built
 ├── 107 (context ports) ──────► ⟂  ✅ ports shipped
 └── 108 (memory field) ───────► ⟂  🟡 adapter shipped, experiment not run

82A (sequential MVP) ──────► ✅ core path works
82B (intelligence layer) ──► 🟡 replan/templates/archive shipped; history scoring, consistency, telemetry API partial
82C (parallel + budget) ───► ✅ parallel core + admission gate + wiring tests
82D (guidance + contractors) ► ❌ does not exist yet
```

---

## Appendix: grep verification log

Commands run and files checked (for reproduction):

```
orchestrator/modules/coordination/*.py            → 8 files present (planner, dispatcher, reconciler, agent_matcher, verification, deterministic_checks, templates, __init__)
orchestrator/services/coordinator_service.py      → present (~2000 lines)
orchestrator/core/models/orchestration{,_enums}.py → present
orchestrator/core/ports/context.py                → present
orchestrator/modules/context/adapters/vector_field.py → VectorFieldSharedContext:43
orchestrator/modules/context/factory.py           → SHARED_CONTEXT_BACKEND read
orchestrator/alembic/versions/prd82a_orchestration_tables.py  → present
orchestrator/alembic/versions/prd82b_us005_replan_support.py  → present
orchestrator/alembic/versions/prd82b_us009_orchestration_archive.py → present
orchestrator/alembic/versions/prd82c_parallel_schema.py → parallel_group, complexity, estimated_tokens columns
orchestrator/tests/test_82c_wiring.py             → present
orchestrator/tests/test_budget_gate.py            → present
orchestrator/tests/test_coordinator_parallel.py   → present
orchestrator/tests/test_synthesis_{executor,auto_insert}.py → present
orchestrator/tests/test_parallel_decomposition.py → present
orchestrator/tests/test_complexity_detection.py   → present
orchestrator/tests/test_dispatcher_parallel.py    → present

NOT FOUND:
  ContractorConfig, create_ephemeral_agent          (PRD-104)
  MissionBudgetManager, BudgetExceededError         (PRD-105)
  TelemetryService, mission_events, mv_model_performance, MISSION_TASKS_TOTAL  (PRD-106)
  experiment_results, VERIFIER_MODEL_SELECTION       (PRD-108, PRD-103)
  mission_task_id in usage_tracker.py               (PRD-105/106)
```

Ground truth recorded at session start. If code lands after this audit, re-run the grep commands in this appendix to refresh.
