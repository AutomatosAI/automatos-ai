# PRD-125 — Workflow Decoupling & Mission Migration

**Version:** 1.0
**Type:** Migration / Cleanup
**Status:** Draft
**Priority:** P0 (Phase 0), P1 (Phases 1-5)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-04-01
**Supersedes:** PRD-10 (Workflow Orchestration Engine), PRD-13 (Workflows Enhancement), PRD-59 (Workflow Engine V2), PRD-68 Phase 2 (Workflow Bridge)
**Dependencies:** PRD-82A (Sequential Missions), PRD-82B (Mission Extensions), PRD-108 (Memory Field)
**Touches:** PRD-68 (Progressive Complexity Routing), PRD-52 (Unified Analytics), PRD-72 (Activity Command Centre)

---

## 1. Goal

Remove the legacy Workflow execution system and route all complex-task handling through the Mission system (PRD-82A/B/C). The old workflow engine is dead code that actively causes production bugs — the chat-to-workflow bridge hangs when triggered because `WorkspaceManager` tries to create filesystem directories on Railway (read-only). This PRD retires ~5,000 lines of execution code, ~15 frontend components, and consolidates analytics onto `OrchestrationRun`/`OrchestrationTask` data.

## 2. Problem Statement

### The Production Bug

When a user sends a complex message (organ/organism complexity), AutoBrain in `consumers/chatbot/auto.py` classifies it as `Action.WORKFLOW`. This triggers `_stream_workflow_bridge()` in `api/chat.py:70-196`, which:

1. Creates a transient `Workflow` + `WorkflowExecution` record
2. Calls `execute_workflow_with_progress()` as a fire-and-forget `asyncio.create_task()`
3. Opens an SSE stream waiting for events from the execution

The execution immediately crashes because `WorkspaceManager` (line 1562 of `api/workflows.py`) calls `mkdir` on `/var/lib/automatos/results` — a path that doesn't exist and can't be created on Railway. The crash is silent (async, no error propagation). The SSE stream never receives events and hangs forever, sending heartbeat pings into the void.

**User experience:** Chat freezes. No error. No recovery. Must reload the page.

### The Architecture Mismatch

The platform has moved to a new execution model:

| Old System (Dead) | New System (Live) |
|---|---|
| `Workflow` + `WorkflowExecution` models | `OrchestrationRun` + `OrchestrationTask` models |
| `execute_workflow_with_progress()` 9-stage pipeline | `coordinator_service.py` 5s tick loop |
| `WorkspaceManager` filesystem workspaces | S3 document storage + shared fields (PRD-108) |
| `WorkflowStreamManager` SSE events | `OrchestrationEvent` table + React Query polling |
| `WorkflowAnalyticsService` | `GET /api/missions/stats` |
| Single-agent execution | Multi-agent with DAG dependencies |
| No verification | Two-stage verification (deterministic + LLM judge) |

The two systems share **zero code paths**. They coexist as parallel stacks with independent models, APIs, services, and frontend components. The only caller of the old system is the chat workflow bridge.

### Scope of Dead Code

**Backend (36+ files):**
- `api/workflows.py` — 3,400 lines, ~2,900 are execution-only
- `consumers/workflows/` — 4 files (streaming, analytics, usage tracker)
- `core/services/workspace_manager.py` — filesystem workspace creation
- `modules/orchestrator/service.py` — legacy orchestrator service
- `modules/orchestrator/pipeline.py` — workflow pipeline runner
- `api/workflow_templates.py` — replaced by recipes
- `api/widget_workflows.py` — widget execution control

**Frontend (15+ files):**
- `app/api/workflows/stream/route.ts` — SSE proxy
- `app/api/workflow/stream/route.ts` — legacy SSE proxy
- `components/workflows/workflow-stream-viewer.tsx`
- `components/workflows/interactive-workflow-execution.tsx`
- Various execution modal/panel components

---

## 3. What Ships

| # | Change | Phase | Risk | Files |
|---|--------|-------|------|-------|
| 0a | Add `MISSION` to Action enum | Phase 0 | None | 1 |
| 0b | Route organ/organism → mission suggestion | Phase 0 | Low | 2 |
| 0c | Error-guard `_stream_workflow_bridge()` | Phase 0 | None | 1 |
| 1a | Chat → Mission launch flow (backend) | Phase 1 | Low | 2 |
| 1b | Mission suggestion UI (frontend) | Phase 1 | Low | 2-3 |
| 2a | Analytics queries → OrchestrationRun | Phase 2 | Med | 3-4 |
| 2b | Activity feed → mission data | Phase 2 | Med | 2-3 |
| 3a | Delete workflow execution pipeline | Phase 3 | High | 8-10 |
| 3b | Trim `api/workflows.py` to CRUD-only | Phase 3 | High | 1 |
| 3c | Delete frontend execution components | Phase 3 | Med | 5-8 |
| 4a | Recipe execution → missions | Phase 4 | Med | 2-3 |
| 5a | Deprecate workflow DB tables | Phase 5 | Low | 1 migration |

## 4. What Does NOT Ship (Deferred)

| Deferred | Why |
|----------|-----|
| Drop workflow/workflow_execution DB tables | 90-day data retention; tables deprecated but kept |
| Migrate historical workflow execution data to mission tables | Different schemas, not worth the effort; keep read-only access |
| Auto-launch missions without user confirmation | UX decision: always ask first in Phase 0-1 |
| Remove `Workflow` model from ORM | Still needed for historical CRUD reads |
| Frontend workflow CRUD UI removal | Users may still browse historical workflows/recipes |

---

## 5. Phase 0 — Unbreak Chat (Day 1, Hotfix)

### 5.1 Add `MISSION` Action to AutoBrain

#### Problem

`Action` enum has only 3 values: `RESPOND`, `DELEGATE`, `WORKFLOW`. The LLM classification prompt maps organ/organism → `"workflow"`. There is no mission-aware action.

#### Design

Add `MISSION` to the enum and update the classification prompt:

```python
class Action(str, Enum):
    RESPOND = "respond"
    DELEGATE = "delegate"
    WORKFLOW = "workflow"    # DEPRECATED — kept for backward compat
    MISSION = "mission"      # NEW — complex multi-step tasks
```

Update the LLM prompt (line ~675) to map:
- atom → `"respond"`
- molecule/cell → `"delegate"`
- organ/organism → `"mission"` (was `"workflow"`)

Update `_parse_assessment()` to map both `"mission"` and `"workflow"` strings → `Action.MISSION` (handles LLM still saying "workflow").

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/consumers/chatbot/auto.py` | Add `MISSION` to enum, update prompt, update parser |

#### Acceptance Criteria

1. `Action.MISSION` exists and is usable
2. LLM prompt instructs organ/organism → `"mission"`
3. Both `"mission"` and `"workflow"` strings parse to `Action.MISSION`
4. Existing RESPOND/DELEGATE paths unchanged

---

### 5.2 Route MISSION → Suggestion (Not Execution)

#### Problem

When AutoBrain returns `Action.WORKFLOW`, chat.py calls `_stream_workflow_bridge()` which hangs. We need to intercept `Action.MISSION` before it hits the dead code.

#### Design

In `api/chat.py`, add a new routing block for `Action.MISSION` that:

1. Sends a **mission suggestion** SSE event to the frontend:
   ```
   d:{"type":"mission-suggestion","goal":"<user message>","complexity":"organ","agent_id":"<routed agent>"}
   ```
2. Falls through to `DELEGATE` path so the user still gets an immediate response from a single agent
3. The agent response serves as an initial analysis; the user can then choose to escalate to a full mission

This means the user always gets a response AND a suggestion card — never a blank hang.

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/api/chat.py` | Add `Action.MISSION` handling block (~line 479), emit suggestion event, fall through to delegate |

#### Acceptance Criteria

1. Message classified as organ/organism → user gets a text response AND a mission-suggestion event
2. No call to `_stream_workflow_bridge()` for `Action.MISSION`
3. SSE stream completes normally (no hang)
4. Simple messages (atom/molecule/cell) unaffected

---

### 5.3 Error-Guard the Workflow Bridge (Safety Net)

#### Problem

Even though Phase 0.2 prevents new traffic from hitting `_stream_workflow_bridge()`, the function still exists and could be called by other paths. If it crashes, the SSE stream hangs forever.

#### Design

Wrap the fire-and-forget `asyncio.create_task()` in `_stream_workflow_bridge()` with error propagation:

```python
# Before (line 161):
asyncio.create_task(_run_workflow())

# After:
try:
    await asyncio.wait_for(_run_workflow(), timeout=120)
except Exception as e:
    yield format_aisdk_data({"type": "workflow-update", "status": "error", "error": str(e)})
    yield format_aisdk_data({"type": "finish", "finishReason": "error"})
    return
```

This converts the infinite hang into a 2-minute timeout with an error message.

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/api/chat.py` | Wrap `_run_workflow()` in try/await with timeout, emit error on failure |

#### Acceptance Criteria

1. If `_stream_workflow_bridge()` is ever called and execution fails, SSE stream terminates with error event within 120s
2. Error message is sent to frontend (not swallowed)
3. No change to happy-path behavior

---

## 6. Phase 1 — Chat → Mission Bridge (Week 1)

### 6.1 Backend: Launch Mission from Chat

#### Problem

Phase 0 sends a suggestion but the user can't act on it from chat yet. Need a way to accept the suggestion and create a mission.

#### Design

The frontend already has `useCreateMission()` in `hooks/use-missions-api.ts` and a "Launch as Mission" button in `message-actions.tsx`. The missing piece is connecting the mission-suggestion event to this existing UI.

No new backend endpoint needed — the frontend calls `POST /api/missions` directly with the goal text.

**Chat context enrichment:** When a mission is created from chat, include the chat_id and recent history in the mission config so the planner has conversational context:

```python
# In POST /api/missions handler, if source == "chat":
config = {
    "source": "chat",
    "chat_id": chat_id,
    "context_messages": last_5_messages,
}
```

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/api/missions.py` | Accept optional `chat_id` + `context_messages` in `MissionCreateRequest.config` — pass to planner |
| `orchestrator/modules/coordination/planner.py` | If `config.source == "chat"`, include conversation context in decomposition prompt |

#### Acceptance Criteria

1. `POST /api/missions` with `config.source = "chat"` includes conversation context in planning
2. Planner produces better task decomposition when given chat history
3. Existing non-chat mission creation unaffected

---

### 6.2 Frontend: Mission Suggestion Card

#### Problem

The `mission-suggestion` SSE event from Phase 0 needs to render as an actionable UI element in chat.

#### Design

When the chat stream receives a `mission-suggestion` event:

1. Render an inline card below the agent's text response:
   - Title: "This task could benefit from a Multi-Agent Mission"
   - Body: Shows the detected complexity and suggested goal
   - Actions: **"Launch Mission"** button + **"No thanks"** dismiss
2. "Launch Mission" calls `useCreateMission({ goal, config: { source: "chat", chat_id } })`
3. On success, show a toast with link to mission detail page
4. "No thanks" dismisses the card (no action)

#### Files to Change

| File | Change |
|------|--------|
| `frontend/components/chatbot/chat.tsx` | Handle `mission-suggestion` event type in stream parser |
| `frontend/components/chatbot/mission-suggestion-card.tsx` | **NEW** — renders suggestion card with launch/dismiss actions |
| `frontend/components/chatbot/message-actions.tsx` | Wire existing "Launch as Mission" to also work from suggestion |

#### Acceptance Criteria

1. Complex messages show agent response + suggestion card
2. "Launch Mission" creates a mission and shows confirmation
3. "No thanks" dismisses cleanly
4. Card does not appear for simple messages

---

## 7. Phase 2 — Analytics Migration (Week 2)

### 7.1 Dashboard Analytics → Mission Data

#### Problem

Analytics endpoints query `WorkflowExecution` for execution counts, success rates, and model usage. With no new workflow executions being created (Phase 0 redirects to missions), these numbers are stale.

#### Design

Update analytics queries to UNION or replace with `OrchestrationRun`/`OrchestrationTask` data:

| Metric | Old Source | New Source |
|--------|-----------|------------|
| Total executions | `COUNT(workflow_executions)` | `COUNT(orchestration_runs)` |
| Success rate | `executions WHERE status='completed'` | `runs WHERE state='completed'` |
| Active executions | `executions WHERE status='running'` | `runs WHERE state='running'` |
| Model usage | `execution.models_used` JSONB | `SUM(task.tokens_used)` grouped by agent model |
| Execution duration | `completed_at - started_at` on executions | Same fields on `orchestration_runs` |
| Agent performance | Joined via `execution.agent_id` | `orchestration_tasks.assigned_agent_id` (per-task granularity) |

**Transition strategy:** For 90 days, UNION both sources (old + new). After Phase 5 deprecation window, drop old source.

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/api/analytics.py` | Update dashboard queries to include `orchestration_runs` |
| `orchestrator/api/analytics_real.py` | Update real-time stats to include mission data |
| `orchestrator/api/llm_analytics.py` | Add mission token spend to LLM cost tracking |
| `orchestrator/consumers/workflows/analytics.py` | Mark deprecated; new callers use `GET /api/missions/stats` |

#### Acceptance Criteria

1. Dashboard shows combined workflow + mission execution counts
2. Success rate reflects mission completions
3. LLM cost tracking includes mission token spend
4. Historical workflow data still visible (not dropped)

---

### 7.2 Activity Feed → Mission Events

#### Problem

The Activity page shows workflow execution history. New executions are all missions.

#### Design

Update the activity feed to show `OrchestrationRun` entries alongside (and eventually replacing) `WorkflowExecution` entries.

The mission system already has `OrchestrationEvent` with 30+ event types — richer than the workflow execution_log JSONB.

#### Files to Change

| File | Change |
|------|--------|
| `frontend/components/activity/execution-detail.tsx` | Support both workflow and mission execution detail views |
| `frontend/hooks/use-missions-api.ts` | Already has `useMissions()` — ensure activity page uses it |
| `orchestrator/api/analytics.py` | Add `/api/analytics/activity-feed` that merges both sources |

#### Acceptance Criteria

1. Activity feed shows mission runs with state, duration, task count
2. Clicking a mission run opens mission detail page
3. Old workflow executions still visible in feed (historical)

---

## 8. Phase 3 — Remove Workflow Execution Path (Week 3)

> **IMPORTANT — Corrected Execution Order (Review 2026-04-08)**
>
> The original order (3a → 3b → 3c) is WRONG. `analytics.py`, `usage_tracker.py`, and `workspace_manager.py` are actively imported by `api/workflows.py` execution code. They can only be deleted AFTER workflows.py is trimmed.
>
> **Correct order:** 3b (trim workflows.py) → 3a (delete now-dead files) → 3c (frontend cleanup)
>
> Files NOT safe to delete until 3b completes:
> - `consumers/workflows/analytics.py` — used by `api/analytics.py` (4 endpoints), `api/execution_history.py`
> - `consumers/workflows/usage_tracker.py` — used in `execute_workflow_with_progress()`
> - `core/services/workspace_manager.py` — used extensively in workflows.py execution + result APIs
>
> `modules/orchestrator/pipeline.py` is NEW PRD-59 code (not legacy) — keep if needed for Phase 4 recipe execution.

### 8.1 Delete Execution Pipeline

#### Problem

With Phase 0-2 complete, no code path calls the workflow execution pipeline. It's dead code.

#### Design

Delete the following files entirely:

| File | Lines | Reason |
|------|-------|--------|
| `orchestrator/consumers/workflows/streaming.py` | ~500 | `WorkflowStreamManager` — replaced by mission events |
| `orchestrator/consumers/workflows/analytics.py` | ~200 | `WorkflowAnalyticsService` — replaced Phase 2 |
| `orchestrator/consumers/workflows/usage_tracker.py` | ~150 | `ModelUsageTracker` — replaced by mission token tracking |
| `orchestrator/core/services/workspace_manager.py` | ~200 | `WorkspaceManager` — filesystem workspaces not needed |
| `orchestrator/modules/orchestrator/service.py` | ~800 | `EnhancedOrchestratorService` — legacy, not in live path |
| `orchestrator/modules/orchestrator/pipeline.py` | ~300 | `WorkflowPipeline` — replaced by coordinator tick |

**DO NOT DELETE** (shared by mission system):
- `orchestrator/modules/orchestrator/stages/` — Task decomposer, agent selector, context engineering, etc. are reused by the coordinator.

#### Files to Change

| File | Action |
|------|--------|
| Files listed above | DELETE |
| `orchestrator/consumers/workflows/__init__.py` | Remove deleted exports |
| Any file importing deleted modules | Remove import + usage |

#### Acceptance Criteria

1. `grep -r "WorkflowStreamManager" orchestrator/` → 0 results
2. `grep -r "WorkspaceManager" orchestrator/` → 0 results
3. `grep -r "execute_workflow_with_progress" orchestrator/` → 0 results
4. `grep -r "EnhancedOrchestratorService" orchestrator/` → 0 results
5. All existing tests pass
6. Server starts cleanly

---

### 8.2 Trim `api/workflows.py`

#### Problem

`api/workflows.py` is 3,400 lines. ~2,900 lines are execution pipeline code. The remaining ~500 lines are CRUD operations for listing/viewing historical workflows — still needed.

#### Design

Remove from `api/workflows.py`:
- `execute_workflow_with_progress()` and all helper functions
- All streaming endpoints (`/stream`, `/stream/aisdk`, `/live-progress`)
- `POST .../execute`, `POST .../execute-advanced`
- All `WorkflowStageTracker` usage
- All Redis event publishing

Keep in `api/workflows.py`:
- `GET /api/workflows` — list workflows
- `GET /api/workflows/{id}` — get workflow detail
- `POST /api/workflows` — create workflow (for recipe creation)
- `PUT /api/workflows/{id}` — update workflow
- `DELETE /api/workflows/{id}` — delete workflow
- `GET /api/workflows/executions/` — list historical executions (read-only)
- `GET /api/workflows/executions/{id}` — get historical execution detail (read-only)
- `GET /api/workflows/stats/dashboard` — historical stats

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/api/workflows.py` | Remove execution functions, streaming endpoints; keep CRUD |

#### Acceptance Criteria

1. `api/workflows.py` is under 600 lines
2. Historical workflow/execution data still accessible via API
3. No execution endpoints remain (POST execute, streaming)
4. `config.AUTOMATOS_RESULTS_BASE` can be removed from `config.py`

---

### 8.3 Delete Frontend Execution Components

#### Design

Delete:

| File | Reason |
|------|--------|
| `frontend/app/api/workflows/stream/route.ts` | SSE proxy for dead execution streaming |
| `frontend/app/api/workflow/stream/route.ts` | Legacy SSE proxy |
| `frontend/components/workflows/workflow-stream-viewer.tsx` | Real-time execution viewer |
| `frontend/components/workflows/interactive-workflow-execution.tsx` | Interactive execution dashboard |

Keep (historical viewing):
- `execution-kitchen.tsx` — render historical execution details (read-only)
- `workflow-management.tsx` — recipe/workflow CRUD UI
- `create-workflow-modal.tsx`, `edit-workflow-modal.tsx` — recipe editing

#### Files to Change

| File | Action |
|------|--------|
| Files listed above | DELETE |
| Any component importing deleted files | Remove import + usage |

#### Acceptance Criteria

1. No frontend SSE workflow stream proxies exist
2. Frontend builds cleanly (`npm run build`)
3. Historical execution viewing still works
4. Recipe management UI still works

---

## 9. Phase 4 — Recipe Execution via Missions (Week 4)

### 9.1 Recipes Execute as Missions

#### Problem

Recipes currently create `WorkflowExecution` records and run through the workflow pipeline. With Phase 3 deleting the pipeline, recipe execution needs to route through missions.

#### Design

The mission planner already supports template-based planning (PRD-82B US-001). Recipes become mission templates:

1. When a recipe is executed, create an `OrchestrationRun` with `template_id` pointing to the recipe
2. The planner's template matching checks recipe definitions first
3. Recipe steps map to `OrchestrationTask` entries with explicit agent roles and prompts
4. `RecipeExecution` model records link to `OrchestrationRun.id` instead of `WorkflowExecution.id`

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/api/workflow_recipes.py` | Execution endpoint calls `coordinator.create_mission()` with template config |
| `orchestrator/modules/coordination/templates.py` | Add recipe-to-template adapter |
| `orchestrator/core/models/core.py` | `RecipeExecution.execution_id` → nullable FK to `orchestration_runs` (add column) |

#### Acceptance Criteria

1. Recipe execution creates an `OrchestrationRun` (not a `WorkflowExecution`)
2. Recipe steps map to mission tasks correctly
3. Recipe execution history shows in both recipes UI and missions list
4. Existing recipe CRUD unaffected
5. Cron-scheduled recipes trigger missions on schedule

---

## 10. Phase 5 — Database Cleanup (Week 5)

### 10.1 Deprecate Workflow Tables

#### Problem

`workflows` and `workflow_executions` tables still exist with old data. No code writes to them after Phase 3.

#### Design

Create an Alembic migration that:
1. Adds `deprecated_at TIMESTAMP DEFAULT NOW()` to `workflows` and `workflow_executions`
2. Adds a comment to each table: `'DEPRECATED by PRD-125. Read-only. Drop after 2026-07-01.'`
3. Does NOT drop the tables — 90-day retention for historical queries

After 90 days (separate follow-up migration):
- Drop `workflows`, `workflow_executions`, `workflow_agents` tables
- Remove `Workflow`, `WorkflowExecution` models from ORM

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/alembic/versions/new_deprecate_workflow_tables.py` | **NEW** — migration adding deprecated_at |
| `orchestrator/core/models/core.py` | Add `deprecated_at` column to Workflow/WorkflowExecution; add code comment |

#### Acceptance Criteria

1. `alembic upgrade head` succeeds
2. Tables have `deprecated_at` column set
3. No code writes new rows to deprecated tables
4. Historical reads still work

---

## 11. Shared Stage Components — DO NOT DELETE

The 9-stage components in `modules/orchestrator/stages/` are actively reused by the mission coordinator:

| Stage Component | Used By (Missions) |
|---|---|
| `RealTaskDecomposer` | `MissionPlanner.decompose()` |
| `IntelligentAgentSelector` | `AgentMatcher` |
| `ContextEngineeringIntegrator` | `coordinator._execute_task()` |
| `AgentExecutionManager` | `coordinator._execute_task()` |
| `ResultAggregator` | `MissionReconciler` |
| `OutputQualityAssessor` | `VerificationService` |
| `WorkflowMemoryIntegrator` | Field context injection |

Only the **wrappers** are deleted (`WorkflowPipeline`, `WorkflowContext`, `EnhancedOrchestratorService`). The stage implementations stay.

---

## 12. Migration & Rollback

| Phase | Backward Compatible | Rollback Strategy |
|-------|--------------------|--------------------|
| 0 | Yes — adds new action, doesn't remove old | Revert commit |
| 1 | Yes — new UI element, no behavior change for existing users | Revert commit |
| 2 | Yes — UNION queries include both sources | Revert commit |
| 3 | **No** — deletes code | Revert PR; code is in git history |
| 4 | Partially — recipes change execution target | Revert PR + restore old execution endpoint |
| 5 | Yes — tables kept, just marked deprecated | Revert migration |

**Deploy order:** Phase 0 as hotfix. Phases 1-5 as separate PRs on a feature branch, merged sequentially.

---

## 13. Test Plan

### Per-Phase Verification

| Phase | Tests to Run |
|-------|-------------|
| 0 | `pytest tests/api/test_chat.py tests/api/test_chat_errors.py -v` |
| 1 | `pytest tests/api/test_chat.py tests/api/test_mission_journeys.py -v` |
| 2 | `pytest tests/api/test_analytics.py tests/api/test_llm_analytics.py -v` |
| 3 | `python3 tests/run_nightly.py` (full 376-test suite) + grep verification |
| 4 | `pytest tests/api/test_recipes.py tests/api/test_recipe_cron.py -v` |
| 5 | `alembic upgrade head` + `python3 tests/run_health_regression.py` |

### Manual Verification (Phase 0)

1. Send "check my calendar and post on Slack" in chat
2. Verify: response received (not hung) + mission suggestion card appears
3. Verify: simple messages still route normally

### Grep Verification (Phase 3)

```bash
grep -r "execute_workflow_with_progress" orchestrator/  # expect 0
grep -r "WorkspaceManager" orchestrator/                # expect 0
grep -r "_stream_workflow_bridge" orchestrator/          # expect 0
grep -r "WorkflowStreamManager" orchestrator/           # expect 0
grep -r "AUTOMATOS_RESULTS_BASE" orchestrator/           # expect 0 (after config cleanup)
```

---

## 14. Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| Chat hang rate on complex messages | 100% (always hangs) | 0% |
| Dead code lines (backend) | ~5,000 | 0 |
| Dead frontend components | ~15 | 0 |
| Workflow-related files | 36+ | ~5 (CRUD-only) |
| `api/workflows.py` line count | 3,400 | <600 |
| Analytics data freshness | Stale (no new executions) | Live (mission data) |
