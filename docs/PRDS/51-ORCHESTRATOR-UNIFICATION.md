# PRD-51: Recipe Execution Stabilization & Orchestrator Unification

## Context

The recipe system redesign (PRD current `prd.json`, 25 user stories) is feature-complete. However, end-to-end recipe execution has gaps that prevent real-world usage. Separately, PRD-50 (Universal Orchestrator Router) is ~80% implemented on branch `ralph/universal-orchestrator-router` in worktree `automatos-ai-pr/`. These two efforts must be sequenced correctly to avoid a merge nightmare (78 files, 15k+ lines of divergence between branches).

### What's Working
- Recipe CRUD, marketplace, sharing, installation
- 9-stage workflow engine (execute_workflow_with_progress)
- Cook button → Workflow → WorkflowExecution → ExecutionTheater
- Theater sub-components (StageProgress, StepExecution, SelfLearningPanel)
- Composio tool bridge in agent_factory.py (agents in recipe execution now receive `composio_execute`)
- Action name reconstruction from display names to Composio API identifiers

### What's Broken / Incomplete
- DB `composio_actions_cache` stores display names ("Fetch emails") not API identifiers ("GMAIL_FETCH_EMAILS")
- Gmail OAuth tokens need refresh in Composio (not a code issue, admin action)
- Uncommitted changes on `ralph/recipe-system-redesign`: agent_factory.py, tool_executor.py, + 12 other files
- Two separate tool-loading code paths: chatbot (`service.py → tool_router.py → ToolRegistry`) vs recipe (`agent_factory.py → _build_tool_schemas()`)
- PRD-50 Universal Orchestrator is 80% built on a diverged branch that can't merge cleanly yet

---

## Phase 1: Stabilize Recipe Execution (Current Branch)

**Branch:** `ralph/recipe-system-redesign`
**Goal:** Get recipe execution working end-to-end with Composio tools, commit all changes, merge to main.

### US-051-001: Commit Composio tool bridge changes

**Description:** The Composio tool bridge (composio_execute injection into agent_factory.py + action name reconstruction in tool_executor.py) is working but uncommitted.

**Files changed (uncommitted):**
- `orchestrator/modules/agents/factory/agent_factory.py` — Added ComposioActionCache import, composio_execute schema injection in `execute_with_prompt()`, `_build_composio_hints()` method, workspace_id passthrough to tool execution
- `orchestrator/core/composio/tool_executor.py` — Added action name reconstruction (display name → API identifier format)

**Checklist:**
- [ ] Review all uncommitted changes across 14 files (run `git diff --name-only`)
- [ ] Remove any debug logging (e.g., writes to `/tmp/composio_debug.log`)
- [ ] Verify agent_factory.py changes are purely additive (no existing behavior modified)
- [ ] Verify tool_executor.py reconstruction logic handles edge cases (action names with special characters, multi-word names)
- [ ] Run typecheck on orchestrator (`./venv/bin/python -c "import api.main"`)
- [ ] Commit with message: `fix: Bridge Composio tools into recipe execution pipeline`

---

### US-051-002: Fix Composio action cache sync to store proper identifiers

**Description:** The `composio_actions_cache` table stores human-readable display names ("Fetch emails", "Create email draft") instead of Composio API identifiers ("GMAIL_FETCH_EMAILS", "GMAIL_CREATE_EMAIL_DRAFT"). The current workaround reconstructs API names at runtime (`APP_NAME + "_" + display_name.upper().replace(" ", "_")`), but this is fragile — action names with special characters, apostrophes (e.g., "Revoke a file's public url"), or non-standard casing will break.

**Root cause:** The sync process that populates `composio_actions_cache` from Composio SDK stores `action.display_name` instead of `action.name` (the enum identifier).

**Files to investigate:**
- `orchestrator/core/composio/` — Find the sync/cache population code
- `orchestrator/core/models/composio_cache.py` — ComposioActionCache model definition

**Checklist:**
- [ ] Find where `composio_actions_cache` is populated (likely a sync script or background task)
- [ ] Fix to store both `action.name` (API identifier like "GMAIL_FETCH_EMAILS") and `action.display_name` (human readable)
- [ ] Add `action_identifier` column to `composio_actions_cache` if needed, or fix `action_name` to store the API identifier
- [ ] Re-run the sync for existing data
- [ ] Update `_build_composio_hints()` in agent_factory.py to use the proper identifier directly (remove reconstruction logic)
- [ ] Update `tool_executor.py` to use the proper identifier directly (remove reconstruction logic)
- [ ] Test: Execute a recipe with Gmail agent → verify action name sent to Composio is correct without reconstruction

---

### US-051-003: Verify end-to-end recipe execution with Composio tools

**Description:** Test the full chain: Cook button → Workflow creation → WorkflowExecution → execute_workflow_with_progress → agent calls composio_execute → Composio API returns results → ExecutionTheater displays results.

**Prerequisites:**
- US-051-001 committed
- Gmail OAuth tokens refreshed in Composio dashboard
- Test recipe exists with 2+ steps, at least one agent with Gmail/Slack tools assigned

**Checklist:**
- [ ] Refresh Gmail OAuth connection in Composio dashboard for workspace `ae8320bc-95e1-4de1-bbe9-396bef19cbf8`
- [ ] Create (or use existing) test recipe: Step 1 = "Read today's emails" (Communication Agent with Gmail), Step 2 = "Summarize findings" (any agent)
- [ ] Click Cook on the recipe card
- [ ] Verify toast shows "Recipe execution started"
- [ ] Verify ExecutionKitchen opens and shows the workflow
- [ ] Verify 9-stage progress bar advances (should skip stages 1-2 in RECIPE mode)
- [ ] Verify Step 1 agent receives `composio_execute` tool and calls `GMAIL_FETCH_EMAILS`
- [ ] Verify Composio returns actual email data (not OAuth error)
- [ ] Verify Step 2 agent receives Step 1's output and produces a summary
- [ ] Verify execution completes with status='completed'
- [ ] Verify TheaterStepExecution shows tool calls, results, and messages for each step
- [ ] Check DB: `workflow_executions` has status=completed, `output_data` populated
- [ ] Check DB: `recipe_executions` has `execution_metadata` with workflow references

---

### US-051-004: Fix remaining frontend TypeScript errors

**Description:** There are ~21 pre-existing TypeScript errors across hooks/ and temp-api-client.ts files (merge conflicts, unterminated strings). These should be cleaned up before merging to main.

**Checklist:**
- [ ] Run `npx tsc --noEmit` from `frontend/` directory
- [ ] Fix errors in hooks/ directory (likely merge conflict artifacts)
- [ ] Fix errors in temp-api-client.ts
- [ ] Remove temp-api-client.ts if it's unused
- [ ] Verify no new errors introduced by recipe system changes
- [ ] All frontend components compile clean

---

### US-051-005: Commit all changes and merge to main

**Description:** Final commit of all Phase 1 work, create PR, merge to main.

**Checklist:**
- [ ] Run full typecheck (backend + frontend)
- [ ] Verify dev server starts without errors (`uvicorn main:app --reload`)
- [ ] Verify frontend builds (`npm run build`)
- [ ] Commit any remaining changes
- [ ] Push `ralph/recipe-system-redesign` to remote
- [ ] Create PR to main with summary of all recipe system changes
- [ ] Merge PR (or get approval if required)

---

## Phase 2: Rebase Universal Orchestrator (Separate Branch)

**Branch:** `ralph/universal-orchestrator-router` (worktree at `automatos-ai-pr/`)
**Goal:** Rebase PRD-50 work onto updated main that includes the recipe system.

### US-051-006: Rebase and resolve conflicts

**Description:** Rebase the 14 PRD-50 commits onto updated main after recipe system merge.

**Expected conflict areas:**
- `orchestrator/api/chat.py` — Both branches modify this
- `orchestrator/api/composio.py` — PRD-50 adds webhook dispatch
- `orchestrator/main.py` — Both add router registrations
- `orchestrator/config.py` — Both add env vars
- `frontend/components/chatbot/` — PRD-50 modifies chat components

**Checklist:**
- [ ] `git fetch origin` to get updated main
- [ ] `git rebase origin/main` on `ralph/universal-orchestrator-router`
- [ ] Resolve conflicts in each file (prefer PRD-50 changes for routing, keep recipe changes for execution)
- [ ] Verify after rebase: `import api.main` works
- [ ] Verify after rebase: `npm run build` succeeds
- [ ] Run the dev server and test both chatbot and recipe execution still work

---

## Phase 3: Complete Universal Orchestrator

**Branch:** `ralph/universal-orchestrator-router` (rebased)
**Goal:** Complete the remaining ~20% of PRD-50 and ship the Universal Orchestrator.

### US-051-007: Create Alembic migrations for routing tables

**Description:** PRD-50 defines 3 new tables but the worktree has no migration files.

**Tables needed:**
- `routing_decisions` — Audit log of all routing decisions (id, request_id, envelope_hash, route_type, agent_id, workflow_id, confidence, cached, was_corrected, corrected_agent_id, created_at)
- `routing_rules` — User-defined routing rules (id, workspace_id, source_pattern, intent_keywords, target_agent_id, target_workflow_id, priority, active, created_at)
- `unrouted_events` — Events that couldn't be routed (id, workspace_id, envelope_json, source, reason, created_at, reviewed)

**Checklist:**
- [ ] Create migration `20260202_add_routing_tables.py`
- [ ] Use `IF NOT EXISTS` pattern for idempotent creation
- [ ] Add indexes: `ix_routing_decisions_workspace_created`, `ix_routing_rules_workspace`, `ix_unrouted_events_workspace`
- [ ] Run migration against Railway PostgreSQL
- [ ] Verify tables exist

---

### US-051-008: Wire chatbot to Universal Router

**Description:** Modify `api/chat.py` to route through `UniversalRouter` when no agent is explicitly selected.

**Existing code in worktree:** `ChatbotIngestor` class and `UniversalRouter.route()` already exist. Need to verify integration is correct after rebase.

**Checklist:**
- [ ] Verify `ChatbotIngestor` correctly converts `ChatRequest → RequestEnvelope`
- [ ] Verify `stream_chat` endpoint calls `UniversalRouter.route(envelope)` when `agentId` is not provided
- [ ] Verify user-selected agent (override) bypasses router (Tier 0)
- [ ] Verify conversation history maintained across agent switches
- [ ] Test: send message without selecting agent → verify router picks correct agent
- [ ] Test: send "check my emails" → verify routes to Communication Agent
- [ ] Test: send "help me write code" → verify routes to Coder Agent

---

### US-051-009: Wire webhook dispatch through Universal Router

**Description:** Replace the TODO at `api/composio.py:509` with actual routing logic.

**Existing code in worktree:** `JiraTriggerIngestor` and webhook dispatch logic already exist. Need to verify after rebase.

**Checklist:**
- [ ] Verify webhook endpoint creates `RequestEnvelope` via `JiraTriggerIngestor`
- [ ] Verify `UniversalRouter.route(envelope)` dispatches to correct agent/workflow
- [ ] Verify async dispatch (return 200 immediately, process in background)
- [ ] Verify unroutable events stored in `unrouted_events` table
- [ ] Test: simulate Jira webhook payload → verify routing decision logged
- [ ] Test: create Jira ticket in PILOT → verify webhook fires and agent dispatched

---

### US-051-010: Wire recipe execution as a routing target

**Description:** When the Universal Router routes to a recipe/workflow (via trigger or chatbot), it should use the same execution path we stabilized in Phase 1 — create Workflow from recipe, launch `execute_workflow_with_progress`.

**Checklist:**
- [ ] Add `"recipe"` as a valid `route_type` in `RoutingDecision`
- [ ] When `route_type == "recipe"`: look up recipe by workflow_id, call same logic as `POST /api/workflow-recipes/{recipe_id}/execute`
- [ ] Extract shared execution logic into `execute_recipe(recipe_id, input_data, workspace_id)` function that both the API endpoint and the router can call
- [ ] Test: create routing rule "jira_trigger → Bug Triage recipe"
- [ ] Test: simulate Jira webhook → verify recipe execution starts
- [ ] Verify ExecutionTheater can display the execution (via workflow_id from routing)

---

## Phase 4: Unify Tool-Loading Paths

**Branch:** `ralph/universal-orchestrator-router` (after Phase 3)
**Goal:** Eliminate the duplicate tool-loading code paths. One provider for all execution contexts.

### US-051-011: Create UnifiedToolProvider

**Description:** Today, tools are loaded differently depending on context:
- **Chatbot:** `chatbot/tool_router.py → get_chatbot_tools()` → ToolRegistry → composio_execute + hints
- **Recipe/Workflow:** `agent_factory.py → _build_tool_schemas()` → research/file tools + our Composio bridge patch

Both should use the same provider that: (1) loads base tools for the agent, (2) checks Composio app assignments, (3) adds composio_execute with action hints, (4) resolves workspace_id for tool execution.

**Checklist:**
- [ ] Create `orchestrator/modules/tools/unified_tool_provider.py`
- [ ] Implement `get_tools_for_agent(agent_id, workspace_id, db_session) → List[ToolSchema]` method
- [ ] Method loads: base tools (research, file) + agent-specific skills + Composio tools (if assigned)
- [ ] Method builds hints following the pattern from `_build_composio_hints()` and `chatbot/service.py`
- [ ] Method returns tool schemas in OpenAI function-calling format
- [ ] Implement `get_system_hints(agent_id, prompt) → str` for Composio action hints
- [ ] Add `resolve_workspace_id(agent_id) → Optional[UUID]` helper

---

### US-051-012: Migrate agent_factory.py to use UnifiedToolProvider

**Description:** Replace `_build_tool_schemas()` and the Composio bridge patch in `execute_with_prompt()` with a call to `UnifiedToolProvider.get_tools_for_agent()`.

**Checklist:**
- [ ] Import `UnifiedToolProvider` in agent_factory.py
- [ ] Replace `_build_tool_schemas()` call with `UnifiedToolProvider.get_tools_for_agent()`
- [ ] Remove `_build_composio_hints()` method (moved to UnifiedToolProvider)
- [ ] Remove inline composio_execute schema injection block
- [ ] Verify workspace_id is still passed to tool execution
- [ ] Test: recipe execution still works with Composio tools
- [ ] Test: non-Composio recipe execution still works (research tools only)

---

### US-051-013: Migrate chatbot to use UnifiedToolProvider

**Description:** Replace `get_chatbot_tools()` in `chatbot/tool_router.py` and the Composio hint injection in `chatbot/service.py` with `UnifiedToolProvider`.

**Checklist:**
- [ ] Modify `chatbot/tool_router.py` to call `UnifiedToolProvider.get_tools_for_agent()` instead of `get_chatbot_tools()`
- [ ] Remove Composio hint injection block from `chatbot/service.py` (lines 606-727)
- [ ] Verify chatbot still receives composio_execute tool when agent has Composio apps
- [ ] Verify chatbot still works without Composio apps (basic tools only)
- [ ] Test: chat with Communication Agent → verify Gmail/Slack tools available
- [ ] Test: chat with Coder Agent → verify GitHub tools available
- [ ] Test: chat with Research Agent → verify no Composio tools, only research tools

---

### US-051-014: Remove duplicate tool-loading code

**Description:** Clean up old tool-loading code that's been replaced by UnifiedToolProvider.

**Checklist:**
- [ ] Remove `_build_tool_schemas()` from agent_factory.py (if still present)
- [ ] Remove `get_chatbot_tools()` from chatbot/tool_router.py (if now unused)
- [ ] Remove Composio-specific hint building from chatbot/service.py
- [ ] Verify no other consumers of the removed functions (grep for function names)
- [ ] Run full typecheck
- [ ] Run dev server, test chatbot and recipe execution

---

## Dependency Order

```
Phase 1 (current branch - recipe stabilization):
  US-051-001 (commit bridge)
    → US-051-002 (fix action cache)
      → US-051-003 (e2e test)
  US-051-004 (fix TS errors) — parallel with 001-003
    → US-051-005 (merge to main)

Phase 2 (rebase):
  US-051-006 (rebase universal orchestrator) — after Phase 1 merged

Phase 3 (complete orchestrator):
  US-051-007 (migrations) + US-051-008 (chatbot) + US-051-009 (webhooks) — parallel after rebase
    → US-051-010 (recipe as routing target)

Phase 4 (unify tools):
  US-051-011 (UnifiedToolProvider) — after Phase 3
    → US-051-012 (migrate agent_factory) + US-051-013 (migrate chatbot) — parallel
      → US-051-014 (cleanup)
```

---

## Key Architecture Decisions

### Why not merge PRD-50 now?
- 78 files changed, 15k+ lines of divergence between branches
- Recipe execution on current branch has uncommitted changes to 14 files
- Merging two divergent feature branches simultaneously creates compounding conflicts
- Sequential merge (recipe first, then rebase orchestrator) is cleaner

### Why UnifiedToolProvider?
- Today: chatbot loads tools via ToolRegistry + service.py hints; recipe loads tools via agent_factory.py + our bridge patch
- Both paths do the same thing: check agent → load base tools → check Composio assignments → add composio_execute → build hints
- Single provider eliminates duplication and ensures all execution contexts get the same tools
- Future channels (Slack bot, WhatsApp) automatically get tool loading for free

### Jira Bug Triage (PRD-50 US-008) Correction
The Jira Bug Triage workflow is NOT CodeGraph autonomously patching repos. It's:
1. Jira creates ticket → fires webhook to Automatos
2. Webhook dispatched through Universal Router → matches "jira_trigger" rule → dispatches to Bug Triage Recipe
3. Bug Triage Recipe has pre-defined steps with Claude 4.5 as the agent, assigned GitHub + Jira tools
4. Recipe executes through the same `execute_workflow_with_progress` 9-stage pipeline
5. Agent reads ticket (JIRA_GET_ISSUE), clones repo (GitHub tools), creates branch, applies fix, opens PR (GITHUB_CREATE_PULL_REQUEST), updates Jira (JIRA_UPDATE_ISSUE)

The recipe execution pipeline (Phase 1) is the foundation. The Universal Router (Phase 3) adds the trigger dispatch layer on top.

---

## Files Reference

### Phase 1 Files (Current Branch)
| File | Status | Change |
|------|--------|--------|
| `orchestrator/modules/agents/factory/agent_factory.py` | Uncommitted | Composio bridge: composio_execute injection, _build_composio_hints(), workspace_id passthrough |
| `orchestrator/core/composio/tool_executor.py` | Uncommitted | Action name reconstruction (display name → API identifier) |
| `orchestrator/api/workflow_recipes.py` | Committed | Recipe execute endpoint (creates Workflow + WorkflowExecution + launches pipeline) |
| `frontend/lib/api-client.ts` | Committed | executeRecipe() method |
| `frontend/hooks/use-recipe-api.ts` | Committed | useExecuteRecipe() hook |
| `frontend/components/workflows/recipes-tab.tsx` | Committed | Cook button wiring, onExecuteRecipe prop |
| `frontend/components/workflows/workflow-management.tsx` | Uncommitted | onExecuteRecipe handler, data mapping fixes |

### Phase 3-4 Files (Universal Orchestrator Branch)
| File | Status | Change Needed |
|------|--------|---------------|
| `orchestrator/core/routing/engine.py` | Exists | Verify after rebase |
| `orchestrator/core/routing/cache.py` | Exists | Verify after rebase |
| `orchestrator/core/routing/ingestors/chatbot.py` | Exists | Verify after rebase |
| `orchestrator/core/routing/ingestors/jira_trigger.py` | Exists | Verify after rebase |
| `orchestrator/api/routing.py` | Exists | Verify after rebase |
| `orchestrator/api/chat.py` | Exists (modified) | Resolve conflicts with recipe branch |
| `orchestrator/api/composio.py` | Exists (modified) | Resolve conflicts, verify webhook dispatch |
| `orchestrator/modules/tools/unified_tool_provider.py` | New | Phase 4 — create from scratch |
| `orchestrator/alembic/versions/20260202_add_routing_tables.py` | New | Phase 3 — create migration |

---

## Success Criteria

1. **Phase 1 Complete:** Recipe with Composio tools executes end-to-end. Cook button → agent calls Gmail API → results displayed in ExecutionTheater.
2. **Phase 2 Complete:** Universal Orchestrator branch rebased on main with no conflicts. Both recipe execution and chatbot work.
3. **Phase 3 Complete:** Chatbot auto-routes without agent selection. Jira webhook dispatches to recipe. Routing decisions logged.
4. **Phase 4 Complete:** Single `UnifiedToolProvider` serves all execution contexts. No duplicate tool-loading code. Chatbot and recipe execution both use the same provider.
