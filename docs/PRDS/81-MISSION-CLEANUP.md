# PRD-81: MISSION CLEANUP — Complete the Context & Memory Unification

**Version:** 1.0
**Status:** Ready for Execution
**Priority:** P0
**Author:** Gar Kavanagh + Claude
**Created:** 2026-03-13
**Dependencies:** PRD-79 (Unified Memory — INCOMPLETE), PRD-80 (Unified Context Service — INCOMPLETE)

---

## Why This PRD Exists

PRD-79 was supposed to unify memory across ALL execution paths. It only did chatbot.
PRD-80 was supposed to migrate ALL callers to ContextService. It only fully migrated 3 of 6.

Both PRDs were marked done. Both are half-finished. This PRD finishes the job.

**The root cause is the same both times:** the legacy `_build_agent_system_prompt()` in `agent_factory.py` caches a prompt at activation time. When `execute_with_prompt()` runs, it checks for this cached prompt FIRST (line 768). If found — which is almost always — ContextService is never called. This means:

- No Mem0 memory for task-executing agents
- No Mem0 memory for heartbeat agents
- No Mem0 memory for channel messages
- No Composio section in ContextService (still inline in the legacy function)
- No Plugin section in ContextService (still inline in the legacy function)
- No token budget management for these paths
- 173 lines of dead prompt-building code that should have been deleted

This PRD has **4 phases**, each independently deployable and testable. Each phase has tiny, numbered tasks with exact file paths, line numbers, and acceptance criteria.

---

## Current State (Post PRD-80)

### What ContextService HAS (working)

| Component | Status | File |
|-----------|--------|------|
| ContextService.build_context() | Working | `modules/context/service.py` |
| IdentitySection | Working | `modules/context/sections/identity.py` |
| SkillsSection | Working | `modules/context/sections/skills.py` |
| PlatformActionsSection | Working | `modules/context/sections/platform_actions.py` |
| MemorySection | Working | `modules/context/sections/memory.py` |
| TaskContextSection | Working | `modules/context/sections/task_context.py` |
| RecipeContextSection | Working | `modules/context/sections/recipe_context.py` |
| DatetimeContextSection | Working | `modules/context/sections/datetime_context.py` |
| ConversationSection | Working | `modules/context/sections/conversation.py` |
| CustomSection | Working | `modules/context/sections/custom.py` |
| ToolsSection (load_tools) | Working | `modules/context/sections/tools.py` |
| TokenBudgetManager | Working | `modules/context/budget.py` |
| TokenEstimator | Working | `modules/context/estimator.py` |

### What ContextService is MISSING

| Component | Status | Why It's Missing |
|-----------|--------|-----------------|
| ComposioSection | NOT BUILT | Composio app hints are still inline at `agent_factory.py:1258-1289`. No section file exists. |
| PluginsSection | NOT BUILT | Plugin tier1/tier2 content still inline at `agent_factory.py:1232-1255`. No section file exists. |
| HEARTBEAT_AGENT mode | NOT BUILT | Only one `HEARTBEAT` mode exists (`modes.py:16`). Agent ticks go through factory, not ContextService. |

### Caller Migration Status

| # | Caller | Status | Evidence |
|---|--------|--------|----------|
| 1 | Chatbot (`smart_orchestrator.py`) | MIGRATED | Uses `ContextService(CHATBOT)` at line 194 |
| 2 | Heartbeat Orchestrator (`heartbeat_service.py`) | MIGRATED | Uses `ContextService(HEARTBEAT)` at line 439 |
| 3 | Recipe Executor (`recipe_executor.py`) | MIGRATED | Uses `ContextService(RECIPE)` at line 143 |
| 4 | Agent Factory (`agent_factory.py`) | BYPASSED | `activate_agent()` caches `_build_agent_system_prompt()` at line 657. `execute_with_prompt()` prefers cache at line 768. ContextService only called as fallback at line 772 — almost never reached. |
| 5 | Heartbeat Agent (`heartbeat_service.py`) | NOT MIGRATED | Calls `factory.execute_with_prompt(use_memory=False)` at line 705. Goes through factory cache. |
| 6 | Execution Manager (`execution_manager.py`) | BYPASSED | Calls `factory.execute_with_prompt()` at line 869 without `system_prompt=`. Factory cache takes priority. |
| 7 | Channels (`channels/base.py`) | NOT MIGRATED | Calls `factory.execute_with_prompt()`. No ContextService reference. |

### Memory Injection Status

| Path | Has Mem0? | Has Daily Logs? | Reason |
|------|-----------|-----------------|--------|
| Chatbot | YES | YES | ContextService MemorySection |
| Heartbeat Orch | NO (correct) | NO | HEARTBEAT mode excludes memory section |
| Recipe | NO | NO | RECIPE mode config doesn't include "memory" section |
| Task Execution (factory) | NO | NO | Factory cache bypass — ContextService never called |
| Heartbeat Agent | NO | NO | `use_memory=False`, goes through factory |
| Execution Manager | NO | NO | Delegates to factory, cache bypass |
| Channels | NO | NO | Delegates to factory, cache bypass |

---

## Phase 1: Build Missing Sections (No Breaking Changes)

**Goal:** Create ComposioSection and PluginsSection. Add HEARTBEAT_AGENT mode. Zero changes to existing callers.

**Duration:** 2-3 hours
**Risk:** ZERO — new files only, nothing calls them yet.

---

### Task 1.1: Create `composio.py` section

**What:** Create a new file `orchestrator/modules/context/sections/composio.py` that renders the Composio app descriptions block.

**Where to copy logic from:** `agent_factory.py` lines 1258-1289. That code:
1. Queries `AgentAppAssignment` for active EXTERNAL apps
2. Queries `ComposioAppCache` for app descriptions
3. Builds a markdown block: `## Available External Apps (Composio)` with app names and descriptions

**New file content must:**
1. Subclass `BaseSection` from `modules/context/sections/base.py`
2. Set `name = "composio"`, `priority = 5`, `max_tokens = 1000`
3. Implement `async def render(self, ctx: SectionContext) -> str`
4. Access `ctx.db_session` for database queries
5. Access `ctx.agent` to get the agent ID
6. Return empty string `""` if no Composio apps are assigned
7. Return the markdown block if apps exist
8. Catch all exceptions, log a warning, return `""` on failure

**Acceptance test:**
- Create a test file `orchestrator/tests/test_context/test_sections/test_composio.py`
- Test 1: Agent with no Composio apps → returns `""`
- Test 2: Agent with 2 Composio apps → returns markdown containing both app names
- Test 3: Database error → returns `""`, logs warning

---

### Task 1.2: Create `plugins.py` section

**What:** Create a new file `orchestrator/modules/context/sections/plugins.py` that renders plugin tier1 + tier2 content.

**Where to copy logic from:** `agent_factory.py` lines 1232-1255. That code:
1. Calls `PluginContextService(db).get_assigned_plugins(agent.id)`
2. Filters to non-materialized plugins (ones without `materialized_skill_ids`)
3. Calls `plugin_svc.build_tier1_summary(non_materialized)` → returns a summary string
4. Calls `plugin_svc.build_tier2_content_sync(non_materialized, task_context=task_context)` → returns detailed content

**New file content must:**
1. Subclass `BaseSection`
2. Set `name = "plugins"`, `priority = 5`, `max_tokens = 2000`
3. Implement `async def render(self, ctx: SectionContext) -> str`
4. Import `PluginContextService` from `core.services.plugin_context_service`
5. Get `task_context` from `ctx.task_description` (this maps to the old `task_context` param)
6. Concatenate tier1 and tier2 with `\n\n` between them
7. Return empty string if no plugins assigned or all are materialized
8. Catch all exceptions, log warning, return `""`

**Acceptance test:**
- Test file: `orchestrator/tests/test_context/test_sections/test_plugins.py`
- Test 1: Agent with no plugins → returns `""`
- Test 2: Agent with plugins → returns string containing tier1 and tier2
- Test 3: All plugins are materialized → returns `""`

---

### Task 1.3: Register both sections in SECTION_REGISTRY

**What:** Add ComposioSection and PluginsSection to the registry so ContextService can find them by name.

**File to edit:** `orchestrator/modules/context/sections/__init__.py`

**What to do:**
1. Import `ComposioSection` from `.composio`
2. Import `PluginsSection` from `.plugins`
3. Add both to the `SECTION_REGISTRY` dict with keys `"composio"` and `"plugins"`

**Acceptance test:** In a Python shell or test:
```python
from modules.context.sections import SECTION_REGISTRY
assert "composio" in SECTION_REGISTRY
assert "plugins" in SECTION_REGISTRY
```

---

### Task 1.4: Add HEARTBEAT_AGENT mode to modes.py

**What:** Split the single `HEARTBEAT` mode into `HEARTBEAT_ORCHESTRATOR` and `HEARTBEAT_AGENT`.

**File to edit:** `orchestrator/modules/context/modes.py`

**Changes:**
1. Rename `HEARTBEAT = "heartbeat"` to `HEARTBEAT_ORCHESTRATOR = "heartbeat_orchestrator"`
2. Add `HEARTBEAT_AGENT = "heartbeat_agent"`
3. Rename the `ContextMode.HEARTBEAT` key in `MODE_CONFIGS` to `ContextMode.HEARTBEAT_ORCHESTRATOR`
4. Add a new entry for `ContextMode.HEARTBEAT_AGENT`:

```python
ContextMode.HEARTBEAT_AGENT: ModeConfig(
    sections=[
        "identity", "skills", "composio", "plugins",
        "platform_actions", "task_context", "datetime_context",
    ],
    tool_loading="full",
    personality=False,
    max_tokens=16000,
),
```

Note: HEARTBEAT_AGENT gets `tool_loading="full"` (agents need their tools), `composio` and `plugins` (agents have external integrations), but NO `memory` section (keep context lean for now — can be added later after testing).

5. Update `HEARTBEAT_ORCHESTRATOR` config to stay the same as current `HEARTBEAT`:
```python
ContextMode.HEARTBEAT_ORCHESTRATOR: ModeConfig(
    sections=[
        "identity", "skills", "platform_actions", "task_context",
        "datetime_context",
    ],
    tool_loading="dispatcher_only",
    personality=False,
    max_tokens=8000,
),
```

**Acceptance test:**
```python
from modules.context.modes import ContextMode, MODE_CONFIGS
assert ContextMode.HEARTBEAT_ORCHESTRATOR in MODE_CONFIGS
assert ContextMode.HEARTBEAT_AGENT in MODE_CONFIGS
assert "composio" in MODE_CONFIGS[ContextMode.HEARTBEAT_AGENT].sections
assert MODE_CONFIGS[ContextMode.HEARTBEAT_AGENT].tool_loading == "full"
```

---

### Task 1.5: Add composio + plugins to existing mode configs

**What:** Update CHATBOT, TASK_EXECUTION, RECIPE, and EXECUTION_MANAGER mode configs to include the new sections.

**File to edit:** `orchestrator/modules/context/modes.py`

**Changes to section lists (insert after "skills" in each):**

```python
ContextMode.CHATBOT: ModeConfig(
    sections=[
        "identity", "skills", "composio", "plugins",  # <-- ADD composio, plugins
        "platform_actions", "memory",
        "datetime_context", "conversation",
    ],
    ...
),
ContextMode.TASK_EXECUTION: ModeConfig(
    sections=[
        "identity", "skills", "composio", "plugins",  # <-- ADD composio, plugins
        "platform_actions", "memory",
        "task_context", "datetime_context", "conversation",
    ],
    ...
),
ContextMode.RECIPE: ModeConfig(
    sections=[
        "identity", "skills", "composio", "plugins",  # <-- ADD composio, plugins
        "platform_actions", "memory",                  # <-- ADD memory
        "recipe_context", "datetime_context",
    ],
    ...
),
```

Note: Also adding `"memory"` to RECIPE mode — recipes should have user context.

**Acceptance test:**
```python
from modules.context.modes import ContextMode, MODE_CONFIGS
for mode in [ContextMode.CHATBOT, ContextMode.TASK_EXECUTION, ContextMode.RECIPE]:
    assert "composio" in MODE_CONFIGS[mode].sections
    assert "plugins" in MODE_CONFIGS[mode].sections
assert "memory" in MODE_CONFIGS[ContextMode.RECIPE].sections
```

---

### Task 1.6: Update budget.py DEFAULT_BUDGETS

**What:** Add budget entries for the new modes.

**File to edit:** `orchestrator/modules/context/budget.py`

**What to do:** Find the `DEFAULT_BUDGETS` dict. Add entries:
```python
ContextMode.HEARTBEAT_ORCHESTRATOR: TokenBudget(total=128_000, reserved_for_response=2_048, reserved_for_messages=0),
ContextMode.HEARTBEAT_AGENT: TokenBudget(total=128_000, reserved_for_response=4_096, reserved_for_messages=0),
```

Remove the old `ContextMode.HEARTBEAT` entry if it exists.

**Also:** update any imports of `ContextMode.HEARTBEAT` in budget.py to use the new name.

---

### Task 1.7: Update heartbeat_service.py to use HEARTBEAT_ORCHESTRATOR

**What:** The orchestrator tick currently uses `ContextMode.HEARTBEAT`. Rename to `ContextMode.HEARTBEAT_ORCHESTRATOR`.

**File to edit:** `orchestrator/services/heartbeat_service.py`

**What to do:** Find the line (around 439) that says:
```python
mode=ContextMode.HEARTBEAT,
```
Change it to:
```python
mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
```

**Acceptance test:** Start heartbeat, verify orchestrator tick logs show `mode=heartbeat_orchestrator`.

---

### Phase 1 Done Checklist

- [ ] `modules/context/sections/composio.py` exists and renders Composio app block
- [ ] `modules/context/sections/plugins.py` exists and renders plugin tier1+tier2
- [ ] Both registered in `SECTION_REGISTRY`
- [ ] `ContextMode.HEARTBEAT_ORCHESTRATOR` exists in modes.py
- [ ] `ContextMode.HEARTBEAT_AGENT` exists in modes.py with full tools + composio + plugins
- [ ] `composio` + `plugins` added to CHATBOT, TASK_EXECUTION, RECIPE section lists
- [ ] `memory` added to RECIPE section list
- [ ] `DEFAULT_BUDGETS` updated for new modes
- [ ] Heartbeat orchestrator uses `HEARTBEAT_ORCHESTRATOR` mode
- [ ] All existing tests still pass
- [ ] New section tests pass

**STOP HERE. Deploy. Test. Verify logs. Then continue to Phase 2.**

---

## Phase 2: Kill the Cache — Make ContextService the Only Path

**Goal:** Remove the legacy `_build_agent_system_prompt()` cache so ALL agent execution goes through ContextService.

**Duration:** 3-4 hours
**Risk:** MEDIUM — this changes the prompt for every task-executing agent. Test carefully.

---

### Task 2.1: Remove `system_prompt` from AgentRuntime

**What:** The `AgentRuntime` dataclass (or NamedTuple) has a `system_prompt` field that caches the legacy prompt. Remove it.

**File to find:** Search for `class AgentRuntime` in `orchestrator/modules/agents/` — likely in `agent_factory.py` or a separate models file.

**What to do:**
1. Find the `AgentRuntime` class definition
2. Remove the `system_prompt` field
3. Remove the `skill_tool_schemas` field (skills are now handled by SkillsSection)
4. Search the entire codebase for any code that reads `agent_runtime.system_prompt` or `agent_runtime.skill_tool_schemas`. Fix each reference.

**Known references to fix:**
- `agent_factory.py:673` — `system_prompt=system_prompt` in AgentRuntime constructor → remove
- `agent_factory.py:674` — `skill_tool_schemas=skill_tool_schemas` → remove
- `agent_factory.py:768` — `elif getattr(agent_runtime, "system_prompt", None):` → remove this branch entirely
- `agent_factory.py:770` — `skill_tool_schemas_from_prompt = getattr(agent_runtime, "skill_tool_schemas", [])` → remove

**How to find ALL references (run this grep):**
```bash
grep -rn "system_prompt" orchestrator/modules/agents/factory/agent_factory.py
grep -rn "skill_tool_schemas" orchestrator/modules/agents/factory/agent_factory.py
grep -rn "agent_runtime.system_prompt" orchestrator/
grep -rn "runtime.system_prompt" orchestrator/
```

**Acceptance test:** `AgentRuntime(...)` constructor works without `system_prompt` param. No import errors.

---

### Task 2.2: Make execute_with_prompt() always use ContextService

**What:** Remove the 3-way prompt resolution (`explicit > cached > ContextService`) and replace with 2-way (`explicit > ContextService`).

**File to edit:** `orchestrator/modules/agents/factory/agent_factory.py`

**Current code (lines 762-783):**
```python
# System prompt: explicit > cached > ContextService
if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
elif getattr(agent_runtime, "system_prompt", None):
    messages.append({"role": "system", "content": agent_runtime.system_prompt})
    skill_tool_schemas_from_prompt = getattr(agent_runtime, "skill_tool_schemas", [])
else:
    # --- ContextService path ---
    from modules.context import ContextService, ContextMode
    db_agent = self.db_session.query(Agent).filter_by(id=agent_runtime.agent_id).first()
    if db_agent:
        context_result = await ContextService(self.db_session).build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=db_agent,
            workspace_id=agent_runtime.workspace_id,
            task_description=prompt,
        )
        messages.append({"role": "system", "content": context_result.system_prompt})
```

**Replace with:**
```python
# System prompt: explicit OR ContextService
if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
else:
    from modules.context import ContextService, ContextMode
    db_agent = self.db_session.query(Agent).filter_by(id=agent_runtime.agent_id).first()
    if db_agent:
        context_result = await ContextService(self.db_session).build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=db_agent,
            workspace_id=agent_runtime.workspace_id,
            task_description=prompt,
        )
        messages.append({"role": "system", "content": context_result.system_prompt})
    else:
        # Last resort — should not happen
        messages.append({"role": "system", "content": f"You are agent {agent_runtime.agent_id}."})
```

Also remove `skill_tool_schemas_from_prompt` variable and any code that uses it for tool loading (ContextService handles tools now via `context_result.tools`).

**IMPORTANT:** When `context_result` is populated, use `context_result.tools` and `context_result.tool_choice` instead of calling `get_tools_for_agent()` separately. Find where tools are loaded later in `execute_with_prompt()` and add:
```python
if context_result:
    tools = context_result.tools
    tool_choice = context_result.tool_choice
else:
    # explicit system_prompt path — load tools normally
    tools = get_tools_for_agent(agent_id, self.db_session, agent_runtime.workspace_id)
    tool_choice = "auto"
```

**Acceptance test:**
- Call `execute_with_prompt(agent=123, prompt="test")` WITHOUT `system_prompt=`
- Verify logs show `[ContextService] mode=task_execution`
- Verify prompt contains identity, skills, composio, plugins, platform_actions, memory sections

---

### Task 2.3: Delete `_build_agent_system_prompt()`

**What:** Delete the entire method from agent_factory.py.

**File to edit:** `orchestrator/modules/agents/factory/agent_factory.py`

**What to delete:** Lines 1143-1319 (the entire `_build_agent_system_prompt` method). That's 176 lines.

**Before deleting, verify no other code calls it:**
```bash
grep -rn "_build_agent_system_prompt" orchestrator/
```

Expected matches:
- `agent_factory.py` — the definition and calls in `activate_agent()` + `refresh_agent_prompt()`
- Maybe `execution_manager.py` — if it was calling it directly (check)

ALL calls should have been removed by Task 2.1 and 2.2.

**Acceptance test:** `grep -rn "_build_agent_system_prompt" orchestrator/` returns zero results.

---

### Task 2.4: Delete `refresh_agent_prompt()`

**What:** This method (lines 688-705) only exists to rebuild the cached system prompt. Since there's no more cache, it's dead code.

**File to edit:** `orchestrator/modules/agents/factory/agent_factory.py`

**Before deleting, verify no callers:**
```bash
grep -rn "refresh_agent_prompt" orchestrator/
```

Delete all callers too (they're now no-ops).

**Acceptance test:** `grep -rn "refresh_agent_prompt" orchestrator/` returns zero results.

---

### Task 2.5: Update `activate_agent()` to NOT build a prompt

**What:** `activate_agent()` currently calls `_build_agent_system_prompt()` at line 657 and stores the result. Since we deleted that method, we need to remove this call.

**File to edit:** `orchestrator/modules/agents/factory/agent_factory.py`

**What to do:**
1. Remove lines 657-660 (the `_build_agent_system_prompt` call)
2. Remove `system_prompt=system_prompt` from the AgentRuntime constructor (already done in Task 2.1)
3. Remove `skill_tool_schemas=skill_tool_schemas` from the constructor (already done in Task 2.1)
4. The `activate_agent()` method should now only: create LLM manager, create AgentRuntime (without prompt), store in `self.active_agents`, set status to ACTIVE

**Acceptance test:** `activate_agent(agent_id)` succeeds. The returned `AgentRuntime` has no `system_prompt` attribute.

---

### Task 2.6: Migrate heartbeat agent tick to ContextService

**What:** `_agent_tick()` at line 705 calls `factory.execute_with_prompt(use_memory=False)`. Change it to use ContextService directly with the new HEARTBEAT_AGENT mode.

**File to edit:** `orchestrator/services/heartbeat_service.py`

**Current code (around lines 700-710):**
```python
factory = AgentFactory(db_session=db)
exec_result = await factory.execute_with_prompt(
    agent=agent_id,
    prompt=prompt,
    context={"source": "heartbeat", "workspace_id": workspace_id},
    use_memory=False,
)
```

**Two options — pick the simpler one:**

**Option A (recommended):** Keep using factory but DON'T pass `use_memory=False`. Since factory now uses ContextService (after Task 2.2), and HEARTBEAT_AGENT mode doesn't include "memory" in its sections, memory will naturally be excluded. But we need to tell the factory to use HEARTBEAT_AGENT mode instead of TASK_EXECUTION.

Add a `context_mode` parameter to `execute_with_prompt()`:
```python
exec_result = await factory.execute_with_prompt(
    agent=agent_id,
    prompt=prompt,
    context={"source": "heartbeat", "workspace_id": workspace_id},
    context_mode=ContextMode.HEARTBEAT_AGENT,
)
```

And in `execute_with_prompt()`, use this mode instead of hardcoded `ContextMode.TASK_EXECUTION`:
```python
mode = context_mode or ContextMode.TASK_EXECUTION
context_result = await ContextService(self.db_session).build_context(
    mode=mode,
    ...
)
```

**Option B:** Call ContextService directly in `_agent_tick()` and pass the result to factory. More invasive.

**Go with Option A.**

**What to change in `execute_with_prompt()` signature:**
Add parameter: `context_mode: Optional[ContextMode] = None`

**What to change in `execute_with_prompt()` body:**
Where it calls `ContextService(...).build_context(mode=ContextMode.TASK_EXECUTION, ...)`, change to:
```python
mode = context_mode or ContextMode.TASK_EXECUTION
context_result = await ContextService(self.db_session).build_context(mode=mode, ...)
```

**What to change in `heartbeat_service.py`:**
```python
from modules.context.modes import ContextMode

exec_result = await factory.execute_with_prompt(
    agent=agent_id,
    prompt=prompt,
    context={"source": "heartbeat", "workspace_id": workspace_id},
    context_mode=ContextMode.HEARTBEAT_AGENT,
)
```

Remove `use_memory=False` — memory exclusion is now handled by the mode config.

**Acceptance test:**
- Trigger a heartbeat agent tick
- Verify logs show `[ContextService] mode=heartbeat_agent`
- Verify prompt contains skills, composio, platform_actions
- Verify prompt does NOT contain memory section

---

### Phase 2 Done Checklist

- [ ] `AgentRuntime` has no `system_prompt` field
- [ ] `AgentRuntime` has no `skill_tool_schemas` field
- [ ] `_build_agent_system_prompt()` is deleted (0 results from grep)
- [ ] `refresh_agent_prompt()` is deleted (0 results from grep)
- [ ] `activate_agent()` does not build a prompt
- [ ] `execute_with_prompt()` has 2-way resolution: `explicit > ContextService`
- [ ] `execute_with_prompt()` accepts `context_mode` parameter
- [ ] `execute_with_prompt()` uses `context_result.tools` when ContextService provides them
- [ ] Heartbeat agent tick uses `HEARTBEAT_AGENT` mode
- [ ] Heartbeat agent tick does NOT pass `use_memory=False`
- [ ] All existing tests still pass

**STOP HERE. Deploy. Test these scenarios:**
1. Send a chat message → chatbot responds (ContextService CHATBOT)
2. Trigger a heartbeat orchestrator tick → orchestrator runs (ContextService HEARTBEAT_ORCHESTRATOR)
3. Trigger a heartbeat agent tick → agent runs with tools (ContextService HEARTBEAT_AGENT)
4. Execute a recipe → recipe step runs (ContextService RECIPE)
5. Execute a complex task (execution_manager path) → subtask runs (ContextService TASK_EXECUTION)
6. Send a message through a channel adapter → agent responds (ContextService TASK_EXECUTION)

**Verify in ALL 6 scenarios that logs show `[ContextService] mode=...` with correct mode.**

**Then continue to Phase 3.**

---

## Phase 3: Memory Gap Fixes

**Goal:** Ensure every path that SHOULD have memory actually gets it.

**Duration:** 1-2 hours
**Risk:** LOW — MemorySection already works for chatbot. We're just enabling it for more modes.

---

### Task 3.1: Verify TASK_EXECUTION mode includes memory

**What:** Check that `ContextMode.TASK_EXECUTION` in `modes.py` has `"memory"` in its sections list.

**File to check:** `orchestrator/modules/context/modes.py`

**Expected:** `"memory"` should already be in the list (it was added in PRD-80). Verify it's there. If not, add it after `"platform_actions"`.

**Acceptance test:**
```python
assert "memory" in MODE_CONFIGS[ContextMode.TASK_EXECUTION].sections
```

---

### Task 3.2: Verify RECIPE mode includes memory

**What:** After Task 1.5, RECIPE mode should have `"memory"`. Verify.

**Acceptance test:**
```python
assert "memory" in MODE_CONFIGS[ContextMode.RECIPE].sections
```

---

### Task 3.3: Test memory injection for task execution

**What:** Execute a task through AgentFactory (which now uses ContextService after Phase 2) and verify the system prompt contains memory.

**How to test:**
1. Ensure there are memories stored in Mem0 for the workspace
2. Call `execute_with_prompt(agent=agent_id, prompt="What do you know about me?")`
3. Check logs for `[ContextService] mode=task_execution sections=[..., memory, ...]`
4. If "memory" shows up in `sections_included`, it's working
5. If "memory" shows up in `sections_trimmed`, the budget is too tight (check budget.py)
6. If "memory" doesn't appear at all, the mode config is wrong

---

### Task 3.4: Test memory injection for recipes

**What:** Same as 3.3 but for recipe execution.

**How to test:**
1. Execute a recipe step
2. Check logs for `[ContextService] mode=recipe sections=[..., memory, ...]`

**Also check:** `recipe_executor.py` lines 206-217 have manual Mem0 injection for step 1. This is now redundant if MemorySection is in the RECIPE mode. Either:
- Remove the manual injection (recommended — let MemorySection handle it)
- Or keep it as recipe-specific context (if it adds recipe-scoped memories that MemorySection doesn't)

Read the code at lines 206-217 to determine which. If it calls the same `SmartMemoryManager.retrieve_memories()` that MemorySection uses, remove it.

---

### Task 3.5: Decide on HEARTBEAT_AGENT memory

**What:** Currently HEARTBEAT_AGENT mode does NOT include memory (Task 1.4 excluded it to keep context lean).

**Decision needed from product owner:**
- If heartbeat agents should learn from past runs → add `"memory"` to HEARTBEAT_AGENT sections
- If heartbeat should stay stateless → leave as-is

**For now:** Leave as-is. Document this as a conscious decision, not an oversight. Add a comment in modes.py:
```python
# NOTE: memory intentionally excluded from HEARTBEAT_AGENT to keep context lean.
# Add "memory" here if agents need cross-run learning. See PRD-81 Task 3.5.
```

---

### Phase 3 Done Checklist

- [ ] TASK_EXECUTION mode has "memory" in sections
- [ ] RECIPE mode has "memory" in sections
- [ ] Task execution logs show memory section included
- [ ] Recipe execution logs show memory section included
- [ ] HEARTBEAT_AGENT has a comment explaining why memory is excluded
- [ ] Redundant manual memory injection in recipe_executor.py is removed (if applicable)

**STOP HERE. Deploy. Test. Then continue to Phase 4.**

---

## Phase 4: Dead Code Cleanup + Dependency Instructions

**Goal:** Remove all legacy prompt-building code that was superseded by ContextService. Clean up leftover instructions that lived in the deleted method.

**Duration:** 1-2 hours
**Risk:** LOW — only deleting code confirmed unreachable.

---

### Task 4.1: Handle "Dependency Context" instructions

**What:** The deleted `_build_agent_system_prompt()` had boilerplate instructions at lines 1291-1301:

```
## IMPORTANT: Working with Context and Dependencies
When you receive '## DEPENDENCY CONTEXT' at the beginning of your task:
1. This contains outputs from previous tasks that you need to use
...
```

These instructions are useful but were NOT extracted into a ContextService section. Two options:

**Option A (recommended):** Add a `DependencyInstructionsSection` or fold them into `TaskContextSection`:
- Edit `modules/context/sections/task_context.py`
- After rendering the task description, append the dependency context instructions
- These are static text, no DB queries needed

**Option B:** Add them to the agent's skill SKILL.md files instead (more appropriate if only some agents need them).

**Go with Option A** — add to TaskContextSection since ALL task-executing agents benefit from knowing how to handle dependency context.

---

### Task 4.2: Handle "Skill Tool Usage" instructions

**What:** The deleted method also had instructions at lines 1303-1309 about using skill tools:

```
## IMPORTANT: Using Your Skill Tools
You have access to: {tool_names}
When your task requires capabilities provided by these tools, you MUST use them via function calling.
```

**Where to put this:** In `SkillsSection`. After rendering the skill content, if skill tool schemas were extracted, append the usage instructions.

**File to edit:** `modules/context/sections/skills.py`

After the skill content rendering loop, if any skill tools were found, append:
```python
if skill_tool_names:
    parts.append(
        "\n## Using Your Skill Tools\n"
        f"You have access to: {', '.join(skill_tool_names)}\n"
        "When your task requires capabilities provided by these tools, "
        "you MUST use them via function calling. "
        "Analyze your task, check if any tools match, and CALL them."
    )
```

---

### Task 4.3: Handle "Response Formatting" instructions

**What:** The deleted method had response formatting rules at lines 1311-1316:

```
## Response Formatting Rules
When you receive API/tool results:
- Synthesize data into clear, human-friendly prose — do NOT dump raw JSON
- NEVER use code blocks or inline code backticks
- Use bullet points or short paragraphs for a non-technical reader
```

**Where to put this:** This is generic enough to go in `IdentitySection` or a new `FormattingSection`. Since it applies to all agent-facing modes, add it to the identity section after the persona block.

**File to edit:** `modules/context/sections/identity.py`

Add after the persona rendering:
```python
# Response formatting guidance (from legacy _build_agent_system_prompt)
parts.append(
    "\n## Response Formatting\n"
    "When you receive API/tool results:\n"
    "- Synthesize data into clear, human-friendly prose — do NOT dump raw JSON\n"
    "- Use bullet points or short paragraphs for a non-technical reader"
)
```

**Note:** Removed the "NEVER use code blocks" rule — that's too restrictive for agents working with code.

---

### Task 4.4: Verify no orphaned imports

**What:** After deleting `_build_agent_system_prompt()`, some imports in agent_factory.py may be orphaned.

**File to check:** `orchestrator/modules/agents/factory/agent_factory.py`

**What to do:**
1. Check if these imports are still used elsewhere in the file:
   - `from core.services.plugin_context_service import PluginContextService`
   - `from core.models.composio_cache import AgentAppAssignment, ComposioAppCache, ComposioActionCache`
   - `from modules.agents.services.skill_loader import get_skill_loader`
2. If ONLY used by `_build_agent_system_prompt()` → delete the import
3. If used elsewhere → keep

**How to verify:** For each import, grep the file (excluding the import line itself) for the imported name.

---

### Task 4.5: Run final grep audit

**What:** Verify all legacy prompt-building patterns are gone.

Run these greps. Each should return ZERO results from caller files (OK to match in section files or tests):

```bash
# Should only match in modules/context/sections/*, NOT in agent_factory, smart_orchestrator, heartbeat_service
grep -rn "get_happy_system_prompt" orchestrator/ --include="*.py" | grep -v "modules/context/" | grep -v "test_" | grep -v "__pycache__"

# Should return ZERO results
grep -rn "_build_agent_system_prompt" orchestrator/ --include="*.py" | grep -v "__pycache__"

# Should return ZERO results
grep -rn "refresh_agent_prompt" orchestrator/ --include="*.py" | grep -v "__pycache__"

# Should return ZERO results
grep -rn "professional_system_prompt" orchestrator/ --include="*.py" | grep -v "__pycache__"

# Should only match in modules/context/sections/platform_actions.py and action_registry.py
grep -rn "build_prompt_summary" orchestrator/ --include="*.py" | grep -v "modules/context/" | grep -v "action_registry" | grep -v "test_" | grep -v "__pycache__"
```

**Acceptance test:** All greps above return zero results from non-ContextService files.

---

### Task 4.6: Update tests

**What:** Existing tests may reference deleted methods or old prompt paths.

```bash
grep -rn "_build_agent_system_prompt\|refresh_agent_prompt\|system_prompt=system_prompt" orchestrator/tests/ --include="*.py"
```

Fix any test that:
1. Calls `_build_agent_system_prompt()` directly → rewrite to use ContextService
2. Mocks `agent_runtime.system_prompt` → remove the mock
3. Tests `refresh_agent_prompt()` → delete the test

---

### Phase 4 Done Checklist

- [ ] Dependency context instructions moved to TaskContextSection
- [ ] Skill tool usage instructions moved to SkillsSection
- [ ] Response formatting instructions moved to IdentitySection
- [ ] Orphaned imports removed from agent_factory.py
- [ ] All 5 grep audits pass (zero results from non-ContextService files)
- [ ] All tests updated and passing

---

## Phase 5: Remaining Audit Findings (Tech Debt)

**Goal:** Address every remaining finding from the March 2026 system audit that Phases 1-4 didn't cover.

**Duration:** 2-3 hours
**Risk:** LOW — cleanup, consolidation, no new features.

---

### Task 5.1: F3 — Decide on personality for non-chatbot paths

**The finding:** Personality (workspace tone/style settings) only applies to chatbot. All other agent paths produce tonally inconsistent responses.

**Current state in modes.py:**
- `CHATBOT`: `personality=True`
- `HEARTBEAT_ORCHESTRATOR`: `personality=False` (was partial before PRD-80)
- Everything else: `personality=False`

**Decision needed:** Should task-executing agents match the workspace's tone?

**Option A (recommended):** Add `personality=True` to `HEARTBEAT_ORCHESTRATOR` only. Orchestrator represents the platform to the user. Task agents should be professional/neutral regardless of workspace personality settings.

**Option B:** Add `personality=True` to ALL agent-facing modes. Risk: personality template may be too chatbot-specific for task execution.

**What to do:**
1. Read `modules/context/sections/identity.py` — find the personality rendering logic
2. Check if `AutomatosPersonality.get_base_system_prompt()` produces output suitable for task agents or only chatbot
3. If chatbot-specific → go with Option A
4. If generic enough → go with Option B
5. Update `modes.py` accordingly
6. Add a comment on each mode explaining the personality decision

**File to edit:** `orchestrator/modules/context/modes.py`

---

### Task 5.2: D2 — Consolidate tool loading aliases

**The finding:** 3 entry points all resolve to `get_tools_for_agent()`:
- `get_tools_for_agent()` in `tool_router.py` — primary
- `get_agent_tools()` in `tool_router.py` — convenience wrapper
- `get_chat_tools()` re-exported via `consumers/chatbot/tool_router.py`

**What to do:**
1. `grep -rn "get_agent_tools\|get_chat_tools" orchestrator/ --include="*.py"` — find all callers
2. If callers exist outside of ContextService → replace with `get_tools_for_agent()` directly
3. If the only caller is ContextService → delete the aliases
4. If `consumers/chatbot/tool_router.py` only re-exports and adds nothing → delete the file

**Files to edit:**
- `modules/tools/tool_router.py` — remove `get_agent_tools()` alias if unused
- `consumers/chatbot/tool_router.py` — remove `get_chat_tools()` re-export if unused

**Acceptance test:** `grep -rn "get_agent_tools\|get_chat_tools" orchestrator/` returns zero results outside of ContextService.

---

### Task 5.3: R1 — Document SmartToolRouter future plan

**The finding:** `consumers/chatbot/smart_tool_router.py` (442 lines) is used only by ContextService for `ToolLoadingStrategy.FILTERED`. It works but is a large standalone module for one strategy.

**What to do (NOT delete — document):**
1. Add a comment at the top of `smart_tool_router.py`:
```python
# NOTE: This module is consumed by ContextService (ToolLoadingStrategy.FILTERED).
# It is the only caller. Future work: absorb filtering logic into
# modules/context/sections/tools.py and delete this file.
# See PRD-81 Task 5.3 and system audit R1 finding.
```

2. Add to MIGRATION-ROADMAP.md Phase 4 section.

**No code changes — documentation only.**

---

### Task 5.4: R2 — Remove `get_platform_skill()` from personality.py

**The finding:** `AutomatosPersonality.get_platform_skill()` returns a hardcoded platform skill block. It was used by the chatbot's old prompt path.

**What to do:**
1. `grep -rn "get_platform_skill" orchestrator/` — find all callers
2. If zero callers (expected after PRD-80 migration) → delete the method
3. If still called → trace the caller and determine if it should use a ContextService section instead

**File to edit:** `orchestrator/consumers/chatbot/personality.py`

**Acceptance test:** `grep -rn "get_platform_skill" orchestrator/` returns zero results.

---

### Task 5.5: Daily logs for non-chatbot paths

**The finding:** Daily logs are rendered inside `MemorySection`. If a mode doesn't include "memory" in its sections, it gets no daily logs either.

**After Phase 3:** TASK_EXECUTION and RECIPE have "memory" (which includes daily logs). HEARTBEAT_AGENT does not.

**Decision:** This is acceptable. Heartbeat agents don't need daily logs — they have their own scheduled context. Document this.

**What to do:** Add a comment in `modes.py` on HEARTBEAT_AGENT:
```python
# NOTE: No "memory" section — heartbeat agents are stateless by design.
# This also means no daily logs. If agents need cross-run awareness,
# add "memory" here. See PRD-81 Task 5.5.
```

---

### Task 5.6: Gemini finding — Agent resolution scatter

**The finding:** `db.query(Agent)` appears in many files with different join patterns (some load skills, some don't, some load composio apps).

**What to do:**
1. `grep -rn "db.query(Agent)" orchestrator/ --include="*.py" | wc -l` — count occurrences
2. `grep -rn "\.query(Agent)" orchestrator/ --include="*.py"` — list all
3. Identify the common patterns:
   - Agent with skills: `db.query(Agent).options(joinedload(Agent.skills)).filter_by(id=id)`
   - Agent with persona: `db.query(Agent).options(joinedload(Agent.persona))`
   - Agent bare: `db.query(Agent).filter_by(id=id).first()`
4. Create a utility function in `core/models/` or `modules/agents/`:
```python
def get_agent_with_context(db: Session, agent_id: int) -> Agent:
    """Load an agent with all associations needed for ContextService."""
    return (
        db.query(Agent)
        .options(
            joinedload(Agent.skills),
            joinedload(Agent.persona),
        )
        .filter_by(id=agent_id)
        .first()
    )
```
5. Replace scattered `db.query(Agent)` calls in ContextService paths with this utility
6. Leave non-ContextService paths alone (e.g., admin endpoints that only need the name)

**File to create:** `orchestrator/modules/agents/queries.py` (or add to existing `agent_registry.py`)

**Acceptance test:** ContextService paths all use `get_agent_with_context()`. Scattered bare queries still exist in admin/API endpoints (that's fine).

---

### Task 5.7: Verify `_inject_composio_hints()` is still needed

**The finding:** `agent_factory.py` has `_inject_composio_hints()` (around line 1100) which injects Composio action hints as system messages. After Phase 1 creates ComposioSection, this might be redundant.

**What to do:**
1. Read `_inject_composio_hints()` in `agent_factory.py`
2. Compare what it injects vs what ComposioSection renders
3. If ComposioSection covers the same info → delete `_inject_composio_hints()`
4. If `_inject_composio_hints()` adds action-level detail (specific action names, parameter hints) that ComposioSection doesn't → either:
   a. Expand ComposioSection to include this detail, OR
   b. Keep `_inject_composio_hints()` but note it as tech debt

**Key difference to check:** ComposioSection (from Task 1.1) renders app-level descriptions. `_inject_composio_hints()` may render action-level hints (specific actions like `GMAIL_SEND_EMAIL`). If so, both are needed until ComposioSection is expanded.

---

### Task 5.8: Verify inline datetime in heartbeat agent prompt

**The finding (D4):** Heartbeat agent tick has inline datetime in the user prompt at `heartbeat_service.py:690`:
```python
f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
```

**After Phase 2:** HEARTBEAT_AGENT mode includes `datetime_context` in its sections, so DatetimeContextSection renders datetime in the system prompt.

**What to do:** The inline datetime in the user prompt is now redundant. Remove the `Time:` line from the heartbeat prompt string at line 690.

**File to edit:** `orchestrator/services/heartbeat_service.py`

**Acceptance test:** `grep -n "strftime" orchestrator/services/heartbeat_service.py` — should only appear in the orchestrator tick path (if at all), not in `_agent_tick()`.

---

### Phase 5 Done Checklist

- [ ] Personality decision made and documented in modes.py
- [ ] Tool loading aliases consolidated or documented
- [ ] SmartToolRouter has future-plan comment
- [ ] `get_platform_skill()` deleted from personality.py (if unused)
- [ ] Daily logs exclusion from HEARTBEAT_AGENT documented
- [ ] `get_agent_with_context()` utility created
- [ ] `_inject_composio_hints()` evaluated — kept or deleted
- [ ] Inline datetime removed from heartbeat agent prompt

---

## Final Verification: The 9-Path Audit

After ALL phases are deployed, run this checklist against production logs:

| # | Path | Expected Mode | Expected Sections | Memory? | Check |
|---|------|--------------|-------------------|---------|-------|
| 1 | Chat message | `chatbot` | identity, skills, composio, plugins, platform_actions, memory, datetime, conversation | YES | [ ] |
| 2 | Heartbeat orchestrator tick | `heartbeat_orchestrator` | identity, skills, platform_actions, task_context, datetime | NO | [ ] |
| 3 | Heartbeat agent tick | `heartbeat_agent` | identity, skills, composio, plugins, platform_actions, task_context, datetime | NO (by design) | [ ] |
| 4 | Task execution (factory) | `task_execution` | identity, skills, composio, plugins, platform_actions, memory, task_context, datetime, conversation | YES | [ ] |
| 5 | Recipe step | `recipe` | identity, skills, composio, plugins, platform_actions, memory, recipe_context, datetime | YES | [ ] |
| 6 | Execution manager subtask | `task_execution` | identity, skills, composio, plugins, platform_actions, memory, task_context, datetime, conversation | YES | [ ] |
| 7 | Channel message | `task_execution` | identity, skills, composio, plugins, platform_actions, memory, task_context, datetime, conversation | YES | [ ] |
| 8 | Router classification | `router` | identity, datetime | NO | [ ] |
| 9 | NL2SQL | `nl2sql` | identity, datetime | NO | [ ] |

**Every row must show `[ContextService] mode=...` in the logs.** If any row shows a different prompt path, something was missed.

---

## Success Criteria

| Metric | Before PRD-81 | After PRD-81 |
|--------|-------------|-------------|
| Paths using ContextService | 3/6 agent-facing (50%) | 6/6 (100%) |
| Paths with Mem0 memory | 1/6 (chatbot only) | 4/6 (chatbot, task, recipe, exec manager) |
| Paths with Composio context | 0/6 via ContextService | 5/6 (all except heartbeat orch) |
| Paths with Plugin context | 0/6 via ContextService | 5/6 (all except heartbeat orch) |
| Paths with daily logs | 1/6 (chatbot only) | 4/6 (via MemorySection) |
| Paths with datetime | 3/6 (inconsistent) | 6/6 (all via DatetimeContextSection) |
| Paths with token budget | 0/6 | 6/6 (all via TokenBudgetManager) |
| Platform action catalog coverage | 2/6 | 6/6 |
| Audit findings addressed | 0/16 | 16/16 |
| Lines of dead code deleted | 0 | ~250 (`_build_agent_system_prompt` + aliases + `get_platform_skill` + orphans) |
| `_build_agent_system_prompt()` exists | Yes (173 lines) | No |
| `agent_runtime.system_prompt` cache | Active, bypassing ContextService | Deleted |
| Tool loading aliases | 3 (`get_tools_for_agent`, `get_agent_tools`, `get_chat_tools`) | 1 |
| Agent resolution pattern | Scattered `db.query(Agent)` | `get_agent_with_context()` utility |

---

## Appendix: File Change Summary

### New Files (Phase 1)
| File | Lines (est.) |
|------|-------------|
| `modules/context/sections/composio.py` | ~60 |
| `modules/context/sections/plugins.py` | ~70 |
| `tests/test_context/test_sections/test_composio.py` | ~50 |
| `tests/test_context/test_sections/test_plugins.py` | ~50 |

### Modified Files
| File | Phase | Change |
|------|-------|--------|
| `modules/context/sections/__init__.py` | 1 | Register ComposioSection + PluginsSection |
| `modules/context/modes.py` | 1 | Split HEARTBEAT, add composio/plugins/memory to modes |
| `modules/context/budget.py` | 1 | Add HEARTBEAT_ORCHESTRATOR + HEARTBEAT_AGENT budgets |
| `services/heartbeat_service.py` | 1+2 | HEARTBEAT_ORCHESTRATOR mode, agent tick uses HEARTBEAT_AGENT |
| `modules/agents/factory/agent_factory.py` | 2 | Remove cache, delete `_build_agent_system_prompt`, add `context_mode` param |
| `modules/context/sections/task_context.py` | 4 | Add dependency context instructions |
| `modules/context/sections/skills.py` | 4 | Add skill tool usage instructions |
| `modules/context/sections/identity.py` | 4 | Add response formatting instructions |

### Modified Files (Phase 5)
| File | Phase | Change |
|------|-------|--------|
| `modules/context/modes.py` | 5 | Document personality decisions, daily logs exclusion |
| `modules/tools/tool_router.py` | 5 | Remove `get_agent_tools()` alias |
| `consumers/chatbot/tool_router.py` | 5 | Remove `get_chat_tools()` re-export (or delete file) |
| `consumers/chatbot/smart_tool_router.py` | 5 | Add future-plan comment |
| `consumers/chatbot/personality.py` | 5 | Delete `get_platform_skill()` |
| `services/heartbeat_service.py` | 5 | Remove inline datetime from agent tick |

### New Files (Phase 5)
| File | Lines (est.) |
|------|-------------|
| `modules/agents/queries.py` | ~30 (agent resolution utility) |

### Deleted Code
| Code | File | Lines Deleted | Phase |
|------|------|--------------|-------|
| `_build_agent_system_prompt()` | agent_factory.py | ~176 | 2 |
| `refresh_agent_prompt()` | agent_factory.py | ~18 | 2 |
| `system_prompt` field on AgentRuntime | agent_factory.py | ~2 | 2 |
| `skill_tool_schemas` field on AgentRuntime | agent_factory.py | ~2 | 2 |
| Cache check branch in `execute_with_prompt()` | agent_factory.py | ~3 | 2 |
| `get_agent_tools()` alias | tool_router.py | ~5 | 5 |
| `get_chat_tools()` re-export | chatbot/tool_router.py | ~5-file | 5 |
| `get_platform_skill()` | personality.py | ~15 | 5 |
| `_inject_composio_hints()` (if redundant) | agent_factory.py | ~40 | 5 |
| Inline datetime in agent tick | heartbeat_service.py | ~1 | 5 |
| **Total** | | **~270 lines** | |

---

## Audit Finding Cross-Reference

Every finding from SYSTEM-AUDIT-2026-03.md mapped to a PRD-81 task:

| Finding | ID | Severity | PRD-81 Task | Phase |
|---------|-----|----------|------------|-------|
| Platform actions missing 4/6 paths | F1 | HIGH | Phase 2 (cache kill → ContextService for all) | 2 |
| Memory inconsistency across paths | F2 | HIGH | Phase 2 + Phase 3 (verify memory in modes) | 2+3 |
| Personality only on chatbot | F3 | MEDIUM | Task 5.1 (decide + document) | 5 |
| No token budgets anywhere | F4 | HIGH | Phase 2 (cache kill → TokenBudgetManager) | 2 |
| `_build_agent_system_prompt` 3 callers | D1 | HIGH | Task 2.3 (delete method) | 2 |
| Tool loading 3 aliases | D2 | LOW | Task 5.2 (consolidate) | 5 |
| Platform summary in 2 places | D3 | MEDIUM | Phase 2 (only PlatformActionsSection) | 2 |
| Datetime 3 different ways | D4 | LOW | Phase 2 + Task 5.8 (only DatetimeSection) | 2+5 |
| SmartToolRouter 442 lines | R1 | LOW | Task 5.3 (document future plan) | 5 |
| `get_platform_skill()` orphan | R2 | LOW | Task 5.4 (delete) | 5 |
| `professional_system_prompt` | R3 | HIGH | Already deleted (PRD-80) | — |
| Daily logs missing non-chatbot | 4.4 | MEDIUM | Phase 3 (memory in TASK_EXECUTION, RECIPE) | 3 |
| Datetime missing 3 paths | 4.4 | LOW | Phase 2 (all paths get DatetimeSection) | 2 |
| Session context (L1) non-chatbot | 4.4 | LOW | MemorySection handles if chat_id present | — |
| Agent resolution scatter | Gemini | LOW | Task 5.6 (utility function) | 5 |
| Composio hints redundancy | Audit | LOW | Task 5.7 (evaluate + decide) | 5 |
