# PRD-81: MISSION CLEANUP — Implementation Plan

> **Scope**: Backend (orchestrator) | **Risk**: Medium (cache removal changes all agent execution paths) | **Branch**: `ralph/81-mission-cleanup`

## Summary

Complete the half-finished PRD-79 and PRD-80 unification. Kill the legacy `_build_agent_system_prompt()` cache that bypasses ContextService, add ComposioSection + PluginsSection, enable memory for task/recipe paths, and clean up ~270 lines of dead prompt-building code. After this, ALL 6 agent-facing code paths use ContextService exclusively.

## Reference

- **PRD**: `docs/PRDS/81-MISSION-CLEANUP.md`
- **System Audit**: `docs/audits/SYSTEM-AUDIT-2026-03.md` (findings F1-F4, D1-D4, R1-R3)
- **ContextService**: `orchestrator/modules/context/service.py`
- **Context sections**: `orchestrator/modules/context/sections/` (identity, skills, platform_actions, memory, tools, etc.)
- **Context modes**: `orchestrator/modules/context/modes.py` (ContextMode enum + MODE_CONFIGS)
- **Context budget**: `orchestrator/modules/context/budget.py` (TokenBudgetManager + DEFAULT_BUDGETS)
- **Section registry**: `orchestrator/modules/context/sections/__init__.py` (SECTION_REGISTRY)
- **Agent Factory**: `orchestrator/modules/agents/factory/agent_factory.py` (_build_agent_system_prompt, execute_with_prompt, activate_agent)
- **Heartbeat Service**: `orchestrator/services/heartbeat_service.py` (_orchestrator_tick_llm, _agent_tick)
- **Recipe Executor**: `orchestrator/api/recipe_executor.py` (_execute_step)
- **Execution Manager**: `orchestrator/modules/agents/execution/execution_manager.py` (_execute_subtask, professional_system_prompt)
- **Tool Router**: `orchestrator/modules/tools/tool_router.py` (get_tools_for_agent, get_agent_tools)
- **Plugin Context**: `orchestrator/core/services/plugin_context_service.py`
- **Composio Models**: `orchestrator/core/models/composio_cache.py` (AgentAppAssignment, ComposioAppCache, ComposioActionCache)
- **Personality**: `orchestrator/consumers/chatbot/personality.py` (AutomatosPersonality, get_platform_skill)
- **Smart Tool Router**: `orchestrator/consumers/chatbot/smart_tool_router.py`
- **Chatbot Tool Router**: `orchestrator/consumers/chatbot/tool_router.py` (get_chat_tools re-export)
- **Channel Adapter**: `orchestrator/channels/base.py` (delegates to factory.execute_with_prompt)
- **Config**: `orchestrator/config.py`

## Tasks

### Phase 1: Add Missing Sections + Mode Configs

- [x] **US-001: Create ComposioSection and PluginsSection** — Port Composio app rendering from agent_factory.py lines 1247-1278 into modules/context/sections/composio.py. Port plugin tier1+tier2 rendering from lines 1222-1243 into modules/context/sections/plugins.py. Both follow existing section patterns (render() method, graceful error handling). Also registered both in SECTION_REGISTRY (__init__.py)
- [ ] **US-002: Register new sections and update mode configs** — Register composio + plugins in SECTION_REGISTRY. Split ContextMode.HEARTBEAT into HEARTBEAT_ORCHESTRATOR (dispatcher_only tools, simple) and HEARTBEAT_AGENT (full tools + composio + plugins). Add composio/plugins to CHATBOT, TASK_EXECUTION, RECIPE modes. Add memory to RECIPE. Update DEFAULT_BUDGETS for new modes
- [ ] **US-003: Update heartbeat_service.py** — Change ContextMode.HEARTBEAT to ContextMode.HEARTBEAT_ORCHESTRATOR

### Phase 2: Kill the Cache — ContextService Only Path

- [ ] **US-004: Remove system_prompt cache + delete legacy methods** — Remove system_prompt and skill_tool_schemas fields from AgentRuntime. Delete _build_agent_system_prompt() (~176 lines). Delete refresh_agent_prompt() (~18 lines). Update activate_agent() to not build a prompt. Fix ALL references across codebase. CRITICAL: grep every file for callers before deleting — check execution_manager.py especially
- [ ] **US-005: Make execute_with_prompt() always use ContextService** — Add context_mode parameter. Remove cached prompt branch. 2-way resolution: explicit > ContextService. Use context_result.tools when ContextService provides them. Remove skill_tool_schemas_from_prompt variable
- [ ] **US-006: Migrate heartbeat agent tick + verify channel adapters** — Pass context_mode=ContextMode.HEARTBEAT_AGENT to execute_with_prompt(). Remove use_memory=False. Verify channels/base.py does NOT pass explicit system_prompt (so it auto-uses ContextService)
- [ ] **US-006b: Migrate execution_manager.py to ContextService** — Remove direct _build_agent_system_prompt() call. Remove professional_system_prompt hardcoded block. Make execution_manager go through execute_with_prompt() without explicit system_prompt so ContextService handles it. Move useful instructions to user prompt prefix

### Phase 3: Memory Gap Verification + Daily Logs

- [ ] **US-007: Verify and fix memory + daily logs** — Confirm memory in TASK_EXECUTION + RECIPE modes. Verify MemorySection renders daily logs (## Recent Activity) for non-chatbot modes. Document HEARTBEAT_AGENT memory exclusion. Check recipe_executor.py for redundant manual Mem0 injection and remove if duplicate

### Phase 4: Dead Code Cleanup

- [ ] **US-008: Move legacy prompt instructions to sections** — Dependency context instructions → TaskContextSection. Skill tool usage instructions → SkillsSection. Response formatting → IdentitySection
- [ ] **US-009: Orphaned imports + grep audit + test fixes** — Remove orphaned imports from agent_factory.py. Run 5 grep audits (all must return zero non-ContextService results). Fix any broken tests

### Phase 5: Tech Debt

- [ ] **US-010: Consolidate tool loading aliases** — Replace get_agent_tools() and get_chat_tools() calls with get_tools_for_agent(). Delete aliases. Add SmartToolRouter future-plan comment
- [ ] **US-011: Personality decisions + get_platform_skill() deletion** — Delete get_platform_skill() if unused. Document personality=True/False rationale on each mode. Document daily logs exclusion from HEARTBEAT_AGENT
- [ ] **US-012: Agent resolution utility** — Create get_agent_with_context() in modules/agents/queries.py with joinedload(skills, persona). Replace bare db.query(Agent) in ContextService paths
- [ ] **US-013: Composio hints evaluation + datetime cleanup** — Compare _inject_composio_hints() with ComposioSection — delete if redundant, expand section if needed. Remove inline strftime from heartbeat agent prompt

---

## Discovered Issues

_(Ralph will populate during execution)_

## Notes

- 14 stories total across 5 phases
- Phase 2 is the riskiest — changes prompt for every task-executing agent
- US-006b is the key addition — execution_manager.py has its own prompt-building that would break when _build_agent_system_prompt() is deleted in US-004
- PRD-80 US-016 may have partially migrated execution_manager already — Ralph should check before doing work
- BEFORE DELETING CODE: always grep for callers (learn from intent_classifier.py deletion mistake)
- PRD-80 is complete (all 22 stories done) — this PRD finishes what PRD-80 left behind
