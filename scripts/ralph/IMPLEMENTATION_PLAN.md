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
- [x] **US-002: Register new sections and update mode configs** — Split ContextMode.HEARTBEAT into HEARTBEAT_ORCHESTRATOR (dispatcher_only, 8K max) and HEARTBEAT_AGENT (full tools + composio + plugins, 128K max). Added composio/plugins to CHATBOT, TASK_EXECUTION, RECIPE, HEARTBEAT_AGENT modes. Added memory to RECIPE. Updated DEFAULT_BUDGETS. Updated all tests (test_modes, test_service, test_budget_manager). SECTION_REGISTRY already had composio+plugins from US-001
- [x] **US-003: Update heartbeat_service.py** — Changed ContextMode.HEARTBEAT to ContextMode.HEARTBEAT_ORCHESTRATOR (done as part of US-002, single reference at line 440)

### Phase 2: Kill the Cache — ContextService Only Path

- [x] **US-004: Remove system_prompt cache + delete legacy methods** — Removed system_prompt and skill_tool_schemas fields from AgentRuntime. Deleted _build_agent_system_prompt() (~176 lines). Deleted refresh_agent_prompt() (~18 lines). Updated activate_agent() to skip prompt building. Removed cached prompt branch from execute_with_prompt() (3-way → 2-way: explicit > ContextService). Removed skill_tool_schemas_from_prompt variable. Fixed chatbot/service.py _load_agent_context() (removed dead cache branch). Removed orphaned imports (get_skill_loader, ComposioActionCache). Updated comment in agent_endpoints.py. All 114 context tests pass
- [x] **US-005: Make execute_with_prompt() always use ContextService** — Added context_mode: Optional[str] parameter to execute_with_prompt(). ContextService path uses `context_mode or ContextMode.TASK_EXECUTION` instead of hardcoded mode. Cached prompt branch already removed in US-004. context_result.tools already used (line 789). skill_tool_schemas_from_prompt already removed. Note: context_result.tool_choice exists but generate_response() interface doesn't accept tool_choice — documented as discovered issue for future LLM interface enhancement. All 114 context tests pass
- [x] **US-006: Migrate heartbeat agent tick + verify channel adapters** — Passed context_mode=ContextMode.HEARTBEAT_AGENT to execute_with_prompt() in _agent_tick(). Removed use_memory=False (memory exclusion handled by HEARTBEAT_AGENT mode not including "memory" section; short-term conversation memory is empty for heartbeat agents anyway). Verified channels/base.py does NOT pass system_prompt= — will auto-use ContextService TASK_EXECUTION mode after cache removal
- [x] **US-006b: Migrate execution_manager.py to ContextService** — ALREADY DONE (verified): No _build_agent_system_prompt() calls remain. No professional_system_prompt block exists. execute_with_prompt() called at line 869 without system_prompt= param, so ContextService TASK_EXECUTION mode handles identity/skills/tools/memory. Execution-specific guidance (professional instructions, workspace rules, conversion tool hints) lives in user prompt prefix (lines 822-865). Note: MemoryPromptInjector still used at line 550 to inject pre-fetched workflow memories into user prompt — potentially redundant with ContextService MemorySection (tracked for US-007)

### Phase 3: Memory Gap Verification + Daily Logs

- [x] **US-007: Verify and fix memory + daily logs** — Verified: TASK_EXECUTION and RECIPE modes both include "memory" in sections (lines 48 and 76 of modes.py). MemorySection renders daily logs via _retrieve_daily_logs() for all modes (uses workspace_id only, no chatbot dependency). _extract_query() falls back to task_description for non-chat modes. Added HEARTBEAT_AGENT memory exclusion comment to modes.py (PRD-81 Task 3.5/5.5). Recipe executor Mem0 injection (lines 206-217) is COMPLEMENTARY — uses RecipeMemoryService (recipe-scoped: past run learnings) vs MemorySection (workspace-scoped: user memories). Kept as-is. Execution manager MemoryPromptInjector (line 550) injects pre-fetched workflow-scoped memories into user prompt — different scope from MemorySection's live search but potential for duplicate content (tracked as DI-002)

### Phase 4: Dead Code Cleanup

- [x] **US-008: Move legacy prompt instructions to sections** — Added dependency context instructions (DEPENDENCY CONTEXT handling + document writing guidance) to TaskContextSection._build() after task metadata. Added skill tool usage instructions to SkillsSection._build() — extracts tool names from skills' tools_schema JSONB field via new _extract_skill_tool_names() static method, appends "Using Your Skill Tools" block when tools exist. Added response formatting guidance to IdentitySection._build() (non-chatbot path only) — synthesize API results into prose, use bullet points. Bumped max_tokens: TaskContextSection 1000→1500, IdentitySection 500→600. Fixed test_identity_section.py max_tokens assertion. All 114 context tests pass
- [x] **US-009: Orphaned imports + grep audit + test fixes** — Verified: PluginContextService, get_skill_loader, ComposioActionCache already removed in US-004. AgentAppAssignment + ComposioAppCache still live (used by _inject_composio_hints). All 5 grep audits pass with zero results: _build_agent_system_prompt (cleaned comment refs too), refresh_agent_prompt, professional_system_prompt, build_prompt_summary (non-ContextService), get_happy_system_prompt (non-ContextService). No broken tests found. Cleaned legacy method name references from docstrings/comments in identity.py, skills.py, task_context.py. All 114 context tests pass

### Phase 5: Tech Debt

- [x] **US-010: Consolidate tool loading aliases** — Deleted get_agent_tools() wrapper + _session_scope() from tool_router.py. Removed get_chat_tools/get_chatbot_tools re-exports from consumers/chatbot/tool_router.py. Removed get_tools()/CHAT_TOOLS lazy cache from consumers/chatbot/__init__.py. Removed get_chat_tools from consumers/__init__.py. Updated chatbot_llm.py to import get_tools_for_agent directly from modules.tools.tool_router. Removed get_agent_tools from modules/tools/__init__.py exports. Added SmartToolRouter future-plan comment (PRD-81 Task 5.3). Cleaned up stale docstring in get_tools_for_agent(). Note: mcp_executor.py:344 and composio_analytics.py:306 have their own get_agent_tools methods (different signatures, unrelated). All 114 context tests pass
- [x] **US-011: Personality decisions + get_platform_skill() deletion** — get_platform_skill() NOT deleted: has 1 active caller (identity.py:157 in _build_chatbot_identity()). Documented personality=True/False rationale on every mode in modes.py. Key decision: personality stays CHATBOT-only because get_base_system_prompt() is chatbot-specific (greetings, conversation counts, "never show code" rules — inappropriate for task/heartbeat agents). HEARTBEAT_AGENT daily logs exclusion comment already present from US-007. All 114 context tests pass
- [x] **US-012: Agent resolution utility** — Created get_agent_with_context() in modules/agents/queries.py with joinedload(skills, persona). Replaced bare db.query(Agent) in 3 ContextService paths: execute_with_prompt() (agent_factory.py:743), activate_agent() (agent_factory.py:579), _load_agent() (smart_orchestrator.py:264). Left non-ContextService paths alone (get_agent_status, admin endpoints, bulk queries). execution_manager.py already had joinedload(Agent.skills) at line 675. recipe_executor.py receives agent as param from bulk query — different pattern, left as-is. All 114 context tests pass
- [x] **US-013: Composio hints evaluation + datetime cleanup** — Evaluated: _inject_composio_hints() and _inject_composio_recipe_hints() are COMPLEMENTARY to ComposioSection, NOT redundant. ComposioSection renders static app-level descriptions in system prompt. Hint methods do dynamic per-request work: semantic action matching via ComposioHintService + constraining composio_execute tool schema enum with matched actions. Both kept with detailed comment block (PRD-81 Task 5.7). Removed inline strftime Time: line from _agent_tick() prompt (line 690) — datetime now provided by DatetimeContextSection in HEARTBEAT_AGENT mode. Remaining strftime in heartbeat_service.py is in _orchestrator_tick (line 451, different method) and daily summary (lines 834/880, date formatting). All 114 context tests pass

---

## Discovered Issues

- **DI-001: LLM interface missing tool_choice support** — `ContextResult.tool_choice` exists (default "auto") but `LLMManager.generate_response()` and all LLM client `generate_response()` methods only accept `messages` and `tools` — no `tool_choice` param. Adding it would require changes to the base class + all 8 client implementations. Low priority since "auto" is the default everywhere.
- **DI-002: Execution manager MemoryPromptInjector may duplicate ContextService MemorySection** — `execution_manager.py:550` uses `MemoryPromptInjector.inject_memory_into_prompt()` to inject pre-fetched workflow memories into user prompt. Now that ContextService TASK_EXECUTION mode includes MemorySection (live Mem0 search in system prompt), agents may receive overlapping memory content from both paths. The scopes differ (workflow-planned vs live-search) but content could overlap. Future cleanup: either remove MemoryPromptInjector and rely on MemorySection, or add dedup logic.

## Notes

- 14 stories total across 5 phases
- Phase 2 is the riskiest — changes prompt for every task-executing agent
- US-006b is the key addition — execution_manager.py has its own prompt-building that would break when _build_agent_system_prompt() is deleted in US-004
- PRD-80 US-016 may have partially migrated execution_manager already — Ralph should check before doing work
- BEFORE DELETING CODE: always grep for callers (learn from intent_classifier.py deletion mistake)
- PRD-80 is complete (all 22 stories done) — this PRD finishes what PRD-80 left behind
