# PRD-80: Unified Context Service — Implementation Plan

> **Scope**: Backend (orchestrator) | **Risk**: Medium (new module, then incremental migration) | **Branch**: `ralph/80-unified-context-service`

## Summary

Replace 9 fragmented prompt-building code paths with a single `ContextService`. Composable sections (identity, skills, platform_actions, memory, tools, task_context, recipe_context, datetime, conversation, custom), token budget manager with priority-based trimming, mode-based declarative assembly.

## Reference

- **PRD**: `docs/PRDS/80-UNIFIED-CONTEXT-SERVICE.md`
- **Agent Factory**: `orchestrator/modules/agents/factory/agent_factory.py` (main execution path)
- **Smart Orchestrator**: `orchestrator/consumers/chatbot/smart_orchestrator.py` (chatbot path)
- **Personality**: `orchestrator/consumers/chatbot/personality.py` (get_happy_system_prompt)
- **Smart Tool Router**: `orchestrator/consumers/chatbot/smart_tool_router.py` (intent filtering)
- **Tool Router**: `orchestrator/modules/tools/discovery/tool_router.py` (get_tools_for_agent)
- **Action Registry**: `orchestrator/modules/tools/discovery/action_registry.py` (build_prompt_summary, to_dispatcher_schema)
- **Heartbeat Service**: `orchestrator/services/heartbeat_service.py` (inline prompt)
- **Recipe Executor**: `orchestrator/api/recipe_executor.py` (recipe prompts)
- **Execution Manager**: `orchestrator/modules/agents/execution/execution_manager.py` (thin wrapper)
- **Universal Router**: `orchestrator/core/routing/engine.py` (tier prompts)
- **Orchestrator Stages**: `orchestrator/modules/orchestrator/stages/*.py` (per-stage prompts)
- **NL2SQL**: `orchestrator/modules/nl2sql/service.py` (schema prompts)
- **Smart Memory**: `orchestrator/consumers/chatbot/smart_memory.py` (retrieve_memories)
- **Config**: `orchestrator/config.py` (all constants here)

## Tasks

### Phase 1: Build the Module (No Breaking Changes)

- [x] **US-001: Create package skeleton** — result.py (ContextResult frozen dataclass), modes.py (ContextMode enum + ModeConfig + MODE_CONFIGS), estimator.py (TokenEstimator), sections/__init__.py, __init__.py
- [x] **US-002: Create BaseSection ABC** — sections/base.py with SectionContext dataclass, render() ABC, estimate_tokens(), truncate()
- [x] **US-003: Create TokenBudgetManager** — budget.py with TokenBudget, RenderedSection, allocate() with priority-based trimming, DEFAULT_BUDGETS per mode
- [x] **US-004: Create IdentitySection** — Agent name, role, persona, personality. Replaces get_happy_system_prompt() identity + _build_agent_system_prompt() opening
- [x] **US-005: Create SkillsSection** — SKILL.md content from agent_skills → skills table. Replaces skill injection in agent_factory
- [x] **US-006: Create PlatformActionsSection** — Wraps ActionRegistry.build_prompt_summary(). Replaces inline injection in 3 files
- [x] **US-007: Create MemorySection** — Wraps SmartMemoryManager.retrieve_memories() + daily logs. Replaces memory injection in smart_orchestrator + agent_factory
- [x] **US-008: Create ToolsSection** — Unified tool loading with 4 strategies (FULL/FILTERED/DISPATCHER_ONLY/NONE). Replaces get_tools_for_agent + smart_tool_router.route + inline to_dispatcher_schema
- [x] **US-009: Create TaskContextSection** — Task description, status, priority, board context
- [x] **US-010: Create RecipeContextSection** — Recipe step name, instructions, previous results
- [x] **US-011: Create DatetimeContext + Conversation + Custom sections** — Three lightweight sections to complete the library. SECTION_REGISTRY (10 entries) exported from sections/__init__.py, all MODE_CONFIGS validated against it
- [x] **US-012: Create ContextService.build_context()** — Main orchestrator: section composition, parallel rendering, budget allocation, tool loading, message formatting → ContextResult

### Phase 2: Migrate Callers (One at a Time, Least Risk First)

- [x] **US-013: Migrate Heartbeat Service** — Replaced inline f-string prompt + to_dispatcher_schema() in _orchestrator_tick_llm with ContextService(HEARTBEAT). Added task_context to HEARTBEAT mode sections for heartbeat-specific instructions. Uses SimpleNamespace pseudo-agent for orchestrator identity. Single db session for both context building and tool execution
- [x] **US-014: Migrate Agent Factory** — In execute_with_prompt(), when no explicit/cached system_prompt exists, uses ContextService(TASK_EXECUTION) for both prompt and tools. When caller provides system_prompt (execution_manager, channels), keeps existing behavior. _build_agent_system_prompt() NOT deleted (Phase 3). Composio hints, tool loop, retries all preserved unchanged
- [x] **US-015: Migrate Recipe Executor** — Replaced _build_system_prompt() + get_chat_tools() in _execute_step() with ContextService(RECIPE). Added recipe_name and total_steps params to _execute_step(), builds recipe_step dict with previous_output from scratchpad. Base tools from ContextService, Composio overlay + scratchpad tools preserved. _build_system_prompt() is now dead code (Phase 3 cleanup). Scope instruction, memories, input data kept as additional system messages
- [x] **US-016: Migrate Execution Manager** — Removed _build_agent_system_prompt() call and explicit system_prompt pass from _execute_with_retries(). Now omits system_prompt → factory uses ContextService(TASK_EXECUTION) for identity, skills, platform actions, memory, tools. Professional instructions + workspace guidance moved to user prompt prefix. Skill tool schemas no longer manually extracted (ContextService handles tool loading)
- [x] **US-017: Migrate Smart Orchestrator (Chatbot)** — Replaced prepare_request() prompt/memory/tool assembly with ContextService(CHATBOT). Intent classification stays separate for response routing. Enhanced IdentitySection with full chatbot personality via AutomatosPersonality (base_system_prompt + platform_skill + tool_guidance + action_response + self_learning) when personality=True. Enhanced MemorySection with Context Router (PRD-79) support, skip_memory kwarg, and session hydration from Redis. Enhanced ToolsSection FILTERED strategy to always include platform_* tools (PRD-64 self-awareness). Added db_session passthrough: service.py → SmartChatIntegration → SmartChatOrchestrator → ContextService. Loads full Agent record from DB for SkillsSection. _build_compat_memory_result() provides backward compat for CTO override + SSE events. Removed imports: personality.py, smart_tool_router.py, ToolRoutingResult (all handled by ContextService sections). store_exchange() unchanged
- [ ] **US-018: Migrate Universal Router** — Replace per-tier prompts with ContextService(ROUTER)
- [ ] **US-019: Migrate Orchestrator Stages + NL2SQL** — Replace stage prompts and schema prompts with ContextService

### Phase 3: Cleanup

- [ ] **US-020: Dead code cleanup** — Delete _build_agent_system_prompt, simplify get_happy_system_prompt, delete smart_tool_router.py if unused. GREP BEFORE DELETING

### Phase 4: Tests

- [ ] **US-021: Unit tests** — Sections, budget manager, estimator, modes
- [ ] **US-022: Integration tests** — build_context() for each mode, failure resilience, immutability

---

## Discovered Issues

_(populated during implementation)_

## Notes

- 22 stories total
- Phase 1 creates all new code without touching existing paths — safe to ship at any point
- Phase 2 migrates one caller at a time — can stop mid-phase
- Phase 3 only after ALL callers migrated and verified
- Each story follows acceptance criteria in prd.json exactly
