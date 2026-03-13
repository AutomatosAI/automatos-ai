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
- [ ] **US-005: Create SkillsSection** — SKILL.md content from agent_skills → skills table. Replaces skill injection in agent_factory
- [ ] **US-006: Create PlatformActionsSection** — Wraps ActionRegistry.build_prompt_summary(). Replaces inline injection in 3 files
- [ ] **US-007: Create MemorySection** — Wraps SmartMemoryManager.retrieve_memories() + daily logs. Replaces memory injection in smart_orchestrator + agent_factory
- [ ] **US-008: Create ToolsSection** — Unified tool loading with 4 strategies (FULL/FILTERED/DISPATCHER_ONLY/NONE). Replaces get_tools_for_agent + smart_tool_router.route + inline to_dispatcher_schema
- [ ] **US-009: Create TaskContextSection** — Task description, status, priority, board context
- [ ] **US-010: Create RecipeContextSection** — Recipe step name, instructions, previous results
- [ ] **US-011: Create DatetimeContext + Conversation + Custom sections** — Three lightweight sections to complete the library
- [ ] **US-012: Create ContextService.build_context()** — Main orchestrator: section composition, parallel rendering, budget allocation, tool loading, message formatting → ContextResult

### Phase 2: Migrate Callers (One at a Time, Least Risk First)

- [ ] **US-013: Migrate Heartbeat Service** — Replace inline f-string prompt + to_dispatcher_schema() with ContextService(HEARTBEAT)
- [ ] **US-014: Migrate Agent Factory** — Replace _build_agent_system_prompt() + get_tools_for_agent() with ContextService(TASK_EXECUTION)
- [ ] **US-015: Migrate Recipe Executor** — Replace recipe prompt assembly with ContextService(RECIPE)
- [ ] **US-016: Migrate Execution Manager** — Verify delegation works through factory, remove any redundant prompt building
- [ ] **US-017: Migrate Smart Orchestrator (Chatbot)** — Replace prepare_request() prompt/memory/tool assembly with ContextService(CHATBOT). Keep intent classification separate
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
