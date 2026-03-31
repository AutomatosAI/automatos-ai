# PRD-122: Tool Routing Promotion & Permission Enforcement — Implementation Plan

## Overview

Two problems, one fix:
1. **Auto can't reliably call platform tools** — they're behind `platform_execute` dispatcher indirection. Promote ~13 high-value actions to first-class OpenAI tool schemas.
2. **`permission_level` is declared but never enforced** — 6 infrastructure tools expose system internals to every user. Enforce admin gating before promoting.

## Architecture

```
get_tools_for_agent() [tool_router.py:129]
  ├── ToolRegistry.get_all() → core tool schemas
  ├── ActionRegistry.to_dispatcher_schema() → platform_execute (non-promoted only)
  ├── ActionRegistry.to_first_class_schemas() → promoted schemas [NEW]
  └── ComposioActionCache enrichment → composio_execute params

Execution: unified_executor.py routes platform_* → PlatformActionExecutor (no changes needed)
```

## Key Files

| File | Purpose |
|------|---------|
| `orchestrator/modules/tools/discovery/action_registry.py` | ActionDefinition dataclass (line 27) + ActionRegistry class (line 53) |
| `orchestrator/modules/tools/discovery/platform_executor.py` | execute() at line 273, permission checks at 286-334 |
| `orchestrator/modules/tools/tool_router.py` | get_tools_for_agent() at line 129, dispatcher at 242 |
| `orchestrator/modules/tools/discovery/actions_monitoring.py` | 6 infrastructure tool registrations |
| `orchestrator/modules/tools/discovery/handlers_search.py` | search_chat_history handler, broken WHERE at line 33 |
| `orchestrator/consumers/chatbot/smart_tool_router.py` | TOOL_CATEGORIES at 65, INTENT_TO_TOOLS at 77 |
| `orchestrator/modules/context/sections/platform_actions.py` | PlatformActionsSection._build() at ~50 |
| `orchestrator/modules/tools/discovery/actions_agents.py` | Agent action registrations |
| `orchestrator/modules/tools/discovery/actions_marketplace.py` | Marketplace action registrations |
| `orchestrator/modules/tools/discovery/actions_search.py` | Memory + search action registrations |
| `orchestrator/modules/tools/unified_executor.py` | execute_tool() routing at ~310 |

## Tasks

### Phase 0: Permission Enforcement (P0)

- [x] **US-001**: Add `admin_only: bool = False` to ActionDefinition + mark 6 infra tools (5 in actions_monitoring.py, 1 in actions_workspace.py)
- [x] **US-002**: Thread caller_context (user_id, system_role, workspace_role) through unified_executor → platform_executor
- [x] **US-003**: Add admin gate (before requires_confirmation) + destructive safety check (after rate limit) in platform_executor.py
- [x] **US-004**: Add is_admin param to get_tools_for_agent() + exclude_admin to to_dispatcher_schema/build_prompt_summary
- [x] **US-005**: Fix search_chat_history WHERE clause to scope by workspace_id

### Phase 1: Promote High-Value Actions (P1)

- [x] **US-006**: Add `promoted: bool = False` to ActionDefinition + get_promoted() + to_first_class_schemas() methods
- [ ] **US-007**: Update build_prompt_summary(exclude_promoted=True) + to_dispatcher_schema(exclude_promoted=True)
- [ ] **US-008**: Mark ~13 actions as promoted=True in actions_agents, actions_marketplace, actions_monitoring, actions_search
- [ ] **US-009**: Append promoted schemas in tool_router.py + remove hardcoded _FIELD_TOOL_SCHEMAS + promote field tools
- [ ] **US-010**: Update SmartToolRouter with ALWAYS_INCLUDE set + promoted tool categories
- [ ] **US-011**: Update PlatformActionsSection._build() to pass exclude_promoted=True

### Phase 2: Dispatcher Enum (P2)

- [ ] **US-012**: Add enum of non-promoted action names to to_dispatcher_schema()

## Constraints

- **No execution layer changes** — unified_executor.py already routes `platform_*` directly (line 392). Only adding caller_context passthrough.
- **No migration needed** — workspace_members.role column already exists with owner|admin|editor|viewer|member values.
- **Backward compatible** — all new parameters default to None/False. Existing callers unaffected.
- **Field tool consolidation** — must verify to_openai_schema() output matches old hardcoded _FIELD_TOOL_SCHEMAS exactly.

## Quality Bar

- Every change is additive (new fields with defaults, new optional parameters)
- Admin gate is fail-closed (no caller_context = denied)
- Promoted actions are excluded from dispatcher (no dual paths)
- Log all permission denials at WARNING level
