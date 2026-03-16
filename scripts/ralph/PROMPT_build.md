# Build Mode

Implement ONE task from the plan, validate, commit, exit.

## Phase 0: Orient

Study with subagents:
- @CLAUDE.md (how to build/test)
- @docs/PRDS/81-MISSION-CLEANUP.md (full requirements)
- @scripts/ralph/IMPLEMENTATION_PLAN.md (current state)
- @scripts/ralph/prd.json (acceptance criteria for each story)

### Key References

- **ContextService**: `orchestrator/modules/context/service.py` — the unified context builder (PRD-80)
- **Context sections**: `orchestrator/modules/context/sections/` — identity.py, skills.py, platform_actions.py, memory.py, etc.
- **Context modes**: `orchestrator/modules/context/modes.py` — ContextMode enum + MODE_CONFIGS
- **Context budget**: `orchestrator/modules/context/budget.py` — TokenBudgetManager + DEFAULT_BUDGETS
- **Section registry**: `orchestrator/modules/context/sections/__init__.py` — SECTION_REGISTRY dict
- **AgentFactory**: `orchestrator/modules/agents/factory/agent_factory.py` — _build_agent_system_prompt (to delete), execute_with_prompt, activate_agent
- **Heartbeat service**: `orchestrator/services/heartbeat_service.py` — _orchestrator_tick_llm, _agent_tick
- **Recipe executor**: `orchestrator/api/recipe_executor.py` — _execute_step, _build_system_prompt
- **Execution manager**: `orchestrator/modules/agents/execution/execution_manager.py` — _execute_subtask
- **Tool router**: `orchestrator/modules/tools/tool_router.py` — get_tools_for_agent, get_agent_tools
- **Plugin context**: `orchestrator/core/services/plugin_context_service.py` — PluginContextService
- **Composio models**: `orchestrator/core/models/composio_cache.py` — AgentAppAssignment, ComposioAppCache, ComposioActionCache
- **Personality**: `orchestrator/consumers/chatbot/personality.py` — AutomatosPersonality, get_platform_skill
- **Smart tool router**: `orchestrator/consumers/chatbot/smart_tool_router.py` — SmartToolRouter
- **Chatbot tool router**: `orchestrator/consumers/chatbot/tool_router.py` — get_chat_tools re-export
- **System audit**: `docs/audits/SYSTEM-AUDIT-2026-03.md` — findings F1-F4, D1-D4, R1-R3
- **Config pattern**: `orchestrator/config.py` — ALL config constants live here, no os.getenv() elsewhere

### Check for completion

```bash
grep -c "^\- \[ \]" scripts/ralph/IMPLEMENTATION_PLAN.md || echo 0
```

- If 0: Run validation → commit → output **RALPH_COMPLETE** → exit
- If > 0: Continue to Phase 1

## Phase 1: Implement

1. **Study the plan** — Choose the FIRST unchecked task from @scripts/ralph/IMPLEMENTATION_PLAN.md
2. **Read prd.json** — Find the matching US-XXX story in @scripts/ralph/prd.json and follow its acceptance criteria exactly
3. **Search first** — Don't assume not implemented. Check if the component/service already exists
4. **Read existing code** — Before creating a new file, read the reference files listed in the story notes and follow existing patterns
5. **Implement** — ONE task only. Implement completely — no placeholders or stubs
6. **Validate** — Run typecheck/import check. All acceptance criteria must be met

### Architecture Rules (CRITICAL)

- NO hardcoded config values — all constants in `config.py`
- NO os.getenv() outside of `config.py`
- Follow existing section patterns in `modules/context/sections/` — each section has a `render()` method
- DB sessions acquired per-request from async pool — NEVER stored on singleton
- All methods include logging with exc_info=True on exceptions
- Follow immutable data patterns — return new objects, don't mutate
- BEFORE DELETING ANY CODE: grep EVERY file for callers. Remember the intent_classifier.py lesson — never delete code with live callers
- When removing fields from dataclasses/NamedTuples: search ALL references across the entire codebase

### Validation

For backend changes:
```bash
cd orchestrator && python -c "import api.main" 2>&1
```

For new modules:
```bash
cd orchestrator && python -c "from modules.context.service import ContextService" 2>&1
```

For grep audits (Phase 4+):
```bash
grep -rn "_build_agent_system_prompt" orchestrator/ --include="*.py" | grep -v __pycache__
grep -rn "refresh_agent_prompt" orchestrator/ --include="*.py" | grep -v __pycache__
```

If validation cannot run (e.g., missing deps), verify via grep that all imports resolve.

## Phase 2: Update & Learn

**Update scripts/ralph/IMPLEMENTATION_PLAN.md:**
- Mark completed task `- [x] Completed`
- Add any discovered bugs or issues
- Note new tasks discovered during implementation

**Update CLAUDE.md** (if you learned something new):
- Add correct commands or patterns discovered
- Keep it brief and operational

## Phase 3: Commit & Exit

```bash
git add -A && git commit -m "feat(context): [description of what was implemented]"
```

Check remaining:
```bash
grep -c "^\- \[ \]" scripts/ralph/IMPLEMENTATION_PLAN.md || echo 0
```

- If > 0: Say "X tasks remaining" and EXIT
- If = 0: Output **RALPH_COMPLETE**

## Guardrails

99999. Capture the why — tests and implementation importance.
999999. Single sources of truth, no migrations/adapters.
9999999. Implement functionality completely. No placeholders or stubs.
99999999. Keep @scripts/ralph/IMPLEMENTATION_PLAN.md current with learnings.
999999999. For any bugs you notice, resolve them or document them even if unrelated.
9999999999. ONE task per iteration. Search before implementing. Validation MUST pass. Never output RALPH_COMPLETE if tasks remain.
