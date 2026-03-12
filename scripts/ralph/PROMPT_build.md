# Build Mode

Implement ONE task from the plan, validate, commit, exit.

## Phase 0: Orient

Study with subagents:
- @CLAUDE.md (how to build/test)
- @docs/PRDS/79-UNIFIED-MEMORY-CONTEXT-ARCHITECTURE.md (full requirements)
- @scripts/ralph/IMPLEMENTATION_PLAN.md (current state)
- @scripts/ralph/prd.json (acceptance criteria for each story)

### Key References

- **Existing Mem0Client**: `orchestrator/modules/memory/integrations/mem0_client.py` — HTTP wrapper, circuit breaker, retry
- **SmartMemoryManager**: `orchestrator/consumers/chatbot/smart_memory.py` — primary chat memory path (854 lines)
- **Redis client**: `orchestrator/core/redis/client.py` — get_redis_client() singleton
- **Config pattern**: `orchestrator/config.py` — ALL config constants live here, no os.getenv() elsewhere
- **Platform tool pattern**: 3-file pattern — `platform_actions.py` (ActionDefinition), `platform_executor.py` (handler), `auto.py` (keywords)
- **Memory module**: `orchestrator/modules/memory/` — operations/, storage/, types/, integrations/
- **RecipeMemoryService**: `orchestrator/core/services/recipe_memory_service.py`
- **Widget memory**: `orchestrator/api/widget_memory.py`
- **Memory stats API**: `orchestrator/api/memory_stats.py`
- **Platform executor**: `orchestrator/modules/tools/discovery/platform_executor.py` — 5 inline Mem0Client() calls
- **Workflow memory**: `orchestrator/api/workflows.py` (~line 2075), `orchestrator/api/workflow_recipes.py` (~line 682)
- **MemoryInjector (deprecated)**: `orchestrator/modules/memory/operations/injection.py`
- **Mem0System adapter**: `orchestrator/modules/memory/storage/mem0_system.py`
- **Alembic migrations**: `orchestrator/alembic/versions/`

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
- NO direct Mem0Client() instantiation outside UnifiedMemoryService
- ALL memory user_ids built via MemoryNamespace helper — never raw string concatenation
- DB sessions acquired per-request from async pool — NEVER stored on singleton
- Redis failures must NEVER break chat — always try/except with graceful degradation
- Background writes (L2/L3 storage) use asyncio.create_task() — must not block TTFT
- All methods include logging with exc_info=True on exceptions
- Follow immutable data patterns — return new objects, don't mutate

### Validation

For backend changes:
```bash
cd orchestrator && python -c "import api.main" 2>&1
```

For new modules:
```bash
cd orchestrator && python -c "from modules.memory.unified_memory_service import UnifiedMemoryService" 2>&1
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
git add -A && git commit -m "feat(memory): [description of what was implemented]"
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
