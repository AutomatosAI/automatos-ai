# Build Mode

Implement ONE task from the plan, validate, commit, exit.

## Phase 0: Orient

Study with subagents:
- @CLAUDE.md (how to build/test)
- @docs/PRDS/82A-SEQUENTIAL-MISSION-COORDINATOR.md (full requirements)
- @scripts/ralph/IMPLEMENTATION_PLAN.md (current state)
- @scripts/ralph/prd.json (acceptance criteria for each story)

### Key References

- **PRD-82A**: `docs/PRDS/82A-SEQUENTIAL-MISSION-COORDINATOR.md` — state machine (Section 4), board mapping (Section 4.3), dispatch claim pattern (Section 4.4), output summary (Section 6), failure codes (Section 8), budget tracking (Section 9)
- **PRD-101**: `docs/PRDS/101-ORCHESTRATION-DATA-SCHEMA.md` — canonical schema definitions, column types, constraints
- **PRD-102**: `docs/PRDS/102-COORDINATOR-ARCHITECTURE.md` — coordinator design, tick pattern, lifecycle methods
- **PRD-103**: `docs/PRDS/103-VERIFICATION-QUALITY.md` — verification service, deterministic checks, cross-model judge
- **Existing models**: `orchestrator/core/models/core.py` — Agent, BoardTask, Workspace models (reference for FK types, patterns)
- **Model registry**: `orchestrator/core/models/__init__.py` — how models are exported
- **AgentFactory**: `orchestrator/modules/agents/factory/agent_factory.py` — execute_with_prompt() signature
- **Heartbeat service**: `orchestrator/services/heartbeat_service.py` — scheduler tick pattern to follow
- **Board model**: `orchestrator/core/models/core.py` — BoardTask columns, source_type values
- **Context modes**: `orchestrator/modules/context/modes.py` — ContextMode enum + MODE_CONFIGS
- **Context budget**: `orchestrator/modules/context/budget.py` — TokenBudgetManager + DEFAULT_BUDGETS
- **Context sections**: `orchestrator/modules/context/sections/` — identity.py, skills.py etc (pattern to follow)
- **Section registry**: `orchestrator/modules/context/sections/__init__.py` — SECTION_REGISTRY dict
- **Tool router**: `orchestrator/modules/tools/tool_router.py` — get_tools_for_agent()
- **Config pattern**: `orchestrator/config.py` — ALL config constants live here, no os.getenv() elsewhere
- **Existing API routers**: `orchestrator/api/` — pattern for auth, workspace isolation, Pydantic models

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
- workspace_id is UUID type, created_by is String (Clerk user ID like 'user_xxx')
- All orchestration DB columns use `orchestration_*` naming, API uses `mission` naming
- `completed` task state is NOT terminal — only `verified`, `failed`, `skipped` are terminal
- Board task `done` status ONLY maps from `verified` state, NOT from `completed`
- Follow existing SQLAlchemy model patterns in `core/models/core.py`
- Follow existing section patterns in `modules/context/sections/` — each section has a `render()` method
- DB sessions acquired per-request from async pool — NEVER stored on singleton
- All methods include logging with exc_info=True on exceptions
- Follow immutable data patterns — return new objects, don't mutate
- Use datetime.now(timezone.utc) NOT datetime.utcnow()
- BEFORE DELETING ANY CODE: grep EVERY file for callers

### Validation

For new model imports:
```bash
cd orchestrator && python -c "from core.models.orchestration_enums import RunState, TaskState" 2>&1
```

For new modules:
```bash
cd orchestrator && python -c "from services.orchestration_state import transition_task" 2>&1
```

For coordination modules:
```bash
cd orchestrator && python -c "from modules.coordination.planner import MissionPlanner" 2>&1
```

For API:
```bash
cd orchestrator && python -c "from api.missions import router" 2>&1
```

For migrations:
```bash
cd orchestrator && alembic upgrade head 2>&1
```

If validation cannot run (e.g., missing deps), verify via grep that all imports resolve.

## Phase 2: Update & Learn

**Update scripts/ralph/IMPLEMENTATION_PLAN.md:**
- Mark completed task `- [x] Completed`
- Add any discovered bugs or issues
- Note new tasks discovered during implementation

**Update scripts/ralph/progress.txt:**
- Log what was completed this iteration

## Phase 3: Commit & Exit

```bash
git add -A && git commit -m "feat(orchestration): [description of what was implemented]"
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
