# Build Mode

Implement ONE task from the plan, validate, commit, exit.

## Phase 0: Orient

Study with subagents:
- @CLAUDE.md (how to build/test)
- @scripts/ralph/IMPLEMENTATION_PLAN.md (current state — tasks + key references)
- @scripts/ralph/prd.json (acceptance criteria for each story)

### Key References

- **Coordination modules**: `orchestrator/modules/coordination/` — planner.py, dispatcher.py, reconciler.py, agent_matcher.py, verification.py, deterministic_checks.py
- **Coordinator service**: `orchestrator/services/coordinator_service.py` — lifecycle methods, create_mission, approve_plan, review_mission
- **Orchestration models**: `orchestrator/core/models/orchestration.py` — OrchestrationRun, OrchestrationTask, OrchestrationEvent
- **State enums**: `orchestrator/core/models/orchestration_enums.py` — RunState, TaskState, EventType, StateType
- **Missions API**: `orchestrator/api/missions.py` — existing REST endpoints (create, approve, reject, cancel, review, list, detail)
- **Config**: `orchestrator/core/config.py` — ALL config constants go here
- **Agent matcher**: `orchestrator/modules/coordination/agent_matcher.py` — _ROLE_SYNONYMS (line 62), _score_agent() (line 275), match() (line 105)
- **Verification**: `orchestrator/modules/coordination/verification.py` — VerificationResult, VerificationService, VERIFIER_MODEL_SELECTION
- **Templates**: `orchestrator/modules/coordination/templates.py` — DecompositionTemplate, TaskTemplate, match_template, render_template
- **Dependencies**: `orchestrator/services/orchestration_deps.py` — DependencyResolver.get_ready_tasks()
- **Budget manager**: `orchestrator/modules/orchestrator/stages/token_budget_manager.py` — TokenBudgetManager (exists but unwired)
- **Frontend missions**: `frontend/components/missions/` — mission-detail-page.tsx, mission-dag-canvas.tsx, mission-task-node.tsx
- **Frontend hooks**: `frontend/hooks/use-missions-api.ts` — React Query hooks
- **Frontend types**: `frontend/types/missions.ts` — TypeScript interfaces
- **PRD**: `docs/PRDS/82C-PARALLEL-EXECUTION-BUDGET-DECOMPOSITION.md`

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
4. **Read existing code** — Before creating or editing a file, read the files listed in the story notes and the Key References above to follow existing patterns
5. **Implement** — ONE task only. Implement completely — no placeholders or stubs
6. **Validate** — Run tests and import checks. All acceptance criteria must be met

### Architecture Rules (CRITICAL)

- Python 3.11+ with type hints on all public functions
- SQLAlchemy ORM with sync Session (not async) — follow existing patterns in coordinator_service.py
- FastAPI endpoints with Pydantic BaseModel for request/response
- ALL config values go in orchestrator/core/config.py — NO os.getenv() anywhere else
- Dual-write pattern: state change + orchestration_events append in SAME transaction
- Optimistic locking via version_id column — check existing patterns in orchestration_state.py
- Cross-model verification: verifier model family != executor model family
- Agent roles in templates: use categories from _ROLE_SYNONYMS in agent_matcher.py (researcher, writer, analyst, reviewer, search, etc.)
- NO hardcoded values — use config constants
- Follow immutable data patterns — frozen dataclasses, no mutation
- Error handling: log detailed context, raise clean HTTPExceptions for API layer
- BEFORE DELETING ANY CODE: grep EVERY file for callers

### Validation

For Python imports and basic syntax:
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && python -c "
from orchestrator.modules.coordination import templates, planner, agent_matcher, verification
print('All coordination imports OK')
" 2>&1 | tail -5
```

For tests (if any exist):
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && python -m pytest orchestrator/tests/ -x -q --timeout=30 2>&1 | tail -20
```

For frontend changes (US-011 only):
```bash
cd frontend && npx tsc --noEmit 2>&1 | grep -iE "mission-detail|mission-dag|budget-bar" | head -10
```

Note: Pre-existing errors may exist in other files. Only check for NEW errors introduced by your changes.

If no new errors appear, validation passes.

## Phase 2: Update & Learn

**Update scripts/ralph/IMPLEMENTATION_PLAN.md:**
- Mark completed task `- [x] Completed`
- Add any discovered bugs or issues
- Note new tasks discovered during implementation

**Update scripts/ralph/progress.txt:**
- Log what was completed this iteration

## Phase 3: Commit & Exit

```bash
git add -A && git commit -m "feat(missions): [description of what was implemented]"
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
