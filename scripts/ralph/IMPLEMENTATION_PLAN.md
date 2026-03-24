# PRD-82C Parallel Execution, Budget & Decomposition — Implementation Plan

## Overview
Make missions parallel, budget-aware, and intelligently decomposed. Wire all scaffolded-but-unused code from 82A/B. Every story includes wiring tests that prove the code is called, not just defined.

## Branch: ralph/prd-82c-parallel-budget-decomposition

---

## Tasks

### Phase 1: Schema & Foundation
- [x] US-001: Add complexity, parallel_group, estimated_tokens columns. ComplexityTier + BudgetStatus enums. COMPLEXITY_TOKEN_BUDGET config. Change max_concurrent default to 3.

### Phase 2: Parallel Dispatch
- [x] US-002: Replace has_active_task() with count_active_tasks(). Add dispatch_ready() for multi-task dispatch.
- [x] US-003: Wire dispatch_ready() into coordinator tick. Execute via asyncio.gather().

### Phase 3: Intelligent Decomposition
- [x] US-004: Add _detect_complexity() to planner. Set max_concurrent on DecompositionResult.
- [x] US-005: Update planner system prompt for parallel groups + complexity. Parse new fields. Validate parallel_group cross-deps.
- [x] US-006: Rewrite all 4 templates with parallel groups and synthesis tasks.

### Phase 4: Synthesis & Budget
- [x] US-007: Build synthesis executor. _build_synthesis_prompt(). Detect TaskType.SYNTHESIS in _execute_task().
- [x] US-008: Auto-insert synthesis tasks when parallel branches converge without explicit synthesis.
- [x] US-009: Wire budget admission gate. Pre-dispatch can_afford() check. Graduated response. Pause on exceeded.

### Phase 5: API & Frontend
- [x] US-010: Enrich plan approval API with budget estimate, max_concurrent override.
- [x] US-011: Budget bar component, parallel DAG rendering, approval overrides.

### Phase 6: Wiring Tests
- [ ] US-012: Dedicated test suite proving all 82C features are wired end-to-end.

---

## Key References

| File | Purpose |
|------|---------|
| `orchestrator/modules/coordination/dispatcher.py` | Task dispatch — has_active_task (line ~78), dispatch_next |
| `orchestrator/modules/coordination/planner.py` | Goal decomposition — _SYSTEM_PROMPT (line ~530), _parse_plan (line ~769), _validate_plan (line ~862) |
| `orchestrator/modules/coordination/templates.py` | Template library — TaskTemplate (line ~30), TEMPLATE_REGISTRY (line ~60), render_template (line ~448) |
| `orchestrator/services/coordinator_service.py` | Tick loop — _process_run(), _execute_task(), token tracking (line ~624) |
| `orchestrator/services/orchestration_deps.py` | DAG resolution — DependencyResolver.get_ready_tasks() |
| `orchestrator/core/models/orchestration.py` | DB models — OrchestrationRun, OrchestrationTask |
| `orchestrator/core/models/orchestration_enums.py` | State machine — RunState, TaskState, TaskType, EventType |
| `orchestrator/core/config.py` | All config constants |
| `orchestrator/modules/orchestrator/stages/token_budget_manager.py` | Budget (exists but unwired — can_afford() never called) |
| `orchestrator/modules/coordination/verification.py` | Verification — VerificationService, ConsistencyResult |
| `frontend/components/missions/mission-detail-page.tsx` | Mission detail UI |
| `frontend/components/missions/mission-dag-canvas.tsx` | DAG visualization |
| `frontend/hooks/use-missions-api.ts` | React Query hooks |
| `frontend/types/missions.ts` | TypeScript interfaces |
| `docs/PRDS/82C-PARALLEL-EXECUTION-BUDGET-DECOMPOSITION.md` | Full PRD with architecture details |

## Architecture Rules (CRITICAL)

- Python 3.11+ with type hints on all public functions
- SQLAlchemy ORM with sync Session — follow existing patterns
- FastAPI endpoints with Pydantic BaseModel
- ALL config values go in orchestrator/core/config.py — NO os.getenv() anywhere else
- Dual-write pattern: state change + orchestration_events append in SAME transaction
- Optimistic locking via version_id column
- Agent roles in templates: use categories from _ROLE_SYNONYMS in agent_matcher.py
- NO hardcoded values — use config constants
- Frozen dataclasses for immutable data
- BEFORE DELETING ANY CODE: grep EVERY file for callers
- React Query v4 on frontend (isLoading not isPending)

## Validation

Backend Python imports:
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && python -c "
from orchestrator.modules.coordination import templates, planner, dispatcher, agent_matcher, verification
from orchestrator.services.coordinator_service import CoordinatorService
from orchestrator.core.models.orchestration_enums import TaskType, ComplexityTier, BudgetStatus
print('All imports OK')
" 2>&1 | tail -5
```

Tests:
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && python -m pytest orchestrator/tests/ -x -q --timeout=30 2>&1 | tail -20
```

Frontend:
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai/frontend && npx tsc --noEmit 2>&1 | grep -iE "mission-detail|mission-dag|budget-bar" | head -10
```

## Discovered Issues

(Ralph will log issues here during implementation)
