# PRD-82B Mission Intelligence Layer — Implementation Plan

## Overview
Make missions smarter: decomposition templates, history-based agent scoring, replanning on failure, telemetry queries, cross-task consistency verification, verification caching, save-as-routine, and archival.

## Branch: ralph/prd-82b-mission-intelligence

---

## Tasks

- [x] US-001: Create decomposition template library (templates.py)
- [x] US-002: Wire template matching into MissionPlanner
- [x] US-003: Wire history-based agent scoring in AgentMatcher
- [x] US-004: Add telemetry query endpoints for missions
- [ ] US-005: Add replanning state and replan endpoint
- [ ] US-006: Add cross-task consistency verification
- [ ] US-007: Add verification result caching
- [ ] US-008: Add save-as-routine conversion endpoint
- [ ] US-009: Add orchestration archive table and cleanup job

---

## Key References

- **Coordination modules**: `orchestrator/modules/coordination/` — planner.py, dispatcher.py, reconciler.py, agent_matcher.py, verification.py, deterministic_checks.py
- **Coordinator service**: `orchestrator/services/coordinator_service.py` — lifecycle methods, tick loop
- **Orchestration models**: `orchestrator/core/models/orchestration.py` — OrchestrationRun, OrchestrationTask, OrchestrationEvent, OrchestrationTaskDependency
- **State enums**: `orchestrator/core/models/orchestration_enums.py` — RunState, TaskState, EventType
- **Missions API**: `orchestrator/api/missions.py` — REST endpoints
- **Config**: `orchestrator/core/config.py` — all config constants
- **Agent matcher synonyms**: `orchestrator/modules/coordination/agent_matcher.py` — _ROLE_SYNONYMS dict (line 62)
- **Verification models**: `orchestrator/modules/coordination/verification.py` — VerificationResult, VERIFIER_MODEL_SELECTION
- **Board bridge**: `orchestrator/services/orchestration_board_bridge.py` — board task creation/sync
- **PRD**: `docs/PRDS/82B-MISSION-INTELLIGENCE-LAYER.md`

## Architecture Notes

- Python 3.11+ with SQLAlchemy ORM (sync sessions via `Session`)
- FastAPI endpoints with Pydantic request/response models
- Alembic for migrations (orchestrator/alembic/versions/)
- All DB writes use dual-write pattern: state change + orchestration_events append in same transaction
- Optimistic locking via version_id column on runs and tasks
- Config: ALL config values in orchestrator/core/config.py — NO os.getenv() elsewhere
- Cross-model verification: verifier must use different model family from executor
- Agent roles in templates must match _ROLE_SYNONYMS categories in agent_matcher.py
- React Query v4 on frontend (isLoading not isPending)
- Existing test patterns: check orchestrator/tests/ for conventions

## Validation

For backend Python:
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && python -c "
import orchestrator.modules.coordination.templates as t
import orchestrator.modules.coordination.planner as p
import orchestrator.modules.coordination.agent_matcher as am
import orchestrator.modules.coordination.verification as v
print('All imports OK')
"
```

For any new tests:
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && python -m pytest orchestrator/tests/ -x -q 2>&1 | tail -20
```

For frontend changes (US-008 only):
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai/frontend && npx tsc --noEmit 2>&1 | grep -iE "mission-detail|save-as-routine" | head -10
```
