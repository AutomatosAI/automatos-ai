# PRD-82B — Mission Intelligence Layer

**Version:** 1.0
**Type:** Implementation
**Status:** Draft
**Priority:** P1
**Depends On:** PRD-82A (Sequential Mission Coordinator)
**Research Base:** PRDs 102 (Coordinator), 103 (Verification), 104 (Model Selection), 105 (Budget Governance), 106 (Telemetry)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-16

---

## 1. Goal

Make missions smarter. 82A proved the lifecycle works (plan → assign → execute → verify → review). 82B adds the intelligence: learn from past missions, decompose with templates, replan on failure, query telemetry, and convert successful missions into reusable routines. Still sequential. Still roster agents only. The goal is quality and reliability, not scale.

## 2. 82A Production Learnings (What This PRD Fixes)

These are real bugs and gaps hit running missions in production on 2026-03-16:

| Issue | Root Cause | 82A Fix | 82B Fix |
|-------|-----------|---------|---------|
| Wrong agents assigned (ATLAS to research, QUILL to analysis) | Skill matching was too loose; `tool_coverage` gave 1.0 freebie when no tools required | Rebalanced weights, token-level synonym matching | **History scoring**: agents that verified well on similar tasks score higher |
| SCRIBE (gpt-4 8k) failed task 5 with upstream context overflow | No model context awareness for task position | Added `_LARGE_CONTEXT_MODELS` set | **Model selection per task position**: later tasks auto-prefer 128k+ models |
| Verification death loop (3 retries, all fail) | `must_pass=true` default too strict; LLM verification has ~15% false-fail rate | Changed default to `must_pass=false` | **Adaptive thresholds**: adjust based on historical override rate per task type |
| Mission fails → dead end | No replanning after max retries exhausted | N/A — mission enters `failed` | **Replanning**: redecompose failed subtree with learned context |
| LLM decomposition quality varies wildly | No templates, no structural guidance for common patterns | N/A — LLM-only | **Template library**: pre-built decomposition templates for research, analysis, content |
| No visibility into mission performance | Events captured but no query layer | N/A — write-only | **Telemetry API + dashboard data**: query events, agent scores, cost tracking |
| Good mission patterns lost | No way to reuse a successful mission structure | N/A | **Save as routine**: convert completed mission → reusable recipe |

## 3. What Ships

| Component | Source PRD | Description |
|-----------|-----------|-------------|
| History-based agent scoring | 102 §6.2 | Wire `AgentMatcher._score_history()` using accumulated `verifier_score` data |
| Decomposition template library | 102 §5.3 | Template dataclasses + keyword/embedding lookup + fallback to LLM |
| Replanning on failure | 102 §5.2 | `MissionPlanner.replan()` — redecompose failed subtree with failure context |
| Cross-task consistency verification | 103 §3.1 | Post-all-verified pass: check output coherence across tasks |
| Telemetry query API | 106 | Events by run/agent/type/time, agent score aggregates, cost breakdowns |
| Verification result caching | 103 | Don't re-verify unchanged output on retry (hash-based) |
| Adaptive verification thresholds | 103 | Track human override rate → auto-adjust `quality_threshold` |
| Save-as-routine conversion | 82A deferred | Completed mission → recipe with task template |
| `orchestration_archive` table | 101 | Move terminal runs older than 30d to archive |

## 4. What Does NOT Ship (Deferred)

| Deferred | Target | Why |
|----------|--------|-----|
| Parallel dispatch (`max_concurrent > 1`) | 82C | Sequential works; parallel adds race conditions |
| Contractor/ephemeral agents | 82C | Needs AgentFactory changes |
| Budget enforcement gates | 82C | Sequential is cheap |
| Multi-judge verification ensemble | 82C | Single cross-model judge works for v1 |
| Complexity detection ("this should be a mission") | 82D | Needs telemetry data first |
| Prompt coaching | 82D | Needs outcome data first |
| Mission planning conversation in chat | 82C | Modal creation works; conversational flow needs prompt pipeline changes |

---

## 5. History-Based Agent Scoring

### 5.1 Problem

82A's `AgentMatcher` has a placeholder `history_score = 0.5` for all agents. The `_score_history()` method exists in PRD-102 spec but was deferred. Without history, the same wrong agent gets picked repeatedly.

### 5.2 Design

Wire `_score_history()` to query `OrchestrationTask` for completed tasks by the same agent with similar `task_type` or `agent_role`. Use the average `verifier_score` as the history signal.

```python
def _score_history(
    db: Session,
    agent_id: int,
    task_type: Optional[str],
    agent_role: Optional[str],
    lookback_days: int = 30,
) -> float:
    """
    Average verifier_score for this agent on similar tasks in the last N days.

    Returns 0.5 (neutral) if fewer than 3 data points.
    """
    filters = [
        OrchestrationTask.assigned_agent_id == agent_id,
        OrchestrationTask.state == TaskState.VERIFIED.value,
        OrchestrationTask.verifier_score.isnot(None),
        OrchestrationTask.updated_at >= datetime.utcnow() - timedelta(days=lookback_days),
    ]
    if agent_role:
        filters.append(OrchestrationTask.agent_role == agent_role)

    scores = db.query(OrchestrationTask.verifier_score).filter(and_(*filters)).all()

    if len(scores) < 3:
        return 0.5  # Not enough data

    avg = sum(s[0] for s in scores) / len(scores)
    return round(avg, 4)
```

### 5.3 Integration

In `AgentMatcher._score_agent()`, replace `history_score = 0.5` with the DB query. Batch the query for all candidate agents to avoid N+1.

### 5.4 Cold Start

New agents or agents with < 3 verified tasks get `0.5` (neutral). This means established agents with good track records naturally float to the top. New agents still get selected via skill/tool/model scores.

### 5.5 Files Changed

| File | Change |
|------|--------|
| `modules/coordination/agent_matcher.py` | Wire `_score_history()`, add batch query |
| `modules/coordination/agent_matcher.py` | Accept `db` parameter in `_score_agent()` |

---

## 6. Decomposition Template Library

### 6.1 Problem

LLM-only decomposition quality varies. "Research top AI frameworks" sometimes produces 3 tasks, sometimes 7. Task descriptions are inconsistent. Agent roles are made up ("data collector" doesn't match any roster agent).

### 6.2 Template Dataclass

From PRD-102 §5.3:

```python
@dataclass(frozen=True)
class DecompositionTemplate:
    id: str                          # e.g. "research_and_report"
    name: str                        # "Research & Report"
    description: str                 # When to use this template
    keywords: list[str]              # Trigger keywords for matching
    task_templates: list[TaskTemplate]
    min_tasks: int
    max_tasks: int
    output_format: str               # "markdown_report" | "structured_data" | "comparison_table"

@dataclass(frozen=True)
class TaskTemplate:
    sequence: int
    agent_role: str                  # Must match roster agent skills
    title_pattern: str               # e.g. "Research {topic}"
    description_pattern: str         # Template with {goal} placeholders
    required_tools: list[str]        # e.g. ["TAVILY", "GOOGLEDOCS"]
    expected_output: str             # "list", "report", "analysis", "synthesis"
    verification_criteria: list[str] # What to check
```

### 6.3 Built-in Templates

| Template | Tasks | Pattern | Use When |
|----------|-------|---------|----------|
| `research_and_report` | 5 | Search → Deep Research → Analyse → Synthesise → Report | Goal contains "research", "compare", "evaluate" |
| `content_pipeline` | 4 | Research → Outline → Draft → Edit | Goal contains "write", "blog", "article", "content" |
| `competitive_analysis` | 4 | Identify Players → Research Each → Compare → Recommendation | Goal contains "competitive", "market analysis", "compare companies" |
| `data_investigation` | 3 | Gather Data → Analyse → Report Findings | Goal contains "investigate", "audit", "diagnose", "track" |

### 6.4 Template Matching

```python
def match_template(goal: str) -> Optional[DecompositionTemplate]:
    """
    Keyword-based template matching. Returns None if no template scores above threshold.

    Scoring: count keyword hits in goal text. Threshold: 2+ keywords match.
    """
    goal_lower = goal.lower()
    best_match = None
    best_score = 0

    for template in TEMPLATE_REGISTRY:
        score = sum(1 for kw in template.keywords if kw in goal_lower)
        if score >= 2 and score > best_score:
            best_match = template
            best_score = score

    return best_match
```

### 6.5 Planner Integration

`MissionPlanner.decompose()` pipeline becomes:

1. **Template match** — keyword lookup against goal
2. **If matched**: use template structure, LLM fills in specifics (task titles, descriptions, tool requirements)
3. **If no match**: fall back to current LLM-only decomposition (82A behavior)
4. **Structural validation** — same as 82A (DAG validation, agent role validation)

The template constrains task count, agent roles, and output format. The LLM customizes within those constraints.

### 6.6 Files

| File | Change |
|------|--------|
| `modules/coordination/templates.py` | **NEW** — Template dataclasses + registry + match function |
| `modules/coordination/planner.py` | Add template matching step before LLM decomposition |

---

## 7. Replanning on Failure

### 7.1 Problem

82A: task fails after `max_retries` → task enters `failed` → mission enters `failed`. Dead end. User has to create a new mission from scratch.

### 7.2 Design

When a task fails after max retries, the coordinator can **replan** the remaining subtree:

1. Gather failure context: what went wrong, which task, what the agent tried
2. Call `MissionPlanner.replan()` with the failure context + completed task outputs
3. Generate replacement tasks for the failed task and any dependent tasks
4. Insert new tasks into the DAG, mark old failed tasks as `skipped`
5. Resume execution from the new tasks

### 7.3 Triggers

| Trigger | Source | Action |
|---------|--------|--------|
| `task_failed_max_retries` | Reconciler | Auto-replan if `config.auto_replan = true`, else transition mission to `failed` |
| `human_replan_request` | API: `POST /missions/{id}/replan` | User requests replan with optional new instructions |
| `budget_warning` | Coordinator | Log warning, continue (no auto-replan on budget) |

### 7.4 Constraints

- **Max replans per mission:** 2 (configurable via `config.COORDINATOR_MAX_REPLANS`)
- **Replan scope:** Only the failed task + its downstream dependents. Completed/verified tasks are locked.
- **Context injection:** Replan prompt includes: original goal, completed task summaries, failure reason, user notes (if any)

### 7.5 Interface

```python
async def replan(
    self,
    db: Session,
    run: OrchestrationRun,
    trigger: ReplanTrigger,
    context: ReplanContext,
) -> DecompositionResult:
    """
    Re-decompose the remaining work after a failure.

    Args:
        run: The mission run
        trigger: What caused the replan (task_failure, human_request)
        context: Failure details, completed outputs, user notes

    Returns:
        New DecompositionResult with replacement tasks.
    """
```

### 7.6 State Machine Addition

```
running → replanning       # replan triggered
replanning → running       # new tasks inserted, execution resumes
replanning → failed        # replan failed or max replans exhausted
```

New `RunState`: `replanning` (ACTIVE type)

### 7.7 Files

| File | Change |
|------|--------|
| `core/models/orchestration_enums.py` | Add `replanning` to `RunState` |
| `modules/coordination/planner.py` | Implement `replan()` method |
| `modules/coordination/reconciler.py` | On task failure after max retries, trigger replan instead of failing mission |
| `services/coordinator_service.py` | Add `replan_mission()` method + API wiring |
| `api/missions.py` | Add `POST /missions/{id}/replan` endpoint |

---

## 8. Cross-Task Consistency Verification

### 8.1 Problem

82A verifies each task independently. Task 3 might contradict Task 1's findings. With sequential execution the risk is lower (each task sees upstream outputs), but contradictions still happen when agents interpret the same data differently.

### 8.2 Design

After all tasks are verified, run a single cross-task consistency check:

```python
async def verify_cross_task_consistency(
    self,
    db: Session,
    run_id: UUID,
    task_outputs: list[dict],  # [{task_id, title, output}]
) -> VerificationResult:
    """
    Check that all task outputs are consistent with each other and the original goal.

    Uses a different model family from the executor agents.
    """
```

### 8.3 Prompt Template

```
You are reviewing the combined outputs of a multi-agent mission.

ORIGINAL GOAL: {goal}

TASK OUTPUTS:
{for each task: Task {n}: {title}\n{output}\n---}

Check for:
1. Contradictions between task outputs
2. Missing coverage (goal aspects not addressed)
3. Redundant duplication
4. Logical coherence of the overall narrative

Return:
- overall_consistent: true/false
- issues: [{task_ids: [int], description: str, severity: "high"|"medium"|"low"}]
- summary: one paragraph assessment
```

### 8.4 Integration

In `reconciler.py`, after the last task is verified:
1. Collect all verified task outputs
2. Call `verify_cross_task_consistency()`
3. If consistent → proceed to `awaiting_human` (or `completed` if auto-approve)
4. If inconsistent with high-severity issues → flag for human review with the issues

### 8.5 Files

| File | Change |
|------|--------|
| `modules/coordination/verification.py` | Implement `verify_cross_task_consistency()` |
| `modules/coordination/reconciler.py` | Wire cross-task check after all tasks verified |

---

## 9. Telemetry Query API

### 9.1 Problem

82A writes `orchestration_events` on every state transition but there's no way to read them. No dashboards, no cost tracking, no agent performance visibility.

### 9.2 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/missions/{id}/events` | Events for a mission run, paginated, filterable by event_type |
| `GET` | `/api/missions/{id}/cost` | Token usage breakdown by task + total |
| `GET` | `/api/missions/stats` | Aggregate stats: success rate, avg duration, avg cost, top agents |
| `GET` | `/api/agents/{id}/mission-history` | Agent's mission performance: tasks completed, avg verifier score, failure rate |

### 9.3 Stats Response

```json
{
  "total_missions": 42,
  "success_rate": 0.78,
  "avg_duration_minutes": 12.5,
  "avg_tokens_used": 45000,
  "avg_tasks_per_mission": 4.2,
  "top_agents": [
    { "agent_id": 101, "name": "NOVA", "tasks_completed": 38, "avg_score": 0.82 }
  ],
  "common_failure_reasons": [
    { "reason": "verification_fail", "count": 8 },
    { "reason": "context_overflow", "count": 3 }
  ]
}
```

### 9.4 Files

| File | Change |
|------|--------|
| `api/missions.py` | Add 4 new endpoints |
| `services/coordinator_service.py` | Add query methods for stats/events |

---

## 10. Adaptive Verification Thresholds

### 10.1 Problem

82A uses a static `quality_threshold = 0.6` for all verification. Some task types need higher bars (final reports), some need lower (intermediate data gathering). Human overrides (approving despite verification failure) indicate the threshold is too strict.

### 10.2 Design

Track the human override rate per `agent_role` / task type:
- If humans approve > 30% of verification failures for a role → lower threshold by 0.05
- If humans reject > 20% of verification passes → raise threshold by 0.05
- Bounds: `[0.4, 0.85]`

Store adjusted thresholds in a new `verification_thresholds` table or in config.

### 10.3 Implementation

Phase 1 (82B): Track override rates. Log recommendations but don't auto-adjust.
Phase 2 (82C): Auto-adjust with human confirmation.

### 10.4 Files

| File | Change |
|------|--------|
| `modules/coordination/verification.py` | Query override rate, log threshold recommendation |
| `services/coordinator_service.py` | Capture human review decisions (approve/reject) with verification context |

---

## 11. Verification Result Caching

### 11.1 Problem

When a task retries (verification fail → retry with feedback), the agent might produce the same or very similar output. Re-running the full LLM verification wastes tokens.

### 11.2 Design

Hash the task output (`sha256(output_text)`). Before calling the LLM verifier, check if we've already verified this exact output. If so, return the cached result.

Cache key: `(task_id, output_hash)`
Cache TTL: duration of the mission run (not persisted beyond run completion)

### 11.3 Files

| File | Change |
|------|--------|
| `modules/coordination/verification.py` | Add hash check before LLM call |

---

## 12. Save as Routine

### 12.1 Problem

User runs a successful "Research top AI frameworks" mission. Next week they want to do "Research top LLM providers" — same structure, different topic. Currently they have to start from scratch.

### 12.2 Design

After a mission completes successfully, offer a "Save as Routine" action:

1. Extract the task graph structure (roles, dependencies, verification criteria)
2. Templatize: replace specific content with `{goal}` / `{topic}` placeholders
3. Save as a recipe in the existing recipes system
4. Next time a similar goal is detected, suggest the saved template

### 12.3 API

```
POST /api/missions/{id}/save-as-routine
{
  "name": "Research & Report",
  "description": "5-task research pipeline",
  "tags": ["research", "analysis"]
}
```

Response: the created recipe ID.

### 12.4 Frontend

Add a "Save as Routine" button on the mission detail page (only visible when mission is `completed`).

### 12.5 Files

| File | Change |
|------|--------|
| `api/missions.py` | Add save-as-routine endpoint |
| `services/coordinator_service.py` | Extract template from completed run |
| Mission detail page | Add button + confirmation dialog |

---

## 13. Orchestration Archive

### 13.1 Problem

`orchestration_runs` and `orchestration_tasks` will grow unbounded. Terminal runs (completed, failed, cancelled) older than 30 days should be archived.

### 13.2 Design

- New table: `orchestration_archive` — same schema as `orchestration_runs` + JSON blob of all tasks and events
- Cron job: daily, moves terminal runs older than `config.ARCHIVE_AFTER_DAYS` (default 30)
- Archive is read-only, queryable via `GET /api/missions/archive`
- Active queries (list, stats) exclude archived runs

### 13.3 Files

| File | Change |
|------|--------|
| `core/models/orchestration.py` | Add `OrchestrationArchive` model |
| Alembic migration | Create `orchestration_archive` table |
| `services/coordinator_service.py` | Add `archive_old_runs()` method |
| Cron job or reconciler tick | Trigger archival |

---

## 14. Configuration

New config entries for 82B:

| Key | Default | Description |
|-----|---------|-------------|
| `COORDINATOR_MAX_REPLANS` | `2` | Max replan attempts per mission |
| `COORDINATOR_AUTO_REPLAN` | `false` | Auto-replan on task failure (vs fail mission) |
| `COORDINATOR_HISTORY_LOOKBACK_DAYS` | `30` | Days of history for agent scoring |
| `COORDINATOR_HISTORY_MIN_DATAPOINTS` | `3` | Min verified tasks before using history score |
| `COORDINATOR_ARCHIVE_AFTER_DAYS` | `30` | Days before terminal runs are archived |
| `COORDINATOR_CONSISTENCY_CHECK` | `true` | Run cross-task consistency verification |

---

## 15. Implementation Order

| Phase | Components | Effort | Dependencies |
|-------|-----------|--------|-------------|
| 1 | History-based agent scoring | S | Needs accumulated verifier_score data from 82A runs |
| 2 | Decomposition templates | M | Standalone — improves plan quality immediately |
| 3 | Telemetry query API | S | Standalone — unlocks visibility |
| 4 | Verification caching + adaptive thresholds | S | Standalone |
| 5 | Cross-task consistency | S | After telemetry (to measure impact) |
| 6 | Replanning on failure | L | Needs state machine addition + planner changes |
| 7 | Save as routine | M | Needs recipe system integration |
| 8 | Orchestration archive | S | Can land anytime |

**Recommended order:** 2 → 1 → 3 → 4 → 5 → 6 → 7 → 8

Templates (phase 2) has the highest immediate impact — every mission benefits from better decomposition. History scoring (phase 1) depends on having enough data from 82A runs, so it naturally comes after some missions have been executed.

---

## 16. Success Criteria

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Template match rate | > 60% of missions use a template | Planner logs `template_used` field |
| Verification pass rate (first attempt) | > 70% (up from ~55% in 82A) | Query `orchestration_events` for verification outcomes |
| Agent selection accuracy | < 10% human overrides of agent assignment | Track agent reassignment rate |
| Mission success rate | > 85% (up from ~80% in 82A) | Terminal state ratio |
| Replan success rate | > 50% of replanned missions complete | Track replan → completed transitions |
| Avg mission duration | < 10 min for 5-task missions | Event timestamps |

---

## 17. Non-Goals

- This PRD does **not** add parallel execution — that's 82C
- This PRD does **not** add contractor/ephemeral agents — that's 82C
- This PRD does **not** add budget enforcement — sequential missions are cheap
- This PRD does **not** change the frontend mission detail page (except Save as Routine button)
- This PRD does **not** add the conversational mission planning flow in chat
