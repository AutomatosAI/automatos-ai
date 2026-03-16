# PRD-102 — Coordinator Architecture

**Version:** 1.0
**Type:** Research + Design
**Status:** Complete — Ready for Peer Review
**Priority:** P0
**Dependencies:** PRD-100 (Research Master), PRD-101 (Mission Schema)
**Blocks:** PRD-103 (Verification), PRD-104 (Ephemeral Agents), PRD-107 (Context Interface)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-15

---

## 1. Problem Statement

### 1.1 The Gap

Automatos has **no coordination layer**. The closest existing component is `heartbeat_service.py:_orchestrator_tick_llm()` (line ~382), which runs a 5-iteration tool loop with an 8,000-token budget and `dispatcher_only` tools — it does health checks and reporting, not goal decomposition or agent dispatch.

The platform can execute single-agent tasks beautifully. What it cannot do is take a complex goal — "Research EU AI Act compliance for our product" — and decompose it into subtasks, assign agents, execute with dependency ordering, verify outputs, handle failures, and track everything on the board.

### 1.2 What Exists vs What's Missing

| What Exists | What's Missing |
|---|---|
| `_orchestrator_tick_llm()` — LLM tool loop for workspace health checks | Goal decomposition: breaking complex goals into 3-20 subtasks with dependency edges |
| `AgentFactory.execute_with_prompt()` — per-agent execution with 10-iteration tool loop | Parallel dispatch: running independent subtasks concurrently via `asyncio.gather` |
| `AgentCommunicationProtocol` — Redis pub/sub messaging (built, not wired to heartbeat) | Cross-task data flow: passing Task 1's output as input to Task 2 |
| `BoardTask` with `assigned_agent_id` — manual task assignment | Automatic agent selection: matching task requirements to agent capabilities |
| `SharedContextManager` — in-process shared state with Redis backing (2h TTL) | Mission state machine: tracking plan → execute → verify → review lifecycle |
| `TaskReconciler` — stall detection for `recipe_executions` only | Mission-scoped stall detection, dependency-aware retry, escalation on failure |
| `ContextMode.HEARTBEAT_ORCHESTRATOR` — 8k tokens, 5 sections, dispatcher tools | `ContextMode.COORDINATOR` — full tools, mission context section, no token cap |

### 1.3 What This PRD Delivers

The architecture for a **CoordinatorService** that:
1. Takes a natural language goal + autonomy settings
2. Decomposes it into a dependency graph of 3-20 tasks (using PRD-101's `orchestration_tasks` schema)
3. Assigns each task to a roster agent or contractor agent
4. Dispatches tasks respecting dependency ordering
5. Monitors execution, handles failures (continuation vs retry)
6. Triggers verification (PRD-103) and human review gates
7. Detects mission completion and offers "save as routine"

### 1.4 What This PRD Does NOT Cover

| Out of Scope | Covered By |
|-------------|------------|
| How verification/scoring works | PRD-103 (Verification & Quality) |
| Ephemeral "contractor" agent lifecycle | PRD-104 (Ephemeral Agents & Model Selection) |
| Budget enforcement and approval gates | PRD-105 (Budget & Governance) |
| Outcome telemetry queries and learning | PRD-106 (Outcome Telemetry) |
| Context interface abstraction for Phase 3 | PRD-107 (Context Interface Abstraction) |
| Neural field prototype | PRD-108 (Memory Field Prototype) |
| SQL DDL and Alembic migrations | PRD-101 (already delivered) and PRD-82A (implementation) |

### 1.5 Design Philosophy

Four principles guided every decision:

1. **Stateless coordinator, DB-authoritative.** The coordinator holds no in-process state. Every tick reads from `orchestration_runs` / `orchestration_tasks` and writes back. Any coordinator instance can take over after a crash. This is the Airflow scheduling pattern validated at massive scale.

2. **Two-phase tick (Symphony pattern).** Every coordinator cycle runs dispatch (find ready tasks, assign agents) then reconcile (check running tasks for stalls, completions, failures). Clean separation, predictable behavior.

3. **HTN-inspired hybrid planning.** Template library for known mission types + LLM for novel goals + structural validation for all plans. Never pure LLM (non-deterministic), never pure rules (brittle).

4. **BDI intention commitment.** Once committed to a plan, the coordinator does not replan on every tick. Replanning triggers are explicit: task failure after max retries, user sends new instructions, budget warning. This prevents thrashing.

---

## 2. Prior Art: Coordination Patterns

### 2.1 Overview

Seven systems and architectural patterns were studied to inform the coordinator design. Each addresses a different facet of the coordination problem: how to plan, how to track state, how to handle failure, how to involve humans.

### 2.2 Comparison Table

| Aspect | Blackboard (Nii 1986, LbMAS 2025) | HTN Planning (ChatHTN 2025, Hsiao 2025) | BDI Agents (Rao & Georgeff 1995, ChatBDI 2025) | Symphony (OpenAI) | CrewAI | AutoGen | LangGraph |
|---|---|---|---|---|---|---|---|
| **Coordination model** | Shared state + event-driven knowledge source activation | Hierarchical decomposition of compound tasks into primitives | Belief-Desire-Intention deliberation cycle | Reconciliation loop (dispatch + reconcile) with policy-as-code | Sequential or hierarchical (LLM-as-manager) process | Turn-based group chat with LLM speaker selection | Typed state graph with deterministic conditional edges |
| **State management** | Blackboard data structure (shared, hierarchical) | World state updated at each primitive step | Belief base (agent's model of world) | External tracker (Linear) + workspace filesystem | In-memory crew state; Flows add SQLite persistence | In-memory message list (ephemeral) | Typed schema + pluggable checkpointers (Postgres, SQLite) |
| **Planning approach** | Opportunistic — no predetermined path | Method library for known decompositions; backtracking for alternatives | Plan library indexed by triggering events; LLM can generate plans dynamically | No planning — work comes from external tracker | LLM-as-manager in hierarchical mode; AgentPlanner pre-generates steps | No planning — conversation-driven emergence | Graph defined at compile time; conditional routing for branching |
| **Failure handling** | Knowledge sources produce competing hypotheses; control resolves conflicts | Backtrack and try alternative method | Plan failure propagation with alternative plan selection; bold/cautious reconsideration | Continuation (1s) vs retry (exponential backoff); workspace preserved | Guardrail retry loop (max 3); soft failure — proceeds with bad output | No built-in failure handling | Checkpoint enables resume from last successful step |
| **Human review** | Not built-in | Not built-in | Not built-in (agent is autonomous) | PR review is the human gate; no mid-execution review | `human_input=True` per task; `@human_feedback` in Flows | `human_input_mode` on UserProxyAgent | `interrupt()` pauses execution; resume with human input |

### 2.3 System-by-System Analysis

#### Blackboard Architecture (Nii 1986; LbMAS, arxiv:2507.01701, 2025)

The blackboard pattern coordinates multiple "knowledge sources" (KS) through a shared workspace. Each KS has activation preconditions — it fires when data it can process appears on the blackboard. A control component resolves conflicts when multiple KS are eligible.

LbMAS (2025) modernized this for LLM multi-agent systems and demonstrated a 5% improvement over static agent configurations. The key insight: **event-driven activation** (agent fires when its dependencies appear on the shared state) outperforms polling.

**What we adopt:** The mission state object (`orchestration_runs` + `orchestration_tasks` from PRD-101) acts as a blackboard. Agents write results to it; the coordinator reads it to decide next actions. Task activation is dependency-driven — a task becomes `queued` when all its parent dependencies reach terminal success state.

**What we reject:** The BB1 control blackboard (a second blackboard to manage the first — overkill for 3-20 tasks). Distributed blackboard partitioning (premature for our scale).

#### HTN Planning (Nau et al. JAIR 2003; ChatHTN, arxiv:2505.11814, 2025; Hsiao et al., arxiv:2511.07568, 2025)

Hierarchical Task Network planning decomposes compound tasks into primitive actions using a library of decomposition methods. SHOP2 (Nau et al.) proved this formally correct for forward-search decomposition.

ChatHTN (2025) proved that a hybrid approach — symbolic HTN structure with LLM filling in the gaps — is provably sound. The LLM generates decomposition candidates; the HTN validator ensures structural correctness (no cycles, valid dependencies, feasible agent assignments).

Hsiao et al. (2025) showed that hand-coded HTN structures enable 20-70B parameter models to outperform 120B baselines. **Structure improves LLM planning quality.** This means our decomposition templates aren't just efficiency shortcuts — they make planning better.

**What we adopt:** Template library for known mission types (the "methods" in HTN terminology). LLM generates decomposition for novel goals. All plans validated structurally before execution — DAG check, agent availability, budget estimate. This is the ChatHTN hybrid.

**What we reject:** Full formal HTN domain models (too rigid for natural language goals). Requiring hand-authored methods for every decomposition (LLM handles novel cases).

#### BDI Agents (Rao & Georgeff, ICMAS 1995; ChatBDI, AAMAS 2025)

Belief-Desire-Intention architecture models rational agent behavior. The critical insight for coordinators: **intention commitment**. Once an agent commits to an intention (plan), it should not reconsider on every deliberation cycle. Kinny & Georgeff proved that bold agents (reconsider rarely) outperform cautious agents (reconsider constantly) in stable environments.

ChatBDI (2025) adapted BDI for LLM agents, showing that the intention stack prevents the "thrashing" problem where agents constantly replan instead of executing.

**What we adopt:** The bold/cautious spectrum maps directly to the autonomy toggle. `approve` mode = cautious (human gates at plan approval and result review). `autonomous` mode = bolder (replan only on failure). In both cases, the coordinator commits to a plan and does not replan on every tick — only on explicit triggers (Section 5.5).

**What we reject:** The full BDI deliberation cycle (belief revision, desire filtering, plan selection). Our coordinator is simpler — it has one goal (the mission), one plan (the decomposition), and reconsiders only when reality diverges from the plan.

#### Symphony (OpenAI)

Symphony's defining contribution is the **two-phase reconciliation tick**:
1. **Dispatch phase:** Find tasks whose dependencies are met, claim them, assign agents
2. **Reconcile phase:** Check running tasks for stalls, completions, external state changes

This separation is cleaner than a single monolithic loop because dispatch decisions don't interleave with reconciliation decisions. Each phase has a clear contract: dispatch reads `pending` tasks, reconcile reads `running` tasks.

Symphony's continuation vs retry distinction (Section 3.5 of PRD-101) is adopted wholesale. A clean agent exit → continuation (1s delay, same workspace). A failure → retry (exponential backoff). This prevents backoff on normal multi-turn agent work while protecting against failure loops.

**What we adopt:** Two-phase tick. Continuation vs retry. WORKFLOW.md-style state-specific coordinator instructions (the coordinator prompt changes based on mission state). Stall detection via elapsed time since last event.

**What we reject:** Linear-as-coordinator (we have our own board). In-memory-only state (we need persistent mission history). Single-agent-per-task constraint (we support contractor fan-out within PRD-104).

#### CrewAI

CrewAI's `context=[task_a, task_b]` dependency declaration maps directly to PRD-101's `orchestration_task_dependencies` join table. The explicit, declarative, queryable dependency model is what we need.

The guardrail validation pattern — a function that checks output before accepting it — is a simplified version of what PRD-103 (Verification) delivers.

**What we adopt:** Explicit dependency declarations. The `async_execution` + join pattern for parallel tasks.

**What we reject:** LLM-as-manager for agent selection (non-deterministic, untestable). Soft guardrail failure mode (bad output proceeds — unacceptable for missions).

#### AutoGen

AutoGen's Swarm handoff pattern defines priority ordering for task transitions: tool-returned agent → OnCondition → AFTER_WORK fallback. The `context_variables` dict as shared mutable state across agents maps to our mission-scoped context.

**What we adopt:** Priority ordering for coordinator task transitions (dependency-resolved tasks first, then stalled task recovery, then budget checks). Shared mutable context per mission (via `SharedContextManager` in Phase 2, neural field in Phase 3).

**What we reject:** LLM-based speaker selection per turn (expensive, non-deterministic). Magic-string termination conditions. Ephemeral state.

#### LangGraph

LangGraph's typed state schema with checkpoint-per-step is the closest to our DB-authoritative model. The `interrupt()` mechanism for human review maps to our `awaiting_approval` and `awaiting_human` states.

**What we adopt:** Typed state schema (our `orchestration_runs`/`orchestration_tasks` tables). Checkpoint per state transition (our dual-write to event log). `interrupt()` for human review (our `awaiting_human` task state). Send API for dynamic parallelism (our coordinator dispatching multiple tasks concurrently).

**What we reject:** Full boilerplate burden of graph compilation. Static graph definition at compile time (our plans are generated per-mission). LangSmith vendor lock-in.

### 2.4 Architectural Decisions Summary

| Decision | Pattern | Source | Rationale |
|----------|---------|--------|-----------|
| **Tick structure** | Two-phase: dispatch + reconcile | Symphony | Clean separation; each phase has a clear contract |
| **Planning** | HTN-inspired hybrid: templates + LLM + validation | ChatHTN, Hsiao et al. | Templates improve quality; LLM handles novel goals; validation catches structural errors |
| **State authority** | DB-authoritative, stateless coordinator | Airflow, LangGraph | Crash-safe; any instance can take over |
| **Replanning policy** | BDI intention commitment — replan on explicit triggers only | Rao & Georgeff, ChatBDI | Prevents thrashing; matches autonomy toggle |
| **Mission state** | Blackboard pattern — shared state with event-driven activation | Nii, LbMAS | Tasks activate when dependencies met; coordinator reads blackboard each tick |
| **Dependencies** | Explicit join table, queryable both directions | CrewAI, Airflow | Declarative, queryable, validates DAG structure |
| **Failure handling** | Continuation vs retry + infrastructure/quality failure classification | Symphony, Prefect | Different strategies for different failure types |
| **Human review** | Interrupt-based: plan approval + result review | LangGraph, Symphony | Two human gates; configurable per autonomy level |
| **Agent selection** | Deterministic scoring, not LLM-based | (Anti-pattern from CrewAI) | Reproducible, testable, debuggable |

---

## 3. CoordinatorService Architecture

### 3.1 Module Hierarchy

```
orchestrator/
├── services/
│   └── coordinator_service.py          # Main service: tick registration, lifecycle
├── modules/
│   └── coordination/
│       ├── __init__.py
│       ├── planner.py                  # MissionPlanner: goal → task graph
│       ├── dispatcher.py               # MissionDispatcher: resolve ready tasks, assign agents
│       ├── reconciler.py               # MissionReconciler: stall detection, completion, failure
│       ├── agent_matcher.py            # AgentMatcher: score agents against task requirements
│       └── templates/
│           ├── __init__.py
│           └── decomposition_templates.py  # HTN-inspired template library
```

**Rationale:** `coordinator_service.py` lives in `services/` alongside `heartbeat_service.py` and `task_reconciler.py` — it's a service that registers its tick on the shared scheduler. Supporting classes live in `modules/coordination/` because they encapsulate domain logic (planning, dispatching, reconciling) that doesn't belong in the service entry point.

### 3.2 Class Diagram

```
┌─────────────────────────────────────────┐
│           CoordinatorService            │
├─────────────────────────────────────────┤
│ - _planner: MissionPlanner              │
│ - _dispatcher: MissionDispatcher        │
│ - _reconciler: MissionReconciler        │
│ - _scheduler: UnifiedScheduler          │
│ - _db: AsyncSession                     │
├─────────────────────────────────────────┤
│ + register_tick()                       │
│ + tick()                                │
│ + create_mission(goal, config) → Run    │
│ + approve_plan(run_id) → Run            │
│ + reject_plan(run_id, reason) → Run     │
│ + review_mission(run_id, verdict) → Run │
│ + pause_mission(run_id) → Run           │
│ + resume_mission(run_id) → Run          │
│ + cancel_mission(run_id) → Run          │
└─────────────────────────────────────────┘
              │ uses
    ┌─────────┼────────────┐
    ▼         ▼            ▼
┌──────────┐ ┌───────────┐ ┌──────────────┐
│ Mission  │ │ Mission   │ │  Mission     │
│ Planner  │ │ Dispatcher│ │  Reconciler  │
├──────────┤ ├───────────┤ ├──────────────┤
│ decompose│ │ dispatch  │ │ reconcile    │
│ validate │ │ assign    │ │ detect_stalls│
│ replan   │ │ schedule  │ │ handle_done  │
└──────────┘ └───────────┘ └──────────────┘
    │              │              │
    ▼              ▼              ▼
┌──────────┐ ┌───────────┐ ┌──────────────┐
│ Agent    │ │ Agent     │ │  Task        │
│ Matcher  │ │ Factory   │ │  Reconciler  │
│ (new)    │ │ (existing)│ │  (existing)  │
└──────────┘ └───────────┘ └──────────────┘
```

### 3.3 Public Interface

```python
class CoordinatorService:
    """
    Stateless mission coordinator.

    Reads mission state from DB on every tick.
    Writes state changes back within transactions.
    Any instance can take over after a crash.
    """

    def __init__(
        self,
        db_session_factory: Callable[[], AsyncSession],
        planner: MissionPlanner,
        dispatcher: MissionDispatcher,
        reconciler: MissionReconciler,
        scheduler: UnifiedScheduler,
    ) -> None: ...

    async def register_tick(self) -> None:
        """Register coordinator tick on shared scheduler. Default: 5s interval."""
        ...

    async def tick(self) -> None:
        """
        Two-phase coordinator tick.
        Phase A: Dispatch ready tasks for all active missions.
        Phase B: Reconcile running tasks across all active missions.
        """
        ...

    # --- Mission Lifecycle ---

    async def create_mission(
        self,
        workspace_id: uuid.UUID,
        goal: str,
        config: MissionConfig,
        created_by: str,  # Clerk user ID (e.g., "user_2abc...")
    ) -> OrchestrationRun:
        """
        Create a new mission. Transitions: → pending → planning.
        Calls planner.decompose() to generate task graph.
        If autonomy mode: auto-approves and transitions to running.
        If approve mode: transitions to awaiting_approval.
        """
        ...

    async def approve_plan(
        self, run_id: uuid.UUID, modifications: Optional[PlanModification] = None
    ) -> OrchestrationRun:
        """Human approves plan. Transitions: awaiting_approval → running."""
        ...

    async def reject_plan(
        self, run_id: uuid.UUID, reason: str
    ) -> OrchestrationRun:
        """Human rejects plan. Transitions: awaiting_approval → failed."""
        ...

    async def review_mission(
        self, run_id: uuid.UUID, verdict: ReviewVerdict
    ) -> OrchestrationRun:
        """
        Human reviews completed mission.
        verdict.accept → completed
        verdict.reject(task_ids, feedback) → tasks re-queued for retry
        verdict.reject_all → failed
        """
        ...

    async def pause_mission(self, run_id: uuid.UUID) -> OrchestrationRun:
        """Human pauses. Transitions: running → paused."""
        ...

    async def resume_mission(self, run_id: uuid.UUID) -> OrchestrationRun:
        """Human resumes. Transitions: paused → running."""
        ...

    async def cancel_mission(self, run_id: uuid.UUID) -> OrchestrationRun:
        """Human cancels. Transitions: any non-terminal → cancelled."""
        ...
```

---

## 4. Coordinator Tick Algorithm

### 4.1 Overview

The coordinator tick runs on a configurable interval (default: 5 seconds, matching Symphony's default). Each tick processes ALL active missions in the workspace, not just one.

```
Every 5 seconds:
    active_missions = SELECT * FROM orchestration_runs
                      WHERE workspace_id = :ws
                      AND state = 'running'

    For each mission in active_missions:
        Phase A: Dispatch (find and launch ready tasks)
        Phase B: Reconcile (check running tasks)
```

### 4.2 Phase A: Dispatch

```python
async def _dispatch_phase(self, run: OrchestrationRun) -> None:
    """
    Find tasks whose dependencies are met and dispatch them.
    Runs within a single DB transaction per mission.
    """
    if run.state == RunState.PAUSED:
        return  # Skip paused missions (budget_exceeded is also PAUSED)

    # 1. Find ready tasks
    ready_tasks = await self._find_ready_tasks(run.id)

    # 2. Respect concurrency limits
    running_count = await self._count_running_tasks(run.id)
    max_concurrent = run.config.get("max_concurrent_tasks", 3)
    slots_available = max(0, max_concurrent - running_count)

    if slots_available == 0:
        return

    # 3. Dispatch up to available slots (priority: earlier task_order first)
    for task in ready_tasks[:slots_available]:
        await self._dispatcher.dispatch_task(run, task)


async def _find_ready_tasks(self, run_id: uuid.UUID) -> list[OrchestrationTask]:
    """
    A task is ready when:
    1. status = 'queued' (dependencies already resolved)
    2. No budget block (PRD-105 pre-check)

    Dependency resolution happens via a trigger: when a task completes,
    check all tasks that depend on it. If all their dependencies are
    now terminal-success, transition them from 'pending' to 'queued'.
    This is more efficient than scanning all pending tasks every tick.
    """
    return await self._db.execute(
        select(OrchestrationTask)
        .where(
            OrchestrationTask.run_id == run_id,
            OrchestrationTask.state == TaskState.QUEUED,
        )
        .order_by(OrchestrationTask.sequence_number)
    )
```

### 4.3 Phase B: Reconcile

```python
async def _reconcile_phase(self, run: OrchestrationRun) -> None:
    """
    Check running tasks for stalls, completions, and mission-level state.
    """
    # 1. Check for stalled tasks
    running_tasks = await self._get_tasks_by_state(
        run.id, [TaskState.RUNNING, TaskState.ASSIGNED, TaskState.CONTINUING]
    )
    for task in running_tasks:
        await self._reconciler.check_task_health(run, task)

    # 2. Check if all tasks are terminal
    task_summary = await self._get_task_summary(run.id)

    if task_summary.all_terminal:
        if task_summary.all_passed:
            # All tasks completed successfully → advance to verification/review
            await self._advance_to_review(run)
        elif task_summary.has_unrecoverable_failure:
            # At least one task failed with no retries remaining
            await self._transition_run(
                run, RunState.FAILED,
                reason=f"Task {task_summary.failed_task_ids[0]} failed after max retries"
            )

    # 3. Check budget status (PRD-105)
    budget_status = await self._check_budget(run)
    if budget_status == BudgetStatus.EXCEEDED:
        await self._transition_run(run, RunState.BUDGET_EXCEEDED)
    elif budget_status == BudgetStatus.WARNING:
        await self._emit_event(run.id, "budget_warning", {
            "spent": budget_status.spent,
            "cap": budget_status.cap,
            "pct": budget_status.pct,
        })
```

### 4.4 Dependency Resolution

When a task completes, the coordinator must check whether downstream tasks are now unblocked. This is event-driven, not polling-based (blackboard pattern).

```python
async def _on_task_completed(self, run_id: uuid.UUID, completed_task_id: int) -> None:
    """
    Called when a task reaches terminal success state.
    Check all downstream dependents — if all their parents are
    terminal-success, transition them from 'pending' to 'queued'.
    """
    # Find tasks that depend on the completed task
    dependents = await self._db.execute(
        select(OrchestrationTask)
        .join(
            OrchestrationTaskDependency,
            OrchestrationTaskDependency.task_id == OrchestrationTask.id,
        )
        .where(
            OrchestrationTaskDependency.depends_on_id == completed_task_id,
            OrchestrationTask.state == TaskState.PENDING,
        )
    )

    for dependent in dependents:
        # Check ALL dependencies of this dependent task
        all_deps_met = await self._all_dependencies_met(dependent.id)
        if all_deps_met:
            await self._transition_task(
                dependent, TaskState.QUEUED,
                reason=f"All dependencies met (last: task {completed_task_id})"
            )
```

### 4.5 Stall Detection

```python
async def check_task_health(
    self, run: OrchestrationRun, task: OrchestrationTask
) -> None:
    """
    Detect stalled tasks using elapsed time since last event.

    Stall thresholds (configurable per mission):
    - ASSIGNED state: 30s (agent should start quickly)
    - RUNNING state: 300s default (configurable per task type)
    - CONTINUING state: 10s (continuation should dispatch fast)
    """
    last_event_time = await self._get_last_event_time(task.id)
    elapsed = (datetime.utcnow() - last_event_time).total_seconds()

    stall_threshold = self._get_stall_threshold(task)

    if elapsed > stall_threshold:
        if task.attempt_number < task.max_retries:
            await self._transition_task(
                task, TaskState.AWAITING_RETRY,
                reason=f"Stalled for {elapsed:.0f}s (threshold: {stall_threshold}s)"
            )
        else:
            await self._transition_task(
                task, TaskState.FAILED,
                reason=f"Stalled for {elapsed:.0f}s, max retries ({task.max_retries}) exhausted"
            )
```

### 4.6 Concurrency Safety

Multiple coordinator ticks could overlap if a tick takes longer than the interval. Two tasks could complete simultaneously, both triggering dependency resolution for the same downstream task.

**Solution: Optimistic locking with version column.**

PRD-101 defines `version` on `orchestration_tasks`. Every state transition includes `WHERE version = :expected_version`. If the version changed (another process already transitioned the task), the UPDATE affects 0 rows and the transition is skipped.

```python
async def _transition_task(
    self, task: OrchestrationTask, new_state: TaskState, reason: str
) -> bool:
    """
    Transition task state with optimistic locking.
    Returns True if transition succeeded, False if version conflict.
    """
    result = await self._db.execute(
        update(OrchestrationTask)
        .where(
            OrchestrationTask.id == task.id,
            OrchestrationTask.version == task.version,  # Optimistic lock
        )
        .values(
            state=new_state,
            state_type=TASK_STATE_TYPE[new_state],
            version=task.version + 1,
            updated_at=datetime.utcnow(),
        )
    )

    if result.rowcount == 0:
        # Version conflict — another process already transitioned this task
        return False

    # Append to event log (dual-write pattern from PRD-101)
    await self._emit_event(task.run_id, f"task_{new_state}", {
        "task_id": task.id,
        "reason": reason,
        "from_state": task.state,
        "to_state": new_state,
    })
    return True
```

---

## 5. Plan Decomposition

### 5.1 Decomposition Pipeline

```
User: "Research EU AI Act compliance for our product"
         │
         ▼
   ┌─────────────────────────────────────┐
   │ Step 1: Template Match              │
   │  Search template library by         │
   │  semantic similarity (embedding)    │
   │  Threshold: cosine_sim > 0.85       │
   │  If match → use template, go to 3   │
   │  If no match → go to Step 2         │
   └──────────────┬──────────────────────┘
                  │ no match
                  ▼
   ┌─────────────────────────────────────┐
   │ Step 2: LLM Decomposition           │
   │  Call coordinator LLM with:         │
   │  - Goal text                        │
   │  - Available agents (roster)        │
   │  - Available tools (workspace)      │
   │  - Success criteria (if provided)   │
   │  - Budget constraints               │
   │  Output: task graph as JSON         │
   └──────────────┬──────────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────┐
   │ Step 3: Structural Validation       │
   │  - DAG check (no cycles)            │
   │  - All referenced agents exist      │
   │  - All referenced tools available   │
   │  - Task count within bounds (1-20)  │
   │  - Budget estimate within limits    │
   │  - Success criteria parseable       │
   │  If fails → retry LLM (max 2)      │
   └──────────────┬──────────────────────┘
                  │ valid
                  ▼
   ┌─────────────────────────────────────┐
   │ Step 4: Write to DB                 │
   │  - Create orchestration_tasks rows  │
   │  - Create dependency edges          │
   │  - Emit plan_ready event            │
   │  - Transition run to                │
   │    awaiting_approval or running     │
   └─────────────────────────────────────┘
```

### 5.2 MissionPlanner Interface

```python
@dataclass(frozen=True)
class DecompositionResult:
    """Immutable result from plan decomposition."""
    tasks: list[TaskSpec]
    dependencies: list[DependencyEdge]
    estimated_cost_usd: float
    estimated_tokens: int
    planning_model: str
    planning_tokens_used: int
    template_used: Optional[str]  # None if LLM-generated


@dataclass(frozen=True)
class TaskSpec:
    """Specification for a single task in the decomposition."""
    task_order: int
    title: str
    description: str
    task_type: str  # research, writing, coding, review, analysis, simple
    success_criteria: list[dict]  # [{criterion: str, weight: float, must_pass: bool}]
    suggested_agent_id: Optional[int]  # None → coordinator assigns
    suggested_model: Optional[str]  # Role-based default if None
    required_tools: list[str]  # Tool names needed for this task
    max_retries: int  # Default: 3
    stall_timeout_s: int  # Default: 300
    estimated_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class DependencyEdge:
    """Dependency between two tasks."""
    from_task_order: int  # This task...
    to_task_order: int    # ...depends on this task
    trigger_rule: str     # "all_success" (default) or "all_done"


class MissionPlanner:
    """
    Decomposes natural language goals into task graphs.
    Uses HTN-inspired hybrid: template matching + LLM generation + validation.
    """

    async def decompose(
        self,
        goal: str,
        workspace_id: uuid.UUID,
        available_agents: list[AgentSummary],
        config: MissionConfig,
    ) -> DecompositionResult:
        """
        Decompose a goal into a validated task graph.

        1. Try template match (fast, deterministic)
        2. Fall back to LLM decomposition (flexible, non-deterministic)
        3. Validate structural correctness
        4. Estimate costs

        Raises PlanValidationError if validation fails after max retries.
        """
        ...

    async def replan(
        self,
        run: OrchestrationRun,
        trigger: ReplanTrigger,
        context: ReplanContext,
    ) -> DecompositionResult:
        """
        Revise an existing plan. Only modifies pending/scheduled tasks.
        Completed tasks are immutable.

        Increments run.plan_version and emits PLAN_REVISED event.
        """
        ...
```

### 5.3 Decomposition Templates

Templates are Python dataclasses registered in a template library. They provide structural scaffolding that the LLM customizes with mission-specific details.

```python
@dataclass(frozen=True)
class DecompositionTemplate:
    """HTN-inspired decomposition template."""
    name: str
    description: str
    embedding: Optional[list[float]]  # For semantic matching
    keywords: list[str]  # For keyword matching fallback
    task_specs: list[TaskSpec]  # Template tasks (descriptions are templates with {goal} placeholders)
    dependencies: list[DependencyEdge]
    min_complexity: int  # Minimum task count this template applies to
    max_complexity: int  # Maximum task count


# Example: Research and Report template
RESEARCH_AND_REPORT = DecompositionTemplate(
    name="research_and_report",
    description="Research a topic and produce an analysis report",
    keywords=["research", "report", "analysis", "investigate", "study", "compliance"],
    task_specs=[
        TaskSpec(
            task_order=1,
            title="Research: {topic}",
            description="Gather comprehensive information about {topic}. "
                        "Use web search and document analysis tools. "
                        "Produce structured findings with sources.",
            task_type="research",
            success_criteria=[
                {"criterion": "Covers all major aspects of {topic}", "weight": 0.4, "must_pass": True},
                {"criterion": "All claims cite specific sources", "weight": 0.3, "must_pass": False},
                {"criterion": "Findings are structured with clear categories", "weight": 0.3, "must_pass": False},
            ],
            suggested_agent_id=None,
            suggested_model=None,
            required_tools=["platform_search_web", "workspace_read_file"],
            max_retries=3,
            stall_timeout_s=600,  # Research tasks can take longer
            estimated_tokens=50_000,
            estimated_cost_usd=0.50,
        ),
        TaskSpec(
            task_order=2,
            title="Analyze: {topic}",
            description="Analyze research findings and identify key patterns, "
                        "gaps, and recommendations. Cross-reference multiple sources.",
            task_type="analysis",
            success_criteria=[
                {"criterion": "Identifies at least 3 key findings", "weight": 0.4, "must_pass": True},
                {"criterion": "Recommendations are actionable", "weight": 0.3, "must_pass": False},
                {"criterion": "Analysis is grounded in research data", "weight": 0.3, "must_pass": True},
            ],
            suggested_agent_id=None,
            suggested_model=None,
            required_tools=["workspace_read_file"],
            max_retries=3,
            stall_timeout_s=300,
            estimated_tokens=30_000,
            estimated_cost_usd=0.30,
        ),
        TaskSpec(
            task_order=3,
            title="Write report: {topic}",
            description="Produce a professional report combining research findings "
                        "and analysis. Include executive summary, detailed findings, "
                        "and recommendations.",
            task_type="writing",
            success_criteria=[
                {"criterion": "Has Executive Summary, Findings, and Recommendations sections", "weight": 0.3, "must_pass": True},
                {"criterion": "Writing quality score >= 3.5/5", "weight": 0.3, "must_pass": False},
                {"criterion": "All analysis findings are reflected in report", "weight": 0.4, "must_pass": True},
            ],
            suggested_agent_id=None,
            suggested_model=None,
            required_tools=["workspace_write_file"],
            max_retries=2,
            stall_timeout_s=300,
            estimated_tokens=40_000,
            estimated_cost_usd=0.40,
        ),
        TaskSpec(
            task_order=4,
            title="Review report: {topic}",
            description="Review the final report for accuracy, completeness, "
                        "and quality. Flag any issues.",
            task_type="review",
            success_criteria=[
                {"criterion": "Identifies any factual errors", "weight": 0.5, "must_pass": True},
                {"criterion": "Assesses completeness against original goal", "weight": 0.5, "must_pass": True},
            ],
            suggested_agent_id=None,
            suggested_model=None,  # PRD-104: reviewer MUST use different model family
            required_tools=["workspace_read_file"],
            max_retries=1,
            stall_timeout_s=300,
            estimated_tokens=20_000,
            estimated_cost_usd=0.20,
        ),
    ],
    dependencies=[
        DependencyEdge(from_task_order=2, to_task_order=1, trigger_rule="all_success"),
        DependencyEdge(from_task_order=3, to_task_order=2, trigger_rule="all_success"),
        DependencyEdge(from_task_order=4, to_task_order=3, trigger_rule="all_success"),
    ],
    min_complexity=3,
    max_complexity=6,
)
```

### 5.4 LLM Decomposition Prompt

When no template matches, the coordinator calls an LLM to generate the decomposition. The prompt is structured to produce valid JSON matching the `TaskSpec` schema.

```python
COORDINATOR_DECOMPOSITION_PROMPT = """You are a mission coordinator for the Automatos AI platform.

## Your Task
Decompose the following goal into 3-20 subtasks with dependency edges.

## Goal
{goal}

## Available Agents
{agent_roster}

## Available Tools
{available_tools}

## Budget Constraints
Max cost: ${max_cost_usd}
Max tokens: {max_tokens}

## Output Format
Respond with a JSON object matching this schema exactly:
```json
{{
  "tasks": [
    {{
      "task_order": 1,
      "title": "Short descriptive title",
      "description": "Detailed instructions for the agent",
      "task_type": "research|writing|coding|review|analysis|simple",
      "success_criteria": [
        {{"criterion": "Specific, measurable criterion", "weight": 0.5, "must_pass": true}}
      ],
      "suggested_agent_id": null,
      "required_tools": ["tool_name"],
      "estimated_tokens": 30000,
      "estimated_cost_usd": 0.30
    }}
  ],
  "dependencies": [
    {{"from_task_order": 2, "to_task_order": 1, "trigger_rule": "all_success"}}
  ]
}}
```

## Rules
1. Tasks MUST form a valid DAG (no circular dependencies)
2. Task 1 should have no dependencies (the starting point)
3. Every task needs at least one success criterion with must_pass=true
4. Use task_type to guide model selection (research=mid-tier, review=different-family)
5. Keep task count proportional to goal complexity (simple goal = 3-4 tasks)
6. Independent tasks CAN run in parallel (no dependency edge between them)
7. Estimated costs must sum to less than the budget constraint
"""
```

### 5.5 Plan Validation

```python
async def _validate_plan(
    self, result: DecompositionResult, workspace_id: uuid.UUID
) -> list[ValidationError]:
    """
    Structural validation of a decomposition.
    Returns empty list if valid, list of errors otherwise.
    """
    errors = []

    # 1. DAG check (no cycles)
    try:
        graph = TopologicalSorter()
        for dep in result.dependencies:
            graph.add(dep.from_task_order, dep.to_task_order)
        graph.prepare()  # Raises CycleError if cycles exist
    except CycleError as e:
        errors.append(ValidationError("CYCLE", str(e)))

    # 2. Task count bounds
    if not (1 <= len(result.tasks) <= 20):
        errors.append(ValidationError(
            "TASK_COUNT", f"Got {len(result.tasks)} tasks, expected 1-20"
        ))

    # 3. All referenced agents exist
    for task in result.tasks:
        if task.suggested_agent_id is not None:
            agent = await self._get_agent(task.suggested_agent_id, workspace_id)
            if agent is None:
                errors.append(ValidationError(
                    "AGENT_NOT_FOUND", f"Agent {task.suggested_agent_id} not found"
                ))

    # 4. All referenced tools available
    available_tools = await self._get_available_tools(workspace_id)
    for task in result.tasks:
        for tool in task.required_tools:
            if tool not in available_tools:
                errors.append(ValidationError(
                    "TOOL_NOT_FOUND", f"Tool '{tool}' not available"
                ))

    # 5. At least one success criterion per task with must_pass=True
    for task in result.tasks:
        if not any(c.get("must_pass") for c in task.success_criteria):
            errors.append(ValidationError(
                "NO_MUST_PASS", f"Task {task.sequence_number} has no must_pass criterion"
            ))

    # 6. Budget estimate within limits
    total_estimated = sum(t.estimated_cost_usd for t in result.tasks)
    if total_estimated > result.estimated_cost_usd * 1.5:  # 50% buffer
        errors.append(ValidationError(
            "BUDGET_ESTIMATE", f"Task estimates (${total_estimated:.2f}) exceed budget"
        ))

    # 7. Task orders are unique and sequential
    orders = [t.sequence_number for t in result.tasks]
    if len(orders) != len(set(orders)):
        errors.append(ValidationError("DUPLICATE_ORDER", "Task orders must be unique"))

    return errors
```

---

## 6. Agent Assignment

### 6.1 Assignment Strategy

For each task in the plan, the coordinator assigns an agent using a deterministic scoring algorithm — not LLM-based selection (CrewAI's approach, which is non-deterministic and untestable).

| Strategy | When Used |
|---|---|
| **Roster match** | Task requirements match a roster agent's skills/tools. Preferred — agent has memory, personality, history. |
| **Contractor spawn** | No roster agent scores above threshold, or task needs a specialist model not available on roster. Ephemeral — mission-scoped lifecycle (PRD-104). |
| **User override** | In `approve` mode, user can reassign agents before execution starts. |

### 6.2 Scoring Algorithm

```python
class AgentMatcher:
    """
    Deterministic agent scoring for task assignment.
    Returns ranked list of (agent_id, score) pairs.
    """

    async def score_agents(
        self,
        task: TaskSpec,
        available_agents: list[AgentSummary],
        mission_context: MissionContext,
    ) -> list[tuple[int, float]]:
        """
        Score each agent against task requirements.

        Score components (0.0 - 1.0 each):
        - tool_coverage: % of required tools the agent has
        - skill_match: semantic similarity of agent skills to task_type
        - model_fit: does agent's model match the role tier?
        - availability: is agent currently idle vs overloaded?
        - history: agent's past verifier_score on similar task_types

        Final score = weighted sum, normalized to 0.0-1.0.
        """
        scored = []
        for agent in available_agents:
            tool_score = self._score_tool_coverage(task.required_tools, agent.tools)
            skill_score = self._score_skill_match(task.task_type, agent.skills)
            model_score = self._score_model_fit(task.task_type, agent.model)
            avail_score = self._score_availability(agent)
            history_score = await self._score_history(agent.id, task.task_type)

            final = (
                tool_score * 0.35     # Tools are most important — can't execute without them
                + skill_score * 0.25  # Skill alignment matters for quality
                + model_score * 0.15  # Right model tier for the task type
                + avail_score * 0.10  # Prefer idle agents
                + history_score * 0.15  # Past performance predicts future
            )
            scored.append((agent.id, final))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _score_tool_coverage(
        self, required: list[str], agent_tools: list[str]
    ) -> float:
        """Fraction of required tools the agent has. 1.0 = all tools present."""
        if not required:
            return 1.0
        return len(set(required) & set(agent_tools)) / len(required)

    def _score_model_fit(self, task_type: str, agent_model: str) -> float:
        """
        Score model appropriateness for the task type.
        Uses the role→tier mapping from PRD-104.
        """
        expected_tier = ROLE_MODEL_TIERS.get(task_type, "mid")
        actual_tier = self._classify_model_tier(agent_model)

        if actual_tier == expected_tier:
            return 1.0
        elif actual_tier == "top" and expected_tier == "mid":
            return 0.8  # Over-qualified but works
        elif actual_tier == "mid" and expected_tier == "top":
            return 0.5  # Under-qualified
        elif actual_tier == "cheap":
            return 0.3 if expected_tier == "simple" else 0.2
        return 0.5

    async def _score_history(self, agent_id: int, task_type: str) -> float:
        """
        Average verifier_score for this agent on similar task types.
        Returns 0.5 (neutral) if no history.
        """
        avg_score = await self._db.execute(
            select(func.avg(OrchestrationTask.verifier_score))
            .where(
                OrchestrationTask.assigned_agent_id == agent_id,
                OrchestrationTask.task_type == task_type,
                OrchestrationTask.verifier_score.isnot(None),
            )
        )
        return avg_score.scalar() or 0.5


# Minimum score to use a roster agent. Below this → spawn contractor.
ROSTER_MATCH_THRESHOLD = 0.4
```

### 6.3 Dispatch Mechanism

The coordinator dispatches tasks **directly** via `AgentFactory.execute_with_prompt()`. It does NOT create a `BoardTask` and wait for the agent's heartbeat tick to pick it up. Direct dispatch gives the coordinator control over timing, retry, and result collection.

A `BoardTask` is created **for visibility** (kanban tracking) but is NOT the dispatch mechanism.

```python
class MissionDispatcher:
    """Dispatches mission tasks to agents."""

    async def dispatch_task(
        self, run: OrchestrationRun, task: OrchestrationTask
    ) -> None:
        """
        Assign agent and dispatch for execution.

        1. Select agent (roster match or contractor spawn)
        2. Create board task for visibility
        3. Transition task to 'assigned'
        4. Launch execution via AgentFactory (async, non-blocking)
        """
        # 1. Agent selection
        agent_id = task.assigned_agent_id  # May be pre-assigned from plan
        if agent_id is None:
            agent_id = await self._select_agent(run, task)

        # 2. Create board task for kanban visibility
        board_task = await self._create_board_task(run, task, agent_id)
        task.board_task_id = board_task.id

        # 3. Transition to assigned
        await self._transition_task(task, TaskState.ASSIGNED)

        # 4. Build context and dispatch (fire-and-forget, result collected by reconciler)
        #
        # CRASH RECOVERY: Task is ASSIGNED in DB before this coroutine starts.
        # If coordinator crashes between ASSIGNED and execute_with_prompt():
        #   → Reconciler detects ASSIGNED task with no progress events
        #   → After stall timeout (60s), re-dispatches to same or different agent
        # The DB is the source of truth — the in-memory coroutine is expendable.
        asyncio.create_task(
            self._execute_task(run, task, agent_id)
        )

    async def _execute_task(
        self, run: OrchestrationRun, task: OrchestrationTask, agent_id: int
    ) -> None:
        """
        Execute a task via AgentFactory. Handles result collection
        and state transitions on completion/failure.
        """
        try:
            await self._transition_task(task, TaskState.RUNNING)

            # Build mission-aware context
            prior_results = await self._get_completed_task_outputs(run.id, task.id)

            prompt = self._build_task_prompt(task, prior_results)

            result = await self._agent_factory.execute_with_prompt(
                agent_id=agent_id,
                prompt=prompt,
                context_mode="task_execution",
                workspace_id=run.workspace_id,
                mission_context={
                    "mission_id": run.id,
                    "mission_goal": run.goal,
                    "task_id": task.id,
                    "task_title": task.title,
                    "success_criteria": task.success_criteria,
                    "prior_task_outputs": prior_results,
                },
            )

            # Store result and advance to verification
            task.result_reference = await self._store_task_output(run, task, result)
            await self._transition_task(task, TaskState.VERIFYING)

        except Exception as e:
            # Infrastructure failure
            if task.attempt_number < task.max_retries:
                await self._transition_task(
                    task, TaskState.AWAITING_RETRY,
                    reason=f"Infrastructure failure: {str(e)[:500]}"
                )
            else:
                await self._transition_task(
                    task, TaskState.FAILED,
                    reason=f"Infrastructure failure after {task.max_retries} retries: {str(e)[:500]}"
                )
```

---

## 7. ContextMode.COORDINATOR

### 7.1 New Context Mode Definition

The coordinator needs its own context mode to get mission-aware context when making planning and monitoring decisions.

```python
# In orchestrator/modules/context/modes.py

COORDINATOR = ContextMode(
    name="coordinator",
    description="Mission coordinator — full context for planning and monitoring",
    sections=[
        "identity",           # Coordinator agent identity
        "mission_context",    # NEW: current mission state, task statuses, agent assignments
        "agent_roster",       # NEW: available agents with capabilities and recent performance
        "platform_actions",   # Full platform tools including mission management
        "task_context",       # Current tick focus: which tasks need attention
        "datetime_context",   # Current time for scheduling
    ],
    token_budget=0,           # No cap — coordinator needs full context
    tool_loading=ToolLoadingStrategy.FULL,  # All tools available
)
```

### 7.2 New Sections

#### MissionContextSection

```python
class MissionContextSection(BaseSection):
    """Renders current mission state for the coordinator."""
    name = "mission_context"
    priority = 2  # After identity, before everything else

    async def render(self, ctx: SectionContext) -> str:
        run = ctx.kwargs.get("mission_run")
        if not run:
            return ""

        tasks = await self._get_mission_tasks(run.id)

        lines = [
            f"## Current Mission",
            f"**Goal:** {run.goal}",
            f"**Status:** {run.state}",
            f"**Plan version:** {run.plan_version}",
            f"**Budget:** ${run.budget_spent.get('cost_usd', 0):.2f} / "
            f"${run.budget_config.get('max_cost_usd', 'unlimited')}",
            "",
            "### Task Status",
        ]

        for task in tasks:
            status_emoji = {
                "pending": "⏳", "queued": "📋", "assigned": "👤",
                "running": "▶️", "continuing": "🔄", "verifying": "🔍",
                "awaiting_human": "🧑", "awaiting_retry": "⏰",
                "completed": "✅", "failed": "❌", "cancelled": "🚫",
                "skipped": "⏭️",
            }.get(task.state, "❓")

            dep_info = ""
            if task.state == "pending":
                deps = await self._get_pending_deps(task.id)
                dep_info = f" (waiting for: {', '.join(d.title for d in deps)})"

            lines.append(
                f"  {status_emoji} T{task.sequence_number}: {task.title} "
                f"[{task.state}]{dep_info}"
            )

            if task.verifier_score is not None:
                lines.append(f"     Score: {task.verifier_score:.2f}")

        return "\n".join(lines)
```

#### AgentRosterSection

```python
class AgentRosterSection(BaseSection):
    """Renders available agents for the coordinator's planning decisions."""
    name = "agent_roster"
    priority = 4

    async def render(self, ctx: SectionContext) -> str:
        agents = await self._get_workspace_agents(ctx.workspace_id)

        lines = ["## Available Agents"]
        for agent in agents:
            tools = await self._get_agent_tools(agent.id)
            lines.append(
                f"- **{agent.name}** (ID: {agent.id})"
                f"\n  Model: {agent.model_config.get('model', 'default')}"
                f"\n  Skills: {', '.join(agent.skills or [])}"
                f"\n  Tools: {', '.join(t.name for t in tools[:10])}"
                f"{'...' if len(tools) > 10 else ''}"
            )

        return "\n".join(lines)
```

### 7.3 Files That Must Be Modified

| File | Change |
|---|---|
| `orchestrator/modules/context/modes.py` | Add `COORDINATOR` and `VERIFIER` to `ContextMode` enum and `MODE_CONFIGS` |
| `orchestrator/modules/context/service.py` | Register `MissionContextSection` and `AgentRosterSection` section renderers |
| `orchestrator/modules/context/sections/` | New files: `mission_context.py`, `agent_roster.py` |

---

## 8. Failure Handling

### 8.1 Decision Tree

```
Task execution completes
    │
    ├─ Clean exit, work done → output submitted
    │   └─ Transition to VERIFYING → PRD-103 evaluates
    │       ├─ Verifier PASS → COMPLETED
    │       ├─ Verifier PARTIAL + retries remaining → AWAITING_RETRY (with feedback)
    │       ├─ Verifier FAIL + retries remaining → AWAITING_RETRY (with feedback)
    │       ├─ Verifier FAIL + no retries → FAILED
    │       └─ Verifier uncertain (low confidence) → AWAITING_HUMAN
    │
    ├─ Clean exit, needs more turns → CONTINUING
    │   └─ 1s delay → re-dispatch (same attempt, same agent)
    │
    ├─ Infrastructure failure (timeout, crash, OOM)
    │   ├─ Retries remaining → AWAITING_RETRY
    │   │   └─ Backoff: min(10s × 2^(attempt-1), 5min)
    │   └─ No retries → escalate
    │       ├─ Different agent available → reassign + retry
    │       ├─ Different model available → swap model + retry
    │       └─ All alternatives exhausted → FAILED
    │
    └─ Budget exceeded mid-task
        └─ Complete current LLM call (K8s pattern: in-flight completes)
            └─ Run transitions to BUDGET_EXCEEDED
                └─ Human decides: increase budget, downgrade models, or cancel
```

### 8.2 Retry-with-Feedback Protocol

When verification fails but retries remain, the verifier's reasoning is fed back to the executing agent. This is a continuation with guidance, not a blind retry.

```python
async def _retry_with_feedback(
    self, run: OrchestrationRun, task: OrchestrationTask,
    verification_result: VerificationResult
) -> None:
    """
    Schedule a retry with verifier feedback injected into the agent's prompt.
    """
    feedback_context = {
        "previous_attempt": task.attempt_number,
        "verifier_feedback": verification_result.reasoning,
        "failed_criteria": [
            c for c in verification_result.criteria_scores
            if not c["met"]
        ],
        "aggregate_score": verification_result.aggregate_score,
    }

    # Update task with feedback for next attempt
    task.retry_context = feedback_context
    task.attempt_number += 1

    # Calculate backoff delay
    delay_s = min(10 * (2 ** (task.attempt_number - 1)), 300)

    await self._transition_task(
        task, TaskState.AWAITING_RETRY,
        reason=f"Verification failed (score: {verification_result.aggregate_score:.2f}). "
               f"Retry {task.attempt_number}/{task.max_retries} in {delay_s}s."
    )

    # Schedule retry after backoff
    await self._schedule_retry(task, delay_s)
```

### 8.3 Escalation Strategy

When a task fails after max retries:

| Escalation Level | Action | When |
|---|---|---|
| 1. Different agent | Reassign to next-best-scoring agent | Default |
| 2. Different model | Keep same agent, switch to higher-tier model | If agent-specific issue unlikely |
| 3. Coordinator replanning | Remove failed task, find alternative path | If task is on critical path |
| 4. Human escalation | Flag for human review with full context | All automated options exhausted |
| 5. Mission failure | Mark run as failed, cancel remaining tasks | Human rejects or no alternatives |

---

## 9. Replanning Specification

### 9.1 Triggers

| Trigger | Action | Constraint |
|---|---|---|
| Task fails after max retries + all escalations | Replan: find alternative path or substitute task | Completed tasks immutable |
| User sends new instructions mid-mission | Replan: incorporate new requirements | Completed tasks immutable |
| Budget warning (>80% spent) | Replan: cut optional tasks, use cheaper models | Running tasks continue |
| Verification rejects task + coordinator determines task design is wrong | Replan: redesign the task, not just retry | Only pending/queued tasks modified |
| Agent discovers new information requiring additional work | Replan: add tasks dynamically | New tasks get new task_order values |

### 9.2 Replanning Constraints

1. **Completed tasks are immutable.** Their outputs are already consumed by downstream tasks. Removing them would invalidate the dependency graph.
2. **Running tasks continue.** Only cancel running tasks if explicitly directed by human or if budget is exhausted.
3. **Plan version increments.** Every replan bumps `orchestration_runs.plan_version` for audit trail.
4. **New tasks get the next available task_order.** No renumbering of existing tasks.
5. **Dependency graph must remain a valid DAG.** Validated after every replan.

### 9.3 Replanning LLM Prompt

```python
COORDINATOR_REPLAN_PROMPT = """You are revising a mission plan.

## Original Goal
{goal}

## Current Plan State
{current_plan_state}

## Completed Tasks (IMMUTABLE — do not modify)
{completed_tasks}

## Replan Trigger
{trigger_description}

## Instructions
1. You may ONLY modify tasks with status 'pending' or 'queued'
2. You may ADD new tasks (with task_order > {max_existing_order})
3. You may REMOVE pending tasks that are no longer needed
4. You may CHANGE the description, tools, or success criteria of pending tasks
5. You MUST NOT touch completed or running tasks
6. The dependency graph MUST remain a valid DAG

## Output Format
Same JSON schema as the original decomposition, but only include
modified/new tasks and updated dependency edges.
"""
```

---

## 10. API Endpoints

### 10.1 Mission CRUD

```
POST   /api/missions                    Create mission
GET    /api/missions                    List missions (workspace-scoped)
GET    /api/missions/{id}               Get mission details + task graph
DELETE /api/missions/{id}               Cancel mission (soft delete)
```

### 10.2 Mission Lifecycle

```
POST   /api/missions/{id}/approve       Approve plan (with optional modifications)
POST   /api/missions/{id}/reject        Reject plan
POST   /api/missions/{id}/pause         Pause execution
POST   /api/missions/{id}/resume        Resume execution
POST   /api/missions/{id}/review        Submit review verdict (accept/reject/rework)
POST   /api/missions/{id}/budget        Update budget (increase/decrease)
```

### 10.3 Task Operations

```
GET    /api/missions/{id}/tasks         List tasks with statuses and dependencies
GET    /api/missions/{id}/tasks/{tid}   Get task detail including output and verification
POST   /api/missions/{id}/tasks/{tid}/retry   Manual retry of a failed task
POST   /api/missions/{id}/tasks/{tid}/skip    Skip a pending task
```

### 10.4 Request/Response Examples

#### Create Mission

```json
// POST /api/missions
{
    "goal": "Research EU AI Act compliance for our product",
    "config": {
        "autonomy_mode": "approve",
        "max_cost_usd": 5.00,
        "max_concurrent_tasks": 3,
        "model_preferences": {
            "research": "anthropic/claude-sonnet-4-20250514",
            "review": "openai/gpt-4o"
        }
    }
}

// Response: 201 Created
{
    "id": 42,
    "goal": "Research EU AI Act compliance for our product",
    "state": "planning",
    "plan_version": 0,
    "created_at": "2026-03-15T10:30:00Z",
    "estimated_cost_usd": null,
    "tasks": []
}
```

#### Plan Ready (webhook or poll)

```json
// GET /api/missions/42
{
    "id": 42,
    "goal": "Research EU AI Act compliance for our product",
    "state": "awaiting_approval",
    "plan_version": 1,
    "estimated_cost_usd": 2.40,
    "estimated_tokens": 140000,
    "tasks": [
        {
            "task_order": 1,
            "title": "Research EU AI Act requirements",
            "task_type": "research",
            "state": "pending",
            "dependencies": [],
            "assigned_agent": {"id": 12, "name": "Researcher"},
            "estimated_cost_usd": 0.50,
            "success_criteria": [...]
        },
        {
            "task_order": 2,
            "title": "Analyze product against requirements",
            "task_type": "analysis",
            "state": "pending",
            "dependencies": [1],
            "assigned_agent": {"id": 15, "name": "Analyst"},
            "estimated_cost_usd": 0.30
        },
        // ... more tasks
    ]
}
```

---

## 11. Sequence Diagrams

### 11.1 Happy Path: 3-Task Sequential Mission

```
User          CoordinatorService    MissionPlanner    MissionDispatcher    AgentFactory    VerificationService
  │                  │                    │                  │                  │                  │
  │ POST /missions   │                    │                  │                  │                  │
  ├─────────────────►│                    │                  │                  │                  │
  │                  │ decompose(goal)    │                  │                  │                  │
  │                  ├───────────────────►│                  │                  │                  │
  │                  │  DecompositionResult│                  │                  │                  │
  │                  │◄───────────────────┤                  │                  │                  │
  │                  │ state=awaiting_approval               │                  │                  │
  │ 201 Created      │                    │                  │                  │                  │
  │◄─────────────────┤                    │                  │                  │                  │
  │                  │                    │                  │                  │                  │
  │ POST /approve    │                    │                  │                  │                  │
  ├─────────────────►│                    │                  │                  │                  │
  │                  │ state=running       │                  │                  │                  │
  │                  │                    │                  │                  │                  │
  │                  │─── tick() ──────── │                  │                  │                  │
  │                  │ Dispatch Phase:     │                  │                  │                  │
  │                  │ Task 1 ready        │ dispatch_task(1) │                  │                  │
  │                  │                    │─────────────────►│                  │                  │
  │                  │                    │                  │ execute_with_prompt(agent=12)        │
  │                  │                    │                  │─────────────────►│                  │
  │                  │                    │                  │    result         │                  │
  │                  │                    │                  │◄─────────────────┤                  │
  │                  │                    │                  │ Task 1 → verifying│                  │
  │                  │                    │                  │                  │ verify_task(1)   │
  │                  │                    │                  │                  │─────────────────►│
  │                  │                    │                  │                  │   PASS (0.85)    │
  │                  │                    │                  │                  │◄─────────────────┤
  │                  │                    │                  │ Task 1 → completed│                  │
  │                  │                    │                  │                  │                  │
  │                  │ _on_task_completed(1)                 │                  │                  │
  │                  │ Task 2 deps met → queued              │                  │                  │
  │                  │                    │                  │                  │                  │
  │                  │─── tick() ──────── │                  │                  │                  │
  │                  │ Task 2 dispatched → execute → verify → completed         │                  │
  │                  │                    │                  │                  │                  │
  │                  │─── tick() ──────── │                  │                  │                  │
  │                  │ Task 3 dispatched → execute → verify → completed         │                  │
  │                  │                    │                  │                  │                  │
  │                  │ All tasks completed │                  │                  │                  │
  │                  │ state=awaiting_review                  │                  │                  │
  │ notification     │                    │                  │                  │                  │
  │◄─────────────────┤                    │                  │                  │                  │
  │                  │                    │                  │                  │                  │
  │ POST /review     │                    │                  │                  │                  │
  │ {verdict:accept} │                    │                  │                  │                  │
  ├─────────────────►│                    │                  │                  │                  │
  │                  │ state=completed     │                  │                  │                  │
  │                  │ "Save as routine?"  │                  │                  │                  │
  │ 200 OK           │                    │                  │                  │                  │
  │◄─────────────────┤                    │                  │                  │                  │
```

### 11.2 Mission with Task Failure and Retry

```
CoordinatorService    MissionDispatcher    AgentFactory    VerificationService
       │                     │                  │                  │
       │ Task 2 dispatched   │                  │                  │
       ├────────────────────►│                  │                  │
       │                     │ execute(agent=15)│                  │
       │                     ├─────────────────►│                  │
       │                     │   result          │                  │
       │                     │◄─────────────────┤                  │
       │                     │ Task 2 → verifying│                  │
       │                     │                  │ verify(task=2)   │
       │                     │                  ├─────────────────►│
       │                     │                  │ FAIL (0.35)      │
       │                     │                  │ "Missing 3 of 5  │
       │                     │                  │  required sections│
       │                     │                  │◄─────────────────┤
       │                     │                  │                  │
       │ retry_with_feedback │                  │                  │
       │ attempt 2/3         │                  │                  │
       │ delay: 10s          │                  │                  │
       │                     │                  │                  │
       │ ─── 10s later ───   │                  │                  │
       │                     │                  │                  │
       │ Task 2 dispatched   │                  │                  │
       │ (with feedback)     │                  │                  │
       ├────────────────────►│                  │                  │
       │                     │ execute(agent=15,│                  │
       │                     │  feedback=...)   │                  │
       │                     ├─────────────────►│                  │
       │                     │   improved result │                  │
       │                     │◄─────────────────┤                  │
       │                     │ Task 2 → verifying│                  │
       │                     │                  │ verify(task=2)   │
       │                     │                  ├─────────────────►│
       │                     │                  │ PASS (0.78)      │
       │                     │                  │◄─────────────────┤
       │                     │ Task 2 → completed│                  │
       │                     │                  │                  │
```

### 11.3 Mission with Human Review Rejection

```
User          CoordinatorService    MissionDispatcher
  │                  │                    │
  │ (Tasks 1-3 completed, mission in awaiting_review)
  │                  │                    │
  │ POST /review     │                    │
  │ {verdict: reject,│                    │
  │  task_feedback: [│                    │
  │   {task_id: 3,   │                    │
  │    reason: "Report│                   │
  │    is too surface │                   │
  │    level"}       │                    │
  │  ]}              │                    │
  ├─────────────────►│                    │
  │                  │                    │
  │                  │ Task 3 → awaiting_retry
  │                  │ (with human feedback)
  │                  │ Tasks 1,2 stay completed
  │                  │ Run stays running
  │                  │                    │
  │                  │─── tick() ──────── │
  │                  │ Task 3 retry (attempt 2)
  │                  │ Prompt includes:    │
  │                  │  "Previous attempt  │
  │                  │   rejected by human.│
  │                  │   Feedback: Report  │
  │                  │   is too surface    │
  │                  │   level"           │
  │                  ├───────────────────►│
  │                  │                    │ execute → verify → completed
  │                  │                    │
  │                  │ All tasks completed again
  │                  │ state=awaiting_review
  │                  │                    │
  │ POST /review     │                    │
  │ {verdict: accept}│                    │
  ├─────────────────►│                    │
  │                  │ state=completed    │
  │ 200 OK           │                    │
  │◄─────────────────┤                    │
```

---

## 12. Integration Points

### 12.1 Existing Components Used

| Component | How Coordinator Uses It | Changes Required |
|---|---|---|
| `AgentFactory.execute_with_prompt()` | Dispatches each task to its assigned agent | None — accepts `AgentRuntime` already |
| `ContextService.build_context()` | Coordinator builds its own context with `ContextMode.COORDINATOR` | Add new mode + 2 new sections |
| `get_tools_for_agent()` (`tool_router.py:~140`) | Resolves tools for task agents | None for roster agents; PRD-104 adds `explicit_tools` param for contractors |
| `UnifiedToolExecutor.execute_tool()` | Coordinator's own tool loop for mission management | None |
| `BoardTask` model (`core/models/board.py`) | Creates board tasks with `source_type='orchestration'` for kanban visibility | None — existing model supports this |
| `TaskReconciler` (`services/task_reconciler.py`) | Extended to cover `orchestration_tasks` alongside `recipe_executions` | Add mission task query to `_tick()` |
| `SharedContextManager` (`inter_agent.py`) | Stores mission-scoped shared context for cross-task data flow | None — used via `SharedContextPort` (PRD-107) |
| `UnifiedScheduler` | Registers coordinator tick alongside heartbeat tick | None — additive registration |
| `workflow_recipes` table | "Save as routine" converts mission structure to recipe | Conversion function (new) |

### 12.2 New Components Introduced

| Component | Purpose | Location |
|---|---|---|
| `CoordinatorService` | Main service: tick loop, plan generation, dispatch, reconciliation | `orchestrator/services/coordinator_service.py` |
| `MissionPlanner` | LLM-powered decomposition: goal → task graph | `orchestrator/modules/coordination/planner.py` |
| `MissionDispatcher` | Resolves ready tasks, assigns agents, launches execution | `orchestrator/modules/coordination/dispatcher.py` |
| `MissionReconciler` | Stall detection, completion handling, failure escalation | `orchestrator/modules/coordination/reconciler.py` |
| `AgentMatcher` | Deterministic agent-to-task scoring | `orchestrator/modules/coordination/agent_matcher.py` |
| `MissionContextSection` | New context section: mission state for coordinator | `orchestrator/modules/context/sections/mission_context.py` |
| `AgentRosterSection` | New context section: available agents for coordinator | `orchestrator/modules/context/sections/agent_roster.py` |
| `platform_create_mission` | Platform tool: create mission from chat | `platform_actions.py` + `platform_executor.py` |
| `platform_approve_plan` | Platform tool: approve plan from chat | `platform_actions.py` + `platform_executor.py` |
| `platform_mission_status` | Platform tool: check mission progress from chat | `platform_actions.py` + `platform_executor.py` |
| API router | REST endpoints for mission CRUD + lifecycle | `orchestrator/api/missions.py` |

### 12.3 Board Task Bridge

```python
async def _create_board_task(
    self, run: OrchestrationRun, task: OrchestrationTask, agent_id: int
) -> BoardTask:
    """
    Create a board task for kanban visibility.
    Board task is for DISPLAY, not for dispatch.
    """
    return await BoardTask.create(
        workspace_id=run.workspace_id,
        title=task.title,
        description=task.description,
        status="todo",
        assigned_agent_id=agent_id,
        source_type="orchestration",
        source_id=str(run.id),
        labels=[f"MISSION-{run.id}"],
        metadata={
            "mission_id": run.id,
            "task_order": task.sequence_number,
            "task_type": task.task_type,
        },
    )
```

### 12.4 Save as Routine Conversion

```python
async def convert_to_routine(
    self, run: OrchestrationRun
) -> WorkflowRecipe:
    """
    Convert a successful mission into a reusable recipe.
    Strips mission-specific data, keeps structure and agent assignments.
    """
    tasks = await self._get_mission_tasks(run.id)
    deps = await self._get_mission_dependencies(run.id)

    steps = []
    for task in tasks:
        steps.append({
            "order": task.sequence_number,
            "title": task.title,
            "description": task.description,
            "agent_id": task.assigned_agent_id,
            "tools": task.required_tools,
            "success_criteria": task.success_criteria,
            "depends_on": [
                d.depends_on_task_order
                for d in deps if d.task_id == task.id
            ],
        })

    return await WorkflowRecipe.create(
        workspace_id=run.workspace_id,
        name=f"Routine: {run.goal[:100]}",
        description=f"Auto-generated from Mission #{run.id}",
        steps=steps,
        source_mission_id=run.id,
        created_by=run.created_by,
    )
```

---

## 13. Acceptance Criteria

### Must Have

- [x] **Prior art comparison table** — 7 systems compared across coordination model, state management, planning, failure handling, human review. Explicit "adopt" and "reject" per system.
- [x] **CoordinatorService architecture** — class diagram with `CoordinatorService`, `MissionPlanner`, `MissionDispatcher`, `MissionReconciler`, `AgentMatcher`. Method signatures for all public methods.
- [x] **Coordinator tick algorithm** — pseudocode for two-phase tick (dispatch + reconcile) with concurrency handling, error paths, and budget checks.
- [x] **Decomposition pipeline** — template matching → LLM generation → validation. Includes coordinator prompt template and example decomposition template.
- [x] **Agent assignment algorithm** — scoring function with 5 weighted components, threshold for contractor spawn fallback.
- [x] **Failure handling specification** — decision tree covering clean exit, verification failure, infrastructure failure, budget exceeded. Continuation vs retry vs escalation.
- [x] **ContextMode.COORDINATOR definition** — sections, token budget, tool loading strategy. MissionContextSection and AgentRosterSection renderers.
- [x] **Replanning specification** — 5 triggers, 5 constraints, replan prompt template.
- [x] **API endpoint specification** — CRUD + lifecycle endpoints with request/response examples.
- [x] **Sequence diagrams** — 3 diagrams: happy path, failure/retry, human review rejection.

### Should Have

- [ ] **Integration test scenarios** — key test cases for: simple 3-task sequential mission, parallel mission with join, mission with task failure and retry, mission with replanning, budget-exceeded mission.
- [ ] **Performance targets** — tick latency <100ms for 10 active missions, dispatch latency <500ms per task, stall detection within 2 ticks.
- [ ] **Template library seed** — 5-10 decomposition templates for common mission types (research, code review, content creation, data analysis, comparison).

### Nice to Have

- [ ] **Coordinator prompt versioning** — track which prompt version produced which decomposition for future improvement.
- [ ] **Parallel dispatch optimization** — `asyncio.gather` for dispatching multiple independent tasks in a single tick.
- [ ] **Event-driven tick enhancement** — trigger immediate tick on task completion rather than waiting for interval.

---

## 14. Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | **Coordinator complexity** — too many responsibilities in one service | High | Medium | Split into focused classes: Planner, Dispatcher, Reconciler. Coordinator is the orchestrator, not the doer. |
| 2 | **LLM planning reliability** — decomposition quality varies by model and prompt | High | High | Template library for common patterns (ChatHTN hybrid). Validate all plans structurally. Benchmark decomposition quality across models. |
| 3 | **Cost of coordination calls** — coordinator LLM calls add overhead per mission | Medium | Medium | Use cheap models for coordination (Haiku-class). Template matching avoids LLM call entirely for known patterns. |
| 4 | **Tick frequency tradeoff** — too fast = wasted cycles, too slow = delayed dispatch | Medium | Medium | Start with 5s (Symphony default). Make configurable. Event-driven trigger for task completion → immediate dependent dispatch. |
| 5 | **Parallel dispatch race conditions** — two tasks complete simultaneously, both trigger same dependent | Medium | Medium | Optimistic locking with version column on `orchestration_tasks`. Only one transition succeeds. |
| 6 | **Replanning destroys progress** — bad replan discards valid completed work | High | Low | Immutable completed tasks. Replanning only modifies `pending`/`scheduled` tasks. `plan_version` increments for audit. |
| 7 | **Agent unavailability** — assigned agent offline or overloaded | Medium | Medium | Check availability before dispatch. Fallback: reassign or spawn contractor. Stall detection catches unresponsive agents. |
| 8 | **Circular dependencies in task graph** — LLM generates impossible plan | Low | Low | Validate DAG structure via `TopologicalSorter` before accepting any plan. Reject plans with cycles. |
| 9 | **Coordinator single point of failure** | Medium | Low | Stateless design (DB-driven) means any instance can take over. No in-process state to lose. |
| 10 | **Over-engineering v1** | High | High | PRD-100 Risk #3: "Start sequential-only. Get lifecycle right first." Implementation phases: sequential (82A/B) → parallel + replanning (82C). |

---

## 15. Dependencies

| Dependency | Direction | Notes |
|---|---|---|
| PRD-101 (Mission Schema) | **Blocked by 101** | Coordinator reads/writes `orchestration_runs`, `orchestration_tasks`, `orchestration_events`. Schema must exist. |
| PRD-103 (Verification) | **Blocks 103** | Coordinator triggers verification phase. Verification PRD needs coordinator's handoff interface (defined in Section 8). |
| PRD-104 (Ephemeral Agents) | **Blocks 104** | Coordinator spawns contractor agents. Contractor PRD needs coordinator's spawn interface (Section 6). |
| PRD-105 (Budget) | Uses 105 | Coordinator calls budget admission gate before dispatch. Can start with simple checks, enhance later. |
| PRD-106 (Telemetry) | Feeds 106 | Coordinator emits `orchestration_events` that telemetry queries. Event schema supports aggregation. |
| PRD-107 (Context Interface) | **Blocks 107** | Context interface must abstract how coordinator gets/sets context. Coordinator is the primary consumer. |
| `HeartbeatService` | Integration | Coordinator registers its tick alongside heartbeat. Must not conflict with heartbeat scheduling. |
| `AgentFactory` | Integration | Coordinator dispatches via `execute_with_prompt()`. No changes needed to AgentFactory. |
| `TaskReconciler` | Extension | Must extend to cover mission tasks. New `MissionReconciler` or extension of existing class. |
| `ContextService` | Extension | Must add `COORDINATOR` mode and 2 new sections. Non-breaking — adds new mode, doesn't modify existing. |

---

## Appendix A: Coordinator Model Selection

The coordinator LLM call (for planning and replanning) should use a cheap-but-capable model. Planning requires good reasoning but produces relatively short structured output.

| Coordinator Operation | Recommended Model Tier | Rationale |
|---|---|---|
| Template matching | No LLM needed | Embedding similarity check |
| Novel decomposition | Mid-tier (Sonnet 4.6, GPT-4o) | Good reasoning for task decomposition |
| Plan validation | No LLM needed | Structural checks only |
| Replanning | Mid-tier | Same reasoning as decomposition |
| Stall detection | No LLM needed | Time-based threshold check |
| Dependency resolution | No LLM needed | DAG traversal |

Estimated coordinator overhead per mission: 1-2 LLM calls for planning (template miss), 0 for execution (all structural). At ~$0.05-0.10 per planning call, coordinator overhead is <5% of mission cost.

## Appendix B: Research Sources

| Source | What It Informed |
|--------|-----------------|
| Nii 1986, "Blackboard Systems" (AI Magazine) | Shared state coordination, event-driven knowledge source activation |
| LbMAS 2025 (arxiv:2507.01701) | Modern blackboard for LLMs, 5% improvement over static multi-agent |
| Nau et al. JAIR 2003 (SHOP2) | HTN formal correctness, forward-search decomposition |
| ChatHTN 2025 (arxiv:2505.11814) | Hybrid HTN + LLM, provably sound decomposition |
| Hsiao et al. 2025 (arxiv:2511.07568) | HTN structure enables smaller models to outperform larger baselines |
| Rao & Georgeff, ICMAS 1995 | BDI intention commitment, bold vs cautious agent spectrum |
| ChatBDI, AAMAS 2025 | BDI for LLM agents, intention stack prevents thrashing |
| OpenAI Symphony (SPEC.md) | Two-phase tick, continuation vs retry, WORKFLOW.md policy-as-code |
| CrewAI (crewAIInc/crewAI) | `context=[]` dependency declarations, guardrail validation, async_execution |
| AutoGen (microsoft/autogen) | Swarm handoff priority, context_variables shared state |
| LangGraph (langchain-ai/langgraph) | Typed state + checkpoint, interrupt() for human review, Send API |
| Automatos codebase | heartbeat_service.py, agent_factory.py, task_reconciler.py, context/service.py, inter_agent.py, tool_router.py |
