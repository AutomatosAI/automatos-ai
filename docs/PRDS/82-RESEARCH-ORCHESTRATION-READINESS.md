# PRD-82 — Research: Orchestration Readiness Assessment

**Version:** 1.0
**Type:** Research / Strategic Assessment
**Status:** Complete
**Priority:** P0
**Author:** Gerard + Claude
**Date:** 2026-03-14
**Purpose:** Map the trajectory from origin to current state to orchestration-ready future. Honest gap analysis. No dreaming.

---

## 1. Executive Summary

Automatos has evolved from an ambitious multi-agent concept (PRDs 01-06) into a **real, working platform** with substantial infrastructure. But 81 PRDs later, the gap between "what's planned" and "what's built" needs honest accounting before adding an orchestration layer.

**The good news:** The foundations for orchestration are closer than they appear. Context Service, Tool Router, Agent Factory, Heartbeat Service, and Memory are all built and wired. The missing piece is smaller than PRD-82's original 24-section draft suggests.

**The honest news:** Several "foundation" PRDs (01-06) were never implemented as designed. The platform grew organically around the chatbot pipeline, heartbeat system, and tool plumbing — not from those original blueprints. That's fine. The organic path built real things. But it means the orchestration layer needs to be designed for *what actually exists*, not what was originally planned.

---

## 2. Where You Started (The Vision)

### Original Foundation PRDs (01-06)
These were drafted as a connected system:
- **PRD-01** — Core Orchestration Engine (task decomposition, agent assignment, workflow coordination)
- **PRD-02** — Agent Factory & Lifecycle
- **PRD-03** — Context Engineering Layer (atoms → molecules → cells → organs)
- **PRD-04** — Inter-Agent Communication
- **PRD-05** — Memory & Knowledge Systems
- **PRD-06** — Monitoring & Analytics Dashboard

**Reality:** None of these were implemented as written. Instead, the platform evolved through practical needs:
- Agent Factory got rewritten 3+ times (latest: clean 1,235-line version)
- Context became the `ContextService` with 8 modes (not the atom→organ hierarchy)
- Memory became a 5-layer stack (Redis → Postgres → Mem0 → RAG)
- Inter-agent comms became Redis pub/sub + `inter_agent.py` (1,216 lines)
- Monitoring became heartbeat + activity feed + reports (PRDs 72, 76)

**Lesson:** The original PRDs were architectural aspirations. The actual platform was built bottom-up from real user needs. That's not a failure — that's how good software gets built.

### The Pivot Moment
Around PRD-37 (SaaS Foundation), the project shifted from "research platform" to "multi-tenant product." This forced real decisions:
- Clerk auth + workspaces
- Tool assignments per agent (DB-backed)
- ContextService as single entry point
- Config centralization (86 files fixed)
- Security hardening

This was the right call. It built the infrastructure that orchestration needs.

---

## 3. What You've Actually Built (Honest Inventory)

### Working Foundation Systems

| System | State | Key Files | Lines |
|--------|-------|-----------|-------|
| **ContextService** | ✅ Built, 8 modes, 12 sections | `modules/context/service.py` | ~800 |
| **Tool Router** | ✅ Single source of truth | `modules/tools/tool_router.py` | 735 |
| **Tool Registry** | ✅ Core + Platform + Workspace | `tool_registry.py` + `action_registry.py` | ~2,000 |
| **Unified Executor** | ✅ Prefix-based dispatch | `unified_executor.py` + 9 exec modules | ~1,500 |
| **Agent Factory** | ✅ Clean rewrite, tool loop | `agent_factory.py` | 1,235 |
| **Universal Router** | ✅ 7-tier routing | `core/routing/engine.py` | 906 |
| **Heartbeat Service** | ✅ Cron-based autonomous execution | `heartbeat_service.py` | 1,459 |
| **Memory (5-layer)** | ✅ Redis → Postgres → Mem0 → RAG | `unified_memory_service.py` | 2,068 |
| **Chatbot Pipeline** | ✅ SSE streaming + tool loop | `consumers/chatbot/service.py` | 1,963 |
| **Inter-Agent Comms** | ✅ Redis pub/sub + consensus | `inter_agent.py` | 1,216 |
| **Multi-Agent Coordination** | ⚠️ Built but untested at scale | `coordination_manager.py` | 877 |
| **Channel Adapters** | ✅ 11 platforms | `channels/` | ~2,000 |
| **Report Service** | ✅ PRD-76 | `report_service.py` | ~400 |
| **Task Reconciler** | ✅ Stall detection + retry | `task_reconciler.py` | ~200 |
| **Scheduled Tasks** | ✅ PRD-77 agent self-scheduling | `scheduled_task_service.py` | ~300 |

### What Does NOT Exist

| System | Status | Notes |
|--------|--------|-------|
| **orchestration_runs table** | ❌ Not created | No migration, no model, no code |
| **orchestration_tasks table** | ❌ Not created | No migration, no model, no code |
| **Task graph / dependency engine** | ❌ Not built | No DAG, no dependency resolution |
| **Coordinator agent** | ❌ Not built | Heartbeat orchestrator is closest but different purpose |
| **Verifier / critic loop** | ❌ Not built | No output validation against criteria |
| **Aggregator** | ❌ Not built | No multi-output merging |
| **Budget enforcement (per-run)** | ❌ Not built | Token budget exists in ContextService but not per-run |
| **Run trace / explainability** | ❌ Not built | Heartbeat logs exist but no structured run trace |
| **Guidance engine** | ❌ Not built | No prompt coaching, model recommendation, task structuring |
| **Learned patterns** | ❌ Not built | No outcome tracking feeding back to recommendations |
| **Recipe from run** | ❌ Not built | Can't convert a successful run into a reusable recipe |

### Partially Built (Exists But Not Orchestration-Grade)

| System | State | Gap |
|--------|-------|-----|
| **9-stage workflow** | Legacy, mostly dead code | `modules/orchestrator/service.py` marked LEGACY. Stages exist but pipeline isn't wired to live execution path |
| **Recipe execution** | DB table + scheduler exists | Only 1 concrete recipe (Jira bug triage). No dynamic recipe creation |
| **Board tasks** | PRD-72 table + bridge | Task board exists but not wired as orchestration task graph |
| **Workflow executions** | Table exists | No structured run lifecycle (start → tasks → verify → complete) |

---

## 4. Competitive Landscape Analysis

### Agent Zero — Hierarchical Delegation
**Model:** Prompt-driven agents, parent-child delegation, conversation sealing.

| Aspect | Agent Zero | Automatos |
|--------|-----------|-----------|
| Multi-tenancy | ❌ Single user | ✅ Full workspace isolation |
| Persistent state | ❌ In-memory (crashes lose work) | ✅ PostgreSQL + Redis |
| Tool system | Basic (code exec, search, delegate) | ✅ 3-layer registry, 40+ platform actions |
| Memory | FAISS in-process, LLM consolidation | ✅ 5-layer stack with Mem0 |
| Delegation | ✅ Clean parent→child with topic sealing | ⚠️ Inter-agent comms exist but no delegation protocol |
| Verification | ❌ None (prompt-dependent) | ❌ None |
| Context management | Basic history compression | ✅ ContextService with 8 modes, 12 sections, budget |

**What to steal:**
- Conversation sealing after delegation (prevent context bleed)
- Utility model separation (cheap model for memory/compression)
- Skills as on-demand loading (not eager)

**What you already beat them on:**
- Persistence, multi-tenancy, tool richness, context engineering, channels

### OpenClaw — Personal AI Gateway
**Model:** Hub-and-spoke gateway, channel-first, single-user.

| Aspect | OpenClaw | Automatos |
|--------|----------|-----------|
| Channels | ✅ 15+ platforms, native apps | ✅ 11 channels |
| Multi-agent routing | ✅ 6-tier deterministic bindings | ✅ 7-tier Universal Router |
| Multi-tenancy | ❌ Single trusted operator | ✅ Full workspace isolation |
| Persistence | SQLite + JSONL files | ✅ PostgreSQL + Redis + S3 |
| Tool policy layers | ✅ 6-level deny-first | ⚠️ Per-agent assignment, no layered policy |
| Scaling | ❌ Single process | ✅ Multi-worker |

**What to steal:**
- Tool policy layering (gateway > agent > provider > group > sandbox)
- Context compaction with dedicated summarization model
- ACP protocol for external agent integration

**Not relevant:** Different use case (personal assistant vs. platform).

### OpenAI Symphony — Issue Tracker Daemon
**Model:** Linear issues → isolated Codex agents → PRs. One agent per issue, no coordination.

| Aspect | Symphony | Automatos |
|--------|----------|-----------|
| Coordination | ❌ None (isolation is the strategy) | ⚠️ Has inter-agent, needs orchestration |
| Policy as code | ✅ WORKFLOW.md (brilliant) | ⚠️ Agent config in DB, not versioned |
| Workspace isolation | ✅ Strict per-issue sandboxing | ⚠️ Shared workspace with file tools |
| Reconciliation | ✅ Self-healing poll loop | ✅ TaskReconciler exists |
| Persistence | ❌ In-memory only | ✅ PostgreSQL |
| Multi-agent on same task | ❌ Explicitly avoided | Goal for Phase 2 |

**What to steal:**
- WORKFLOW.md / Policy-as-Code pattern (version agent behavior alongside code)
- Reconciliation loop pattern (already have TaskReconciler — extend it)
- Lifecycle hooks (before_run, after_run) for workspace setup/teardown
- Issue tracker as coordination mechanism (board tasks → orchestration tasks?)

**What you already beat them on:**
- Persistence, multi-agent, tool richness, memory, real-time channels

### Perplexity Computer Use
**Model:** Browser automation agent with search-first approach.

**Relevance:** Limited. Different problem domain (web interaction vs. multi-agent orchestration). Worth watching for UX patterns around showing agent work in progress.

---

## 5. The Actual Gap to Orchestration

Here's the honest distance from where you are to a working orchestration layer:

### Already Have (Don't Rebuild)

```
✅ ContextService         → system prompts for any agent role
✅ Tool Router             → tools for any agent
✅ Agent Factory           → execute any agent with tool loop
✅ Universal Router        → route requests to agents
✅ Heartbeat Service       → cron-based autonomous execution
✅ Memory (5-layer)        → context persistence
✅ Inter-Agent Comms       → Redis pub/sub messaging
✅ Task Reconciler         → stall detection + recovery
✅ Scheduled Tasks         → agent self-scheduling
✅ Report Service          → agent output persistence
✅ Board Tasks             → task tracking
```

### Need to Build (The Real Gap)

```
❌ orchestration_runs      → persistent run lifecycle
❌ orchestration_tasks     → persistent task graph
❌ task dependencies       → DAG resolution
❌ coordinator service     → plan → assign → monitor → aggregate
❌ verifier                → output validation against criteria
❌ run budget tracking     → per-run token/tool limits
❌ run event log           → structured execution trace
```

### The Key Insight

**The gap is narrower than PRD-82's original scope suggested.** You don't need a Guidance Engine, Learning Engine, Prompt Coach, Model Recommender, or Recipe Builder to get orchestration working. Those are Phase 2C/2D features.

The core missing piece is:

> **A coordinator that creates a persistent run, decomposes it into tasks with dependencies, assigns agents, executes sequentially/parallel, verifies outputs, and records the trace.**

Everything else (context, tools, agents, memory, scheduling) already works.

---

## 6. Dependency Chain — What Blocks What

### Critical Path to Orchestration

```
PRD-81 (Mission Cleanup)          STATUS: In Progress (current branch)
  └── Consolidates context + memory foundations
      │
PRD-82A (Orchestration Schema)    STATUS: Not Started
  └── orchestration_runs, orchestration_tasks, events tables
  └── Coordinator ContextMode
  └── Verifier ContextMode
      │
PRD-82B (Sequential Coordinator)  STATUS: Not Started
  └── Plan → assign → execute → verify → aggregate
  └── Uses ContextService + AgentFactory + ToolRouter
  └── Persistent run lifecycle
      │
PRD-82C (Parallel + Budget)       STATUS: Not Started
  └── Bounded parallel task execution
  └── Per-run budget enforcement
  └── Run trace UI
      │
PRD-82D (Guidance + Learning)     STATUS: Not Started
  └── Prompt coaching
  └── Model recommendation
  └── Outcome tracking → improved recommendations
```

### What's NOT on the critical path (can be deferred)

- PRD-80 (Unified Context Service) — **already essentially built** as `modules/context/service.py`
- PRD-68 (Progressive Complexity) — nice-to-have for routing, not blocking orchestration
- PRD-64 (Unified Action Discovery) — partially done via ActionRegistry
- PRD-69 (Agent Intelligence Layer) — Phase 2D territory

### What IS blocking

- **PRD-81 (Mission Cleanup)** — if this is cleaning up context/memory foundations, it should land first
- **No orchestration schema** — need tables before services
- **No coordinator logic** — this is the actual new code

---

## 7. Recommended Path Forward

### Step 1: Land PRD-81 (current work)
Finish the mission cleanup. Stabilize context + memory.

### Step 2: PRD-82A — Orchestration Schema + Context Modes
**Scope:** Database only + context modes. No execution logic.

Deliverables:
1. Alembic migration: `orchestration_runs`, `orchestration_tasks`, `orchestration_task_dependencies`, `orchestration_events`
2. SQLAlchemy models
3. Two new ContextModes: `COORDINATOR`, `VERIFIER`
4. ModeConfig for each (which sections, tool loading strategy)
5. API endpoints: create run, get run, list tasks, get events
6. Tests for schema + context modes

**Why separate:** Schema changes are low-risk, high-value. Once tables exist, everything else can be built incrementally.

### Step 3: PRD-82B — Sequential Coordinator Service
**Scope:** The coordinator. Sequential execution only. No parallelism.

Deliverables:
1. `CoordinatorService` — takes a goal, produces a plan (task list with dependencies)
2. Plan execution loop: pick next ready task → assign agent → execute via AgentFactory → verify → mark complete
3. Verification step: LLM-as-judge against success criteria
4. Run lifecycle: created → planning → executing → verifying → completed/failed
5. Event logging: every state transition recorded in `orchestration_events`
6. Integration with existing `ContextService` (COORDINATOR mode for planning, TASK_EXECUTION for agent work, VERIFIER for validation)
7. Tests

**Why sequential first:** Parallel execution adds complexity (race conditions, resource contention, partial failure handling). Get the lifecycle right first.

### Step 4: PRD-82C — Parallel Execution + Budget + UI
**Scope:** Scale the coordinator.

Deliverables:
1. Bounded parallel task execution (asyncio.gather with semaphore)
2. Per-run token budget tracking (increment on each LLM call)
3. Per-run tool call budget
4. Budget exhaustion handling (degrade, pause, or fail)
5. Run trace API for frontend
6. Frontend: run viewer with task graph, status, budget gauge, event timeline
7. Tests

### Step 5: PRD-82D — Guidance + Learning (Future)
**Scope:** Intelligence layer on top of working orchestration.

Deliverables:
1. Prompt coach (analyze request, suggest improvements)
2. Model recommender (task type → model suggestion)
3. Outcome capture (link run results to recommendations)
4. Pattern detection (repeated successful structures → recipe candidates)
5. Recommendation UI (preflight advice panel)

---

## 8. What You Can Reuse (Don't Reinvent)

| Existing System | Reuse For |
|----------------|-----------|
| `ContextService` + `COORDINATOR` mode | Coordinator's system prompt |
| `ContextService` + `TASK_EXECUTION` mode | Agent task execution (already works) |
| `AgentFactory.execute_with_prompt()` | Execute any task agent |
| `get_tools_for_agent()` | Tool loading for task agents |
| `UnifiedToolExecutor` | Tool dispatch |
| `TaskReconciler` pattern | Stall detection for orchestration tasks |
| `HeartbeatService` scheduling | Cron-triggered orchestration runs |
| `report_service` | Run output persistence |
| `board_task_bridge` | UI task display |
| `inter_agent.py` | Agent-to-agent messaging during runs |

---

## 9. What This Means for PRD Count

You asked if you need 20 more PRDs or 200. Here's the honest answer:

### To get orchestration working: 4 PRDs
- 82A (Schema + Context Modes) — ~1 week
- 82B (Sequential Coordinator) — ~2 weeks
- 82C (Parallel + Budget + UI) — ~2 weeks
- 82D (Guidance + Learning) — ~3 weeks

### To get the full Phase 2 vision: ~8-10 PRDs
Add:
- Recipe-from-run generation
- Workflow pattern learning
- Advanced model recommendation
- Cross-run analytics
- Semi-autonomous workflow builder
- External action approval gates

### You do NOT need to rewrite foundations
The original PRDs 01-06 were superseded by what you actually built. Don't go back and implement them as designed. The organic evolution produced something more practical.

---

## 10. Risk Assessment

### Risk 1: Over-scoping PRD-82 (again)
The original draft was 24 sections covering guidance, learning, recipes, coaching, recommendations, AND orchestration. That's 6 systems.
**Mitigation:** This document splits it into 4 focused PRDs.

### Risk 2: Coordinator complexity
The coordinator needs to: decompose tasks, assign agents, manage dependencies, handle failures, retry, verify, aggregate. This is the hardest new code.
**Mitigation:** Start sequential-only. No parallel. No dynamic replanning. Just: plan → execute in order → verify → done.

### Risk 3: Context window pressure during multi-agent runs
Each agent task consumes context. A 5-task run means 5 separate LLM interactions, each needing full context assembly.
**Mitigation:** ContextService already handles this. COORDINATOR mode gets planning context. TASK_EXECUTION mode gets agent context. They're separate calls, not one bloated prompt.

### Risk 4: Cost blowout
Coordinator call + N task agent calls + N verifier calls = 2N+1 LLM calls minimum.
**Mitigation:** Budget tracking from day 1 (PRD-82C). Coordinator can use cheaper model. Verifier can be rule-based for simple criteria.

### Risk 5: Nobody uses orchestration if simple chat works
If 90% of requests are simple chat, building orchestration is premature.
**Mitigation:** Orchestration is opt-in (triggered by recipes, heartbeats, or explicit "plan this" requests). Don't force it on simple queries.

---

## 11. Conclusion

**Where you started:** Ambitious 6-PRD foundation that was too abstract to implement directly.

**What you built instead:** A practical, working platform through 81 PRDs of organic evolution. Context Service, Tool Router, Agent Factory, Memory, Heartbeat, Channels, Routing — all real, all wired, all serving users.

**What's actually missing for orchestration:** Persistent run/task schema, a coordinator service, verification, and budget tracking. That's it. The execution infrastructure (agents, tools, context, memory) already works.

**The path:** 4 focused PRDs, building on what exists. No rewriting foundations. No 24-section fantasy documents. Schema first, sequential coordinator second, parallel + budget third, intelligence fourth.

You're closer than you think. The foundations are there. The next step is PRD-82A: the schema.

---

## Appendix A: Competitive Pattern Matrix

| Pattern | Agent Zero | OpenClaw | Symphony | Automatos (Current) | Automatos (After 82A-D) |
|---------|-----------|----------|----------|---------------------|------------------------|
| Persistent runs | ❌ | ❌ | ❌ | ❌ | ✅ |
| Task graph | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multi-tenant | ❌ | ❌ | ❌ | ✅ | ✅ |
| Tool richness | Low | Medium | Low | ✅ High | ✅ High |
| Memory system | FAISS only | SQLite | None | ✅ 5-layer | ✅ 5-layer |
| Context engineering | Basic | Basic | None | ✅ 8 modes | ✅ 10 modes |
| Verification | ❌ | ❌ | CI-based | ❌ | ✅ LLM-as-judge |
| Budget control | ❌ | ❌ | ❌ | ❌ | ✅ Per-run |
| Delegation | ✅ | Subagents | ❌ | ⚠️ Pub/sub | ✅ Coordinator |
| Channels | Web only | ✅ 15+ | None | ✅ 11 | ✅ 11 |
| Self-learning | ❌ | ❌ | ❌ | ❌ | ✅ (82D) |
| Policy as code | Prompts | ✅ JSON5 | ✅ WORKFLOW.md | DB config | DB + SKILL.md |
| Run explainability | ❌ | ❌ | Logs | ❌ | ✅ Event trace |

## Appendix B: Patterns to Adopt from Research

### From Agent Zero
1. **Conversation sealing** — after coordinator delegates to agent, seal that context to prevent bleed into next task
2. **Utility model** — use cheap model for coordinator planning, memory ops, verification heuristics
3. **On-demand skill loading** — load SKILL.md content only when agent is assigned task that needs it

### From OpenClaw
1. **Tool policy layers** — consider gateway > workspace > agent > task layering for tool access control
2. **Context compaction model** — dedicated cheaper model for summarization during long runs

### From Symphony
1. **Reconciliation loop** — extend TaskReconciler to cover orchestration runs (detect stalled runs, orphaned tasks)
2. **Lifecycle hooks** — `before_task`, `after_task` hooks for workspace setup/teardown
3. **Tracker as coordinator** — board tasks / mission board as the human-visible coordination layer (agents read from and write to it)
4. **Continuation vs. retry distinction** — continuation (same thread, 1s delay) vs. failure retry (fresh attempt, exponential backoff)
