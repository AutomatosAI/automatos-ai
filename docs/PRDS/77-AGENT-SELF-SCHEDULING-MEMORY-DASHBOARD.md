# PRD-77: Agent Self-Scheduling & Memory Dashboard

**Status:** DRAFT
**Priority:** P1
**Author:** Auto-generated from Agent Zero research (2026-03-10)
**Dependencies:** PRD-05 (Memory & Knowledge), PRD-55 (Heartbeat), PRD-72 (Activity Command Centre)
**Research Reference:** See `memory/agent-zero-research.md` for competitive analysis

---

## Problem Statement

1. **Agents can't schedule their own follow-up work.** The UnifiedScheduler (APScheduler) powers heartbeats and recipes, but agents have no tool to create their own scheduled tasks. An agent that discovers "I should check this again tomorrow" has no way to act on it.

2. **Memory is opaque to users.** Memories are stored in Mem0 but there's no UI to browse, search, manage, or consolidate them. Users can't see what their agents remember, can't fix bad memories, and can't prune stale ones. The memory stats page shows counts but no content.

3. **Memory search was broken** (fixed in `fix-memory` branch — `Mem0Client.search()` was calling the list endpoint instead of the vector search endpoint). Remaining issues: potential double-injection of memories in prompts, `get_daily_logs()` never wired into prompt assembly.

---

## Phase 1: Agent Self-Scheduling Tool

### User Stories

- **US-001:** As an agent, I can schedule a one-shot follow-up task so I can revisit work later without human intervention.
- **US-002:** As an agent, I can schedule a recurring task (cron) so I can perform periodic checks autonomously.
- **US-003:** As an admin, I can view and cancel agent-scheduled tasks so I maintain control over what runs.

### Implementation

#### New Platform Tool: `platform_schedule_task`

```python
# platform_actions.py — new ActionDefinition
ActionDefinition(
    name="platform_schedule_task",
    description="Schedule a follow-up task for yourself or request a task for another agent",
    parameters={
        "task_type": "one_shot | recurring",
        "description": "What the task should accomplish",
        "schedule": "ISO datetime for one_shot, cron expression for recurring (e.g. '0 9 * * 1' = every Monday 9am)",
        "target_agent_id": "Optional — defaults to self. ID of agent to run the task.",
        "max_runs": "Optional — for recurring, max number of executions before auto-cancel",
    },
)
```

#### Execution Flow

1. Agent calls `platform_schedule_task` during conversation
2. `PlatformActionExecutor` validates parameters, creates DB record in `agent_scheduled_tasks` table
3. Registers job with `UnifiedScheduler` (APScheduler) — same scheduler heartbeats use
4. When job fires, creates a new chat session with the target agent, injecting the task description as the user message
5. Results stored as activity feed entry + optional report via `platform_submit_report`

#### Database Schema

```sql
CREATE TABLE agent_scheduled_tasks (
    id SERIAL PRIMARY KEY,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
    created_by_agent_id INTEGER NOT NULL REFERENCES agents(id),
    target_agent_id INTEGER NOT NULL REFERENCES agents(id),
    task_type VARCHAR(20) NOT NULL,  -- 'one_shot' | 'recurring'
    description TEXT NOT NULL,
    schedule VARCHAR(100) NOT NULL,  -- ISO datetime or cron expression
    max_runs INTEGER,
    run_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',  -- 'active' | 'paused' | 'completed' | 'cancelled'
    last_run_at TIMESTAMP,
    next_run_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### Admin Controls

- View all scheduled tasks in Activity → Missions tab (or new Scheduled tab)
- Pause/resume/cancel any task
- Set workspace-level limits: max scheduled tasks per agent, max recurring tasks total

#### Registration

- `platform_actions.py` — ActionDefinition
- `platform_executor.py` — handler + `_handlers` dict entry
- `auto.py` — Tier 2 keywords: `schedule`, `follow-up`, `remind`, `later`, `recurring`, `check again`

---

## Phase 2: Memory Explorer Dashboard

### User Stories

- **US-004:** As a user, I can browse all memories stored by my agents so I understand what they've learned.
- **US-005:** As a user, I can search memories by keyword so I can find specific knowledge.
- **US-006:** As a user, I can delete individual memories so I can correct mistakes.
- **US-007:** As a user, I can view memory metadata (created by, when, access count) so I can assess quality.

### Backend API Additions (`api/memory.py` or `api/memory_stats.py`)

```
GET  /api/v1/memory/browse          — paginated list with search, filters
GET  /api/v1/memory/{id}            — single memory detail
DELETE /api/v1/memory/{id}          — delete a memory
GET  /api/v1/memory/health          — staleness report, duplicate detection
POST /api/v1/memory/search          — vector similarity search (proxies to Mem0)
```

### Frontend: Memory Tab

**Location:** Activity → Memory tab (alongside Dashboard, Feed, Reports, Missions)

**Components:**
- `components/activity/activity-memory.tsx` — main tab with table + search bar
- `components/activity/memory-card.tsx` — individual memory display (content, agent, date, score)
- `components/activity/memory-health.tsx` — health stats (total, stale count, duplicates)

**Table columns:**
| Content (truncated) | Agent | Created | Last Accessed | Score | Actions |
|---------------------|-------|---------|---------------|-------|---------|
| "User prefers..." | Sentinel | 2h ago | 1h ago | 0.92 | View / Delete |

**Filters:** By agent, date range, keyword search

---

## Phase 3: Memory Consolidation & Pruning

### User Stories

- **US-008:** As a user, I can select multiple related memories and merge them into one so I reduce duplication.
- **US-009:** As a user, I can see auto-flagged stale memories (no access in 30+ days) so I can prune them.
- **US-010:** As a user, I can trigger AI-powered consolidation that summarises related memories into fewer, richer entries.

### Implementation

#### Consolidation Endpoint

```
POST /api/v1/memory/consolidate
Body: { "memory_ids": [1, 2, 3], "strategy": "merge" | "summarise" }
```

- `merge` — concatenate content, keep latest metadata
- `summarise` — call LLM to produce a single summary, delete originals, store new

#### Auto-Pruning Service

- Cron job (via UnifiedScheduler) runs weekly
- Flags memories with no access in configurable window (default 30 days)
- Does NOT auto-delete — flags for user review in Memory Health dashboard
- Detects near-duplicate memories via embedding similarity (>0.95 cosine)

---

## Phase 4: Remaining Memory Bug Fixes

### P1: Double Memory Injection Check

Verify whether memories are injected twice in `SmartChatOrchestrator`:
1. Via `get_happy_system_prompt(memories=...)` in system prompt
2. As a separate system message

If confirmed, remove the duplicate path. Single injection point = less token waste.

### P1: Wire `get_daily_logs()`

`get_daily_logs()` exists but is never called during prompt assembly (PRD-55 incomplete). Wire it into the prompt with a 500-token cap so agents have access to their daily activity context.

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Agent-scheduled tasks created per week | >10 across all workspaces |
| Memory search returning scored results | 100% (was 0% before fix) |
| User memory management actions (view/delete) | >5/week per active workspace |
| Memory consolidation reducing total count | 20% reduction in duplicates |
| Stale memories flagged and reviewed | 90% reviewed within 7 days |

---

## Out of Scope (Future PRDs)

- **Agent-to-Agent messaging** (A2A protocol) — see agent-zero-research.md, ties into PRD-68/69
- **Behaviour self-adjustment** — agents tuning own parameters mid-conversation
- **Dynamic sub-agent spawning** — runtime agent creation without roster pre-assignment
- These three form a future "Agent Autonomy" super-PRD building on PRD-68, PRD-50, PRD-69
