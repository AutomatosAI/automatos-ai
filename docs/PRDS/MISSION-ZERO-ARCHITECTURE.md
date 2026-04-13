# Mission Zero: Onboarding Architecture

> Internal design document — how the wizard builds a workspace from scratch using 4 specialist agents.

---

## Overview

Mission Zero is the automated onboarding system that transforms a business URL into a fully configured Automatos workspace. The user provides a domain and goals; the system scrapes the site, builds a knowledge graph, decomposes the work into tasks, and dispatches 4 specialist agents to research the business, design the workspace, write the personas, and wire everything up.

The key innovation is **ephemeral agents** — per-mission clones of global templates that run in the user's workspace, have access to their data, and are destroyed when the mission completes.

---

## The 4 Hidden Agents

Four global system agents serve as **templates**. They have `workspace_id=NULL`, `is_system_agent=True`, `required_role="onboarding"`, and never appear in any user's agent roster.

| Agent | Model | Temp | Role |
|-------|-------|------|------|
| **VOYAGER** | anthropic/claude-opus-4 | 0.7 | Deep business & market research |
| **BLUEPRINT** | anthropic/claude-sonnet-4 | 0.5 | Evidence-based profile extraction from corpus + knowledge graph |
| **SCRIBE** | openai/gpt-4.1 | 0.7 | Synthesises research into onboarding brief, brand voice guide, recommendations |
| **FORGE** | anthropic/claude-sonnet-4 | 0.3 | Proposes onboarding agent team, tool assignments, workspace configuration |

### VOYAGER — The Researcher

Uses Composio web search tools (COMPOSIO_SEARCH_WEB, COMPOSIO_SEARCH_NEWS, COMPOSIO_SEARCH_TAVILY, COMPOSIO_SEARCH_FINANCE, COMPOSIO_SEARCH_SCHOLAR) to research the company externally. Cross-references multiple sources. Produces: company overview, market analysis, competitive landscape, customer segments, pain points, growth opportunities.

### BLUEPRINT — The Architect

Browses the Automatos marketplace (agents, skills, plugins, LLMs, tools) and designs the workspace. Reads the ingested corpus and knowledge graph for evidence. Constraints: max 8 agents, cost-conscious model selection. Produces: agent roster, marketplace installs, playbooks, governance blueprint, cost breakdown.

### SCRIBE — The Writer

Crafts agent personas (system prompts), heartbeat instructions, playbooks, and the final onboarding brief. Tailors everything to the business voice detected during intake. Produces: per-agent persona, heartbeat prompt, heartbeat checklist, playbook steps.

### FORGE — The Builder

Executes the workspace build sequentially: installs models/skills/plugins, creates agents, wires tools, configures heartbeats. Uses verification calls to confirm each step. Produces: a fully operational workspace ready for the user.

### Seeding

Defined in `orchestrator/core/seeds/seed_onboarding_agents.py`. Idempotent upsert by slug (`voyager`, `blueprint`, `scribe`, `forge`). Run via `seed_onboarding_agents(db)` at startup or manually.

---

## Ephemeral Agents

### Problem

Global template agents have `workspace_id=NULL`. They can't access workspace-scoped data (RAG documents, knowledge graph, files). If we inject the workspace at runtime, a cached agent serving two concurrent onboardings would leak data across tenants.

### Solution

On the first coordinator tick of a Mission Zero run, the system **clones** each template into the user's workspace as an ephemeral agent:

```
Template (global)              →  Ephemeral Clone (workspace-scoped)
─────────────────                 ─────────────────────────────────
workspace_id = NULL               workspace_id = user's workspace
agent_type   = "system"           agent_type   = "ephemeral"
slug         = "voyager"          slug         = "voyager-a1b2c3d4"
is_system    = True               is_system    = False
```

### Lifecycle

1. **Clone** — `_clone_onboarding_agents(db, run)` in `coordinator_service.py`
   - Queries global templates: `is_system_agent=True`, `required_role="onboarding"`, `status="active"`
   - Creates workspace-scoped copies with `agent_type="ephemeral"`
   - Stores clone IDs in `run.config["ephemeral_agent_ids"]`
   - One-time operation: subsequent ticks read IDs from config

2. **Execute** — Ephemeral agents are loaded into the agent pool alongside workspace agents
   - AgentMatcher scores them by role name (voyager, blueprint, scribe, forge)
   - They have native `workspace_id` so RAG, knowledge graph, and file access all work
   - No cross-tenant risk — each mission gets its own agent instances

3. **Cleanup** — `_cleanup_ephemeral_agents(db, run)` on terminal state
   - Fires on COMPLETED, FAILED, or CANCELLED
   - Deletes all agents where `id IN ephemeral_agent_ids AND agent_type = "ephemeral"`
   - Agents vanish from the workspace — no roster clutter

### Multi-Tenancy Safety

300 users onboarding simultaneously = 300 × 4 = 1,200 ephemeral agents, each scoped to their own workspace. Zero shared mutable state. No cross-tenant data leakage.

---

## The Wizard Pipeline

The frontend wizard (`/onboarding/wizard`) drives a 6-step intake process:

```
Step 1: Goals       → User picks business priorities
Step 2: Domain      → User enters their website URL
Step 3: Scan        → Firecrawl maps the site, detects archetype
Step 4: Pages       → User selects which pages to scrape
Step 5: Intake      → Background pipeline (scrape → ingest → graphify → profile)
Step 6: Profile     → User reviews/edits extracted business profile
        ↓
     [Launch Mission Zero]
```

### Backend Pipeline (Step 5)

Triggered by `POST /api/wizard/scrape/{profile_id}` → returns 202 immediately. Background task runs:

1. **Scrape** — Firecrawl extracts each selected URL with schema-based extraction
2. **Ingest** — Documents stored in S3, chunked, embedded via RAG pipeline (S3 Vectors, qwen3-embedding-8b)
3. **Graphify** — Entity extraction builds knowledge graph (nodes, edges, communities). Controlled by `WIZARD_SKIP_GRAPHIFY` env var.
4. **Profile synthesis** — Extracts company_name, sectors, brands, standards, voice_notes from scraped content

Progress streamed via SSE (`GET /api/wizard/progress/{profile_id}`) through Redis pub/sub.

---

## Goal → Plan → Mission

### Goal Generation

`build_mission_goal()` in `plan_generator.py` renders a structured goal string from the BusinessProfile:

```
Mission Zero: bootstrap the Automatos workspace for {company}.
Primary site: {domain}. Archetype detected: {archetype}.
Sectors served: {sectors}.
Notable brands/products: {brands}.
...

## Agent Roles — MANDATORY
You MUST assign tasks to the following specialist agents by their exact role name:
- **voyager** — deep business & market research
- **blueprint** — evidence-based business profile extraction
- **scribe** — synthesises research into final onboarding brief
- **forge** — proposes the onboarding agent team and workspace config

Do NOT assign tasks to 'auto'. Every task must use one of the four roles above.
```

The role instructions are critical — without them, the planner assigns everything to "auto" and the AgentMatcher can't find a match.

### Plan Decomposition

`MissionPlanner.decompose()` in `planner.py`:

1. Attempts template matching first (fast path for known patterns)
2. Falls back to LLM decomposition (currently `zhipu/glm-5-1` via `PLANNER_MODEL` config)
3. LLM returns a task DAG: 3-20 tasks with titles, descriptions, agent_roles, dependencies, verification_criteria
4. Validated: DAG must be acyclic, roles must exist, task count in bounds
5. 3 retry attempts on validation failure
6. Complexity detection: SIMPLE (1 concurrent) / MODERATE (2) / COMPLEX (3)

### Mission Creation

`CoordinatorService.create_mission()`:

1. Creates `OrchestrationRun` in PENDING state
2. Transitions to PLANNING → calls planner
3. Creates `OrchestrationTask` rows from decomposition
4. Creates dependency edges (`OrchestrationTaskDependency`)
5. If `auto_approve=True` (Mission Zero): PLANNING → RUNNING immediately
6. Otherwise: PLANNING → AWAITING_APPROVAL (user must click Approve)

---

## Execution: The Tick Loop

`CoordinatorService.tick()` runs every 5 seconds via APScheduler:

```
For each run in RUNNING state:
  1. Load workspace agents + ephemeral clones
  2. If mission_zero source && no ephemeral_ids yet → clone templates
  3. MissionDispatcher.dispatch_ready() → match tasks to agents
  4. MissionReconciler.reconcile() → verify completed tasks
  5. Check for terminal state → cleanup ephemeral agents
```

### Task Dispatch

`MissionDispatcher.dispatch_ready()`:

- Counts active tasks (ASSIGNED + RUNNING)
- Calculates `available_slots = max_concurrent - active_count`
- Finds candidates: QUEUED tasks with all dependencies met
- For each candidate, `AgentMatcher.match()` scores agents by role name (threshold: 0.40)
- Optimistic claim via version_id check (prevents double-dispatch)
- Budget gate: HEALTHY → allow, WARNING → allow + event, CRITICAL → defer, EXCEEDED → pause run

### Task Execution

`AgentFactory.execute_with_prompt()`:

- Builds prompt from task description + any verification feedback from previous attempts
- Loads tools via `get_tools_for_agent()` (ToolRegistry + ActionRegistry)
- Runs LLM + tool loop (max 10 iterations)
- Returns output text → stored on `task.output`

---

## Verification & Retry

`MissionReconciler._verify_completed_tasks()`:

For each COMPLETED task:

1. Transition COMPLETED → VERIFYING
2. Run deterministic checks first (min_length, required sections)
3. If deterministic checks pass, run LLM verification against task's `verification_criteria`
4. Verdict:
   - **PASS** → VERIFIED (terminal success)
   - **FAIL** → If retries remain: inject feedback into task context, transition to RETRYING → re-dispatch. If exhausted: FAILED (permanent)
   - **PARTIAL** → Same as FAIL but flagged for human escalation if retries exhausted

### Retry Feedback

On retry, the dispatcher builds a revision prompt:

> "Your previous output (attempt N) needs revision. Do NOT rewrite from scratch — revise the content below to address the feedback."

Includes: previous output, failure reasons, verifier reasoning. Saves ~80% tokens vs full rewrite.

---

## State Machine

### Run States

```
PENDING → PLANNING → AWAITING_APPROVAL → RUNNING → COMPLETED
                                            ↓
                                          PAUSED (budget/stall)
                                            ↓
                                          FAILED / CANCELLED
```

### Task States

```
PENDING → QUEUED → ASSIGNED → RUNNING → COMPLETED → VERIFYING → VERIFIED
                                                        ↓
                                                     RETRYING → (re-dispatch)
                                                        ↓
                                                      FAILED
```

**Important**: COMPLETED is NOT terminal — only VERIFIED, FAILED, and SKIPPED are.

---

## Token Budget

- Budget estimated at plan time from task complexity tiers
- Tracked per-task and per-run
- Thresholds: WARNING at 80%, CRITICAL at 100%
- At CRITICAL: non-priority tasks (analysis, research) are deferred; synthesis/review tasks still dispatch
- At EXCEEDED: run paused, human intervention required

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `WIZARD_SKIP_GRAPHIFY` | `0` | Set to `1` to skip knowledge graph build (dev only) |
| `PLANNER_MODEL` | `zhipu/glm-5-1` | LLM model for plan decomposition |
| `LLM_MODEL` | (env) | Default model for agent execution |
| `COORDINATOR_ENABLED` | `true` | Enable/disable the 5s tick loop |
| `COMPOSIO_KEY` | (env) | Required for VOYAGER web search tools |
| `FIRECRAWL_API_KEY` | (env) | Required for wizard site scraping |

---

## Key Files

| File | Purpose |
|------|---------|
| `core/seeds/seed_onboarding_agents.py` | 4 agent template definitions |
| `api/wizard.py` | Wizard endpoints + background pipeline |
| `modules/intake/plan_generator.py` | Goal string generation |
| `modules/coordination/planner.py` | LLM plan decomposition |
| `modules/coordination/dispatcher.py` | Task → agent matching + dispatch |
| `modules/coordination/reconciler.py` | Verification + retry loop |
| `services/coordinator_service.py` | Tick loop, ephemeral lifecycle, mission CRUD |
| `core/models/orchestration.py` | Run, Task, Event, Archive models |
| `core/models/orchestration_enums.py` | State machines + transitions |
