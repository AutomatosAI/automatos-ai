# AUTOMATOS 0.2 — Current State Audit

**Purpose:** Baseline inventory. Numbers as of 2026-04-24. This is the "before" photo; [09-SUCCESS-METRICS.md](./09-SUCCESS-METRICS.md) is the "after".

Sources:
- Static scan: `grep -r __tablename__ orchestrator/`, `ls orchestrator/api/`, `grep include_router orchestrator/main.py`, `ls orchestrator/alembic/versions/`.
- Dynamic scan: blocked (local Postgres not running; Railway creds intentionally not pulled). Supplement with PRD-135 Phase 1 once scanner ships.

---

## 1. Headline numbers

| Surface | Count | Notes |
|---|---|---|
| Orchestrator Python files | 801 | 228,297 lines of code |
| `orchestrator/api/` router files | 103 (mounted in main.py) | Target: ≤25 |
| `orchestrator/services/` files | 24 | Target: ≤15 (some merge into core) |
| `orchestrator/core/` top-level subdirs | 19 | Auth, cache, composio, credentials, database, llm, math, models, monitoring, neural_field, ports, redis, routing, security, seeds, services, task_runner, utils, workspaces |
| `orchestrator/modules/` subdirs | 18 | agents, attachments, codegraph, context, coordination, documents, evaluation, intake, knowledge, learning, memory, nl2sql, rag, search, tools, voice, workflows |
| SQLAlchemy `__tablename__` declarations | 109 | Target: ≤75 post-wave-1 |
| Tables in migrations with no model | 16 | Prime DROP candidates (see §4) |
| Alembic migrations | 94 versions | Cumulative history; several no-op or reverted |
| `main.py` | 1,456 lines | Most of it is the 103 include_router calls |
| Frontend TS files (app/components/hooks/lib) | 599 | Target: ≤400 |
| React components | 429 | Target: ≤300 |
| `use-*-api.ts` hooks | ~60 | Target: ≤25, one per canonical domain |

---

## 2. API router duplication census (the top offenders)

Grouped by concept. Count is # of router files that live on the concept today.

### Analytics (7 routers — worst offender)
- `analytics.py`
- `analytics_api.py`
- `analytics_real.py`
- `analytics_charts.py`
- `composio_analytics.py`
- `database_analytics.py`
- `llm_analytics.py`
- (plus `kpi_api.py` and `insights.py` overlap in spirit)

**Canonical target:** one `analytics.py` with sub-paths `/llm`, `/composio`, `/database`, `/charts`. Kill `_real`, `_api` suffixes.

### Marketplace (5 routers)
- `marketplace.py`
- `marketplace_plugins.py`
- `llm_marketplace.py`
- `openrouter_marketplace.py`
- `widget_marketplace.py`
- (plus admin_plugins, agent_plugins, workspace_plugins, plugin_security, widget_installation — plugin-domain partially overlaps)

**Canonical target:** one `marketplace.py` (catalog browse + install for skills, models, plugins, widgets, templates). Admin flows separate.

### Memory (3 routers)
- `memory.py`
- `memory_stats.py`
- `widget_memory.py`

**Canonical target:** one `memory.py`. Stats is a sub-path, widget_memory is a deprecated alias.

### Knowledge / RAG (4 routers)
- `knowledge.py`
- `knowledge_graph.py`
- `knowledge_multimodal.py`
- `database_knowledge.py` (+ `database_knowledge_simple.py` — already duplicated)

**Canonical target:** one `knowledge.py` with sub-paths `/graph`, `/database`, `/multimodal`. The `database_knowledge_simple.py` file is a textbook dead duplicate (was superseded).

### Workflows / Recipes / Playbooks (7+ routers)
- `workflows.py`
- `workflow_history.py`
- `workflow_recipes.py`
- `workflow_templates.py`
- `widget_workflows.py`
- `recipe_executor.py`
- `api_playbooks.py` + `playbooks.py`

**Canonical target:** one `recipes.py` (per PRD-125's decoupling direction). Mission/coordinator handles execution; recipes are recurring mission schedules. Playbooks → recipes at the naming layer (user-facing term unifies).

### Chat (3 routers)
- `chat.py`
- `chatbot_llm.py`
- `chat_voice.py`

**Canonical target:** one `chat.py`. Voice is a sub-path if live (verify).

### Tasks / Board (3 routers)
- `tasks.py`
- `board_tasks.py`
- (+ orchestration tasks in coordinator)

**Canonical target:** one `tasks.py` once the board/mission split reconciles. Board tasks = coordinator tasks with board-style UI metadata; should not be a separate API.

### Widgets (8+ routers)
- `widget_email.py`
- `widget_marketplace.py`
- `widget_memory.py`
- `widget_workflows.py`
- `widget_installation.py`
- `widget_review.py`
- `marketplace_widget.py`
- `widgets/` subdir

**Canonical target:** one `widgets.py` (install, review, render). Per-widget sub-routers only if the widget needs bespoke backend.

### Misc duplicate/edge
- `system.py` + `system_settings.py` + `system_prompts.py` + `admin_prompts.py` — four "system" surfaces. Collapse to one admin surface (`admin/system.py` with sub-paths).
- `codegraph.py` + `patterns.py` + `insights.py` + `synthesis.py` + `learning.py` + `recommendations.py` + `solutions.py` + `problems.py` — the "smart" suite from PRD-11 era. Likely 60% dead or never wired to frontend. Needs sub-audit.
- `anthropic_client.py` as a router file — suspicious (client code, not route code). Verify; likely dead.
- `blog.py` — single-purpose router; candidate to move to CMS or delete.

**Estimated API collapse:** 103 → ~25 by Wave 2 end. See [04-API-SURFACE.md](./04-API-SURFACE.md) for the canonical list.

---

## 3. Core vs services vs modules — the three-way split

Same domain appearing in multiple trees:

| Domain | In `core/` | In `services/` | In `modules/` |
|---|---|---|---|
| Composio | `core/composio/` | `services/composio_api_service.py` | — |
| Credentials | `core/credentials/` | — | — |
| Workspaces | `core/workspaces/` | — | — |
| Memory | — | `services/memory_archival_job.py`, `services/memory_jobs.py` | `modules/memory/` |
| Documents | — | — | `modules/documents/` + top-level `orchestrator/documents` imports |
| Tools | — | `services/tool_manifest_service.py` | `modules/tools/` |
| Coordination | — | `services/coordinator_service.py`, `services/orchestration_*.py` | `modules/coordination/` |
| Deliverables | — | `services/deliverable_service.py`, `services/report_service.py` | — |
| Agents | — | `services/heartbeat_service.py`, `services/activity_service.py` | `modules/agents/` |

**The rule today:** none. Some domains live in one tree, others span all three, and the reason is often historical (whichever tree the PRD author preferred at the time).

**The rule after 0.2** ([03-DOMAIN-MODEL.md](./03-DOMAIN-MODEL.md)):
- `core/` = long-lived infrastructure primitives (auth, db, redis, s3, llm routing).
- `modules/` = domain packages with their own models + logic.
- `services/` = cross-domain orchestration (e.g. a service that spans agents + tasks + deliverables).

That rule eliminates ~60% of the three-way ambiguity.

---

## 4. Orphan table suspects (in migrations, no SQLAlchemy model)

Extracted from `orchestrator/alembic/versions/*.py` `op.create_table(...)` vs `__tablename__` declarations.

**16 orphan candidates:**

| Table | Likely status | Probable replacement |
|---|---|---|
| `agent_executions` | dead | `orchestration_runs` + `agent_reports` |
| `agent_skills` | dead | `workspace_enabled_skills` + agent assignments |
| `agent_tool_assignments_v2` | dead (v2 suffix = migration artifact) | `agent_tool_permissions` |
| `agent_tools` | dead | `agent_tool_permissions` |
| `chat_messages` | dead (renamed) | `messages` |
| `context_usage` | unclear | likely `llm_usage` |
| `conversations` | dead (renamed) | `chats` |
| `document_chunks` | dead (moved to vector store) | S3 Vectors index |
| `document_usage` | dead | (no replacement; telemetry folded elsewhere) |
| `performance_data` | dead | `component_metrics` |
| `system_alerts` | unclear | likely dead (notifications system took over) |
| `system_metrics` | dead | `component_metrics` |
| `tenant_tool_config` | dead (renamed) | `workspace_tool_config` |
| `User` | dead (capitalized duplicate) | `users` |
| `user_activities` | unclear | likely `audit_logs` |
| `workflow_agents` | dead | `agents` + `workflow_executions` join |

**Action:** each needs a 7-day `pg_stat_user_tables.seq_scan = 0 AND n_tup_ins/upd/del = 0` confirmation before DROP. Blocked on PRD-135 Phase 1 + 4 (runtime overlay) being live. Until then, ship as **model stubs that raise on first access** so any live caller surfaces loudly in logs.

Plus: per memory, 11 `b_*_<date>` backup tables from the Phase B rename pass — watch-period confirmation then DROP.

---

## 5. Known canonical-vs-duplicate pairs from memory

These are already-documented consolidation targets the user confirmed but hasn't yet scheduled:

| Canonical | Duplicate(s) | Status | PRD |
|---|---|---|---|
| `agent_reports` | `deliverables` | in flight (PRD-133b) | PRD-129 → PRD-133b |
| `orchestration_runs` | `agent_executions`, `workflow_executions`? | confirm | — |
| `tasks` (board/mission) | `board_tasks`, `orchestration_tasks` | confirm | — |
| `messages` | `chat_messages` | dead | — |
| `chats` | `conversations` | dead | — |
| `workspace_tool_config` | `tenant_tool_config` | dead | — |
| `component_metrics` | `performance_data`, `system_metrics` | dead | — |
| `audit_logs` | `user_activities` | confirm | — |

---

## 6. Frontend hook duplication

Confirmed triplets/duplicates from `ls frontend/hooks/`:

- `use-memory-api.ts` / `use-memory-v1-api.ts` / `use-memory-explorer-api.ts` → **merge to one**
- `use-multi-agent.ts` / `use-multi-agent-verified.ts` → **merge to one, keep verified path**
- `use-board-tasks.ts` / `use-board-tasks-api.ts` → **merge**
- `use-reports-api.ts` / `use-deliverables-api.ts` → **merge after Wave 4 deliverables unification**
- `use-api.ts` / `use-api-debug.ts` / `use-api-toggle.js` → JS file is an outlier; migrate to TS, merge debug into main via flag
- `use-processing-queue-api.ts` / `use-processing-queue-websocket.ts` → fine if SSE vs REST split is intentional; else merge

Additional fresh suspects once Wave 2 collapses backend routes — any hook still calling a killed route needs to be killed or re-pointed.

---

## 7. main.py mount-list audit (quick read)

Per header comment in `main.py`:
> `DO NOT COMMENT OUT ANYTHING IN THIS FILE.`

…which is itself a consolidation tell. Treat main.py as append-only has been the operating principle; Wave 2 lifts that by making mounts declarative (e.g. auto-mount from `api/__init__.py` with explicit tag → domain mapping).

Ordering comments inside `main.py` reveal **catch-all-route conflicts** we've patched by hand:
- `widget_workflows_router` MUST be before `workflows_router` (else `/{id}` eats it).
- `memory_stats_router` MUST be before `memory_router` (same pattern).
- `document_generation_router` MUST be before `documents_router` (same pattern).

This manual ordering discipline is a Wave 2 refactor target — move ambiguous catch-alls under explicit path prefixes and ordering becomes irrelevant.

---

## 8. What we don't know yet

Blocked on PRD-135 Phase 1 (DB snapshot) and Phase 2 (code→DB edge walker) for these questions:

1. Of the 109 modelled tables, how many get zero writes/reads in the last 7 days? (needs `pg_stat_user_tables`)
2. Of the 103 routers, how many handle zero requests in the last 7 days? (needs log-relay or frontend call-graph)
3. Which columns in live tables are never read by any handler? (needs AST walker)
4. Which indexes have `idx_scan = 0`? (needs `pg_stat_user_indexes`)
5. Where are the heavy writes concentrated (write amplification)? (needs pg_stat_statements)

**These are the Wave 1 data-driven decisions.** Until PRD-135 Phases 1-3 land, we work from the static catalog in this doc + the duplicate list from memory. The static list is ~80% complete; runtime closes the last 20%.

---

## 9. Companion agent report

An Explore agent was dispatched alongside this audit to produce a dense duplication map (API × model × route × hook). When that report lands it will be appended as `01a-EXPLORE-REPORT.md` and referenced from §2 and §4 of this doc.

---

**Summary:** 0.2 starts with 103 routers, 109 tables, 16 orphan tables, 7 "analytics" routers, 5 "marketplace" routers, three overlapping code trees, and about 60% of the autonomous-flow story intact but not composed. Target: 25 routers, ≤75 tables, one canonical per concept, one UX shell, and the story composed end-to-end.
