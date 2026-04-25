# AUTOMATOS 0.2 — Domain Model

**Purpose:** Declare the 10 canonical domains, their code home, their canonical models/tables, and the code-organization rule. This is the filter every Wave-2/3 PR uses: "does this code live in the right place?"

---

## 1. The 10 domains

| # | Domain | What it owns | Primary code home |
|---|---|---|---|
| 1 | **Goals** | Chats, missions, recipes, plans — the one run object | `core/coordinator/` + `modules/coordination/` |
| 2 | **Agents** | Agent definitions, roster, heartbeat, skill/tool assignments | `modules/agents/` + `core/agent_factory.py` |
| 3 | **Skills** | Skill catalog, install, version, injection | `modules/tools/skills/` + `modules/context/sections/skills.py` |
| 4 | **Tools** | Tool registry (core, platform, workspace, Composio, MCP) | `core/routing/` + `modules/tools/` |
| 5 | **Knowledge** | Documents, vectors, graph, field memory, database NL2SQL | `modules/knowledge/` + `modules/memory/` + `modules/rag/` |
| 6 | **Deliverables** | Every agent output — unified object | `services/deliverable_service.py` + `modules/deliverables/` (new) |
| 7 | **Workspaces** | Tenant isolation, members, roles, config | `core/workspaces/` |
| 8 | **Marketplace** | Skills, models, plugins, widgets, templates catalog | `core/marketplace/` (consolidated) |
| 9 | **Analytics** | Runtime stats, KPIs, insights across runs and agents | `modules/analytics/` (consolidated from 7 existing surfaces) |
| 10 | **Admin** | Auth, billing/budget, system settings, system prompts, admin workspaces | `core/auth/` + `core/billing/` (new) + `core/admin/` (new) |

Anything that doesn't fit one of these 10 is either infrastructure (`core/db/`, `core/redis/`, `core/llm/`) or it's wrong.

---

## 2. Domain boundaries (who owns what, with examples)

### 1. Goals
- **Owns:** `chats`, `messages`, `orchestration_runs` (→ renamed `missions`), `workflow_recipes`, `recipe_executions`, `business_profiles` (goal=configure-workspace).
- **Does NOT own:** task-level execution records (that's Agents domain), deliverables.
- **Key concept:** One `run` object with a `kind` enum (`chat | mission | recipe | plan`). All current surfaces collapse into this.

### 2. Agents
- **Owns:** `agents`, `agent_blueprints`, `agent_tool_permissions`, `agent_app_assignments`, agent heartbeats, agent_reports (until Wave 4 collapses to deliverables), onboarding agents.
- **Does NOT own:** skill definitions (Skills domain), tool definitions (Tools), orchestration logic (Goals).

### 3. Skills
- **Owns:** `skills`, `skill_files`, `skill_versions`, `skill_sources`, `skill_audit_log`, `workspace_enabled_skills`.
- **Key split:** Skill = user-facing capability bundle; the implementation uses tools + prompts. Skills live in `../automatos-skills/` repo (gold standards: sentinel, scout).

### 4. Tools
- **Owns:** `tools`, `tool_categories`, `tool_configurations`, `tool_credentials`, `tool_reviews`, `tool_installation_requests`, `workspace_tool_config`, Composio tables.
- **Ownership split:** Core tools (workspace exec, read_file, grep) + platform tools (install, browse, heartbeat) + Composio tools (500+ integrations) + MCP tools. All dispatch through `core/routing/tool_router.py`.

### 5. Knowledge
- **Owns:** `documents`, `document_templates`, `external_knowledge`, `knowledge_nodes`, `knowledge_edges`, `memory_items`, `memory_short_term`, `learning_outcomes`, `rag_configurations`, `database_knowledge_sources`, `database_relationships`, `database_query_templates`, NL2SQL training/benchmark tables.
- **Key split (domain sub-lanes):** Documents (uploaded/generated) | Memory (ephemeral + field) | Graph (nodes + edges) | Database (NL2SQL).

### 6. Deliverables
- **Owns (after Wave 4):** A single `deliverable` row per agent output. Storage path, mime type, agent_id, run_id, workspace_id, mission_task_id (nullable), grade, tags.
- **Collapses:** `artifacts` (chat artifacts), `agent_reports.file_path`, workspace file outputs from missions, `generated_images` metadata.
- **Does NOT own:** the files themselves — those stay in S3 under `s3://automatos-ai/workspaces/{ws}/deliverables/{id}/`.

### 7. Workspaces
- **Owns:** `workspaces`, `workspace_members`, `workspace_invitations`, `workspace_models`, `business_profiles` (ties to Goals-plan but stored here).
- **Infra layer:** multi-tenant filter applied at every query via `core/workspaces/context.py`. No route in any domain may skip the workspace guard.

### 8. Marketplace
- **Owns:** `marketplace_plugins`, `plugin_categories`, `plugin_security_scans`, `plugin_sync_history`, `workspace_enabled_plugins`, `marketplace_widgets`, `widget_installations`, `widget_reviews`, `openrouter_models_cache`, `openrouter_sync_jobs`.
- **Concept:** Catalog + install is one domain. Install affects Skills/Tools/Agents/Workspaces but the catalog/transaction lives here.

### 9. Analytics
- **Owns:** `llm_usage`, `component_metrics`, `evaluation_results`, `benchmark_assessments`, `integration_analyses`, `routing_decisions`, `unrouted_events`, `database_query_audit`, `semantic_metrics`.
- **Read-mostly domain:** aggregates data produced by Goals/Agents/Tools/Knowledge for dashboards. Must not double-write business state.

### 10. Admin
- **Owns:** `users`, `credentials`, `credential_types`, `credential_audit_logs`, `api_keys`, `user_api_keys`, `sdk_api_keys`, `system_settings`, `system_configurations`, `system_prompts` + versions + eval_runs, `permission_audit_logs`, `context_policies`.
- **Includes:** budget/billing (new — not modelled today; placeholder for PRD-105).

---

## 3. The code-organization rule (after 0.2)

```
orchestrator/
├── core/                   # infrastructure primitives (long-lived)
│   ├── auth/               # sessions, RBAC, SDK keys
│   ├── db/                 # SQLAlchemy engine, session factory
│   ├── redis/              # redis client, pubsub
│   ├── s3/                 # S3 + S3 Vectors client
│   ├── llm/                # OpenRouter/Anthropic/OpenAI dispatch + defaults
│   ├── routing/            # tool_router.py, intent_classifier (core)
│   ├── coordinator/        # sequential mission coordinator (PRD-82A)
│   ├── agent_factory.py    # agent runtime
│   ├── graph_storage.py    # knowledge graph persistence
│   ├── workspaces/         # tenant isolation + context
│   ├── marketplace/        # catalog service (new — consolidates marketplace surfaces)
│   ├── billing/            # budget + cost (new — PRD-105)
│   └── admin/              # system settings, admin actions (new)
├── modules/                # domain packages (each owns models + logic)
│   ├── agents/
│   ├── coordination/       # mission scheduling (parallel extensions land here)
│   ├── deliverables/       # new — unified deliverable domain (Wave 4)
│   ├── knowledge/
│   ├── memory/
│   ├── rag/
│   ├── nl2sql/             # sub-package of knowledge? or standalone? — decide in Wave 2
│   ├── tools/
│   ├── analytics/          # new — consolidates 7 analytics files
│   └── context/            # context-engineering sections
├── services/               # cross-domain orchestration
│   ├── heartbeat_service.py
│   ├── coordinator_service.py
│   ├── deliverable_service.py
│   ├── report_service.py   # deprecated alias, folds into deliverable_service
│   └── ...
├── api/                    # thin HTTP handlers, one file per domain
│   ├── goals.py            # chat + missions + recipes + plans
│   ├── agents.py
│   ├── skills.py
│   ├── tools.py
│   ├── knowledge.py
│   ├── deliverables.py
│   ├── workspaces.py
│   ├── marketplace.py
│   ├── analytics.py
│   └── admin.py
├── channels/               # external channels (email, slack, telegram)
├── consumers/              # chatbot, workers
├── main.py                 # thin; auto-mount from api/__init__.py
└── config.py               # single source of env
```

**Rule:** if a file doesn't fit the above tree, it's wrong.

---

## 4. Ownership matrix (domain × code tree)

| Domain | `core/` | `modules/` | `services/` | `api/` | Notes |
|---|---|---|---|---|---|
| Goals | coordinator/ | coordination/ | coordinator_service, task_reconciler | goals.py | mission = orchestration_run renamed |
| Agents | agent_factory.py | agents/ | heartbeat_service, activity_service | agents.py | |
| Skills | — | tools/skills/ | — | skills.py | skill source-of-truth in separate repo |
| Tools | routing/ | tools/ | tool_manifest_service | tools.py | |
| Knowledge | graph_storage.py | knowledge/, memory/, rag/, nl2sql/ | memory_archival_job, memory_jobs | knowledge.py | |
| Deliverables | — | deliverables/ (new) | deliverable_service, report_service (fold) | deliverables.py | |
| Workspaces | workspaces/ | — | — | workspaces.py | |
| Marketplace | marketplace/ (new) | — | — | marketplace.py | |
| Analytics | — | analytics/ (new) | — | analytics.py | |
| Admin | auth/, admin/ (new), billing/ (new) | — | — | admin.py | |

---

## 5. What this model kills

- **The 103-router flat list.** Replaced by 10 routers (one per domain). See [04-API-SURFACE.md](./04-API-SURFACE.md).
- **The core/services/modules ambiguity.** Each has a defined role now.
- **The "analytics lives everywhere" problem.** `analytics/` module is the single home for aggregate telemetry; per-domain metrics (like `composio_stats_cache`) stay in their domain module.
- **The "where does output live" problem.** `deliverables/` module + service is the only answer.

---

## 6. What this model does NOT answer (deferred to Wave 5+)

- **Where do Channels (email, slack, telegram) live?** Currently `orchestrator/channels/`. Probably stays — they are I/O adapters, not domain. But: should channel-linked messages write into the Goals domain as chat runs? Yes (Wave 5 decision).
- **Where does Consumer (chatbot background worker) live?** Currently `orchestrator/consumers/`. Stays; it's a process type, not a domain.
- **Where does `codegraph` live?** Currently `orchestrator/api/codegraph.py` + `orchestrator/modules/codegraph/`. Stays as sub-module of Knowledge.
- **Where does `intake` (Mission Zero wizard) live?** Currently `modules/intake/`. Becomes a sub-package of Goals (intake → plan-mode run).

These are deliberate deferrals — Wave 5 picks them up with more context.

---

## 7. Frontend mirror (the four tabs × 10 domains)

| Tab | Covers domains |
|---|---|
| **Goals** | Goals + Agents (as the "crew" visible at run-time) |
| **Deliverables** | Deliverables only (plus grading + tags from Analytics) |
| **Knowledge** | Knowledge (all sub-lanes) + Skills (installed skills are "what the workspace knows how to do") |
| **Agents** | Agents + Tools + Marketplace (install new) |

Admin, Analytics, Workspaces collapse behind a **Workspace Settings** gear + **Advanced** sub-tabs. See [06-FRONTEND-SURFACE.md](./06-FRONTEND-SURFACE.md).

---

**Test for this doc:** handed this page, a new engineer can answer "where does a new feature for X live?" in under 30 seconds for 9 of 10 X's. If they can't, a domain definition needs sharpening.
