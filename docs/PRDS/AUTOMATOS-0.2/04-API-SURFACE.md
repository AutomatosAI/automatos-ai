# AUTOMATOS 0.2 — Target API Surface

**Purpose:** Concrete target for the 103 → ~25 router collapse. Every current router file has a verdict: keep, merge, rename, or delete.

---

## 1. Target router list (10 canonical + infra + legacy-until-deprecated)

### Canonical domain routers (one per domain from [03-DOMAIN-MODEL.md](./03-DOMAIN-MODEL.md))

```
orchestrator/api/
├── goals.py           # chat, missions, recipes, plans — the run object
├── agents.py          # roster, CRUD, skills/tools assignment, onboarding
├── skills.py          # catalog, install, version
├── tools.py           # registry, exec, permissions, Composio, MCP
├── knowledge.py       # documents, memory, graph, rag, database_knowledge (sub-paths)
├── deliverables.py    # list, get, grade, download (Wave 4)
├── workspaces.py      # CRUD, members, invitations, models
├── marketplace.py     # catalog browse + install for skills/models/plugins/widgets/templates
├── analytics.py       # dashboards, KPIs, insights, routing decisions
└── admin.py           # system settings, auth, budget, system prompts
```

### Infra / integrations (kept separate on purpose)

```
├── webhooks/
│   ├── github.py       # github PR automation
│   ├── recipe.py       # recipe trigger webhook (no auth)
│   ├── workspace.py    # general workspace webhook (no auth)
│   └── shopify.py      # shopify app store provisioning + webhook
├── cache.py            # cache admin + stats
├── system.py           # healthz, readyz, version, status
└── ws.py               # websocket channels (activity, run stream, heartbeat)
```

### Widgets (kept — each embeds on marketing / customer sites)

```
├── widgets/
│   ├── __init__.py
│   ├── memory.py       # US-013 — inline agent memory panel
│   ├── workflows.py    # US-014 — workflow pause/resume/cancel control
│   ├── email.py        # email-in widget
│   └── marketplace.py  # embedded widget marketplace
```

### Legacy (mounted with 301 redirect until deprecation window closes)

```
├── deprecated/
│   ├── chatbot_llm.py  # redirect to goals/chat
│   └── workflows.py    # redirect to goals/recipes
```

**Total target:** 10 canonical + 4 infra + 4 widgets + 2 legacy = **~20-25 files**, down from 103.

---

## 2. Per-current-router verdict table

Legend: **K** = keep as canonical, **M** = merge into canonical, **R** = rename, **D** = delete, **I** = investigate.

| # | Current file | Verdict | Target | Notes |
|---|---|---|---|---|
| 1 | `activity.py` | M | `goals.py` (sub: activity feed) OR `agents.py` (heartbeat activity) | PRD-72 command centre; feeds both domains |
| 2 | `admin_plugins.py` | M | `admin.py` (sub: `/admin/plugins`) | |
| 3 | `admin_prompts.py` | M | `admin.py` (sub: `/admin/prompts`) | |
| 4 | `admin_workspaces.py` | M | `admin.py` (sub: `/admin/workspaces`) | |
| 5 | `agent_endpoints.py` | M | `agents.py` | Why two agent files? historical |
| 6 | `agent_plugins.py` | M | `agents.py` (sub: `/agents/{id}/plugins`) OR `marketplace.py` | probably agents |
| 7 | `agents.py` | K | `agents.py` | canonical |
| 8 | `analytics.py` | M | `analytics.py` (new consolidated) | |
| 9 | `analytics_api.py` | M | `analytics.py` | |
| 10 | `analytics_real.py` | M | `analytics.py` (sub: `/analytics/dashboard`, `/analytics/performance`) | |
| 11 | `analytics_charts.py` | M | `analytics.py` (sub: `/analytics/charts`) or keep separate (PandasAI) |
| 12 | `anthropic_client.py` | D | — | utility file mis-placed in api/; move to `core/llm/` and delete the router stub |
| 13 | `api_keys.py` | M | `admin.py` (sub: `/admin/api-keys`) | |
| 14 | `api_playbooks.py` | M | `goals.py` (sub: `/goals/recipes` — playbook = recipe at user level) | |
| 15 | `attachments.py` | M | `goals.py` (sub: `/goals/runs/{id}/attachments`) | PRD-127 |
| 16 | `blog.py` | **D** | — | **unmounted, dead** |
| 17 | `board_tasks.py` | M | `agents.py` (sub: `/agents/tasks/board`) — or `goals.py` if board tasks become run-subtasks | |
| 18 | `bug_reports.py` | M | `admin.py` or keep as widget (`widgets/bug-reports.py`); low priority |
| 19 | `cache.py` | K | `cache.py` | infra; keep |
| 20 | `channels.py` | K (but move) | `channels/` (not in api/) | external I/O, not route |
| 21 | `chat_voice.py` | M | `goals.py` (sub: `/goals/chat/voice`) |
| 22 | `chat.py` | M | `goals.py` (sub: `/goals/chat`) |
| 23 | `chatbot_llm.py` | M→deprecated | `deprecated/chatbot_llm.py` | 301 to goals/chat |
| 24 | `cloud_documents.py` | M | `knowledge.py` (sub: `/knowledge/cloud`) |
| 25 | `codegraph.py` | M | `knowledge.py` (sub: `/knowledge/graph/code`) |
| 26 | `composio_analytics.py` | M | `analytics.py` (sub: `/analytics/composio`) |
| 27 | `composio.py` | M | `tools.py` (sub: `/tools/composio`) |
| 28 | `context_engineering.py` | M | `admin.py` (sub: `/admin/context`) or `knowledge.py` sub | context is infra-ish |
| 29 | `context_policy.py` | M | `admin.py` (sub: `/admin/context/policies`) |
| 30 | `context_summarization.py` | M | `admin.py` (sub: `/admin/context/summarization`) |
| 31 | `context.py` | M | `admin.py` (sub: `/admin/context`) |
| 32 | `credentials.py` | M | `admin.py` (sub: `/admin/credentials`) |
| 33 | `dashboard_integration.py` | I | likely → `analytics.py` | audit use |
| 34 | `database_analytics.py` | M | `analytics.py` (sub: `/analytics/database`) |
| 35 | `database_knowledge_simple.py` | **D** | — | confirmed dead duplicate |
| 36 | `database_knowledge.py` | M | `knowledge.py` (sub: `/knowledge/database`) |
| 37 | `deliverables.py` | K | `deliverables.py` | canonical (Wave 4) |
| 38 | `document_generation.py` | M | `knowledge.py` (sub: `/knowledge/documents/generate`) |
| 39 | `documents.py` | M | `knowledge.py` (sub: `/knowledge/documents`) |
| 40 | `execution_history.py` | M | `goals.py` (sub: `/goals/runs/history`) — runs = missions/workflows unified |
| 41 | `generated_images.py` | M | `deliverables.py` (sub: `/deliverables?type=image`) OR `knowledge.py` |
| 42 | `github_webhooks.py` | M→move | `webhooks/github.py` | |
| 43 | `heartbeat.py` | M | `agents.py` (sub: `/agents/heartbeat`) + `ws.py` for stream |
| 44 | `insights.py` | M | `analytics.py` (sub: `/analytics/insights`) |
| 45 | `knowledge_graph.py` | M | `knowledge.py` (sub: `/knowledge/graph`) |
| 46 | `knowledge_multimodal.py` | M | `knowledge.py` (sub: `/knowledge/multimodal`) |
| 47 | `knowledge.py` | K | `knowledge.py` | canonical |
| 48 | `kpi_api.py` | M | `analytics.py` (sub: `/analytics/kpi`) |
| 49 | `learning.py` | M | `analytics.py` (sub: `/analytics/learning`) — feedback loop telemetry |
| 50 | `llm_analytics.py` | M | `analytics.py` (sub: `/analytics/llm`) |
| 51 | `llm_marketplace.py` | M | `marketplace.py` (sub: `/marketplace/models`) |
| 52 | `marketplace_plugins.py` | M | `marketplace.py` (sub: `/marketplace/plugins`) |
| 53 | `marketplace.py` | K | `marketplace.py` | canonical |
| 54 | `memory_stats.py` | M | `knowledge.py` (sub: `/knowledge/memory/stats`) |
| 55 | `memory.py` | M | `knowledge.py` (sub: `/knowledge/memory`) |
| 56 | `missions.py` | M→rename | `goals.py` (sub: `/goals/missions`) — kind=`mission` on run |
| 57 | `models_endpoints.py` | M | `admin.py` (sub: `/admin/models`) — LLM model admin |
| 58 | `notifications.py` | K (but move) | `modules/notifications/` → `api/notifications.py` mini surface | PRD-128 |
| 59 | `onboarding_agents.py` | M | `agents.py` (sub: `/agents/onboarding`) |
| 60 | `openrouter_marketplace.py` | M | `marketplace.py` (sub: `/marketplace/models/openrouter`) |
| 61 | `patterns.py` | M | `analytics.py` or `knowledge.py` | patterns = learned coordination patterns (PRD-11) |
| 62 | `permissions.py` | M | `admin.py` (sub: `/admin/permissions`) |
| 63 | `personas.py` | M | `agents.py` (sub: `/agents/personas`) |
| 64 | `problems.py` | I, likely M | `analytics.py` (sub: `/analytics/problems`) — part of smart-diagnosis suite |
| 65 | `query.py` | I | audit; possibly duplicate of `knowledge.py/search` |
| 66 | `rag_feedback.py` | **D** | — | unmounted, no backing model |
| 67 | `recipe_executor.py` | I→move | `services/recipe_executor_service.py` — not an API router |
| 68 | `recommendations.py` | M | `analytics.py` (sub: `/analytics/recommendations`) |
| 69 | `reports.py` | M→deprecated | merged into `deliverables.py` (Wave 4) |
| 70 | `routing.py` | M | `admin.py` (sub: `/admin/routing`) OR `tools.py` (sub) |
| 71 | `scheduled_tasks.py` | M | `goals.py` (sub: `/goals/recipes/schedules`) or `agents.py` — PRD-77 agent self-scheduling |
| 72 | `shopify.py` | M→move | `webhooks/shopify.py` |
| 73 | `skills.py` | K | `skills.py` | canonical |
| 74 | `solutions.py` | M | `analytics.py` (sub: `/analytics/solutions`) |
| 75 | `statistics.py` | M | `analytics.py` — likely dupe of KPI/analytics |
| 76 | `synthesis.py` | M | `analytics.py` (sub: `/analytics/synthesis`) |
| 77 | `system_settings.py` | M | `admin.py` (sub: `/admin/system/settings`) |
| 78 | `system.py` | K (rename) | `system.py` — healthz only; admin pieces move |
| 79 | `tasks.py` | M | `agents.py` (sub: `/agents/tasks`) — workspace tasks per agent |
| 80 | `team.py` | M | `workspaces.py` (sub: `/workspaces/{ws}/members`) |
| 81 | `templates.py` | M | `marketplace.py` (sub: `/marketplace/templates`) |
| 82 | `tool_assignments.py` | M | `agents.py` (sub: `/agents/{id}/tools`) |
| 83 | `tools.py` | K | `tools.py` | canonical |
| 84 | `user_api_keys.py` | M | `admin.py` (sub: `/admin/api-keys/user`) |
| 85 | `voice_profiles.py` | M | `agents.py` (sub: `/agents/{id}/voice`) if kept, else deprecate |
| 86 | `webhooks.py` | M→move | `webhooks/__init__.py` root |
| 87 | `widget_email.py` | M→move | `widgets/email.py` |
| 88 | `widget_marketplace.py` | M→move | `widgets/marketplace.py` |
| 89 | `widget_memory.py` | M→move | `widgets/memory.py` |
| 90 | `widget_workflows.py` | M→move | `widgets/workflows.py` |
| 91 | `widgets/*` | K (reorganize) | `widgets/` | |
| 92 | `wizard.py` | M | `goals.py` (sub: `/goals/plan` — wizard = plan-mode run) |
| 93 | `workflow_history.py` | M | `goals.py` (sub: `/goals/runs/history`) |
| 94 | `workflow_recipes.py` | M | `goals.py` (sub: `/goals/recipes`) |
| 95 | `workflow_templates.py` | M→deprecated | redirect to `/goals/recipes` (renamed per PRD-US-009) |
| 96 | `workflows.py` | M→deprecated | redirect to `/goals/recipes` |
| 97 | `workspace_exec.py` | **I/D** | investigate; likely delete |
| 98 | `workspace_files.py` | M | `workspaces.py` (sub: `/workspaces/{ws}/files`) |
| 99 | `workspace_github.py` | M | `workspaces.py` (sub: `/workspaces/{ws}/github`) |
| 100 | `workspace_plugins.py` | M | `workspaces.py` (sub: `/workspaces/{ws}/plugins`) |
| 101 | `workspace_skills.py` | M | `workspaces.py` (sub: `/workspaces/{ws}/skills`) |
| 102 | `workspaces.py` | K | `workspaces.py` | canonical |
| — | `widget_installation.py`, `widget_review.py` | M→move | `widgets/` |

**Confirmed deletes:** 4 (blog, database_knowledge_simple, rag_feedback, workspace_exec + anthropic_client router stub)
**Confirmed investigates:** 3 (recipe_executor, workspace_exec depth, query)
**Everything else collapses into the 10 canonical + 4 infra + 4 widgets + 2 deprecated.**

---

## 3. URL path scheme (after collapse)

```
/api/goals/                          — list runs
/api/goals/chat                      — new chat run
/api/goals/missions                  — new mission run
/api/goals/recipes                   — list/create recipes
/api/goals/recipes/{id}/run          — kick off a scheduled recipe
/api/goals/plan                      — plan-mode (wizard)
/api/goals/runs/{id}                 — run detail (any kind)
/api/goals/runs/{id}/events          — SSE stream
/api/goals/runs/{id}/attachments     — attachments

/api/agents/                         — list
/api/agents/{id}                     — CRUD
/api/agents/{id}/skills              — skill assignment
/api/agents/{id}/tools               — tool assignment
/api/agents/{id}/personas            — persona library
/api/agents/{id}/heartbeat           — heartbeat control
/api/agents/tasks                    — task queue per agent
/api/agents/tasks/board              — board view

/api/skills/                         — catalog browse
/api/skills/{id}                     — detail
/api/skills/{id}/install             — install to workspace

/api/tools/                          — registry
/api/tools/composio                  — composio apps/actions
/api/tools/mcp                       — MCP servers

/api/knowledge/documents             — CRUD
/api/knowledge/documents/generate    — document generation
/api/knowledge/cloud                 — cloud sync (Drive, Dropbox, S3)
/api/knowledge/memory                — memory CRUD
/api/knowledge/memory/stats          — memory observability
/api/knowledge/graph                 — knowledge graph
/api/knowledge/graph/code            — codegraph (PRD-11)
/api/knowledge/multimodal            — image/audio search
/api/knowledge/database              — NL2SQL & database knowledge
/api/knowledge/search                — unified search

/api/deliverables/                   — list (filter by run, agent, type, grade)
/api/deliverables/{id}               — detail
/api/deliverables/{id}/download      — file bytes (S3 pre-signed)
/api/deliverables/{id}/grade         — set grade + tags

/api/workspaces/                     — list
/api/workspaces/{id}                 — CRUD
/api/workspaces/{id}/members         — team
/api/workspaces/{id}/invitations     — invite
/api/workspaces/{id}/files           — file browser (PRD-66)
/api/workspaces/{id}/github          — github connection
/api/workspaces/{id}/plugins         — enabled plugins
/api/workspaces/{id}/skills          — enabled skills
/api/workspaces/{id}/models          — model config

/api/marketplace/skills              — browse skills
/api/marketplace/models              — browse models (openrouter + providers)
/api/marketplace/plugins             — browse plugins
/api/marketplace/widgets             — browse widgets
/api/marketplace/templates           — browse workspace templates

/api/analytics/dashboard             — KPI dashboard
/api/analytics/llm                   — token + cost
/api/analytics/agents                — agent ranking
/api/analytics/performance           — SLA, bottlenecks, p95/p99
/api/analytics/routing               — routing decisions
/api/analytics/insights              — generated insights
/api/analytics/charts                — PandasAI chart gen

/api/admin/system/settings           — system settings
/api/admin/system/prompts            — system prompts + versions
/api/admin/routing                   — routing rules
/api/admin/credentials               — credentials
/api/admin/api-keys                  — SDK + user API keys
/api/admin/permissions               — permissions
/api/admin/models                    — admin LLM models catalog
/api/admin/workspaces                — admin-level workspace control
/api/admin/context/*                 — context engineering

/webhooks/github                     — github PR
/webhooks/recipe                     — recipe trigger
/webhooks/workspace                  — generic
/webhooks/shopify                    — shopify app store

/api/system/healthz                  — health
/api/system/readyz                   — readiness
/api/system/version                  — build info

/ws/activity                         — activity stream
/ws/runs/{id}                        — run stream
/ws/heartbeat                        — heartbeat stream
```

---

## 4. Response envelope (canonical after 0.2)

All domains adopt a single envelope (per `common/patterns.md` API response format rule):

```json
{
  "success": true,
  "data": { /* domain-specific */ },
  "error": null,
  "meta": {
    "trace_id": "...",
    "workspace_id": "...",
    "pagination": {"total": 123, "limit": 20, "offset": 0}  // when paginated
  }
}
```

Legacy non-enveloped responses from pre-0.2 routers get wrapped in an adapter during Wave 2 to avoid frontend churn.

---

## 5. Migration mechanics (Wave 2 playbook)

1. **Create `api/{domain}.py`** as the canonical router.
2. **Copy handlers** from source router(s). Keep handler names; change only the router prefix.
3. **Adapt tests** (pytest imports move one path up; otherwise unchanged).
4. **Mount the new router in `main.py`** before the legacy one.
5. **Add redirect to legacy path:** old router returns `308 Permanent Redirect` with `Deprecation: true` header and `Sunset: <date>` header pointing to new path.
6. **Update frontend hook** to call canonical path.
7. **Measure:** with PRD-135 Phase 4 runtime overlay, watch legacy-path traffic for 14 days.
8. **Delete legacy router** when traffic = 0.

**Frontend side:** one sweep per domain during Wave 3 replaces hooks; the 308 redirect buys time if a hook is missed.

---

## 6. What's not in the target

- **No versioning prefix** (`/v2/...`). 0.2 is not a breaking API release; it's a reshape with 308s. A versioned split is 1.0.
- **No GraphQL.** REST stays. The envelope and pagination gives us ~70% of GraphQL's selection-set ergonomics.
- **No per-widget API files.** Widgets are HTML that hit the same canonical API as the app; the only widget-specific endpoints are render hooks (`/widgets/{id}/render`).

---

## 7. Success metric for Wave 2

- `ls orchestrator/api/*.py | wc -l` ≤ 25
- `grep -c include_router orchestrator/main.py` ≤ 30
- `/graphify db-report dead-routes` returns 0 rows
- 0 frontend hooks calling paths with `Deprecation: true` header
- Alias paths 308-redirect to canonical paths for one release cycle, then 410 Gone.

See [09-SUCCESS-METRICS.md](./09-SUCCESS-METRICS.md) for the full scorecard.
