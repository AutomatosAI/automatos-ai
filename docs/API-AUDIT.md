# API Endpoint Audit — Automatos AI Platform

**Date:** 2026-02-23
**Scope:** All `orchestrator/api/*.py` files cross-referenced with `main.py` mounts and frontend usage

---

## Executive Summary

| Metric | Count |
|--------|-------|
| API files in `orchestrator/api/` | 85 |
| Files with route decorators | 76 |
| Total route decorators (`@router.get/post/put/delete/patch`) | 564 |
| Routers mounted in `main.py` (`app.include_router()`) | 79 |
| Optional routers (try/except) | 12 |
| **Unmounted API files** | **4** (see below) |

---

## Unmounted API Files (Dead or Orphaned)

These files define routers but are NOT imported or mounted in `main.py`:

| File | Endpoints | Status |
|------|-----------|--------|
| `database_knowledge_simple.py` | 4 | Likely superseded by `database_knowledge.py` (22 endpoints) |
| `rag_feedback.py` | 3 | Orphaned — prefix `/api/rag/feedback` |
| `anthropic_client.py` | 0 | Utility file, not a router |
| `recipe_executor.py` | 0 | Utility file, not a router |

**Note:** `document_processing.py` is also unmounted, with a comment in `main.py`: "document_processing_router removed — use api/documents.py instead."

---

## Duplicate Router Prefixes (Route Conflicts)

Multiple files share the same prefix, which can cause route shadowing:

| Prefix | Files | Risk |
|--------|-------|------|
| `/api/agents` | `agents.py` (12), `agent_endpoints.py` (13), `agent_plugins.py` (3) | HIGH — 28 endpoints on same prefix |
| `/api/analytics` | `analytics_api.py` (11), `analytics_real.py` (14) | HIGH — later mount shadows earlier |
| `/api/documents` | `documents.py` (17), `document_generation.py` (9) | MEDIUM — may have distinct sub-paths |
| `/api/knowledge` | `knowledge.py` (1), `knowledge_graph.py` (7), `knowledge_multimodal.py` (6) | MEDIUM — 14 endpoints |
| `/api/v1/memory` | `memory.py` (18), `memory_stats.py` (3) | LOW — likely distinct sub-paths |
| `/api/system` | `system.py` (20), `statistics.py` (5) | MEDIUM — 25 endpoints |
| `/api/workspaces` | `workspaces.py` (8), `workspace_plugins.py` (3) | LOW — likely distinct sub-paths |
| `/api/knowledge/sources/database` | `database_knowledge.py` (22), `database_knowledge_simple.py` (4, unmounted) | NONE — simple variant unmounted |

---

## API Versioning Inconsistencies

| Prefix | File | Issue |
|--------|------|-------|
| `/analytics` (no `/api/`) | `analytics.py` | Missing `/api/` prefix — inconsistent with all other routers |
| `/permissions` (no `/api/`) | `permissions.py` | Missing `/api/` prefix |
| `/v1/workflows` (no `/api/`) | `workflow_history.py` | Should be `/api/v1/workflows` |
| `/api/v1/memory` | `memory.py`, `memory_stats.py` | Versioned while all others are unversioned |
| `/api/v1/skills` | `skills.py` | Versioned while all others are unversioned |
| `/api/v1/benchmarking` | `benchmarking.py` | Versioned while all others are unversioned |

---

## Endpoints by Domain

### Core Chat & Messaging (20 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `chat.py` | `/api/chat` | 9 | GET, POST |
| `chatbot_llm.py` | `/api/chatbot` | 2 | POST |
| `channels.py` | `/api/channels` | 8 | GET, POST, PUT, DELETE |
| `heartbeat.py` | `/api/heartbeat` | 6 | GET, POST |

### Agents (28 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `agents.py` | `/api/agents` | 12 | GET, POST, PUT, DELETE |
| `agent_endpoints.py` | `/api/agents` | 13 | GET, POST, PUT |
| `agent_plugins.py` | `/api/agents` | 3 | GET, POST, DELETE |

### Workflows & Recipes (54 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `workflows.py` | `/api/workflows` | 24 | GET, POST, PUT, DELETE |
| `workflow_recipes.py` | `/api/workflow-recipes` | 18 | GET, POST, PUT, DELETE |
| `workflow_templates.py` | `/api/workflow-templates` | 8 | GET, POST, PUT, DELETE |
| `workflow_history.py` | `/v1/workflows` | 4 | GET |

### Documents & Knowledge (75 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `documents.py` | `/api/documents` | 17 | GET, POST, PUT, DELETE |
| `document_generation.py` | `/api/documents` | 9 | GET, POST |
| `cloud_documents.py` | `/api/cloud-documents` | 9 | GET, POST, PUT |
| `knowledge.py` | `/api/knowledge` | 1 | POST |
| `knowledge_graph.py` | `/api/knowledge` | 7 | GET, POST |
| `knowledge_multimodal.py` | `/api/knowledge` | 6 | GET, POST |
| `database_knowledge.py` | `/api/knowledge/sources/database` | 22 | GET, POST, PUT, DELETE |

### Analytics & Reporting (49 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `analytics_real.py` | `/api/analytics` | 14 | GET |
| `analytics_api.py` | `/api/analytics` | 11 | GET, POST |
| `analytics.py` | `/analytics` | 6 | GET |
| `llm_analytics.py` | `/api/analytics/llm` | 10 | GET |
| `composio_analytics.py` | `/api/analytics/composio` | 3 | GET |
| `analytics_charts.py` | `/api/analytics/charts` | 2 | GET, POST |
| `database_analytics.py` | `/api/database/analytics` | 3 | GET |

### Tools & Integrations (30 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `tools.py` | `/api/tools` | 16 | GET, POST, PUT, DELETE |
| `composio.py` | `/api/composio` | 14 | GET, POST |

### Marketplace (26 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `marketplace.py` | `/api/marketplace` | 9 | GET, POST |
| `llm_marketplace.py` | `/api/marketplace/llm` | 9 | GET, POST |
| `marketplace_plugins.py` | `/api/marketplace/plugins` | 4 | GET |
| `community_skills.py` | `/api/skills/community` | 4 | GET, POST |

### Memory (21 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `memory.py` | `/api/v1/memory` | 18 | GET, POST, PUT, DELETE |
| `memory_stats.py` | `/api/v1/memory` | 3 | GET |

### System & Admin (57 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `system.py` | `/api/system` | 20 | GET, POST, PUT |
| `system_settings.py` | `/api/system-settings` | 10 | GET, POST, PUT, DELETE |
| `statistics.py` | `/api/system` | 5 | GET |
| `admin_prompts.py` | `/api/admin/prompts` | 11 | GET, POST, PUT, DELETE |
| `admin_plugins.py` | `/api/admin/plugins` | 9 | GET, POST, PUT |

### Auth & Security (29 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `credentials.py` | `/api/credentials` | 15 | GET, POST, PUT, DELETE |
| `permissions.py` | `/permissions` | 7 | GET, POST |
| `user_api_keys.py` | `/api/keys` | 7 | GET, POST, DELETE |

### Workspace & Team (15 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `workspaces.py` | `/api/workspaces` | 8 | GET, POST, PUT |
| `workspace_plugins.py` | `/api/workspaces` | 3 | GET, POST, DELETE |
| `team.py` | `/api/workspaces/{workspace_id}/team` | 4 | GET, POST, PUT, DELETE |

### Context Engineering & Intelligence (47 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `context.py` | `/api/context` | 15 | GET, POST |
| `context_engineering.py` | `/api/context-engineering` | 12 | POST |
| `context_policy.py` | `/api/policy` | 5 | GET, POST, PUT |
| `context_summarization.py` | `/api/context-summarization` | 2 | POST |
| `field_theory.py` | `/api/field-theory` | 13 | GET, POST, PUT |

### Multi-Agent & Orchestration (22 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `multi_agent.py` | `/api/multi-agent` | 10 | POST |
| `orchestrator.py` | `/api/orchestrator` | 5 | GET, POST |
| `routing.py` | `/api/routing` | 7 | GET, POST |

### Code & GitHub (13 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `codegraph.py` | `/api/code-graph` | 11 | GET, POST |
| `github_webhooks.py` | `/api/github` | 2 | POST |

### Skills & Templates (19 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `skills.py` | `/api/v1/skills` | 15 | GET, POST, PUT, DELETE |
| `templates.py` | `/api/templates` | 4 | GET, POST |

### Miscellaneous (34 endpoints)

| File | Prefix | Endpoints | Methods |
|------|--------|-----------|---------|
| `personas.py` | `/api` | 7 | GET, POST, PUT, DELETE |
| `benchmarking.py` | `/api/v1/benchmarking` | 4 | GET, POST |
| `execution_history.py` | `/api/execution-history` | 4 | GET |
| `playbooks.py` | `/api/playbooks` | 2 | GET |
| `patterns.py` | `/api/patterns` | 4 | GET, POST |
| `cache.py` | `/api/cache` | 3 | GET, DELETE |
| `bug_reports.py` | `/api/bug-reports` | 1 | POST |
| `generated_images.py` | `/api/generated-images` | 1 | GET |
| `webhooks.py` | `/api/webhooks` | 2 | POST |
| `models_endpoints.py` | `/api/models` | 7 | GET, PUT |
| `openrouter_marketplace.py` | `/api/openrouter` | 5 | GET, POST |
| `insights.py`, `learning.py`, `problems.py`, `query.py`, `recommendations.py`, `solutions.py`, `synthesis.py` | Various `/api/` | 13 combined | GET, POST |

---

## HTTP Method Distribution

| Method | Count | Percentage |
|--------|-------|------------|
| GET | ~330 | 58.5% |
| POST | ~165 | 29.3% |
| PUT | ~40 | 7.1% |
| DELETE | ~25 | 4.4% |
| PATCH | ~4 | 0.7% |

The system is read-heavy. Over half the endpoints are GET requests.

---

## Frontend Cross-Reference

**Frontend API client:** `frontend/lib/api-client.ts`

The frontend uses a centralized `ApiClient` class. Key observations:

1. **Actively called from frontend:**
   - `/api/agents/*`, `/api/workflows/*`, `/api/documents/*`
   - `/api/chat/*`, `/api/analytics/*`, `/api/credentials/*`
   - `/api/tools/*`, `/api/marketplace/*`, `/api/knowledge/*`
   - `/api/composio/*`, `/api/system/*`, `/api/workspaces/*`
   - `/api/memory/*`, `/api/code-graph/*`, `/api/cloud-documents/*`

2. **Known broken endpoints (marked in api-client.ts):**
   - `/api/multi-agent/behavior/learn` → 404 not implemented
   - `/api/multi-agent/optimization/adaptive` → 404 not implemented

3. **Backend-only endpoints (no frontend calls found):**
   - `/api/v1/benchmarking/*`
   - `/api/field-theory/*` (13 endpoints)
   - `/api/context-engineering/*` (12 endpoints)
   - `/api/orchestrator/*`
   - `/api/rag/feedback/*` (unmounted)
   - `/analytics/*` (non-/api prefix)
   - `/permissions/*` (non-/api prefix)

---

## Recommendations

### HIGH Priority — Fix Route Conflicts

1. **`/api/agents` triple-mount**: Consolidate `agents.py`, `agent_endpoints.py`, and `agent_plugins.py` or use distinct sub-prefixes (`/api/agents/plugins`, `/api/agents/v2`)
2. **`/api/analytics` double-mount**: `analytics_api.py` and `analytics_real.py` both mount at same prefix — merge or differentiate
3. **`/api/documents` double-mount**: `documents.py` and `document_generation.py` overlap

### MEDIUM Priority — Cleanup

4. **Remove unmounted files**: `database_knowledge_simple.py` (superseded), `rag_feedback.py` (orphaned)
5. **Fix prefix inconsistencies**: `analytics.py` → `/api/analytics`, `permissions.py` → `/api/permissions`, `workflow_history.py` → `/api/v1/workflows`
6. **Standardize versioning**: Either version all routes or none — currently only memory, skills, benchmarking, and workflow_history use `/v1/`

### LOW Priority — Future Consolidation

7. **Knowledge endpoints**: 3 files at `/api/knowledge` could merge
8. **Context endpoints**: 4 files for context engineering could consolidate
9. **Analytics**: 7 files could merge into 2-3 by sub-domain (platform, LLM, composio)

---

## Platform Action Candidates (for PRD-64)

High-value endpoints suitable for platform self-awareness actions:

| Domain | Candidate Actions | Source Files |
|--------|------------------|--------------|
| Agents | create, list, get, update, delete, assign tools | `agents.py`, `agent_endpoints.py` |
| Recipes | list, create, execute, get status | `workflow_recipes.py` |
| Analytics | LLM usage, costs, agent ranking | `llm_analytics.py`, `analytics_real.py` |
| Documents | list, upload, search, generate | `documents.py`, `document_generation.py` |
| Memory | recall, store, stats | `memory.py`, `memory_stats.py` |
| Tools | list connected, browse marketplace | `tools.py`, `composio.py` |
| Workspace | info, team, integrations | `workspaces.py`, `team.py` |
| Code | search, index, call graph | `codegraph.py` |
| System | health, config, settings | `system.py`, `system_settings.py` |

Estimated: ~50-60 curated actions from 564 total endpoints.
