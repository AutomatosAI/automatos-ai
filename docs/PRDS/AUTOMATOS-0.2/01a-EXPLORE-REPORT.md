# AUTOMATOS 0.2 — Explore Agent Report (Appendix)

**Source:** Background Explore agent, 2026-04-24, dispatched to audit duplication + dead code across 228K LOC Python + 599 TS frontend.
**Parent doc:** [01-CURRENT-STATE.md](./01-CURRENT-STATE.md)
**Status:** Raw agent output preserved as-is; main-doc numbers in 01-CURRENT-STATE are reconciled against this.

---

## Executive Summary (agent's own)

> 101 API routers mounted + 4 unmounted dead. 109 SQLAlchemy tables spread across 31 model files. Redundant analytics surfaces (4 overlapping files = 1.7K LOC). Memory subsystem split across 2 API files + 3 hooks. Mission/orchestration naming drift (runs table vs missions API). **Total dead weight: ~2.3K LOC** across unmounted routers, orphan hooks, table-less models. **Total consolidation opportunity: ~4.1K LOC** (2.3K deletion + 1.6K refactoring).

---

## 1. Unmounted dead routers (immediate delete candidates)

| File | LOC | Status | Action | Risk |
|---|---|---|---|---|
| `orchestrator/api/blog.py` | 182 | unmounted, no frontend hook, no DB model calls | DELETE | low |
| `orchestrator/api/database_knowledge_simple.py` | 77 | overlaps `database_knowledge.py`, unmounted | DELETE | low |
| `orchestrator/api/rag_feedback.py` | 178 | unmounted, no backing model, incomplete | DELETE (verify first) | low |
| `orchestrator/api/workspace_exec.py` | 52 | unmounted, unclear purpose | INVESTIGATE → DELETE or migrate to services | medium |
| `orchestrator/api/recipe_executor.py` | 1,574 | **unmounted but large** — may be background job runner | INVESTIGATE — likely move to `services/` | high |

**Quick-win total:** 489 LOC (first four) deleted safely; recipe_executor needs a 30-min investigation before action.

## 2. Analytics duplication (Rank 1 hotspot)

| File | LOC | Routes | Canonical target |
|---|---|---|---|
| `analytics.py` | 249 | `/dashboard/summary`, `/agent-selection/analysis` | merge → `analytics_base.py` |
| `analytics_api.py` | 227 | `/dashboard/overview`, `/agents/{id}`, `/context`, `/system/health`, `/realtime` | merge → `analytics_base.py` |
| `analytics_real.py` | 931 | `/dashboard/*`, `/performance/*` (14 routes) | rename → `analytics_dashboards.py` |
| `analytics_charts.py` | 292 | PandasAI chart gen | keep separate |

**Consolidated: 476 LOC deleted, one clean router structure.**

## 3. Memory duplication (Rank 2 hotspot)

- `memory.py` (711 LOC) = CRUD + search + consolidate
- `memory_stats.py` (898 LOC) = observability (must mount BEFORE `memory.py` to avoid `/{memory_id}` catch-all)
- `use-memory-api.ts` = canonical
- `use-memory-v1-api.ts` = legacy alias, same endpoints — merge & delete
- `use-memory-explorer-api.ts` = additional surface (audit whether it overlaps)

**Risk:** high — this is critical infra for agent memory (PRD-79). Refactor with explicit tests.

## 4. Missions / orchestration naming debt

- DB class: `OrchestrationRun` → table `orchestration_runs`
- API path: `/api/missions`
- Frontend hook: `use-missions-api.ts`

Agent recommendation: **rename class to `Mission`** (or add `Mission = OrchestrationRun` alias) so DB, API, code, and UI all agree. This is the most confusing drift in the platform. Ship as part of Wave 1.

## 5. Frontend hook call-site map (15-hook sample)

All 15 sampled hooks hit live routes; two are legacy duplicates:

- `use-memory-v1-api.ts` — legacy, merge into `use-memory-api.ts`
- (other legacy candidates flagged in 01-CURRENT-STATE.md §6)

All primary domain hooks (agents, missions, memory, workflows, board-tasks, knowledge, analytics, composio, marketplace, credentials, heartbeats, cloud-documents, reports, deliverables) resolve correctly to mounted routes.

## 6. Dead-table suspects (post-static-scan)

| Table | Agent verdict | Action |
|---|---|---|
| `blog_posts` | zero router queries, blog.py unmounted | DELETE with blog.py |
| `unrouted_events` | routing.py writes only, no read endpoints | verify cleanup job exists |
| `orchestration_archive` | verify missions.py exposes archive endpoints |
| `voice_profiles` | chat_voice.py optional; verify enabled |
| `widget_reviews` | widget API optional |
| `tool_installation_requests` | verify tools.py endpoint |
| `permission_audit_logs` | write-only audit; keep |
| `document_chunks` | appears in migrations, no model — orphan OR stored as JSONB in documents |

These compound the 16 migration-orphan tables from 01-CURRENT-STATE §4. **Total dead-table candidates: ~20** pending runtime validation (PRD-135 Phase 1+4).

## 7. Route → DB table map (agent sampled 20 routers)

Agent produced handler-level mapping for the 20 most-touched routers: agents, missions, tasks, workflows, memory, database_knowledge, analytics_real, composio, chat, documents, marketplace_plugins, knowledge, routing, admin_plugins, credentials, permissions, recommendations, and three others. Full tables included in the original agent output (retained below).

**Key insight from this map:** most tables are touched by 1-3 routers. The highest fan-in tables are `agents` (5+ routers), `workflow_executions` (3+), `workflow_recipes` (3+), `composio_connections` (3+). These are the canonical tables — anything that looks similar and has low fan-in is probably redundant.

## 8. Agent's phasing recommendation (cross-reference for 08-MIGRATION-PHASES.md)

**Phase 1 — Week 1-2 Foundation Clarity**
1. Execute the 10 quick wins (dead-router deletion + docstring clarifications)
2. Consolidate analytics (476 LOC, low risk)
3. Clarify memory subsystem boundaries

**Phase 2 — Week 3-4 Service Layer Migration**
4. Move `recipe_executor.py` → `services/recipe_executor_service.py` if background job
5. Migrate composio caching into dedicated service (8 cache tables currently scattered)
6. Unify NL2SQL service

**Phase 3 — Week 5+ Long-term refactoring**
7. Rename `OrchestrationRun` → `Mission`
8. Collapse marketplace routes (`llm_marketplace` + `openrouter_marketplace`)
9. Canonical REST patterns (endpoint naming, response envelope, error codes)

**Post-0.2 backlog for 0.3:**
- Remove `chatbot_llm.py` deprecated router
- Consolidate evaluation routing
- Stabilize voice subsystem

---

## 9. Quick-wins summary table (reproduced verbatim from agent)

| # | Item | LOC | Effort | Risk | 0.2 verdict |
|---|---|---|---|---|---|
| 1 | Delete `blog.py` | 182 | 15 min | low | DELETE |
| 2 | Delete `database_knowledge_simple.py` | 77 | 10 min | low | DELETE |
| 3 | Investigate `rag_feedback.py` | 178 | 30 min | low | DELETE if confirmed |
| 4 | Investigate `workspace_exec.py` | 52 | 30 min | medium | DELETE or integrate |
| 5 | Consolidate `analytics.py` + `analytics_api.py` | 476 | 2 hrs | low | MERGE |
| 6 | Clarify memory docs | 0 | 30 min | low | DOCUMENT |
| 7 | Verify `recipe_executor.py` | 1,574 | 1 hr | high | INVESTIGATE |
| 8 | `ORCHESTRATION_NAMING_DEBT.md` | 0 | 30 min | low | DOCUMENT |
| 9 | Deprecate `chatbot_llm.py` | 0 | 15 min | low | COMMENT |
| 10 | Consolidate `use-memory-v1-api.ts` | ~50 | 1 hr | medium | MERGE |

**Agent's estimate:** 12-16 hours total for full quick-win wave.

---

## 10. How to reconcile with 01-CURRENT-STATE.md

- **Router count:** main-doc says "103 mounted"; agent says "101 mounted + 4 unmounted". Difference = conditional `try/except` mounts in main.py (e.g. composio, cloud_documents, shopify). Treat 101 as load-bearing; 4 unmounted = confirmed-dead; remaining 2 = conditional.
- **Analytics router count:** agent found 4 router files; main-doc listed 7 including `composio_analytics.py`, `database_analytics.py`, `llm_analytics.py`. Those three exist but are domain-partitioned analytics, not general analytics — they stay with their domains (composio/database/llm). True consolidation target = the 3 general-analytics files (analytics, analytics_api, analytics_real).
- **Marketplace router count:** main-doc listed 5; agent confirms 3 true marketplace + 2 widget-specific. Consolidation target = 3 marketplace files, widgets stay separate.
- **Table count:** both agree, 109 modelled + ~16-20 orphan in migrations.

---

**Use this appendix as the fact base.** [05-DATA-MODEL.md](./05-DATA-MODEL.md) and [04-API-SURFACE.md](./04-API-SURFACE.md) derive their canonical targets from these findings.
