# AUTOMATOS 0.2 — Migration Phases (Executable Plan)

**Purpose:** Each phase below is a standalone PRD that can be shipped on its own. Phases are ordered by blockers; several can parallelize once Wave 0 is green.

Treat this document as the table of contents for the PR stream.

---

## Wave 0 — Instrumentation (blocker for every subsequent wave)

Ship PRD-135 Phases 1-3 first so the entire rest of 0.2 is data-driven.

### Phase 0.1 — PRD-135 Phase 1: DB snapshot scanner
**Scope:** `/graphify db-scan` emits `graphify-out/db.json` from live Railway Postgres.
**Success:** runs in <60s against production DB; produces node/edge JSON matching graphify schema.
**PR size:** ~300 LOC (one Python script + CLI wiring).
**Who blocks:** read-only Postgres user, `DATABASE_URL` in CI.

### Phase 0.2 — PRD-135 Phase 2: Code→DB edge walker
**Scope:** Walk `orchestrator/**/*.py` for `__tablename__` + `text("...")` + `db.query(Model)` patterns.
**Success:** ≥80% of call sites produce module→table edges with confidence tier.
**PR size:** ~500 LOC (AST walker + edge emitter).

### Phase 0.3 — PRD-135 Phase 3: Three reports
**Scope:** `dead-tables`, `dead-routes`, `consolidation-candidates` markdown reports generated from the merged graph.
**Success:** `consolidation-candidates` returns `deliverables ↔ agent_reports` pair (retroactive validation against pre-PRD-133b main).
**PR size:** ~200 LOC (report queries + markdown formatter).

### Phase 0.4 — PRD-135 Phase 4: Runtime overlay (optional, recommended)
**Scope:** Nightly pull of `pg_stat_statements` + `pg_stat_user_tables` merged as edge weights.
**Success:** zero-traffic signal per-table; 14-day trend line for use in Wave 1 DROP gating.
**PR size:** ~150 LOC.

**Exit criteria for Wave 0:** the rest of 0.2 becomes a series of reports + PRs that close rows in those reports.

---

## Wave 1 — Data-model collapse

Each phase is one alembic migration + zero or minimal app code.

### Phase 1.1 — Drop orphan Tier 1 tables (safe 11)
**Tables:** `User`, `blog_posts`, `chat_messages`, `conversations`, `tenant_tool_config`, `agent_tool_assignments_v2`, `document_chunks`, `document_usage`, `performance_data`, `system_metrics`, `workflow_agents`.
**Preconditions:** Phase 0.4 shows 7-day zero-traffic on all 11.
**Migration:** `prd-0.2-001-drop-orphan-tables-tier1.py` (simple DROP TABLE IF EXISTS with IF EXISTS guards).
**PR size:** ~100 LOC migration.
**Risk:** low.

### Phase 1.2 — Drop orphan Tier 1 guarded (verify 3)
**Tables:** `user_activities`, `system_alerts`, `context_usage`.
**Preconditions:** confirm replacement (audit_logs / notifications / llm_usage) fully adopted.
**Migration:** `prd-0.2-002-drop-orphan-tables-tier1-guarded.py`.
**PR size:** ~60 LOC + 3 code audits.
**Risk:** medium — require explicit sign-off.

### Phase 1.3 — Collapse agent permissions
**Scope:** migrate any residual rows from `agent_tools`, `agent_skills`, `agent_tool_assignments_v2` into `agent_tool_permissions`; then drop sources.
**Migration:** `prd-0.2-003-collapse-agent-permissions.py` with data migration block.
**PR size:** ~150 LOC migration + ORM model deletion.
**Risk:** medium.

### Phase 1.4 — Rename orchestration → runs (naming debt)
**Scope:** rename `orchestration_runs` → `runs`, `orchestration_tasks` → `run_tasks`, `orchestration_events` → `run_events`.
- Create view alias (`CREATE VIEW orchestration_runs AS SELECT * FROM runs`) for one release.
- Rename ORM class `OrchestrationRun` → `Mission` OR keep ORM as `Run` with domain aliases.
- Update all call sites.
**Migration:** `prd-0.2-004-rename-orchestration-to-runs.py`.
**PR size:** ~400 LOC including call-site updates.
**Risk:** medium — many call sites.
**Tactic:** ship in two PRs — migration first (with alias views), call-site update second.

### Phase 1.5 — Collapse recipes
**Scope:** merge `workflow_templates`, `workflow_recipes` → `recipes`.
**Migration:** `prd-0.2-005-collapse-recipes.py`.
**PR size:** ~200 LOC.
**Risk:** low — PRD-US-009 already renamed templates→recipes; this is finishing that.

### Phase 1.6 — Drop b_*_<date> backup tables (11)
**Tables:** the 11 backup tables from memory.
**Preconditions:** 14-day watch from Phase 0.4.
**Migration:** `prd-0.2-006-drop-backup-tables.py`.
**PR size:** ~80 LOC.
**Risk:** low.

**Wave 1 exit:** `/graphify db-report dead-tables` returns ≤5 rows (tolerance for column-level cleanup that's out of scope for this wave).

---

## Wave 2 — API-surface collapse (103 → ~25 routers)

Each phase tackles one domain. Order chosen to minimize cross-phase conflicts.

### Phase 2.1 — Delete confirmed-dead routers (quick win)
**Scope:** delete `blog.py`, `database_knowledge_simple.py`, `rag_feedback.py`, `workspace_exec.py`, `anthropic_client.py` (router stub); move `recipe_executor.py` → `services/`.
**PR size:** ~1.5K LOC deleted.
**Risk:** low (all unmounted or confirmed unused).

### Phase 2.2 — Consolidate analytics
**Scope:** merge `analytics.py` + `analytics_api.py` → `analytics.py` (canonical); rename `analytics_real.py` sub-paths to `/analytics/dashboard/*` and `/analytics/performance/*`; merge `analytics_charts.py` sub-path.
Absorb: `composio_analytics.py`, `database_analytics.py`, `llm_analytics.py`, `kpi_api.py`, `insights.py`, `learning.py`, `recommendations.py`, `patterns.py`, `problems.py`, `solutions.py`, `synthesis.py`, `statistics.py`, `dashboard_integration.py`, parts of `routing.py`.
**PR size:** 1-2K LOC reorganized, ~500 LOC deleted.
**Risk:** medium — many call sites. Ship with 308 redirects.

### Phase 2.3 — Consolidate marketplace
**Scope:** merge `marketplace.py` + `marketplace_plugins.py` + `llm_marketplace.py` + `openrouter_marketplace.py` + `widget_marketplace.py` + `templates.py` → `marketplace.py` (sub-paths per kind).
**PR size:** ~800 LOC reorganized.
**Risk:** low-medium.

### Phase 2.4 — Consolidate knowledge
**Scope:** `knowledge.py` absorbs `knowledge_graph.py`, `knowledge_multimodal.py`, `database_knowledge.py`, `cloud_documents.py`, `codegraph.py`, `documents.py`, `document_generation.py`, `memory.py`, `memory_stats.py`, `widget_memory.py` (sub-paths), `context_engineering.py`, `context_summarization.py`, `context_policy.py`, `context.py`, `query.py`.
**PR size:** 2-3K LOC reorganized.
**Risk:** medium-high — memory subsystem is critical infra. Mount-order tricks must be preserved during transition (see memory mount-order comment in main.py).

### Phase 2.5 — Consolidate goals (chat + missions + recipes + plans)
**Scope:** `goals.py` absorbs `chat.py`, `chatbot_llm.py` (→ deprecated/), `chat_voice.py`, `missions.py`, `workflows.py` (→ deprecated/), `workflow_recipes.py`, `workflow_templates.py` (→ deprecated/), `workflow_history.py`, `recipe_executor.py` (API parts), `api_playbooks.py`, `attachments.py`, `scheduled_tasks.py`, `execution_history.py`, `wizard.py`.
**PR size:** 2-3K LOC.
**Risk:** high — chat + missions are most-used surfaces. Feature-flag + 308 + 14-day telemetry mandatory.

### Phase 2.6 — Consolidate agents
**Scope:** `agents.py` absorbs `agent_endpoints.py`, `agent_plugins.py`, `onboarding_agents.py`, `heartbeat.py`, `board_tasks.py`, `tasks.py`, `personas.py`, `voice_profiles.py`, `tool_assignments.py`.
**PR size:** 1-2K LOC.
**Risk:** medium.

### Phase 2.7 — Consolidate admin
**Scope:** `admin.py` absorbs `admin_plugins.py`, `admin_prompts.py`, `admin_workspaces.py`, `system.py` (admin parts), `system_settings.py`, `credentials.py`, `api_keys.py`, `user_api_keys.py`, `permissions.py`, `models_endpoints.py`, `routing.py` (admin parts).
**PR size:** 1K LOC.
**Risk:** low.

### Phase 2.8 — Consolidate workspaces
**Scope:** `workspaces.py` absorbs `workspace_files.py`, `workspace_github.py`, `workspace_plugins.py`, `workspace_skills.py`, `team.py`.
**PR size:** 500 LOC.
**Risk:** low.

### Phase 2.9 — Consolidate tools
**Scope:** `tools.py` absorbs `composio.py`.
**PR size:** 500 LOC.
**Risk:** low.

### Phase 2.10 — Reorganize widgets + webhooks
**Scope:** move `widget_*.py` → `widgets/` subpackage; `github_webhooks.py`, `shopify.py`, `webhooks.py` → `webhooks/` subpackage.
**PR size:** ~300 LOC (mostly moves).
**Risk:** low.

### Phase 2.11 — Retire deprecated routers
**Scope:** after 14 days of 308-redirect telemetry showing zero traffic, delete the `deprecated/` subpackage.
**PR size:** ~500 LOC deleted.
**Risk:** low.

**Wave 2 exit:** `ls orchestrator/api/*.py | wc -l` ≤ 25; `grep -c include_router orchestrator/main.py` ≤ 30; dead-routes report = 0.

---

## Wave 3 — Frontend surface collapse

### Phase 3.1 — Consolidate duplicate hooks (first sweep)
**Scope:** merge `use-memory-v1-api.ts` → `use-memory-api.ts`; merge `use-multi-agent-verified.ts` → `use-multi-agent.ts`; merge `use-board-tasks.ts` + `use-board-tasks-api.ts`; merge `use-memory-explorer-api.ts` into memory hook if redundant.
**PR size:** ~300 LOC deleted.
**Risk:** low.

### Phase 3.2 — Create canonical domain hooks
**Scope:** ship `useGoalsApi`, `useAgentsApi`, `useSkillsApi`, `useToolsApi`, `useKnowledgeApi`, `useDeliverablesApi`, `useWorkspacesApi`, `useMarketplaceApi`, `useAnalyticsApi`, `useAdminApi`.
Each maps to the canonical backend router from Wave 2.
**PR size:** ~2K LOC (one file per hook).
**Risk:** low — additive; no deletions yet.

### Phase 3.3 — Migrate call sites (codemod)
**Scope:** run jscodeshift codemod to replace old hook imports with canonical domain hooks across `frontend/app/` + `frontend/components/`.
**PR size:** ~5K LOC touched, 0 net LOC change per-file (just imports).
**Risk:** medium — must test each page.

### Phase 3.4 — Delete old hooks
**Scope:** after call sites migrated, delete legacy hook files.
**PR size:** ~2K LOC deleted.
**Risk:** low.

### Phase 3.5 — App route migration (four-tab shell)
**Scope:** move `/chat`, `/missions`, `/playbooks`, `/activity`, `/tools`, `/field-theory`, `/context`, `/team`, `/workspace`, `/dashboard` into the four-tab structure per [06-FRONTEND-SURFACE.md §4](./06-FRONTEND-SURFACE.md).
**PR size:** ~1K LOC moved.
**Risk:** medium.

### Phase 3.6 — Component consolidation sweep
**Scope:** dedup component clusters per [06-FRONTEND-SURFACE.md §3](./06-FRONTEND-SURFACE.md). Use `knip` / `ts-prune` for unused-export detection.
**PR size:** ~1.5K LOC deleted, ~500 LOC changes.
**Risk:** low-medium.

### Phase 3.7 — Redirect cleanup
**Scope:** after 14 days of redirect telemetry, remove `next.config.js` 301s.
**PR size:** ~100 LOC.
**Risk:** low.

**Wave 3 exit:** `ls frontend/hooks/use-*-api*.ts | wc -l` ≤ 15; no `*-v1-api.ts` etc.; all routes reachable from four tabs in ≤2 clicks.

---

## Wave 4 — Deliverables unification

### Phase 4.1 — Create `deliverables` + `deliverable_grades` tables
**Scope:** Wave 1 migration already listed as `prd-0.2-007-create-deliverables.py`; ship it as part of Wave 4 if Wave 1 didn't land it.
**PR size:** ~150 LOC.
**Risk:** low.

### Phase 4.2 — DeliverableService + dual-write
**Scope:** every place that creates an `artifact`, an `agent_report.file_path`, a mission task output, or a generated image now also writes to `deliverables`.
**PR size:** ~500 LOC changes across ~15 call sites.
**Risk:** medium — must preserve all legacy call-site semantics.

### Phase 4.3 — Deliverables tab UI
**Scope:** `/deliverables` page, grid + filters + `<DeliverableView />` preview + grader.
**PR size:** ~1K LOC.
**Risk:** low — unified preview already exists per memory.

### Phase 4.4 — Read migration
**Scope:** all UI places that used to list `agent_reports` or `artifacts` now list `deliverables`.
**PR size:** ~400 LOC.
**Risk:** medium — grading history must be preserved.

### Phase 4.5 — Skill-promotion flow
**Scope:** "Promote this deliverable as a skill training example" button → writes a `skill_source`.
**PR size:** ~300 LOC.
**Risk:** low — additive feature.

### Phase 4.6 — Retire legacy write paths
**Scope:** after 90 days of dual-write telemetry, stop writing to `artifacts` + `agent_reports.file_path`. Reads from old tables continue for audit.
**PR size:** ~200 LOC.
**Risk:** medium.

**Wave 4 exit:** every new agent output lands in `deliverables`; Deliverables tab is the single answer.

---

## Wave 5 — Autonomous flow composition

### Phase 5.1 — Unified `run` object (the heart)
**Scope:** migrate `orchestration_runs` (now `runs` post-Wave 1.4) into the canonical `run` shape from [07-DELIVERABLES-FLOW §1](./07-DELIVERABLES-FLOW.md). Add `kind` column. Backfill chat rows into runs.
**PR size:** ~600 LOC migration + backfill.
**Risk:** high — load-bearing schema change.

### Phase 5.2 — Unified compose box
**Scope:** Goals tab compose box with mode picker (chat | mission | recipe | plan). Coordinator auto-picks mode if user doesn't.
**PR size:** ~500 LOC.
**Risk:** medium — UX-heavy.

### Phase 5.3 — Run event stream UI
**Scope:** `/ws/runs/{id}` endpoint + live event list on run detail page.
**PR size:** ~600 LOC.
**Risk:** low — replaces multiple fragmented feeds.

### Phase 5.4 — Human-gate UI
**Scope:** gate events surface as action cards in the Goals tab; approve/reject/modify → resume event.
**PR size:** ~400 LOC.
**Risk:** medium — covers PRD-82A gate + future gates.

### Phase 5.5 — Mission Zero upgrade to Plan-mode run
**Scope:** wizard becomes a kind=`plan` run; VOYAGER/BLUEPRINT/FORGE emit deliverables per configured resource.
**PR size:** ~500 LOC.
**Risk:** medium.

**Wave 5 exit:** the four journey tests in [07-DELIVERABLES-FLOW §10](./07-DELIVERABLES-FLOW.md) pass.

---

## Wave 6 — Skills & marketplace as the extension layer

### Phase 6.1 — Marketplace tab unification
**Scope:** `/marketplace/{skills,models,plugins,widgets,templates}` as one tab with five kinds; per [04-API-SURFACE §3](./04-API-SURFACE.md).
**PR size:** ~400 LOC.
**Risk:** low.

### Phase 6.2 — Workspace template installer
**Scope:** "install template" creates a `kind=plan` run that configures the workspace (ties to PRD-120).
**PR size:** ~600 LOC.
**Risk:** medium.

### Phase 6.3 — Skill promotion from deliverable (wired to Wave 4.5)
**Scope:** already specced; wire UI to the flow.
**PR size:** already in 4.5; this is the activation PR.

**Wave 6 exit:** a new vertical (Shopify, HubSpot, finance team) installs as one template with zero orchestrator code changes.

---

## Wave 7 — Instrumentation as CI

### Phase 7.1 — Nightly `/graphify db-scan` on CI
**Scope:** GitHub Action runs Phase 0.1 + 0.4 nightly against production Postgres; commits reports to `graphify-out/`.
**PR size:** ~100 LOC.
**Risk:** low.

### Phase 7.2 — PR gate on new router/model without retiring equivalent
**Scope:** CI check: if a PR adds a new router file in `orchestrator/api/` without the PRD description referencing a canonical consolidation, fail.
**PR size:** ~100 LOC.
**Risk:** low — purely a linter.

### Phase 7.3 — Weekly 0.2-close report
**Scope:** scheduled report aggregating dead-tables, dead-routes, consolidation-candidates, trend line of LOC deleted per week.
**PR size:** ~200 LOC.
**Risk:** low.

---

## Dependency graph

```
Wave 0 (instrumentation) ─────────────┬───────────────────────────┐
                                      │                           │
                                      ▼                           ▼
Wave 1.1-1.2 (orphan drops) ─────► Wave 1.3-1.6 (collapses)      Wave 7 (CI, continuous)
                                      │
                                      ▼
        ┌────────────────────────────Wave 2 (API) ─────────────────┐
        │                                                          │
        ▼                                                          ▼
Wave 3 (frontend) ──────────────► Wave 4 (deliverables) ────────► Wave 5 (autonomous) ──► Wave 6 (marketplace)
```

- **Wave 0 blocks everything.** Ship first.
- **Wave 1 and Wave 7 can start in parallel** with Wave 0 landing.
- **Wave 2 can start** as soon as Wave 1.4 (rename) is green — other Wave 1 phases don't block.
- **Wave 3 waits on Wave 2.5-2.6** (goals + agents) to avoid constant re-migration.
- **Wave 4 can start** in parallel with late Wave 2 (table created; dual-write hooks go in).
- **Wave 5 waits on Wave 4** (deliverables exist) and Wave 1.4 (runs renamed).
- **Wave 6 waits on Wave 5** (plan-mode runs) and Wave 4 (deliverables).

---

## Estimated effort (rough, no calendar commitments)

| Wave | Phases | PR count | LOC touched | LOC net deleted |
|---|---|---|---|---|
| 0 | 4 | 4 | ~1.2K | 0 |
| 1 | 6 | 6 | ~1K | -29 tables, -small code |
| 2 | 11 | ~15 | ~12K touched | ~6K deleted (routers merged) |
| 3 | 7 | ~10 | ~10K touched | ~4K deleted |
| 4 | 6 | ~6 | ~3K touched | 0 net (additive) |
| 5 | 5 | ~5 | ~2.5K touched | small |
| 6 | 3 | ~3 | ~1K touched | 0 |
| 7 | 3 | ~3 | ~400 | 0 |
| **Total** | **45** | **~52 PRs** | **~31K touched** | **~10K deleted** |

Comparable scale to PRD-131 (which deleted 22.6K LOC). Different in character: PRD-131 was "delete things we replaced"; 0.2 is "reorganize what's left."

---

## Risk controls (shared across all waves)

1. **Feature-flag everything possible.** Gate new routers behind `ENABLE_PRD_02_NEW_API=true` initially.
2. **Backward-compatible shims.** Legacy paths 308 or view-alias for one release cycle.
3. **14-day telemetry per cut-over.** Only delete legacy after zero-traffic for 14 days.
4. **One concept per PR.** No "refactor + feature" PRs. Revert-safe.
5. **Pre-migration DB snapshot.** Tag Railway backup before each Wave 1 migration.
6. **Pixel-diff for frontend.** Visual regression threshold 2%.
7. **Escalate on surprise.** If a consolidation reveals unknown callers (e.g. external SDK user hitting a "dead" route), halt that phase.

---

## How to pick up a phase

1. Open the relevant doc section above.
2. Find the phase number matching current wave state.
3. Read the cross-linked design doc (02-07) for context.
4. Open a PR branch `fix/prd-0.2-<phase>-<slug>`.
5. Ship as its own mini-PRD; link back to this phase ID in the description.
6. On merge, cross out the phase in this doc or open a follow-up if scope split.

---

**Cross-references:**
- [00-README.md](./00-README.md) — phase-wave summary
- [01-CURRENT-STATE.md](./01-CURRENT-STATE.md) — inputs to phases
- [03-DOMAIN-MODEL.md](./03-DOMAIN-MODEL.md) — target layout
- [04-API-SURFACE.md](./04-API-SURFACE.md) — Wave 2 target
- [05-DATA-MODEL.md](./05-DATA-MODEL.md) — Wave 1 + 4 target
- [06-FRONTEND-SURFACE.md](./06-FRONTEND-SURFACE.md) — Wave 3 target
- [07-DELIVERABLES-FLOW.md](./07-DELIVERABLES-FLOW.md) — Waves 4-5 target
- [09-SUCCESS-METRICS.md](./09-SUCCESS-METRICS.md) — exit criteria per wave
- PRD-135 — Wave 0 foundation
