# AUTOMATOS 0.2 — Target Data Model

**Purpose:** Declare the canonical table list (109+ → ≤75), the collapse map, and the drop queue.

---

## 1. Canonical tables per domain (target state)

### Goals (4 tables — was 7+)
| Table | Purpose | Collapses |
|---|---|---|
| `runs` | unified run object (kind=chat|mission|recipe|plan) | `orchestration_runs`, `workflow_executions`, `recipe_executions` |
| `run_tasks` | sub-tasks of a run | `orchestration_tasks`, parts of `tasks` (mission-owned), parts of `board_tasks` |
| `run_events` | event stream per run | `orchestration_events` |
| `recipes` | recurring schedules that instantiate runs | `workflow_recipes`, `workflow_templates` |
| (separate) `chats`, `messages`, `votes` | retained for chat ergonomics | no collapse; chat is a `run` of kind=chat, but message tape stays |

**Legacy retired:** `conversations` (= `chats`), `chat_messages` (= `messages`), `workflow_agents`.

### Agents (5 tables — was 7)
| Table | Purpose | Collapses |
|---|---|---|
| `agents` | roster | unchanged |
| `agent_blueprints` | templates | unchanged |
| `agent_tool_permissions` | tool grants per agent | `agent_tools`, `agent_skills`, `agent_tool_assignments_v2` |
| `agent_app_assignments` | Composio app assignments | `agent_app_features` merges here |
| `personas` | persona library | unchanged |

**Legacy retired:** `agent_executions` (→ `runs` + `run_tasks`), `agent_skills`, `agent_tools`, `agent_tool_assignments_v2`.

### Skills (5 tables)
| Table | Purpose |
|---|---|
| `skills`, `skill_files`, `skill_versions`, `skill_sources`, `skill_audit_log` | unchanged |

### Tools (8 tables — was 9)
| Table | Purpose |
|---|---|
| `tools`, `tool_categories`, `tool_configurations`, `tool_credentials`, `tool_reviews`, `tool_installation_requests` | unchanged |
| `workspace_tool_config` | per-workspace tool enablement | replaces `tenant_tool_config` |
| `composio_action_metadata` | composio action definitions |

### Knowledge (20 tables — was 24)
| Table | Purpose | Collapses |
|---|---|---|
| `documents`, `document_templates`, `rag_configurations` | documents subsystem |
| `knowledge_nodes`, `knowledge_edges` | graph |
| `external_knowledge` | external sources |
| `memory_items`, `memory_short_term`, `learning_outcomes` | memory subsystem |
| `database_knowledge_sources`, `database_relationships`, `database_query_templates`, `database_query_audit`, `semantic_metrics`, `semantic_dimensions` | NL2SQL |
| `nl2sql_training_examples`, `nl2sql_benchmark_runs`, `nl2sql_benchmark_results` | NL2SQL eval |
| `cloud_documents`, `cloud_sync_config`, `cloud_sync_jobs` | cloud sync |

**Retired:** `document_chunks` (moved to S3 Vectors per memory), `document_usage` (dead).

### Deliverables (2 tables — new in Wave 4)
| Table | Purpose | Collapses |
|---|---|---|
| `deliverables` | unified output | `artifacts`, `agent_reports.file_path`, mission output records |
| `deliverable_grades` | star ratings, review comments, tags | agent_reports grading columns |

**Retired:** `artifacts`, `agent_reports` (after migration window; the table survives for historical data, new writes go to `deliverables`).

### Workspaces (6 tables)
| Table | Purpose |
|---|---|
| `workspaces`, `workspace_members`, `workspace_invitations`, `workspace_models`, `workspace_enabled_plugins`, `workspace_enabled_skills` | unchanged |
| `business_profiles` | business context for plan-mode |

### Marketplace (10 tables)
| Table | Purpose |
|---|---|
| `marketplace_plugins`, `plugin_categories`, `plugin_security_scans`, `plugin_sync_history` | plugin catalog |
| `marketplace_widgets`, `widget_installations`, `widget_reviews` | widget catalog |
| `openrouter_models_cache`, `openrouter_sync_jobs` | model catalog cache |
| `composio_apps_cache`, `composio_actions_cache` | composio catalog cache (could move to Tools but they're catalog data) |

### Analytics (6 tables)
| Table | Purpose | Collapses |
|---|---|---|
| `llm_usage` | token + cost telemetry |
| `component_metrics` | component perf | replaces `system_metrics`, `performance_data` |
| `evaluation_results`, `benchmark_assessments`, `integration_analyses` | eval harness |
| `routing_decisions`, `routing_rules`, `unrouted_events` | routing telemetry (move from top-level to analytics) |
| `audit_logs` | system-wide audit | replaces `user_activities` |

### Admin (10 tables)
| Table | Purpose |
|---|---|
| `users`, `user_api_keys`, `sdk_api_keys` | identity |
| `credentials`, `credential_types`, `credential_audit_logs` | external creds |
| `system_settings`, `system_configurations`, `system_prompts`, `system_prompt_versions`, `system_prompt_eval_runs` | system admin |
| `permission_audit_logs` | permission changes |
| `context_policies` | context engineering policy |

### Composio (5 tables — kept separate as sub-domain of Tools)
| Table | Purpose |
|---|---|
| `composio_entities`, `composio_connections`, `composio_stats_cache`, `composio_sync_jobs`, `intent_classification_cache`, `tool_execution_logs`, `tool_execution_cache` | unchanged |

**Target total:** ~75 tables, down from 109 + ~16 orphan.

---

## 2. Drop queue (ordered by safety)

### Tier 1 — confirmed dead (drop after 7-day zero-traffic watch via pg_stat_user_tables)

```
User                            -- capitalized duplicate of users
blog_posts                      -- blog.py unmounted
chat_messages                   -- replaced by messages
conversations                   -- replaced by chats
tenant_tool_config              -- replaced by workspace_tool_config
agent_tool_assignments_v2       -- v2 migration artifact
document_chunks                 -- moved to S3 Vectors
document_usage                  -- dead telemetry
performance_data                -- replaced by component_metrics
system_metrics                  -- replaced by component_metrics
workflow_agents                 -- replaced by agents + workflow_executions
user_activities                 -- replaced by audit_logs (verify first)
system_alerts                   -- replaced by notifications (verify first)
context_usage                   -- likely replaced by llm_usage (verify first)
```

### Tier 2 — soft-retired after deliverables wave (drop in 0.3)

```
artifacts                       -- writes migrate to deliverables; read tier kept for 90 days
agent_reports                   -- same treatment; see PRD-133b
```

### Tier 3 — renames (no drop, just rename via migration)

```
orchestration_runs  -> runs         (canonical name per north-star)
orchestration_tasks -> run_tasks
orchestration_events -> run_events
orchestration_archive -> (drop after 90 days of zero reads)
```

### Tier 4 — consolidation (collapse into parent)

```
agent_tools, agent_skills       -> agent_tool_permissions (single grants table)
agent_app_features              -> agent_app_assignments (single composio assignment table)
workflow_templates, workflow_recipes -> recipes
```

### Tier 5 — backup remnants (confirmed dead per memory)

```
b_*_<date>  (11 backup tables from Phase B rename pass — watch-period DROP)
```

---

## 3. Column-level cleanup candidates

Deferred to PRD-135 Phase 2-3 (AST walker + dead-column report). Do not guess columns in this doc. Manual scrub pass after Wave 1 drops tables — column cleanup of surviving tables happens with static AST scan results.

Initial suspects to watch for once the walker runs:
- `agents.model_config` — dropped per memory; verify via alembic
- `users` — pre-hybrid-auth columns may be unused now
- `workspaces` — legacy config columns

---

## 4. Migration safety rules

1. **No DROP without watch period.** Every candidate needs 7 days of `seq_scan = 0 AND n_tup_* = 0` in `pg_stat_user_tables` before DROP ships to prod.
2. **Every DROP is reversible by reloading from backup.** Railway Postgres has PITR; pre-DROP snapshot tagged `prd-0.2-drop-<n>` before each migration.
3. **Every rename gets a view alias.** E.g. `CREATE VIEW orchestration_runs AS SELECT * FROM runs` for one release cycle so any missed caller doesn't 500.
4. **Every collapse has a 90-day soft-migration window.** Writes go to canonical; reads union both until telemetry proves zero-traffic on the old.
5. **Every migration ships as its own alembic version.** No bundled "cleanup" migration — one concept per migration, one revert.
6. **Foreign keys re-pointed before rename.** E.g. any FK on `orchestration_runs.id` re-pointed to `runs.id` in the same migration with `ALTER TABLE ... DROP CONSTRAINT ... ADD CONSTRAINT ...`.

---

## 5. Alembic migration plan (Wave 1)

Proposed migration order. Each is its own version file under `orchestrator/alembic/versions/`:

```
prd-0.2-001-drop-orphan-tables-tier1.py
   -- drops: User, blog_posts, chat_messages, conversations, tenant_tool_config,
              agent_tool_assignments_v2, document_chunks, document_usage,
              performance_data, system_metrics, workflow_agents

prd-0.2-002-drop-orphan-tables-tier1-guarded.py
   -- drops after verification: user_activities, system_alerts, context_usage

prd-0.2-003-collapse-agent-permissions.py
   -- migrates agent_tools + agent_skills + agent_tool_assignments_v2 residue
      into agent_tool_permissions; then drops sources

prd-0.2-004-rename-orchestration-to-runs.py
   -- renames orchestration_runs→runs, orchestration_tasks→run_tasks,
      orchestration_events→run_events (alias views created)

prd-0.2-005-collapse-recipes.py
   -- merges workflow_templates, workflow_recipes into recipes

prd-0.2-006-drop-backup-tables.py
   -- drops the 11 b_*_<date> tables

prd-0.2-007-create-deliverables.py
   -- Wave 4; creates deliverables + deliverable_grades

prd-0.2-008-migrate-artifacts-to-deliverables.py
   -- backfills; dual-writes begin

prd-0.2-009-drop-alias-views.py
   -- after one release cycle, drops view aliases from 004
```

---

## 6. Data contracts (what frontend/SDK consumers will see)

| Before | After | Change type |
|---|---|---|
| `GET /missions` returns `orchestration_runs` shape | `GET /goals/missions` returns `run` shape (kind=mission) | additive fields; legacy paths 308 |
| `GET /reports` returns agent_reports with file_path | `GET /deliverables` returns unified deliverable | dual surface for one release; legacy 308 |
| `GET /workflows` returns workflow | `GET /goals/recipes` returns recipe | 308 redirect |

No field deletions in 0.2. Deletions in 0.3 after telemetry.

---

## 7. Cross-checks before shipping Wave 1

Ship blockers — do not drop ANY table until all green:

- [ ] PRD-135 Phase 1 (db-scan) produces full table inventory from live Railway Postgres.
- [ ] PRD-135 Phase 2 (code→DB walker) emits `reads/writes/models` edges for ≥80% of call sites.
- [ ] PRD-135 Phase 4 (runtime overlay) produces 7-day zero-traffic signal for each Tier 1 candidate.
- [ ] Alembic migrations reviewed; reversible.
- [ ] Pre-migration backup tag exists in Railway.
- [ ] All FKs accounted for.
- [ ] Frontend call-site grep for each dropped name returns 0 hits.

---

## 8. Estimated scope

- **Migrations:** 9 alembic files, each <200 lines.
- **Tables dropped:** ~20 direct + 11 backup = **~31 tables removed**.
- **Tables renamed:** 3.
- **Tables collapsed:** 6 → 2.
- **Target after 0.2:** ≤75 active tables (from 109 + orphan).
- **New tables added:** 2 (`deliverables`, `deliverable_grades`) in Wave 4.

Net: **-29 tables**, more predictable schema, alembic history unambiguous.

---

**Cross-references:**
- [01-CURRENT-STATE.md §4](./01-CURRENT-STATE.md) — orphan table list
- [01a-EXPLORE-REPORT.md §6](./01a-EXPLORE-REPORT.md) — dead-table suspects
- [08-MIGRATION-PHASES.md](./08-MIGRATION-PHASES.md) — executable phasing
- PRD-135 — the scanner + report tooling this plan depends on
