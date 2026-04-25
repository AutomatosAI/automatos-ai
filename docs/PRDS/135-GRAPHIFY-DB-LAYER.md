# PRD-135 — Graphify DB Layer (Dead-Code & Consolidation Intelligence)

**Version:** 0.1 (draft, background)
**Type:** Tooling / Internal Infrastructure
**Status:** Draft — intended to be designed/refined in parallel with PRD-133b finish-out
**Priority:** P2 (pays off on every future cleanup PRD, but not blocking pilot)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-04-24
**Related:** PRD-131 (platform consolidation), PRD-133b (outputs view), PRD-134 (placeholder — post-133 DB cleanup pass)

---

## 1. Problem

Every cleanup PRD (131, 133, the half-done Phase B table-rename pass, the pending 4-table cleanup) runs into the same question:

> "Is this table / route / column / index actually used anywhere?"

Right now, answering that means: grep across the orchestrator repo, grep across frontend, read ORM models, cross-check migrations, and still miss dynamic SQL or downstream consumers. That's how we ended up with:

- `deliverables` as a shadow registry drifting from `agent_reports` (PRD-129 → PRD-133b).
- 8+ tables renamed blindly in the Phase B pass because we couldn't cheaply prove they were unused.
- 4 tables (`agent_performance`, `memory_items`, `tasks`, `artifacts`) still open one-at-a-time because each audit is a full half-hour of grep.
- Unknown number of dead routes, dead columns, unused indexes — silently accumulating cost and surface area.

Graphify already builds a code graph. The missing half is: **the DB as nodes in that same graph, with edges from code call sites to the tables/columns they touch.** Once those edges exist, dead-code detection and consolidation candidates fall out of single-query reports instead of ad-hoc grep.

## 2. Goal

A background scan (`/graphify db-scan`) that extends the existing graphify graph with DB-shape nodes and code↔DB edges, plus three saved report queries (dead tables, dead routes, consolidation candidates). Scope is **intelligence, not automation** — graphify suggests, a human approves any DROP.

## 3. Non-Goals

- No automated schema changes. Reports only.
- No ORM migration. Existing SQLAlchemy models stay as source-of-truth for app behaviour.
- No runtime instrumentation that slows production queries. `pg_stat_statements` is optional and read-only.
- No replacement of alembic or any migration tooling.
- Not a documentation generator. If someone wants a data dictionary, that's downstream and separate.

## 4. What Ships

| Component | Description |
|---|---|
| **DB snapshot scanner** | Single subcommand that reads `pg_catalog` + `information_schema` and emits JSON-shaped nodes/edges compatible with the existing graphify graph |
| **Code-to-DB edge extractor** | Walks the Python + TS AST / source, emits `module → table` edges labelled `reads` / `writes` / `models` |
| **Report queries** | Three saved graph queries: dead-tables, dead-routes, consolidation-candidates |
| **Optional runtime overlay** | Nightly pull of `pg_stat_statements` + `pg_stat_user_tables` merged as edge weights — gated behind a flag, off by default |
| **CLI / output format** | `/graphify db-scan`, `/graphify db-report <name>`. Outputs markdown for humans, JSON for programmatic use. |

## 5. Architecture

### 5.1 Node & edge schema (additions to graphify)

New node types:

| Type | Key fields | Source |
|---|---|---|
| `table` | name, schema, row_estimate, size_bytes, last_autovacuum | `pg_class`, `pg_stat_user_tables` |
| `column` | table_name, name, type, nullable, default | `information_schema.columns` |
| `index` | table_name, name, is_unique, is_primary, idx_scan_count | `pg_index`, `pg_stat_user_indexes` |
| `view` | name, definition | `pg_views` |
| `function` (SQL) | name, arg_types, return_type | `pg_proc` |
| `trigger` | name, table_name, event | `pg_trigger` |
| `route` | method, path, handler_fqn | FastAPI router introspection (already partially in graphify) |

New edge types:

| From → To | Label | Emitted by |
|---|---|---|
| `column → table` | `column_of` | DB scan |
| `table → table` | `fk_to` | DB scan (`pg_constraint`) |
| `view → table` | `depends_on` | DB scan (`pg_depend`) |
| `index → table` | `index_of` | DB scan |
| `function (code) → table` | `reads` / `writes` / `models` | Code walker |
| `route → function (code)` | `handler` | Code walker (already exists) |
| `table → table` (runtime overlay) | `cooccurs_in_query` (weighted) | pg_stat_statements parse |

### 5.2 Three scan passes

**Pass 1 — DB snapshot (`/graphify db-scan`):**
- Connect read-only to Railway Postgres (credentials from `DATABASE_URL`).
- Run ~8 `pg_catalog` queries (tables, columns, indexes, FKs, views, functions, triggers, stats).
- Emit nodes + structural edges to `graphify-out/db.json`.
- Merge into `graphify-out/graph.json` alongside code nodes.

**Pass 2 — Code→DB edges (part of `/graphify --update`):**
- **Python (orchestrator):**
  - SQLAlchemy `__tablename__` → `class → table` (`models` edge).
  - Grep-then-parse for `text("...")` string literals; extract table names via regex on `FROM|JOIN|UPDATE|INSERT INTO|DELETE FROM`.
  - Session method calls (`db.query(Model)`, `session.execute(text(...))`) → `function → table`.
- **TypeScript (frontend):**
  - Frontend rarely hits DB directly, but Drizzle/Prisma schemas (if any) emit `module → table`.
- **Confidence flag on each edge** (`static_match` / `ast_resolved` / `runtime_observed`) so reports can filter.

**Pass 3 — Runtime overlay (`/graphify db-scan --runtime`, optional):**
- Read `pg_stat_statements` for the last 7 days (requires extension already enabled in Railway Postgres — check; if not, skip pass 3 entirely).
- Read `pg_stat_user_tables.seq_scan / idx_scan / n_tup_ins/upd/del`.
- Emit weighted `runtime` edges on existing nodes so reports can distinguish "no static reference" from "no static reference AND no runtime traffic".

### 5.3 Report queries

Three report commands, each a saved graph query:

**`/graphify db-report dead-tables`**
- Tables with no inbound `reads`/`writes`/`models` edge from code.
- If runtime overlay enabled: AND zero traffic over the last 7 days.
- Output: markdown table with table name, row count, size, last_autovacuum, confidence.

**`/graphify db-report dead-routes`**
- Routes with no inbound reference from frontend code AND no test coverage AND no traffic in log-relay (if available).
- Output: method + path + handler file:line + last-known-hit.

**`/graphify db-report consolidation-candidates`**
- Pairs of tables ranked by:
  1. Column-name Jaccard similarity (≥ 0.5 columns in common).
  2. FK density between them.
  3. Shared-writer count (functions writing to both).
- Output: ranked pairs with overlap %, shared writers, and a brief reason. This is the report that would have caught `deliverables` drifting from `agent_reports` before PRD-129 shipped.

Bonus reports (low-cost follow-ons, same scan data):
- `dead-columns` — columns never read in any SQL AST AND never serialized by a Pydantic model.
- `dead-indexes` — `idx_scan = 0` over 7 days.
- `write-amplification` — tables written from >3 distinct code paths (PRD-129 would have flagged itself).
- `slow-query-attribution` — `pg_stat_statements` mean_exec_time joined back to the function that issued it.

## 6. Implementation phases

**Phase 1 (spike, ~0.5 day) — DB snapshot only.**
Ship `/graphify db-scan` that produces `graphify-out/db.json`. No code walker yet. Prove the `pg_catalog` extraction works and fits the graphify JSON shape.

**Phase 2 (~1 day) — Code-to-DB extractor.**
Extend the existing Python walker. Start with static `text("...")` + SQLAlchemy `__tablename__` — covers ~80% of call sites. Skip ORM query inference (`db.query(Model).filter(...)`) for v1; it's derivable from `models` edge + class reference.

**Phase 3 (~0.5 day) — Reports.**
Three markdown reports. Stop here unless specific follow-ons pay their way.

**Phase 4 (optional, ~0.5 day) — Runtime overlay.**
Only if pg_stat_statements is already on. Nightly cron into graphify-out.

**Total: ~2 days core, +0.5 day optional. No code shipped to production runtime — this is dev tooling.**

## 7. Concrete first use

Run Phase 1+2+3 against current main and answer, in one pass:

1. Are `agent_performance`, `memory_items`, `tasks`, `artifacts` actually used by live routes? (The 4 open items from the Phase B cleanup queue.)
2. Post-PRD-133b, how thin is the edge from any API code into `deliverables`? (Signal for when to plan PRD-134 deprecation.)
3. Of the 11 `b_*_<date>` renamed backup tables, are any still being read? (They shouldn't be — this is the watch-period check.)
4. What other consolidation risks exist that we haven't spotted yet?

## 8. Risks & caveats

| Risk | Mitigation |
|---|---|
| Dynamic SQL (table name built from a variable) missed by static pass | Runtime overlay closes the gap; also flag low-confidence edges explicitly in reports |
| SQLAlchemy models without explicit `__tablename__` | Walker must resolve class→table via DeclarativeBase inference; test case required |
| Views-of-views (transitive `pg_depend`) | Resolve recursively; trivial but easy to forget |
| Frontend dead-route detection depends on frontend being scanned too | Out of scope for Phase 1; frontend scan is already separate in graphify |
| False positives on "dead" | All reports are **advisory**. Never auto-drop. Human gate on every DELETE. |
| Reports become stale | Runs on `/graphify --update`; cron once a week is enough for most signals |
| Credentials | Read-only DB user for the scanner; never commit `DATABASE_URL` into graphify config |

## 9. Success criteria

1. `/graphify db-scan` runs in < 60s against Railway Postgres.
2. Code walker emits edges for ≥ 80% of `text("...")` call sites on first pass (manual sample audit on 20 files).
3. `dead-tables` report, when first run against main, flags only tables we agree are dead on manual review (false-positive rate ≤ 1 per 20 tables flagged).
4. `consolidation-candidates` report surfaces `deliverables ↔ agent_reports` as a top-5 pair when run against pre-PRD-133b main (retroactive validation).
5. Zero impact on production runtime (tooling-only).

## 10. Out of scope / future

- **Automated DROP.** Explicitly rejected. Reports stop at recommendation.
- **Cross-database joins** (only single DB at launch — Railway Postgres).
- **Redis / S3 usage mapping.** Same principle applies but different sources; separate PRD if valuable.
- **Vector store (S3 Vectors / Qdrant) lineage.** Arguably high-value but structurally different; separate PRD.
- **"Graph-the-agents"** (agent → tool → table). Conceptually related but different graph; out of scope here.

## 11. Open questions

1. Does Railway Postgres have `pg_stat_statements` enabled already? (Check before committing to Phase 4.)
2. Should the DB scan live in the `orchestrator` repo (near the code walker) or in the graphify CLI itself? Leaning toward graphify CLI — keeps the orchestrator repo clean.
3. Confidence-tier colouring in reports — worth the effort in v1, or plain markdown enough?
4. When is "dead" truly dead — 7 days no traffic? 30? Parametrize with default 7.

---

## Appendix A — The `pg_catalog` queries (ready to port)

Rough sketch of the six queries that feed Phase 1. These are read-only, safe to run on production, and return in < 1s each on our DB size.

```sql
-- tables + size + last autovacuum
SELECT c.relname AS table_name,
       c.reltuples::bigint AS row_estimate,
       pg_total_relation_size(c.oid) AS size_bytes,
       s.last_autovacuum
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
  LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
 WHERE c.relkind = 'r' AND n.nspname = 'public';

-- columns
SELECT table_name, column_name, data_type, is_nullable, column_default
  FROM information_schema.columns
 WHERE table_schema = 'public';

-- foreign keys
SELECT conrelid::regclass AS from_table,
       confrelid::regclass AS to_table,
       conname, pg_get_constraintdef(oid) AS def
  FROM pg_constraint
 WHERE contype = 'f';

-- indexes + usage
SELECT i.relname AS index_name,
       t.relname AS table_name,
       ix.indisunique, ix.indisprimary,
       s.idx_scan
  FROM pg_index ix
  JOIN pg_class i ON i.oid = ix.indexrelid
  JOIN pg_class t ON t.oid = ix.indrelid
  LEFT JOIN pg_stat_user_indexes s ON s.indexrelid = ix.indexrelid
 WHERE t.relkind = 'r';

-- views + dependencies
SELECT v.viewname, v.definition
  FROM pg_views v WHERE v.schemaname = 'public';

SELECT d.refobjid::regclass AS depends_on,
       dep.objid::regclass AS used_by
  FROM pg_depend dep
  JOIN pg_rewrite r ON r.oid = dep.objid
  JOIN pg_class d  ON d.oid = dep.refobjid
 WHERE dep.classid = 'pg_rewrite'::regclass;
```

## Appendix B — Graph query sketches (report side)

```
# dead-tables
MATCH (t:table) WHERE NOT (:function)-[:reads|writes|models]->(t) RETURN t

# dead-routes
MATCH (r:route) WHERE NOT (:module)-[:references]->(r) RETURN r

# consolidation candidates
MATCH (a:table), (b:table) WHERE a<>b
WITH a, b, jaccard(a.columns, b.columns) AS sim
WHERE sim >= 0.5
RETURN a, b, sim ORDER BY sim DESC
```

(Pseudo-Cypher — graphify's actual query language may differ; translate at implementation time.)

---

**Next action:** Open a fresh context on this PRD. Phase 1 is a half-day spike — ship it, validate against the 4 open Phase B cleanup items, and iterate from what the first report actually surfaces.

---

# §12 — First Cleanup: 51 Dead Tables Drop Plan

**Date drafted:** 2026-04-25
**Status:** Plan, awaiting execution
**Branch:** `chore/prd-135-dead-tables-cleanup`
**Inputs:** `graphify-out/REPORT_dead_tables.md` (post-runtime-overlay, 11h sweep window)

## §12.1 Headline numbers

| Metric | Value |
|---|---:|
| Tables to drop | **51** |
| Total disk to reclaim | **14 MB** (all empty bloat) |
| Live rows across all 51 | **0** |
| FK constraints to drop | **39** (all outbound — dead → live) |
| Live tables affected | **0** (no inbound FKs from live tables) |
| Risk of data loss | **None** (zero rows everywhere) |
| Risk of breaking live code | **None** (zero static + zero runtime references) |

The "snapshot, delete, test" pattern still applies, but snapshots are schema-only — there's nothing else to preserve.

## §12.2 Confirmed carve-outs (NOT in the drop list)

These showed runtime traffic during the sweep — keep them, fix the AST walker:

| Table | Why kept | Walker fix needed |
|---|---|---|
| `workspace_graphs` | PRD-130 graph artefact storage — 9 calls, 5 writes during sweep. 4 raw SQL paths in `orchestrator/core/graph_storage.py`. | Walker should parse `sa_text("...")` literals. |
| `workflow_agents` | Cascade-delete association table (`workflows ↔ agents`). Will go with PRD-125 when `workflows` is decommissioned. | Walker should track SQLAlchemy `Table()` declarations in addition to `__tablename__` model classes. |

## §12.3 Bucket strategy

Six buckets, ordered safest-first. **One PR per bucket** — incremental commits within the same branch. Each bucket: snapshot → migration → deploy → smoke test on Railway → next bucket.

### Bucket 1 — `b_*_<date>` backup tables (11 tables, 11 MB)

Pure backup remnants from earlier rename/cleanup passes. Zero ambiguity.

| Table | Size |
|---|---:|
| `b_backup_document_chunks_20251024_20260424` | 9.3 MB |
| `b_mcp_tools_backup_20260424` | 520 kB |
| `b_tools_backup_20260424` | 520 kB |
| `b_agent_messages_20260424` | 32 kB |
| `b_agent_performance_tracking_20260424` | 32 kB |
| `b_field_states_20260424` | 32 kB |
| `b_field_interactions_20260424` | 24 kB |
| `b_historical_tasks_20260424` | 16 kB |
| `b_task_assignments_20260424` | 16 kB |
| `b_agent_runtimes_20260424` | 8 kB |
| `b_task_decompositions_20260424` | 8 kB |

**Smoke test after drop:** none required. These were already dead before any rename pass.

### Bucket 2 — Context-engineering experiments (10 tables, 1.7 MB)

Old "context engine" wave that didn't ship. The current context system uses `documents` / `document_chunks` / `agent_memories_short_term` (memory_short_term).

`context_examples`, `context_optimizations`, `context_patterns`, `context_queries`, `context_sources`, `context_templates`, `context_usage`, `context_permissions`, `entity_clusters`, `shared_contexts`

**Smoke test after drop:** RAG / document retrieval flow. Upload a doc, ask a question, verify chunks return.

### Bucket 3 — Multi-agent reasoning experiments (7 tables, ~112 kB)

Coordination/consensus features designed but never wired. Agent-to-agent today goes through `messages` (alive: 150 calls) and `board_tasks` (alive: 159 calls).

`agent_coordination`, `multi_agent_reasoning`, `agent_behavior_monitoring`, `agent_performance`, `collaboration_proposals`, `consensus_votes`, `message_broadcasts`

**Smoke test after drop:** run a mission with ≥2 agents — verify they handoff via the live path.

### Bucket 4 — Superseded RAG/memory (4 tables, ~1.1 MB)

`vector_documents` (1 MB historical) and `document_embeddings` were the legacy RAG path; replaced by S3 Vectors + `document_chunks` (alive: 644 calls). `agent_memories` was replaced by `memory_short_term` (alive: 418 calls). `analytics_snapshots` was the dashboard prototype path; replaced by widgets reading live tables.

`vector_documents`, `document_embeddings`, `agent_memories`, `analytics_snapshots`

**Smoke test after drop:** RAG question on uploaded doc; agent recalls prior turn from short-term memory; widget renders on dashboard.

### Bucket 5 — Built-but-never-wired admin features (15 tables, ~256 kB)

Surfaces designed in the platform's expansion phase but never delivered or replaced by alternatives.

| Table | Replacement / fate |
|---|---|
| `dashboard_configs` | Widgets are config-less today (auto-discovery) |
| `custom_metrics` | Not yet implemented |
| `alert_configs` | Not yet implemented |
| `compliance_events` | Not yet implemented |
| `marketplace_submissions` | No submission flow live |
| `knowledge_collections`, `knowledge_collection_items`, `knowledge_usage` | Replaced by `documents` + `kb_types` (alive) |
| `usage_logs`, `usage_summary` | Replaced by `llm_usage` (alive: 534 calls) |
| `search_analytics` | Not implemented |
| `execution_contexts` | Workflow-era; PRD-125 supersedes |
| `integration_analysis` | Not implemented |
| `workspace_shares` | No sharing flow live |
| `api_keys` | ⚠ verify before drop — see §12.5 |

**Smoke test after drop:** marketplace browse, knowledge bases page, billing/usage page, settings pages.

### Bucket 6 — Misc legacy (5 tables, ~144 kB)

`playbooks` — confirmed legacy (was learning-patterns table, the canonical Playbook is `workflow_recipes`).
`schema_versions` — superseded by alembic.
`code_symbols`, `code_edges` — graphify-related tables that were never wired into the platform's own code graph.

**Smoke test after drop:** none — these are pure legacy.

## §12.4 Per-bucket execution recipe

```
For each bucket:

  1. Snapshot (schema only, since rows = 0):
     pg_dump $DATABASE_URL --schema-only --table=<t1> --table=<t2> ... \
       > graphify-out/snapshots/bucket-N-pre-drop.sql

  2. Generate migration:
     orchestrator/alembic/versions/prd135_bucket_N_<name>.py
       upgrade(): drop FKs (if any) → drop tables
       downgrade(): recreate from snapshot

  3. Run locally:
     alembic upgrade head
     orchestrator/tests run smoke suite

  4. Deploy to Railway:
     git push (auto-deploys)
     wait for health check

  5. Smoke test (per bucket §12.3):
     hit relevant pages, verify no errors

  6. Re-run runtime overlay:
     python3 scripts/graphify_runtime_overlay.py --update-dead-tables
     verify dropped tables disappear, no new "investigate" entries

  7. Commit. Move to next bucket.
```

## §12.5 Pre-execution checks (before bucket 5)

Two tables on the drop list need a manual confirmation before dropping. Both have user-flow-driven write paths that the 11h sweep wouldn't have triggered:

| Table | What to check |
|---|---|
| `api_keys` | Verify there's no `Settings → API Keys` page or token-issuance flow before dropping. Has FKs to `users` + `workspaces`. The current auth path is `auth.py` + JWT — `api_keys` looks like a never-built personal-token feature. |
| `marketplace_submissions` | Confirm no plugin/skill submission flow exists today. Per VISION.md §4.10, marketplace stays as-is — but submission flow may not be live yet. |

If either turns up live code, lift to "investigate" and remove from drop list.

## §12.6 Rollback procedure

If a smoke test fails after a bucket:

```
1. alembic downgrade -1   (reverts the bucket's migration)
2. Or: psql $DATABASE_URL < graphify-out/snapshots/bucket-N-pre-drop.sql
3. git revert <commit>
4. push, redeploy
5. Re-investigate the failed table — it had a code path we missed.
```

Recovery is fast because there's no data: structure restore alone gets us back.

## §12.7 Branch & PR shape

- **Branch:** `chore/prd-135-dead-tables-cleanup` (new, off `main`)
- **PR strategy:** **One PR per bucket.** Six PRs total. Each PR includes:
  - Migration file
  - Updated `REPORT_dead_tables.md`
  - Snapshot SQL file in `graphify-out/snapshots/`
  - Smoke test notes in PR description
- **Why per-bucket not single PR:** lets you pause between buckets if anything wobbles, and gives a clean revert surface.

## §12.8 Walker improvements (follow-up, not blocking)

Post-cleanup, two AST walker enhancements close the carve-out gap so future runs catch raw SQL:

1. **`sa_text("...")` literal parsing** — extract table refs from string args of `sa.text()` / `sa_text()` / `text()`.
2. **`Table('name', metadata, ...)` recognition** — track SQLAlchemy core `Table()` declarations alongside ORM `__tablename__`.

Both are ~50 lines in `scripts/graphify_db_scan.py`. Defer to a follow-on `chore/graphify-walker-improvements` branch.

## §12.9 Total impact summary

| Outcome | Value |
|---|---|
| Tables removed | 51 (200 → 149) |
| Disk reclaimed | 14 MB |
| FK constraints removed | 39 |
| Schema cognitive load | -25% |
| Migration files added | 6 |
| Risk | Negligible (0 rows, 0 inbound code refs, 0 inbound runtime traffic) |
| Time to execute | ~2-3 hours including smoke tests |

---
