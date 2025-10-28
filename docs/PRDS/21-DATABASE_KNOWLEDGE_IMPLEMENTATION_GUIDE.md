# PRD-21: Database Knowledge - Implementation Guide

Status: Implemented (backend MVP)  
Scope: Introspection, Safe SQL Execution, Auditing  
Impacts: Backend API, Services, DB (existing tables)

---

## 1) What Was Added/Changed

- Added
  - `orchestrator/services/database_introspection.py`
    - Postgres/MySQL schema introspection: tables, columns, sample values, row counts, FK relationships
  - `orchestrator/services/sql_validator.py`
    - Minimal SQL validator: SELECT-only, LIMIT enforcement, denylist (DDL/DML), optional table allowlist from schema
- Updated
  - `orchestrator/api/database_knowledge.py`
    - `POST /api/knowledge/sources/database/{id}/introspect`: resolve DB creds → introspect → persist `schema_metadata` + `last_introspected`
    - `GET  /api/knowledge/sources/database/{id}/schema`: return persisted `schema_metadata` (uses cache if available)
    - `POST /api/knowledge/sources/database/{id}/query/sql`: dev-only safe SQL exec; validates then runs SELECT with timeouts; audits success/failure to `database_query_audit`
- Removed
  - `orchestrator/database/migrations/20251025_add_database_knowledge.sql` (redundant with single-source schema)

No import-time credential lookups; startup remains safe without API keys in env.

---

## 2) Database Impact

- No new migrations required (schema already defined in single-source SQL):
  - `database_knowledge_sources` (stores `schema_metadata`, `semantic_layer`, settings, stats)
  - `database_query_audit` (query audit trail)
- File: `orchestrator/database/init_complete_schema.sql` contains the canonical definitions and indexes.

If your DB was initialized with `init_complete_schema.sql`, you are ready. Otherwise initialize using your standard process that executes this file.

---

## 3) Dependencies

- Required (already common): `psycopg2-binary`
- Optional (for MySQL support): add to `orchestrator/requirements.txt`
```txt
PyMySQL==1.1.0
```
- SQLAlchemy is used for execution/metadata access (already present in project).

---

## 4) Credentials Integration

- Credentials are resolved at runtime via `CredentialStore` (from DB; audited). No `.env` required.
- A `database_knowledge_sources` row must reference a valid credential (`credential_id`) and specify `dialect` (e.g., `postgresql`, `mysql`).

---

## 5) API Usage (Backend)

Replace `{SOURCE_ID}` with the ID of an existing database source that points to your DB credential.

- Re-introspect schema
```bash
curl -X POST http://localhost:8000/api/knowledge/sources/database/{SOURCE_ID}/introspect
```
- Fetch schema metadata
```bash
curl http://localhost:8000/api/knowledge/sources/database/{SOURCE_ID}/schema
```
- DEV ONLY: Run validated SELECT (LIMIT enforced; audited)
```bash
curl -X POST http://localhost:8000/api/knowledge/sources/database/{SOURCE_ID}/query/sql \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT 1 AS ok"}'
```
Response shape:
```json
{
  "sql": "SELECT 1 AS ok LIMIT 1000",
  "reasons": ["LIMIT 1000 injected"],
  "columns": ["ok"],
  "rows": [{"ok": 1}],
  "stats": {"duration_ms": 3, "row_count": 1}
}
```

---

## 6) How It Works (Flow)

- Introspection
  - API fetches creds via `CredentialStore` → `DatabaseIntrospectionService` queries information_schema and FK metadata → persists JSONB `schema_metadata` and `last_introspected`.
- Safe SQL Execution
  - `SQLValidator` checks SELECT-only, denies DDL/DML, caps/injects LIMIT, optionally verifies referenced tables exist in `schema_metadata`.
  - For Postgres, `statement_timeout` is set per-connection (`SET LOCAL`).
  - Results streamed to JSON; success/failure recorded in `database_query_audit` with timings and row count.
- Auditing
  - Every execution (including validation failures) writes an entry into `database_query_audit`.

---

## 7) Rollout Steps

1. (Optional) Add MySQL support
```bash
echo "PyMySQL==1.1.0" >> automatos-ai/orchestrator/requirements.txt
```
2. Rebuild backend when you’re ready to run (not required just to merge code).
3. Ensure you have a `database_knowledge_sources` record that:
   - references an existing DB credential (`credential_id`)
   - sets `dialect` to `postgresql` or `mysql`
4. Introspect → Verify schema → (dev) Run a safe SELECT using endpoints above.

---

## 8) Testing & Verification

- Happy path
  - Introspect completes; `schema_metadata` contains tables, columns, `relationships`, `stats.introspection_ms`.
  - Schema endpoint returns the same structure.
  - DEV SQL endpoint executes SELECT and returns rows; an audit record exists with `success=true`.
- Safety checks
  - Non-SELECT or DDL/DML statements are rejected with 400; audit record has `success=false` and error.
  - LIMIT is injected or capped to `source.max_rows_limit`.
- Timeouts
  - For Postgres, long-running queries are terminated per `query_timeout_seconds`.

---

## 9) Troubleshooting

- "Unsupported dialect"
  - Ensure `database_knowledge_sources.dialect` starts with `postgresql` or `mysql`.
- "Failed to resolve credentials"
  - Verify `credential_id` is valid, active, and decryptable in the target environment.
- Introspection returns no tables
  - Check DB user permissions and that the schema is not excluded by filters; for Postgres, system schemas are skipped by default.
- Query rejected by validator
  - Ensure SELECT-only; remove CTEs with DML; reference only tables present in `schema_metadata`.

---

## 10) Next (Out of Scope of This Commit)

- Natural Language → SQL generation with LLM prompts and semantic layer hints
- Frontend: Knowledge Base source detail (schema browser, quick query, audit)
- Frontend: Context Engineering analytics UI (charts, save/export)
- Result caching (Phase 2)

---

## 11) Change Log (This Implementation)

- New: `services/database_introspection.py`, `services/sql_validator.py`
- Updated: `api/database_knowledge.py` (introspection, schema fetch, dev-only SQL exec + auditing)
- Removed: redundant migration in `orchestrator/database/migrations/`
- DB: No new migrations; uses `init_complete_schema.sql`
