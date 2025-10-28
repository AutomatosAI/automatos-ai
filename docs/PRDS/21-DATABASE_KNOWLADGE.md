# PRD-21: Database Knowledge Source (Wren-style Direct Query)

Status: Draft for Final Review  
Priority: P0  
Owners: Orchestrator, Knowledge, Frontend, Security  
Depends on: PRD-18 (Credentials), PRD-17 (Dynamic Tool Assignment), PRD-16 (LLM Orchestrator), PRD-09 (Context Engineering)

---

## 0. Executive Summary
Enable users to connect their own databases as first-class Knowledge Sources and query them via natural language using schema-driven text-to-SQL. Data remains in-place (no ingestion). The system introspects schema metadata, applies strict SQL validation (SELECT-only, LIMIT/timeout, allowlist), executes queries against the user’s database using stored credentials, and returns structured results with basic visualizations. Agents can combine DB answers with Documents (RAG) and Code (CodeGraph) to produce unified insights.

---

## 1. Goals & Non-Goals

### Goals
1. Add Database as a Knowledge Source type with in-place querying (no copying data).
2. Introspect and persist schema metadata (tables, columns, types, PK/FK, samples, counts).
3. Generate safe SQL from NL using LLM + schema + optional semantic layer.
4. Enforce safety: SELECT-only, LIMIT (default 1000), execution timeout (default 30s), table/column allowlist.
5. Provide UI to add sources (Knowledge Base), run queries and chart results (Context Engineering).
6. Expose per-source and generic agent tools (query_database) for workflows and agents.
7. Audit all queries (who/when/source/SQL/rows/duration/success/error). Multi-tenant safe by design.

### Non-Goals (MVP)
- ETL or full data sync of SaaS sources (future). 
- Complex BI (pivoting, dashboarding beyond basic charts). 
- Complex governance/versioning for semantic layer (MVP supports simple JSON definitions + validation).

---

## 2. User Stories
- As a Product Analyst, I can connect our production Postgres and ask “Top 10 products by revenue last month” and get a table and bar chart.
- As a RevOps user, I can define business metrics (total_revenue, active_customers) and reuse them across questions to standardize results.
- As an Agent, I can decide when to use DB vs Documents vs Code and synthesize a single answer.
- As a Security Admin, I see an audit log of all DB queries executed (user, time, SQL, row count, success) and confirm SELECT-only.

---

## 3. Architecture

### 3.1 High-Level Flow
```
Settings > Credentials (encrypted) → Knowledge Base > Add Source > Database → Test + Introspect schema
  → Persist metadata (JSONB) → Context Engineering: NL→SQL → Validator (SELECT/LIMIT/timeout/allowlist)
  → Execute on user's DB → Return rows + meta → Table/Bar/Line → Save/Export → Agent tools available
```

### 3.2 Components
- Backend Services:
  - DatabaseIntrospectionService: connects via credential, introspects schema, samples, relationships, row counts.
  - TextToSQLService: builds LLM prompt from schema + semantic layer (optional); parses/validates SQL; executes.
  - SQLValidator: SELECT-only enforcement, LIMIT injection, timeout, allowlist by introspected schema.
- Data Models:
  - `database_knowledge_sources(id, tenant_id, name, credential_id, schema_metadata JSONB, semantic_layer JSONB, stats JSONB, last_introspected TIMESTAMP, created_at, updated_at)`
  - `database_query_audit(id, tenant_id, source_id, user_id, sql, duration_ms, row_count, success, error, created_at)`
- Frontend:
  - Knowledge Base: Add Database Source (wizard), Source Detail (schema preview, quick query, audit tab).
  - Context Engineering: Query editor, result table, basic charts (table, bar, line), save/export (CSV), add-to-context.
- Agent Tools:
  - Per-source tool: `query_{source_slug}(question)` auto-created.
  - Generic tool: `query_database(source_id, question)`.

---

## 4. Data Model & Migrations

### 4.1 Tables
```sql
-- database_knowledge_sources
CREATE TABLE IF NOT EXISTS database_knowledge_sources (
  id SERIAL PRIMARY KEY,
  tenant_id INTEGER NOT NULL REFERENCES tenants(id),
  name VARCHAR(255) NOT NULL,
  credential_id INTEGER NOT NULL REFERENCES credentials(id),
  schema_metadata JSONB NOT NULL,     -- tables, columns, pk/fk, samples, counts
  semantic_layer JSONB DEFAULT '{}'::jsonb, -- metrics/dimensions MVP
  stats JSONB DEFAULT '{}'::jsonb,
  last_introspected TIMESTAMP NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dks_tenant ON database_knowledge_sources(tenant_id);
CREATE INDEX IF NOT EXISTS idx_dks_updated ON database_knowledge_sources(updated_at DESC);

-- database_query_audit
CREATE TABLE IF NOT EXISTS database_query_audit (
  id SERIAL PRIMARY KEY,
  tenant_id INTEGER NOT NULL REFERENCES tenants(id),
  source_id INTEGER NOT NULL REFERENCES database_knowledge_sources(id),
  user_id VARCHAR(255) NOT NULL,
  sql TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  row_count INTEGER NOT NULL,
  success BOOLEAN NOT NULL,
  error TEXT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dqa_tenant ON database_query_audit(tenant_id);
CREATE INDEX IF NOT EXISTS idx_dqa_source_created ON database_query_audit(source_id, created_at DESC);
```

### 4.2 `schema_metadata` JSONB Structure (example)
```json
{
  "database": {"engine": "postgres", "version": "15", "timezone": "UTC"},
  "tables": [
    {
      "name": "customers",
      "schema": "public",
      "row_count": 15234,
      "columns": [
        {"name": "id", "type": "integer", "nullable": false, "primary_key": true},
        {"name": "email", "type": "varchar(255)", "nullable": false, "unique": true, "samples": ["a@x.com"]},
        {"name": "tier", "type": "varchar(50)", "nullable": true, "samples": ["premium", "enterprise"]},
        {"name": "created_at", "type": "timestamp", "nullable": false}
      ]
    }
  ],
  "relationships": [
    {"from_table": "orders", "from_column": "customer_id", "to_table": "customers", "to_column": "id", "type": "many_to_one"}
  ]
}
```

### 4.3 `semantic_layer` JSONB Structure (MVP)
```json
{
  "metrics": {
    "total_revenue": {
      "sql": "SUM(orders.total_amount)",
      "description": "Sum of all order amounts",
      "type": "currency",
      "tables": ["orders"]
    },
    "active_customers": {
      "sql": "COUNT(DISTINCT customers.id) WHERE EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = customers.id AND orders.created_at > NOW() - INTERVAL '90 days')",
      "description": "Customers with orders in last 90d",
      "type": "count",
      "tables": ["customers", "orders"]
    }
  },
  "dimensions": {
    "time": {
      "day": "DATE(created_at)",
      "month": "DATE_TRUNC('month', created_at)",
      "quarter": "DATE_TRUNC('quarter', created_at)"
    },
    "customer_segment": {
      "tier": "customers.tier",
      "region": "customers.region"
    }
  }
}
```

---

## 5. API Design (MVP)

Base: `/api/knowledge/sources/database`

1) `POST /` (create → test → introspect)
- Body: `{ name, credential_id, options: { include_samples, include_stats, timezone } }`
- Flow: test connection → introspect → persist → return source with schema summary

2) `POST /{id}/introspect`
- Re-run introspection; update `schema_metadata`, `last_introspected`

3) `GET /{id}`
- Detail: source, schema summary, last_introspected, stats

4) `GET /{id}/schema`
- Full `schema_metadata` for UI (schema browser, prompt context)

5) `POST /{id}/semantic`
- Upsert metrics/dimensions; validate expressions compile against schema

6) `POST /{id}/query`
- Body: `{ question: string, params?: {...} }`
- Steps: build LLM prompt from schema subset + semantic; generate SQL; validate; execute; audit
- Response: `{ sql, explanation, columns, rows, visualization: { type, x, y }, stats: { duration_ms, row_count } }`

Security headers: tenant scoping, auth, rate limiting per tenant.

---

## 6. NL→SQL Prompting & Validation

### 6.1 Prompt Inputs
- Relevant tables/columns (limited subset by keyword/metric match)
- Relationships (FK join hints)
- Sample values (a few per column)
- Semantic metrics/dimensions (if defined)
- Rules: SELECT-only, LIMIT 1000, timeout expectations, dialect hints

### 6.2 Validator Rules
- Parse with sqlparse; reject if statement type != SELECT
- Inject/verify LIMIT ≤ configured max (default 1000)
- Enforce execution timeout (default 30s)
- Verify referenced tables/columns exist in `schema_metadata`
- Deny keywords: UPDATE/INSERT/DELETE/ALTER/DROP/TRUNCATE/CTE writing into …

### 6.3 Execution
- Use dialect-specific driver (MVP: psycopg2-binary for Postgres, PyMySQL for MySQL)
- Stream rows to avoid memory blowups; truncate with LIMIT

---

## 7. Frontend UX

### 7.1 Knowledge Base
- Add Source → Database (wizard): name, credential select, options (samples, stats, timezone) → Test → Introspect → Summary
- Source Detail: schema browser, quick query (NL→SQL), audit tab (latest queries, export CSV), refresh schema

### 7.2 Context Engineering
- Query editor (natural language)
- Result viewer: table; chart picker with defaults (table, bar, line)
- Save view; export CSV; add result into prompt context for downstream workflows
- Display `sql` and explanation; allow copy

---

## 8. Agent & Workflow Integration
- Auto-create per-source tool `query_{slug}` and generic `query_database`
- Tool return schema: `{ data, sql_generated, explanation, chart_suggestion }`
- Orchestrator intent routing: analytical → DB; conceptual → RAG; code → CodeGraph; mixed → multi-source synthesize

---

## 9. Security, Compliance, Multi-Tenancy
- Credentials remain encrypted; never persisted in logs
- SELECT-only, LIMIT, timeout, allowlist guardrails
- Per-tenant scoping across sources, audits, and access control
- Optional RLS filters (Phase 2): append `WHERE tenant_id = {ctx.tenant_id}`
- Full audit trail for compliance

---

## 10. Performance & Caching
- Schema metadata cached in DB; manual/cron refresh
- Query result caching (Phase 2) via Redis (hash of SQL → payload with TTL)
- Large rowsets: enforce LIMIT, pagination UI (Phase 2)

---

## 11. Dependencies & Drivers
- MVP: Postgres (psycopg2-binary), MySQL (PyMySQL)
- Phase 2: Snowflake (`snowflake-connector-python`), MSSQL (`pyodbc` + driver image), BigQuery (`google-cloud-bigquery`)

---

## 12. Success Metrics
- Connect → first query in < 3 minutes
- 10 canned queries pass on Postgres & MySQL
- 95% queries: <5s small schema, <15s large schema
- 0 destructive queries executed
- Agents produce unified answers using DB + Docs + Code in 3 demo tasks

---

## 13. Phasing & Timeline
- Week 1:
  - Migrations, Postgres driver, introspection service, endpoints: create/test/introspect
- Week 2:
  - NL→SQL generation, validator, `/query` endpoint, audits
  - Knowledge Base UI + Source Detail (quick query, schema preview, audit)
- Week 3:
  - Context Engineering integration (results, charts), agent tools, tests, docs
- Week 4:
  - MySQL support; semantic layer MVP UI; polish, metrics

---

## 14. Open Questions
1. Which Phase 2 DB to prioritize (Snowflake vs MSSQL)?
2. Add optional SSH tunnels for on-prem sources?
3. Column-level masking for PII in Phase 2?
4. Chart library choice confirmation (Recharts vs Chart.js)?

---

## 15. Appendix: How This Differs From RAG/CodeGraph
- Documents/Code use vector similarity (embeddings) for semantic retrieval
- Databases use schema-driven text-to-SQL for exact computation
- Unified orchestrator picks the right method per question; multi-source synthesis produces richer answers
