# Database Knowledge Source (Wren-style) - PRD

## 1) Overview
Enable users to add their own databases as knowledge sources and query them via natural language using text-to-SQL (schema-driven) while keeping data in-place. Integrates with existing Credentials, Agent tools, Knowledge Base, and Context Engineering for results review and charting.

- Deployment: Native service (no Wren dependency) [1A]
- Connectivity: Direct-query and optional import/sync later for SaaS [2C]
- Initial DBs: Use existing credential types (PostgreSQL, MySQL, MongoDB, Snowflake, etc.)
- Credentials: Use Settings > Credentials (encrypted, reusable) [4A]
- Security: SELECT-only enforcement [5A]; Full validation (limit, timeout, whitelist) [6D]
- Semantic Layer: Full UI at MVP (metrics/dimensions) [7C]
- Agent Tools: Both per-source tools and generic parameterized tool [8C]
- UI Placement: Knowledge Base (source setup) + Context Engineering (querying, charts) [9D]
- Caching: Cache schema metadata; refresh on demand/schedule [10B]
- Multi-tenant: Per-tenant isolation for creds/sources [11A]
- Success: MVP targets include agent-driven charts and synthesis [12(all feasible)]
- Parity Targets: Semantic layer + charts [13BC]

## 2) Goals
1. Users can add a database source, introspect schema (no data copy), and run NL→SQL queries.
2. Agents can use DB tools alongside Documents (RAG) and Code (CodeGraph) for unified answers.
3. Provide an MVP semantic layer UI for defining metrics/dimensions per source.
4. Display tabular results and basic charts; allow saving/summarizing results.
5. Enforce strict safety: read-only SQL, timeouts, row limits, schema whitelist, audit logs.

## 3) User Stories
- As a user, I can add my product database as a knowledge source and run natural language queries (e.g., “top 10 products by revenue last month”).
- As an analyst, I can define metrics/dimensions in a semantic layer to standardize business logic (e.g., total_revenue, active_users by month).
- As a developer, my agent can decide when to use DB vs Docs vs Code and combine results.
- As an admin, I can review audit logs for all DB queries (who/when/what SQL).

## 4) Functional Requirements
1. Add Database Source
   1.1 Select existing credential (category=database) from Settings > Credentials.
   1.2 Test connection (status, latency, role capabilities).
   1.3 Introspect schema: tables, columns, types, PK/FK, row counts, sample values (configurable).
   1.4 Store metadata in platform DB; no user data is copied.
   1.5 Optional: schedule schema refresh; manual refresh supported.

2. Query (Text-to-SQL)
   2.1 Generate SQL from natural language using LLM with schema + optional semantic layer.
   2.2 Validate SQL: SELECT-only; reject DDL/DML; enforce LIMIT (default 1000); max execution time (default 30s); table/column whitelist.
   2.3 Execute against user’s database using credential; return rows + column meta.
   2.4 Support parameters (time ranges, groupings); suggest follow-up questions.

3. Semantic Layer (MVP UI)
   3.1 Per-source metrics: name, SQL expression, description, type (count/currency/rate), tables.
   3.2 Per-source dimensions: time grain (day/week/month/quarter), categories (region/tier/etc.).
   3.3 Validation: verify referenced tables/columns exist; preview SQL.

4. Agent Tools
   4.1 Auto-create per-source tool: e.g., query_customer_db(question).
   4.2 Provide generic tool: query_database(source_id, question).
   4.3 Tools return: data, generated SQL, explanation, visualization suggestion.

5. UI/UX
   5.1 Knowledge Base: Add Source > Database dialog; status (Connected/Indexed), schema summary.
   5.2 Source Detail: Quick Query input; recent queries; schema browser; refresh button.
   5.3 Context Engineering: Results review; charts (table/bar/line), save view, export CSV; add to prompt context.

6. Security & Audit
   6.1 Enforce SELECT-only at multiple layers (validator + connection role guidance).
   6.2 Enforce LIMIT, timeout, max rows; configurable per source.
   6.3 Optional row-level filters (future) appended to generated SQL.
   6.4 Audit log each query: user, source_id, SQL, duration, row count, success/error.

7. Caching & Performance
   7.1 Cache schema metadata (DB table); refresh on demand/schedule.
   7.2 Optional: cache query results hash(SQL)->payload in Redis (future phase).

## 5) Non-Goals (Out of Scope for MVP)
- Data import/synchronization pipelines for SaaS sources (future; 2C acknowledges later).
- Advanced BI features (pivot tables, alerting, dashboards beyond basic charts).
- Complex semantic governance (versioned metrics, approvals).

## 6) Design Considerations
- Query routing: database (text-to-SQL) vs documents (vector) vs code (vector) decided by intent classifier.
- LLM prompts include: schema subset, relationships, sample values, semantic metrics, and rules.
- Visualization suggestion derived from column types and grouping (e.g., time series → line chart).
- Multi-tenant: all credentials and knowledge sources scoped by tenant_id.

## 7) Technical Considerations
- Data models (new):
  - database_knowledge_sources(id, tenant_id, name, credential_id, schema_metadata JSONB, semantic_layer JSONB, stats JSONB, last_introspected TIMESTAMP)
  - database_query_audit(id, tenant_id, source_id, user_id, sql, duration_ms, row_count, success, error, created_at)

- API (MVP):
  - POST /api/knowledge/sources/database (create + test + introspect)
  - POST /api/knowledge/sources/database/{id}/introspect (refresh)
  - POST /api/knowledge/sources/database/{id}/query (NL→SQL→execute)
  - GET  /api/knowledge/sources/database/{id} (detail)
  - GET  /api/knowledge/sources/database/{id}/schema (schema metadata)
  - POST /api/knowledge/sources/database/{id}/semantic (upsert metrics/dimensions)

- Validation:
  - SQL parser (sqlparse) + allowlist (introspected tables/columns)
  - Enforce SELECT-only, LIMIT, timeout

- LLM:
  - Use existing OpenAI/Anthropic integration; temperature low for determinism

## 8) Success Metrics
- Time to first value: connect → query in < 3 minutes
- 10 canned NL→SQL queries pass (PostgreSQL + MySQL)
- 95% queries complete < 5s (schema small), < 15s (schema large)
- 0 destructive queries executed (validator + role guardrail)
- Agent successfully synthesizes DB + docs + code in at least 3 demo scenarios

## 9) Open Questions
1. Which DBs to certify at MVP beyond Postgres/MySQL (Snowflake/MS SQL first)?
2. Do we add optional RLS UI in MVP or Phase 2?
3. Charting library choice (Recharts/Chart.js) and theming.
4. Governance for semantic layer (who can edit metrics?).
5. Where to expose saved query views (Context Engineering vs Dashboards?).

## Appendix: Credentials & Searching Differences (Answer to 3)
- Credentials don’t change “searching”—they change the **connector & dialect** used.
- Unstructured search (docs/code) uses **embeddings + vector similarity** (no credentials).
- Structured queries (databases) use **text-to-SQL** with credentials per source to execute SQL.
- Each credential type influences: connection string, introspection queries, SQL dialect rules, and driver (e.g., psycopg2, mysqlclient, snowflake-connector, pyodbc).
