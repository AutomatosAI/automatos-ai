# PRD 61: NL2SQL v2 — Top-10 Competitive Upgrade

**Status:** Draft
**Priority:** High
**Effort:** 35-45 hours
**Dependencies:** PRD-21 (Database Knowledge), PRD-30 (Modular Architecture)
**Created:** 2026-02-18
**Research Base:** Deep analysis of 9 leading open-source NL2SQL projects + Automatos codebase audit

---

## Executive Summary

Deep research across the top NL2SQL/Text-to-SQL open-source projects (PandasAI 23.2K stars, Vanna 22.7K, DB-GPT 16K, WrenAI, Dataherald 3.6K, SQLCoder 4K, XiYan-SQL, DBHub, SQLBot 5.5K) reveals that **Automatos already has a surprisingly strong NL2SQL implementation** — more complete than most open-source alternatives. However, critical gaps remain in error correction, training data management, benchmark-quality generation, and MCP exposure that prevent it from being competitive with the best.

This PRD closes those gaps across 8 phases, taking Automatos from "good internal tool" to "competitive with Vanna/Dataherald" level.

---

## Part 1: How the Top Projects Work

### Tier 1: Full Platforms (Closest Competitors)

#### 1. Vanna (22.7K stars) — RAG-Based NL2SQL
- **Core Innovation:** Uses vector stores to store DDL schemas, documentation, and verified question/SQL pairs. Retrieves similar examples at generation time for few-shot prompting.
- **Pipeline:** `generate_sql()` → `run_sql()` → `generate_plotly_code()` → `get_plotly_figure()`
- **Training:** `vn.train(question, sql)` / `vn.add_ddl()` / `vn.add_documentation()` — builds a RAG corpus of schema + examples
- **Auto-train:** If query fails, automatically adds correction to training data
- **Follow-ups:** `generate_followup_questions()` suggests next questions
- **License:** MIT
- **Key Lesson:** RAG-based few-shot example retrieval dramatically improves SQL accuracy

#### 2. PandasAI (23.2K stars) — Code Generation + SQL
- **Core Innovation:** Generates Python code (not just SQL) for data analysis. Agent-based architecture routes between SQL queries and pandas operations.
- **Connectors:** MySQL, PostgreSQL, Snowflake, Databricks (production license required for Snowflake/Databricks)
- **License:** MIT + EE directory (enterprise features require commercial license)
- **Key Lesson:** Smart routing between SQL and code-based analysis; already partially adopted in Automatos via PandasAI integration

#### 3. DB-GPT (16K stars) — Full AI Data Platform
- **Core Innovation:** AWEL (Agentic Workflow Expression Language) DAG-based orchestration. Five-stage Text2SQL pipeline: Query Understanding → Schema Linking → SQL Generation → Execution → Visualization
- **Databases:** MySQL, PostgreSQL, Oracle, MSSQL, ClickHouse, DuckDB, Hive, Spark (8 databases)
- **Benchmark:** 82.5% Spider accuracy
- **Architecture:** Modular packages (dbgpt-core, dbgpt-ext, dbgpt-serve, dbgpt-client)
- **Key Lesson:** Modular package architecture; schema linking as explicit pipeline stage; fine-tuning framework for Text2SQL

#### 4. WrenAI — Semantic Layer Focus
- **Core Innovation:** Modeling Definition Language (MDL) for semantic layer. Strong emphasis on business metric definitions over raw SQL.
- **API:** SQL Generation + Chart Generation endpoints (commercial plan required)
- **Key Lesson:** Semantic layer as first-class citizen improves accuracy for business queries

### Tier 2: Specialized Tools

#### 5. Dataherald (3.6K stars) — Enterprise NL2SQL API
- **Core Innovation:** 4 SQL generator implementations (Langchain Agent, Langchain Chain, LlamaIndex, Dataherald Agent). Agent with 7 specialized tools. "Golden SQL" training system.
- **Agent Tools:** QuerySQLDatabase, GetCurrentTime, TablesSQLDatabase, SchemaSQLDatabase, InfoRelevantColumns, ColumnEntityChecker, GetFewShotExamples
- **Databases:** PostgreSQL, BigQuery, Databricks, Snowflake
- **Training:** Golden SQL pairs stored in vector DB + used for fine-tuning (min 20-30 samples/table)
- **Key Lesson:** Golden SQL concept; agent with specialized database tools; self-correction

#### 6. SQLCoder (4K stars) — Fine-Tuned Model
- **Core Innovation:** Purpose-built fine-tuned model for Text-to-SQL (based on StarCoder/Code Llama)
- **Key Lesson:** Fine-tuned models can beat GPT-4 on specific domains at lower cost

#### 7. XiYan-SQL — MCP-Native NL2SQL
- **Core Innovation:** State-of-the-art benchmark scores delivered via MCP server. Remote mode (API) or local mode.
- **Databases:** MySQL, PostgreSQL
- **Key Lesson:** MCP as delivery mechanism for NL2SQL; 2-22% better than raw database MCP servers

#### 8. DBHub — MCP Database Gateway
- **Core Innovation:** Zero-dependency MCP server for multi-database access. Custom parameterized SQL tools.
- **Databases:** PostgreSQL, MySQL, SQL Server, MariaDB, SQLite
- **Key Lesson:** MCP + custom tools pattern; read-only mode with row limits

#### 9. SQLBot (5.5K stars) — Conversational SQL
- **Core Innovation:** Multi-turn conversational interface for SQL
- **Key Lesson:** Conversation state management for iterative query refinement

### Benchmark Reality Check

| Method | Spider EX | Notes |
|--------|-----------|-------|
| MCS-SQL + GPT-4 | 89.6% | With Spider training data |
| CHASE-SQL + Gemini 1.5 | 87.6% | Without training data |
| DAIL-SQL + GPT-4 | 86.6% | With training data |
| DIN-SQL + GPT-4 | 85.3% | With training data |
| DB-GPT | 82.5% | Reported |
| Spider 2.0 (best) | 23.8% | Real-world enterprise complexity |
| Real enterprise datasets | ~24% | "NL2SQL is a solved problem... Not!" |

**Critical insight:** Academic benchmarks (85-91%) dramatically overstate real-world accuracy (~24% on enterprise schemas). Schema linking brittleness causes 10-20% accuracy drops on paraphrased queries. Models are only effective for ~20% of realistic user queries.

---

## Part 2: Automatos Current State (Honest Assessment)

### What Already Works (Strengths)

Automatos has a **substantial, production-ready NL2SQL system** across 16+ files in `orchestrator/modules/nl2sql/`:

| Feature | Status | Quality | Files |
|---------|--------|---------|-------|
| NL→SQL Generation | Working | Production | `query/nl2sql_service.py` (343 lines) |
| SQL Validation (3-tier) | Working | Excellent | `query/validator.py` (254 lines) |
| Schema Introspection | Working | Good | `schema/introspection.py` |
| Schema Provider + Cache | Working | Good | `schema/provider.py` (374 lines) |
| Smart Agent (multi-turn) | Working | Good | `intelligence/agent.py` (403 lines) |
| Query Clarification | Working | Good | `intelligence/clarifier.py` (261 lines) |
| Query Rephrasing | Working | Basic | `intelligence/rephraser.py` |
| Result Explanation | Working | Basic | `intelligence/explainer.py` |
| Visualization Suggestion | Working | Good | `intelligence/visualizer.py` (310 lines) |
| PandasAI Integration | Working | Good | `tools/services/pandas_ai_service.py` |
| Semantic Layer (metrics/dims) | Working | Good | `core/models/database_knowledge.py` |
| Database Tool Auto-creation | Working | Good | `tools/services/database_tool_integration.py` |
| Full REST API | Working | Complete | `api/database_knowledge.py` (612 lines) |
| Multi-tenant Isolation | Working | Excellent | Workspace-based throughout |
| Audit Trail | Working | Complete | `database_query_audit` table |
| Frontend UI | Working | Extensive | `DatabaseQueryExplorer.tsx` (NL input, SQL, results, viz), `SemanticLayerBuilder`, `QueryTemplatesGrid`, `DataWidget` (chatbot results with table/pagination/CSV export), `DataVisualizationModal`, `AddDatabaseModal`, plus 15 analytics components |
| Credential Management | Working | Secure | Encrypted, runtime-resolved |
| Query Templates (20+) | Working | Good | Per-dialect templates |
| Analytics Dashboard | Working | Basic | `api/database_analytics.py` |

**Total: ~3,500+ lines of working NL2SQL code**

### What's Missing (Gaps vs Top Projects)

| Gap | Impact | Who Has It | Priority |
|-----|--------|-----------|----------|
| **1. RAG-Based Few-Shot Examples** | Accuracy drops 15-25% without examples | Vanna, Dataherald | Critical |
| **2. SQL Error Self-Correction** | First-try failures become dead ends | Vanna (auto-train), DB-GPT, Dataherald | Critical |
| **3. Golden SQL Training System** | No way to improve over time | Vanna, Dataherald | High |
| **4. Schema Linking Stage** | Sending full schema wastes tokens, reduces accuracy | DB-GPT, CHASE-SQL, DAIL-SQL | High |
| **5. Multi-Database Connectors** | Only PostgreSQL + MySQL | DB-GPT (8), Dataherald (4) | Medium |
| **6. MCP Server Exposure** | NL2SQL not available via MCP | XiYan-SQL, DBHub | Medium |
| **7. Query Confidence Scoring** | Users can't judge result reliability | Dataherald | Medium |
| **8. SQL Generation Benchmarking** | No way to measure/track accuracy | All serious projects | Medium |

### 5 Bugs Found

| # | Bug | Severity | File | Fix |
|---|-----|----------|------|-----|
| 1 | **No table relevance scoring** — sends ALL tables to LLM | High | `query/nl2sql_service.py:_get_relevant_tables()` | Uses keyword matching only; needs embedding-based similarity |
| 2 | **Clarifier patterns too aggressive** — triggers on simple queries | Medium | `intelligence/clarifier.py` | Time-range pattern matches "last" in "last_name"; grouping matches "by" in "baby" |
| 3 | **Validator allows subqueries with mutations** | Medium | `query/validator.py` | Nested `INSERT INTO ... SELECT` bypasses single-statement check |
| 4 | **Schema cache never invalidates on DDL changes** | Medium | `schema/provider.py` | `_schema_cache` uses TTL but no hook for ALTER TABLE / new columns |
| 5 | **Visualizer chart config missing axis labels** | Low | `intelligence/visualizer.py` | `_build_chart_config()` returns x/y fields but no labels/titles for charts |

---

## Part 3: Build vs. Adopt Analysis

### The Question
> "Is it worth dropping the in-house version and adopting a really good open-source project?"

### Verdict: **BUILD ON EXISTING (Enhance, Don't Replace)**

#### Why NOT to adopt Vanna/Dataherald/DB-GPT:

| Concern | Detail |
|---------|--------|
| **Multi-tenant isolation** | None of the 9 projects support workspace-based multi-tenancy. You'd have to retrofit it — harder than building from scratch. |
| **Credential management** | Automatos uses encrypted, runtime-resolved credentials via CredentialStore. Open-source projects use connection strings in config files. |
| **Agent tool integration** | PRD-17 auto-creates 3 database tools per source. No open-source project has this. |
| **Audit trail** | Your `database_query_audit` table with user/agent/session tracking is more complete than any open-source project. |
| **Architecture fit** | FastAPI + SQLAlchemy + workspace isolation. Vanna is a standalone library, DB-GPT is a full platform, Dataherald is Docker-only. None fit cleanly into your modular architecture. |
| **License risk** | PandasAI EE requires commercial license. WrenAI API is paid. DB-GPT is AGPL-friendly but massive dependency. |
| **Migration cost** | Estimated 80-120 hours to rip out existing + integrate + retrofit multi-tenancy + restore feature parity. |

#### What TO adopt (techniques, not codebases):

| Technique | Source | Effort | Impact |
|-----------|--------|--------|--------|
| RAG-based few-shot retrieval | Vanna | 8h | +15-25% accuracy |
| Golden SQL training pairs | Dataherald | 4h | Continuous improvement |
| SQL error self-correction loop | DB-GPT, Vanna | 6h | Reduces failure rate 30-50% |
| Explicit schema linking stage | CHASE-SQL, DB-GPT | 4h | Better token efficiency + accuracy |
| MCP tool exposure | XiYan-SQL, DBHub | 3h | New integration channel |

**Bottom line:** Your existing implementation is worth ~3,500 lines of working, multi-tenant, production-ready code. Adopting would cost more than enhancing. Instead, adopt the **techniques** (RAG for SQL, Golden SQL, self-correction) that the top projects use.

### Same Verdict for RAG (from PRD-60)
The RAG system has a similar story — substantial existing implementation (semantic chunker, hybrid search, ingestion pipeline). Adopt techniques (parent-child chunks, RRF, knowledge graph) not codebases.

---

## Existing Frontend Reality

The NL2SQL frontend is **already extensive** — unlike RAG (where the DocumentWidget broke), the database UI is largely intact:

| Component | File | What It Does |
|-----------|------|-------------|
| `DatabaseQueryExplorer.tsx` | `frontend/components/knowledge/` | NL query input, generated SQL display, query results table, visualization toggle, query history, source selection |
| `SemanticLayerBuilder` | `frontend/components/knowledge/` | Define metrics, dimensions, business terms |
| `QueryTemplatesGrid` | `frontend/components/knowledge/` | 20+ query templates per dialect |
| `DataWidget/index.tsx` | `frontend/components/widgets/` | Chatbot widget — table with pagination, search, CSV export, sort, chart view |
| `DataVisualizationModal` | `frontend/components/knowledge/` | Full chart rendering |
| `SimpleDataVisualization` | `frontend/components/knowledge/` | Inline chart in query explorer |
| `AddDatabaseModal` | `frontend/components/knowledge/` | Database connection setup |
| `document-management.tsx` | `frontend/components/documents/` | Main orchestrator with tabs for all of the above |

**Frontend work in this PRD is minimal** — only 2 new things:
1. `TrainingExamplesManager.tsx` (Phase 5) — genuinely new, adds as tab in existing structure
2. Confidence badge in `DatabaseQueryExplorer.tsx` (Phase 6) — small addition to existing component
3. Benchmark charts in `DatabaseQueryAnalytics.tsx` (Phase 8) — small addition to existing component

---

## Part 4: Implementation Plan

### Phase 1: RAG-Based Few-Shot Example Retrieval (8h) — CRITICAL

**What:** Store verified question/SQL pairs in vector store. At query time, retrieve the most similar examples and include them in the LLM prompt (like Vanna's core approach).

**Why:** This is the single highest-impact improvement. Vanna's entire value proposition is built on this pattern. Few-shot examples improve accuracy by 15-25% on real-world queries.

#### Backend Changes

**New file: `orchestrator/modules/nl2sql/training/example_store.py`**

```python
"""
SQL Example Store — Vanna-inspired RAG for NL2SQL

Stores verified question/SQL pairs in vector store for
few-shot retrieval at generation time.
"""

class SQLExampleStore:
    """Manages training examples (question/SQL pairs) for few-shot retrieval."""

    def __init__(self, embedding_manager=None):
        self.embedding_manager = embedding_manager
        self._collection_name = "nl2sql_examples"

    async def add_example(
        self,
        question: str,
        sql: str,
        database_source_id: str,
        tables_used: List[str],
        is_verified: bool = False,
        metadata: Dict = None
    ) -> str:
        """Store a question/SQL pair with embedding for retrieval."""
        ...

    async def get_similar_examples(
        self,
        question: str,
        database_source_id: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """Retrieve most similar verified examples for few-shot prompting."""
        ...

    async def mark_verified(self, example_id: str) -> None:
        """Mark an example as verified (Golden SQL)."""
        ...

    async def add_ddl(self, ddl: str, database_source_id: str) -> str:
        """Store DDL for schema context retrieval (like Vanna's add_ddl)."""
        ...

    async def add_documentation(self, doc: str, database_source_id: str) -> str:
        """Store business context documentation (like Vanna's add_documentation)."""
        ...
```

**Modify: `orchestrator/modules/nl2sql/query/nl2sql_service.py`**

Add few-shot examples to the SQL generation prompt:

```python
# In _build_sql_prompt(), after schema context:
similar_examples = await self.example_store.get_similar_examples(
    question=question,
    database_source_id=source_id,
    limit=5
)

if similar_examples:
    prompt += "\n\n--- SIMILAR VERIFIED QUERIES (use as reference) ---\n"
    for ex in similar_examples:
        prompt += f"Question: {ex['question']}\nSQL: {ex['sql']}\n\n"
```

#### Database Migration

```sql
-- New table for SQL training examples
CREATE TABLE nl2sql_training_examples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    database_source_id UUID NOT NULL REFERENCES database_knowledge_sources(id),
    question TEXT NOT NULL,
    sql TEXT NOT NULL,
    tables_used TEXT[] DEFAULT '{}',
    is_verified BOOLEAN DEFAULT FALSE,
    verification_source VARCHAR(50), -- 'user', 'auto', 'import'
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    embedding_id VARCHAR(255), -- reference to vector store
    metadata JSONB DEFAULT '{}',
    created_by UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_nl2sql_examples_workspace ON nl2sql_training_examples(workspace_id);
CREATE INDEX idx_nl2sql_examples_source ON nl2sql_training_examples(database_source_id);
CREATE INDEX idx_nl2sql_examples_verified ON nl2sql_training_examples(is_verified) WHERE is_verified = TRUE;
```

#### API Endpoints

```
POST /api/knowledge/sources/database/{id}/examples      — Add training example
GET  /api/knowledge/sources/database/{id}/examples      — List examples
PUT  /api/knowledge/sources/database/{id}/examples/{eid} — Update/verify example
DELETE /api/knowledge/sources/database/{id}/examples/{eid} — Delete example
POST /api/knowledge/sources/database/{id}/examples/import — Bulk import from CSV/JSON
```

**Files to modify:**
- `orchestrator/modules/nl2sql/query/nl2sql_service.py` — inject examples into prompt
- `orchestrator/api/database_knowledge.py` — add example management endpoints
- `orchestrator/core/models/database_knowledge.py` — add SQLAlchemy model

---

### Phase 2: SQL Error Self-Correction Loop (6h) — CRITICAL

**What:** When generated SQL fails execution, automatically retry with error context. If retry succeeds, optionally auto-save as training example (like Vanna's `auto_train`).

**Why:** Currently, a failed SQL query is a dead end. DB-GPT, Vanna, and Dataherald all implement self-correction. This alone can reduce failure rates by 30-50%.

#### Backend Changes

**Modify: `orchestrator/modules/nl2sql/service.py` — `query_database()` method**

```python
async def query_database_with_correction(
    self,
    question: str,
    source_id: str,
    max_retries: int = 2,
    auto_train: bool = True
) -> Dict:
    """Execute NL query with automatic error correction."""

    last_error = None
    attempted_sqls = []

    for attempt in range(max_retries + 1):
        # Generate SQL (include previous errors as context)
        sql = await self.nl2sql_service.generate_sql(
            question=question,
            schema_metadata=schema,
            error_context=last_error,
            previous_attempts=attempted_sqls
        )

        # Validate
        validation = self.validator.validate_and_rewrite(sql, schema)
        if not validation.is_valid:
            last_error = f"Validation failed: {'; '.join(validation.reasons)}"
            attempted_sqls.append({"sql": sql, "error": last_error})
            continue

        # Execute
        try:
            result = await self._execute_sql(validation.rewritten_sql, source)

            # Success! Optionally auto-train
            if auto_train and attempt == 0:
                await self.example_store.add_example(
                    question=question,
                    sql=validation.rewritten_sql,
                    database_source_id=source_id,
                    is_verified=False,  # auto-generated, not manually verified
                    verification_source='auto'
                )

            return {
                "success": True,
                "sql": validation.rewritten_sql,
                "results": result,
                "attempts": attempt + 1,
                "corrections": attempted_sqls
            }

        except Exception as e:
            last_error = f"Execution error: {str(e)}"
            attempted_sqls.append({"sql": validation.rewritten_sql, "error": last_error})

    return {
        "success": False,
        "error": last_error,
        "attempts": max_retries + 1,
        "corrections": attempted_sqls
    }
```

**Modify: `orchestrator/modules/nl2sql/query/nl2sql_service.py`**

Add error context to prompt building:

```python
def _build_sql_prompt(self, question, schema, error_context=None, previous_attempts=None):
    prompt = self._build_base_prompt(question, schema)

    if error_context and previous_attempts:
        prompt += "\n\n--- PREVIOUS ATTEMPTS (fix these errors) ---\n"
        for attempt in previous_attempts:
            prompt += f"Attempted SQL: {attempt['sql']}\n"
            prompt += f"Error: {attempt['error']}\n\n"
        prompt += "Generate a CORRECTED SQL query that avoids these errors.\n"

    return prompt
```

**Files to modify:**
- `orchestrator/modules/nl2sql/service.py` — add correction loop
- `orchestrator/modules/nl2sql/query/nl2sql_service.py` — add error context to prompts
- `orchestrator/api/database_knowledge.py` — add `auto_correct` parameter to query endpoint

---

### Phase 3: Explicit Schema Linking Stage (4h) — HIGH

**What:** Before SQL generation, run a dedicated schema linking step that identifies the most relevant tables/columns for the query. Send only relevant schema to the LLM instead of the full schema.

**Why:** Current implementation uses keyword matching in `_get_relevant_tables()` which is brittle. DB-GPT and CHASE-SQL use explicit schema linking as a separate pipeline stage. This improves both accuracy (less noise) and token efficiency (smaller prompts).

#### Backend Changes

**New file: `orchestrator/modules/nl2sql/query/schema_linker.py`**

```python
"""
Schema Linker — Maps natural language entities to database schema elements.

Uses a combination of:
1. Embedding similarity (question vs table/column descriptions)
2. Entity extraction (detect table/column name mentions)
3. Semantic layer matching (map business terms to metrics/dimensions)
"""

class SchemaLinker:
    """Links natural language queries to relevant schema elements."""

    async def link(
        self,
        question: str,
        schema_metadata: Dict,
        semantic_layer: Dict = None,
        top_k_tables: int = 5,
        top_k_columns: int = 20
    ) -> LinkedSchema:
        """
        Returns:
            LinkedSchema with ranked tables, columns, and relationships
            relevant to the question.
        """
        # Step 1: Semantic layer matching (highest priority)
        metric_matches = self._match_metrics(question, semantic_layer)

        # Step 2: Embedding-based table ranking
        table_scores = await self._score_tables(question, schema_metadata)

        # Step 3: Column-level matching
        column_scores = await self._score_columns(question, top_tables)

        # Step 4: Relationship expansion (add JOIN tables)
        expanded = self._expand_relationships(top_tables, schema_metadata)

        return LinkedSchema(
            tables=expanded,
            columns=top_columns,
            relationships=relevant_relationships,
            metrics=metric_matches,
            confidence=overall_confidence
        )
```

**Modify: `orchestrator/modules/nl2sql/query/nl2sql_service.py`**

Replace `_get_relevant_tables()` with schema linker:

```python
# Before:
relevant_tables = self._get_relevant_tables(question, schema_metadata)

# After:
linked = await self.schema_linker.link(
    question=question,
    schema_metadata=schema_metadata,
    semantic_layer=semantic_layer
)
# Only send linked.tables and linked.columns to LLM prompt
```

**Files to modify:**
- `orchestrator/modules/nl2sql/query/nl2sql_service.py` — replace `_get_relevant_tables()`
- `orchestrator/modules/nl2sql/__init__.py` — export SchemaLinker

---

### Phase 4: Bug Fixes (3h) — HIGH

Fix the 5 bugs identified during the code audit.

#### Bug 1: Table Relevance Scoring (in `nl2sql_service.py`)

**Current:** Keyword-only matching in `_get_relevant_tables()` — sends all tables if no keywords match.

**Fix:** Use embedding similarity + keyword matching. This is partially addressed by Phase 3 (Schema Linker), but as an immediate fix:

```python
def _get_relevant_tables(self, question: str, schema_metadata: Dict) -> List[str]:
    """Score tables by relevance using keywords + column names + descriptions."""
    tables_with_scores = []
    question_lower = question.lower()
    question_words = set(question_lower.split())

    for table in schema_metadata.get('tables', []):
        score = 0
        table_name = table['name'].lower()

        # Direct name mention
        if table_name in question_lower or table_name.replace('_', ' ') in question_lower:
            score += 10

        # Column name mentions
        for col in table.get('columns', []):
            col_name = col['name'].lower()
            if col_name in question_lower or col_name.replace('_', ' ') in question_lower:
                score += 5

        # Description keyword overlap
        desc = (table.get('description', '') + ' ' + table.get('comment', '')).lower()
        desc_words = set(desc.split())
        overlap = question_words & desc_words - {'the', 'a', 'an', 'is', 'are', 'in', 'on', 'for'}
        score += len(overlap) * 2

        if score > 0:
            tables_with_scores.append((table['name'], score))

    # Sort by score, take top 10
    tables_with_scores.sort(key=lambda x: x[1], reverse=True)
    relevant = [t[0] for t in tables_with_scores[:10]]

    # If no matches, return top 5 tables by row count (most likely to be important)
    if not relevant:
        all_tables = schema_metadata.get('tables', [])
        all_tables.sort(key=lambda t: t.get('row_count', 0), reverse=True)
        relevant = [t['name'] for t in all_tables[:5]]

    return relevant
```

#### Bug 2: Clarifier False Positives (in `clarifier.py`)

**Current:** Regex patterns match inside words — "last_name" triggers time_range, "baby" triggers grouping.

**Fix:** Use word boundary matching:

```python
# Change patterns to use word boundaries
self._ambiguity_patterns = {
    'time_range': [
        r'\b(last|recent|past|previous)\b(?!_)',  # negative lookahead for underscore
        r'\b(this|current)\b\s+(week|month|quarter|year)',
        r'\b(between|from|since|until)\b',
    ],
    'grouping': [
        r'\bgroup\s+by\b',
        r'\b(by|per|for each)\b\s+\w+',  # "by" must be followed by a word
        r'\b(breakdown|split|segment)\b',
    ],
    ...
}
```

**File:** `orchestrator/modules/nl2sql/intelligence/clarifier.py`

#### Bug 3: Subquery Mutation Bypass (in `validator.py`)

**Current:** Only checks top-level for INSERT/UPDATE/DELETE. Nested subqueries can contain mutations.

**Fix:** Parse recursively:

```python
def _check_mutations(self, sql: str) -> List[str]:
    """Check for mutation keywords anywhere in the SQL, including subqueries."""
    errors = []
    # Normalize: remove string literals to avoid false positives
    clean_sql = re.sub(r"'[^']*'", "''", sql)
    clean_sql = re.sub(r'"[^"]*"', '""', clean_sql)

    mutation_pattern = r'\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE)\b'
    matches = re.findall(mutation_pattern, clean_sql, re.IGNORECASE)

    if matches:
        errors.append(f"Mutation keywords found: {', '.join(matches)}")

    return errors
```

**File:** `orchestrator/modules/nl2sql/query/validator.py`

#### Bug 4: Schema Cache Invalidation (in `provider.py`)

**Fix:** Add cache invalidation method + hook on introspection:

```python
def invalidate_cache(self, database_source_id: str = None):
    """Invalidate schema cache, optionally for a specific source."""
    if database_source_id:
        keys_to_remove = [k for k in self._schema_cache if database_source_id in k]
        for k in keys_to_remove:
            del self._schema_cache[k]
    else:
        self._schema_cache.clear()
    logger.info(f"Schema cache invalidated for {database_source_id or 'all'}")
```

**File:** `orchestrator/modules/nl2sql/schema/provider.py`

#### Bug 5: Chart Config Missing Labels (in `visualizer.py`)

**Fix:** Add axis labels to chart config:

```python
def _build_chart_config(self, chart_type, data_columns, query_hints):
    config = {
        "type": chart_type.value,
        "x": x_col,
        "y": y_col,
        "xLabel": x_col.replace('_', ' ').title(),  # Add
        "yLabel": y_col.replace('_', ' ').title(),  # Add
        "title": self._generate_chart_title(chart_type, x_col, y_col),  # Add
        ...
    }
```

**File:** `orchestrator/modules/nl2sql/intelligence/visualizer.py`

---

### Phase 5: Golden SQL Training System (4h) — HIGH

**What:** A complete system for managing verified question/SQL pairs, inspired by Dataherald's "Golden SQL" concept. Users can verify auto-generated pairs, import external examples, and track which examples are most effective.

#### Frontend: Training Examples Manager

**New component: `frontend/components/knowledge/TrainingExamplesManager.tsx`**

**Integration note:** The existing `document-management.tsx` orchestrator already has 10+ tab panels (CodeGraphPanel, MultimodalKnowledgePanel, DatabaseQueryExplorer, QueryTemplatesGrid, SemanticLayerBuilder, etc.). This new component should be added as a new tab/panel within that existing structure, NOT as a standalone page.

The existing `DatabaseQueryExplorer.tsx` already has NL query input, generated SQL display, query results, visualization toggle, and query history. The TrainingExamplesManager should complement it — accessed via a "Training Examples" tab or button within the database section.

Features:
- List all training examples for a database source
- Filter by verified/unverified/auto-generated
- Inline edit SQL
- One-click verify button
- Import from CSV/JSON
- Usage statistics (how often each example is retrieved)
- Delete/archive examples

#### API Endpoints (in `api/database_knowledge.py`)

```python
@router.get("/{source_id}/examples")
async def list_training_examples(source_id: UUID, verified_only: bool = False):
    """List training examples for a database source."""

@router.post("/{source_id}/examples")
async def add_training_example(source_id: UUID, body: TrainingExampleCreate):
    """Add a new training example (question/SQL pair)."""

@router.put("/{source_id}/examples/{example_id}/verify")
async def verify_example(source_id: UUID, example_id: UUID):
    """Mark an example as verified (Golden SQL)."""

@router.post("/{source_id}/examples/import")
async def import_examples(source_id: UUID, file: UploadFile):
    """Bulk import examples from CSV or JSON."""

@router.get("/{source_id}/examples/stats")
async def get_example_stats(source_id: UUID):
    """Get training example statistics."""
```

**Files to create:**
- `frontend/components/knowledge/TrainingExamplesManager.tsx`

**Files to modify:**
- `orchestrator/api/database_knowledge.py` — add endpoints
- `orchestrator/core/models/database_knowledge.py` — add NL2SQLTrainingExample model

---

### Phase 6: Query Confidence Scoring (3h) — MEDIUM

**What:** Assign a confidence score to each generated SQL query before execution, so users know how reliable the result is likely to be.

**Why:** Dataherald implements this. Users need to know when to trust automated SQL and when to review manually.

#### Scoring Factors

```python
class QueryConfidenceScorer:
    """Score confidence of generated SQL queries."""

    def score(self, context: ScoringContext) -> ConfidenceScore:
        factors = {}

        # Factor 1: Similar examples found (0-30 points)
        similar_count = len(context.similar_examples)
        best_similarity = max(ex.similarity for ex in context.similar_examples) if similar_count else 0
        factors['example_match'] = min(30, similar_count * 5 + best_similarity * 20)

        # Factor 2: Schema linking confidence (0-25 points)
        linked_tables = len(context.linked_schema.tables)
        factors['schema_link'] = min(25, context.linked_schema.confidence * 25)

        # Factor 3: Query complexity (0-20 points, simpler = higher confidence)
        join_count = context.sql.lower().count('join')
        subquery_count = context.sql.lower().count('select') - 1
        complexity_penalty = (join_count * 3 + subquery_count * 5)
        factors['complexity'] = max(0, 20 - complexity_penalty)

        # Factor 4: Semantic layer coverage (0-15 points)
        metrics_used = len(context.metrics_matched)
        factors['semantic'] = min(15, metrics_used * 5)

        # Factor 5: Validation passed cleanly (0-10 points)
        factors['validation'] = 10 if context.validation_clean else 5

        total = sum(factors.values())

        return ConfidenceScore(
            score=total,
            level='high' if total >= 70 else 'medium' if total >= 40 else 'low',
            factors=factors,
            recommendation='auto_execute' if total >= 70 else 'review_sql' if total >= 40 else 'manual_review'
        )
```

**Files to create:**
- `orchestrator/modules/nl2sql/query/confidence.py`

**Files to modify:**
- `orchestrator/modules/nl2sql/service.py` — integrate scorer
- `orchestrator/api/database_knowledge.py` — return confidence in query response
- `frontend/components/knowledge/DatabaseQueryExplorer.tsx` — display confidence badge (small addition — component already has NL input, SQL display, results table, chart toggle, and query history)

---

### Phase 7: MCP Server Exposure (3h) — MEDIUM

**What:** Expose NL2SQL as an MCP tool so external agents (Claude Desktop, Cursor, etc.) can query databases through Automatos.

**Why:** XiYan-SQL and DBHub prove that MCP is the dominant integration pattern for database tools. Automatos already has MCP gateway infrastructure (PRD-33).

#### MCP Tool Definitions

```python
# In MCP tool registry
{
    "name": "query_database",
    "description": "Ask a natural language question about data in a connected database. Returns SQL, results, and optional visualization.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Natural language question about the data"},
            "database_source": {"type": "string", "description": "Name or ID of the database source"},
            "auto_execute": {"type": "boolean", "default": True},
            "include_chart": {"type": "boolean", "default": False}
        },
        "required": ["question"]
    }
},
{
    "name": "explore_database_schema",
    "description": "Explore the schema of a connected database. Returns tables, columns, relationships, and sample data.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "database_source": {"type": "string"},
            "table_name": {"type": "string", "description": "Optional: specific table to explore"}
        }
    }
},
{
    "name": "add_sql_training_example",
    "description": "Add a verified question/SQL pair to improve future query generation.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "sql": {"type": "string"},
            "database_source": {"type": "string"}
        },
        "required": ["question", "sql"]
    }
}
```

**Files to modify:**
- `orchestrator/modules/tools/services/database_tool_integration.py` — add MCP tool definitions
- MCP gateway configuration

---

### Phase 8: Benchmarking & Monitoring (4h) — MEDIUM

**What:** Implement a benchmarking system to measure and track NL2SQL accuracy over time.

**Why:** "What gets measured gets improved." Without benchmarks, you can't know if changes help or hurt.

#### Benchmark Test Suite

**New file: `orchestrator/modules/nl2sql/benchmarks/runner.py`**

```python
class NL2SQLBenchmarkRunner:
    """Run benchmarks against verified Golden SQL examples."""

    async def run_benchmark(
        self,
        database_source_id: str,
        test_examples: List[Dict] = None
    ) -> BenchmarkResult:
        """
        Test current system against verified examples.

        For each example:
        1. Generate SQL from question
        2. Compare generated SQL to verified SQL
        3. Execute both and compare results
        4. Score: exact match, execution match, partial match
        """
        if test_examples is None:
            # Use 20% of verified examples as test set
            test_examples = await self._get_test_set(database_source_id)

        results = []
        for example in test_examples:
            generated_sql = await self.nl2sql_service.generate_sql(
                question=example['question'],
                schema_metadata=schema
            )

            # Compare
            exact_match = self._normalize_sql(generated_sql) == self._normalize_sql(example['sql'])

            # Execute both and compare results
            if not exact_match:
                exec_match = await self._compare_execution(generated_sql, example['sql'])
            else:
                exec_match = True

            results.append({
                'question': example['question'],
                'expected_sql': example['sql'],
                'generated_sql': generated_sql,
                'exact_match': exact_match,
                'execution_match': exec_match
            })

        return BenchmarkResult(
            total=len(results),
            exact_match_rate=sum(1 for r in results if r['exact_match']) / len(results),
            execution_match_rate=sum(1 for r in results if r['execution_match']) / len(results),
            details=results,
            timestamp=datetime.now()
        )
```

#### API Endpoints

```
POST /api/knowledge/sources/database/{id}/benchmark/run  — Run benchmark
GET  /api/knowledge/sources/database/{id}/benchmark/history — View historical results
```

#### Monitoring Dashboard

Add to existing `DatabaseQueryAnalytics.tsx` (part of the 15 existing analytics components):
- Accuracy trend chart (benchmark scores over time)
- Error category breakdown (syntax, schema linking, wrong aggregation, etc.)
- Latency percentiles (p50, p95, p99)
- Most common failure patterns

**Note:** The chatbot `DataWidget` (`frontend/components/widgets/DataWidget/index.tsx`) already displays NL2SQL results with table, pagination (25 rows/page), search, CSV export, sort, and chart view toggle. No changes needed to the chatbot widget — enhancements are to the analytics/management UI only.

**Files to create:**
- `orchestrator/modules/nl2sql/benchmarks/runner.py`
- `orchestrator/modules/nl2sql/benchmarks/comparator.py`

**Files to modify:**
- `orchestrator/api/database_knowledge.py` — add benchmark endpoints
- `frontend/components/context/DatabaseQueryAnalytics.tsx` — add accuracy charts

---

## Priority Matrix

| Phase | Effort | Impact | Priority | Dependencies |
|-------|--------|--------|----------|-------------|
| 1. RAG Few-Shot Examples | 8h | Critical (+15-25% accuracy) | P0 | None |
| 2. Error Self-Correction | 6h | Critical (-30-50% failures) | P0 | None |
| 3. Schema Linking | 4h | High (token efficiency + accuracy) | P1 | None |
| 4. Bug Fixes | 3h | High (correctness) | P1 | None |
| 5. Golden SQL Training | 4h | High (continuous improvement) | P1 | Phase 1 |
| 6. Confidence Scoring | 3h | Medium (user trust) | P2 | Phase 3 |
| 7. MCP Exposure | 3h | Medium (integration channel) | P2 | None |
| 8. Benchmarking | 4h | Medium (measurability) | P2 | Phase 5 |

**MVP (Phases 1-4): 21 hours** — Gets you to competitive parity with Vanna/Dataherald
**Full (All phases): 35 hours** — Best-in-class for a multi-tenant NL2SQL platform

---

## Competitive Comparison (After Implementation)

| Feature | Automatos (Current) | Automatos (After PRD-61) | Vanna | Dataherald | DB-GPT |
|---------|---------------------|--------------------------|-------|------------|--------|
| NL→SQL Generation | Basic LLM | RAG-enhanced + few-shot | RAG-based | Agent-based | Schema-linked |
| Error Self-Correction | None | Auto-retry + context | Auto-train | Agent retry | LLM feedback |
| Training Data Mgmt | None | Golden SQL system | `train()` API | Golden SQL API | Fine-tuning hub |
| Schema Linking | Keyword-only | Embedding + semantic layer | DDL retrieval | Schema+column tools | Dedicated stage |
| SQL Validation | Excellent (3-tier) | Excellent (3-tier + subquery fix) | Basic | Basic | Basic |
| Multi-tenant | Full isolation | Full isolation | None | None | None |
| Audit Trail | Complete | Complete | None | Basic | None |
| Confidence Scoring | None | Multi-factor scoring | None | Partial | None |
| MCP Tools | None | 3 tools exposed | None | None | None |
| Benchmarking | None | Automated + tracking | None | None | Fine-tuning eval |
| Databases | 2 (PG, MySQL) | 2 (PG, MySQL) | 8+ | 4 | 8 |
| Agent Tools | Auto-created | Auto-created + MCP | None | 7 agent tools | Agent framework |
| Visualization | 10 chart types | 10 chart types + labels | Plotly | None | GPT-Vis |
| Semantic Layer | Metrics + dimensions | Metrics + dimensions | None | None | None |
| License | Proprietary | Proprietary | MIT | MIT | MIT |

---

## Files Summary

### New Files
| File | Phase | Purpose |
|------|-------|---------|
| `orchestrator/modules/nl2sql/training/example_store.py` | 1 | RAG-based example storage/retrieval |
| `orchestrator/modules/nl2sql/training/__init__.py` | 1 | Package init |
| `orchestrator/modules/nl2sql/query/schema_linker.py` | 3 | Explicit schema linking stage |
| `orchestrator/modules/nl2sql/query/confidence.py` | 6 | Query confidence scoring |
| `orchestrator/modules/nl2sql/benchmarks/runner.py` | 8 | Benchmark test runner |
| `orchestrator/modules/nl2sql/benchmarks/comparator.py` | 8 | SQL result comparison |
| `orchestrator/modules/nl2sql/benchmarks/__init__.py` | 8 | Package init |
| `frontend/components/knowledge/TrainingExamplesManager.tsx` | 5 | Training examples UI |

### Modified Files
| File | Phases | Changes |
|------|--------|---------|
| `orchestrator/modules/nl2sql/service.py` | 2, 6 | Add correction loop, confidence scoring |
| `orchestrator/modules/nl2sql/query/nl2sql_service.py` | 1, 2, 3, 4 | Few-shot injection, error context, schema linker, bug fix |
| `orchestrator/modules/nl2sql/query/validator.py` | 4 | Subquery mutation fix |
| `orchestrator/modules/nl2sql/intelligence/clarifier.py` | 4 | Word boundary fix |
| `orchestrator/modules/nl2sql/intelligence/visualizer.py` | 4 | Chart labels fix |
| `orchestrator/modules/nl2sql/schema/provider.py` | 4 | Cache invalidation |
| `orchestrator/modules/nl2sql/__init__.py` | 1, 3, 6 | Export new classes |
| `orchestrator/api/database_knowledge.py` | 1, 2, 5, 6, 8 | New endpoints |
| `orchestrator/core/models/database_knowledge.py` | 1, 5 | New models |
| `orchestrator/modules/tools/services/database_tool_integration.py` | 7 | MCP tool definitions |
| `frontend/components/knowledge/DatabaseQueryExplorer.tsx` | 6 | Confidence display |
| `frontend/components/context/DatabaseQueryAnalytics.tsx` | 8 | Benchmark charts |

### Database Migration
| Table | Phase | Purpose |
|-------|-------|---------|
| `nl2sql_training_examples` | 1 | Training example storage |
| `nl2sql_benchmark_runs` | 8 | Benchmark history |
| `nl2sql_benchmark_results` | 8 | Individual benchmark results |

---

## Success Criteria

- [ ] Phase 1: At least 5 verified examples improve SQL generation for similar queries
- [ ] Phase 2: Failed queries auto-retry with correction, success rate improves by 20%+
- [ ] Phase 3: Token usage per query drops by 40%+ (only relevant tables sent to LLM)
- [ ] Phase 4: All 5 bugs verified fixed
- [ ] Phase 5: Users can add/verify/import training examples via UI
- [ ] Phase 6: Every query response includes confidence score with factors
- [ ] Phase 7: External MCP clients can query databases through Automatos
- [ ] Phase 8: Benchmark runs complete and track accuracy trends

---

## Out of Scope (Future PRDs)

- Additional database connectors (Snowflake, BigQuery, Redshift, ClickHouse)
- Fine-tuned model training (like SQLCoder approach)
- Multi-database JOINs (cross-source queries)
- Natural language to NoSQL (MongoDB, etc.)
- Real-time schema change detection (webhooks from database)
- A/B testing different generation strategies
- Query plan optimization suggestions
- Row-level security integration

---

**Estimated Total Effort:** 35-45 hours
**MVP (Phases 1-4):** 21 hours
**Priority:** High
**Dependencies:** PRD-21 (completed), PRD-30 (completed)
