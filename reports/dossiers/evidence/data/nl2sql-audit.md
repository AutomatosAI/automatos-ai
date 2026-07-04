# NL2SQL — audit log, benchmarks, training examples

Captured 2026-07-04, read-only.

## Query

```sql
-- census + aux UNION queries
SELECT 'database_query_audit', count(*) FROM database_query_audit;        -- 0
SELECT 'nl2sql_benchmark_runs', count(*) FROM nl2sql_benchmark_runs;      -- 0
SELECT 'nl2sql_benchmark_results', count(*) FROM nl2sql_benchmark_results;-- 0
SELECT 'nl2sql_training_examples', count(*) FROM nl2sql_training_examples;-- 2
```

| table | rows |
|---|---:|
| database_query_audit (PRD-160 audit choke-point) | **0** |
| nl2sql_benchmark_runs | 0 |
| nl2sql_benchmark_results | 0 |
| nl2sql_training_examples | 2 |

No generated-SQL log rows exist to sample (the requested 15-row sample is therefore empty by fact, not by omission).

## In-repo counterpart (pinned tree)

- `orchestrator/tests/nl2sql_eval/questions.json` — 20 questions; `baseline.json` records `accuracy: 0.0` with note: *"0.0 until a generation run with a real LLM records a number; the gate is no-regression-vs-baseline"* (recorded 2026-06-12).
- `orchestrator/tests/nl2sql_eval/seed_schema.sql` — SQLite eval schema.

## First look

NL2SQL's accuracy stack (PRD-160) shipped its scaffolding — audit table, benchmark tables, example store with JSONB embeddings — but production has never exercised it: zero audited queries, zero benchmark runs, two training examples, and the repo baseline still holds the placeholder 0.0 accuracy from 2026-06-12. Either the NL2SQL feature is unused by real workspaces or its call path doesn't reach the audit choke-point; both are worth a dossier check.
