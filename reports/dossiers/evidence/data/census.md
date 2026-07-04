# Live Postgres census — Railway production DB

- Captured: 2026-07-04 (read-only session: `PGOPTIONS="-c default_transaction_read_only=on"`, `PGCONNECT_TIMEOUT=8`)
- Server: PostgreSQL 18.3 (Debian), database `railway`, host per `DATABASE_URL` in `automatos-ai/orchestrator/.env` (Railway public proxy; credentials elided)
- Public base tables: **152** (`information_schema.tables`, `table_schema='public' AND table_type='BASE TABLE'`)
- Note: 152 live vs ~109 cited in the July OS review — the live DB carries extras such as `alembic_version_backup_20260520` plus tables from waves W7–W14 (`approval_grants`, `tool_routing_*`, `unrouted_events`, `error_events`, ...).

## Row-count census (29 key tables, single UNION ALL query)

```sql
SELECT 'workspaces' t, count(*) n FROM workspaces UNION ALL ... ORDER BY t;  -- one SELECT, 29 subcounts
```

| table | rows | first-look |
|---|---:|---|
| workspaces | 21 | |
| users | 22 | |
| agents | 237 | |
| documents | 644 | |
| document_chunks | 19,130 | RAG corpus is real |
| chats | 332 | newest 2026-06-27 |
| messages | 1,278 | newest 2026-06-27 |
| memory_items | **0** | durable/L3 memory table EMPTY |
| memory_short_term | 2,211 | see memory-short-term.md |
| tool_execution_logs | 2,341 | 100% synthetic — see tool-telemetry.md |
| tool_routing_edges | 29 | all synthetic workspace — see operating-graph-edges.md |
| tool_routing_affinities | 232 | same |
| tool_routing_intent_clusters | 20 | same |
| rag_feedback | **0** | never one row |
| board_tasks | 1,449 | active daily — see board-tasks.md |
| database_query_audit | **0** | NL2SQL audit never fired |
| orchestration_runs | 17 | last created 2026-06-13 |
| orchestration_tasks | 122 | |
| notifications | 5,423 | see notifications.md |
| deliverables | 2,242 | newest 2026-06-16 |
| workspace_graphs | 155 | fresh 2026-07-01 — see workspace-graphs.md |
| knowledge_nodes | **0** | KG persistence empty |
| knowledge_edges | **0** | KG persistence empty |
| agent_reports | 3,845 | newest 2026-07-03 |
| heartbeat_results | 148,362 | newest 2026-07-03; by far the largest table sampled |
| llm_usage | 31,081 | newest 2026-07-03 |
| unrouted_events | 135 | |
| approval_grants | **0** | W11 (PRD-181) table exists, unused yet |
| error_events | 12 | W10 table live, small |

## Auxiliary counts (second UNION query)

| table | rows |
|---|---:|
| nl2sql_benchmark_runs | 0 |
| nl2sql_benchmark_results | 0 |
| nl2sql_training_examples | 2 |
| routing_decisions | 542 (newest 2026-06-10) |
| intent_classification_cache | 0 |
| recipe_executions | 735 (newest 2026-07-03) |
| workflow_executions | 21 |
| task_executions | 9 |
| harness_prescriptions | 0 |
| skill_audit_log | 6 |
| codegraph_projects | 1 |
| codegraph_symbols | 7,702 |

## Freshness (max timestamp per activity table)

| table | newest row |
|---|---|
| heartbeat_results | 2026-07-03 |
| llm_usage | 2026-07-03 |
| agent_reports | 2026-07-03 |
| recipe_executions | 2026-07-03 |
| chats / messages | 2026-06-27 |
| deliverables | **2026-06-16** — client-facing output stopped ~2.5 weeks ago |
| routing_decisions | 2026-06-10 |
| orchestration_runs (created) | 2026-06-13 |

## First look (3–5 lines)

The platform's *heartbeat* plane is alive and current (heartbeats, reports, LLM usage, playbook cron), but its *learning* plane is essentially empty: memory_items, rag_feedback, knowledge_nodes/edges, database_query_audit, intent_classification_cache and approval_grants all have 0 rows, and everything the operating graph "learned" comes from one synthetic seed batch. Deliverables — the client-visible output — stopped on 2026-06-16, which lines up with playbook LLM calls failing on OpenRouter 402 (see memory-short-term.md) while board tasks kept getting marked done.
