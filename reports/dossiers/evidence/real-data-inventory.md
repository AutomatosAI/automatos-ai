# Real-Data Inventory — Automatos AI deep review (Phase 2 recon)

Captured 2026-07-04 by the real-data reconnaissance agent. All live-store access was STRICTLY READ-ONLY (`default_transaction_read_only=on`, GET-only HTTP, ≤50-row LIMITs, 8s timeouts). Credentials referenced by variable name only.

## Reachability

| store | reachable | how / why not |
|---|---|---|
| Postgres (Railway prod) | **YES** | `DATABASE_URL` in `automatos-ai/orchestrator/.env` → Railway public proxy; PostgreSQL 18.3, db `railway`, **152 public base tables**; 24 read-only SELECT invocations used |
| mem0 server | **NO** | `MEM0_API_URL` host answers with Railway edge `404 "Application not found"` on every path incl. spec-valid `GET /api/v1/stats/` — the deployment behind that hostname no longer exists → [data/mem0.md](data/mem0.md) |
| Qdrant (field memory) | **NO** | no `QDRANT_*` var in either env file; code default `http://localhost:6333` (`orchestrator/config.py:859`) refused; only committed value is compose-internal `http://qdrant:6333` (`envs/api.defaults`) → [data/qdrant.md](data/qdrant.md) |
| Redis | not attempted | out of scope for data sampling (no redis-cli guarantee; queue state, not stored knowledge) |

Non-inspectable ≠ missing: mem0 + Qdrant contents (the durable-memory and field-memory halves of the memory stack) **could not be audited from this machine** — that is itself a review finding. Verifying them needs Railway-internal access.

## What exists where

### Live Postgres (fresh, production)
Full census + freshness: [data/census.md](data/census.md). Headline row counts:

| surface | rows | freshness | sample file |
|---|---:|---|---|
| workspaces / users / agents | 21 / 22 / 237 | — | census.md |
| documents / document_chunks | 644 / 19,130 | — | census.md |
| memory_short_term | 2,211 | 2026-07-03 | [data/memory-short-term.md](data/memory-short-term.md) |
| memory_items (L3 durable) | **0** | never | memory-short-term.md |
| memory_access_log | 6 | 2026-03-11 | memory-short-term.md |
| tool_execution_logs | 2,341 (**100% synthetic**) | frozen 2026-05-05 | [data/tool-telemetry.md](data/tool-telemetry.md) |
| tool_routing_edges / affinities / intent_clusters | 29 / 232 / 20 (all synthetic ws) | frozen 2026-05-05 | [data/operating-graph-edges.md](data/operating-graph-edges.md) |
| rag_feedback, knowledge_nodes, knowledge_edges, database_query_audit, intent_classification_cache, harness_prescriptions, approval_grants | **0 each** | never | [data/rag-feedback.md](data/rag-feedback.md), [data/nl2sql-audit.md](data/nl2sql-audit.md) |
| board_tasks | 1,449 | 2026-07-03 (daily) | [data/board-tasks.md](data/board-tasks.md) |
| orchestration_runs / _tasks (Missions) | 17 / 122 | last run created 2026-06-13 | [data/missions-orchestration.md](data/missions-orchestration.md) |
| notifications | 5,423 | 2026-07-03 | [data/notifications.md](data/notifications.md) |
| deliverables | 2,242 | **stopped 2026-06-16** | [data/deliverables.md](data/deliverables.md) |
| workspace_graphs (graph blobs) | 155 | rebuilt 2026-07-01 (42 MB max blob) | [data/workspace-graphs.md](data/workspace-graphs.md) |
| heartbeat_results / llm_usage / agent_reports | 148,362 / 31,081 / 3,845 | 2026-07-03 | census.md |
| recipe_executions / routing_decisions | 735 / 542 | 07-03 / 06-10 | census.md |
| nl2sql_* (runs/results/examples) | 0 / 0 / 2 | — | nl2sql-audit.md |
| codegraph_projects / codegraph_symbols | 1 / 7,702 | — | census.md |

### In-repo (pinned tree @ 77bc9c6d5)
Details: [data/repo-eval-artifacts.md](data/repo-eval-artifacts.md)

- **Eval sets**: tool-routing 47-query `eval_set.jsonl` + seed corpus/harness (`orchestrator/scripts/eval/tool_routing/`); W7 uplift eval `orchestrator/evals/operating_graph_uplift.py` (+ CI test); NL2SQL 20-question set with **0.0 placeholder baseline** (`orchestrator/tests/nl2sql_eval/baseline.json`, 2026-06-12); Shopify opener golden fixtures (`orchestrator/integrations/shopify/tests/fixtures/`).
- **Seeds**: `orchestrator/core/seeds/` — 10 seeders + Auto soul + 44 KB platform-management skill.
- **Recorded telemetry/eval outputs**: none committed. Gitignored local results exist ONLY on the primary checkout: `orchestrator/scripts/eval/tool_routing/results/{summary.csv,report.md,results.jsonl}` (2026-05-05/08; 22 model×mode pairs — gpt-4.1-mini filtered_schema 93.6% @ ~$0.0007/call; `graph (no-edges)` mode 83.0%, below baseline).
- **Migrations**: 137 files, **single head `e773c09189a9`** (post-W6/W11 merge chain) — heads hygiene held.
- **Pre-drop schema snapshots**: `graphify-out/snapshots/bucket-{1..6}-pre-drop.sql` (2026-04-25 Railway cleanup).

### Structure graph (STALE)
`graph.json` is **not in the pinned tree**; stats from primary checkout copy (34.5 MB, built 2026-06-09, predates all 14 waves): 19,996 nodes / 63,575 links; calls 23,444, uses 21,092, contains 7,787, rationale_for 6,614, method 3,595, inherits 686, imports_from 356. Shape-only; verify against live code.

## Cross-surface first-look (the story the data tells)

1. **The learning plane is unfed.** Every table that is supposed to accumulate real usage signal is either empty (rag_feedback, memory_items, knowledge_nodes/edges, database_query_audit, approval_grants) or contains only the 2026-05-05 synthetic seed batch (tool_execution_logs and everything the operating graph learned from it). The 21 real workspaces have zero learned edges. W7's per-tenant machinery has real scaffolding and no real diet.
2. **Memory quality is confirmably LOW.** ~87% of stored memories are duplicated operational chatter (playbook-failure spam + heartbeat summaries stored twice each); zero L3 promotions ever; recall last logged 2026-03-11; the only informative memories are raw chat clippings ending 2026-06-07 — several of which are the user telling Auto it is lying.
3. **A silent production outage is in the data.** Playbook LLM steps have failed with OpenRouter 402 (insufficient credits) daily since ~mid-June; deliverables and playbook_complete notifications both stop 2026-06-16; yet board_tasks closes those runs `done` every day (194 done-with-error, 484 done-without-result). Failed-marked-done is real and current.
4. **Missions stall at the approval gate.** 8/17 runs `awaiting_approval` (some since April) with 0 tokens; a single `action_required` notification; only one workspace ever ran Missions; none since 2026-06-13.
5. **What IS healthy:** heartbeat/report/llm_usage planes are current (07-03); the RAG corpus is real (19k chunks); per-workspace graph blobs rebuilt 2026-07-01 (though a 42 MB JSON in a TEXT column is a storage smell); Mission failure semantics look honest when runs actually execute.

## Sample-file index

| file | surface |
|---|---|
| [data/census.md](data/census.md) | table census + freshness |
| [data/memory-short-term.md](data/memory-short-term.md) | durable/distilled memory + access log |
| [data/tool-telemetry.md](data/tool-telemetry.md) | tool_execution_logs |
| [data/operating-graph-edges.md](data/operating-graph-edges.md) | learned edges/affinities/intent clusters |
| [data/rag-feedback.md](data/rag-feedback.md) | rag_feedback + empty learning tables |
| [data/board-tasks.md](data/board-tasks.md) | Board + failed-marked-done probe |
| [data/nl2sql-audit.md](data/nl2sql-audit.md) | NL2SQL audit/benchmarks/examples |
| [data/missions-orchestration.md](data/missions-orchestration.md) | Missions runs/tasks |
| [data/notifications.md](data/notifications.md) | notifications distribution |
| [data/workspace-graphs.md](data/workspace-graphs.md) | per-workspace graph blobs |
| [data/deliverables.md](data/deliverables.md) | client-facing outputs |
| [data/qdrant.md](data/qdrant.md) | Qdrant non-inspectability |
| [data/mem0.md](data/mem0.md) | mem0 unreachability |
| [data/repo-eval-artifacts.md](data/repo-eval-artifacts.md) | in-repo eval sets/seeds/migrations/graph stats |

PII policy applied throughout: emails → `x***@domain`, personal names → initials, no tokens/keys/credential values anywhere.
