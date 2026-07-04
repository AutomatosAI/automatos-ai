# In-repo data: eval sets, seeds, fixtures, recorded eval outputs, migrations, structure graph

Pinned tree: `p2-src` = origin/main @ 77bc9c6d5 (all relative paths below are in it). Captured 2026-07-04.

## Eval sets

| asset | where | size/shape |
|---|---|---|
| Tool-routing eval set | `orchestrator/scripts/eval/tool_routing/eval_set.jsonl` | **47 queries** (q000–q046), fields: query, correct_actions, category, difficulty; categories incl. agents/analytics/cross/memory/missions/playbooks/workspace_code |
| Tool-routing seed corpus | `orchestrator/scripts/eval/tool_routing/eval_seed.yaml` (15.6 KB) + `seed_telemetry.py` (17.6 KB) | generates the synthetic telemetry month found in the live DB (all-zeros workspace, 2026-04-05→05-05) |
| Eval harness | `run_eval.py`, `score.py`, `prompt_builder.py`, `models.yaml`, `snapshot.py`, `_registry_bootstrap.py` in same dir | model matrix incl. gpt-4.1-mini / claude-* / gemini / llama |
| Operating-graph uplift eval (W7) | `orchestrator/evals/operating_graph_uplift.py` (17.2 KB) | offline, pure BM25 vs embedding-proxy vs learned-edge top-1 accuracy on TRAIN/TEST split; docstring: honest sub-threshold outcome must not flip `TOOL_ROUTING_GRAPH` |
| Uplift eval test | `orchestrator/tests/test_prd177_uplift_eval.py` | CI-side |
| NL2SQL eval | `orchestrator/tests/nl2sql_eval/` — `questions.json` (20 q), `baseline.json` (accuracy **0.0** placeholder, 2026-06-12), `seed_schema.sql`, `harness.py` | gate = no-regression-vs-baseline; never bumped by a real run |
| Shopify opener fixtures | `orchestrator/integrations/shopify/tests/fixtures/` — cart_idle_context.json, product_page_context.json, expected_*_opener.txt, inbuild_graph_snapshot.json, regenerate.py | golden-output style |

## Recorded eval outputs (LOCAL-ONLY, gitignored — found on the primary checkout, absent from pinned tree)

`orchestrator/scripts/eval/tool_routing/.gitignore` excludes `results/` and `benchmarks/`. On the primary checkout, `orchestrator/scripts/eval/tool_routing/results/` holds:

- `summary.csv` + `report.md` (2026-05-05) — PRD-138/139 routing eval: 22 (model,mode) pairs × 47 queries = 1,034 cells. Headlines: gpt-4.1-mini full 93.6% / filtered_schema 93.6% @ $0.0007/call; claude-opus-4.7 full 95.7%; sonnet-4.6 filtered_schema 93.6%; gemini-2.5-pro 46.8% (worst); mode `graph (no-edges)` 83.0% — *below* filtered_schema, i.e. the graph prompt-shape without learned edges under-performed the schema-filtered baseline in this run.
- `results.jsonl` (2.2 MB, 2026-05-08) — per-cell records.
- `benchmarks/` does not exist.

These are the only recorded eval outputs found anywhere; none are in git.

## Seeds

`orchestrator/core/seeds/` — 10 python seeders + 2 content files: seed_auto_agent.py, seed_cto_agent.py, seed_models.py, seed_onboarding_agents.py, seed_personas.py, seed_plugin_categories.py, seed_shopify_agents.py (28 KB), seed_skills.py, seed_system_prompts.py, seed_system_settings.py (39 KB), auto-cto-custom-soul.txt, platform-management-skill.md (44 KB).

## Recorded telemetry artifacts in-repo

- None checked in (the only .jsonl in the tree is the eval set itself).
- `graphify-out/snapshots/bucket-{1..6}-pre-drop.sql` — schema-only snapshots of the 2026-04-25 Railway table-drop cleanup ("Captured at 2026-04-25 ... Source: pg_catalog (Railway live)") — evidence of dropped legacy tables (b_backup_document_chunks_*, b_mcp_tools_backup_*, ...).

## Alembic migration census

- `orchestrator/alembic/versions/`: **137 migration files**.
- Head analysis (grep revision/down_revision incl. multi-line tuples): single head **`e773c09189a9`** (`e773c09189a9_merge_prd176_prd181_heads.py` merges `prd176_merge_heads` + `prd181_s2_approval_grants`; `prd176_merge_heads.py` had merged the 4 pre-W6 heads: 20260612_nl2sql_example_embedding, prd158_cloud_default_team, prd161_sla_breach, prd164_doc_source_type). W6's "4→1 heads" held through the later waves.
- Live DB agrees it is migratable: 152 public tables incl. wave tables (approval_grants, error_events, unrouted_events).
- ORM census: 58 distinct `__tablename__` strings under `orchestrator/` — much of the schema (94 tables' worth) exists only via migrations/raw DDL, not ORM models.

## Structure graph (STALE — build artifact dated 2026-06-09, predates all 14 waves)

- **Not in the pinned tree** (`p2-src/graphify-out/` contains only `snapshots/`); stats computed from the primary checkout's `automatos-ai/graphify-out/graph.json` (34.5 MB, mtime Jun 9 21:52). `GRAPH_REPORT.md` absent from both.
- jq stats: **19,996 nodes / 63,575 links / 0 hyperedges**. Node kinds: code 13,097, rationale 6,899 (`file_type`). Link relations: calls 23,444; uses 21,092; contains 7,787; rationale_for 6,614; method 3,595; inherits 686; imports_from 356; imports 1.
- Use for shape only; every claim must be re-verified against the pinned tree.

## First look

The eval scaffolding is real and thoughtfully built (honest-outcome language in the W7 uplift eval, model-matrix routing eval with recorded results), but every recorded number is from synthetic fixtures, the NL2SQL baseline is still the 0.0 placeholder, and the only eval outputs live gitignored on one laptop. Nothing in-repo or in the DB yet measures the system against real tenant behaviour.
