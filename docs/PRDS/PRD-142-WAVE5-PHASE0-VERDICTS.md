# PRD-142 Wave 5 — Phase 0 Verdicts (WS-V / US-V2)

> **Recorded:** 2026-06-09 · **Ground:** `feat/prd142-wave5-cuts` @ `origin/main` (`6ad7e0ee2`, post-#430).
> Evidence greps ran on the wave-4 worktree @ `b0f58e4a5` (= old main + the §12.3 commit). Transfer is valid:
> the two deltas between the grounds (#430: `platform_actions.py` + one test; §12.3: harness/config/autonomy files)
> touch **no cut-list surface**.
> Method per parent §11: the graph finds candidates; **inbound-reachability grep on live code is the verdict.**
> Degree ≠ liveness — re-proven this wave (see items 2, 6a, 6b, 7, 8: every stale §11 number moved).

## §12 open decisions — defaults applied

| # | Decision | Default applied |
|---|---|---|
| 1 | DB DROPs human-gated? | **Yes** — loop/agent preps reversible migrations; **Gerard applies on prod** |
| 2 | `jobs/` + `integrations/` | **Out of scope** (not on §11; separate audit if wanted) |
| 3 | `chatbot_llm` canonical target | **Moot** — live Python callers were already migrated in earlier PRDs; shim had zero importers |
| 4 | Full-tree re-graph vs grep | **Grep** — absence checks ran across the full repo (and sibling repos for route paths) |

## Item verdicts

| # | Item | Verdict | Evidence (live grep) | Action |
|---|---|---|---|---|
| 1 | `core/neural_field/` + `AgentExecutionManager` | **KILL — executed** | `.execution_manager` property: 0 accesses anywhere; `execute_workflow_subtasks`: 0 callers; `neural_field` only importer = `execution_manager.py:184`; `ExecutionPlan`/`SubtaskExecution`/`SubtaskStatus`: 0 consumers outside the unit | WS-W: deleted both packages, the lazy property, and the `modules/agents` re-exports |
| 2 | `api/chatbot_llm.py` (547 ln) + `chatbot_router` | **KILL — executed** (migrate-first **not needed**) | §11's "6 edges / 5 files" was stale — only remaining refs were the `main.py` import + mount. Route check: only `/api/chatbot`-prefix router; frontend's 4 wrapper methods had **0 callers**, and 3 of the 4 wrapped endpoints (`/execute`, `/history`, `/feedback`) no longer existed server-side | WS-X: deleted shim, unmounted, deleted the 4 dead frontend wrappers |
| 3 | `_stream_workflow_bridge` (`chat.py`) | **KILL — executed** | 0 inbound — only its own `def` at `chat.py:37`; all its imports were function-local | WS-W: deleted (−142 ln). `consumers/workflows` **survives** — `api/workflows.py` still imports `streaming` ×3 |
| 4 | `seed_recipes_marketplace_v2.py` | **KILL — executed** (PRD's "not found" was **wrong** — file existed) | 0 invocations (no CI / Dockerfile / Makefile / *.sh); writes `marketplace_items` `type='recipe'` while the live marketplace browse serves recipes from `WorkflowRecipe` (`owner_type='marketplace'`) | WS-W: deleted + removed the stale docs link in `marketplace-backend.md` |
| 5 | Context Forge remnants (PRD-33/34/35) | **ALREADY GONE** | Full-tree code grep (`context[_ -]?forge`, py/ts/tsx/js/sql): zero | None |
| 6a | PRD-12 Playbook miner | **KEEP — §11 claim falsified** | NOT "never wired": mounted at `main.py:62`; `api_playbooks.py:31` instantiates `PlaybookMiner`; `test_playbook_launch_parity.py:312` **enforces** the 49-line read-only stub as a human-gate contract | Removed from the wave |
| 6b | PRD-20 MCP scaffolding | **KILL — executed** (doubly dead) | `mcp_executor.py`: 0 importers. Stronger: the models it imported (`MCPTool`, `AgentToolAssignment`, `ToolUsageLog`) **no longer exist in `core/models`** — its import line would have raised `ImportError` if ever executed | WS-W: deleted |
| 7 | "~198 unreferenced routes" | **RE-MINED: 290 candidates** (not verdicts) | 531 routes / 100 router files, **all mounted**; 290 have zero path-string refs across frontend/tests/scripts/services | WS-Z input below — per-route verification before any cut |
| 8 | "~53 dead tables" | **RE-MINED: 10 dead + 1 flagged** | 106 tables; 10 with zero refs outside model+migrations and no live FKs; `intent_classification_cache` is FK'd by live `ToolExecutionLog.intent_cluster_id` | WS-AA input below — human-gated |

## Executed gates (WS-W + WS-X, this branch)

- **Cut-greps: zero** for every executed symbol (`neural_field`, `AgentExecutionManager`, `execute_workflow_subtasks`, `ExecutionPlan`/`SubtaskExecution`/`SubtaskStatus`, `_stream_workflow_bridge`, `chatbot_llm`, `chatbot_router`, `mcp_executor`/`MCPToolExecut*`, `seed_recipes_marketplace_v2`, `/api/chatbot`). One residual hit is a JSX *comment string* ("Neural field visualization") in the live `mission-field-panel.tsx` — prose, not a reference.
- `py_compile` green on every touched module. Net change: **18 files, −3,849 / +2 lines.**
- Test net (`orchestrator-tests` CI) gates the PR — runs on push.

## WS-Z input — 290 route candidates (re-mined 2026-06-09; **candidates, not verdicts**)

Mandatory per-route checks before any cut (the re-mine has known blind spots):

1. **Parametric paths** — the miner grepped literal `{param}` strings; frontend template literals (e.g. `` `/api/team/members/${id}/role` ``) won't match. Re-grep by **path segments**. Mission state mutations (`/approve`, `/pause`, `/cancel`, `/replan`…), team-member routes, and similar are **likely false-dead**.
2. **Flag-gated consumers** (`HARNESS_*`, channels, feature flags) are invisible to grep — check the flag registry before declaring dead.
3. **External callers** — widgets/* (merchant-site JS), `chat_voice` audio serving, Composio/Shopify/Clerk/Stripe webhooks + OAuth callbacks → flag `KEEP-external`, never cut on internal evidence alone.
4. **Internal schedulers** — heartbeat/cron paths may be called by the scheduler, not HTTP consumers.

Strongest clusters (highest-confidence starting points): `analytics_real.py` (18 routes / 0 refs), `heartbeat.py` (12 / 0 — but see check 4), the `problems.py`/`solutions.py`/`synthesis.py`/`insights.py` subsystem, `context_engineering.py` info-theory utilities (`/entropy`, `/mutual-information`, `/graph/centrality`…), legacy `workflows.py` execution-streaming routes.

## WS-AA input — table candidates (human-gated; routes cut first per parent rule)

**Verified dead** (zero refs outside model + migrations; no live FKs): `benchmark_assessments`, `component_metrics`, `database_relationships`, `evaluation_results`, `external_knowledge`, `integration_analyses`, `tool_credentials`, `tool_execution_cache`, `tool_installation_requests`, `tool_reviews`.

**Flagged:** `intent_classification_cache` — FK from live `ToolExecutionLog.intent_cluster_id`; dropping requires touching a live model → needs explicit decision.

**Residual-schema check:** the mcp-era models are gone from code, but prod may still hold their tables (e.g. `mcp_tools`-family) created by old migrations. The WS-AA live-schema pass should list tables **with no corresponding model** and fold them into the DROP review.

Process: WS-Z route cuts land first → re-verify each table on the live schema → author **reversible** DROP migrations → **Gerard applies**; the loop never runs them.

## Flags for Gerard

1. **`automatos-mobile`** references `/api/chatbot/history` and `/api/chatbot/stream` in its api-client — endpoints that no longer exist (most never existed post-consolidation). Sibling-repo cleanup, not this PR.
2. **`seed_recipes_marketplace.py` (v1)** has the same evidence profile as the deleted v2 (zero invocations; writes the same superseded `marketplace_items` `type='recipe'` path). Not on §11, so left in place — say the word and it folds into the wave.
3. `marketplace_items` rows with `type='recipe'` in prod look vestigial (live browse reads `WorkflowRecipe`) — candidate for the WS-AA data-cleanup review.

---

# Addendum — WS-Z / WS-AA execution (2026-06-09, post-merge of #431/#432)

Per-story outcomes on `ralph/prd-142-wave5-cut-list` (fresh main):

| Story | Outcome | Evidence |
|---|---|---|
| **W5-S1** `analytics_real.py` | **KEEP — all 18 routes live.** The "18/0" stale flag was hook-indirection blindness: `use-analytics-api.ts` hooks → IsItWorkingStrip / Command Center; composite routes delegate to sub-handlers as functions. | No cuts |
| **W5-S2** `heartbeat.py` | **KEEP — all 12 routes live.** Routes are the HTTP control surface of the live `HeartbeatService` (APScheduler, `HEARTBEAT_ENABLED`, started in main.py lifespan); FE callers in agent-configuration-modal, use-heartbeats-api, Routines tab, analytics-overview. | No cuts |
| **W5-S3** problems/solutions/synthesis/insights | **KILLED end-to-end.** 7 stub routes (hardcoded JSON), zero callers, zero tables; 3 orphaned FE hook files whose api-client methods didn't even exist (broken at runtime). | 4 routers + 3 hooks + mounts deleted |
| **W5-S4** `context_engineering.py` | **KILLED — whole router** (12 routes, 0 callers, 2 handlers were commented-out husks) + route-only orphans `QueryProcessor`/`MultiModalProcessor`. Engines KEPT (core.math, SemanticChunker, ContextOptimizer — live in RAG). Live `ContentModality` users import the separate `modules.rag` copy. | Router + 2 classes deleted |
| **W5-S5** workflows.py legacy streaming | **KEEP — all streaming/execution routes are live Mission-UI paths** (execution-kitchen.tsx → `/stream/aisdk` is the canonical protocol; template-literal callers). Only cut: the PRD-125 410-Gone `/results` stub pair + orphaned hook/query-key/client method. `consumers/workflows/streaming` survives. | Micro-cut only |
| **W5-S6** remainder sweep | **VERDICTS REJECTED — no cuts.** The remainder agent labelled 375/534 routes KILL via literal path-grep, skipping the api-client method indirection. Sample-disproof: `agents.py` CRUD live via `use-agent-api.ts`; `attachments.py` live at `api-client.ts:1725`; `api_playbooks.py /mine` contradicts the Phase-0 parity-test KEEP. The remainder sweep returns to the Ralph loop with the corrected method (trace api-client wrapper → hook → component before any verdict). | No cuts on unsafe evidence |
| **W5-S7** WS-AA | **DONE (prep).** All 10 tables re-verified zero-ref on post-merge main; reversible standalone migration `prd142_wave5_drop_dead_tables` authored (DROP IF EXISTS ×10; downgrade recreates from DDL snapshots); 10 model classes + 2 live back-refs removed. `intent_classification_cache` EXCLUDED (FK from live `ToolExecutionLog`). **Apply = Gerard only.** | Migration + models committed |

**Gates:** every cut grep-zero; `py_compile` green; full net **1807 passed** with only the 2 pre-existing L2-transcript flakes (identical to the pristine-main baseline).

**Method lesson (now in the loop PROMPT):** a route's liveness verdict REQUIRES tracing the frontend indirection chain — api-client wrapper method → hook → component callers. Literal path-grep alone produced 375 false-KILLs in one pass.
