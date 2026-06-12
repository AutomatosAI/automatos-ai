# Automatos AI — Platform Deep Review (June 2026)

Synthesized from 13 review/research streams (10 components + 3 cross-cutting sweeps). All file:line references are against the repo at review time. Constraints honored throughout: PRD-150 open-core split (no deepening Clerk coupling), Auto gets ALL platform APIs except the super_admin observability tier, board SDK-key reads stay on the narrow scope-gated dependency, and the locked decisions (Teams-based RAG scoping with multi-team tags + retrieval-time enforcement; visual block editor over JSON → PDF+DOCX for templates; Code Canvas embeds the Claude Agent SDK headless per workspace; execution as a Ralph PRD chain, one PRD per workstream).

---

## 1. Executive Summary

Five themes explain most of the pain:

**Theme 1 — Agents are starved at the LLM boundary while humans get the full data.** Every "agents feel dumb" complaint traces to truncation at the exact point where data enters the LLM. RAG agents reason over 500-char S3 metadata previews while the frontend artifact viewer fetches the full document (`modules/rag/ingestion/manager.py:1300` vs `result_formatter.py:589-616`); there is no read-a-document tool; code search results reach the LLM without line numbers or qualified names; NL2SQL results are JSON cut mid-structure at 3000 chars; chat hard-truncates tool context at 6000 chars (`consumers/chatbot/service.py:1269-1271`). The fixes are cheap (batched full-text hydration, read tools, token-budgeted formatting) and are the highest-yield work in this report.

**Theme 2 — Learning loops are write-only.** The platform diligently records mission summaries, task failures, field patterns, golden SQL, and graph data — and then never reads any of it back. Mission planning consults no RAG/memory/graph (`modules/coordination/planner.py:17-66`); L2 memory decays in ~15 hours and promotion is dead code (nothing calls `touch_short_term`); the per-mission field is destroyed at completion (`coordinator_service.py:711-737`); NL2SQL embeddings are computed then discarded (`example_store.py:90-101`). The write side exists almost everywhere; the read side is the missing 20% that would make the platform feel like it learns.

**Theme 3 — In-process state and silent failure in a 4-worker deployment.** A whole class of "feels flaky, is actually deterministic" bugs: the calendar reads APScheduler state that exists in 1 of 4 uvicorn workers, so 75% of loads return an empty 200 (`Dockerfile:140`, `main.py:445-464`, `activity_service.py:683-742`); board pickup depends on per-agent heartbeats that nothing enables and whose enable API silently no-ops on 3 of 4 workers; ~25 routers mount inside try/except that swallows ImportError (two imports already fail silently every boot, `main.py:115,123`); three Knowledge-Base tabs fetch with no auth against the wrong origin and render as silently empty; the RAG team filter fails OPEN on DB error (`modules/rag/service.py:391-393`).

**Theme 4 — Auto's tool surface is broken exactly where lifecycle management matters.** Auto can create missions but `platform_get_mission` crashes on every call (`int()` on a UUID PK, `handlers_missions.py:96`); there are no approve/pause/cancel/replan tools and no plan-ready notification, so Auto-created missions silently stall in `awaiting_approval`; codegraph's call-graph/dependency/architecture tools are implemented but never registered; deliverables, document templates, and the unified schedule are invisible to agents; and platform-action results never render as widgets in chat (`frontend/components/widgets/router.ts` has zero `platform_execute` handling) — so the platform "doesn't flow" even when the data round-trip succeeds.

**Theme 5 — Trust-eroding fake states and unguarded tenant boundaries.** The UI fabricates data: StudioTicker's hardcoded "live" metrics on every page, 85.5% agent performance (`api/agents.py:331-336`), hardcoded Studio tab counts, fake green validation checkmarks, placebo RAG settings — while roughly half of all success/error feedback never renders because sonner's `<Toaster>` was never mounted (49 files call it; `providers.tsx:91` mounts only react-hot-toast). Meanwhile real boundaries are missing: multimodal search has no workspace filter, legacy entity-KG endpoints are unauthenticated and cross-tenant, template CRUD has cross-workspace IDOR plus unsandboxed Jinja2 SSTI, and the mem0 server has no auth. These must close before the PRD-150 open-core split widens exposure.

---

## 2. Per-Component Findings

Thirteen review streams. For each: verdict, confirmed root causes of the owner's complaints, additional issues (P0 broken / P1 high-value / P2 polish), and top adoptable OSS ideas where a research stream was commissioned.

### 2.1 RAG / Documents & Team Scoping

**Verdict:** Genuinely sophisticated retrieval machinery (HyDE, RRF, rerank, knapsack budgeting) undermined by one content-starvation bug, missing read tools, and team/workspace enforcement holes.

**Confirmed root causes (owner-reported):**
- Agents see only 500-char chunk previews: S3 Vectors metadata stores `chunk_text=content[:500]` (`modules/rag/ingestion/manager.py:1300`), retrieval reads it back (`s3_vectors_backend.py:160`), and the PRD-136 4000-char per-doc cap (`result_formatter.py:354-356`) can never be filled — while the human-facing viewer fetches the FULL document (`result_formatter.py:589-616`). This is the "opens it in the file browser instead of reading it" symptom.
- Parent-child expansion fetches full chunks into `expanded_content` (`modules/rag/service.py:855-909`, one asyncpg query per candidate) and nothing ever consumes it (`:544`, `:754`) — the fix already half-exists as dead code.
- No tool lets an agent read a full document: `actions_documents.py` registers only list/delete/reprocess; the `[View Document]` links are human-clickable only (`consumers/chatbot/tool_router.py:134`).
- Team upload control is free text, not a dropdown (`document-upload.tsx:448-482`, `upload-provider-modal.tsx:204-232`) even though the distinct-teams list exists at `GET /api/agents/org-chart` (`api/agents.py:573-669`). Typos create phantom security domains.
- Team selector exists only client-side inside LocalStorageBrowser (`local-storage-browser.tsx:75-150`); `GET /api/documents` has no team param (`api/documents.py:539-560`); no UI to edit a document's teams after upload; the PATCH team-access endpoints crash on nonexistent `title`/`updated_at` columns (`api/documents.py:1871-1914`).
- Teams are case-inconsistent across surfaces: docs normalize lowercase, agents/org-chart return raw case (`core/team_access.py` vs `api/agents.py:662-669`).

**Additional issues:**
- **P0:** Multimodal search tools (`search_tables/images/formulas/multimodal`) have NO workspace or team filter — cross-workspace leak (`multimodal_knowledge_tools.py:65-99,175,274,401`). Team filter fails OPEN on any DB error (`service.py:391-393`). `platform_delete_document` lets a support-team agent delete another team's docs (`handlers_documents.py:13-105`). RAG analytics aggregate across ALL workspaces (`service.py:1007-1168` via `api/context.py:86-151`). Widget/SDK docs API queries columns that don't exist in the schema (`api/widgets/docs.py:108-197`). `GET /api/documents/content` (by path) has no auth dependency (`api/documents.py:411-452`).
- **P1:** Multimodal similarity subquery is functionally broken (exact-match `WHERE content = :query`, `multimodal_knowledge_tools.py:77-92`); agent-configured `min_similarity` silently dropped (`agent_platform_tools.py:291-297` → `service.py:505-524`); UI semantic search maps S3 hits by filename with N+1 queries (`api/documents.py:935-942`); multimodal upload accepts `team_access` and drops it (`knowledge_multimodal.py:258`).
- **P2:** Placebo `rag_configurations` knobs never read by retrieval (`configure-rag-modal.tsx:47-114`); unmounted 700-line duplicate knowledge router; unused RAG feedback hook; fake fallback documents on API failure (`use-document-api.ts:29-43`); tags never persisted; per-retrieval perf traps (pure-Python knapsack DP, fresh S3 backend + up to 7 SessionLocals per instantiation, `service.py:47-95,674-803`).

**Top adoptable ideas (research):**
- **Onyx (MIT core; ee/ is enterprise-licensed — patterns only):** index-time ACL pre-filter with a `__PUBLIC__` sentinel token written into filterable vector metadata, plus ONE central filter builder every search path must pass through — structurally fixes the multimodal leak class and the post-filter recall loss, while staying within the locked retrieval-time-enforcement decision.
- **Letta (Apache-2.0):** `open_files`/`grep_files` paged, line-numbered read tools with view-window limits derived from the model's context window — the direct answer to "no read-a-document tool."
- **AnythingLLM (MIT):** full chunk text guaranteed at retrieval + document pinning (whole-doc injection for small canonical docs) + 4-column per-workspace RAG tuning (similarity threshold, topN, query-mode refusal).
- **RAGFlow (Apache-2.0):** `citation_prompt.md` portable verbatim (inline `[ID:n]` markers, must-cite rules), token-budget whole-chunk accumulation at 97% of max_tokens, and a chunk-management UI (edit/boost/disable individual chunks).
- **Open WebUI (v0.6.6+ non-OSI license — patterns only, no vendored code):** the `access_control` JSON-blob → normalized `access_grant` table migration is the schema blueprint if team grants ever outgrow `team_access TEXT[]`.

### 2.2 Memory (extraction → storage → recall)

**Verdict:** The L1/L2/L3 architecture and PRD-131d distillation direction are right; the system fails on knobs and dead code — double extraction mangles facts, operational memories are excluded by prompt and then killed by a 15-hour decay, and the delete endpoint has never worked.

**Confirmed root causes (owner-reported):**
- Junk memories: curated facts are RE-extracted server-side by mem0's "Personal Information Organizer" prompt because `infer` defaults true (`smart_memory.py:441-473` → `openmemory memories.py:415-423`; prompt at `automatos-mem0/mem0/configs/prompts.py:14-59`). The raw-exchange fallback and all non-chat writers (mission, playbook, widget, tools) bypass distillation entirely and feed unfiltered text to that junk-prone prompt.
- Missing operational memories — three compounding causes: the distill prompt explicitly forbids them ("Do NOT record transient interaction events", `smart_memory.py:611-613`); L2 rows die in ~15 hours (`MEMORY_DECAY_RATE=0.1/hr`, `unified_memory_service.py:1114-1121`, archive threshold `config.py:104-106`); and L2→L3 promotion is dead code — `touch_short_term` has zero callers so `access_count > 3` is unreachable (`unified_memory_service.py:1990-1996`).
- Recipe/workflow memory namespaces are invisible to chat recall AND Memory Explorer (`memory_stats.py:66-101`; `context_router.py:507-509`); L2 operational rows are only recalled on temporal-regex queries with raw ILIKE (`context_router.py:489`, `unified_memory_service.py:833-844`) — "why did the deploy mission fail?" hits neither.
- "Memory stored" SSE fires on scheduling, not persistence (`smart_orchestrator.py:384,433-434`; `service.py:1099-1107`), with a `_last_tier` read race.

**Additional issues:**
- **P0:** Single-memory DELETE calls an endpoint that doesn't exist on the mem0 server — every delete 405s, trips circuit breakers, makes Explorer delete fail, guarantees consolidation duplication, and breaks daily-log cleanup + L3 archival (`mem0_client.py:498-506` vs bulk-only `mem0_openapi.json`). L1→L2 session consolidation is dead (`end_session` has zero callers; `decisions/action_items` have no writer).
- **P1:** OpenMemory server has NO authentication — the orchestrator's Token header is ignored (`openmemory memories.py`, `mem0_client.py:160`). Legacy `/api/v1/memory` router serves mock CRUD and `random.uniform` time-series on the same prefix as the real Explorer API (`api/memory.py:354-438,586-677`). Daily-log retrieval reads an unordered first-50 of an ever-growing namespace (`mem0_client.py:466-496`). Vector-store metadata is hardcoded, dropping tier/category provenance (`memories.py:418-421`); UPDATE/DELETE events ignored so SQL view drifts from pgvector (`memories.py:432-433`).
- **P2:** Default tier 'both' double-writes most exchanges (`smart_memory.py:169`); 3 LLM calls per chat turn for memory alone; MemoryWidget frontend is a simulation; widget delete has no ownership check; no relevance floor on recall — junk gets replayed into every conversation.

**Top adoptable ideas (research):**
- **mem0 upstream V3 (Apache-2.0):** `infer=False` for pre-distilled facts (the fork already plumbs the flag end-to-end — a near one-line client change), and the V3 single-pass ADDITIVE extraction prompt whose rules (no echo-extraction, no greetings, agent-attribution framing, extract from both sides) should replace the current distill prompt's content — including deleting the operational exclusion.
- **Zep/Graphiti (Apache-2.0):** bi-temporal contradiction invalidation (`valid_at`/`invalid_at`/`expired_at`) replacing Ebbinghaus decay — facts live until superseded, not 15 hours; typed ontology with `Procedure` as a first-class memory type; 0-1 fact ratings with a `min_rating` retrieval floor.
- **Letta (Apache-2.0):** message-count-triggered sleep-time consolidation pass replacing the dead hourly jobs; memory mutations as visible operations.
- **cognee (Apache-2.0):** SHA-256 content-hash dedup BEFORE any LLM/network call (~15 lines, Redis SETNX) — kills retry-loop duplicate writes; retrieval-as-touch feedback to revive promotion.
- **LangMem (MIT):** Profile pattern (single strict-schema doc, continuously patched) for the L1 session summary instead of 500-char truncation.

### 2.3 NL2SQL / Databases

**Verdict:** The SQL Explorer path (few-shot + self-correction + validator) is real and decent; 3 of 6 sub-tabs are dead from wrong fetch wiring; the agent path is a security hole that can't reach customer databases at all.

**Confirmed root causes (owner-reported):**
- Semantic Layer, Query Templates, and Training tabs use raw relative `fetch('/api/knowledge/...')` with no auth against the Next.js origin — rewrites are disabled (`next.config.js:18`), so all three silently render empty in any deployed environment (`SemanticLayerBuilder.tsx:125,161`; `QueryTemplatesGrid.tsx:90,136`; `TrainingExamplesManager.tsx:78-171`).
- Semantic Layer is dead end-to-end: the GET route doesn't exist (only POST), the save handler calls two methods that don't exist (always HTTP 400), never commits, and generation never reads the semantic layer anyway (`service.py:518-530`, `:349-357` vs `nl2sql_service.py:51`).
- Query Templates is a mock shell: real prop ignored, SQL/params/usage fabricated client-side (`QueryTemplatesGrid.tsx:70-101,362-418`), execute route mismatched and the backend handler is a hardcoded empty stub (`api/database_knowledge.py:433-457`), table never seeded.
- Audit History is empty because the NL query path never writes `DatabaseQueryAudit` rows — only the dev-only `/query/sql` endpoint does (`api/database_knowledge.py:489-615`).
- Dialect selected in Add Database is silently discarded — everything stored as postgresql (`service.py:202`); LLM generation failure produces an executable `SELECT 'Error...'` that returns success (`nl2sql_service.py:109-113`); SQL Explorer renders three hardcoded green "validation passed" rows (`DatabaseQueryExplorer.tsx:148,395-409`); all sub-tabs except the Explorer are pinned to `databaseSources[0]` with no selector (`document-management.tsx:1089-1131`).
- Training feedback is real but inert: only verified examples enter prompts, the verification UI can't reach the backend, and embeddings are computed then thrown away — retrieval is keyword Jaccard (`example_store.py:90-165`).

**Additional issues:**
- **P0:** Agent `query_database` makes an UNAUTHENTICATED httpx self-call that 401s in production and silently falls back to running LLM-generated SQL against the platform's own multi-tenant Postgres guarded only by `startswith('SELECT')` — no validator, LIMIT, timeout, or workspace scoping (`exec_research.py:38-55,95-140`). Audit listing endpoint has no workspace scoping — cross-tenant read of NL queries and generated SQL (`api/database_knowledge.py:667-681`; also `database_analytics.py:25-119`).
- **P1:** `smart_query_database` can never query customer DBs (always platform schema, `exec_research.py:184-240,297`) and its multi-turn clarification is broken by stateless construction; 4-5 LLM calls per invocation with no budget control; tool output JSON hard-truncated at 3000 chars mid-structure (`tool_router.py:186`); connection strings built via f-string with engine leaks and no statement timeout (`service.py:398-404`).
- **P2:** Dead scaffolding everywhere — per-source tools pointing at nonexistent executor methods (`database_tool_integration.py:262,276,364`), unregistered simulation router, unused `useSemanticLayer` hook that would ReferenceError; hardcoded `user_id='1'`/`tenant_id=1`; only the validator has tests.

**Top adoptable ideas (research):**
- **Vanna (MIT):** persist the embeddings you already compute into pgvector (already in requirements) and retrieve cosine top-k; the `intermediate_sql` two-pass for WHERE-literal grounding gated by `allow_llm_to_see_data`; token-budgeted prompt assembly; auto-train only when rows > 0.
- **WrenAI (app is AGPL-3.0 — concepts only; core engine Apache-2.0):** in-process typed toolkit for agents (no HTTP self-calls anywhere), dry-run validation, bounded `retries=2` self-correction with structured errors, MDL-lite JSON as the semantic-layer schema with TWO real consumers (prompt rendering + validator allowlist).
- **Dataherald (Apache-2.0, dormant — pattern-mine only):** low-cardinality column-value sampling at schema sync; audit-first persistence (one row per generation, success or failure, with confidence/tokens); per-source admin "instructions" as the v1 semantic layer (numbered rules in the prompt) — shippable in days.
- **sqlglot (MIT):** AST-based validator replacing the regex one — then CTEs/UNION can safely return.
- **defog sql-eval (Apache-2.0):** execution-accuracy regression harness over verified golden pairs (subset dataframe matching) — the missing safety net for every prompt/model change.

### 2.4 Semantic Field Memory (PRD-108)

**Verdict:** The Qdrant core adapter is solid (58 tests); everything visible is broken — the legacy page calls a backend deleted in March, mission fields are destroyed the moment you'd want to look at them, and all writes are attributed to "System".

**Confirmed root causes (owner-reported):**
- The visual graph never renders: `orchestrator/api/field_theory.py` was deleted 2026-03-01 (commit 6f15c1944, PRD-68 cleanup) but the frontend survived (`api-client.ts:2466-2496`); every call 404s, the page is orphaned from all navigation, and even with data the canvas drew synthetic art, not field state (`field-visualization.tsx:63-137`).
- Mission fields are destroyed on terminal state every tick, so the 3D field tab is blank for any completed mission (`coordinator_service.py:711-737`); the OOM rationale is stale since the single-collection refactor.
- Multi-agent viz collapses: all tool-path writes/reads attributed to `agent_id=0` — the executor passes no kwargs while handlers read kwargs (`platform_executor.py:737` vs `handlers_field.py:28,92`); `exec_platform.py:66-72` injects `_agent_id` into params that handlers never read.
- Token limits: full upstream outputs (8000 chars each) are stuffed into downstream prompts AND injected into the field — the field is token-additive, never token-saving (`coordinator_service.py:1319-1340`); `platform_field_query` returns up to ~10k tokens unbudgeted (`handlers_field.py:59-74`); injects silently truncate at 4000 chars (`:112`).

**Additional issues:**
- **P0:** `field_id` resolves to `.first()` running mission in the workspace — concurrent missions cross-pollinate (`platform_executor.py:716-733`); agent-supplied `field_id` honored with no ownership check — cross-workspace read/write (`handlers_field.py:27,91`).
- **P1:** Silent degradation to hash-noise embeddings when no provider configured (`core/llm/clients/base.py:200-227`); embedding-dimension config drift can break the collection; instrumentation metrics leak memory and vanish on restart/replicas (`instrumentation.py:103-110`); redis backend silently blanks the panel.
- **P2:** Full-collection scrolls polled every 8s; N sequential Qdrant round-trips per Hebbian reinforce; field tools pinned into EVERY chat turn yet hard-fail outside missions; `system.py:1100` hardcodes `field_theory: healthy`; zero tests on the agent-facing surface.

**Top adoptable ideas (research):**
- **Graphiti (Apache-2.0):** soft-archive with `expired_at` instead of destroy — one payload field + query filter makes completed missions render their final field; episode-style provenance payloads (source_type/agent_id/task_id).
- **Mem0 (Apache-2.0):** mandatory scoping keys on every field op (field_id from the task's own run, agent_id, task_id) — structurally kills both the `.first()` and agent-0 bugs.
- **Stanford Generative Agents (Apache-2.0) + MemoryBank (MIT):** three-factor retrieval (relevance × recency-since-last-access × LLM importance) and adaptive per-pattern half-life.
- **react-force-graph + getzep/zep-graph-visualization (both MIT, same Next.js/shadcn stack):** rebuild the viz on real node+edge data with a click-detail provenance panel; delete the dead /field-theory surface entirely. Do NOT ship Cosmograph app/@cosmograph/react (commercial) — the cosmos.gl engine is MIT if GPU scale is ever needed.
- **Letta (Apache-2.0):** pinned char-capped "field digest" block in every dispatch prompt (agents often don't call the tool), plus an eviction-warning handshake at the 80% budget gate telling agents to checkpoint findings to the field.

### 2.5 Knowledge Graphs UI

**Verdict:** Good build pipeline and well-designed agent tools; the viz is crippled by a link-mutation bug, a Shopify-only palette, permanent overlay chips, and clusters that are just numbers.

**Confirmed root causes (owner-reported):**
- Filter chips are permanent overlays inside the viewport with no hide/show toggle, plus controls scattered across three other clusters (`BusinessGraphPanel.tsx:627-784`).
- Every non-Shopify graph renders uniformly gray: `TYPE_COLORS`/`RELATION_COLORS` contain only 5 shopify types (`BusinessGraphVisualization.tsx:92-136`) while the platform's own extraction emits concept/entity/process/metric/rule/agent/table/integration (`graph_extraction.py:115-478`).
- Drill-in is broken: react-force-graph mutates shared link objects (source/target string → object); the panel handles only the string form, so the detail panel shows no connections and post-render filters drop all edges (`BusinessGraphPanel.tsx:403-415` vs the viz's own defense at `BusinessGraphVisualization.tsx:204-205`).
- Cluster sidebar and agent tools expose only "Cluster 47": `score_all` is computed then discarded (`graph_service.py:240-242,405`), `_format_communities` writes ids+members only (`:838-849`).
- CodeGraph drill-in is a stub: node click never fetches symbol details (`CodeGraphVisualization.tsx:310-319`); its Type selector and heatmap are non-functional.

**Additional issues:**
- **P0:** Legacy entity-KG endpoints are unauthenticated AND unscoped — full `knowledge_items.content` for any tenant to any caller (`api/knowledge_graph.py:84-559`, mounted bare at `main.py:1009`). `platform_graph_communities` reads the wrong storage backend (worker FS instead of Postgres `workspace_graphs`) so it returns "No communities data" for most workspaces (`handlers_graph.py:274-277`).
- **P1:** Legacy KnowledgeGraphVisualizer is dead in production (relative unauthenticated fetches, `KnowledgeGraphVisualizer.tsx:79,96`); entire graph.json (up to 50MB) downloaded and parsed on the main thread with no server-side subgraph endpoint; no search debounce — every keystroke re-heats a 50k-node simulation.
- **P2:** Stale "5,000-node" copy vs 50k cap; GraphDiffBanner button console.logs; duplicate import path + localStorage workspace-ID; god_nodes serialized as Python tuple reprs (`graph_service.py:863-866`); `platform_query_graph` traverses from only one top node (no two-concept paths); `graph_neighbors` unbounded for god nodes.

**Top adoptable ideas (research):**
- **Microsoft GraphRAG (MIT):** `community_reports` schema — one cheap LLM pass per community emitting {title, summary, rank, findings}; the discarded `score_all` output becomes `rank` for free. Fixes "Cluster 47" in both UI and agent tools with one artifact change. Plus gleaning loops for extraction density.
- **Neo4j Bloom (proprietary — patterns only):** collapsible legend with live per-type counts and click-to-isolate; "unique values" auto-coloring from the existing 24-color community wheel.
- **LightRAG (MIT):** server-side subgraph endpoint (`label + max_depth + max_nodes`) reusing the traversal the agent tools already have; clean model/renderer split (graphology source-of-truth, renderer never mutates data) — the architectural fix for the mutation bug class.
- **graphrag-visualizer (MIT):** existence proof the current react-force-graph stack handles this workload — no renderer migration needed; communities/provenance as first-class navigable nodes.
- Do NOT vendor: Neo4j NVL (proprietary LICENSE.txt), Ogma/Linkurious (commercial), Gephi Lite (GPL-3.0).

### 2.6 CodeGraph

**Verdict:** Indexing pipeline is solid (tree-sitter 15 langs, SHA-256 incremental, SSRF-validated clones); the agent surface exposes 1 of ~6 capabilities, and a query-log INSERT bug likely fails every search on schema-matching databases.

**Confirmed root causes (owner-reported):**
- Graph-traversal tools are dead code: `get_call_graph`/`find_dependencies`/`analyze_architecture` have full schemas AND working executors (`agent_platform_tools.py:136-206,614-697`) but are absent from ToolRegistry (`tool_registry.py:470` registers only `search_codebase`) and `unified_executor.tool_routes` (`:95-159`) — dispatch hits "Unknown tool". The entire relationships table is unreachable by any agent.
- `search_codebase` never uses semantic search — always ILIKE on names (`agent_platform_tools.py:531`), so NL "concept" queries return nothing despite the description inviting them; the purpose-built `prompt_block` from semantic search is generated and discarded.
- No freshness: `auto_reindex` is write-only (no scheduler/webhook consumes it); the UI Reindex wipes all rows and re-embeds everything, losing exclude_patterns (`api/codegraph.py:442-476`).
- LLM context drops line numbers, signature, docstring, and qualified_name, with a hardcoded ```` ```python ```` fence for all languages (`tool_router.py:163-166`; `result_formatter.py:814-817`) — agents cannot cite file:line or chain queries.

**Additional issues:**
- **P0:** Query-log INSERT uses `execution_time_ms` and omits NOT NULL `workspace_id` vs the migration's `duration_ms` + `workspace_id` (`codegraph_service.py:1264-1277,1363-1376,1445-1458` vs `20260218_fix_codegraph_schema_v2.py:118-128`) — on a migration-matching DB every search raises AFTER computing results, masked as "no indexed project". Embedding column is never created on fresh DBs (migration omits it; service only ALTERs an existing column, `:837-911`) — semantic search silently empty.
- **P1:** Call edges resolved by bare-name globally-unique fuzzy match — common method names dropped, methods not class-scoped (`codegraph_service.py:706-716,1066-1081`); hardcoded default `project_name='Automatos-ai'` in the tool schema defeats workspace auto-resolution (`tool_registry.py:494-499`); clone failures can leak GitHub tokens into logs (`:468,489`); architecture analysis runs O(V×E) betweenness + exponential simple_cycles synchronously in the event loop, uncached (`architecture_analyzer.py:179,229`); service-level project INSERT casts JSON into a text[] column (`:196`).
- **P2:** PageRank "ranking" only ranks the ≤20 candidates while loading ALL relationships per query; leading-wildcard ILIKE with no pg_trgm index; NameError on zero-parseable-file repos (`:321`); fictional README architecture; tests broken by the PRD-70 URL allowlist; misleading collapse of all errors to "no indexed project" (`agent_platform_tools.py:543-552`).

**Top adoptable ideas:** No dedicated research stream. Apply cross-stream patterns: register the three executors via the proven `platform_graph_*` ActionRegistry pattern; the knowledge-graph stream's server-side subgraph + shared GraphView shell; the board stream's webhook-driven freshness (GitHub push → incremental reindex via existing `workspace_github.py`); Letta-style `get_symbol` follow-up reads after truncated hits.

### 2.7 Board / Task Pickup

**Verdict:** Solid model, tenancy, and bridges; the pickup architecture is wrong — heartbeat-gated batch pseudo-execution instead of assignment-triggered dispatch. "Assigned" is a dead-end column by default.

**Confirmed root causes (owner-reported):**
- Assigned tasks have NO pickup path unless that agent's heartbeat is explicitly enabled — and agent creation never enables it (`board_tasks.py:461-466,709` only trigger on in_progress; the sole automation is the heartbeat scan `heartbeat_service.py:912-931`, gated by `configuration['heartbeat']['enabled']` at `:286-294`).
- Enabling a heartbeat via the API silently no-ops on 3 of 4 uvicorn workers until redeploy (`Dockerfile:140` + fcntl leader `main.py:440-459`; `api/heartbeat.py:113-125` swallows the AttributeError and returns `{"ok": true}`).
- Even when enabled: 60-min interval, 08:00–20:00 active hours, 90% cooldown, and a 3-task cap (`heartbeat_service.py:854-868,920`) — worst case ~12h wait. Priority ordering is lexicographic so HIGH is picked up LAST (`:919`).
- "Pickup" isn't execution: up to 3 tasks are folded into one heartbeat LLM call capped at a 500-char summary, and the SAME truncated text is stamped onto every picked task (`heartbeat_service.py:956-966,1004-1018`).
- Failure path makes work silently die as "done": in_progress is committed BEFORE execution and never reverted; the reconciler then force-closes stalls as `done` + 'Stalled' with no retry (`heartbeat_service.py:926-931,1026-1037`; `task_reconciler.py:139-154`).

**Additional issues:**
- **P0:** AI plan→refine flow 500s on every refine (frontend sends `Record<string,number>`, backend iterates `[{question,answer}]` — `create-task-dialog.tsx:59,329` vs `board_tasks.py:927-929`). Dragging a mission-mirror card to In Progress fires a rogue duplicate agent execution outside the coordinator (`board_tasks.py:465,709` exclude only 'recipe').
- **P1:** No double-launch guard (re-drag re-triggers and wipes state, `:686-719`); heartbeat completions skip task_complete notifications and reports; `sla_deadline` is written but never read anywhere; board fetch caps at 200 newest across ALL statuses so done rows push live tasks off the board (`use-board-tasks.ts:44`); failures buried in the Done column (no failed state).
- **P2:** 'blocked' missing from agent tool enums (`handlers_board_tasks.py:267`); blocked tasks render a blank viewer panel; unvalidated `int()` casts 500; LaneMode drag implies reassignment but only changes status; `review_mode='llm'` is a dead enum; `_auto_create_task_report` reads a nonexistent `execution_id`.

**Top adoptable ideas (research):**
- **GitHub Copilot coding agent / Linear Agents / Devin (all proprietary — patterns only):** assignment IS the execution trigger, with an ack SLA (first activity within seconds or badge "unresponsive"); Linear's typed activity feed (thought/action/elicitation/response/error) + plan-steps schema with state DERIVED from the last activity; SLA color escalation on the existing `sla_deadline`.
- **Procrastinate / PgQueuer (both MIT, Postgres-native):** `SELECT ... FOR UPDATE SKIP LOCKED` claim loop safe on ALL 4 workers (no fcntl leader needed for pickup) + LISTEN/NOTIFY instant wakeup with poll fallback — zero new infrastructure.
- **LangGraph Platform / Temporal (MIT):** lease + attempt columns with a sweeper that RE-ENQUEUES expired-lease work instead of closing it as done; terminal 'failed' only after max_attempts.
- **OpenHands (MIT — code portable):** StuckDetector heuristics (~150 lines) for in-run loop safety inside `AgentFactory.execute_with_prompt`.
- **Plane (AGPL-3.0 — UX reference only, never vendor code).**

### 2.8 Calendar Performance

**Verdict:** Not slow — deterministically empty. The schedule endpoint reads in-process APScheduler state that exists in 1 of 4 workers; the crons it should read live in the DB the whole time.

**Confirmed root causes (owner-reported):**
- 75% of calendar loads return an empty 200: `--workers 4` (`Dockerfile:140`) + fcntl leader (`main.py:445-464`) means `get_schedule` reads scheduler singletons that are None on non-leaders (`heartbeat_service.py:1559-1560`, `playbook_scheduler.py:226-227`, consumed at `activity_service.py:683-742`); React Query treats empty as success (`use-activity-api.ts:171-181`) — ~4 minutes average until a poll randomly hits the owning worker. The crons are in `agents.configuration['heartbeat']` and `workflow_templates.schedule_config` but the endpoint never reads them.
- The classic calendar already contains a `lastGoodData` workaround masking exactly this (`activity-calendar.tsx:39-45`) — the Studio tab has no such guard and blanks.
- `/api/heartbeat/workspace` is an N+1 (one query + one scheduler lookup per agent) polled every 30s by two always-mounted components (`api/heartbeat.py:370-398`; `stats-strip.tsx:32`, `calendar-tab.tsx:164`).
- `next_run_at` is null on 3/4 workers, so recurring events anchor at `new Date()` and shift on every refresh (`heartbeat.py:391-400` → `calendar-tab.tsx:223-225`).
- No fetch timeout anywhere in apiClient — hung requests wait browser-default minutes (`api-client.ts:927-930`).

**Additional issues:**
- **P1:** Shared-scheduler job contamination — first-digit job-id parsing turns `playbook_cron_42` into a phantom routine for agent 42, and the expected prefix is stale (`activity_service.py:690-750` vs `playbook_scheduler.py:116`); `range_days` is accepted and never used (`:675`); client cron expansion renders server-TZ crons in browser-local time and misses MON-FRI/step values (`calendar-tab.tsx:86-104,333-351`); frequency is a lossy string re-parsed by three divergent regex parsers.
- **P2:** Month view has no loading/empty state (`calendar-tab.tsx:520-526`); heartbeat expansion ignores active_hours/timezone; sub-hourly recipes inconsistently handled; two parallel calendar implementations consume the same endpoint with divergent parsing; auth dependency opens a second DB session + sync httpx inside async per request (`hybrid.py:681`, `clerk.py:251`).

**Top adoptable ideas:** No dedicated research stream. The fix is architectural and internal: read configs from the DB and compute `next_run_at` statelessly (croniter/CronTrigger) so all workers agree; the board stream's LISTEN/NOTIFY + DB-claim patterns apply to making schedule state worker-independent; `cron-parser` (frontend) replaces the three regex parsers.

### 2.9 Missions / Plan Mode for Auto

**Verdict:** Disciplined state machine and a robust planner; but plan mode is a tool-less prompt hack whose output is thrown away, and Auto cannot read or manage the missions it creates.

**Confirmed root causes (owner-reported):**
- Plan mode disables ALL tools and injects a markdown prompt (`consumers/chatbot/service.py:771-792`) — Auto literally cannot list agents, search memory, or query the graph while "researching" a plan. The iterated plan text is then discarded: launch re-decomposes from scratch with only the last 5 chat messages truncated to 500 chars each (`planner.py:713-721`; `mission-suggestion-card.tsx:45-48`).
- `auto_approve` exists in the backend (`coordinator_service.py:1839`) but is unreachable: no modal option, undocumented in the tool's config schema (`actions_missions.py:29-37`), and the workspace full-autonomy dial gates tool confirmation but never feeds mission approval (`platform_executor.py:451-509`).
- Auto has no lifecycle tools — no approve/reject/pause/resume/cancel/replan — and no plan-ready notification fires (`coordinator_service.py:1555,2948` cover only step/complete), so an Auto-created mission silently stalls in `awaiting_approval` until a human visits the UI.
- `platform_get_mission` is hard-broken: `int(mission_id)` on a UUID PK always raises; it also reads nonexistent `t.result` (column is `output`); the schema declares mission_id as integer (`handlers_missions.py:96,116`; `actions_missions.py:96-98`).
- Agent selection is hardcoded keyword matching (`agent_matcher.py:64-80`, 10 canonical role words); mission memory is written (`mission_memory_service.py`) but never read into planning or matching — every mission rediscovers known dead ends.
- Layered token limits: heuristic plan-time estimate becomes a hard dispatch gate (defer at 80%, pause >100%, `dispatcher.py:418-508`); upstream outputs truncated 8k/task and 30k total; Max power mode promises 50 iterations but every task dies at the 240s wall-clock timeout (`coordinator_service.py:86-90` vs `:1491-1500`, `config.py:581`).

**Additional issues:**
- **P1:** Plan-edit-on-approve is an illusion — modifications validated, stored in `run.plan['modifications']`, never applied; the frontend setter has zero call sites (`missions.py:92-118`; `coordinator_service.py:1883-1885`; `mission-store.ts:65`). The `awaiting_human` review stage is dead code (auto-completes instead, `coordinator_service.py:2882-2929`) — endpoint and HumanReviewPanel unreachable. DAG canvas fabricates edges from sequence adjacency, not real dependencies (`mission-dag-canvas.tsx:115-149`). Field document seeding reads a config key nothing writes (`coordinator_service.py:1010` vs modal sending `attachment_ids`). The list-tool state enum omits `awaiting_approval` (`actions_missions.py:65`).
- **P2:** Power-mode tooltips advertise token caps the backend removed (`create-mission-modal.tsx:43-57`); mission detail returns full outputs twice and is polled at 10s; re-run clones stale runtime config; dead boto3 import; dead `mission_mode` chat param; saved routines invisible to the template picker/planner hints.

**Top adoptable ideas:** No dedicated research stream. The memory and field streams supply the learning side (planning context pack, distillation); the board stream's Linear/Devin patterns supply lifecycle UX (plan + confidence gate, activity-derived state, notify-on-gate).

### 2.10 Document Templates (PRD-63)

**Verdict:** Clean render plumbing (Jinja2→WeasyPrint, docxtpl, S3 outputs) carrying critical security holes; authoring is three raw textareas — unusable for the target non-technical user. The locked block-editor decision is the right call.

**Confirmed root causes (owner-reported):**
- Authoring = hand-writing a complete HTML+CSS+Jinja2 document, a JSON Schema, and sample-data JSON in three monospace textareas with no preview, validation, or variable help (`template-manager.tsx:286-312`); JSON.parse failures surface as raw 'Unexpected token' toasts. The preview and DOCX-upload endpoints exist but have zero UI surface (Eye/Upload icons imported, never rendered), and there's no generate-from-template button anywhere.
- No logo/brand slots and no profile-variable resolution: `generate()` consumes only the caller's data dict (`generation_service.py:97-145`) — `User.name/avatar_url` (`core.py:980-982`) and `BusinessProfile.company_name/brands` (`business_profiles.py:38-41`) are never injected, and seeds hardcode Automatos branding (`templates/basic_report.html:8-12`, `executive_summary.html:10-41`, fallback `generation_service.py:49-63`).

**Additional issues:**
- **P0:** Cross-workspace IDOR — get/update/delete/preview never compare `template.workspace_id` to `ctx.workspace_id` (`api/document_generation.py:138-235`; `template_service.py:57-62`). Server-side template injection — user `template_content` rendered in unsandboxed `jinja2.Environment` (`generation_service.py:91,179`). WeasyPrint renders user HTML with an unrestricted url_fetcher (SSRF/local file read, `:190`). The seeded 'Meeting Notes' DOCX starter always errors (`template_file=None`, `seed_templates.py:142-147` vs `generation_service.py:217-221`). Uploaded .docx templates live on ephemeral container disk — lost on every redeploy (`api:255-261`).
- **P1:** Validation errors silently swallowed — missing required fields backfilled empty, so users and agents get blank documents reported as success (`generation_service.py:350-383`; generic 400s at `api:233,324`); versioning vestigial (never incremented, ordered-by anyway); per-workspace seed copies mean design fixes never propagate.
- **P2:** XLSX ignores templates entirely (`:248-308`); update can't clear fields and allows arbitrary attribute writes; dead `TemplateGallery.tsx` (zero importers); zero test coverage for the whole module.

**Top adoptable ideas:** No dedicated research stream — the build is defined by the locked decision: blocks JSONB (heading/paragraph/list/table/metrics/image/logo with inline variable chips) compiled to HTML→WeasyPrint for PDF and python-docx for DOCX, with `data_schema` DERIVED from the chips; a `{{user.*}}/{{company.*}}/{{brand.*}}` resolution service; `list_document_templates`/`get_template_schema` agent tools so the LLM stops guessing data shapes from two hardcoded prose examples (`tool_registry.py:1216`).

### 2.11 Integration Seams (cross-cutting sweep)

**Verdict:** Two strong spines exist — the unified tool registry/platform_execute dispatcher and the declarative context-mode system — but the bridges are mostly write-side. The "doesn't flow" feeling is the read side: outputs of one system rarely become inputs of another.

**Key missing seams (all confirmed):**
- Mission planning is sealed off from RAG, memory, and the graph: planner imports neither (`planner.py:17-66`); COORDINATOR context mode has no memory/graph section (`modes.py:125-133`); `MissionMemoryService` writes "we tried X, it failed because Y" narratives that no planning path ever reads — a write-only learning loop.
- Mission outputs die as text blobs on `task.output` (`dispatcher.py:705-707`; rendered raw with a copy button, `mission-results-panel.tsx:77-78,317-320`) — never deliverables (schema supports `source_type='mission'`, nothing writes it), never documents, never knowledge.
- Calendar shows 2 of 4 scheduling systems (`activity_service.py:675-793`): `platform_schedule_task` tasks, board SLA deadlines, and mission runs are invisible — Auto says "scheduled for tomorrow," the calendar shows nothing.
- Platform-action results have ZERO widget rendering in chat: `TOOL_WIDGET_MAP` maps only first-class tools plus stale names that no longer exist backend-side (`router.ts`); no `platform_execute` handling anywhere in frontend/.
- The knowledge graph never enriches RAG retrieval (zero graph refs in `rag/service.py:210-505`); report→graph extraction is stubbed — the trigger fires but `_incremental_build` filters `type=='document'` only, and `extract_from_report` has zero callers (`handlers_reports.py:82-88` vs `graph_service.py:1083-1086`; `graph_extraction.py:319`).
- Codegraph is an island (no platform actions, no board/mission/canvas linkage); generated documents are a parallel output system absent from the `v_workspace_outputs` view; Auto cannot enumerate deliverables at all (no `actions_deliverables.py`); field learnings evaporate at mission end; memory plays no role in agent selection; NL2SQL's injected `rag_service` is dead code (`nl2sql/service.py:86-92`).

**What works (build on it):** ContextService modes, platform_execute with semantic enum narrowing, recipe/mission→board bridges with status sync, doc-ingest→graph rebuild with debounce, workspace-file→deliverable auto-registration, mission→memory writes, chat→mission launch with PRD-125 context extraction.

### 2.12 Completeness Sweep (everything else)

**Verdict:** ~795 backend routes vs ~480 distinct frontend-called paths; 70 orphaned frontend paths and ~15 zero-consumer routers; fabricated metrics platform-wide; and a silent-mounting pattern that can drop whole API surfaces without error.

**P0 / broken:**
- StudioTicker shows fabricated "live" metrics on every page (hardcoded UPTIME 99.84%, ERR/HR 6, QUEUE 14 with a LIVE dot and an aria-label claiming live data, `studio-ticker.tsx:35-47`, mounted in `main-layout.tsx:104`).
- /agents shows hardcoded 85.5% average performance and a frozen 2025-08-01 timestamp (`api/agents.py:331-336`; mirrored in `system.py:543-544,849-852`) — and PRD-143 gives Auto these APIs, so agents will confidently report fake health.
- Workspace Explorer terminal always 404s: `workspace_exec.py` defines the router but is never imported in `main.py` (`InteractiveTerminal.tsx:79`).
- Global search silently returns zero tasks and zero agents — wrong prefixes `/api/v1/activity/feed` and `/api/v1/agents`, both caught into `return []` (`use-global-search.ts:43,59`).
- Cloud-provider document upload silently saves to local storage with no user-facing warning (`document-management.tsx:470-478`).
- Silent router mounting: ~25 routers in try/except ImportError that only logs; two imports already fail silently every boot (`main.py:115` api/auth.py, `:123` api/evaluation.py — neither file exists). A transitive error in missions.py would make the missions API vanish with no startup failure.
- Email widget rewrites all external images to a nonexistent `/api/image-proxy` (`EmailViewer.tsx:65-80`).

**P1:** Three analytics routers mounted for one concept, two dead (`analytics.py` unreachable prefix, `analytics_api.py` zero consumers, `analytics_real.py` live); router graveyard (api_playbooks, templates.py with hardcoded data, patterns, context_policy, cache, execution_history, permissions/workflow_history with broken prefixes; unmounted rag_feedback, database_knowledge_simple, workspace_exec); orphaned frontend clusters calling nonexistent endpoints (agent-coordination, execution-theater/ 7 files, performance-analytics + use-orchestration-data, WorkspaceShareDialog, TemplateGallery, use-projects, api-config.ts); api-client.ts megafile (2,700+ lines) with prod console spam and stale endpoint methods.

**P2:** context_summarization permanently 501 yet mounted; activity router imported+mounted twice; byte-identical duplicate Next stream proxy; three debug pages in the prod bundle (one throws on click, one runs fake tests); inert affordances ('not yet implemented' attach button, Coming Soon tabs, no-op analytics tracker); legacy auth redirect routes.

**Adoptable ideas:** Route-contract CI (regex-diff frontend paths vs mounted routes — the sweep ran it in under a minute) + startup assertion that every expected router actually mounted; wire StudioTicker to the already-mounted kpi_api or hide it.

### 2.13 UX Consistency Audit

**Verdict:** The shared component layer (PageHeader/FilterTabs/StatsBar/SearchInput) is genuinely good where adopted; feedback and graph layers are fragmented across parallel systems, and one critical bug silences half of all user feedback.

**P0 / broken:**
- Sonner toasts never render — only react-hot-toast's `<Toaster>` is mounted (`providers.tsx:13,91`) while 49 files call `toast()` from 'sonner' (missions, knowledge, field-theory, chat). Mission create/approve/reject, DB query failures, and graph imports all report into the void.

**P1:**
- /field-theory renders with no navigation chrome and is unreachable from any nav; /context is likewise unreachable (`app/field-theory/page.tsx:10`; `sidebar.tsx:35-125`; `studio-menu.ts:55-72`) — the page meant for judging the RAG quality the owner complains about can't be found.
- StudioPageTabs shows permanent fake counts (Outputs 41/Blogs 6/Templates 12... hardcoded in `studio-menu.ts:88-92`) AND double-renders with page-owned FilterTabs on /deliverables.
- Three toast systems coexist (react-hot-toast 17 files, sonner 49, shadcn use-toast 21 + a duplicated store impl); 4+ graph implementations across 4 rendering libraries (d3, reactflow, react-force-graph-2d, three.js) each with their own zoom/legend/palette/states — GraphErrorBoundary protects only the Business Graph.
- Two full design languages (classic glass vs Studio) ship simultaneously with bespoke shells per theme; DESIGN_SYSTEM.md predates Studio and is contradicted by it (native selects, serif fonts).

**P2:** KnowledgeGraphVisualizer hardcodes dark `#111827 !important` (broken in light theme); 94 hardcoded color violations; 7 dead components in components/documents/ (zero importers); divergent loading/empty/delete-confirm patterns (Skeleton vs Loader2 vs text vs window.confirm vs window.location.reload()); three different item-inspection patterns (Dialog/inline panel/Sheet); a11y gaps (unlabeled icon buttons, non-semantic Canvas tabs, global ESC hijack in Explorer that exits while dialogs are open); Knowledge Base IA of 17 nested surfaces with 3-level tab nesting and a duplicated RAG test surface; the 'Business Graph' tab label violates the repo's own canonical-terms table.

**Adoptable ideas:** One shared GraphView shell (toolbar/legend/palette-from-CSS-vars/error boundary) consumed by all graph surfaces; standardize on reactflow (DAGs) + one force lib and drop the rest from the bundle; pick sonner as the single toast system (most call sites) and mount its Toaster.

### 2.14 Code Canvas (Claude Code-style) — added post-synthesis

**Verdict:** The server-side substrate for a Claude Code-style loop already exists; what's missing is the session loop and the front-of-house (editor, diffs, approvals). Locked decision: embed the Claude Agent SDK headless per workspace.

**What already exists (reuse, don't rebuild — CLAUDE.md §2):**
- Agent tools registered and live: `workspace_read_file`, `workspace_write_file`, `workspace_list_dir`, `workspace_grep`, `workspace_exec`, `workspace_git`, `workspace_html_to_png`, `workspace_get_public_url` (`orchestrator/modules/tools/discovery/workspace_actions.py:19-341`) — agents can already read/edit/grep/exec/commit server-side.
- HTTP file API mounted (`api/workspace_files.py`, `main.py:1023`): GET files / files/content / files/raw, PUT files/content, POST exec — proxied to the **workspace worker container** which mounts a persistent volume (PRD-66).
- DB-backed fallback for `graph/` paths via `DbWorkspaceClient` (PRD-130) because **wizard-created workspaces have no worker container** (`workspace_files.py:33-47`).

**Gaps to the Agent SDK target:**
- P0: `CodingCanvasWidget` is a read-only viewer — `useWorkspaceFiles.ts:70,114` calls only the two GET endpoints; no write path, no diff view, no git surface, no chat binding (1,072 lines total across 6 files).
- P0: no agent-session plumbing — nothing streams a coding session's events (tool calls, diffs, approvals) to the UI; chat SSE + the widget router would be the natural carrier but `router.ts` has stale tool names and zero platform_execute handling (per 2.12).
- P1: worker-container provisioning is partial — wizard workspaces have none, so the SDK session has nowhere to run for them; provisioning (or a shared sandboxed runner) is a prerequisite.
- P1: writes are immediate with no review step — `PUT files/content` (`workspace_files.py:131`) applies directly; a diff/approve flow needs staged writes or git-based proposal branches.
- P2: the separate `workspace_exec.py` router (Interactive Terminal, PRD-66) is never mounted while `workspace_files.py` POST /exec is — consolidate to one exec surface when mounting the terminal (Q85).

**Shape of the build (one net-new Ralph PRD, outside the remediation chain):** run `claude` headless (Agent SDK) inside the worker container per workspace session; bridge its event stream over the existing SSE channel; render file tree (existing), Monaco editor + diff view (new), approval gates mapped to SDK permission prompts; commits/pushes via the existing `workspace_git` tool with platform identity. Q41 (index local workspaces in codegraph) and Q82 (codegraph as dev-tooling) feed this PRD.

---

## 3. Cross-Component Themes

### 3.1 Integration seams to build (priority order)

1. **Planning Context Pack** — RAG retrieve(goal) + mission-memory search (summaries/failures) + graph neighborhood injected into `planner.decompose` and the COORDINATOR context mode. Converts three write-only systems into one learning loop. (Feeds: missions, memory, RAG, graph.)
2. **Mission completion → deliverable assembly** — write synthesized output to workspace files + `DeliverableService(source_type='mission')`, optionally through `generate_document`. The plumbing on both ends exists.
3. **Unified calendar feed** — `get_schedule()` UNIONs heartbeats (from DB config), cron playbooks, `scheduled_tasks`, board SLA deadlines, and active mission runs; expose the same payload as a `platform_get_schedule` tool so Auto and the UI agree.
4. **Widget routing for platform actions** — thread the platform action name through `frontend_data`, add board/mission/playbook/graph widget types, fix stale names. Auto's answers then open actual platform surfaces in the canvas.
5. **Graph ⇄ RAG enrichment + report→graph wire fix** — graph-neighbor query expansion / chunk boosting in `RAGService`; one-line type-filter fix to ingest reports into the graph (`graph_service.py:1083`), calling the orphaned `extract_from_report`.
6. **Field distillation on mission end** — top-stability patterns → workspace memory (and optionally graph) before soft-archive, tagged with run_id; planner retrieves them for similar goals.
7. **Auto visibility actions** — `platform_list_deliverables`, `platform_list_document_templates`/`get_template_schema`, codegraph actions, schedule tool. Auto can currently create but not see.
8. **Memory-informed agent selection + heartbeat memory** — embedding similarity + mission-memory history blended into `agent_matcher`; memory section enabled for HEARTBEAT_AGENT mode.

### 3.2 Shared infrastructure to consolidate

- **One GraphView shell** (toolbar, collapsible legend with live counts, deterministic palette from CSS tokens, loading/empty/error + error boundary) consumed by Knowledge Graph, CodeGraph, multimodal, mission DAG, and the rebuilt field viz. Standardize: reactflow for DAGs, react-force-graph-2d for force views; delete the d3 view and the hand-rolled canvas; three.js stays only for the mission field if kept.
- **One centralized retrieval filter builder** (`build_index_filters(workspace_id, team, modality)`) that EVERY search path must pass through — search_knowledge, semantic_search, the four multimodal tools, widget docs, NL2SQL example retrieval. The Onyx pattern; makes the leak class structurally impossible.
- **One Teams source of truth** — small `teams` table backfilled from `DISTINCT agents.team` + `sdk_api_keys.team` + `unnest(documents.team_access)`, normalized at every write, exposed via `GET /api/teams` (extracted from the org-chart endpoint), feeding upload dropdowns, library filters, SDK key locks, and the RAG filter builder. Consistent with the locked reuse-agent-Teams decision.
- **Postgres claim/queue primitives** — `FOR UPDATE SKIP LOCKED` + LISTEN/NOTIFY + lease/attempt columns, shared by board pickup, the schedule dispatcher, and any future background work. Kills the fcntl-leader read-side bugs (board, calendar) in one pattern. Procrastinate/PgQueuer (MIT) if hand-rolling is unwanted.
- **One token-budgeted LLM context formatter** — numbered `<source id=N>` blocks, whole-chunk/token accumulation (never mid-structure char cuts), file:line/identity preserved, inline [n] citations mapped to artifact-viewer widgets — shared by RAG, codegraph, and NL2SQL result formatting. Replaces the 6000-char chat truncation, the 3000-char JSON cut, and the line-number-less code blocks.
- **Route-contract CI + startup mount assertions** — fail builds on new orphaned frontend paths; fail (or loudly health-report) when an expected router doesn't mount. Replaces the silent try/except pattern.
- **One toast system (sonner), one canonical calendar surface, one canonical board surface, CSS token palette for all charts/graphs** — each currently exists in 2-4 parallel copies.

### 3.3 Platform-flow improvements

- **State derived from activity, not stamped:** board tasks (and mission tasks mirrored to them) get a typed activity feed; status is derived from the last activity; the reconciler keys off last-activity liveness instead of a 300s guess; failures become a visible `failed` state instead of `done+error`.
- **Honest events:** `memory_stored` fires after persistence with the actual distilled facts; `mission_plan_ready` notification on the approval gate; ack-deadline "unresponsive" badges; no SSE/UI signal that isn't backed by a state change.
- **Deep links everywhere:** graph state in URL searchParams, citations [n] → artifact viewer widgets, mission/board/report cross-links from agent answers — the cheap way to make the platform feel like one product.
- **Kill fake data as policy:** no hardcoded metrics, counts, validation results, or mock fallbacks outside NODE_ENV=development. Every instance found is listed in 2.12/2.13.

---

## 4. Dependency-Ordered Roadmap

Structured so each workstream maps to one Ralph PRD (per the locked execution decision). Sizes: S ≈ days, M ≈ 1-2 weeks, L ≈ 2-4 weeks of focused agent work.

### Wave 0 — Quick wins (each ≤ a day, independently shippable, no dependencies)

1. Mount sonner `<Toaster>` in `providers.tsx` (unsilences 49 files of feedback).
2. Fix global-search prefixes (`use-global-search.ts:43,59`) and mount `workspace_exec` router (or remove the terminal panel pending Q85).
3. Fix `platform_get_mission`: UUID lookup, `t.output`, string mission_id in schema (`handlers_missions.py:96,116`; `actions_missions.py:96-98`); add `awaiting_approval` to the list-tool enum.
4. Fix codegraph query-log INSERT (`duration_ms` + `workspace_id`, wrap in try/except) and `ADD COLUMN IF NOT EXISTS embedding` (`codegraph_service.py:1264-1458,:837-911`).
5. Mem0: switch delete to the bulk endpoint that exists; stop counting 4xx as breaker failures; send `infer=false` for distilled facts (`mem0_client.py:498-506,:312-376`).
6. Board: numeric-CASE priority ordering (`heartbeat_service.py:919`); fix the plan/refine request shape (`board_tasks.py:927-929`); exclude orchestration mirrors from drag-launch (`:465,709`).
7. RAG: make the team filter fail CLOSED (`service.py:391-393`); fix PATCH team-access SQL (drop `title`/`updated_at`); workspace-scope analytics queries.
8. Graphs: swap `handlers_graph.py:274` to DbWorkspaceClient; clone link objects + adjacency index in BusinessGraphPanel's filter memo (fixes drill-in + filter-drop); deterministic palette from the 24-color wheel.
9. Field: read `params._agent_id` in `handlers_field.py`; bind field_id to the calling task's run; ownership check on supplied field_id.
10. Calendar stopgap: throw on `scheduler_active=false` so React Query retries immediately; `AbortSignal.timeout(15000)` in apiClient.
11. NL2SQL: migrate the three broken tabs to apiClient (revives Training instantly); honor the dialect field; return failure instead of executable error-string SELECT.
12. Auth-gate or delete the legacy entity-KG endpoints (`api/knowledge_graph.py:84-559`).
13. Strip the GitHub token from clone-error messages (`codegraph_service.py:489`).
14. Hide StudioTicker (or wire to kpi_api) and remove the 85.5%/fake-stat fields.

### WS-1 — Security & tenancy hardening (S-M) — **blocks the PRD-150 open-core cut and WS-12**
Cross-tenant and injection closures beyond Wave 0: workspace+team filters on all four multimodal search tools; workspace scoping on NL2SQL audit + analytics endpoints; doc-template IDOR checks + Jinja2 `SandboxedEnvironment` + WeasyPrint url_fetcher allowlist; token auth on the OpenMemory server; widget memory delete ownership check; unauth `GET /api/documents/content`; DISABLE the NL2SQL `query_main_database` fallback and the unauthenticated self-call (proper path rebuilt in WS-5); delete the mock `/api/v1/memory` surface (fake 200s). No dependencies — do first.

### WS-2 — RAG content & retrieval quality (M) — **blocks WS-9**
Full-chunk hydration from PG after S3 search (wire the dead `expanded_content`, one batched IN-query); centralized filter builder routed through all six search paths (implements the locked retrieval-time enforcement; optionally mirror team ACL into S3 filterable metadata with `__PUBLIC__` as pre-filter defense-in-depth); `read_document`/`grep_documents` paged tools (Letta pattern); numbered-source injection + RAGFlow citations + token-budget accumulation replacing the 6000-char cut; document pinning; perf pass (cached RAGConfig, reused backend, async access tracking); delete placebo settings + dead routers.

### WS-3 — Teams model & knowledge UX (M) — depends loosely on WS-2 (filter builder)
Teams table + `GET /api/teams` (extracted from org-chart) + normalization at every write; upload dropdowns replacing free-text chips; server-side `team=` param on document list; team-edit chips in document details + bulk action; per-team doc counts; multimodal upload persists team_access. Implements the locked Teams decision end-to-end.

### WS-4 — Memory quality & lifecycle (M) — depends on Wave 0 items 5; **blocks WS-9**
Distill-prompt rewrite on mem0-V3 rules with the operational exclusion DELETED, emitting typed `{fact, type, rating}` (Zep ontology incl. `procedure`); contradiction-based invalidation replacing the 15h decay; retrieval-as-touch reviving promotion; sleep-time consolidation pass replacing the dead jobs (message-count triggered); operational recall section + recipe/L2 namespaces in default recall and Explorer; tool-execution outcome capture as first-class memories; honest `memory_stored` event; content-hash dedup; relevance floor + ordered daily logs. Schedule the fork upgrade to mem0 V3 as the structural follow-up.

### WS-5 — NL2SQL agent path & accuracy stack (M) — depends on WS-1
In-process, workspace-scoped agent call into `DatabaseKnowledgeService.smart_query` (delete the HTTP self-call); `smart_query_database` honors database_name; audit rows written from the NL path (fills the Audit tab); persist embeddings to pgvector for few-shot retrieval; sqlglot AST validator; EXPLAIN dry-run + statement_timeout + bounded retries=2 self-correction; low-cardinality value sampling; admin-instructions v1 semantic layer (then MDL-lite); implement-or-delete Query Templates (Q19); shared source selector across tabs; sql-eval-style regression harness over verified pairs.

### WS-6 — Board execution engine (M) — no deps; **feeds WS-7, WS-9**
Assignment-triggered dispatch through the existing `_launch_task_execution`; per-worker `FOR UPDATE SKIP LOCKED` claim loop + LISTEN/NOTIFY (no fcntl leader for pickup); one real execution per task — delete the heartbeat fold-in; lease/attempt columns + sweeper that requeues instead of closing as done; terminal `failed` state + task_failed notification; typed activity feed + plan-steps schema with derived state; ack-deadline "unresponsive" badge; SLA escalation on the existing `sla_deadline`; per-agent concurrency slots + double-texting policy; OpenHands StuckDetector port; Run Now button + no-heartbeat warning; archive done > N days.

### WS-7 — Calendar & schedule truth (S) — no hard deps; pairs with WS-6
Rewrite `get_schedule` to read `agents.configuration['heartbeat']` + `workflow_templates.schedule_config` from the DB and compute next_run statelessly (identical on all workers) returning structured `{cron_expression, interval_minutes, timezone, active_hours}`; collapse the heartbeat-workspace N+1 to one DISTINCT ON query; unified feed (scheduled_tasks + SLA + missions per 3.1.3); `platform_get_schedule` tool; one shared `cron-parser` util replacing three regex parsers; month-view states; delete the losing calendar implementation (Q58).

### WS-8 — Missions lifecycle & plan mode (M) — depends on Wave 0 item 3; **blocks WS-9**
Register approve/reject/pause/resume/cancel/replan tools over existing coordinator methods; document auto_approve/power_mode/template_id in the tool config schema + modal toggle + autonomy/cost-threshold tie-in (Q61); plan mode gets read-only tools (filter dispatcher schema to permission_level=read); structured plan handoff — `plan_only` create returning the full plan + `platform_update_mission_plan` accepting the planner JSON; `mission_plan_ready` notification; dollar-ceiling budget policy replacing token-estimate pause; timeout scaling per power mode; real dependency edges in TaskResponse + DAG; implement-or-strip plan modifications; delete the dead awaiting_human path.

### WS-9 — Planning intelligence & integration seams (L) — depends on WS-2, WS-4, WS-8, WS-11
Planning Context Pack into planner + COORDINATOR mode; semantic agent matching (embeddings + mission-memory history, honoring agent_overrides); mission completion → deliverables (+ optional generate_document); field distillation at mission end; widget routing for platform actions; graph-assisted RAG + report→graph fix; Auto actions for deliverables/templates; heartbeat-agent memory.

### WS-10 — Graph consolidation: Knowledge Graph UX + CodeGraph agent surface (M) — depends on Wave 0 items 4, 8
GraphRAG community reports (titles/summaries/rank) at build time feeding sidebar + agent tools; collapsible legend with live counts; server-side subgraph endpoint (LightRAG pattern) ending full graph.json downloads; path-finding mode/tool; shared GraphView shell (consumed by WS-11 and WS-14); register codegraph's three executors as platform actions + `list_projects`/`get_symbol`; semantic routing in search_codebase + file:line context; webhook/scheduled incremental reindex honoring auto_reindex; delete the legacy entity-KG surface + dead UI controls.

### WS-11 — Field memory core (M) — depends on Wave 0 item 9; feeds WS-9 distillation
Soft-archive with `expired_at` (completed missions render frozen fields); episode provenance payloads; three-factor scoring + adaptive half-life; token-budgeted query + honest inject (truncated flag/chunking); pinned field digest in dispatch prompts + budget-gate checkpoint warning; viz rebuild on real node+edge data via the WS-10 shell; delete /field-theory page + client methods; real field health check (no more hardcoded 'healthy'); redis backend decision (Q29).

### WS-12 — Document templates block editor (L) — depends on WS-1
Blocks JSONB schema + editor (variable chips deriving data_schema); compile blocks→HTML→WeasyPrint (PDF) and blocks→python-docx (DOCX per Q71); `{{user.*}}/{{company.*}}/{{brand.*}}` resolution service + `GET /api/documents/variables`; workspace brand kit (logo upload + tokens) replacing hardcoded Automatos branding; preview pane + thumbnails; S3 template storage; strict validation in UI/preview with real errors; global shared starters with copy-on-customize; agent tools (list/get_schema, optional create); register generated docs as deliverables.

### WS-13 — Platform hygiene: dead code, route contract, honest metrics (M) — anytime; do early; **feeds WS-14**
Route-contract CI + startup mount assertions (replace silent try/except); the confirmed-kill list from 2.12 (analytics×2, api_playbooks, templates.py, patterns, context_policy, debug pages, orphaned frontend clusters, duplicate mounts/routes, dead document components, legacy field-theory remnants); api-client split + prod logging strip; real metrics or removed fields everywhere fabricated data was found; mount rag_feedback + chat thumbs (Q87); cloud-upload visible block (Q89).

### WS-14 — UX consistency & design system (M) — depends on WS-13, WS-10
Toast consolidation completion; loading/empty/destructive-confirm standardization (Skeleton + DeleteConfirmationModal, kill window.confirm/reload); CSS token palette for charts/graphs (fixes light theme); StudioPageTabs honesty (live counts or no counts; de-dup with FilterTabs per Q97); a11y pass (icon labels, Canvas tab semantics, scoped ESC); Knowledge Base IA flattening + single RAG-test surface + 'Knowledge Graph' rename; DESIGN_SYSTEM.md updated for Studio (or migration declared per Q91).

**Critical path:** Wave 0 → WS-1 → {WS-2, WS-4, WS-6, WS-8 in parallel} → WS-9. WS-5/7/10/11 slot beside the parallel block; WS-12 anytime after WS-1; WS-13/14 are continuous-improvement tracks that don't block the path. Note: the Code Canvas / Claude Agent SDK embed (locked decision) is a separate net-new PRD outside this remediation chain; Q41/Q82 feed its scoping.

---

## 5. Open Questions for the Owner

Deduplicated across all 13 streams (cross-component duplicates noted in their primary home), numbered continuously, each phrased as a decision. Defaults are recommendations, not assumptions.

### RAG / Documents & Teams

**Q1 — No-team agent visibility.** Should an agent with NO team see team-restricted documents (current PRD-124: no team = no filter = sees everything) or only public docs? Options: (a) keep fail-open, (b) fail-closed for team-restricted docs (empty `team_access` stays visible to all). **Default: (b)** — least-surprise security; teamless agents are usually generic, not privileged.

**Q2 — Teams entity shape.** Keep free-text `agents.team` strings, or add a small `teams` table backfilled from existing strings, shared by agents/documents/SDK keys? Can an agent belong to multiple teams? **Default: teams table, single-team agents for now** — consistent with the locked reuse-agent-Teams decision; multi-team agents only if a concrete need appears.

**Q3 — Human visibility on the knowledge page.** Team selector as a pure filter (humans see all), or user-level team membership restricting humans too? **Default: pure filter** — human RBAC is a separate, later decision; don't conflate with agent scoping.

**Q4 — Chat doc widgets out of agent scope.** When a team-scoped agent answers, should document widgets/links render for docs the agent itself couldn't access? Options: render / suppress. **Default: suppress** — matching the agent's scope avoids both confusion and a soft leak.

**Q5 — Default team for cloud-synced documents.** Options: per-connection team setting / inherit from syncing user / empty (all teams, today's default). **Default: per-connection setting, defaulting to empty** — one dropdown on the connection covers 90% of cases.

**Q6 — Multimodal knowledge_items fate.** Invest in team-scoping the tables/images/formulas system, or fold/freeze it into the documents/RAG path? (Its similarity ranking is broken and its tools are unscoped.) **Default: freeze + workspace-scope only (WS-1), fold later** — don't invest in a possibly-dead subsystem.

**Q7 — Widget/SDK docs schema drift.** The surface was built against `documents.title/content/updated_at` which don't exist in the repo schema — is it live in production (implying DB drift to reconcile), and should it use vector search instead of ILIKE? **Default: verify prod schema first; rebuild on the real schema + the WS-2 retrieval path.**

**Q8 — Agent read budget.** How much document content may an agent pull per turn — full docs on demand vs capped excerpts? **Default: paged read tool with a model-context-derived window (Letta pattern), ~2-4k tokens per page** — full docs reachable, never dumped.

### Memory

**Q9 — Operational memory priority.** Rank: tool-execution outcomes (channel IDs, auth quirks) / mission-task failure learnings / playbook patterns / user-business facts. **Default: tool outcomes first, mission failures second** — highest volume and highest reuse; decides where write hooks land first.

**Q10 — Chat-derived operational events.** The PRD-131d distill prompt deliberately excludes "user asked to run a mission"-type events; reverse it? **Default: yes** — include as typed `procedure/operational` facts with ratings; the exclusion is the root cause of the missing-operational-memory complaint.

**Q11 — L2 lifespan target.** Days, weeks, or "until promoted/contradicted"? Should mission/task memories decay at all? **Default: contradiction-based invalidation (no time decay); mission/task memories never decay, promote on use.**

**Q12 — mem0's role.** Keep server-side extraction (upgrade fork to V3) or demote mem0 to a dumb vector store (`infer=False` everywhere) with the orchestrator owning extraction? **Default: orchestrator owns extraction, `infer=False` everywhere now; upgrade the fork to V3 afterward for dedup/hybrid-search**, not for extraction.

**Q13 — Legacy `/api/v1/memory` mock surface.** Safe to delete, or does anything external depend on the fake responses? **Default: delete (WS-1)** — fake 200s on a real prefix are actively dangerous.

**Q14 — Widget-customer memories in Explorer.** Visible to workspace operators or isolated? **Default: isolated** — end-customer memory shouldn't co-mingle with operator memory without an explicit reveal.

**Q15 — mem0 server auth.** Is Railway-internal networking an acceptable trust boundary for multi-tenant memories? **Default: no — add token auth now (WS-1)**; the orchestrator already sends the header.

**Q16 — Per-turn memory cost budget.** Memory currently spends ~3 LLM calls/turn; auto-capturing tool outcomes adds writes. Ceiling? May the distiller use a cheaper model? **Default: one distill call/turn average on a cheap model (Haiku-class); server-side extraction calls eliminated by Q12.**

### NL2SQL / Databases

**Q17 — Agent query target.** Customer-connected DBs, the Automatos platform DB, or both? The current silent platform-DB fallback looks accidental. **Default: customer DBs via the in-process scoped path; platform-DB querying only as a separate, explicitly-gated tool if a product need exists.**

**Q18 — Multi-database per workspace now?** Every sub-tab except the Explorer is hardwired to `sources[0]`. **Default: yes, minimally** — one shared source selector (cheap); deeper multi-source features later.

**Q19 — Query Templates tab.** Implement (seed + parameterized execution through the validator) or cut? **Default: cut the mock tab now; reintroduce later as saved/parameterized golden queries** backed by the training store.

**Q20 — Dev-only `/query/sql` endpoint.** Promote to a real SQL-editor mode or remove from prod? It's currently the only audit writer. **Default: keep, admin-gated**, until WS-5 makes the NL path the audit writer.

**Q21 — Auto-train policy.** Every successful execution auto-trains today ("success" = no error, not "correct"). **Default: gate on rows>0 + confidence threshold (Vanna pattern), opt-out per source.**

**Q22 — Semantic-layer direction.** Keep building the custom metrics/dimensions model, adopt a standard (dbt metrics / cube.dev), or MDL-lite? **Default: admin-instructions v1 now, MDL-lite JSON next; no external dependency yet.**

**Q23 — Audit UX.** Show agent-initiated queries distinctly from human Explorer queries? Retention/pagination requirements? **Default: yes, distinct source field; 90-day retention with pagination.**

**Q24 — External MCP exposure of NL2SQL (PRD-61 US-016).** Still wanted? Definitions exist, nothing serves them. **Default: defer** until the in-process path is solid; then mirror the wren-engine MCP shape.

**Q25 — Dialect enum.** Snowflake/BigQuery/MSSQL appear but only postgres/mysql are implemented. **Default: shrink the enum to reality**; re-add with implementations.

### Semantic Field Memory

**Q26 — Canonical field viz home.** Resurrect the standalone /field-theory page or make the mission-detail Field tab the single home (page deleted)? (Also raised by the UX sweep.) **Default: mission Field tab; delete the page and its client methods.**

**Q27 — Field persistence after completion.** Retention window / frozen snapshot vs ephemeral-by-design? (Also raised by missions and integration streams.) **Default: soft-archive with `expired_at` + distillation into workspace memory** — render frozen fields, learn across missions, keep live queries clean.

**Q28 — Cross-mission field memory.** Should agents learn across missions via the field itself (seeding from past missions), or only via distilled workspace memory? **Default: distillation only** — keeps per-mission isolation guarantees intact.

**Q29 — Redis A/B backend (PRD-107).** Experiment still live, or delete the backend + switch? **Default: delete if concluded** — it currently silently blanks the panel.

**Q30 — Which token pain is primary?** (a) 4000-char inject truncation, (b) budget pause gates, (c) prompt bloat from upstream-output duplication. **Default: (c) first** — replace stuffing with field queries; then honest inject (a); (b) is addressed by Q65's dollar ceilings.

**Q31 — Agent selection inputs.** Use live field content (who contributed relevant patterns) or only static profiles + history? **Default: static embeddings + mission-memory history first; field-resonance as a later signal.**

**Q32 — Scalar/vector/tensor field-theory math API (PRD-68 deletion).** Permanently dead, or returning? The legacy UI's controls only make sense if it returns. **Default: permanently dead; delete the remaining UI.**

### Knowledge Graphs

**Q33 — Legacy entity KG (kb_entities endpoints + Multimodal graph tab).** Delete outright or secure? It's unauthenticated, cross-tenant, and unreachable in prod today. **Default: delete** (repo delete-what-you-replace policy).

**Q34 — Converge CodeGraph and Knowledge Graph viewers?** They duplicate filter/color/drill-in concerns on different renderers. **Default: one shared GraphView shell, separate data products** — converge chrome, not semantics.

**Q35 — >50k-node strategy.** Build cluster-first server-side drill-in now, or accept "agents-only, no viz"? **Default: build the subgraph endpoint (LightRAG pattern)** — it also kills full-file downloads for every graph size.

**Q36 — Filter legend default state + persistence.** Hidden vs visible on first load; localStorage vs workspace setting? **Default: visible but collapsed legend; localStorage.**

**Q37 — LLM spend to auto-name communities at build time?** One pass over the top ~20 clusters. **Default: yes, cheap model** — fixes "Cluster 47" for humans AND agents.

**Q38 — Agents get code-graph traversal tools** (call-graph, impact-of-change) or is business-knowledge-only intentional? **Default: yes — register the existing executors (WS-10)**; PRD-143 says Auto gets all APIs.

**Q39 — Rename 'Business Graph' tab → 'Knowledge Graph'?** The canonical-terms table says Knowledge Graph; is 'Business Graph' deliberate for the Shopify wedge? **Default: rename now** unless the wedge marketing needs it.

### CodeGraph

**Q40 — Tool-surface convention.** Dedicated registry tools (search_codebase pattern) vs platform_execute actions (platform_graph_* pattern)? Both conventions exist. **Default: platform_execute actions** for the new graph tools; keep `search_codebase` pinned as-is.

**Q41 — Index local workspaces/worktrees** (the IDE/Code Canvas direction) or stay GitHub-HTTPS-only? **Default: yes, as part of the Code Canvas SDK PRD** — agents must be able to query the code they're editing; keep the allowlist for remote clones.

**Q42 — Seed the Automatos repo for the CTO agent by default?** The hardcoded 'Automatos-ai' tool default implies yes, but nothing seeds it. **Default: no hardcoded default; per-workspace opt-in seed**; remove the literal.

**Q43 — Freshness SLA.** Reindex on push (webhook) / nightly cron / on-demand? **Default: push webhook → incremental hash-based reindex, nightly sweep as fallback.**

**Q44 — Embedding spend.** Build per-symbol content-hash embedding caching, or eat full re-embeds? **Default: build it** — trivial since file hashes already exist; the UI reindex path currently maximizes spend.

**Q45 — NL `/ask` endpoint.** Make it LLM-backed (as its docstring claims), keep regex, expose to agents? **Default: keep regex for the UI; do NOT expose to agents** — agents reason over structured tools themselves.

**Q46 — Context budget per code search result.** Today top-4 × 500 chars. **Default: top-5 with file:line + signature at ~600-800 tokens each, with `get_symbol` for full-body expansion.**

### Board / Task Pickup

**Q47 — Does assignment imply execution?** Is 'assigned' a queue the agent drains ASAP (event-driven auto-run) or a human-curated staging column with cadence pickup? **Default: assignment = execution trigger** — the universal pattern (Copilot/Linear/Devin) and the single change that resolves the complaint.

**Q48 — Pickup without heartbeats.** Dedicated pickup loop independent of heartbeats, or is enabling a heartbeat the deliberate "activation" step (and should agent creation enable a default one)? **Default: pickup independent of heartbeats (claim loop); heartbeats stay ambient/proactive only.**

**Q49 — SDK TASKS_WRITE scope (PRD-09 slice 3).** Should external/IDE agents get a narrow claim/complete-own-tasks scope, or stay read-only? **Default: yes, narrow scoped claim/complete** — consistent with the scope-gated-dependency constraint, no shared-auth redesign.

**Q50 — review_mode='llm'.** Still on the roadmap (LLM-as-reviewer before done) or remove the dead enum? **Default: remove until designed.**

**Q51 — Real 'failed' terminal state** vs the done+error_message convention reconciler/reaper rely on? **Default: add `failed`** — failures buried in Done is a trust problem; migrate reconciler/reaper to it.

**Q52 — Mission-mirror cards.** Disable drag entirely (coordinator owns status) or route drag-to-in_progress as a coordinator retry signal? **Default: disable drag on mirrors**; retry stays an explicit mission action.

**Q53 — Done-task retention.** Archive policy? **Default: exclude done > 14 days from the default board fetch; separate archive view.**

**Q54 — Per-agent concurrency.** Heartbeat batches 3; direct triggers run unbounded in parallel. **Default: per-agent slots (default 3) + 'enqueue' double-texting policy**, configurable in agent configuration.

### Calendar

**Q55 — Scheduler-down display.** Show the intended schedule from DB config with a 'scheduler offline' banner, or hide jobs that won't fire? **Default: show from DB + banner** — DB is the source of truth after WS-7.

**Q56 — Month view density.** Full recurrence expansion (hundreds of chips for hourly routines) or daily aggregates? **Default: aggregates ('24× Inbox Agent').**

**Q57 — Sub-hourly handling.** Is excluding sub-30-min heartbeats to the 24/7 band deliberate, and should sub-hourly playbooks get the same band? **Default: yes — band treatment for both.**

**Q58 — Canonical calendar surface.** Studio CalendarTab vs classic ActivityCalendar — which survives? **Default: Studio CalendarTab; delete classic after parity** (consistent with Q91).

**Q59 — Auto's schedule access vs PRD-143 obs gating.** Is the unified schedule operational data (full agent access) or observability-tier (super_admin)? **Default: operational — give `platform_get_schedule` to all agents**; it's what's-going-to-run, not system telemetry.

**Q60 — Render timezone.** Workspace tz, viewer browser tz, or per-heartbeat tz? **Default: viewer browser tz, with the source tz shown per event.**

### Missions / Plan Mode

**Q61 — Auto-approval policy.** Auto-approve ALL plans under full autonomy, or graduated gates? **Default: graduated — auto-approve under an estimated-cost threshold, human approval above it**; threshold per workspace.

**Q62 — Plan-mode product shape.** Persist iterated plans as editable draft missions (a real state in the missions UI) or stay conversational with a one-shot "launch exactly this plan"? **Default: structured draft** — `plan_only` create + `platform_update_mission_plan`; the conversational flow writes into it.

**Q63 — Can Auto self-approve under full autonomy,** or is mission-plan approval always a human gate (like the obs boundary)? **Default: Auto may self-approve below the Q61 cost threshold; above it, human-only.**

**Q64 — Semantic agent-selection mechanism.** (a) embeddings inside AgentMatcher, (b) planner LLM assigns named agents from the roster, (c) Auto assigns at plan time honored via agent_overrides. **Default: (a) now, plus implementing agent_overrides so (b)/(c) become honorable later.**

**Q65 — Which token limit is the real pain, and are dollar ceilings acceptable?** Budget pause / 8k-per-upstream truncation / 4k field cap / per-task output ceilings. **Default: replace the token-estimate pause with a workspace dollar ceiling**; fix truncations via WS-2/WS-11 budgeting.

**Q66 — awaiting_human review stage.** Delete permanently or revive as an optional per-mission setting? **Default: delete** — it's dead code and board review_mode covers the human-gate need.

**Q67 — Saved routines as mission templates.** Unify WorkflowTemplate with TEMPLATE_REGISTRY so save-as-routine output appears in the create modal and planner hints? **Default: yes, unify.**

### Document Templates

**Q68 — Brand assets home.** New workspace brand kit vs extend `BusinessProfile.brands` (already populated by the onboarding crawler)? One brand per workspace or per-template overrides? **Default: extend BusinessProfile.brands + workspace logo upload; per-template overrides allowed.**

**Q69 — Existing-template migration.** Automated HTML→blocks migration, or blocks ship net-new with raw HTML as an 'advanced mode' escape hatch? **Default: net-new blocks + advanced-mode HTML; hand-migrate the 3 seed templates.**

**Q70 — Starter template distribution.** Global/shared records (fixes propagate) with copy-on-customize, vs per-workspace copies? **Default: global + copy-on-customize.**

**Q71 — DOCX parity timing.** Blocks→python-docx at parity on day one, or PDF-first with DOCX following? **Default: PDF-first, DOCX fast-follow** — don't let DOCX gate the editor ship.

**Q72 — Variable scope + privacy.** Which chips: Clerk user (name/email/avatar), BusinessProfile, workspace metadata? Is auto-injecting user emails acceptable? **Default: name/company/brand by default; email behind an explicit per-template opt-in.**

**Q73 — Agent-driven template creation.** Is "Auto, build me an invoice template with our logo" a first-class path (template CRUD tools for agents)? **Default: yes** — cheap once the block schema exists, and it's the strongest demo of the feature.

**Q74 — Template permissions under PRD-143 tiering.** All workspace members create/edit, or owner/admin only? **Default: all members create/edit; delete admin-only.**

**Q75 — XLSX fate.** Keep in the template system (templates are no-ops for it) or reposition as a separate data-export feature? **Default: reposition as data export; block editor targets paged documents only.**

**Q76 — Document links after presigned expiry.** Is re-fetch via `/generated/{filename}` the intended permanent link, and does that meet sharing requirements? **Default: yes — make it the documented permanent link**; presigned URLs are transport detail.

### Integration / Platform Flow

**Q77 — Planning context retrieval.** Auto-consult RAG/memory/graph on every plan (latency + cost per mission) or per-mission toggle? Acceptable planning latency? **Default: on by default with a ~10s retrieval budget + opt-out toggle.**

**Q78 — Generated documents vs Deliverables.** Merge PRD-63 outputs into the Deliverables/outputs view and workspace storage, or keep deliberately separate? **Default: merge — register as deliverables** so `v_workspace_outputs` is truly the one outputs surface.

**Q79 — The 'everything scheduled' surface.** Is the command-center calendar canonical, and should board SLA deadlines + one-off scheduled tasks appear there? **Default: yes and yes** — one calendar, typed event sources.

**Q80 — Heartbeat agents' statelessness.** They're amnesiac by explicit design (`modes.py:74-87`); enable cross-run memory now that mission memories exist, accepting per-tick token cost? **Default: yes, with a capped memory section.**

**Q81 — Mission deliverable format.** Raw file registration only, or auto-format through `generate_document` (PDF/DOCX) on completion? **Default: raw registration always; formatted output as a per-mission/template option.**

**Q82 — Codegraph's product role.** Dev-tooling for Code Canvas/coding missions, or a knowledge surface converging with the business graph? Determines codegraph→missions vs codegraph→graph-merge ordering. **Default: dev-tooling first** (aligns with the locked Code Canvas SDK decision); graph convergence later if at all.

### Completeness Sweep

**Q83 — Debug pages.** Delete all three (/api-debug, /api-diagnostics, /api-control) or keep one real diagnostics page? **Default: delete api-debug + api-control; rebuild one honest diagnostics page behind super_admin** (obs tier).

**Q84 — Orphaned ambition clusters.** Agent coordination, execution theater, benchmarking: roadmap features to finish, or delete in the Wave-5-style cut? **Default: delete the orphans**; rebuild from scratch if the roadmap ever demands them.

**Q85 — Workspace Explorer terminal.** Mount `workspace_exec` with scope gating, or remove the terminal panel? **Default: mount, super_admin/scope-gated** — Code Canvas will want shell exec anyway.

**Q86 — Confirmed S6 router kills.** Treat analytics.py + analytics_api.py + api_playbooks.py + templates.py as confirmed kills (zero consumers via full api-client→hook→component trace)? **Default: yes — confirmed kills.**

**Q87 — RAG feedback loop.** Mount rag_feedback.py + add thumbs in chat now, or defer? **Default: now** — cheap, and it instruments the exact RAG quality complaints driving WS-2.

**Q88 — StudioTicker.** Hide until kpi_api wiring lands, or are pilot-snapshot numbers acceptable short-term? **Default: hide until wired** — fabricated "LIVE" data is worse than no ticker.

**Q89 — Cloud-provider document upload.** Block with a visible "saved to Automatos library" message, or prioritize building provider upload via Composio connections? **Default: block visibly now; build later** — silent wrong-destination writes are a trust killer.

### UX Consistency

**Q90 — /context page.** Fix and add to nav, fold RAG observability into a Knowledge Base tab, or delete? It duplicates the RAG test surface and is unreachable today. **Default: fold into Knowledge Base; delete the route.**

**Q91 — Studio end-state.** Command Centre/chat/assignments each maintain two full shells while most users see classic (defaultTheme='system'). Which dies, on what timeline? **Default: Studio is the end-state; delete classic per-surface as each reaches parity** — but this needs an explicit owner call; dual-shell maintenance is taxing every feature.

**Q92 — Canonical graph stack.** Consolidate to reactflow (DAGs) + react-force-graph-2d (force views), dropping d3 and the hand-rolled canvas? Changes the Business Graph feel slightly. **Default: yes — consolidate**; graphrag-visualizer proves the stack handles the workload.

**Q93 — Toast winner.** react-hot-toast (the only one mounted) or sonner (the most imported, 49 files)? **Default: sonner** — mount its Toaster (Wave 0), codemod the other two out (WS-14).

**Q94 — Templates' IA home.** Deliverables > Templates embeds components/documents/template-manager — do templates belong to Deliverables or Knowledge Base long-term? **Default: Deliverables** — templates produce outputs, not knowledge.

**Q95 — Chat canvas takeover.** The widget canvas is a full-viewport overlay hiding all navigation (`chat.tsx:828`) — keep the takeover, or move to a split view where Command Centre stays glanceable? **Default: split view (resizable panel)** — users monitoring agents mid-task shouldn't have to close their workspace.

**Q96 — Item-inspection standard.** Board uses a Dialog, missions an inline panel, deliverables a Sheet — pick one pattern? **Default: Sheet (right drawer)** for all record inspection; Dialogs reserved for confirmations/forms.

**Q97 — Sub-navigation ownership.** Shell-owned StudioPageTabs vs page-owned FilterTabs — both exist and double-render on /deliverables. **Default: page-owned FilterTabs; StudioPageTabs becomes count-free nav or is removed.**

---

*End of report. 5 sections; 13 component streams; 14 workstreams + Wave 0; 97 owner decisions.*

