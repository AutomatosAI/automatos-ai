# Automatos — Brain Blueprint

> The architecture constitution. How the platform *should* work, as if written from
> scratch, with a gap analysis against today's code. Every future PRD checks itself
> against this document. Companion docs: `DIAGRAMS.md`, `GUARDRAILS.md`, `TEST-PLAN.md`.
>
> **Status:** baseline draft — 2026-05-29. Verified against source at branch
> `feat/widget-page-context-on-regular-chat`. Supersedes nothing yet; this is the
> reference the Rock-Solid PRD will be cut from.

---

## 0. North star

Automatos is an **operating system for AI agents that can run a business**. A user
brings their business (its data, tools, and goals); Automatos gives them a workforce of
agents that perceive (memory + RAG + graph), reason (router + agent runtime), act (tools
+ channels), and coordinate (missions + playbooks) — all isolated per workspace.

Two things must stay true at all times:

1. **The core is vertical-agnostic.** It can, on paper, run *any* business. Verticals
   (Shopify first) are folder-isolated plugins, never baked into the core. This is the
   "can run any business" promise — losing it collapses the product into a Shopify app.
2. **The moat is the knowledge graph + business memory**, not the widgets. Widgets are
   the distribution wedge (Shopify App Store → 4.6M stores). The defensible asset is the
   compounding model of a merchant's business — products, orders, FBT relationships, how
   their shipping/refunds actually work, past campaigns, learned playbooks. That can't be
   exported to a competitor. Lead with it.

If a change makes the core know about Shopify, or makes the moat shallower, it is wrong
regardless of how much it helps one demo.

---

## 1. The Arc — one request, end to end

Every interaction, whatever the surface, runs the same loop. This is the brain.

```
INPUT  →  CONTEXT  →  ASSESS  →  ROUTE  →  RUNTIME  →  TOOL LOOP  →  PERSIST  →  OUTPUT
(chat/    (Request-   (Auto-    (Univ.   (Agent-     (Unified-     (memory +   (deliverable
 channel/  Context:   Brain:    Router:  Factory     ToolExec:     graph       .md → S3,
 widget/   tenancy    complex-  tiers    builds      prefix        write)      stream to
 mission/  spine)     ity)      0→3)     runtime)    dispatch)                 surface)
 schedule)
```

**Canonical happy path (chat):**

1. `POST /api/chat` → `stream_chat()` (`orchestrator/api/chat.py`). `get_request_context_hybrid`
   injects a **`RequestContext`** (workspace_id, user, auth_type) — the tenancy spine that
   threads through every layer below.
2. Message persisted; history loaded; `StreamingChatService` constructed bound to the workspace.
3. **Assess** — `AutoBrain.assess()` (`consumers/chatbot/auto.py`) returns a complexity verdict
   (ATOM→ORGANISM) and an action: `RESPOND` / `DELEGATE` / `MISSION`. 3-tier: cache → regex → LLM.
4. **Route** — for DELEGATE/MISSION, `UniversalRouter.route()` (`core/routing/engine.py`) resolves an
   agent through ordered tiers: `0` override → `1` Redis cache → `2a` rules → `2b` trigger →
   `2.5` semantic embeddings → `2c` keyword `IntentClassifier` → `3` LLM fallback.
5. **Runtime** — `AgentFactory.activate_agent()` (`modules/agents/factory/agent_factory.py`) builds an
   `AgentRuntime`: resolves the LLM provider/key (BYOK → platform → env), loads Composio apps,
   attaches a `UnifiedToolExecutor`.
6. **Tool loop** — tools assembled via `get_tools_for_agent()`; the loop calls the LLM with tool
   schemas, dispatches each tool call through `ToolRouter.execute_and_format()` →
   `UnifiedToolExecutor.execute_tool()` (`modules/tools/execution/unified_executor.py`), which routes
   **by name-prefix**: `platform_*` → ActionRegistry, `workspace_*` → WorkspaceClient,
   `composio_*` → ComposioToolExecutor, file/shell → exec modules.
7. **Persist** — after the turn, memory is written (5-layer stack, §4.2) and, where relevant, the
   knowledge graph updated.
8. **Output** — final text streamed back as AI-SDK chunks; any deliverable is written as a markdown
   file to S3 (§4.6). The assistant message is persisted.

The same loop serves channels (adapter → `RequestEnvelope` → router → runtime), widgets (plugin
dispatch → `ChatService`), and missions (coordinator drives the runtime per task). **One loop, many
front doors.** Keeping it that way is the single most important structural rule.

---

## 2. The platform in layers

Read bottom-up; each layer may only depend on the ones below it.

| Layer | Name | What lives here | Key entry points |
|---|---|---|---|
| **L7** | Ops | Observability, voice | `core/monitoring/*`, `voice-service`, `voice-pipeline` |
| **L6** | SaaS surface | Marketplace, onboarding, tenancy admin, Next.js frontend | `api/marketplace.py`, wizard, `frontend/lib/api-client.ts` |
| **L5** | Reach | Channels (11 adapters), widgets, integrations/Composio, vertical plugins | `channels/`, `api/widgets/`, `integrations/`, `core/composio/` |
| **L4** | Orchestration | Missions, playbooks, tasks, scheduler | `services/coordinator_service.py`, `api/recipe_executor.py` |
| **L3** | Knowledge | Memory stack, RAG, knowledge graph, NL2SQL | `modules/memory/`, `modules/rag/`, `modules/knowledge/`, `modules/nl2sql/` |
| **L2** | Cognitive core | Router, AgentFactory, tool executor, ActionRegistry | `core/routing/`, `modules/agents/factory/`, `modules/tools/` |
| **L1** | Core services | Config, auth/RequestContext, DB sessions, LLM manager | `config.py`, `core/auth/`, `core/database/`, `core/llm/` |
| **L0** | Substrate | Postgres, S3, S3 Vectors, Qdrant, Redis, Mem0 | (external stores) |

**Dependency rule:** L2 must never import L4/L5/L6. The cognitive core does not know what a Shopify
widget is, what a marketplace is, or what a mission's UI looks like. It knows agents, tools, context.
Today this mostly holds (the widget CI gate enforces it for generic widget code); §8 lists where it leaks.

---

## 3. The eight primitives — contracts

Each primitive is defined by a **contract**: what it owns, what it depends on, the interface others
use, and an ideal **definition of done** (DoD). The DoD is what "rock solid" means for that primitive —
it's the acceptance bar the test plan asserts.

### 3.1 Chat
- **Owns:** `chats`, `messages`; the streaming response contract (AI-SDK chunks).
- **Depends on:** router (agent selection), runtime (execution), memory (context).
- **Interface:** `POST /api/chat` → SSE stream. One endpoint, one streaming format.
- **DoD:** every turn is assessed, routed deterministically where a rule exists, streamed without
  dropping tool events, and persisted. No turn silently fails; errors surface to the user.

### 3.2 Memory
- **Owns:** the 5-layer stack (§4.2): L1 Redis session, L2 Postgres short-term (Ebbinghaus decay),
  L3 Mem0 long-term facts.
- **Depends on:** Mem0 service, Redis, Postgres.
- **Interface:** `UnifiedMemoryService.store_transcript()` (write-after-turn),
  `retrieve_context()` (read-before-turn) → budget-capped `ContextBundle`.
- **DoD:** exactly one write path per layer (no double-store, no write-only); reads are
  budget-bounded and degrade gracefully when Mem0 is down (circuit breaker, already built).

### 3.3 RAG / Knowledge Base
- **Owns:** `documents`, `document_chunks`; S3 raw docs; S3 Vectors index (`documents-index`, 2048-dim).
- **Depends on:** S3, S3 Vectors, embeddings, chunker.
- **Interface:** `RAGService.retrieve()` (exposed to agents as a tool, not pre-fetched).
- **DoD:** ingest → chunk → embed → index is atomic per document; retrieval is reranked and
  parent-expanded; a document deleted from Postgres is removed from the index (no orphan vectors).

### 3.4 NL2SQL
- **Owns:** `database_knowledge_sources`, `nl2sql_training_examples`, schema metadata.
- **Depends on:** the user's connected DB (creds decrypted at query time), LLM.
- **Interface:** `DatabaseKnowledgeService.query_database()` → generate → **validate** → execute →
  self-correct (≤2 retries) → auto-save training example.
- **DoD:** generated SQL is always validated/rewritten before execution (read-only enforcement);
  no query runs unvalidated; creds never logged.

### 3.5 Knowledge Graph (the moat)
- **Owns:** the workspace graph (NetworkX → S3 JSON), Shopify entity edges (`frequently_bought_with`),
  `document_relationships`.
- **Depends on:** sources (Shopify sync, documents), LLM/deterministic extraction.
- **Interface:** `GraphifyService.build_graph()` / `load_graph()` (BFS/DFS, `subgraph_to_text`).
- **DoD:** rebuildable from sources idempotently; FBT/collection/vendor edges queryable for proactive
  openers; graph state survives restart (persisted, not in-memory only).

### 3.6 Missions
- **Owns:** `orchestration_runs`, `orchestration_tasks`, `orchestration_task_dependencies`,
  `orchestration_events`, `orchestration_archive`; mirrored to `board_tasks` via the board bridge.
- **Depends on:** agents (runtime), tools, memory, the shared APScheduler.
- **Interface:** `CoordinatorService` 5s tick: planner → dispatcher → reconciler → verifier.
- **DoD:** **DB-authoritative and restart-durable** (already true — §8.2); stalled tasks re-dispatched;
  retries feed verifier critique back into the next attempt (NOT yet true — gap).

### 3.7 Playbooks
- **Owns:** `workflow_templates` (steps), `recipe_executions`.
- **Depends on:** agents, scheduler.
- **Interface:** `handlers_playbooks` + `api/recipe_executor.py`; one skill per step.
- **DoD:** an execution survives a process restart (NOT true today — fire-and-forget, §8.2);
  canonical noun is **Playbook**, not Recipe (massively violated today, §8.3).

### 3.8 Channels
- **Owns:** `channel_connections` (creds, mode, activity).
- **Depends on:** router, runtime.
- **Interface:** `BaseChannelAdapter` (`_to_envelope()` → `handle_message()` → `send_message()`).
  11 adapters: telegram, slack, discord, teams, google_chat, signal, imessage, irc, matrix, line, whatsapp.
- **DoD:** every adapter implements the same contract; a new channel is a new adapter file, zero core
  changes; inbound/outbound counts tracked.

---

## 4. The connective tissue — how everything connects

This is the part the user asked for: *how tools, agents, graphs, files all connect.*

### 4.1 Agents
Agents are **database rows** (`agents` table), not files. `AgentFactory` builds an `AgentRuntime` from
the row at activation: system prompt (assembled by `ContextService`), tool set, LLM provider/key
(BYOK → platform → env), Composio apps. Runtime is cached in-process (`active_agents`) but
**`workspace_id` is passed per-execution** (the cross-tenant fix), not read off the cache.

### 4.2 Memory (5-layer)
- **L1 Redis** — rolling session summary (`mem:session:*`).
- **L2 Postgres** — verbatim short-term (`memory_short_term`), Ebbinghaus decay, hourly consolidation.
- **L3 Mem0** — extracted long-term facts, namespaced `mem:{ws}[:agent:{id}]`, async httpx client with
  per-workspace circuit breaker (PRD-141), 5-min Redis-cached search.
- **Write** = `store_transcript()` after a turn. **Read** = `retrieve_context()` → `ContextRouter`
  selects layers → budget-capped `ContextBundle` injected via `MemorySection`.

### 4.3 Tools
The **3-file registration pattern** is the only sanctioned extension point. Tools surface to agents via
`get_tools_for_agent()`, are described by the `ActionRegistry` singleton (platform actions →
`platform_execute` schema), and **all execute through one `UnifiedToolExecutor`** that dispatches by
name-prefix. Platform / Composio / workspace / file / shell are branches of that one executor. This is
the single most reused contract in the system — extend it, never fork it.

### 4.4 Graph
Built from sources (Shopify catalog via `map_shopify_catalog`, documents) → extraction → NetworkX →
clustered/scored → **exported as JSON to workspace files in S3** + an LRU cache. Queried via
`load_graph()` with BFS/DFS and `subgraph_to_text` for injection into prompts. The Shopify plugin walks
FBT/collection/vendor edges to generate proactive widget openers — the moat made visible.

### 4.5 Tenancy spine
`workspace_id` originates in `RequestContext` (from the validated `X-Workspace-ID` header / Clerk JWT)
and threads through router → runtime → tools → memory → graph → output. Server-side membership
validation (`_user_has_workspace_access`) blocks header spoofing. **Nothing below L1 fabricates a
workspace_id**; it is always carried, never guessed (except a dev-only env fallback that must die — §8).

### 4.6 Files / deliverables
Every agent-produced artifact is a **markdown file in S3** (`workspaces/{ws}/generated-documents/*.md`,
agent reports under `reports/`). **Rendering is the consumer's job**, never the API's or a content
transformer's. The file is the source of truth; the UI renders it. This keeps output portable and the
backend dumb about presentation.

### 4.7 Vertical plugins
The generic core dispatches verticals through `integrations/__init__.py` `PLUGIN_REGISTRY` keyed on
`workspaces.settings.vertical`. A plugin implements the `WidgetPlugin` protocol
(`handle_widget_message(...) → WidgetPluginResult`). Today: `generic` + `shopify`. A CI gate
(`scripts/ci/check-no-shopify-in-generic.sh`) keeps Shopify identifiers out of generic widget code.
New vertical = new folder under `integrations/`, register in the map, zero core edits.

---

## 5. State — where everything lives

Single source of truth per concern (full map in `DIAGRAMS.md §7`):

- **Postgres** — system of record. 116 tables across ~12 domains (agents, tenancy, missions,
  workflows, memory, documents/RAG, tools/composio, marketplace, channels/widgets, credentials,
  telemetry, config).
- **S3** — markdown documents + deliverables + marketplace bundles + harness changelog.
- **S3 Vectors** — document RAG embeddings (`documents-index`).
- **Qdrant** — `field_memory` (mission-scoped agent knowledge sharing, PRD-108 single collection).
- **Redis** — session memory, cache, task queue, rate-limit, workflow pub/sub.
- **Mem0** — external long-term memory service.

---

## 6. The ideal request, restated as invariants

These are the load-bearing truths the rest of the constitution protects:

1. **One cognitive loop**, many front doors (§1).
2. **One tool executor**, prefix-dispatched (§4.3).
3. **DB is the system of record**; in-memory state is a cache, never the truth (§5).
4. **workspace_id is carried, never guessed** (§4.5).
5. **Deliverables are markdown in S3; consumers render** (§4.6).
6. **The core is vertical-agnostic; verticals are plugins** (§4.7).
7. **The moat is the graph + memory**, not the surface (§0).
8. **No user-visible work is fire-and-forget** — it survives a restart (§3.6/3.7).

---

## 7. Vertical-agnostic core + plugin boundary (detail)

The PRD-141 widget refactor **fully landed** on the current branch and is the reference
implementation for how every future vertical integrates:

- `chat.py` (widget entry) contains **zero** Shopify identifiers; it resolves the vertical from
  workspace settings and dispatches through `PLUGIN_REGISTRY`.
- Shopify logic (proactive openers, graph-related products) lives entirely in
  `integrations/shopify/`.
- The contract is `WidgetPluginResult` (rewritten message + optional system preamble).
- **Known incompleteness:** `api/shopify.py` (catalog sync) is deliberately excluded from the CI gate
  as a "pre-PRD-141 surface" — it still lives outside `integrations/`. Rehousing it is the next
  vertical-isolation step, not done yet.

---

## 8. Gap analysis — ideal vs today (the PRD backlog)

Severity: **P0** breaks the invariants / data-safety; **P1** reliability; **P2** consistency/debt;
**P3** latent risk / documentation only. This table is the bridge to PRD-142 — each row is a
candidate work item. (Verdicts reconciled against PRD-142's code-verified review, 2026-05-29.)

| # | Sev | Gap | Where | Invariant violated |
|---|---|---|---|---|
| G1 | **P0** | `get_db()` closes but never commits/rolls back → SELECT-hydrated `Agent` held across `await` leaves a 9hr "idle in transaction" that blocks DDL/migrations | `core/database/database.py:105` | DB integrity |
| G2 | **P0** | Migrations wrap ALL versions in one transaction, no `lock_timeout`/`statement_timeout`; `env.py` reads `DATABASE_URL` raw → idle-in-tx hard-blocks any online deploy | `alembic/env.py:31` | Migration safety |
| G3 | **P0** | Fail-open security: `_check_agent_permission()` and `validate_composio_action()` return `True` on error/unknown ("fail open for now") | `agent_factory.py`, composio | Authz |
| G4 | **P1** | **Playbooks/recipes are not durable** — `launch_recipe_task` does genuine `asyncio.create_task` with an in-process dict; a stuck execution has no startup recovery. **The one REWRITE** (PRD-142 §6): consolidate the triplet onto the mission durability model; execution-launch blast radius is **6 call sites** (code-verified) behind 2 entry points — see `PLAYBOOK-ENGINE-DESIGN.md` | `api/recipe_executor.py:903`, `api/workflow_recipes.py:905` | Inv. 8 (no fire-and-forget) |
| G5 | **P1** | Retry without critique — dispatcher re-queues failed tasks storing only `failure_detail`; verifier feedback not injected into the next attempt | `modules/coordination/dispatcher.py:733` | Mission DoD |
| G6 | **P1** | Two parallel tool loops (chat `_run_tool_loop` vs AgentFactory `execute_with_prompt`) — converge on one executor but dedup/retry/truncation logic drifts | `consumers/chatbot/service.py`, `agent_factory.py` | Inv. 1 |
| G7 | **P1** | `os.getenv`/`os.environ` outside `config.py`: ~20 calls across 8 runtime files (worst: whole `core/monitoring/` module) + `database.py` `load_dotenv()` at import | grep set | Config discipline |
| G8 | **P1** | Pervasive bare `except Exception:` then continue/pass (router Tier 2.5, decision logging, `_load_agent_tools`, api_tracking) — masks failures | multiple | Error handling |
| G9 | **P2** | **Recipe naming** — ~1,682 "recipe" references; Playbook is skin-deep (model `WorkflowTemplate`, executions `RecipeExecution`, `api/recipe_executor.py`). Canonical-term drift at scale | repo-wide | Canonical terms |
| G10 | **P2** | Verification is advisory-only (PRD-103) but reconciler docstring still claims a `VERIFIED/RETRYING/FAILED` verdict path — doc/code drift | `modules/coordination/reconciler.py` | Consistency |
| G11 | ~~P2~~ → **P3** | **Downgraded.** The moat is *already* single-sourced in `workspace_graphs` (Postgres, not S3) via `GraphifyService`; Shopify FBT writes there and **nowhere else**, with **zero** dual-write to `knowledge_nodes/edges` (code-verified). Risk is *latent*, not live. Fix = a documented/enforced boundary + a named storage-format scale path (JSON blob → queryable edge tables) — see `KNOWLEDGE-GRAPH-CANONICAL.md` | `modules/knowledge/graph_service.py`, `core/graph_storage.py` | Graph DoD |
| G12 | **P2** | Dual L3 memory write paths (`store_exchange` skips Mem0, smart_memory writes it) → double-store / miss risk; L1 summary is naive `[-500:]` truncation | `modules/memory/` | Memory DoD |
| G13 | **CUT** | `core/neural_field/` (PRD-59) is a **dead twin** of the live `vector_field.py` (PRD-108) — its **only** importer is the workflow-era `AgentExecutionManager`, which is **never instantiated** (its `AgentService.execution_manager` builder property is never read) and **has zero callers** (code-verified 2026-05-29). Superseded by `AgentFactory`. Cut as one unit with `execution_manager`. On the PRD-142 §11 cut list | `core/neural_field/` + `AgentExecutionManager` | Clarity / dead weight |
| G14 | **P2** | Widget conversation owner hardcoded to `users.id=1` | `api/widgets/chat.py:89` | Tenancy hygiene |
| G15 | **P2** | `api/shopify.py` catalog sync still outside `integrations/` (excluded from CI gate) | `api/shopify.py` | Inv. 6 |
| G16 | **P2** | Dev-only `WORKSPACE_ID`/`DEFAULT_WORKSPACE_ID` env fallback in `hybrid.py` — historical cross-tenant leak source | `core/auth/hybrid.py` | Inv. 4 |
| G17 | **P0(test)** | **Testing void** — 1 frontend test across 721 TS/TSX files; no functional RAG test; router tiers / AutoBrain / tool loop / reconciler / recipe-durability untested | repo-wide | "Rock solid" is unmeasurable |

**Corrections to prior assumptions (worth noting):** missions are *already* restart-durable (Mission
Zero P1 is stale for missions; the durability gap is in playbooks, G4). Tenancy is *already* handled
correctly per-execution (the residual risk is the env fallback, G16). The widget refactor is *done*, not
half-landed. The **moat is already single-sourced** (G11 downgraded P2→P3 — the dual-store fear was
latent, not live); `core/neural_field/` is **dead, not just mis-named** (G13 → cut; its only consumer is the never-instantiated workflow-era `AgentExecutionManager`).

---

## 9. How this doc is used

- **Before any PRD:** classify the change against §2 layers and §6 invariants. If it violates one,
  redesign before estimating.
- **During build:** `GUARDRAILS.md` is the enforceable checklist derived from §6.
- **For "rock solid":** `TEST-PLAN.md` turns each §3 DoD into golden-journey tests; §8 is the backlog.
- **For the pitch:** §0 (moat) and the corrected scale facts in §5 are the receipts.
