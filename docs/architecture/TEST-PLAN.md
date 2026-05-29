# Automatos — Test Suite Plan

> How we make "rock solid" measurable. Turns each primitive's Definition of Done
> (`BRAIN-BLUEPRINT.md §3`, `GUARDRAILS.md §H`) into tests. Companion to the Brain
> Blueprint, Diagrams, and Guardrails.
>
> **Status:** baseline plan — 2026-05-29. **Sequencing gate:** the *implementation* of this
> plan waits until BOTH PRD-141s (Platform Reliability + Widget Vertical-Agnostic) are
> merged to main — tests written before then would be thrown away (mem0 async, chat.py
> Shopify-func deletion). This document is design; it can be written now.

---

## 1. Why this exists

Today we cannot answer "is the platform working?" with a number. Backend has real but uneven
coverage; the frontend has **1 test file across 721 TS/TSX files**. A hardening effort with no test
net is just hope. The goal of this plan is a **test net that proves each primitive meets its contract**
and a CI gate that keeps it that way.

---

## 2. Current state (honest map)

Verified 2026-05-29. ✅ = real coverage, ⚠️ = thin/partial, ❌ = none found.

| Primitive / area | State | Evidence |
|---|---|---|
| Mission coordinator / dispatcher / planner | ✅ | `test_coordinator_parallel`, `test_dispatcher_parallel`, `test_parallel_decomposition`, `test_synthesis_executor`, `test_complexity_detection`, `test_82c_wiring`, `test_budget_gate`, `test_prd108_scenarios` |
| Memory stack (L1/L2/L3, Mem0 async, circuit breaker) | ✅ | `test_unified_memory`, `test_mem0_async_client`, `test_mem0_circuit_breaker`, `test_mem0_load`, `test_hierarchical_memory`, `test_memory_section` |
| Tenancy / workspace isolation | ✅ | `test_workspaces`, `test_workspace_errors`, `test_workspace_isolation` (regression — pins the X-Workspace-ID spoof fix), `test_onboarding_journey` |
| Action registry / tool routing models | ✅ | `test_us013/014/015`, `test_action_registry_filtered`, `test_action_semantic_index`, `test_tool_router_semantic` |
| Widgets / vertical plugins / channels | ✅ | `test_registry_contract`, shopify+generic `test_widget_proactive`, `test_prd008a*`, `test_prd008a4_channel_*` |
| Field memory (Qdrant) | ✅ | `test_vector_field` |
| NL2SQL validator (security) | ⚠️ | `test_validator`, `security/test_nl2sql_validator` — validator only, not end-to-end query |
| Knowledge graph build/query | ⚠️ | `test_graph_router`, `test_eval_graph_mode` — routing, not build idempotency |
| **Universal Router tiers** (`route()` 0→3) | ❌ | no direct test |
| **AutoBrain.assess()** | ❌ | no test |
| **IntentClassifier / SmartIntentClassifier** | ❌ | no test |
| **Core tool loop** (both chat + AgentFactory loops) | ❌ | exercised only incidentally via fixtures |
| **RAG functional** (ingest→retrieve) | ❌ | `modules/rag/tests/` has only `__init__.py` |
| **MissionReconciler / VerificationService** | ❌ | no direct test |
| **Playbook/recipe execution durability** | ❌ | only `test_recipe_scheduler` (cron), nothing for execution recovery |
| **Composio executor** | ❌ | no unit test found |
| **DB connection-leak / pool** | ❌ | none (the idle-in-tx leak has no regression test) |
| **Alembic up/down** | ❌ | only PRD-specific `test_prd008a_sites_migration` |
| **Frontend (everything)** | ❌ | 1 file: `playbook-schedule-config.test.tsx`; Vitest configured, unused |

**Read:** the orchestration and tenancy spines are tested; the **reasoning entry path (router →
assess → tool loop), RAG, mission verification, playbook durability, and the entire frontend are
the holes.** Those holes map almost 1:1 onto the `BRAIN §8` gap list — fixing a gap and writing its
test are the same work item.

---

## 3. Philosophy

1. **Golden journeys first, line-coverage second.** Cover 100% of the paths that matter (the journeys
   in §5) before chasing 80% of lines. A green golden-journey suite is the real "is it working" signal.
2. **TDD for new code, characterization for old.** New features follow the global TDD rule
   (red→green→refactor, 80%+ on the new code). Existing untested code gets **characterization tests**
   that pin current behaviour first, then we refactor under green.
3. **Real dependencies for integration tests, not mocks.** Integration tests hit a real Postgres
   (`scripts/init_test_db.py`), real Redis, and a stubbed-but-faithful Mem0/S3 — mocked infra hides
   the migration/transaction bugs that bite in prod (this is why the idle-in-tx leak survived).
   Reserve mocking for external paid APIs (LLM, Composio) via recorded fixtures.
4. **Every gap fix ships with its test.** No `BRAIN §8` item is "done" without a regression test that
   would have caught it (e.g. a connection-leak test for G1, a restart-recovery test for G4).
5. **Coverage targets are per-tier, not global.** 80% on touched/critical-path code; the long tail of
   103 routers reaches coverage opportunistically, not in one sprint.

---

## 4. The pyramid (per primitive)

For each primitive: **unit** (pure logic), **integration** (real DB/Redis, cross-module), **contract**
(stable interface shape), **e2e** (in §5).

### 4.1 Chat / reasoning entry  *(biggest hole)*
- **Unit:** `AutoBrain.assess()` verdict table (ATOM→ORGANISM × RESPOND/DELEGATE/MISSION); cache→regex→LLM tier selection.
- **Unit:** `UniversalRouter.route()` each tier in isolation (0 override, 1 cache hit, 2a rule, 2b trigger, 2.5 semantic, 2c keyword, 3 LLM fallback) — assert the *right tier fires* for crafted inputs.
- **Unit:** `IntentClassifier` vs `SmartIntentClassifier` — distinct expectations, no cross-contamination.
- **Integration:** one tool loop turn — message → router → runtime → `UnifiedToolExecutor` → result fed back → streamed. Assert tool events aren't dropped. **Run against both loops** (chat `_run_tool_loop` and AgentFactory `execute_with_prompt`) until they're unified (G6).
- **Contract:** SSE chunk shape stays AI-SDK compatible.

### 4.2 Memory
- **Unit:** Ebbinghaus decay math; L1 summary roll; ContextRouter layer selection.
- **Integration:** write-after-turn writes exactly once per layer (catches the dual-write G12); read-before-turn respects the budget cap; Mem0-down path uses the circuit breaker and degrades (already covered — keep).
- **Contract:** `ContextBundle` shape.

### 4.3 RAG  *(no functional test today)*
- **Unit:** semantic chunker boundaries; RRF/rerank ordering; knapsack budget.
- **Integration:** ingest a doc → Postgres row + S3 object + S3 Vectors entry all created; retrieve returns it; **delete the doc → vector removed** (no orphan, RAG DoD).
- **Contract:** `external_file_id == documents.id` linkage holds.

### 4.4 NL2SQL
- **Unit:** validator rewrite/read-only enforcement (keep existing).
- **Integration:** question → generate → validate → execute against a seeded test DB → self-correct loop on a deliberately bad gen (≤2 retries) → training example saved. Assert **no unvalidated SQL ever executes** and creds never logged.

### 4.5 Knowledge graph (the moat — test it like it matters)
- **Unit:** `map_shopify_catalog` / `map_orders_to_fbt` edge generation from fixture orders.
- **Integration:** build from sources → persist to S3 JSON → `load_graph()` round-trips; **rebuild is idempotent**; FBT/collection/vendor edges queryable; graph survives a reload (G11 — single source of truth).

### 4.6 Missions
- **Unit:** planner DAG validation/retry; `AgentMatcher` role→agent; `deterministic_checks`.
- **Integration:** full tick lifecycle on a real DB: create → plan → approve → dispatch → complete → reconcile → verify → deliverable. **Restart durability test:** kill and re-create the coordinator mid-run; assert it resumes from DB state (this is the proof Mission Zero P1 is closed). **Stall test:** ASSIGNED>60s / RUNNING>300s → re-dispatch. **Retry-with-critique test** once G5 is fixed.
- **Contract:** board bridge mirrors `OrchestrationTask` ↔ `BoardTask` correctly (`verified`→`done`).

### 4.7 Playbooks  *(durability hole — G4)*
- **Integration:** schedule → run → complete (keep `test_recipe_scheduler`). **New, gating: restart-recovery** — start a recipe, kill the process, restart; assert a stuck `running`/`pending` `RecipeExecution` is recovered or failed-cleanly, not orphaned. This test must exist *before* G4 is called fixed.

### 4.8 Channels
- **Contract (parametrized over all 11 adapters):** each implements `_to_envelope()` → `handle_message()` → `send_message()`; inbound builds a valid `RequestEnvelope`; activity counters update.
- **Integration:** one adapter (slack) full round-trip against a stub transport.

### 4.9 Cross-cutting (infra)
- **DB:** connection-leak/pool test — assert `get_db()` leaves no idle-in-transaction connection after a request that hydrates an `Agent` across an `await` (regression for G1).
- **Migrations:** `alembic upgrade head` then `downgrade base` on a scratch DB in CI; assert `lock_timeout`/`statement_timeout` set (regression for G2).
- **Config:** CI grep gate — zero `os.getenv` outside `config.py` (regression for G7).
- **Authz:** unit tests proving `_check_agent_permission` / `validate_composio_action` **deny** on error (regression for G3).
- **Bare-except gate:** count must not increase (PRD-141 Phase 0 sets this up).

---

## 5. Golden-journey backbone (the e2e suite)

These are the user-visible flows that must never break. Each is one e2e test (backend API-level +
frontend Playwright where there's UI). This list *is* the "is it working" definition.

| # | Journey | Spans |
|---|---|---|
| J1 | **Signup → onboarding wizard → configured workspace** | Clerk → provisioning → VOYAGER/BLUEPRINT/SCRIBE/FORGE → Mission Zero |
| J2 | **Chat → agent → tool call → response** | router → runtime → UnifiedToolExecutor → SSE |
| J3 | **Widget message → vertical plugin → response** | widget auth → PLUGIN_REGISTRY → generic core → SSE (run for both `generic` and `shopify`) |
| J4 | **Mission: create → plan → approve → execute → verify → deliverable** | coordinator full lifecycle + restart durability |
| J5 | **Doc upload → RAG index → retrieval surfaces in chat** | ingest → S3 Vectors → retrieve-in-turn |
| J6 | **Marketplace install → cascade dependencies → agent usable** | install → `cascade_agent_dependencies` → runtime |
| J7 | **Playbook: schedule → run → complete (and recover after restart)** | scheduler → executor → durability |
| J8 | **NL2SQL: connect DB → ask question → validated SQL → answer** | source → generate → validate → execute |
| J9 | **Shopify sync → knowledge graph built → FBT proactive opener** | catalog sync → graph → widget opener (the moat, end-to-end) |
| J10 | **Cross-workspace isolation** | two workspaces, prove no data bleed across J2–J9 |

J9 and J10 are the two that most directly protect the strategy: J9 proves the moat works end-to-end;
J10 proves the multi-tenant promise.

---

## 6. Frontend strategy (from 1 test to a real net)

- **Tooling:** Vitest (already configured, unused) for unit/component; **Playwright** for e2e (J1–J7
  have UI). React Testing Library for components.
- **First targets (highest leverage):**
  1. `lib/api-client.ts` — it carries auth + `X-Workspace-ID` on every call; a bug here is platform-wide. Unit-test header injection, error envelope handling, retry.
  2. The ~5 most-used `use-*-api.ts` hooks (chat, missions, deliverables, marketplace, workspace).
  3. The onboarding wizard (J1) — multi-step, SSE, the first thing every user sees.
  4. Marketplace install flow (J6).
- **Contract tests** against the backend OpenAPI so frontend types can't silently drift from the API.
- **Target:** every valued surface (Command Center, Activity, kanban, analytics, widgets) has at least
  a smoke test that it renders and its primary action fires.

---

## 7. Test infrastructure

- **DB:** `scripts/init_test_db.py` → ephemeral Postgres per CI run; transactional rollback per test
  for speed; a small seed (`seed_onboarding_agents` + fixture workspace/agents).
- **External APIs:** recorded fixtures for LLM and Composio (deterministic, no spend). A nightly job
  may run a small live-smoke subset.
- **Layout:** keep `orchestrator/tests/` (backend) and add `frontend/**/__tests__` + `e2e/`
  (Playwright). Mirror the primitive structure so coverage gaps are obvious.
- **CI gates (blocking):** golden-journey suite green; no new bare excepts; no `os.getenv` outside
  config; alembic up/down passes; coverage on touched files ≥ 80%; the Shopify-in-generic gate
  (existing).
- **Coverage reporting:** publish per-primitive coverage so the dashboard (§9) can show it.

---

## 8. Sequencing

Implementation is **gated on both PRD-141s merging**. Then, in Rock-Solid Wave order:

1. **Wave 0 (parallel, safe now):** finish this plan + the dashboard spec; stand up the e2e harness
   skeleton (no assertions yet) so journeys can be filled in fast.
2. **Wave 2 — Test net (the missing thing):** write J1–J10 golden journeys + the §4 integration tests
   for the untested primitives (router/assess/tool-loop, RAG, reconciler, playbook durability). This
   is where the bulk of the work is.
3. **Per-primitive (Wave 3):** fill unit + contract tests to the DoD as each primitive is hardened;
   each `BRAIN §8` gap fix lands with its regression test.
4. **Frontend:** §6, sequenced with the surfaces being hardened.

Order within Wave 2: do the tests that double as gap regressions first (G1 leak, G4 durability,
G3 fail-closed) — they protect data and reliability, and they're cheap relative to their blast radius.

---

## 9. Tie to "Is it working?"

Wave 0 of the Rock-Solid plan builds one dashboard answering the founding question. This test plan
feeds it: every golden journey (§5) emits a pass/fail signal, and per-primitive coverage (§7) becomes
a tile. The dashboard's "mission success rate," "per-primitive health," and "error rate by subsystem"
are the production mirror of this suite. Tests prove it works in CI; the dashboard proves it works in
prod. Same contracts, two surfaces.

---

## 10. What this plan is NOT

- Not a mandate to hit 80% on all ~919 backend + 721 frontend files immediately — that's the long
  tail, reached opportunistically.
- Not a license to write tests against soon-to-change surfaces before the 141s merge (the gate).
- Not a replacement for the dashboard — tests are pre-prod proof; the dashboard is live proof.
