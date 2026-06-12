# PRD-154 — Wave-0 Quick Wins

**Chain:** Night 1, first. Branch `ralph/prd-154-wave0-quick-wins` from `main`. Size **S** (12 stories, mostly one-to-three-file fixes).
**Source:** review report §4 Wave 0; evidence in §2.1–2.14.

## Overview

Fourteen verified breakages whose fixes are small and independent. Together they resolve a disproportionate share of the owner-reported complaints: agents reading 500-char crumbs, board tasks never picked up, the calendar's minutes-long empty loads, the field graph that never rendered, Auto's broken mission tools, and UI feedback that silently never renders. Every fix follows repo rules: no shims, delete what's replaced, no hardcoded values.

## Goals

- Every owner complaint with a one-file root cause is fixed and regression-tested.
- Zero fabricated data shown to pilot users (D10).
- No new tables, no new tools, no architecture changes — those belong to later PRDs.

## Binding amendments

D5 (`infer:false` for curated writes), D8 (fixes land studio-side where a choice exists), D10 (fake ticker/counts fixed here), D11 (token-budget guardrails come later; do not add new truncation), Q-defaults: Q1 fail-closed teams filter, Q26 deterministic palette, Q73 sonner as the single toast system.

## User Stories

### S1: RAG — full chunk text reaches the LLM
`modules/rag/service.py:855-909` already hydrates full chunks into `expanded_content` (N+1, then unread). Replace the per-candidate asyncpg queries with **one batched `IN` query**, and make the formatter consume the hydrated text (`service.py:544,754` read `content` only; `result_formatter.py:354-356` has a 4000-char budget that can finally fill).
**Acceptance:**
- [ ] Retrieval path returns full chunk text (not the 500-char S3 metadata preview) for chunks whose parents exist in PG
- [ ] Exactly one hydration query per retrieval (assert via query counter in test)
- [ ] New test: `tests/test_rag_hydration.py` — preview-only vs hydrated content; updates any formatter tests asserting 500-char outputs
- [ ] `pytest -q` green

### S2: RAG — team filter fails closed + team-access PATCH SQL fixed
`service.py:391-393` returns ALL candidates on DB error (fail-open). Fail closed: return only `team_access`-empty (public) candidates and log loudly. Fix `api/documents.py:1871-1924` PATCH/bulk SQL referencing non-existent `title`/`updated_at` columns (verify against `core/models/core.py:548-580`).
**Acceptance:**
- [ ] Forced DB error in filter test yields only public docs, never team-restricted ones (Q1 BINDING)
- [ ] PATCH `/api/documents/{id}/team-access` and bulk variant succeed against the real schema (new API test)
- [ ] `pytest -q` green

### S3: Memory — stop junk at the source
Per D5: send `infer:false` on every already-curated write — distilled facts, mission/playbook/widget/tool stores (`mem0_client.py:312-376` call sites). Switch single-delete to the bulk endpoint that actually exists (`mem0_client.py:498-506` → spec in `orchestrator/mem0_openapi.json`) so deletes stop 405-ing and tripping the circuit breaker. Retune `MEMORY_DECAY_RATE`/threshold (`config.py:104,106`) to week-scale. Call `touch_short_term` on recall so promotion can ever fire (`unified_memory_service.py:1990-1996`).
**Acceptance:**
- [ ] All curated writers pass `infer:false` (grep-able assertion test)
- [ ] Delete path uses bulk endpoint; circuit-breaker test no longer trips on 3 deletes
- [ ] Importance-0.8 L2 memory survives ≥7 days in decay unit test
- [ ] Recall increments access_count (promotion-eligibility test)
- [ ] Existing memory tests updated where they asserted old decay numbers; `pytest -q` green

### S4: Board — assignment dispatches execution
Assignment never starts work. Call the existing `_launch_task_execution` (`api/board_tasks.py:724`) from create-with-assignee and assign endpoints; guard double-fire (already-running task) and recipe-mirror tasks. Fix the broken priority `CASE` expression (`services/heartbeat_service.py:919`).
**Acceptance:**
- [ ] Assigning a task to an agent starts execution within one event loop tick (integration test, mock executor)
- [ ] Re-assigning a running task does NOT double-fire (test)
- [ ] Priority ordering test over the CASE expression
- [ ] `tests/test_board_task_handlers.py` updated for the new dispatch behavior; `test_board_sdk_auth.py` untouched and green (PRD-09 narrow dep)

### S5: Missions — Auto's lifecycle tools work
`platform_get_mission` crashes (`int()` on UUID PK, wrong field `t.output`, schema string) — fix in `handlers_missions.py`. Fix chat context extraction (`chat.tsx:870` reads message `parts` now). Delete the byte-duplicate `/resume` route. Document `auto_approve` in the mission-create tool schema and make the create reply honest about `awaiting_approval` state.
**Acceptance:**
- [ ] `platform_get_mission` returns a real mission by UUID (test with seeded mission)
- [ ] Auto-created missions carry recent chat as `context_messages` (parity with `mission-suggestion-card.tsx:48`)
- [ ] Duplicate route gone; route-collision test
- [ ] Tool reply for create names the approval state + how to approve
- [ ] `pytest -q` green; protected recipe suites untouched

### S6: Knowledge Graph — export and guards
`_export_graph` drops `node.community` and confidence — export both (backfill confidence where null). Add the `idOf()` null-guard at the 4 crash sites in the KG frontend. Deterministic palette: hash type→color (Q26 BINDING, palette as data not hardcoded hex). Point `platform_graph_communities` at `DbWorkspaceClient`.
**Acceptance:**
- [ ] Export test asserts community + confidence present
- [ ] Frontend typecheck + the 4 guard sites covered by a vitest unit on `idOf`
- [ ] Same type → same color across reloads (vitest)
- [ ] `platform_graph_communities` returns data for a DB-backed workspace (test)

### S7: Field memory — the graph finally renders
React 18 + R3F v9 can never mount (peer-dep conflict). Swap `mission-field-viz.tsx` to **react-force-graph-3d** (works on React 18) wrapped in an ErrorBoundary with a 2D `react-force-graph-2d` fallback. Archive-don't-destroy at terminal: `_cleanup_terminal_fields` (`coordinator_service.py:711-737`) marks `expired_at`/archived instead of deleting Qdrant data + popping `field_id` (D7 stepping stone). Read `params['_agent_id']` in `handlers_field.py` so field tools work for agents.
**Acceptance:**
- [ ] `@react-three/fiber`/`drei` removed from package.json (delete the losing stack, repo rule §5)
- [ ] Field tab renders nodes+edges for a live mission AND a completed (archived) mission — verify in browser via dev-browser
- [ ] ErrorBoundary fallback renders 2D on 3D failure (vitest)
- [ ] Archived field queryable post-completion (pytest)
- [ ] Field tools no longer 400 when called by an agent in a mission (test)

### S8: CodeGraph — working paths get used
Route `search_type='semantic'` to the working pgvector path; forward the dropped params; remove the `<50-char` content filter that hides most hits; reconcile the query-log schema mismatch and make logging non-fatal; emit `path:line` + symbol signature in results.
**Acceptance:**
- [ ] Semantic search returns results on the seeded index (test); param forwarding asserted
- [ ] Results include `path:line` and signature (snapshot test)
- [ ] Query-log failure cannot fail the search (test)
- [ ] `pytest -q` green

### S9: One toast system
Mount **sonner**'s `<Toaster>` (Q73 BINDING: most call sites), codemod the 17 react-hot-toast + 21 use-toast call sites to sonner, remove the other two systems and the duplicated store impl (`providers.tsx:91` currently mounts only react-hot-toast while 49 sonner call sites render nothing).
**Acceptance:**
- [ ] Mission approve/reject, template save error, field-op feedback all visibly toast — verify in browser via dev-browser
- [ ] `react-hot-toast` and `use-toast` fully removed (package.json + grep zero call sites; delete-what-you-replace)
- [ ] Typecheck + lint green

### S10: Honest surfaces for pilots (D10)
StudioTicker: wire to the live analytics endpoints already serving command-center, or remove it (`studio-ticker.tsx:39-47`, mounted `main-layout.tsx:104`). Remove the hardcoded 85.5% agent performance (`api/agents.py:331-336`), fake Studio tab counts (`studio-menu.ts:88-96`), and fake green validation checkmarks. Fix global search paths (`use-global-search.ts:43,59` → real routes, surface errors). Migrate agent skill create/remove to `/api/v1/skills` (`api-client.ts:1895-1941`). ESLint rule banning raw `fetch('/api`)`.
**Acceptance:**
- [ ] No string-literal metrics/counts in studio components (grep gate in acceptance script)
- [ ] Global search returns live results for a seeded agent name — dev-browser verify
- [ ] Skill add/remove round-trips against `/api/v1/skills` (test)
- [ ] ESLint fails on raw `fetch('/api'` (rule test)

### S11: Calendar stopgap
Real fix is PRD-162. Here: `get_schedule` returning `scheduler_active=false` becomes a thrown error so React Query retries instead of caching an empty 200 (the 1-of-4-workers APScheduler bug); `AbortSignal.timeout(15000)` + visible error state in CalendarTab.
**Acceptance:**
- [ ] Empty-because-wrong-worker response triggers retry, not an empty calendar (vitest on the query fn)
- [ ] Timeout shows error state with retry button — dev-browser verify

### S12: NL2SQL tab revival + safety trims
Migrate the three broken tabs off raw `fetch` to `apiClient` (`DatabaseQueryAnalytics.tsx:25-27`, `SemanticLayerBuilder.tsx:125,161`, `QueryTemplatesGrid.tsx:90,136` — revives Training instantly). Honor the `dialect` field. Return a structured failure instead of an executable error-string `SELECT`. Auth-gate the legacy entity-KG endpoints (`api/knowledge_graph.py:84-559`) pending their WS-10 fate. Strip the GitHub token from clone-error messages (`codegraph_service.py:489`).
**Acceptance:**
- [ ] All six Databases sub-tabs load data in split-origin dev — dev-browser verify
- [ ] Error path test: no SQL string returned as `sql` on failure
- [ ] Entity-KG endpoints 401 without auth (test)
- [ ] Clone error with bad token contains no token substring (test)
- [ ] `pytest -q` green

## Non-Goals

No dispatcher/queue redesign (PRD-161), no DB-first calendar (PRD-162), no extraction-prompt rewrite (PRD-159), no graph-shell consolidation (PRD-165), no team dropdowns (PRD-158), no plan mode (PRD-163).

## Success Metrics

- Agent answering from a seeded 10-page document quotes content beyond char 500.
- Assigned board task reaches `in_progress` without a heartbeat configured.
- Field tab shows a rendered graph for a completed mission.
- Zero hardcoded metric/count literals in studio chrome.
- Full orchestrator suite + lint/typecheck green; no protected suite regressions.

## Testing

Suite gates per chain policy: orchestrator `pytest -q` (Postgres, as `test.yml`), frontend `tsc --noEmit` + ESLint + vitest, dev-browser verification on S7/S9/S10/S11/S12 UI claims. Updated-not-deleted: `test_board_task_handlers.py`, memory decay tests, any formatter tests asserting truncated RAG output.
