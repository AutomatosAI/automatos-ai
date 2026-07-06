# PRD-189: Phase 2 · Wave 1 — Shopify Integrity (stop the FBT wipe, debounce webhooks, test the mappers, un-skip the golden journeys)

**Phase:** Phase 2 — Module Deep-Review remediation · Wave 1 (resurrect the dead client-facing loops)
**Branch:** `feat/p2-w1-shopify-integrity` · **Worktree:** `automatos-ai-p2w1-shopify`
**Dependencies:** **PRD-185** (Wave 0, merged `649482aa3`) — the observability/feed-loop precondition wave. This PRD assumes Wave 0's fail-loud embeddings (S3), real telemetry (S1), and the honest-tile cockpit (S12) are in place; the Commerce integrity tile here plugs into that same strip.
**Build size:** S–M (all four are small, surgical fixes; the un-skip is the one M because it stands up recorded Shopify fixtures) · **Risk:** Low (no rebuild, no new dependency, no new table; the graph-import path is reused, not re-architected)
**Source:** `reports/dossiers/shopify-vertical.md` (C.1–C.7, E, G, J1–J3), `reports/dossiers/knowledge-graphs.md` (C.1, J1–J2), `reports/dossiers/storefront-widget.md` (C.2, J2) — carried as report id **P2-08** in `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md`. Security §5.3 is referenced by this cluster (webhook robustness, mapper-fed autonomy) and handed to the separate Opus pass, not reworked here.

---

## Overview

Shopify is Automatos's one **end-to-end commerce reference pilot** — install → provision → sync catalog/orders into a per-workspace commerce Knowledge Graph → serve a shopper-facing widget whose proactive openers are grounded in that graph's frequently-bought-together edges. It is the concrete proof that the platform's generic machinery (agents, graph, widget, Deliverables, tools) can be pointed at a real business and run largely unattended. Its plumbing is genuinely good. But the review found that **the one capability the merchant actually sees — cross-sell — is silently broken in the live graph**, and the widget that talks to real shoppers **cites a factual grounding that no test protects.** Both are correctness bugs, confirmed in code and in the one production store's data, not cold-start artefacts.

Judged against the **North Star** — *does this make Auto more autonomously capable and the agents' output higher-quality for clients?* — this wave is directly North-Star-positive: it **restores the pilot's marquee feature** (cross-sell survives normal store activity) and **stops the widget confidently fabricating to shoppers** (its citations become a tested, gated number instead of a promise). No new capability is built; the deliverable is that the commerce leg *keeps* the intelligence it computes and *proves* the facts it says.

**Four defects are in scope, all verified in code at HEAD (`649482aa3`):**
1. **Every catalog re-sync wipes the cross-sell edges.** `_product_sync_impl` ends in `import_graph(merge=False)` — a full graph replacement — so the `frequently_bought_with` edges the last orders sync computed are erased on the next product edit. Since F032, a catalog webhook fires this automatically on any inventory change. *The marquee feature does not survive contact with normal store activity.* *(F1)*
2. **The webhook path has no debounce, coalescing, or already-running guard.** A bulk merchant edit emitting N webhooks fires N concurrent full Bulk-Op re-syncs (each of which — until F1 lands — also wipes FBT), with the task reference dropped (GC-eligible mid-flight). *(F2)*
3. **The mappers that produce every opener's facts have zero behavioral tests.** `map_shopify_catalog` / `map_shopify_orders` feed the widget's "bought together in X of Y orders" citations, and F032 now runs the catalog mapper on every webhook — a silent mapper regression corrupts every opener with a confident, provenance-styled sentence and no test catches it. *(F3)*
4. **Both headline golden journeys are skipped.** J3 (widget → vertical plugin → response) and J9 (Shopify sync → Knowledge Graph → FBT opener) are `pytest.skip`-ped, so the marquee journey — the exact one this PRD restores — is unproven in CI. *(F4)*

**PILOT lens (locked):** the pilot is a sample of one, its co-purchase signal is intrinsically thin (16 FBT edges from 57 orders), and deliverables platform-wide stopped 2026-06-16. **None of that is in scope.** Thin signal, one store, and cold counters are *not* defects — see `feedback-pilot-usage-not-quality-signal`. What *is* in scope: the FBT edges being **deleted** (a mechanism, not a data-volume problem) and the widget's facts being **untested** (a correctness gap). We fix the wipe and test the citations; we do **not** try to manufacture more orders or "drive usage." No moat framing — "cross-sell" and "the widget's grounding" are described by what they do for the merchant and the shopper, not as a defensible edge.

---

## Findings & Scope (all `file:line` per the review; confirm by grep before editing — the review tree drifted, HEAD is `649482aa3`)

| Finding | Issue (verified in code) | Fix | Story |
|---|---|---|---|
| **F1** (shopify J1 · knowledge-graphs J1) | `_product_sync_impl` ends in `gs.import_graph(workspace_id, graph, merge=False)` (`api/shopify.py:568`); `_import_graph_unlocked` with `merge=False` **replaces the whole graph** with only the catalog nodes/edges (`modules/knowledge/graph_service.py:466-467`; the merge branch at `:449-463` is skipped). The catalog JSONL carries no FBT edges, so every catalog sync erases the `frequently_bought_with` edges the orders sync computed — **and every non-catalog node** (flywheel syntheses, document/report entities, roster) merged in since. F032 fires this on every `products/update` / `inventory_levels/update` webhook. **Live evidence:** the pilot's `orders_sync` block reports `fbt_edges_added: 16` while the persisted `graph.json` holds **0** `frequently_bought_with` edges. | Make catalog re-sync **preserve** cross-sell (and non-catalog) edges across the rebuild, mirroring the strip-then-merge the **orders path already uses** (`api/shopify.py:809-844`): merge the fresh catalog over the existing graph rather than replacing it, keeping any edge/node the catalog bulk-op does not itself carry. | **S1** |
| **F1-guard** (shopify G · knowledge-graphs J2) | The `product_sync`/`orders_sync` status blocks report a sync *ran*, never that it *produced usable intelligence* — the pilot's block says `fbt_edges_added: 16` while the graph has 0, and nothing reads the drift. | Add an **FBT-persistence-integrity** check — `fbt_edges reported by last orders_sync == frequently_bought_with edges present in the graph` — as a pure assertion in the sync path and a Command Center tile (reuses the Wave-0 honest-tile strip, S12). | **S2** |
| **F2** (shopify J2 · knowledge-graphs C.12) | `POST /events` fires `_asyncio.create_task(_sync_catalog_for_workspace(...))` per catalog event (`api/shopify.py:388`) with **no debounce, no coalescing, no already-running guard**, and the task reference is not held (GC-collectable mid-flight). `_product_sync_impl` sets `product_sync.status="running"` (`:571`) but never checks it. A bulk edit emitting N webhooks fires N concurrent full Bulk-Ops, relying on Shopify's one-bulk-op-per-shop limit with the losers swallowed. | Add a **per-workspace debounce + already-running guard + coalescing** so a webhook burst produces **one** re-sync, mirroring the in-process debounce `GraphifyService` already uses (`graph_service.py:85,200-202,497-530`); hold the task reference. Debounce window through `config.py`, not `os.getenv` inline. | **S3** |
| **F3** (shopify J3 · knowledge-graphs J2 · storefront-widget J2) | `map_shopify_catalog` (`graph_extraction.py:503`) and `map_shopify_orders` (`:693`) have **zero direct behavioral tests** (grep: only a registry identity check exists); the widget's opener cites `co_count`/`total_orders` straight from `map_shopify_orders` (`widget_proactive.py:239-313`, `_build_proactive_opener_message`), and F032 runs the catalog mapper automatically on every webhook. A silent mapper regression makes the widget **confidently fabricate** to real shoppers. | Add **fixture-driven behavioral tests** for both mappers (correct node types, edge relations, FBT `co_count`/`total_orders`/`min_support` math, cancelled-order exclusion). Pure — a JSONL fixture in, assert the graph out. Reuse the existing `tests/fixtures/` shape. | **S4** |
| **F4** (shopify J3 · storefront-widget J2) | `test_j3_widget_plugin_response` (`tests/test_golden_journeys.py:101`) and `test_j9_shopify_sync_to_fbt_opener` (`:221`, self-labelled "the highest-value gap to close next") are `pytest.skip`-ped "needs a live Shopify store + the KG build + the widget." The marquee journey — the one this PRD restores — is unproven in CI. | **Un-skip both** by mocking Shopify at the boundary (recorded catalog/orders JSONL fixtures → real `map_shopify_*` → real `import_graph` against a fixture graph → real `handle_widget_message`), so the sync→graph→FBT-opener path (incl. the S1 preservation guarantee) runs green in CI with **no external service**. The `inbuild_graph_snapshot.json` fixture + `conftest.py` `GraphifyService` stub already exist to build on. | **S5** |

---

## Stories (test-first — write the failing test, make it green, refactor)

> Each story is independently shippable. S1 and S5 have a soft ordering (S5's J9 asserts S1's preservation guarantee, so it is cleanest to land S1 first or together). File ownership is largely disjoint — see **Sequencing**.

### F1 · Stop the catalog re-sync wiping cross-sell

**S1 · Preserve cross-sell across catalog re-sync (the FBT wipe) — S · _the single biggest client-facing restore_**
**Files:** `api/shopify.py:568` (`_product_sync_impl`, the `import_graph(merge=False)` call), reusing the strip-then-merge already at `api/shopify.py:809-844` (orders path) and the merge branch in `modules/knowledge/graph_service.py:449-467`. Catalog nodes are provenance-tagged `source_file` = the `shopify://` catalog source (`graph_extraction.py:503-560`), which is how catalog-vs-other nodes are told apart.
**Test:** `test_catalog_resync_preserves_fbt_edges` — build a graph with catalog nodes **and** `frequently_bought_with` edges (from a fixture orders sync), run `_product_sync_impl` over a fresh catalog JSONL fixture (Composio/httpx mocked at the boundary, per the existing `test_prd183_s1_catalog_webhook.py` pattern), and assert the FBT edges **survive** (today: they are wiped to 0). `test_catalog_resync_preserves_non_catalog_nodes` — assert a flywheel/document-sourced node present before the sync is still present after (today: replaced).
**Notes:** Do **not** re-architect the import path — reuse the orders path's proven pattern (identify the edges/nodes the catalog bulk-op does not itself carry, keep them, merge the fresh catalog over the top). Delete the `merge=False` catalog behaviour in the same change — no "keep both modes" shim. This restores the marquee feature to *work at all* on a live store; it does nothing to increase order volume (out of scope, pilot lens).

**S2 · FBT-persistence-integrity check + Commerce tile — S**
**Files:** the sync status write in `_product_sync_impl` / `_orders_sync_impl` (`api/shopify.py`), the Wave-0 honest-tile strip (PRD-185 S12, `api/analytics_real.py` own-workspace tiles), a read-only integrity helper.
**Test:** `test_fbt_integrity_detects_drift` — given a workspace whose last `orders_sync` reported N FBT edges but whose graph holds M≠N `frequently_bought_with` edges, assert the integrity check reports the drift (this is the single query that would have caught the wipe: 16 reported, 0 present). `test_fbt_integrity_clean_after_resync` — after S1's fix, assert reported == present.
**Notes:** This is the guardrail that stops S1 ever silently regressing — the same role S2 played for S1 in Wave 0. Surfaces as a **Commerce** tile ("cross-sell: N pairs live · last computed M") reachable by the workspace admin, plugged into the Wave-0 strip; no new analytics router, no new table. Own-workspace only, per the Wave-0 de-scope.

### F2 · Tame the webhook path

**S3 · Debounce + coalesce + already-running guard on `/events` — S**
**Files:** `api/shopify.py:355-396` (`forward_event`, the per-event `create_task` at `:388`), `_sync_catalog_for_workspace` (`:322-352`), a new debounce window flag in `config.py` (mirror `PLAYBOOK_BREAKER_THRESHOLD` at `config.py:536`).
**Test:** `test_webhook_burst_coalesces_to_one_resync` — deliver N catalog events for one shop inside the debounce window (the re-sync mocked at the boundary) and assert exactly **one** `_product_sync_impl` runs, not N (today: N concurrent). `test_webhook_already_running_is_skipped` — while a re-sync is in flight, a further event does not launch a second concurrent full sync. `test_webhook_debounce_window_from_config` — the window reads from `config`, not an inline `os.getenv`.
**Notes:** Reuse the in-process debounce shape `GraphifyService` already ships (`graph_service.py:497-530`) rather than inventing a scheduler; hold the task reference so it can't be GC'd mid-flight. Debounce is also the cost lever the review flags — F032 fires a full embedding-bearing rebuild per webhook. In-process debounce is the honest v1; a cross-worker version is not in scope (surface, don't defer, if the reviewer wants it — §12).

### F3 · Prove the openers' facts

**S4 · Mapper behavioral tests (the untested grounding) — S · _the anti-fabrication guarantee_**
**Files:** new `orchestrator/integrations/shopify/tests/test_mappers.py` (or alongside the existing `tests/`), fixtures under `orchestrator/integrations/shopify/tests/fixtures/` (the `inbuild_graph_snapshot.json` / catalog+orders JSONL shape already lives there).
**Test:** `test_map_shopify_catalog_produces_typed_nodes_and_edges` — a small catalog JSONL fixture in, assert `shopify_product`/`shopify_variant`/`shopify_collection`/`shopify_vendor`/`shopify_metafield` nodes and `variant_of`/`in_collection`/`by_vendor`/`has_metafield` edges (`graph_extraction.py:503-560`). `test_map_shopify_orders_fbt_math` — an orders JSONL fixture in, assert `frequently_bought_with` edges with correct `co_count`/`total_orders`, the `min_support` gate (`:783`), and **cancelled-order exclusion** (`:749-759`). `test_map_shopify_orders_emits_no_customer_nodes` — assert the privacy-by-design property (only aggregated product↔product edges, no customer/order nodes).
**Notes:** Both mappers are pure, deterministic, IO-free — ideal for fixture tests, no mocking needed. This is the test that turns "the widget's citations are trustworthy" from a promise into a gated fact; it is the highest North-Star-per-line story here because autonomy that fabricates-with-citations is worse than a canned opener.

### F4 · Un-skip the marquee journeys (CI-runnable, mocked at the boundary)

**S5 · Un-skip J3 + J9 — M · _the marquee journey becomes a gated number_**
**Files:** `tests/test_golden_journeys.py:101` (J3), `:221` (J9); recorded Shopify catalog/orders JSONL fixtures (reuse/extend `integrations/shopify/tests/fixtures/`), the `conftest.py` `GraphifyService` stub that already returns a fixture graph.
**Test:** replace the two `pytest.skip(...)` bodies with real assertions. **J3:** a widget message → `PLUGIN_REGISTRY["shopify"].handle_widget_message` → grounded response, using the recorded fixtures (no app-level client, no live store). **J9:** the full marquee path — recorded catalog JSONL → real `map_shopify_catalog` → `import_graph`, then recorded orders JSONL → real `map_shopify_orders` → merge, then `_resolve_graph_related_products` + `_build_proactive_opener_message` produce a provenance-cited opener — **and, folding in S1, assert a subsequent catalog re-sync leaves the FBT edge intact** so J9 also guards the wipe fix.
**Notes:** The whole point is CI-runnable purity — **mock Shopify (Composio bulk-op + JSONL download) at the boundary** so both journeys run green with no external service, per the Verification section. Do **not** leave them `skip`-ped with a "needs live store" reason; the recorded fixtures + the existing byte-equality harness (`test_widget_proactive.py` US-011 snapshot tests) prove the shape is testable offline. Un-skipping J9 makes the exact journey this PRD restores a tracked, gated CI number.

---

## Sequencing (Wave 1 · Shopify is mostly parallel-safe)

- **S1 → S5** is the only soft ordering: J9 (S5) asserts S1's preservation guarantee, so land S1 first or in the same PR. S2 (integrity guard) also reads cleanest once S1 makes reported==present true.
- **S4 (mapper tests)** is fully independent — pure fixture tests, no dependency on S1/S3.
- **S3 (webhook debounce)** is independent of the graph-import change (it governs *how often* `_product_sync_impl` fires, S1 governs *what it preserves*). They compose but don't block each other.
- If built by parallel agents, file ownership is disjoint except `api/shopify.py`, which S1 (`_product_sync_impl` body) and S3 (`forward_event` / `_sync_catalog_for_workspace`) both touch — coordinate on that one file. The one shared config surface is the S3 debounce flag in `config.py` — never `os.getenv` inline.

---

## Verification (CI is the only gate — no local runs)

Per current project convention (`feedback-no-local-servers`): **do not run servers, builds, `next dev`, headless Chromium, `pytest`, `tsc`, or installs on the dev machine.** Write the code + **pure** tests (no DB / network / Composio / Shopify bulk-op calls — **mock Shopify at the boundary** so they run in CI), commit, push, and let **CI (the PR checks) verify.** Every new test — including the two un-skipped golden journeys (S5) — must be runnable with **no external service**: a recorded JSONL fixture stands in for the Composio bulk-op + signed-URL download, and the `GraphifyService` fixture stub stands in for the workspace blob. The existing `test_prd183_s1_catalog_webhook.py` (mocks `_product_sync_impl` internals at the boundary) and the US-011 byte-equality tests (fixture graph via `conftest.py`) are the patterns to follow. If a golden journey cannot be made pure, it is not in scope to un-skip it as an integration test — but J3/J9 can, because the fixtures already exist.

---

## Conventions (non-negotiable — see `automatos-ai/CLAUDE.md`)

- No `os.getenv()` outside `config.py`; the S3 debounce window goes through the canonical config module (mirror `PLAYBOOK_BREAKER_THRESHOLD`).
- No backward-compat shims — delete the `merge=False` catalog behaviour when S1 replaces it; do not keep both import modes "just in case."
- No new tables where an existing one fits (S2 reuses the sync status blocks + the Wave-0 analytics strip); no new tools where an existing one extends; no new analytics router (S2 plugs into the PRD-185 S12 strip).
- Immutable patterns; small focused functions; comprehensive error handling; no silent `except` swallows — the FBT wipe was invisible precisely because a status block lied about success while the graph was empty.
- Reuse the orders path's strip-then-merge (S1) and `GraphifyService`'s in-process debounce (S3) — do not build parallel machinery.
- Canonical vocab: **Playbook** (not Recipe), **Deliverable**, **Knowledge Graph** (spelled, never "KG" in prose), **Command Center**, **Auto**.
- Branch `feat/p2-w1-shopify-integrity`; commit, push, open a PR; CI is the gate.

## Success metrics (the definition of "Shopify integrity restored")

- **Cross-sell survives a catalog re-sync** — after S1, a catalog sync (or webhook) no longer deletes `frequently_bought_with` edges, and no longer deletes non-catalog (flywheel/document/roster) nodes; proven by `test_catalog_resync_preserves_fbt_edges` (S1) and the J9 re-sync assertion (S5).
- **FBT-persistence integrity is a tracked, non-lying number** — reported FBT edges == present FBT edges, with a Command Center tile that reads 0-live honestly until the fix lands (S2). This is the "16 reported / 0 present" query that would have caught the wipe on day one.
- **A webhook burst produces one re-sync, not N** — debounced, coalesced, already-running-guarded, task-reference held (S3); the daily webhook-storm amplification (and its repeated embedding cost) is gone.
- **The openers' facts are tested** — `map_shopify_catalog` / `map_shopify_orders` have behavioral coverage (node types, edge relations, FBT math, cancelled-order exclusion, no-customer-nodes), so a silent mapper regression is caught before it reaches a shopper (S4).
- **The marquee journeys run green in CI** — J3 (widget → plugin → response) and J9 (sync → Knowledge Graph → FBT opener) are un-skipped and pass with Shopify mocked at the boundary (S5); the journey this PRD restores is now a gated number, not a promise.

## What this wave gates

This is the Wave-1 Shopify slice. It **restores the commerce pilot's client-facing intelligence** (cross-sell that persists, openers whose facts are proven) and makes the marquee journey a CI-gated number — the precondition for any later "learned proactive trigger / opener-outcome feedback" work (dossier J5), which depends on the widget having a *correct* grounding to learn from. It also lands the T1 typed-graph thesis concretely for commerce in the narrowest possible form (preserve non-catalog knowledge across a catalog rebuild) without pre-committing the broader graph-substrate decision (Wave 3 P2-16/P2-17), which stays gated on the Wave-0 memory eval. Separating the commerce Knowledge Graph from conversation-memory pollution (the `l2:` nodes, dossier J4) and down-weighting metafields in community detection are **adjacent** and **not in this PRD** — surfaced here as the reviewer's next-slice call, not silently deferred (§12).

---

*Traceability: every story cites its dossier (`reports/dossiers/{shopify-vertical,knowledge-graphs,storefront-widget}.md`) and the report id **P2-08** in `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md`. `file:line` refs verified by grep against HEAD `649482aa3` (the review's pinned tree `77bc9c6d5` drifted; line numbers re-confirmed during drafting). North-Star framed; PILOT lens applied (thin signal / one store / cold counters are explicitly out of scope — only the FBT deletion and the untested citations, which are real correctness bugs, are in scope); no moat framing. Security §5.3 (webhook signature/robustness, mapper-fed autonomy) is referenced and handed to the separate Opus security pass.*
