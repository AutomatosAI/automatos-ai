# PRD-232 — The Intent Graph: wire it honest, seed it smart, let it learn, prove it, flip it

**Status:** DRAFT for Gerard's review — 2026-08-29
**Owner:** Gerard · **Prepared by:** deep review 2026-08-29 (two-agent sweep: PRD promises vs shipped pipeline)
**Name:** the learned selection layer is the **Intent Graph** (sibling of the Knowledge Graph). Presentation hook: "Auto's Reflexes." The implementation term for cold-start is **synthetic utterance seeding**.

---

## 1. Why now

Live failure, 2026-08-28: Gerard asked Auto to *"close all the blocked tickets from Vector."* Auto — correctly and honestly — reported the board write route absent from its session. The capability exists (`platform_update_task_status`, registered with examples, handler wired). The selection pipeline never gave it to him. November demo needs the opposite story: any phrasing → the right 5–8 tools → acted on.

The deep review found the machine is ~80% built and well-designed — PRDs 64/122/138-139/141/142/143/177/185 built a real learning architecture — but **four wiring inversions mean the learned half never engages**, one of which is a live defect on most chat turns.

## 2. What the review proved (file:line evidence)

### C1 — CRITICAL, live defect: the dispatcher is stripped from the surface on every graph-branch turn
`SmartToolRouter.route()`'s graph branch builds its keep-set from chain names ∪ CORE_TOOLS ∪ always-include ∪ suggested_tools (`consumers/chatbot/smart_tool_router.py:232-246`). `platform_execute` is in **none of them** — it is not an ActionDefinition, not a pin, not a core tool. The branch fires on effectively every tool-requiring turn (see C2), returns early, and **removes the single door to ~136 actions**. Turns that match the literal phrase map are rescued by accident: `tool_hints=["platform"]` fires the hint branch first, and `"platform" in "platform_execute"` preserves the dispatcher. That is why Gerard's board *reads* worked ("how's the board" is in the map) and his *write* didn't ("close/ticket/blocked" is not). No test asserts the dispatcher survives `route()` (`tests/test_us014_graph_router_delegation.py:130-181` uses a fixture list without it).

### C2 — the graph runs under the wrong flag
The graph call is gated on `SEMANTIC_TOOL_ROUTING` (default **true**) at `smart_tool_router.py:224`. `TOOL_ROUTING_GRAPH` (default **false**) is consumed at exactly one site — the prompt catalog (`platform_actions.py:135-139`). Net: the read-side Gerard deliberately held dark (PRD-177 S4/S6 governance) **runs anyway** on the schema path, while the catalog path it was meant to gate stays dark. The flag is inverted across surfaces.

### C3 — the write side of the learning loop is dark
`TOOL_SIGNAL_RECORDER_ENABLED` defaults **false** (`config.py:1059`; no-op at `signal_recorder.py:173-174`). PRD-184 left it explicitly as Gerard's call: "default-true or delete — dark forever is the one wrong state." Today only the nightly `edge_builder` learns (from `tool_execution_logs`, which do carry `user_query` post-PRD-185); intra-day freshness never ships and `api/harness.py` stats read all-zero.

### C4 — the learned structures are lobotomized at read time
- **Intent clusters are write-only**: created nightly (`edge_builder.py:440-470`), **never queried live** — `graph_router.py` never imports them (no query→cluster assignment exists).
- **Cluster-scoped affinities are applied cluster-blind**: `_query_affinities` (`graph_router.py:314-347`) filters by action/workspace/agent but **not `intent_cluster_id`** — `succeeds_for_intent`/`fails_for_intent` are summed across all intents.
- **`failed_after` edges are written and never read** (`edge_builder.py:217-270` vs `_query_edges` following only `used_after`/`meta_sibling`, `graph_router.py:281-284`).

### C5 — promoted-set drift: 13 promised → 47 shipped, ungated
PRD-122 promised ~13 promoted schemas ≈ 3.9k tokens, "promote all 91" explicitly rejected as "overwhelms tool selection." Shipped: **47** promoted schemas attached unconditionally every full-path turn (`tool_router.py:713-718`), ≈ 4–7k tokens — the largest per-turn tool cost, zero query-relevance gating (only tier gating). Meanwhile `rank_actions` runs **4× per turn** on the same query (narrow, shadow, catalog, graph-entry).

### C6 — the vocabulary hole (both gates)
The AutoBrain phrase map for `platform_update_task_status` (`auto.py:541-546`) and the embedding corpus (`action_semantic_index.py:88-96`: name+description+tags+examples+category) both lack "close", "ticket", "blocked" — and **parameter enums are not embedded** (the `done` status value is invisible to the ranker). `matched_tools` from the phrase map is **dead code** — written at `auto.py:873`, only ever read by a log line (`api/chat.py:397`).

### C7 — no gap or negative-selection signal exists
`ToolSignal` carries only name/success/agent/workspace/prior (`signal_recorder.py:56-66`). Nothing records shown-but-unused; nothing records "the model needed a capability it wasn't given"; `record_selection` persists to an in-memory FIFO only. The VECTOR failure was structurally unobservable to the learning loop.

### C8 — `platform_find_tools` can't widen the surface
Its result is prose/JSON (`handlers_capabilities.py:115-127`); the tool array is frozen at `generate_response`. Its `call_with` escape hatch only works because the executor skips enum re-validation — and C1 may have removed the dispatcher it depends on.

### C9 — unreconciled tenancy + threading defects
Global `meta_sibling` bootstrap edges (PRD-143 addendum) vs PRD-177's locked per-tenant read filter were never reconciled. The catalog's graph path can never be agent-scoped (`platform_actions.py:145` reads an `agent_id` kwarg that `smart_orchestrator.py:236-259` never passes).

### C10 — governance context
PRD-138/139 (the founding specs) don't exist as documents — reconstruct from citations only. The flip gate is locked policy (PRD-177 S6): run `operating_graph_uplift.py` per tenant vs BM25 + embedding baselines; **flip only on ≥5–10 point uplift; never fabricate the number**. PRD-223 notes the eval set is stale (47 vs 59 queries). ATOM turns ship only the dispatcher; full turns ship 47 schemas and (per C1) lose the dispatcher — no lane ships both.

## 3. Goals

1. Any reasonable phrasing of a supported intent surfaces the right tool — "close the blocked tickets from Vector" gives Auto `platform_list_tasks` + `platform_update_task_status` (+ dispatcher) and the close happens.
2. The Intent Graph actually engages: seeded day-one, cluster-aware at read time, learning from success, failure, non-use, and gaps.
3. Token cost per turn drops (target: tool surface ≤ ~4k from today's ~5–8k) while hit-rate rises — measured, not asserted.
4. The flip follows Gerard's own gate: eval number first, flag second.

**Non-goals:** no new UI; no skills-axis changes (PRD-120/231 own that); no Composio catalog rework (177 F016 owns per-action telemetry); no new ranking algorithms — we wire, seed, and feed the ones built.

## 4. Stories

### Wave A — wire it honest (small PRs, immediate effect)

**US-001 — The dispatcher survives every route.**
`platform_execute` (and the narrowed enum it carries) is preserved through every `SmartToolRouter.route()` branch — graph, hint, category, fallback. AC: a test asserts the dispatcher is present in `filtered_tools` on each branch with a realistic tool list (the missing test of C1); the VECTOR sentence, replayed through AutoBrain + route(), yields a surface through which `platform_update_task_status` is callable. No new pins-sprawl: one reserved-slot mechanism, reused by the heartbeat lane's existing `_apply_dispatcher_always_include`.

**US-002 — Each flag gates its own surface.**
The `rank_chains` call in `smart_tool_router.py` moves behind `TOOL_ROUTING_GRAPH`; `SEMANTIC_TOOL_ROUTING` continues to gate embedding narrowing everywhere. AC: with graph flag off, no `GraphRouter` query runs anywhere; with it on, both schema path and catalog path use it; config docstrings state the split; no `os.getenv` outside config.py.

**US-003 — One ranking pass per turn.**
`rank_actions` computed once per turn and reused by narrowing, catalog, shadow logging, and graph entry (4×→1). AC: instrumented count in tests = 1; latency delta logged.

**US-004 — Thread the scopes.**
`agent_id` reaches the catalog graph path; `workspace_id` required on all edge reads with global (`workspace_id IS NULL`) rows admitted **only** for `meta_sibling` bootstrap edges at the min-confidence floor — reconciling PRD-143's global seeds with PRD-177's per-tenant lock. AC: tenant-isolation test extended to cover the bootstrap floor; `_build_graph_filtered` receives a real agent_id.

### Wave B — synthetic utterance seeding (the cold-start)

**US-005 — The utterance corpus.**
A one-shot generator (script, human-reviewed output committed as data) produces 15–25 diverse utterances per registered action: colloquial, terse, verbose, wrong-register on purpose (ticket/task/card/item; mail/email/inbox; close/finish/clear/kill). Sources folded in: `ActionDefinition.examples`, the entire AutoBrain phrase map (which then stops being a gate — see US-008), Gerard's known phrasings. AC: corpus file per category under `orchestrator/core/seeds/utterances/`; generator is re-runnable and diff-friendly; su-only actions excluded; ≥90% action coverage.

**US-006 — The corpus reaches the embeddings.**
`_build_embedding_text` gains utterances and parameter enum values (status names like `done` become visible to the ranker). `ensure_indexed` re-embeds on corpus change (hash-keyed like the skill loader). AC: ranking test — "close the blocked tickets" ranks `platform_update_task_status` in top-5 of the semantic floor; embedding cache invalidates on corpus hash change.

**US-007 — The corpus seeds the graph.**
Extend the human-applied `seed_tool_routing_graph.py`: utterance embeddings → seeded intent clusters (`provenance='seeded'`, conservative confidence at the metadata floor) with `action_names_hot` = the action + family; nightly `edge_builder` recompute overrides seeded confidence with organic Wilson confidence as evidence accrues. AC: `test_seeded_cluster_routes_unseen_phrasing` (a phrasing NOT in the corpus lands on the right cluster); seeds never outrank organic rows; idempotent re-run.

**US-008 — The phrase map becomes data, not a gate.**
AutoBrain's `_match_platform_query` stays as a fast-path booster only; its vocabulary moves into the corpus; dead `matched_tools` is either consumed (as a rank boost) or deleted — no dead writes. AC: grep proves no dead field; fast-path behaviour parity test.

### Wave C — learning that learns

**US-009 — Turn the recorder on.** *(decision locked §6.1: default ON now)*
`TOOL_SIGNAL_RECORDER_ENABLED` default true; restart-safety per PRD-142 W4-S9 (no signal loss on drain restart); harness stats show non-zero within a day of deploy. AC: signals/day visible; no pool exhaustion under load test fixture (the 141 US-019 contract).

**US-010 — Cluster-aware reads.**
Live query → nearest intent cluster (cosine to centroid, threshold; miss = no cluster, embedding floor only). `_query_affinities` filters by `intent_cluster_id`; `fails_for_intent` subtracts per-intent as PRD-141 US-017 promised. `failed_after`: read as expansion penalty, or stop writing it — one or the other, no write-only tables. AC: per-cluster affinity test; the write-only and read-never greps both come back empty.

**US-011 — The gap signal (what I didn't have) + shown-unused decay (what I didn't mean).**
(a) `tool_gap` events: recorded when `platform_find_tools` is called for an absent capability, or a tool-requiring assessment ends with zero platform calls. Stored on the existing telemetry lane (no new table if `tool_execution_logs` + a gap marker suffices — builder decides, flag if a table is genuinely needed). (b) `record_selection` persists (batched, same recorder) so the nightly job computes shown-vs-used and decays never-used affinities. (c) Nightly **resolution join**: a gap followed in-session (or same conversation ≤24h) by a successful action becomes a positive cluster→action affinity — the ground truth is *the action that eventually served the intent*. AC: replay of the VECTOR transcript produces a gap row; a scripted gap→resolution fixture produces the affinity; decay test.

### Wave D — prove it, then flip it

**US-012 — Refresh the eval, publish the number.**
Regenerate `eval_set.jsonl` from `eval_seed.yaml` (stale 47 vs 59, PRD-223), add utterance-derived queries including the VECTOR case and abstain rows where no tool applies. Run `operating_graph_uplift.py` per tenant (seeded graph vs BM25 vs embedding floor). AC: the number is published in the PR body and a report file; CI lane stays non-required.

**US-013 — The flip.**
If uplift **≥5 points** (§6.4): `TOOL_ROUTING_GRAPH=true` in Railway; rollback = flip back (documented). (`TOOL_SIGNAL_RECORDER_ENABLED` is already on from US-009.) If below 5: record the number, do **not** flip, list what the number says to fix next. AC: either outcome is a legitimate completion — fabricating or skipping the eval is the only failure mode.

### Promoted-set diet — decided: promotion-as-prior (§6.2)

**US-014 — Promotion becomes a prior, not an attachment.**
A curated pin list of ~10–13 (start from PRD-122's original list at `122-TOOL-ROUTING-PROMOTION-FIRST-CLASS-SCHEMAS.md:317-331`, adjusted for what shipped since) always attaches as first-class schemas. Every other `promoted=True` action keeps its flag but the flag now means a ranking **boost** in `rank_actions` (alongside the intent/CORE boosts from PRD-64) instead of unconditional attachment; when ranked into the surface it attaches as its first-class schema, otherwise it is reachable via the dispatcher enum like any action. AC: full-path tool payload measured before/after in the PR (target ≥2k token reduction); pins list lives in config (no hardcoded values); a ranking test proves a boosted action outranks an equal-cosine unboosted one; tier gating (`super_admin_only`, admin) unchanged and fail-closed; no action becomes unreachable (dispatcher fallback test).

## 5. Success metrics

- VECTOR replay: closed by chat, one turn, no vocabulary luck.
- Selection health metric live (hit-rate, fallback-rate — PRD-143 US-006 finally lands) on the su-locked dashboard; signals/day > 0 with the organic-rows canary green.
- Tool tokens/turn: measured before/after (assembly numbers in the PR); target ≤ ~4k on full path without hit-rate regression.
- A per-tenant uplift number exists in the repo — whatever it says.

## 6. Decisions — LOCKED (Gerard, 2026-08-29)

1. **US-009 recorder default → ON now.** `TOOL_SIGNAL_RECORDER_ENABLED` defaults true with the restart-safety hardening; signals flow from day one of the wave; rollback is one env var. (Resolves PRD-184 item 16.)
2. **US-014 promoted diet → promotion-as-prior.** ~10–13 true pins always attach; every other currently-promoted action loses unconditional attachment and becomes a strong ranking boost competing in the semantic surface. (Settles the tool-surface review §7 promoted-sprawl finding.)
3. **Utterance corpus → committed to the repo.** It is seed input data, not eval gold; gold sets stay local per the standing rule.
4. **Flip threshold → ≥5 points** per-tenant over BM25 + embedding baselines (the lower bound of PRD-177's locked range). Below 5: publish the number, do not flip, list what it says to fix.

## 7. Traps for the builder (pre-verified)

- The keep-set fix (US-001) must not regress the hint branch's accidental rescue — the parity test in US-008 covers both.
- `edge_builder` clusters are delete-and-reinsert per run — seeded clusters must survive the nightly (provenance-aware rebuild), or seeding evaporates at 03:00 UTC.
- `ToolExecutionLog.intent_cluster_id` FK is live (PRD-142-W5) — cluster rebuilds must not orphan it mid-transaction.
- Per-process embedding dict + Redis cache: corpus-hash key change must invalidate both.
- `DeterministicEmbeddingProvider` must never appear in any seeding/eval path (PRD-185 S3) — synthetic vectors would poison every centroid.
- Follow-the-wrapper rule: `route()`'s early returns are the C1 mechanism — any new branch must prove dispatcher survival in the same test.

## 8. Relationship to other work

- **PRD-231** (context diet) — same philosophy, other half of the prompt: skills went small-always-on + pull-on-demand; tools go ranked-surface + learned reflexes. US-006 telemetry there and selection-health here are the two halves of the "what does Auto carry per turn" dashboard.
- **Tool-surface review 2026-07-23** — §7 decisions get their vehicle here (US-014 + C5).
- **PRD-177** F016 (Composio per-action telemetry) and S3 (fail-closed destructive gate) remain owned there; this PRD does not touch them.
- **November presentation** — the Intent Graph *is* the per-tenant outcome-labeled edge dataset the OS-roadmap calls the real moat; the uplift number is the slide.
