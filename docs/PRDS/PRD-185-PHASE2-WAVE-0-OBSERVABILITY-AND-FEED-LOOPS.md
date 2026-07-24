# PRD-185: Phase 2 · Wave 0 — Make the Loops Observable, Honest, and Fed

**Phase:** Phase 2 — Module Deep-Review remediation (the precondition wave)
**Branch:** `feat/p2-w0-observability-feed-loops` · **Worktree:** `automatos-ai-prd185`
**Dependencies:** All 14 hardening waves merged to `main` (`77bc9c6d5`) — esp. **W7** (operating-graph loop, PRD-177), **W8** (field memory, PRD-178), **W10** (observability/SLOs, PRD-180)
**Build size:** M–L (mostly small fixes; the eval substrate is the one M) · **Risk:** Low–Medium (surgical; no rebuilds)
**Source:** `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` §4 (quick-wins) + §6 Wave 0 (P2-01…P2-05); per-item dossiers in `reports/dossiers/`

---

## Overview

The Phase-2 review's one-line finding is **"good bones, open loops":** the architecture is sound almost everywhere, and then the loop that would make it *work* is dead, unmeasured, or switched off. This wave closes that gap. It is **load-bearing for the entire Phase-2 program** — until quality is a tracked number and the platform's nervous system is connected, every larger investment (a graph substrate, a learned router, a new vertical) is spent blind.

Judged against the **North Star** — *does this make Auto more autonomously capable and the agents' output higher-quality for clients?* — Wave 0 is the highest-leverage fortnight in the review, because it is the precondition for judging everything else on evidence instead of faith. **No moat framing; no new capability.** The deliverable is that the platform can finally *see what its agents did* and *whether the work was good.*

**Three loops are currently severed, and this wave reconnects them:**
1. **The telemetry write is type-poisoned** → the operating graph has 0 organic edges after two months and the whole learning plane starves. *(P2-01)*
2. **The one live autonomous line (playbooks) fails silently and marks itself green** → a ~2.5-week production outage told no one. *(P2-02)*
3. **Quality is a tracked number nowhere** (0/28 modules) and the retrieval plane may be silently empty → agents may be answering ungrounded with no signal. *(P2-03, P2-04)*
Plus: the operator can't see any of it — the cockpit 403s to blank for non-super-admins. *(P2-05)*

**PILOT lens (locked):** empty tables / synthetic seed / cold-start counters are **not** failures and are **not** in scope to "fix by driving usage." In scope is the *wiring* that means, when real traffic arrives, the loop records, enforces, and measures. See `feedback-pilot-usage-not-quality-signal`.

---

## Findings & Scope (all `file:line` per the review; confirm by grep before editing — numbers may have drifted)

| Finding | Issue (verified in review) | Fix | Story |
|---|---|---|---|
| **QW-1** | `ToolExecutionLog.user_id = Column(Integer)` (`composio_cache.py:215`) but the chat lane binds a Clerk **string** id → every logged-in tool call fails its INSERT, swallowed at DEBUG (`telemetry.py:91,107-112`). Root cause of 0 organic graph edges across 21 workspaces. | Resolve Clerk-id → integer `User.id` before insert (or retype the column, one migration); raise the swallow to WARNING. | **S1** |
| **QW-1b** | The write failure is invisible — no signal that organic rows/day = 0. | Boot-probe + a per-lane "organic rows/day" canary that alerts on zero. | **S2** |
| **QW-4** | `DeterministicEmbeddingProvider` returns hash-seeded **random vectors** on a missing/failed key (`core/llm/clients/base.py:225`); RAG can't tell `empty` from `error` (`modules/rag/service.py:981-983`; `embedding_manager.py:90-93`). On the 402-failing key, retrieval has plausibly been silent noise since ~06-16. | Remove the deterministic provider from all production paths; fail **loud** on missing/failed embedding; make failure typed (`empty` vs `error`) + a zero-result/error-rate metric. | **S3** |
| **QW-2** | Playbook failures mark their board task `status='done'` (`board_task_bridge.py:114`); no `playbook_failed` event type exists (`notification_dispatcher.py:54-55`; `recipe_executor.py:1588` / `_fail_execution` dispatches nothing). | Add `playbook_failed` to valid event types + defaults; dispatch it from `_fail_execution`; set board status `failed` (not `done`) on failure; add a repeated-failure circuit-breaker so a broken playbook stops re-firing (kills the daily 402 spam). | **S4** |
| **QW-3** | 3 imports still target the March-renamed `recipe_memory_service` / `recipe_learning_service` (now `playbook_*`) — `recipe_executor.py:1118,1742`, `workflow_recipes.py:1231`; a swallowed `ImportError` left playbooks writing memories but reading none, `auto_learning` a no-op, `/learn` 500ing, for 3.5 months. | Repoint the 3 imports; add a regression test that imports every `core.services.*` symbol referenced across the codebase. | **S5** |
| **QW-8** | `get_user_id` returns hardcoded `id=1` (`api/chat.py:82-89`); every chat + PRD-163 approval attributes to user 1; the ownership check compares a constant to itself (`service.py:1268-1275`). Blocks per-user memory signal. | Thread `ctx.user` into `chats.user_id`, message saves, vote checks, and the approval `_driving_clerk`. | **S6** |
| **QW-7** | Chat votes write the `Vote` table; `api/rag/feedback` has **0 callers** so `rag_feedback` has 0 rows ever → the W9 negative-feedback penalty reads an empty table (`frontend/lib/chat/api.ts:47`). | Wire chat votes to also write `rag_feedback` with the turn's retrieved document ids. | **S7** |
| **P2-03** | The committed prod config can't construct the F005-guarded S3 Vectors backend (`config.py` / `s3_vectors_backend.py:51-55`) → the document-vector plane is *plausibly dark since W2*; if dark, every agent answer over workspace documents is ungrounded. | One read-only AWS probe (is prod dark? what index dimension?), then fix env + re-embed, **or** fold into the Wave-3 Qdrant consolidation (P2-16). Decision gate, not a rebuild. | **S8** |
| **P2-04 / T3** | Quality is a tracked number in 0/28 modules; every eval figure is synthetic/stale/placeholder/one-laptop. | Self-host **Langfuse** (MIT); instrument the two chokepoints (tool dispatch, retrieval); do **not** build an eval platform. | **S9** |
| **P2-04 / T3** | The seed complaint — *"memory saves low-quality memories"* — has never been a number. | Author a ~50-Q workspace memory gold-set (recall@5 / MRR, LongMemEval category shape, offline against a store snapshot) + a with-vs-without task-lift A/B reusing the W7 uplift honest-gate shape. | **S10** |
| **QW-9** | Memory injection has no relevance floor / type filter → 402-spam and recorded lies reach every prompt (`context/sections/memory.py:261-267`); heartbeat probe writes pollute `heartbeat_results` (`heartbeat_service.py:244-254,1049-1089`). | Assembly-side: add a relevance floor + content-type exclusion (`playbook_summary`/`heartbeat_log`). Write-side: move the 30s probe writes out of `heartbeat_results`; stop the daily-summary double-write + fabricated `User:/Assistant:` heartbeat "conversation". *(Assembly floor protects clients now, independent of the Wave-1 memory un-split.)* | **S11** |
| **QW-10** | The Command Center "is-it-working" strip 403s to blank for non-super-admins (`analytics_real.py:38-42`); 3 tracked SLOs render nowhere (`/api/analytics/slos` has no frontend caller). | Split the analytics router so own-workspace health tiles (primitive-health, errors, SLOs, activation) are reachable by workspace admins; wire `/slos` into the strip; add a deliverable-freshness tile (would have caught 06-16 day one). | **S12** |

---

## Stories (test-first — write the failing test, make it green, refactor)

> Grouped by the review's Wave-0 PRD ids (P2-01…P2-05). Each story is independently shippable; see **Sequencing** for the few ordering constraints.

### P2-01 · Reconnect the learning nervous system

**S1 · Telemetry write repair (the type-poison) — S · _the single biggest unblock_**
**Files:** `core/models/composio_cache.py:215`, `modules/tools/execution/telemetry.py:91,107-112`, the chat write site that binds `user_id`.
**Test:** `test_telemetry_write_logged_in_user` drives a tool execution under a Clerk **string** principal and asserts a `ToolExecutionLog` row is actually written (today: 0 rows, swallowed). `test_telemetry_failure_is_loud` asserts an insert failure logs at WARNING, not DEBUG.
**Notes:** Prefer resolving Clerk-id → integer `User.id` at the write boundary over retyping the column; if retyping, it's one alembic migration + a nightly-recompute check. This one line un-starves operating-graph edges, affinities, intent clusters, selection-health, the W7 uplift eval, and SLI-1 **all at once** — do it first.

**S2 · Per-lane telemetry canary — S**
**Files:** new boot-probe + a lightweight scheduled check (reuse the W10 SLO/metrics plumbing); `jobs/` scheduler entry.
**Test:** `test_organic_rows_canary` asserts the canary emits a zero-rows alert when no organic telemetry has landed in the window, and clears once S1's writes flow.
**Notes:** This is the guardrail that stops S1 ever silently regressing. "Organic rows/day = 0" is the alarm the platform lacked for two months.

**S3 · Fail-loud embeddings — S · (Security §5.1 / §content-injection)**
**Files:** `core/llm/clients/base.py:225` (`rng.standard_normal`), `core/llm/embedding_manager.py:90-93`, `modules/rag/service.py:981-983`.
**Test:** `test_embedding_missing_key_fails_loud` asserts a missing/failed embedding key raises (not returns random vectors) on every production selection path; `test_retrieval_failure_is_typed` asserts an embedding error surfaces as `error` (not silently `empty`), with an error-rate metric incremented.
**Notes:** Keep the deterministic provider only behind an explicit test-only fixture, never a production fallback. This is the single biggest **silent client-quality** risk in the review.

### P2-02 · Un-silence the one live autonomous line

**S4 · Playbook failure visibility + circuit-breaker — S**
**Files:** `core/services/notification_dispatcher.py:54-55`, `services/board_task_bridge.py:114`, `api/recipe_executor.py:1588` (`_fail_execution`).
**Test:** `test_playbook_failure_emits_event` runs a step that fails (mock the LLM 402) and asserts (a) a `playbook_failed` notification is dispatched, (b) the board task is `failed` not `done`, (c) after N consecutive failures the breaker pauses re-firing.
**Notes:** The board must stop reporting green over a production outage. The breaker turns "fails daily forever, spams 402" into "fails, alerts, stops."

**S5 · Repair the severed learning imports + import-regression guard — S**
**Files:** `api/recipe_executor.py:1118,1742`, `api/workflow_recipes.py:1231` (repoint `recipe_*` → `playbook_*`).
**Test:** `test_playbook_learning_imports_resolve` imports the previously-broken paths and asserts no `ImportError`; `test_no_dangling_service_imports` walks `core.services.*` references across the tree and asserts each resolves (catches the next silent rename).
**Notes:** A rename + swallowed `ImportError` that survived 3.5 months. The guard test is the durable fix.

### P2-01/P2-04 signal-capture (feed the loops *before* the dashboard)

**S6 · Real chat identity — S · (Security §3.x cheapest trust fix)**
**Files:** `api/chat.py:82-89` (`get_user_id` → `id=1`), `consumers/chatbot/service.py:1268-1275`.
**Test:** `test_chat_uses_real_principal` asserts `chats.user_id`, message saves, and the vote-ownership check all resolve to `ctx.user`, not a constant.
**Notes:** Prerequisite for per-user memory signal and honest attribution; unblocks S7 and the Wave-1 memory work.

**S7 · Give the RAG feedback loop a mouth — S**
**Files:** `frontend/lib/chat/api.ts:47` (vote post), the `/api/rag/feedback` writer, the turn's retrieved-doc-id context.
**Test:** `test_chat_vote_writes_rag_feedback` casts a chat vote and asserts a `rag_feedback` row lands with the retrieved document ids (today: 0 rows ever).
**Notes:** Closes both half-tables (reader-no-writer `rag_feedback` + writer-no-reader `Vote`); turns "learning from retrieval feedback" from fiction into a fed loop. Depends on S6 (real principal).

**S11 · Stop memory injecting noise — S · (Security §content-injection)**
**Files:** `modules/context/sections/memory.py:261-267` (assembly floor); `core/services/heartbeat_service.py:244-254,1049-1089` (write-side probe/double-write/fabricated conversation).
**Test:** `test_memory_injection_relevance_floor` asserts sub-threshold and `heartbeat_log`/`playbook_summary`-typed memories are excluded from assembly; `test_heartbeat_probe_not_persisted` asserts the 30s probe no longer writes `heartbeat_results`.
**Notes:** The assembly-side floor protects clients **now**, independent of the Wave-1 memory un-split — ship it here.

### P2-03 · Verify/relight the document-vector plane

**S8 · Prod document-vector probe + decision — S (probe) → gate**
**Files:** read-only probe script under `orchestrator/scripts/`; `core/vector/s3_vectors_backend.py:51-55`, `config.py` (backend construction).
**Test / deliverable:** a read-only probe that reports (a) can the committed prod config construct the active backend? (b) is the document index populated and at what dimension? Output a written finding.
**Gate:** If dark → fix env + re-embed as a fast follow **or** fold into Wave-3 P2-16 (Qdrant consolidation) — **Gerard's call, surfaced not deferred** (§12). This story gates all RAG-quality work (Wave-1 P2-07): grounding must be proven live first.
**Notes:** Analysis-only probe; no destructive action, no re-embed inside this story without the decision.

### P2-04 · Stand up the eval substrate + the first memory number (T3 Phase-0)

**S9 · Eval substrate (adopt Langfuse) + instrument the two chokepoints — M**
**Files:** self-hosted Langfuse (MIT, self-host) config; trace hooks at the tool-dispatch chokepoint and the retrieval chokepoint (reuse the existing choke-points — do not add parallel ones).
**Test:** `test_dispatch_emits_trace` / `test_retrieval_emits_score` assert a trace/score is emitted at each chokepoint (mock the Langfuse client at the boundary — pure test).
**Notes:** Adopt, do not build. Use RAGAS + DeepEval as metric libraries later. Feeding the loops (S1/S6/S7) is Phase-0 of this — an eval substrate over unfed loops is a beautiful empty dashboard.

**S10 · The first memory eval — M · _the seed complaint becomes a number_**
**Files:** new `orchestrator/evals/memory_recall.py` (+ a ~50-Q workspace gold-set fixture); a non-required CI job; reuse the W7 uplift honest-gate shape.
**Test / deliverable:** recall@5 / MRR over the gold-set run offline against a store snapshot (works during pilot), + a with-vs-without task-lift A/B. Emits a number; **exit-0 always, the number is the deliverable** (do not gate CI red on it).
**Notes:** Do **not** chase LOCOMO (contested; Zep's 84% → 58.44%). LongMemEval-v1 baseline is a later story once the loop is live. This number is the exit criterion the T1 graph-substrate decision (Wave 3 P2-17) is gated on.

### P2-05 · Give the operator eyes

**S12 · Operator cockpit reach + honest tiles — S**
**Files:** `api/analytics_real.py:38-42` (`require_super_admin`), the "is-it-working" strip frontend, `/api/analytics/slos` wiring.
**Test:** `test_workspace_admin_sees_health` asserts a non-super-admin workspace admin can read own-workspace primitive-health/errors/SLOs/activation tiles (today: 403 → blank); `test_slo_strip_wired` asserts the strip renders the 3 tracked SLOs.
**Notes:** De-scope carefully — own-workspace only; cross-workspace/platform analytics stay super-admin. Add a deliverable-freshness tile (the one that would have surfaced the 06-16 outage on day one).

---

## Sequencing (Wave 0 is mostly parallel-safe)

- **S1 → S2** (canary guards the repair) and **S6 → S7** (real principal before feedback rows) are the only hard orderings.
- **S9 (eval substrate) before S10 (memory eval)** — the eval needs the trace surface.
- **S1, S3, S4, S5, S11, S12** are independent and can land in any order / in parallel worktrees.
- **S8 is a probe + a Gerard decision** — it blocks Wave-1 P2-07 (RAG quality), not the rest of Wave 0.
- If built by parallel agents, file ownership is disjoint per the Findings table; the only shared file is `config.py` (S3/S8 new flags) — coordinate flag additions, never `os.getenv` inline.

---

## Verification (CI is the only gate — no local runs)

Per current project convention (`feedback-no-local-servers`, tightened 2026-07-03): **do not run servers, builds, `next dev`, headless Chromium, `pytest`, `tsc`, or installs on the dev machine.** Write the code + **pure** tests (no DB / network / Qdrant / Composio / Langfuse calls — mock at the boundary so they run in CI), commit, push, and let **CI (the PR checks) verify.** Every new test must be runnable with no external service. The eval jobs (S2 canary, S10 memory eval) are **non-required** CI lanes that publish a number and exit 0.

---

## Conventions (non-negotiable — see `automatos-ai/CLAUDE.md`)

- No `os.getenv()` outside `config.py`; new flags go through the canonical config module.
- No backward-compat shims — delete what you replace in the same commit (esp. the `DeterministicEmbeddingProvider` prod path, the dead `recipe_*` import names).
- Immutable patterns; small focused functions; comprehensive error handling; no silent `except` swallows (this wave exists because of two of them).
- No new tables where an existing one fits; no new tools where an existing one extends; reuse the W10 metrics plumbing and the existing dispatch/retrieval chokepoints.
- Canonical vocab: **Playbook** (not Recipe), **Deliverable**, **Knowledge Graph**, **Command Center**, **Auto**.
- Branch `feat/p2-w0-observability-feed-loops`; commit, push, open a PR; CI is the gate.

## Success metrics (the definition of "observable, honest, and fed")

- **Organic telemetry rows > 0** across every execution lane, with a canary that alarms on zero (S1/S2).
- **Retrieval fails loud** — no random-vector fallback anywhere in production; `empty` vs `error` are distinguishable and metered (S3).
- **A failed playbook alerts within the hour and shows `failed`, not `done`;** a repeatedly-failing playbook stops re-firing (S4).
- **Every `core.services.*` import resolves** in a regression test; the playbook learning loop reads what it writes (S5).
- **Chat attributes to the real principal; `rag_feedback` accrues rows** from live votes (S6/S7).
- **Memory injection carries a relevance floor;** heartbeat probe noise no longer pollutes `heartbeat_results` (S11).
- **The document-vector plane is proven live or a re-embed/fold decision is on record** (S8).
- **Memory quality is a tracked number** (recall@5 / MRR + task-lift A/B) published by a CI lane (S9/S10).
- **A workspace admin can see own-workspace health + the 3 SLOs** in the Command Center (S12).

## What this wave gates
Wave 1 (resurrect the dead client-facing loops — memory un-split, RAG quality, Shopify integrity, deliverables) is judged against Wave 0's numbers. Wave 3's T1 graph-substrate trial (P2-17) is **gated on the S10 memory eval** showing measured uplift over the repaired baseline. If only one wave ships, it is this one.

---

*Traceability: every item cites its dossier in `reports/dossiers/` and its quick-win/PRD id in `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` (§4 QW-n, §6 P2-0n). `file:line` refs are from the pinned review tree `77bc9c6d5` (spot-verified during synthesis) — confirm by grep before editing; they may have drifted. North-Star framed; PILOT lens applied; no moat framing.*
