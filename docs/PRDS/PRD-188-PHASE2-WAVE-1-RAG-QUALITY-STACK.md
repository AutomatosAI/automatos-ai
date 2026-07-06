# PRD-188: Phase 2 · Wave 1 — RAG Quality Stack (real hybrid, cheaply)

**Phase:** Phase 2 — Module Deep-Review remediation · **Wave 1** (resurrect the client-facing loops)
**Report id:** **P2-07** · **Branch:** `feat/p2-w1-rag-quality` · **Worktree:** `automatos-ai-p2w1-rag`
**Dependencies:**
- **PRD-185 (Wave 0, `649482aa3`)** — reuse its chokepoints, do **not** rebuild them: **S3** fail-loud embeddings (`empty` vs `error`), **S7** `rag_feedback` writer (`modules/rag/feedback_writer.py`), **S9** the retrieval trace seam (`core/observability/tracer.py::fire_retrieval_score`), **S10** the memory-eval harness shape (`orchestrator/evals/memory_recall.py` + `scripts/eval/memory_recall/{gold_set,corpus}.jsonl`).
- **PRD-186 (S3 Vectors relight — the S8 gate)** — **grounding must be proven LIVE first.** Wave-0 S8 is a probe + a Gerard decision (relight S3 Vectors *or* fold into the Wave-3 Qdrant consolidation). Until the document-vector dense plane returns real hits, every quality lever in this PRD is measured against noise. **This wave does not start until the S8 gate says "the dense plane is live."**
**Build size:** M (the two config flips are S; the BM25 leg, contextual annotations, and the eval are the three M's) · **Risk:** Low–Medium (no store rebuild; the sparse index already exists and is maintained; the dense store is PRD-186's concern)
**Source:** `reports/dossiers/rag-retrieval.md` §C, §D, §E, §J (focus **J3–J6**) + `reports/dossiers/vector-substrate.md` §C.6, §J8; report row **P2-07** in `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md`.

---

## Overview

The retrieval plane has **the best-integrated scaffold in the platform and none of the quality levers turned on.** One fail-closed scope choke point, a shared token budgeter, citation-grade assembly, and real agentic `read_document`/`grep_documents` tools — and then, on top of that good spine: rerank is dark, "hybrid search" is three dataclass fields with no sparse leg behind them, chunks carry no situating context, and **nothing measures whether any of it works.** The dossier's verdict is **EXTEND the spine, ADOPT three specific externals into it** (`rag-retrieval` §E) — this PRD does exactly that, and adds the number that proves it.

Judged against the **North Star** — *does this make Auto more autonomously capable and its output higher-quality for the client?* — this wave turns "the agent grounds itself before acting" from a hopeful sentence into a **measured** one. Every retrieval this module serves is an agent reading workspace knowledge before it writes a Deliverable; each lever here makes that grounding land on the right document more often, and the eval makes "more often" a number instead of a vibe.

The dossier cites Anthropic's contextual-retrieval result — **−35%** retrieval-failure from contextual chunk annotations alone, **−49%** adding a BM25 leg, **−67%** (5.7%→1.9% top-20 failure) once rerank is on ([anthropic.com/engineering/contextual-retrieval](https://www.anthropic.com/engineering/contextual-retrieval), via `rag-retrieval` §D). **These are the dossier's cited figures for that stack, not a promise for this corpus** — the whole point of the eval story (S5) is to publish *our* before/after number rather than inherit theirs.

**The three levers Automatos does not have live (all grep-confirmed against the live tree, below):**
1. **Rerank is off and cannot run** — the pipeline's highest-precision stage has never executed. *(J3)*
2. **Chunks carry no context** — no title/section situating text before embedding; the exact gap contextual retrieval attacks. *(J6-derived)*
3. **"Hybrid" is decorative** — the config knobs exist, the sparse index is *already built and maintained on every insert*, and **nothing reads it.** RRF fuses vector variants against themselves. *(J4)*
Plus the cross-cutting one: **no retrieval eval exists** — no gold set, no recall@k, no faithfulness score; quality is unmeasured. *(J5)*

**PILOT lens (locked):** an empty `rag_feedback` table, a cold gold-set, a zero-traffic Langfuse dashboard are **not** defects and are **not** in scope to "fix by driving usage." In scope is the **wiring** — so that when real traffic arrives, the sparse leg fuses, the reranker runs, the annotations are present, and the eval publishes a number. The deliverable is the *measured uplift over the repaired baseline*, not adoption. See `feedback-pilot-usage-not-quality-signal`.

**Explicitly out of scope for this PRD (surfaced, not deferred — §12):** the *dense* document-vector store itself (S3 Vectors relight vs Qdrant fold) is **PRD-186**, the S8 gate this depends on. The embeddings *settings-plane placebo* (managers reading pre-PRD-136 keys — `vector-substrate` §C.2, incl. `RerankManager` reading `rag_rerank_model`) is the vector-substrate module's own repair, not P2-07 — this PRD notes the one line where it touches the rerank flag and routes the flag through `config.py` correctly, but does not take on the settings-plane rewrite.

---

## Findings & Scope (all `file:line` confirmed by grep against the live tree `automatos-ai/orchestrator/` on 2026-07-06; the review tree drifted — numbers below are re-pinned)

| Finding | Issue (verified in code) | Fix | Story |
|---|---|---|---|
| **J3** (rerank dark) | `enable_reranking` defaults `False` (`modules/rag/service.py:130`), loaded from `rag_rerank_enabled` default `"false"` (`service.py:156`); the UI search endpoint hardcodes `enable_reranking=False` (`api/documents.py:1469`); no `COHERE_API_KEY` in prod env (`config.py:198` reads it, grep count 0 in env). The Cohere integration itself is correct and already gated behind a graceful `is_available()` fallback (`core/llm/rerank_manager.py:82-85`), and the call site is sound (`service.py:695-728`). The highest-precision stage has never run. | Route `rag_rerank_enabled` through `config.py` and default it **on**; remove the hardcoded `enable_reranking=False` at `documents.py:1469`; upgrade the default model to Rerank 4 Fast (config, not code). Cohere remains a graceful-degrade seam — no key ⇒ identity order, never an error. **Gerard's call: Cohere key + vendor/cost** (see open questions). | **S1** |
| **J6** (no chunk context) | Chunker runs `TOPIC_COHERENCE` with `_use_embeddings = False` (`modules/rag/chunking/semantic_chunker.py:80`), 500-char targets; **no chunk-level situating context** is generated anywhere before embedding. The persist-and-embed seam is `modules/rag/ingestion/manager.py:1273 _persist_chunks_and_vectors` (called at `:1174`, inside `_process_document` at `:811`); the chunk text INSERTs are at `manager.py:1310,1342`. | Add contextual chunk annotations at ingestion (Anthropic pattern, not a product): a Haiku-class model prepends a ~50–100-token situating context to each chunk **before** embedding, at `_persist_chunks_and_vectors`. Prompt-cached over the parent doc; ~$1.02/M doc tokens one-time (`rag-retrieval` §D/§H). Gated by a `config.py` flag (default OFF until the corpus is re-annotated — see Sequencing). | **S2** |
| **J4** (hybrid decorative) | `hybrid_search_enabled=True` / `hybrid_vector_weight=0.7` / `hybrid_keyword_weight=0.3` are **dataclass fields only** (`service.py:134-136`) — no sparse/BM25/tsvector leg is read anywhere in `modules/rag/` (grep-proven); `_multi_query_retrieval_with_rrf` (`service.py:636-688`, `rrf_k=60` at `:131`) fuses **vector variants against themselves.** The sparse index **already exists and is maintained**: `document_chunks.search_vector` tsvector + GIN `idx_document_chunks_search_vector` + trigger `trg_document_chunks_search_vector` on every insert/update (`alembic/versions/20260218_rag_v3_hybrid_search_and_feedback.py:28-82`); chunk text is in `document_chunks.content`. The only code that reads `search_vector` today is the **dead** `EnhancedVectorStore` (`modules/search/vector_store/store.py:408-447`, grep-proven zero live callers — `vector-substrate` §C.5). | Add a **real** Postgres BM25 (tsvector `ts_rank`) sparse leg over `document_chunks.search_vector`, workspace-scoped through the **same** `build_retrieval_filters` choke point, and **fuse it into the existing RRF** (dense S3-Vectors list + sparse Postgres list → one `_reciprocal_rank_fuse`). Make the `hybrid_*` weights **real** or delete them — no decorative knobs. Reuse the already-maintained index; do not add a second one. | **S3** |
| **J4 / substrate C.6** (dead sparse infra) | The tsvector trigger pays maintenance cost on **every** chunk insert for a column **nothing live reads** (`vector-substrate` §C.6); the dead `EnhancedVectorStore._hybrid_search` is its only reader (`store.py:408-447`). | S3 makes the trigger's cost *earned* (the live BM25 leg reads it). No separate cleanup story — S3 is the reader that was missing. (The dead `EnhancedVectorStore` deletion belongs to the vector-substrate kill-list, not P2-07 — noted, not owned.) | *(folded into S3)* |
| **J5** (no measurement) | No retrieval gold set, no recall@k, no faithfulness score anywhere (`rag-retrieval` §C-10, §G); the only quantified fact is `rag_feedback = 0 rows`. The online grounding signal now **exists** (Wave-0 S9: `fire_retrieval_score` fires from `service.py:264` with `STATUS_HIT`/`STATUS_EMPTY`/`STATUS_ERROR`), but there is no **offline** recall/precision number and no before/after harness for the levers in S1–S3. | Stand up a retrieval eval that **mirrors the S10 memory-eval shape** (`evals/memory_recall.py`): recall@k / MRR over a ~50-question workspace gold-set against a bundled corpus snapshot (works during pilot), run through `RAGService.retrieve` with each lever **on/off** (rerank, BM25, annotations, enhancement), plus a with-vs-without task-lift shape. Emits a number; **non-required CI lane, exit-0 always** — the number is the deliverable. This is the gate the P2-07 row references. | **S5** |
| **J5-online** (funnel invisible offline) | The S9 score surface distinguishes hit/empty/error over live traffic but is not summarised as a per-lever offline metric; there is no single place the eval reads "was this grounded" for a fixture query. | The eval (S5) computes recall/MRR directly from `retrieve()` results; it also asserts `retrieve()` still emits the S9 score for a fixture query (reuse the seam, prove it, don't add a parallel one). No new online metric — S9 already owns that. | *(folded into S5)* |

---

## Stories (test-first — write the failing test, make it green, refactor)

> Every story reuses the existing retrieval choke point (`build_retrieval_filters`), the existing RRF (`_multi_query_retrieval_with_rrf`), the existing rerank seam (`rerank_manager`), the S9 trace hook, and the S10 eval shape. **No parallel chokepoints, no parallel eval, no new tables** (the `search_vector` column and `rag_feedback` already exist). All new tests are **pure** — mock at the boundary (no S3 / Cohere / Postgres / OpenRouter / Langfuse calls), runnable in CI with no external service.

### S1 · Turn on reranking (config flip + un-hardcode) — S

**What:** flip the pipeline's highest-precision stage from dark to live. The Cohere integration and call site are already correct and already degrade gracefully without a key (`rerank_manager.py:82-85`, `service.py:695-728`); the only reason it never runs is two config values and one hardcode.

**Files:**
- `orchestrator/config.py` — add/confirm the canonical `RAG_RERANK_ENABLED` accessor (already present at `config.py:998-1005`) is the single source; add a `RAG_RERANK_MODEL` default (`rerank-3.5-fast` / Rerank 4 Fast) so the model default lives in `config.py`, not in `rerank_manager.py:24`'s hardcode. **No `os.getenv` outside `config.py`.**
- `orchestrator/modules/rag/service.py:130,156` — `RAGConfig.enable_reranking` defaults from the canonical config accessor and defaults **on** (was `"false"`).
- `orchestrator/api/documents.py:1469` — delete the hardcoded `enable_reranking=False`; let the config decide.

**Test:** `test_rerank_enabled_by_default` asserts a fresh `RAGConfig` has `enable_reranking is True` (today: `False`). `test_rerank_runs_when_key_present` drives `_rerank_candidates` with a **mocked** `RerankManager.is_available()→True` and asserts candidates are reordered by `rerank_score`; `test_rerank_degrades_without_key` asserts `is_available()→False` returns the input order **unchanged and without error** (the graceful path stays graceful). `test_search_endpoint_does_not_force_rerank_off` asserts the `documents.py` search path no longer passes `enable_reranking=False`.

**Notes:** Cohere is a **graceful-degrade seam** — with no key it is identity ordering, never a failure, so flipping the default on is safe even before the key exists. The dossier's measured leg is **−67% top-20 failure with rerank on** (`rag-retrieval` §D), *gated by S5's eval to confirm it here.* This story does **not** repair the `RerankManager` settings-plane bug (`vector-substrate` §C.2 — it reads pre-PRD-136 `rag_rerank_model`); that is the vector-substrate module's PRD. Here the flag flows through `config.py`, which is the correct source regardless.

### S2 · Contextual chunk annotations at ingestion — M

**What:** give each chunk a situating context so the embedding (and the BM25 leg) know what the chunk is *about* beyond its 500 characters. Anthropic's pattern: a cheap model reads the parent document and writes a ~50–100-token preface per chunk, prepended before embedding (`rag-retrieval` §D, §E "Adopt 2").

**Files:**
- `orchestrator/modules/rag/ingestion/manager.py:1273` (`_persist_chunks_and_vectors`, called at `:1174` inside `_process_document` at `:811`) — before the chunk goes to `EmbeddingManager`, generate + prepend the annotation. New small module `orchestrator/modules/rag/ingestion/contextual_annotator.py` (immutable: takes chunk + parent text, returns a *new* annotated chunk; does not mutate).
- `orchestrator/config.py` — `RAG_CONTEXTUAL_ANNOTATIONS_ENABLED` (default **OFF** until the corpus is re-annotated), `RAG_CONTEXTUAL_ANNOTATION_MODEL` (Haiku-class id).
- Store the annotation on the chunk (`document_chunks.metadata` JSONB — existing column, no new table) so it is inspectable and re-annotation is idempotent.

**Test:** `test_contextual_annotation_prepended_before_embedding` asserts that with the flag on, the text handed to a **mocked** `EmbeddingManager.generate_embedding` starts with the annotation and contains the original chunk text (mock the annotation LLM at the boundary — no network). `test_annotation_disabled_is_passthrough` asserts flag-off embeds the raw chunk unchanged (byte-identical). `test_annotation_failure_is_loud_not_silent` asserts an annotation-LLM error surfaces (logged at WARNING and the chunk falls back to raw text with a recorded flag) — **no silent `except` swallow** (this module has two such swallows already; do not add a third).

**Notes:** One-time cost ~$1.02/M doc tokens with prompt caching over the parent doc (`rag-retrieval` §H). Re-annotating the existing 19,130-chunk corpus runs through the **existing** background reprocess path (`api/documents.py` reprocess, per `rag-retrieval` §J6) — **not a new pipeline.** The flag stays OFF until that reprocess has run, so live retrieval never sees a half-annotated corpus. Immutable pattern: `contextual_annotator` returns new chunk objects.

### S3 · The real BM25 sparse leg, fused into RRF — M · _the "hybrid is decorative" fix_

**What:** add the sparse retrieval leg the config has always advertised, reading the tsvector index the platform **already builds and maintains on every insert**, and fuse it with the dense S3-Vectors leg through the RRF that already exists — turning client-side vector-only "fusion" into real dense+sparse hybrid.

**Files:**
- new `orchestrator/modules/rag/bm25_leg.py` — a workspace-scoped `ts_rank` query over `document_chunks.search_vector` using `plainto_tsquery('english', …)` (the same tsvector the trigger maintains, `20260218_rag_v3_...py:28-82`), returning `[{id, content, score}]` in the **same shape** `_get_candidates` returns so the fuser is leg-agnostic. Scope **through** `build_retrieval_filters` (`modules/rag/retrieval_filters.py`) — the one fail-closed choke point; no new scoping logic.
- `orchestrator/modules/rag/service.py` — extract the RRF math out of `_multi_query_retrieval_with_rrf` (`:636-688`) into a small pure `_reciprocal_rank_fuse(ranked_lists: list[list[dict]], k) -> list[dict]` and call it with **[dense_list, sparse_list]** when `hybrid_search_enabled`; make `hybrid_vector_weight`/`hybrid_keyword_weight` (`:134-136`) **actually weight** the fusion (or delete them and use plain RRF — no decorative knobs). The existing multi-query RRF becomes one caller of the same pure function.
- `orchestrator/config.py` — `RAG_HYBRID_ENABLED` (canonical accessor; the dataclass field defaults from it). No `os.getenv` in `service.py`.

**Test:** `test_reciprocal_rank_fuse_pure` — table-driven, feeds two hand-built ranked lists and asserts the fused order matches hand-computed RRF (the math is pure, no DB). `test_hybrid_fuses_dense_and_sparse` mocks the dense `_get_candidates` and the sparse `bm25_leg.search` and asserts the final candidate set is the RRF of **both** legs (today: dense only). `test_bm25_leg_is_workspace_scoped` asserts `bm25_leg.search` routes through `build_retrieval_filters` and a missing workspace fails closed to `[]` (mirrors the dense choke point; mock the DB boundary). `test_hybrid_disabled_is_dense_only` asserts flag-off preserves today's exact behaviour.

**Notes:** This is a **wire-up, not a build** — the sparse index, GIN, and maintenance trigger already exist and are already paid for on every insert; the only thing missing was a live reader (`vector-substrate` §C.6). Exact-term queries (SKUs, error codes, names — the widget/commerce cases) are precisely where pure-vector fails hardest and BM25 wins (`rag-retrieval` §J4). The dossier's measured leg is **−49% failure with BM25 added to contextual embeddings** (`rag-retrieval` §D), *gated by S5.* Fusing dense-S3 + sparse-Postgres is the **hybrid design this PRD locks** (see open questions — client-side RRF over the two legs, since S3 Vectors carries no server-side fusion, per `vector-substrate` §D). **Do not** resurrect `EnhancedVectorStore`; the new `bm25_leg.py` is a clean, small reader.

### S5 · The retrieval eval — the number the row is gated on — M · _non-required lane, publishes a number_

**What:** the gate the P2-07 row references. A retrieval quality number, before/after each lever, that mirrors the Wave-0 memory eval so there is **one** eval idiom in the codebase, not two.

**Files:**
- new `orchestrator/evals/retrieval_recall.py` — **modelled on `orchestrator/evals/memory_recall.py`**: `argparse`, `--json`, bundled corpus snapshot default, recall@1/@3/@5 + MRR dataclasses, a `passes()` honest-gate against a published `RECALL_AT_5_TARGET`, and **`exit 0` even when sub-threshold** (the number is published, not massaged — mirrors `memory_recall.py:133-134` and its honesty note).
- new `orchestrator/scripts/eval/retrieval_recall/{gold_set.jsonl,corpus.jsonl}` — ~50 `question → relevant document_id` pairs authored from the real 644-doc / 19,130-chunk corpus shape, with same-topic distractors so recall is honest (mirror the `memory_recall` fixture pattern exactly).
- runs `RAGService.retrieve` with each lever **on/off** (rerank S1, BM25 S3, annotations S2, query-enhancement) so the eval **also finally answers "does HyDE/RRF pay for its 4 LLM calls"** with a number (`rag-retrieval` §G-2).
- a **non-required** CI job (like the S10 memory-eval and the NL2SQL harness) that publishes the number and exits 0.

**Test:** `test_recall_at_k_pure` / `test_mrr_pure` — feed hand-built ranked lists + gold labels and assert recall@k and MRR match hand-computed values (pure math, no retrieval). `test_eval_exit_zero_below_target` asserts a sub-threshold run still exits 0 and reports the number (the lane never reds CI). `test_eval_reuses_retrieve_and_fires_s9_score` runs one fixture query through a **mocked** `retrieve()` and asserts the eval reads its `RAGResult` and that the S9 `fire_retrieval_score` seam is still exercised (reuse, don't parallel).

**Notes:** Offline against a bundled snapshot so it **works during pilot** (no dependence on live traffic — PILOT lens). This number is what proves S1/S2/S3 earned their cost on *this* corpus rather than inheriting the dossier's cited figures; it is the before/after every other story reports against, and the exit criterion Wave-3 quality decisions read. Do **not** build a bespoke LLM-judge first (`rag-retrieval` §E "Adopt 3" — RAGAS/DeepEval faithfulness is a *later* story once the loop is live). **Non-required lane, exit-0 always** — CI red is never gated on the number.

---

## Sequencing

- **PRD-186 S8 gate first (hard blocker).** Nothing in this wave starts until the dense document-vector plane is proven live (relit S3 Vectors, or the Qdrant fold decision made). A quality stack measured against a dark plane is a beautiful lie — this is stated in the Wave-0 S8 gate and repeated here.
- **S5 (eval) is the spine** — stand it up **first among this PRD's stories** (even before S1/S2/S3 land) so every lever reports a before/after against it. It has no dependency on the others; the others are judged by it.
- **S1 (rerank flip)** is independent and lands any time — it is the cheapest lever (config only) and its uplift is read off S5.
- **S3 (BM25 leg)** is independent of S1/S2 in code (different files) and reads the already-maintained index; land it whenever, measure via S5.
- **S2 (annotations)** ships **flag-OFF**; the flag only flips **after** the existing background reprocess has re-annotated the corpus (so live retrieval never sees a half-annotated store). The re-annotation run is operational, gated by the flag — surfaced to Gerard, not auto-run in CI.
- The only shared file is `config.py` (S1/S2/S3 each add a flag) — coordinate additions, never `os.getenv` inline. `service.py` is touched by S1 (flag) and S3 (fuser extraction) — S3 owns the RRF refactor; S1 only touches the rerank default line.
- If built by parallel agents, file ownership is otherwise disjoint per the Findings table.

---

## Verification (CI is the only gate — no local runs)

Per current project convention (`feedback-no-local-servers`, tightened 2026-07-03): **do not run servers, builds, `next dev`, headless Chromium, `pytest`, `tsc`, or installs on the dev machine.** Write the code + **pure** tests (no S3 / Cohere / Postgres / OpenRouter / Langfuse / Qdrant calls — mock at the boundary so they run in CI with no external service), commit, push, and let **CI (the PR checks) verify.** Every new test must be runnable with no external service. The **retrieval eval (S5) is a non-required CI lane that publishes a number and exits 0** — CI red is never gated on the recall figure; the number is the deliverable, exactly as the Wave-0 S10 memory eval.

---

## Conventions (non-negotiable — see `automatos-ai/CLAUDE.md`)

- No `os.getenv()` outside `config.py`; the three new flags (`RAG_RERANK_MODEL` default, `RAG_CONTEXTUAL_ANNOTATIONS_ENABLED`, `RAG_HYBRID_ENABLED`) go through the canonical config module.
- **No backward-compat shims — delete what you replace in the same commit:** the hardcoded `enable_reranking=False` (`documents.py:1469`) goes; the decorative `hybrid_*` weights either become real or are deleted (no dead knobs); **do not** leave the old vector-only "fusion" path beside the new hybrid one — the RRF fuser becomes the one path both callers use.
- **No new tables, no new indexes** — `document_chunks.search_vector` (+ its GIN + trigger) and `rag_feedback` already exist; S3 reads the index that is already maintained. No new tool where an existing one extends.
- **Reuse the existing chokepoints — do NOT add parallel ones:** `build_retrieval_filters` (scope), `_reciprocal_rank_fuse` extracted from the existing RRF (fusion), `rerank_manager` (rerank), `fire_retrieval_score` (S9 online score), `evals/memory_recall.py` shape (S5 offline eval), `feedback_writer.py` (S7's `rag_feedback` writer — the ranking-signal reader, untouched here).
- Immutable patterns (the annotator returns new chunk objects); small focused functions (`_reciprocal_rank_fuse`, `bm25_leg.search`, `contextual_annotator`); **no silent `except` swallows** — this wave exists partly because retrieval already eats errors into empty results (`rag-retrieval` §C-2).
- Canonical vocab: **Playbook**, **Deliverable**, **Knowledge Graph**, **Command Center**, **Auto**.
- Branch `feat/p2-w1-rag-quality`; worktree `automatos-ai-p2w1-rag`; commit, push, open a PR; CI is the gate.

## Success metrics (the definition of "real hybrid, measured")

- **Rerank runs** on every retrieval when a key is present, and degrades to identity order (no error) when it is not; the default is on, the hardcoded off is gone (S1).
- **Chunks carry a situating annotation** before embedding when the flag is on; the corpus is re-annotated via the existing reprocess path; annotation failure is loud, not silent (S2).
- **Retrieval is genuinely hybrid** — a Postgres BM25 leg over the already-maintained `search_vector` fuses with the dense S3-Vectors leg through one RRF; the `hybrid_*` knobs are real or gone; exact-term queries return the right chunk (S3).
- **Retrieval quality is a published number** — recall@5 / MRR over a workspace gold-set, before/after each lever, on a non-required CI lane that exits 0; the number (not the dossier's cited figure) says whether the stack earned its cost on this corpus (S5).
- **The dossier's cited −49%…−67% failure-reduction stack is either confirmed or corrected by our own eval** — the honest outcome, whichever way it lands (a flat number is a valid result under the PILOT lens: "the levers didn't beat plain dense search *here*," not "moat unproven").

## What this wave gates

The retrieval eval (S5) becomes the before/after instrument for **every** later RAG change — the BM25 fusion tuning, a groundedness/faithfulness score (RAGAS/DeepEval, a later story), any embedding-model or dimension change (shared with the vector-substrate eval, `vector-substrate` §G-3). Together with the Wave-0 S10 memory eval, it is half the evidence base the **Wave-3 T1 graph-substrate decision (P2-17)** reads: quality must be a measured number over the repaired baseline before that investment is judged. It also finally answers, with a number, whether query enhancement's 4 sequential LLM calls pay for themselves (`rag-retrieval` §G-2, §H) — an input to the latency/cost cleanups in `rag-retrieval` §J9.

---

*Traceability: every story cites its dossier ref (`reports/dossiers/rag-retrieval.md` §J3–J6 / §C-D-E-G-H; `reports/dossiers/vector-substrate.md` §C.6, §J8) and the report id **P2-07** in `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md`. `file:line` refs were re-pinned by grep against the live tree on 2026-07-06 (the review tree drifted from `77bc9c6d5`); confirm again before editing if the tree moves. Depends on **PRD-185 (Wave 0, `649482aa3`)** chokepoints (S3/S7/S9/S10) and **PRD-186 (S3 Vectors relight — the S8 grounding gate)**. North-Star framed; PILOT lens applied; the −49%…−67% figures are the dossier's cited retrieval-failure reductions (`rag-retrieval` §D), attributed there, to be confirmed or corrected by S5's own number — no moat framing.*
