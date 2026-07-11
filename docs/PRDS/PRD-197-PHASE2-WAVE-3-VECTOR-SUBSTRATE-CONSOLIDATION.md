# PRD-197 — Phase-2 Wave-3: Vector-substrate consolidation (retire S3 Vectors, one dimension authority, kill the zombies, snapshots + SLO) — P2-16

> **Status:** DRAFT for review — spec only, **no build yet** (Gerard is baselining Auto first; this PRD is written to be measured against that baseline).
> **Review id:** P2-16 · **Dossier:** `reports/dossiers/vector-substrate.md` J3–J7 + C.1/C.3/C.5/C.7, thesis-T2 · **Grounded @ `main` 119cc0dc3** (post Wave-2; refs re-confirm at build).

---

## 1. What this is

A **subtraction-and-truth** PRD, not a new engine. The vector substrate's *ideas* are right (one embedding seam, one cheap good model, payload-partitioned Qdrant); its *operation* is not — a document plane that was dark, dimension knobs that disagree, a settings card that does nothing, a destructive migration footgun, and a zombie store with wrong math that still has callers. This PRD makes the substrate **honest and single**: retire the dead S3 Vectors plane, make one dimension authority impossible to violate (CI-gated), migrate the last callers off the zombie store and delete it, back up the Qdrant planes, and put a number on the substrate so the next dark-plane incident is caught in hours, not weeks.

**Framing (CLAUDE.md §3):** this is **Refactor / Consolidation** — pick the canonical path (pgvector for documents, already live post-PRD-188; Qdrant for memory/field), migrate the losers, **delete them** (§5). Almost nothing here is net-new code; the net-new is one CI gate, one snapshot job, and one telemetry seam.

**Build size:** M (mostly deletion + caller migration + one CI gate + one cron) · **Risk:** Medium (deleting a store with live callers — the caller migration is the real work and must be grep-proven complete before deletion; the document plane itself does **not** move, so retrieval quality is held constant by design).

**Why now / why measured:** the whole point of Wave 0 was to make quality a number. This PRD changes the retrieval **substrate** without intending to change retrieval **quality** — so it is the cleanest possible thing to build against a live baseline: recall@k and latency should be **flat or better** across every story. If a story moves recall, that is a regression signal, not a feature. Baseline-capture is a first-class section below.

---

## 2. Current reality (grounded, not the review's July snapshot)

- **Documents run on pgvector, live.** `document_chunks` (~19k chunks, dim 2048 per the PRD-185 S8 probe) is the live document plane; PRD-188 built the hybrid (BM25 leg + Cohere rerank) **on it** (`modules/rag/` — `service.py`, `bm25_leg.py`, `fusion.py`, `retrieval_filters.py`). This plane is **not** dark and **does not move** in this PRD.
- **S3 Vectors is dead.** `modules/search/vector_store/backends/s3_vectors_backend.py` (+ `_mock`), selectable via `get_vector_store(backend="s3_vectors")` (`modules/search/vector_store/__init__.py`); the PRD-185 S8 probe found it **dark** (bucket missing the `{workspace_id}` templating); `scripts/migrate_to_s3_vectors.py` is **destructive** (wipes pgvector before copying — a footgun); `scripts/recreate_s3_index.py` exists; PRD-186 (relight S3) was **retired** in favour of staying on pgvector. Nothing in production reads this plane.
- **The zombie store still has callers.** `EnhancedVectorStore` / `SearchService` / `ContextRetrievalEngine` (`modules/search/vector_store/store.py`, `modules/search/service.py`, `modules/search/retrieval/context_retrieval_engine.py`) — the F079 trio the review flagged (its namesake table was dropped in PRD-135; its "cosine" uses the L2 operator). Live importers outside `modules/search/`: `api/documents.py`, `api/cloud_documents.py`, `modules/rag/ingestion/manager.py`, `modules/codegraph/codegraph_service.py`, `modules/nl2sql/service.py`, `modules/tools/discovery/handlers_documents.py`, `api/context_policy.py`. **Deletion requires migrating these first.**
- **Dimension knobs disagree.** `config.py:911 S3_VECTORS_DIMENSION=2048`, `config.py:958 FIELD_EMBEDDING_DIM=2048`, and `modules/rag/config.py:93` reads the embedding dimension **from system settings** — three independent sources that happen to agree at 2048 today and will silently diverge the day someone changes one.
- **No Qdrant backups.** Field memory (`field_memory`, PRD-166/108) and durable memory (`durable_memory`, PRD-187) live only on the Railway Qdrant with no snapshot — unrecoverable on loss.
- **The eval instrument exists.** `evals/retrieval_recall.py` + `scripts/eval/retrieval_recall/` + the CI `Retrieval-recall eval` lane + the PRD-188 live gold set — this is the measurement instrument this PRD is verified against.

---

## 3. Findings → fix → story

| # (dossier) | Finding (grounded) | Fix | Story |
|---|---|---|---|
| **C.3 / J4 (dimension authority)** | Three independent dimension knobs (`S3_VECTORS_DIMENSION`, `FIELD_EMBEDDING_DIM`, `modules/rag/config.py` system-setting) can silently diverge → a mismatch is discovered at query time as garbage, not at boot. | **One authority** — everything derives from the canonical `(embeddings, dimensions)` system setting; delete the independent knobs; add a **config-integrity CI gate** that fails the build if any plane's declared dimension disagrees with the authority (the same fail-loud posture PRD-192 gave the policy plane). | **S1** |
| **C.1 / J1-fold (retire S3)** | The S3 Vectors plane is dark and was chosen to be *folded*, not relit (PRD-186 retired). The backend, the mock, the **destructive** migrate script, and the boot-abort-on-bad-S3 all still ship — dead weight and a footgun. | **Retire it** (§5 deletion-led): remove the `s3_vectors` / `s3_vectors_mock` backends from the factory and delete their modules; delete `scripts/migrate_to_s3_vectors.py` (destructive) and `scripts/recreate_s3_index.py`; remove the S3-bucket boot-abort (the plane it guarded is gone). `pgvector` becomes the only document backend the factory offers. | **S2** |
| **C.5 / E.4 / J9 (kill the zombies)** | The F079 trio (`EnhancedVectorStore`/`SearchService`/`ContextRetrievalEngine`) is wrong-math dead code that still has 7 live importers — a store future code could resurrect. | **Migrate then delete:** repoint every importer onto the canonical `modules/rag/` pgvector path (documents) or the appropriate Qdrant adapter (memory/field), **grep-prove zero importers**, then delete the three modules + the dead `modules/search/vector_store/store.py`. | **S3** |
| **C.2 / J3 (settings truth)** | The embeddings admin card and the manager reads don't agree on canonical `(embeddings, *)` keys; `modules/memory/__init__` has stale column names; the UI card is a settings-placebo. | Managers read the canonical `(embeddings, *)` keys through the config module; fix the `modules/memory/__init__` column names; **truthify or delete** the UI card (honest OFF over silent placebo — the PRD-196 lesson). | **S4** |
| **C.7 / J7 (backups)** | Qdrant `field_memory` + `durable_memory` have no snapshot → unrecoverable on loss. | **Qdrant snapshot cron** on the existing memory-jobs scheduler pattern + a one-line runbook entry; retention per Gerard's call. | **S5** |
| **G.1 / I.2 / J6 (a number on the substrate)** | No telemetry watches candidates/latency/errors per query seam → the next dark-plane goes unnoticed for weeks (as C.1 did). | A substrate-telemetry seam (candidates returned, latency, errors per retrieval) + a Command Center tile, reusing the PRD-185 observability plane. **This is the watchdog that makes every future substrate change measurable** (and feeds §7 baseline capture). | **S6** |

---

## 4. Stories (test-first; CI is the only gate — no local runs)

### S1 · One dimension authority + config-integrity CI gate — S · _vector-substrate C.3/J4_
Everything that needs an embedding dimension derives it from the canonical `(embeddings, dimensions)` system setting (the `modules/rag/config.py:93` source, lifted to a single accessor in the config module). Delete `S3_VECTORS_DIMENSION` (`config.py:911`) and `FIELD_EMBEDDING_DIM` (`config.py:958`) as independent knobs; the field adapter and any remaining consumer read the one authority. Add a **pure config-integrity test** (CI-gated) that constructs each live plane's dimension and asserts they all equal the authority — a divergence **fails the build**, not a query.
**Test:** `test_dimension_authority_single_source` (all planes resolve to the authority); `test_config_integrity_gate_fails_on_divergence` (a seeded mismatch raises, greppably). Pure — no DB/network.
**Notes:** No `os.getenv` outside `config.py`. When a knob is deleted, repoint every reader in the same commit (grep first — recurring F070 lesson).

### S2 · Retire the dark S3 Vectors plane — S · _vector-substrate C.1/J1-fold_
Remove `s3_vectors` and `s3_vectors_mock` from the `get_vector_store` factory (`modules/search/vector_store/__init__.py`) and delete `backends/s3_vectors_backend.py` + `backends/s3_vectors_mock.py`. Delete `scripts/migrate_to_s3_vectors.py` (destructive) and `scripts/recreate_s3_index.py`. Remove the S3-bucket boot-abort added by #512 (the plane it guards no longer exists). `probe_document_vectors.py` is retained **only** if it still reports pgvector health after S3 is gone (else deleted with it).
**Test:** `test_factory_offers_only_pgvector` (requesting a retired backend raises a clear error, not a silent fallback); `test_no_s3_vectors_imports_remain` (source-grep guard). Pure.
**Notes:** Deletion-led, no `_legacy` shim (§5). If Gerard wants a future AWS-native option it is a *new* PRD against a *working* design, not this dead code kept on life support (Open questions Q2).

### S3 · Migrate callers off the zombie store, then delete it — M · _vector-substrate C.5/E.4/J9_
Repoint the 7 live importers (`api/documents.py`, `api/cloud_documents.py`, `modules/rag/ingestion/manager.py`, `modules/codegraph/codegraph_service.py`, `modules/nl2sql/service.py`, `modules/tools/discovery/handlers_documents.py`, `api/context_policy.py`) onto the canonical path: document reads/writes → `modules/rag/` pgvector; any memory/field use → the Qdrant adapters. **Grep-prove zero importers** of `EnhancedVectorStore`/`SearchService`/`ContextRetrievalEngine`, then delete `modules/search/vector_store/store.py`, `modules/search/service.py`, `modules/search/retrieval/context_retrieval_engine.py` (and any now-orphaned `modules/search/` siblings).
**Test:** per-caller behavioural test that the migrated path returns equivalent results (fixture-fed, mocked at the boundary); `test_no_zombie_store_importers` (source-grep guard). Pure.
**Notes:** This is the story that can move a number if done wrong — each caller migration is verified equivalent against the retrieval eval (§7). If a caller's semantics genuinely differ from the canonical path, that is surfaced as a finding, not silently flattened (§12).

### S4 · Settings-plane truth — S · _vector-substrate C.2/J3_
Managers read canonical `(embeddings, *)` keys through the config module; fix the stale `modules/memory/__init__` column names; the embeddings admin card either reflects real state or is deleted (no placebo — the PRD-196 honest-UI lesson). Update the route-manifest only if a route changes.
**Test:** `test_embeddings_settings_roundtrip` (a written setting is the one the manager reads); `test_no_stale_memory_init_columns`. Pure.

### S5 · Qdrant snapshots — XS · _vector-substrate C.7/J7_
A snapshot job for `field_memory` + `durable_memory` on the existing memory-jobs scheduler pattern (`services/memory_jobs.py`, wired at `main.py`); cadence + retention + destination per Gerard's call (Q4). One runbook line documents restore.
**Test:** `test_snapshot_job_registered`; `compute_next_snapshot` pure helper tested. No live Qdrant in tests.

### S6 · Substrate telemetry → SLO tile — S · _vector-substrate G.1/I.2/J6_
A telemetry seam recording candidates-returned / latency / errors per retrieval seam (documents + memory + field), through the PRD-185 observability plane; a Command Center tile surfaces it. **This is the instrument that makes §7's baseline continuous** — after this story, "is retrieval healthy?" is a live number, and the next dark-plane trips a tile, not a user complaint.
**Test:** `test_substrate_telemetry_records_candidates_latency`; tile renders the healthy + degraded states. Pure/mocked.

---

## 5. Sequencing
- **S1 first** (dimension authority) — everything else assumes one dimension. **S2** (retire S3) is independent and parallel-safe. **S3** (kill zombies) is the long pole — it can land per-caller but the deletion is one commit after the last caller moves. **S4** parallel. **S5/S6** independent, land any time.
- **S6 ideally lands early** if Gerard wants the live substrate tile feeding the baseline while he runs Auto (see §7) — it is the one story whose *output is the measurement instrument*, so pulling it forward makes the rest of the wave measurable in-flight.
- No migration except S1's (if the dimension authority needs a settings backfill) and none that moves document data — the document plane is held constant on purpose.

## 6. Verification (CI is the only gate — no local runs)
Per `feedback-no-local-servers`: **no servers, builds, `pytest`, `tsc`, or installs on the dev machine.** Pure tests (mock Qdrant/pgvector/session at the boundary); the config-integrity gate + the two source-grep guards (no-S3, no-zombie-importers) follow the PRD-185 S5 import-regression shape — **repoint a guard in the same commit its symbol moves.** The **Retrieval-recall eval CI lane is the quality guard**: recall@k must be **flat-or-better** vs the frozen baseline (§7) at every story. Any new route → update the committed `orchestrator/reports/route-manifest.json`. Any migration authored in-PR; Gerard applies (note: the deploy entrypoint runs `alembic upgrade heads` on boot — a merged migration self-applies).

## 7. Baseline capture — run Auto, then measure (the point of building this now)

This PRD is designed to be built **against a live baseline**, because it changes the substrate while holding quality constant — so the measurement is the proof it worked.

**Capture NOW, while Auto runs (before any S-story):**
1. **Retrieval recall@k** on the PRD-188 live gold set — freeze the current numbers as `baseline/retrieval_recall_2026-07.json`. This is the single most important before-number: every story must hold it flat-or-better.
2. **Retrieval latency + candidate counts** per seam (documents / memory / field) — from real traffic. **Turn on the Wave-0 telemetry to get this:** set `TRACING_ENABLED` (PRD-185 S9 Langfuse seam, default OFF) so real Auto usage is traced, and let S6's substrate tile (if pulled forward) accumulate the live distribution.
3. **Dark-plane confirmation** — re-run `probe_document_vectors.py` once to record "pgvector live, S3 dark" as the documented pre-state (so the S2 retirement is provably removing a dead plane, not a used one).
4. **Zombie-store call volume** — before S3, log how often each of the 7 importers actually hits the zombie path in real usage (confirms the migration surface is exercised, not theoretical).

**Success = the delta (measured, not asserted):**
- Recall@k: **flat-or-better** at every story (a drop is a regression, not a feature). ← the core guard.
- Engines: **one fewer** (S3 gone); zombie importers **0** (grep-proven); dimension knobs **1** (was 3).
- Substrate observability: from **0 numbers** to a live tile (S6) — the next dark-plane caught in hours.
- Backups: field + durable memory **recoverable** (was unrecoverable).
- Local edition: open-core RAG **constructs and returns** on a fresh clone (was shipping a broken S3 default) — the P2-23 fresh-clone lane will confirm.

*(The eval-gated Wave-3 bets — Graphiti P2-17, NL2SQL P2-18 — consume this same baseline discipline. Freezing the retrieval baseline now serves the whole wave, not just this PRD.)*

---

## 8. Open questions — Gerard's call (decide, don't let me defer — CLAUDE.md §12)

1. **The consolidation fork (the big one).** The review's E.2 proposed moving document vectors **onto Qdrant** (one engine for all vectors, Qdrant Query-API RRF hybrid, snapshots for docs too). But PRD-188 already built the document hybrid (BM25 + rerank) **on the live pgvector plane**, and it works. **Recommendation: keep pgvector for documents** — retire S3, delete zombies, snapshot the Qdrant *memory* planes, and do **not** migrate document data (least risk, holds recall constant, pgvector hybrid is already shipped). **Alternative:** move docs to Qdrant anyway for true single-engine + Query-API RRF — more work, a re-embed/backfill risk, and it would re-open the recall number this PRD is trying to hold flat. Which posture ships? *(My strong lean: pgvector-for-docs; the "one engine" purity isn't worth re-litigating a working retrieval plane.)*
2. **S3 artifacts: delete outright, or keep behind a flag for a future AWS-native option?** Recommendation: **delete** (the plane never worked; the migrate script is destructive; §5 no-shims). A future AWS-native vector option, if ever wanted, is a new PRD against a working design.
3. **Zombie kill-list scope.** Migrate all 7 callers + delete the trio **in this PRD** (recommended — the point is the subtraction), or split the caller migration across follow-on slices? If any caller's semantics genuinely differ from the canonical path, I surface it as a decision, not a silent flatten.
4. **Qdrant snapshot cadence / retention / destination.** Proposal: **daily snapshot, 7-day retention, to the same object store the platform already uses.** Adjust cadence/retention, and confirm the destination (Railway volume vs S3 bucket vs Qdrant Cloud snapshot).
5. **Baseline metric set (§7).** Confirm the four capture items (recall@k on the live gold set, latency+candidates per seam, dark-plane probe, zombie call volume) are the right before-numbers, and **whether to pull S6 forward** so the substrate tile is live while you run Auto (recommended — it makes the rest of the wave measurable in-flight).

---

*Traceability: every story cites `reports/dossiers/vector-substrate.md` (C.1 dark plane / C.2 settings / C.3 dimensions / C.5 zombies / C.7 backups; J3–J7 + J9 upgrade rows; E.2 consolidate / E.4 kill-list; G.1 metric) and thesis-T2 (stay modular-monolith, one engine less), under review id **P2-16** in `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` §6 (Wave 3). All `file:line` refs re-confirmed by grep against `main @ 119cc0dc3` (post Wave-2: documents live on pgvector via PRD-188; S3 dark per the PRD-185 S8 probe; PRD-186 relight retired) and are current as written. Reuses PRD-185 (observability/telemetry seam), PRD-187 (durable Qdrant), PRD-188 (pgvector hybrid — held constant), PRD-166/108 (field Qdrant). Feeds P2-17/P2-18 (they consume the frozen retrieval baseline). North-Star framed; PILOT lens applied; measurement-forward per Gerard's build-and-measure intent; no moat framing.*
