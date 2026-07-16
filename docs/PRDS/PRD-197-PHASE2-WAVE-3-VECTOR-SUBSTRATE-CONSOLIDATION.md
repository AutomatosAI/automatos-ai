# PRD-197 (Revised 2026-07-14) — Vector-substrate consolidation **reslimmed**: kill the F079 zombie store · settings-truth · Qdrant-memory snapshots · substrate telemetry · open-core local-RAG — P2-16

> **Status:** DRAFT (revised) — spec only, no build yet. **The original 197 was written on an INVERTED premise** ("retire dead S3, keep pgvector"). In fact **S3 Vectors is the LIVE document plane** — the #514 pgvector fallback was reverted (`fa2987ad3`) and the shared bucket fixed (`e2c86f6bd`); pgvector `document_chunks` is now content-hydration only. This revision **drops the two wrong stories** (dimension-authority → now owned by the revised **PRD-186**; retire-S3 → *deleted*, it would kill the live backend) and **adds** the open-core local-RAG gap the revert exposed. **Pairs with PRD-186 (S3 hardening).** Grounded @ `main b4748414a`.

---

## 0. What changed (correction + reslim)

| Original 197 story | Verdict | Disposition |
|---|---|---|
| S1 one dimension authority + config-integrity gate | **moved** — S3 dimension + config-integrity are the live-plane's concern | → **PRD-186** (S2/S3 there) |
| S2 retire the "dark" S3 Vectors plane | **inverted** — S3 is the LIVE plane; deleting the backend breaks prod retrieval | **dropped** |
| S3 kill the F079 zombie store | valid, S3-agnostic | **kept → S1** |
| S4 settings-plane truth | valid | **kept → S2** |
| S5 Qdrant **memory** snapshots | valid (memory planes only) | **kept → S3** |
| S6 substrate telemetry | valid | **kept → S4** |
| — | the #514 revert left the local edition with **no** document backend | **new → S5** |

---

## 1. What this is

The substrate consolidation **minus the S3-directional errors**. With PRD-186 now owning the S3 plane (dimension fail-loud, config-integrity, isolation), PRD-197 is the **S3-agnostic subtraction**: delete the genuinely-dead F079 zombie store, make the settings plane honest, back up the Qdrant *memory* planes, put a number on the substrate, and make **open-core RAG actually work** (the local edition has no document backend now that the pgvector fallback was reverted out).

**Framing (CLAUDE.md §3):** **Refactor / subtraction** — pick the canonical path, migrate the losers, delete them. **Build size:** S–M (mostly deletion + caller migration + one cron + one telemetry seam + one local-edition backend). **Risk:** Low — **nothing here touches the live S3 retrieval path**; recall is held constant by design (the document plane does not move).

---

## 2. Current reality (grounded @ `main` b4748414a — corrected)

- **Documents run on S3 Vectors, live.** `_get_candidates` (`modules/rag/service.py:1017`) constructs `S3VectorsBackend` (`:1010-1012`); `document_chunks` (pgvector) is the **content-hydration** table (`:1176 FROM document_chunks`), not the vector plane. This plane **does not move** in this PRD.
- **The F079 zombie store is still present and still separate from the live path.** `modules/search/service.py`, `modules/search/retrieval/context_retrieval_engine.py`, `modules/search/vector_store/store.py` (the wrong-math `EnhancedVectorStore`/`SearchService`/`ContextRetrievalEngine` trio) — the live RAG path imports `S3VectorsBackend` **directly**, not these. **The importer surface has drifted** from the original 197's list (grep now shows `modules/__init__.py`, `modules/codegraph/codegraph_service.py`, plus *comment-only* "do not resurrect" refs in `modules/rag/bm25_leg.py`/`service.py`) — the real import set must be **re-traced at build**, not taken from the stale list.
- **Settings-plane drift.** The embeddings admin card and `modules/memory/__init__` column names don't agree with the canonical `(embeddings, *)` keys — a settings placebo + stale columns.
- **Qdrant memory planes have no snapshot.** `modules/memory/durable_store.py` (PRD-187 durable memory) + `field_memory` (PRD-166) live on the Railway Qdrant with no backup — unrecoverable on loss. (Document backups are S3's concern, PRD-186 — **not** here.)
- **No substrate telemetry.** No per-seam candidates/latency/errors metric watches retrieval health.
- **Open-core is broken.** With `S3_VECTORS_ENABLED=false` (the local/OSS default — S3 Vectors is AWS-only), the reverted pgvector fallback means the local edition has **no document backend at all** → local RAG constructs nothing and returns empty. This is the "make open-core RAG actually work" the review's P2-16 named — now a real, grounded gap.

---

## 3. Findings → fix → story

| # | Finding (grounded) | Fix | Story |
|---|---|---|---|
| **F079 zombies** | The wrong-math trio still ships; importer surface drifted. | **Re-trace** real importers on main, migrate each to the canonical path (documents → `modules/rag`/S3; memory/field → Qdrant adapters), grep-prove zero, **delete** the trio + orphans. | **S1** |
| **settings placebo** | Admin card + `memory/__init__` columns disagree with canonical `(embeddings,*)` keys. | Managers read canonical keys through config; fix the stale columns; **truthify-or-delete** the placebo card (honest-OFF over silent placebo). | **S2** |
| **no memory backups** | Qdrant `durable_memory` + `field_memory` unrecoverable on loss. | **Snapshot cron** on the memory-jobs scheduler + a runbook line; cadence/retention/destination per Gerard (Q3). **Memory planes only** — documents are S3/PRD-186. | **S3** |
| **no substrate number** | No telemetry watches retrieval health per seam. | A telemetry seam (candidates/latency/errors per retrieval seam) through the PRD-185 observability plane + a Command Center SLO tile. | **S4** |
| **open-core broken** | Local edition (S3 off) has no document backend since the pgvector fallback was reverted. | Wire a **local-edition document backend** (pgvector-local or Qdrant-local, Q1) gated by config so a fresh clone's RAG **constructs and returns**; S3 stays the SaaS path. | **S5** |

---

## 4. Stories (test-first; CI is the only gate — no local runs)

### S1 · Kill the F079 zombie store — M · _vector-substrate C.5/E.4/J9_
**Re-trace** the real importers of `EnhancedVectorStore`/`SearchService`/`ContextRetrievalEngine` on `main` (distinguish live imports from the comment-only "do not resurrect" references — the grep matches both). Repoint each real importer to the canonical path (documents → `modules/rag`/S3VectorsBackend; any memory/field use → the Qdrant adapters), **grep-prove zero importers**, then delete `modules/search/service.py`, `modules/search/retrieval/context_retrieval_engine.py`, `modules/search/vector_store/store.py`, and any now-orphaned siblings.
**Test:** per-caller behavioural test that the migrated path returns equivalent results (fixture-fed, mocked at the boundary); `test_no_zombie_store_importers` (source-grep guard). Pure.
**Notes:** If a caller's semantics genuinely differ from the canonical path, surface it (§8-Q2), don't silently flatten. **Does not touch the live S3 retrieval** — held constant.

### S2 · Settings-plane truth — S · _vector-substrate C.2/J3_
Managers read the canonical `(embeddings, *)` keys through the config module; fix the stale `modules/memory/__init__` column names; the embeddings admin card reflects real state or is deleted (no placebo — the PRD-196 honest-UI lesson). Route-manifest updated only if a route changes.
**Test:** `test_embeddings_settings_roundtrip`; `test_no_stale_memory_init_columns`. Pure.

### S3 · Qdrant **memory** snapshots — XS · _vector-substrate C.7/J7_
A snapshot job for `durable_memory` + `field_memory` on the existing memory-jobs scheduler pattern (`services/memory_jobs.py`, wired at `main.py`); cadence + retention + destination per Gerard (Q3). One runbook line documents restore. **Memory planes only** — document backups are PRD-186's (S3 Vectors).
**Test:** `test_snapshot_job_registered`; `compute_next_snapshot` pure helper tested. No live Qdrant in tests.

### S4 · Substrate telemetry → SLO tile — S · _vector-substrate G.1/I.2/J6_
A telemetry seam recording candidates-returned / latency / errors per retrieval seam (documents/S3 + memory + field), through the PRD-185 observability plane; a Command Center tile surfaces healthy/degraded. Makes "is retrieval healthy?" a live number so the next dark-plane trips a tile, not a user complaint.
**Test:** `test_substrate_telemetry_records_candidates_latency`; tile renders healthy + degraded. Pure/mocked. (Tracer singleton is `config` NOT `settings` — PRD-185 gotcha.)

### S5 · Open-core local-RAG — S · _vector-substrate C.1 open-core / review P2-16_
The local/OSS edition (`S3_VECTORS_ENABLED=false`) must have a working document backend — today it has none (the pgvector fallback was reverted). Wire a **local-edition document backend** (pgvector-local or Qdrant-local, Q1), selected by config, so a fresh clone's `RAGService.retrieve` constructs and returns real results; S3 Vectors stays the SaaS path. The P2-23 fresh-clone lane confirms.
**Test:** `test_local_edition_rag_constructs_and_returns` (S3 off ⇒ the local backend serves candidates, not empty); `test_saas_edition_uses_s3` (S3 on ⇒ S3VectorsBackend path, unchanged). Pure/mocked.
**Notes:** No `os.getenv` outside config.py; the edition switch is config. This is the one net-new backend seam — kept minimal (reuse the existing pgvector `document_chunks` content table as the local vector store, or a local Qdrant collection — Q1).

---

## 5. Sequencing
- **S1** (zombie kill) is the long pole — lands per-caller, deletion is one commit after the last importer moves. **S2/S3/S4** are independent and parallel-safe. **S5** (open-core) is independent; **S4 telemetry ideally lands early** if Gerard wants the tile feeding the §7 baseline while he runs Auto.
- **No migration that moves document data** — the S3 plane is held constant on purpose. S3's snapshot job and S5's local backend add no document-data motion.

## 6. Verification (CI is the only gate — no local runs)
Per `feedback-no-local-servers`: pure/mocked tests (mock Qdrant/S3/session at the boundary); the source-grep guard (no-zombie-importers) follows the PRD-185 S5 shape — repoint in the same commit the symbol moves. **The live S3 retrieval path is untouched → the Retrieval-recall eval must stay flat** (any move is a regression, not a feature). New route → update the committed `orchestrator/reports/route-manifest.json`. Migrations self-apply on boot.

## 7. Baseline capture / success
Consumes the same frozen retrieval baseline as the wave (recall held **flat** — the plane doesn't move). **Success = the delta:** engines **one fewer** (zombie gone, importers 0 grep-proven); settings **honest** (no placebo); Qdrant memory **recoverable** (was not); substrate **observable** (0 numbers → live SLO tile); **open-core RAG constructs + returns** on a fresh clone (was empty).

## 8. Open questions — Gerard's call (§12)
1. **Open-core local document backend (replaces the old moot pgvector-vs-Qdrant-for-*docs* question).** For the OSS/local edition (S3 off), use **pgvector-local** (reuse the `document_chunks` table as the local vector store — one fewer service for self-hosters) or **Qdrant-local** (unify with the memory planes the local edition already runs)? **Recommendation: pgvector-local** — simplest fresh-clone story. Confirm.
2. **Zombie importer scope.** Re-traced at build; if any caller's semantics genuinely differ from the canonical path, surfaced as a decision, not silently flattened. Migrate all in this PR (recommended) or slice?
3. **Qdrant memory snapshot cadence / retention / destination.** Proposal: daily, 7-day retention, to the object store the platform already uses. Adjust + confirm destination.
4. **Pull S4 telemetry forward** so the substrate tile is live while you run Auto (recommended — makes the wave measurable in-flight)?

---

*Supersedes the inverted original 197 (retire-S3/keep-pgvector). Traceability: `reports/dossiers/vector-substrate.md` (C.2 settings / C.5 zombies / C.7 backups; J6/J7/J9) and thesis-T2 (one engine less), under review id **P2-16**. S3 dimension/config-integrity/isolation now owned by the revised **PRD-186**; this PRD is S3-retrieval-agnostic and holds recall constant. All `file:line` re-confirmed @ `main b4748414a` (docs on S3 per `prd-172` revert; zombie trio + Qdrant memory planes present; open-core doc-backend gap real). Reuses PRD-185 (telemetry seam), PRD-187/166 (Qdrant memory — snapshotted here). PILOT lens; measurement-forward; no moat framing.*
